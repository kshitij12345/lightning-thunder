import torch.utils.benchmark
import thunder
from thunder import Transform
from thunder.core.proxies import TensorProxy, variableify
from thunder.core.pytree import tree_flatten, tree_map
from thunder.core.trace import tracectx, from_trace
from thunder.extend import OperatorExecutor
from thunder.core.symbol import BoundSymbol
from thunder.core import prims
from thunder.core.transforms import bsym_list_to_dag, Node, toposort_bsym_dag, TOPOSORT_ORDER

from collections.abc import Sequence
from contextlib import nullcontext

import torch

offload_exec = OperatorExecutor("offload_exec")


def offload_to_cpu_impl(t):
    # Due to https://github.com/Lightning-AI/lightning-thunder/issues/950
    # it may receive tensor on CPU.
    if t.device == torch.device("cpu"):
        return t

    packed = torch.empty(
        t.size(),
        dtype=t.dtype,
        layout=t.layout,
        pin_memory=True,
    )
    packed.copy_(t)
    return packed


offload_to_cpu = offload_exec.register_operator(
    "offload_to_cpu",
    meta=lambda t: TensorProxy("offloaded_" + t.name, like=t, device=thunder.core.devices.Device("cpu")),
    fn=offload_to_cpu_impl,
)


def load_to_gpu_impl(t, device):
    return t.to(device, non_blocking=True)


load_to_gpu = offload_exec.register_operator(
    "load_to_gpu",
    meta=lambda t, device: TensorProxy(like=t, device=thunder.core.devices.Device(device)),
    fn=load_to_gpu_impl,
)


def _get_symbols_to_last_or_first_used_variables(symbols, first_used=False):
    variable_to_last_symbol = {}
    symbol_to_last_variables = {}

    def _mark_last_use(symbol, variable):
        if not variable in variable_to_last_symbol:
            variable_to_last_symbol[variable] = symbol
            symbol_to_last_variables.setdefault(symbol, []).append(variable)

    iter_symbols = symbols if first_used else reversed(symbols)
    for symbol in iter_symbols:
        # If this function is used in the combined nvfuser+torch executor, there are no symbols but regions.
        # Regions do not have args, kwargs
        if hasattr(symbol, "inputs"):
            variables = tuple(symbol.inputs) + tuple(symbol.outputs)
        else:
            variables = (symbol.flat_variableified_proxy_args) + tuple(symbol.flat_variableified_proxy_outs)
        tree_map(lambda x: _mark_last_use(symbol, x), variables)

    return symbol_to_last_variables, variable_to_last_symbol


def get_symbols_to_last_used_variables(symbols):
    return _get_symbols_to_last_or_first_used_variables(symbols)


def get_symbols_to_first_used_variables(symbols):
    return _get_symbols_to_last_or_first_used_variables(symbols, first_used=True)


def get_symbol_to_idx(symbols):
    return {sym: idx for idx, sym in enumerate(symbols)}


def move_closer_to_consumer(execution_trace):
    order_in_trace = {bsym: i for i, bsym in enumerate(execution_trace.bound_symbols)}

    def prefer_ops_closer_to_consumer(eligible_nodes: list[Node]) -> int:
        def key(node: Node) -> int:
            return order_in_trace[node.bsym]

        return min(range(len(eligible_nodes)), key=lambda i: key(eligible_nodes[i]))

    # This moves all del or clear collection at the bottom (as they don't return anything)
    bound_symbols = toposort_bsym_dag(
        bsym_list_to_dag(execution_trace.bound_symbols)[1],
        TOPOSORT_ORDER.BOTTOM_UP,
        selector=prefer_ops_closer_to_consumer,
    )

    for idx, bsym in enumerate(bound_symbols):
        if bsym.sym.id == prims.PrimIDs.DEL:
            break

    new_execution_trace = from_trace(execution_trace)
    new_execution_trace.bound_symbols = bound_symbols[:idx]

    new_execution_trace = thunder.executors.passes.del_last_used(new_execution_trace, clear_mutable_collections=True)
    return new_execution_trace


class CPUOffloading(Transform):
    def __init__(self, save_tensor_policy=None):
        self.forward_pass = None
        self.backward_pass = None
        self._offloaded_tensors = ()
        self.save_tensor_policy = None
        if save_tensor_policy is not None:
            assert callable(save_tensor_policy)
            self.save_tensor_policy = save_tensor_policy

    def _get_tensors_to_offload(self, forward_trace):
        return_bsym = forward_trace.bound_symbols[-1]
        trace_args = return_bsym.args[0]["flat_args"]
        saved_tensors = return_bsym.args[1][0]

        tensor_args_name = tuple(arg.name for arg in trace_args if isinstance(arg, TensorProxy))

        def is_in_tensor_args(t):
            return t.name in tensor_args_name

        def is_cuda_tensor(t):
            return t.device.type == "cuda"

        # Tensors which are intermediate and not argument to the computation trace are
        # the ones we are interested in offloading.
        tensors_to_offload = tuple(t for t in saved_tensors if ((not is_in_tensor_args(t)) and is_cuda_tensor(t)))
        if self.save_tensor_policy is not None:
            tensors_to_offload = self.save_tensor_policy(tensors_to_offload, forward_trace)
        self.tensors_to_offload = tensors_to_offload
        return self.tensors_to_offload

    def _replace_saved_tensors(self, forward_trace, new_output_map):
        return_bsym = forward_trace.bound_symbols[-1]
        return_bsym_args = return_bsym.args
        saved_tensors = return_bsym.args[1][0]

        new_saved_tensors = []
        for t in saved_tensors:
            new_output = new_output_map.get(variableify(t), t)
            new_saved_tensors.append(new_output)

        new_return_bsym = BoundSymbol.from_bsym(
            return_bsym, **{"args": (return_bsym_args[0], (tuple(new_saved_tensors), return_bsym_args[1][1]))}
        )
        forward_trace.bound_symbols.pop(-1)
        forward_trace.bound_symbols.append(new_return_bsym)

    def _offload_tensors_from_forward(self, computation_trace):
        # Find the tensors to offload.
        # We offload saved tensors which are not arguments to the computation trace and are saved for backwards.
        tensors_to_offload = self._get_tensors_to_offload(computation_trace)
        _, variable_to_last_symbol = get_symbols_to_last_used_variables(
            computation_trace.bound_symbols[:-1]
        )  # Ignore the return statement.

        symbol_to_idx = get_symbol_to_idx(computation_trace.bound_symbols)

        # Insert the offloading calls after the last use of the saved tensor (which we want to offload).

        # Book keeping for backward pass update.
        new_output_map = {}
        new_output_dev_map = {}

        # Since we are inserting in the list (we need to obey increasing order) - else the insertions will be incorrect.
        sorted_tensors_to_offload = sorted(
            tensors_to_offload, key=lambda t: symbol_to_idx[variable_to_last_symbol[variableify(t)]]
        )
        for idx, t in enumerate(sorted_tensors_to_offload):
            last_used_symbol = variable_to_last_symbol[variableify(t)]
            last_used_symbol_idx = symbol_to_idx[last_used_symbol]
            computation_trace.push_scope([])
            with tracectx(computation_trace):
                o = offload_to_cpu(t)
                prims.python_del(t)
            scoped_comp = computation_trace.pop_scope()
            scoped_comp[0].header = "Created by CPU Offloading Transform"

            # This will insert `del` first and then push it down when we insert `offload_to_cpu`.
            computation_trace.bound_symbols.insert(last_used_symbol_idx + 1 + (idx * 2), scoped_comp[1])
            computation_trace.bound_symbols.insert(last_used_symbol_idx + 1 + (idx * 2), scoped_comp[0])

            # Update book keeping.
            new_output_map[variableify(t)] = o
            new_output_dev_map[variableify(t)] = t.device.device_str()

        # Update the return symbol to return our offloaded tensors in saved for backward.
        self._replace_saved_tensors(computation_trace, new_output_map)

        # Book keeping for backward pass update.
        self._offloaded_tensors = new_output_map
        self._offloaded_tensors_dev = new_output_dev_map
        return computation_trace

    def _load_tensors_for_backward(self, computation_trace):
        self.backward_pass = computation_trace
        offloaded_tensors = self._offloaded_tensors
        offloaded_tensors_dev_map = self._offloaded_tensors_dev

        compute_producers, compute_consumers = thunder.core.utils.producers_and_consumers(computation_trace)

        # We want to insert `loads` before the first use of offloaded_tensors.
        _, variable_to_first_symbol = get_symbols_to_first_used_variables(computation_trace.bound_symbols)

        symbol_to_idx = get_symbol_to_idx(computation_trace.bound_symbols)

        # Update unpack collection so that it
        # outputs the offloaded tensor proxies (not the original ones).
        unpack_sym = compute_producers[list(offloaded_tensors.keys())[0].proxy]
        unpack_idx = symbol_to_idx[unpack_sym]
        unpack_sym_out = unpack_sym.output
        new_out = []
        for out in unpack_sym_out:
            vout = variableify(out)
            if vout in offloaded_tensors:
                new_out.append(offloaded_tensors[vout])
            else:
                new_out.append(out)
        new_unpack_bsym = BoundSymbol.from_bsym(unpack_sym, output=tuple(new_out))
        computation_trace.bound_symbols[unpack_idx] = new_unpack_bsym

        # Now we again find the first usages of offloaded tensor
        # This will actually point us to the first consumer of the offloaded tensor.
        offset = unpack_idx + 1
        _, variable_to_first_symbol = get_symbols_to_first_used_variables(computation_trace.bound_symbols[offset:])

        # Load the offloaded tensors to GPU before usage.
        # Should iterate in correct order (else it would be problematic).
        for idx, (vt, offloaded_t) in enumerate(
            sorted(offloaded_tensors.items(), key=lambda kv: symbol_to_idx[variable_to_first_symbol[kv[0]]])
        ):
            first_used_symbol = variable_to_first_symbol[vt]
            first_used_symbol_idx = symbol_to_idx[first_used_symbol]
            t = vt.proxy
            device = offloaded_tensors_dev_map[vt]

            with tracectx(computation_trace):
                new_sym = load_to_gpu.bind(offloaded_t, device, output=t)
                # new_wait_sym = wait_till_load.bind(t, output=t)
            new_sym.header = "Created by CPU Offloading Transform"
            # computation_trace.bound_symbols.insert(first_used_symbol_idx + idx, new_wait_sym)
            computation_trace.bound_symbols.insert(first_used_symbol_idx + idx, new_sym)

        return computation_trace

    def transform_trace_post_optimization(self, computation_trace: thunder.TraceCtx, **kwargs):
        if self.forward_pass is None:
            self.forward_pass = computation_trace
            # Processing for the forward pass (only if we are going to compute backward).
            if "augmented_forward" in computation_trace.fn.__name__:
                computation_trace = self._offload_tensors_from_forward(computation_trace)
        else:
            # Skip if no tensor was offloaded.
            if len(self._offloaded_tensors) == 0:
                return computation_trace

            # We need this because in unmodified backward trace, the first consumer of saved_for_backward maybe
            # a reshape or permute op and the actual computation occurs 50-100 (or more) lines later.
            # Because of this we load more tensors than required eagerly (thus decreasing the memory gains from CPU Offloading).
            # This function is currently tailored to pattern observed in Llama-2
            # Eg. on line 92
            #   # Created by CPU Offloading Transform
            #   t1319 = load_to_gpu(offloaded_t1319, 'cuda:0')  # t1319: "cuda:0 f32[8, 1024, 11008]"
            #   t4021 = torch.reshape(t1319, (-1, 11008))  # t4021: "cuda:0 f32[8192, 11008]"
            #     # t4021 = ltorch.reshape(t1319, (-1, 11008))  # t4021: "cuda:0 f32[8192, 11008]"
            #       # t4021 = prims.reshape(t1319, (8192, 11008))  # t4021: "cuda:0 f32[8192, 11008]"
            #   del t1319
            # And it's usage in computation is at 612
            # t4022 = torch.matmul(t4020, t4021)  # t4022: "cuda:0 f32[4096, 11008]"
            #   t4022 = ltorch.matmul(t4020, t4021)  # t4022: "cuda:0 f32[4096, 11008]"
            #     t4022 = prims.matmul(t4020, t4021)  # t4022: "cuda:0 f32[4096, 11008]"
            computation_trace = move_closer_to_consumer(computation_trace)

            # Transform the backward trace to load offloaded tensors back to the device.
            computation_trace = self._load_tensors_for_backward(computation_trace)

        return computation_trace


def benchmark(jmodel, model, args, kwargs):
    stmt = """
# Use the optimized model for prediction and backward
o = jmodel(*args, **kwargs)
o.sum().backward()
for param in model.parameters():  # use original model for clear grads
    param.grad = None
"""
    timer = torch.utils.benchmark.Timer(
        stmt=stmt, globals={"jmodel": jmodel, "model": model, "args": args, "kwargs": kwargs}
    ).timeit(number=10)
    return timer


from thunder.benchmarks.targets import LitGPTConfig, LitGPTBenchmark

name = "thunder_offload"


def get_model_and_args(name):
    with torch.device("cuda"):
        cfg: LitGPTConfig = LitGPTConfig.from_name("Llama-2-7b-hf")
        cfg.n_layer = 20
        # cfg.block_size = 1024
        b = LitGPTBenchmark(cfg, batchdims=(1,))
        model = b.fn()
        args, kwargs = b.make_batch()

    if name == "thunder_offload":
        offload_tfms = CPUOffloading()
        jmodel = thunder.jit(model, transforms=[offload_tfms])
    elif name == "thunder":
        # Use same executors for consistency in comparison.
        jmodel = thunder.jit(model)
    elif name == "eager_offload":

        def jmodel(*args, **kwargs):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return model(*args, **kwargs)

    elif name == "eager":  # OOM
        jmodel = model

    return jmodel, model, args, kwargs, cfg


jmodel, model, args, kwargs, cfg = get_model_and_args(name)

memory_after_model_load = torch.cuda.max_memory_allocated() / 1e9
print(memory_after_model_load)

a = jmodel(*args, **kwargs)

memory_after_forward = torch.cuda.max_memory_allocated() / 1e9
print(memory_after_forward)

g = torch.rand_like(a) + 0.02
actual_grads = torch.autograd.grad(a, model.parameters(), g)

memory_after_backward = torch.cuda.max_memory_allocated() / 1e9
print(memory_after_backward)

# # Sanity Check values vs Eager
# e = model(*args, **kwargs)
# expected_grads = torch.autograd.grad(e, model.parameters(), g)
# torch.testing.assert_close(a, e)
# torch.testing.assert_close(actual_grads, expected_grads)
# del e
# del expected_grads

# Clear every cuda tensor for benchmarking. We will create everything except grad and again!
del a
del g
del actual_grads
del args
del kwargs
del model
del jmodel
import gc

gc.collect()
torch.cuda.empty_cache()

memory_allocated = torch.cuda.memory_allocated() / 1e9
print(memory_allocated)

jmodel, model, args, kwargs, cfg = get_model_and_args(name)

measurement = benchmark(jmodel, model, args, kwargs)

print(measurement)


class Details:
    pass


d = Details()
d.mean = measurement.mean
d.median = measurement.median
d.memory_after_model_load = memory_after_model_load
d.memory_after_forward = memory_after_forward
d.memory_after_backward = memory_after_backward
d.cfg_n_layer = cfg.n_layer
d.cfg_blocksize = cfg.block_size

import json

f_name = f"{name}"
with open(f_name + ".json", "w") as f:
    json.dump(d.__dict__, f, indent=2)
