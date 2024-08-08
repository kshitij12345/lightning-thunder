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

import torch

offload_exec = OperatorExecutor("offload_exec")

offload_onload_stream = torch.cuda.Stream()


def offload_to_cpu_impl(t):
    cuda = torch.device(t.device)
    offload_onload_stream.wait_stream(torch.cuda.default_stream(cuda))
    with torch.cuda.stream(offload_onload_stream):
        return t.to("cpu")


offload_to_cpu = offload_exec.register_operator(
    "offload_to_cpu",
    meta=lambda t: TensorProxy("offloaded_" + t.name, like=t, device=thunder.core.devices.Device("cpu")),
    fn=offload_to_cpu_impl,
)


# def load_to_gpu_impl(t, device):
#     with torch.cuda.stream(offload_onload_stream):
#         return t.to(device)


def load_to_gpu_impl(t, device):
    return t.to(device)


load_to_gpu = offload_exec.register_operator(
    "load_to_gpu",
    meta=lambda t, device: TensorProxy(like=t, device=thunder.core.devices.Device(device)),
    fn=load_to_gpu_impl,
)


def wait_till_load_impl(t):
    cuda = torch.device(t.device)
    t.record_stream(torch.cuda.default_stream(cuda))
    torch.cuda.default_stream(cuda).wait_stream(offload_onload_stream)
    return t


wait_till_load = offload_exec.register_operator("wait_till_load", meta=lambda t: t, fn=wait_till_load_impl)


def sync_stream_impl():
    torch.cuda.synchronize()


sync_stream = offload_exec.register_operator("sync_stream", meta=lambda: None, fn=sync_stream_impl)


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


def move_loads_closer_to_computation_consumer(execution_trace):
    new_execution_trace = from_trace(execution_trace)

    compute_producers, compute_consumers = thunder.core.utils.producers_and_consumers(execution_trace)

    bsyms = execution_trace.bound_symbols
    symbol_to_idx = get_symbol_to_idx(bsyms)
    symbols_to_swap = {}
    for idx, bound_symbol in enumerate(bsyms):
        if (
            bound_symbol.sym.id == load_to_gpu.id
            and (bsyms[idx + 1].sym.id) in ("reshape", "permute")
            and bsyms[idx + 2].sym.id == prims.PrimIDs.DEL
        ):
            # Assumes the first to be the first consumer in trace.
            first_consumer = compute_consumers[bsyms[idx + 1].output][0]
            symbols_to_swap[bound_symbol] = first_consumer

    # Move the loading of tensor - right above the consumer.
    # We move
    # 1. t = load_to_gpu(offloaded_t)
    # 2. t1 = Permute(t) or reshape(t)
    # 3. del t
    NUM_SYMS_TO_MOVE = 3

    # This is not optimal (and we should probably do something smarter here for swapping).
    new_bsyms = [bsym for bsym in bsyms]
    for symbol, consumer_symbol in symbols_to_swap.items():
        symbol_to_idx = {sym: idx for idx, sym in enumerate(new_bsyms)}
        curr_idx = symbol_to_idx[symbol]
        consumer_idx = symbol_to_idx[consumer_symbol]
        for _ in range(NUM_SYMS_TO_MOVE):
            symbol_being_updated = new_bsyms.pop(curr_idx)
            new_bsyms.insert(consumer_idx - 1, symbol_being_updated)

    new_execution_trace.bound_symbols = new_bsyms
    return new_execution_trace


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

        # Don't forget to add `CUDA sync` as the first symbol to the trace.
        # Sync streams between forward and backward (as we offload on different stream).
        sync_bsym = sync_stream.bind(output=None)
        sync_bsym.header = "Inserted by CPU Offloading"
        computation_trace.bound_symbols.insert(0, sync_bsym)

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

            # computation_trace = move_loads_closer_to_computation_consumer(computation_trace)

        return computation_trace


# dim = 2048


# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(dim, dim)
#         self.fc2 = torch.nn.Linear(dim, dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = torch.nn.functional.relu(x)
#         x = self.fc2(x)
#         return x


# with torch.device("cuda"):
#     model = Model()
#     x = torch.randn(dim, dim)
#     args = (x,)
#     kwargs = {}


def benchmark(model, args, kwargs):
    import time

    torch.cuda.synchronize()
    start = time.time_ns()
    for _ in range(10):
        e = model(*args, **kwargs)
        _ = torch.autograd.grad(e, model.parameters(), g)
    torch.cuda.synchronize()
    end = time.time_ns()
    print(f"It took {(end - start) / 1e6} ms!")


from thunder.benchmarks.targets import LitGPTConfig, LitGPTBenchmark

with torch.device("cuda"):
    cfg: LitGPTConfig = LitGPTConfig.from_name("Llama-2-7b-hf")
    cfg.n_layer = 10
    cfg.block_size = 1024
    b = LitGPTBenchmark(cfg)
    model = b.fn()
    args, kwargs = b.make_batch()

offload_tfms = CPUOffloading()
print(torch.cuda.max_memory_allocated() / 1e9)
jmodel = thunder.jit(model, transforms=[offload_tfms])
# jmodel = thunder.jit(model)

a = jmodel(*args, **kwargs)
print(torch.cuda.max_memory_allocated() / 1e9)

g = torch.rand_like(a) / a.numel()
actual_grads = torch.autograd.grad(a, model.parameters(), g)

print(torch.cuda.max_memory_allocated() / 1e9)

# clone_arg = args[0].detach().clone()
# e = model(clone_arg)

# expected_grads = torch.autograd.grad(e, model.parameters(), g)
# torch.testing.assert_close(a, e)
# torch.testing.assert_close(actual_grads, expected_grads)

# benchmark(jmodel, args, kwargs)
# benchmark(model, (clone_arg,), kwargs)

# print(thunder.last_backward_traces(jmodel)[-1])

# 5 Layer
# Without TFMS
# 5.148185088
# 23.921770496
# 25.628853248

# With TFMS
# 5.148185088
# 7.798521856
# 15.084094464

# 10 Layer
# Without TFMS
# 9.195852288
# 45.460013056
# 47.167095808

# With TFMS (move_loads_closer_to_computation_consumer)
# 9.195852288
# 11.846189056
# 24.500470784

# With TFMS (move_closer_to_consumer)
# 9.145504256
# 11.82941184
# 28.388851712
