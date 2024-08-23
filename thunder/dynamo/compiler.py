from enum import Enum, auto
import dataclasses
from typing import List, Dict, Optional
import itertools
import copy
from functools import partial

import torch
import warnings
from collections.abc import Mapping

from thunder.core.baseutils import run_once
from thunder.torch.default_torch_ops import torch_auto_registered_ops

auto_register_ops = set(itertools.chain(*torch_auto_registered_ops.values()))


@run_once
def _warn_thunder_compiler():
    warnings.warn(
        "The ThunderCompiler is in active development and may not work as expected."
        + " Please report any issues you encounter to the Lightning Thunder team."
    )


from torch.fx.passes import operator_support

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from thunder.torch import _torch_to_thunder_function_map


@dataclasses.dataclass
class SubgraphInfo:
    original_graph_module: torch.fx.GraphModule
    is_split: bool
    compiled_functions: list["CompiledFunctions"]
    split_reasons: list | None = None
    split_graph_module: torch.fx.GraphModule | None = None


class SplitReasonType:
    MISSING_OP_SUPPORT = auto()
    EXCEPTION_THUNDER_OP = auto()


class SplitReason:
    type: SplitReasonType
    info: str


def try_execute_symbol(thunder_symbol, node):
    import thunder
    from thunder.core.trace import TraceCtx
    from thunder.core.proxies import proxy

    trc = TraceCtx()
    # We need to be under trace context to generate proxies.
    with thunder.core.trace.tracectx(trc):

        def make_proxy(arg_node):
            # This is a Node in the graph representing a Tensor.
            if isinstance(arg_node, torch.fx.Node):
                fake_t = arg_node.meta["example_value"]
                return proxy(fake_t)

            # This is int, float, etc.
            return proxy(arg_node)

        proxy_args = tuple(map(make_proxy, node.args))
        proxy_kwargs = {k: proxy(v) for k, v in node.kwargs.items()}
        try:
            thunder_symbol(*proxy_args, **proxy_kwargs)
        except Exception as e:
            return False

    return True


class ThunderOperatorSupport(operator_support.OperatorSupport):
    def __init__(self, gm):
        self.gm = gm
        self.unsupported_nodes = set()
        self.find_unsupported_ctx_regions(gm)

    def find_unsupported_ctx_regions(self, gm):
        ctx_cnt = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in (torch.amp.autocast_mode._enter_autocast,):
                ctx_cnt += 1
            elif node.op == "call_function" and node.target in (torch.amp.autocast_mode._exit_autocast,):
                ctx_cnt -= 1
            else:
                if ctx_cnt > 0:
                    self.unsupported_nodes.add(node)

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node):
        import operator

        if node in self.unsupported_nodes:
            return False

        target = node.target
        if node.op == "call_method":
            self_arg = node.args[0]
            target = getattr(torch.Tensor, node.target, None)
            if target is None:
                return False

        if target in (operator.add, operator.sub, operator.mul, operator.getitem):
            return True

        if target in auto_register_ops:
            # Add split reason
            return False

        if target in _torch_to_thunder_function_map:
            if target in [torch.ones]:  # Factory functions.
                return True

            thunder_symbol = _torch_to_thunder_function_map[target]
            did_run = try_execute_symbol(thunder_symbol, node)
            if not did_run:
                # Add split reason
                pass
            return did_run

        # Add split reason
        return False


class GraphModuleSplitter(torch.fx.passes.splitter_base._SplitterBase):
    def starter_nodes(self):
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
        starter_cpu_nodes: NodeSet = set()
        starter_acc_nodes: NodeSet = set()

        for node in self.module.graph.nodes:
            if node.op not in {"placeholder", "get_attr"}:
                continue
            for user in node.users:
                if user in self.acc_nodes:
                    starter_acc_nodes.add(user)
                else:
                    starter_cpu_nodes.add(user)

        for node in self.module.graph.nodes:
            if node.op in {"output", "placeholder", "get_attr"}:
                continue

            if len(self.deps[node]) == 0:
                if node in self.acc_nodes:
                    starter_acc_nodes.add(node)
                else:
                    starter_cpu_nodes.add(node)

        return starter_cpu_nodes, starter_acc_nodes


def is_graph_supported_by_thunder(gm, sample_input):
    op_support = ThunderOperatorSupport(gm)
    supported = True
    for node in gm.graph.nodes:
        if node.op in ["call_method", "call_function"]:
            supported = op_support.is_node_supported(gm, node)
            if not supported:
                break


class ThunderCompiler:
    def __init__(self, **thunder_options):
        """
        A class that compiles a `fx.GraphModule` to a `thunder.ThunderModule`.
        This class is meant to be used as a backend for the `torch.compile`
        function.

        Keyword arguments:
            thunder_options: a dictionary of options to pass to `thunder.jit`.

        Example:
            >>> import torch
            >>> from thunder.dynamo import ThunderCompiler
            >>> backend = ThunderCompiler()
            >>> x = torch.ones(2, requires_grad=True)
            >>> @torch.compile(backend=backend)
            ... def func(x):
            ...     x = torch.sin(x)
            ...     if x.sum() > 0:
            ...         return x + 1
            ...     else:
            ...         return x - 1
            >>> out = func(x)
        """
        from thunder import ThunderModule, jit

        _warn_thunder_compiler()

        # Thunder-compiled functions should be readily available for inspection
        # and testing, so we will store them in a list. The order of the
        # functions in the list will be the same as the order in which they were
        # compiled. In addition, we will store a mapping from the ThunderModule
        # to the GraphModule that was passed to ThunderCompiler. This will allow
        # us to inspect the GraphModule that was compiled by Thunder.
        self.thunder_fns: list[ThunderModule] = []
        self.thunder_to_gm: dict[ThunderModule, torch.fx.GraphModule] = {}
        self.subgraph_infos: list[SubgraphInfo] = []

        self.thunder_options = thunder_options
        self.thunder_jit = partial(jit, **thunder_options)

        # TODO: There will be pieces of Dynamo IR that Thunder cannot compile, so we
        # will need to build a fallback mechanism to handle those cases.
        # Possible stages of the compilation that need to be saved for inspection:
        # 1. The GraphModule as it was passed to ThunderCompiler.
        # 2. The GraphModule after split for Thunder/PyTorch.
        # 3. If the whole GraphModule is not supported, record the reasons why.

    def splitter(self, gm, sample_input):
        """
        This function splits the graph provided by Dynamo
        if it contains any operation or construct that is not supported by thunder.
        For the unsupported subgraph, it is passed to inductor.
        """
        from thunder import jit

        # Setup the splitter class
        settings = torch.fx.passes.splitter_base._SplitterSettingBase(allow_non_tensor=True)
        splitter = GraphModuleSplitter(gm, sample_input, operator_support=ThunderOperatorSupport(gm), settings=settings)
        gm.print_readable()
        # Call the splitter to split GraphModule.
        split_module = splitter()
        split_module.print_readable()
        compiled_funcs = []
        for node in split_module.graph.nodes:
            if node.name.startswith("_run_on_acc_"):
                graph_module = getattr(split_module, node.name)
                jit_fn = self.thunder_jit(graph_module)
                setattr(split_module, node.name, jit_fn)
                compiled_funcs.append(jit_fn)
            if node.name.startswith("_run_on_cpu_") or node.name.startswith("_run_on_gpu_"):
                graph_module = getattr(split_module, node.name)
                jit_fn = torch.compile(graph_module, backend="inductor")
                setattr(split_module, node.name, jit_fn)
                compiled_funcs.append(jit_fn)

        self.subgraph_infos.append(SubgraphInfo(gm, True, compiled_funcs, [], split_module))
        # split_module.print_readable()
        return split_module

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        from thunder import jit

        # Dynamo uses lazy generation of the underlying Python code, so we need to
        # force recompilation of the GraphModule before passing it to Thunder.
        gm.real_recompile()

        # Check if the complete graph `gm` is supported by thunder
        # If yes, pass the whole `gm` to `thunder.jit` and return the compiled function.
        if is_graph_supported_by_thunder(gm, sample_args):
            jitted_gm = self.thunder_jit(gm)
            self.thunder_fns.append(jitted_gm)
            self.thunder_to_gm[jitted_gm] = gm
            self.subgraph_infos.append(SubgraphInfo(gm, False, [jitted_gm]))
            return jitted_gm

        # The whole graph is not supported by `thunder`, so we split it in `thunder` supported sections
        # and unsupported sections which are passed to `torch.compile(backend='inductor')`
        # split_module = capability_partitioner_splitter(gm, sample_args)
        split_module = self.splitter(gm, sample_args)
        return split_module
