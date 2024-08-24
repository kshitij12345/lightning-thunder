from enum import Enum, auto
import dataclasses
from typing import List, Dict, Optional, Tuple
from collections.abc import Callable
import pprint
import itertools
import copy
from functools import partial
import operator

import torch
from torch.fx.passes.split_module import split_module as fx_split_module
import warnings
from collections.abc import Mapping
from torch.fx.passes import operator_support

from thunder.core.baseutils import run_once
from thunder.torch.default_torch_ops import torch_auto_registered_ops
from thunder.torch import _torch_to_thunder_function_map

auto_register_ops = set(itertools.chain(*torch_auto_registered_ops.values()))


@run_once
def _warn_thunder_compiler():
    warnings.warn(
        "The ThunderCompiler is in active development and may not work as expected."
        + " Please report any issues you encounter to the Lightning Thunder team."
    )


class CompilerType(Enum):
    THUNDER = auto()
    TORCH_INDUCTOR = auto()


@dataclasses.dataclass
class CompiledFunction:
    original_graph_module: torch.fx.GraphModule
    compiled_fn: Callable
    compiler: CompilerType


@dataclasses.dataclass
class SubgraphInfo:
    original_graph_module: torch.fx.GraphModule
    compiled_functions: list["CompiledFunctions"]
    is_split: bool
    split_reasons: list | None = None
    split_graph_module: torch.fx.GraphModule | None = None


class SplitReasonType(Enum):
    UNSUPPORTED_NODE = auto()
    MISSING_OP_SUPPORT = auto()
    EXCEPTION_PROXY_THUNDER_OP = auto()
    EXCEPTION_META_THUNDER_OP = auto()


@dataclasses.dataclass
class SplitReason:
    type: SplitReasonType
    info: str | None
    exception: Exception | None = None


def try_execute_symbol(thunder_symbol, node) -> tuple[bool, SplitReason | None]:
    import thunder
    from thunder.core.trace import TraceCtx
    from thunder.core.proxies import proxy

    trc = TraceCtx()
    # We need to be under trace context to generate proxies.
    with thunder.core.trace.tracectx(trc):
        try:

            def make_tensor_proxy(arg_node):
                # This is a Node in the graph representing a Tensor.
                if isinstance(arg_node, torch.fx.Node):
                    example_value = arg_node.meta["example_value"]
                    return proxy(example_value)

                # This is int, float, etc.
                return arg_node

            proxy_args = tuple(map(make_tensor_proxy, node.args))
            proxy_kwargs = {k: make_tensor_proxy(v) for k, v in node.kwargs.items()}
        except Exception as e:
            return False, SplitReason(
                SplitReasonType.EXCEPTION_PROXY_THUNDER_OP,
                f"Failed while creating proxy for node with name: {node.name} and target: {node.target}, see exception field",
                exception=e,
            )

        try:
            thunder_symbol(*proxy_args, **proxy_kwargs)
        except Exception as e:
            return False, SplitReason(
                SplitReasonType.EXCEPTION_META_THUNDER_OP,
                f"Failed while running meta for node with name: {node.name} and target: {node.target}, see exception field",
                exception=e,
            )

    return True, None


class ThunderOperatorSupport:
    def __init__(self, gm):
        self.gm = gm
        self.unsupported_nodes = set()
        self.find_unsupported_ctx_regions(gm)
        self.split_reasons: list[SplitReason] = []

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
        if node in self.unsupported_nodes:
            self.split_reasons.append(
                SplitReason(
                    SplitReasonType.UNSUPPORTED_NODE,
                    info=f"node with name: {node.name} and target: {node.target} is not supported probably because it is in unsupported context.",
                )
            )
            return False

        target = node.target
        if node.op == "call_method":
            self_arg = node.args[0]
            target = getattr(torch.Tensor, node.target, None)
            assert target is not None, f"Failed to find method {node.target}"

        if target in (operator.add, operator.sub, operator.mul, operator.getitem, operator.gt, operator.lt):
            # Example - x: "f32[2]" = l_x_ + 2;  l_x_ = None
            return True

        if target in auto_register_ops:
            self.split_reasons.append(
                SplitReason(
                    SplitReasonType.MISSING_OP_SUPPORT,
                    info=f"node with name: {node.name} and target: {node.target} only has an automatic torch fallback in thunder.",
                )
            )
            return False

        if target in _torch_to_thunder_function_map:
            if target in [torch.ones]:  # Factory functions.
                return True

            thunder_symbol = _torch_to_thunder_function_map[target]
            did_run, opt_split_reason = try_execute_symbol(thunder_symbol, node)
            if not did_run:
                self.split_reasons.append(opt_split_reason)
            return did_run

        self.split_reasons.append(
            SplitReason(
                SplitReasonType.MISSING_OP_SUPPORT,
                info=f"node with name: {node.name} and target: {node.target} didn't have any mapping in thunder.",
            )
        )
        return False


def is_graph_supported_by_thunder(gm, sample_input):
    op_support = ThunderOperatorSupport(gm)
    supported = True
    # gm.graph.print_tabular()
    for node in gm.graph.nodes:
        if node.op in ["call_method", "call_function"]:
            supported = op_support.is_node_supported(gm, node)
            if not supported:
                break
    return supported


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

    def splitter(self, gm, sample_input):
        operator_support = ThunderOperatorSupport(gm)

        prev_value = None
        partition_cnt = 0
        supported_partitions = set()

        def callback(node):
            nonlocal prev_value, partition_cnt
            new_value = operator_support.is_node_supported(gm, node)
            if prev_value == new_value:
                return partition_cnt

            prev_value = new_value
            partition_cnt += 1
            if new_value:
                supported_partitions.add(partition_cnt)
            return partition_cnt

        split_module = fx_split_module(gm, None, callback, keep_original_order=True, keep_original_node_name=True)
        compiled_funcs = []
        for node in split_module.graph.nodes:
            if (
                node.name.startswith("submod") and int(node.name.replace("submod_", "")) in supported_partitions
            ):  # For thunder
                graph_module = getattr(split_module, node.name)
                jit_fn = self.thunder_jit(graph_module)
                setattr(split_module, node.name, jit_fn)
                compiled_funcs.append(CompiledFunction(graph_module, jit_fn, CompilerType.THUNDER))
            elif node.name.startswith("submod"):  # For inductor
                graph_module = getattr(split_module, node.name)
                jit_fn = torch.compile(graph_module, backend="inductor")
                setattr(split_module, node.name, jit_fn)
                compiled_funcs.append(CompiledFunction(graph_module, jit_fn, CompilerType.TORCH_INDUCTOR))

        self.subgraph_infos.append(SubgraphInfo(gm, compiled_funcs, True, operator_support.split_reasons, split_module))
        # pprint.pprint(self.subgraph_infos[-1].split_reasons)
        return split_module

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        from thunder import jit

        # Dynamo uses lazy generation of the underlying Python code, so we need to
        # force recompilation of the GraphModule before passing it to Thunder.
        gm.real_recompile()

        # Check if the complete graph `gm` is supported by thunder
        # If yes, pass the whole `gm` to `thunder.jit` and return the compiled function.
        # if is_graph_supported_by_thunder(gm, sample_args):
        #     jitted_gm = self.thunder_jit(gm)
        #     self.thunder_fns.append(jitted_gm)
        #     self.thunder_to_gm[jitted_gm] = gm
        #     compiled_fn = CompiledFunction(gm, jitted_gm, CompilerType.THUNDER)
        #     self.subgraph_infos.append(SubgraphInfo(gm, [compiled_fn], False))
        #     return jitted_gm

        # The whole graph is not supported by `thunder`, so we split it in `thunder` supported sections
        # and unsupported sections which are passed to `torch.compile(backend='inductor')`
        # split_module = capability_partitioner_splitter(gm, sample_args)
        split_module = self.splitter(gm, sample_args)
        return split_module
