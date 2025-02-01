from __future__ import annotations

from typing import List, Set
from collections.abc import Callable
from collections import defaultdict
from copy import copy
from itertools import chain

import torch

import thunder.core.utils as utils
from thunder.core.trace import TraceCtx, from_trace, tracectx
from thunder.core.symbol import BoundSymbol
from thunder.core.proxies import variableify, Proxy, AnyProxy
import thunder.core.prims as prims
from thunder.core.prims import PrimIDs
from thunder.executors import torchex
from thunder.extend import OperatorExecutor


# Represents a region and its parents (regions it consumes the output of) and
#   children (regions that consume its output)
#   Essentially creates a directional graph of regions showing their
#   producer-consumer relationships.
# TODO These classes could be refactored to be region-independent in their logic
class Node:
    def __init__(self, ID: int, group_bsyms: list[BoundSymbol], group_indices: list[int], start: int, stop: int):
        self.ID = ID
        self.start = start
        self.stop = stop
        self.group_bsyms = group_bsyms
        self.group_indices = group_indices
        self.parents: utils.OrderedSet[Node] = utils.OrderedSet()
        self.children: utils.OrderedSet[Node] = utils.OrderedSet()

    def __repr__(self) -> str:
        s = f"node ID {self.ID} : "
        s += self.group_bsyms.__repr__()
        s += f"\n\tparents ids: "
        for parent in self.parents:
            s += f" {parent.ID}, "
        s += f"\n\tchildren ids: "
        for child in self.children:
            s += f" {child.ID}, "
        s += f"\n"
        return s

    def __hash__(self) -> int:
        return self.ID

    def __eq__(self, other) -> bool:
        return self.ID == other.ID

    @staticmethod
    def merge(a: Node, b: Node):
        merged_bsyms = []
        merged_indices = []

        def push_node(n: Node, index: int):
            nonlocal merged_bsyms
            nonlocal merged_indices
            merged_bsyms.append(n.group_bsyms[index])
            merged_indices.append(n.group_indices[index])

        a_index = 0
        b_index = 0
        while a_index < len(a.group_bsyms):
            if b_index < len(b.group_bsyms) and b.group_indices[b_index] < a.group_indices[a_index]:
                push_node(b, b_index)
                b_index += 1
            else:
                push_node(a, a_index)
                a_index += 1

        while b_index < len(b.group_bsyms):
            push_node(b, b_index)
            b_index += 1

        return merged_bsyms, merged_indices


# assumes bound_symbol comes in as a DAG and in valid topo order
# NOTE: consolidate graph implementations, we have several almost identical
# implementations already
class Graph:
    def __init__(self, trace: TraceCtx):
        self.roots: list[Node] = []
        self.return_node: None | Node = None
        self.counter = len(trace.bound_symbols)
        self.bsym_id_to_node_map = []

        producers = utils.producers(trace, _map_to_numbers=True)
        consumers = utils.consumers(trace, _map_to_numbers=True)

        # Note, even though BoundSymbolInterface is hashable, it's hash is very slow
        # as it appears to be far off from being universal.
        # We use indices as hash values instead.
        bsym_id_to_node_map: list[int] = []
        for bsym_id, bsym in enumerate(trace.bound_symbols):
            node = Node(bsym_id, [bsym], [bsym_id], bsym_id, bsym_id)
            bsym_id_to_node_map.append(node)

            if bsym.sym.id is PrimIDs.RETURN:
                utils.check(
                    self.return_node is None,
                    lambda: f"Found multiple RETURN nodes while converting a list of bound symbols to a dag",
                )
                self.return_node = node

        for bsym_id, node in enumerate(bsym_id_to_node_map):
            bsym = node.group_bsyms[0]
            for inp in bsym.flat_args:
                if not isinstance(inp, Proxy):
                    continue

                producer_id = producers[inp]
                parent = bsym_id_to_node_map[producer_id]
                node.parents.add(parent)

            if not node.parents:
                self.roots.append(node)

            for out in bsym.flat_outs:
                if not isinstance(out, Proxy):
                    continue

                # Checks that the output is actually produced by this function, and not an input to it
                if variableify(out) in (variableify(x) for x in bsym.flat_args):
                    continue

                children_ids = consumers.get(out, [])
                for child_id in children_ids:
                    child_node = bsym_id_to_node_map[child_id]
                    node.children.add(child_node)

        self.bsym_id_to_node_map = bsym_id_to_node_map

    def __repr__(self) -> str:
        s = f"graph roots:"
        for root in self.roots:
            s += f" {root.ID},"
        s += "\ntraversal nodes:\n"
        visit_stack = list(self.roots)
        visited = set()
        while visit_stack:
            cur = visit_stack.pop(0)
            if cur in visited:
                continue
            s += cur.__repr__()
            visited.add(cur)
            visit_stack.extend(cur.children)
            for child in cur.children:
                assert cur in child.parents
        return s

    # merge consumer `b` into producer `a`
    def merge(self, a: Node, b: Node) -> bool:
        ##############################
        # step0: cyclic check
        ##############################
        max_depth = max(a.stop, b.stop)

        if len(b.parents) != 1 or not a in b.parents:
            visit_stack = list()
            visit_stack.extend([x for x in a.children if x != b])
            visit_stack.extend([x for x in b.children if x != a])

            visited = set()
            while visit_stack:
                cur = visit_stack.pop(0)
                if cur in visited:
                    continue
                if cur in [a, b]:
                    # cycle detected, do nothing and return False
                    return False
                visited.add(cur)
                if cur.start <= max_depth:
                    visit_stack.extend(cur.children)

        ##############################
        # step1: merge the two nodes together
        ##############################

        # create a new_node as the merged node with combined bsyms from a and b

        min_start = min(a.start, b.start)
        merged_bsyms, merged_indices = Node.merge(a, b)
        new_node = Node(self.counter, merged_bsyms, merged_indices, min_start, max_depth)
        self.counter = self.counter + 1

        # TODO: this part is slow! we might want to refactor this section and do one merge with a group, instead of do rewiring for every single pair
        for parent in a.parents.union(b.parents):
            if parent is a or parent is b:
                continue
            parent.children.discard(a)
            parent.children.discard(b)
            parent.children.add(new_node)
            new_node.parents.add(parent)

        for child in a.children.union(b.children):
            if child is a or child is b:
                continue
            child.parents.discard(a)
            child.parents.discard(b)
            child.parents.add(new_node)
            new_node.children.add(child)

        if a in self.roots:
            # we want to put new_node at the same spot, i.e. # args: "Collection" would want to stay at where it was before the merge
            if a.parents.union(b.parents).issubset({a, b}):
                self.roots[self.roots.index(a)] = new_node
            else:
                self.roots.remove(a)

        return new_node


# import torch

# def memory_peak_efficient_func(t0s, a):
#     for t0 in t0s:
#         t1 = torch.sin(t0); del t0
#         t2 = t1.cos()
#         a = torch.matmul(t2, a); del t1
#         # t3 = torch.nn.functional.sin(t2); del t2
#         # a = torch.matmul(t3, a); del t3
#     return a

# N_PARALLEL_PATHS = 3
# t0s = [torch.randn(256, 256, device="cuda") for _ in range(N_PARALLEL_PATHS)] # 0.25 MiB * N_PARALLEL_PATHS
# a = torch.randn(256, 256, device="cuda") # 0.25 MiB

# import thunder

# jit_memory_peak_efficient_func = thunder.jit(memory_peak_efficient_func)
# expected = jit_memory_peak_efficient_func(t0s, a)

# fwd_trace = thunder.last_traces(jit_memory_peak_efficient_func)[-1]

import thunder


class StreamParallelization(thunder.Transform):
    def transform_trace_post_optimization(self, computation_trace, **kwargs):
        tmp_trace = from_trace(computation_trace)
        tmp_trace.bound_symbols = list(
            bsym for bsym in computation_trace.bound_symbols if not bsym.sym == thunder.core.prims.python_del
        )
        g = Graph(tmp_trace)

        import networkx as nx

        G = nx.DiGraph(directed=True)

        for bsym_id, node in enumerate(g.bsym_id_to_node_map):
            for children in node.children:
                G.add_edges_from([(node.ID, children.ID)])

        # import matplotlib.pyplot as plt
        # nx.draw(G, with_labels=True)
        # plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')

        # print(g)
        # print(G)

        sink = [node for node, d in G.out_degree() if d == 0][0]
        paths = []
        for bsym_id, node in enumerate(g.bsym_id_to_node_map):
            if len(node.parents) == 0:  # roots
                for path in nx.all_simple_paths(G, node.ID, 10):
                    paths.append(path)

        for path in paths:
            print(path)

        stream_map = {}
        node_to_streams_map = defaultdict(set)
        processor_cnt = 0
        for path in paths:
            for node in path:
                if node not in stream_map:
                    stream_map[node] = processor_cnt
            processor_cnt += 1

        print(stream_map)

        # find sync nodes
        for node in G.nodes():
            if G.in_degree(node) > 1:
                for child in G.predecessors(node):
                    node_to_streams_map[node].add(stream_map[child])
            else:
                node_to_streams_map[node].add(stream_map[node])

        print(node_to_streams_map)

        topological_nodes = list(nx.topological_sort(G))

        for path in set(stream_map.values()):
            if path == 0:  # Hot path skip!
                continue

            # Move the nodes of path closer to start point
            start_idx = None
            nodes_in_path = []
            for idx, node in enumerate(topological_nodes):
                if path in node_to_streams_map[node] and start_idx is None:
                    start_idx = idx
                elif path in node_to_streams_map[node] and len(node_to_streams_map[node]) == 1:
                    nodes_in_path.append(idx)
                elif path in node_to_streams_map[node] and len(node_to_streams_map[node]) > 1:
                    break

            for cnt, idx in enumerate(nodes_in_path):
                topological_nodes.insert(start_idx + 1 + cnt, topological_nodes.pop(idx))

        print(topological_nodes)

        stream_ex = OperatorExecutor("stream_ex")

        def sync_with_default_and_set_new_stream_impl(new_stream: torch.cuda.Stream):
            new_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.set_stream(new_stream)

        sync_with_default_and_set_new_stream = stream_ex.register_operator(
            "sync_with_default_and_set_new_stream",
            meta=lambda stream: None,
            fn=sync_with_default_and_set_new_stream_impl,
        )

        def set_default_stream_impl():
            torch.cuda.set_stream(torch.cuda.default_stream())

        set_default_stream = stream_ex.register_operator(
            "set_default_stream", meta=lambda: None, fn=set_default_stream_impl
        )

        def sync_with_default_stream_and_set_default_stream_impl(new_stream: torch.cuda.Stream):
            torch.cuda.default_stream().wait_stream(new_stream)
            torch.cuda.set_stream(torch.cuda.default_stream())

        sync_with_default_stream_and_set_default_stream = stream_ex.register_operator(
            "sync_with_default_stream_and_set_default_stream",
            meta=lambda stream: None,
            fn=sync_with_default_stream_and_set_default_stream_impl,
        )

        def get_new_stream_impl():
            return torch.cuda.Stream()

        get_new_stream = stream_ex.register_operator(
            "get_new_stream", meta=lambda: AnyProxy(None), fn=get_new_stream_impl
        )

        new_trace = from_trace(computation_trace)

        with tracectx(new_trace):
            new_nodes_with_syncs = []
            prev_node = None
            node_to_bsym_map = {node.ID: node.group_bsyms[0] for node in g.bsym_id_to_node_map}

            stream_proxy_map = {}
            for stream in set(stream_map.values()):
                if stream == 0:
                    continue
                stream_p = get_new_stream()
                stream_proxy_map[stream] = stream_p

            for node in topological_nodes:
                if new_nodes_with_syncs == []:
                    # First node nothing to do.
                    new_nodes_with_syncs.append(node_to_bsym_map[node])  # node
                    prev_node = node
                    continue

                if len(node_to_streams_map[node]) > 1:
                    stream_to_merge = stream_proxy_map[(node_to_streams_map[node] - {0}).pop()]
                    new_nodes_with_syncs.append(
                        sync_with_default_stream_and_set_default_stream.bind(stream_to_merge, output=None)
                    )
                    new_nodes_with_syncs.append(node_to_bsym_map[node])  # node
                    prev_node = node
                    continue

                if stream_map[node] != stream_map[prev_node]:
                    if len(node_to_streams_map[node]) == 1 and stream_map[node] == 0:  # back to default stream
                        new_nodes_with_syncs.append(set_default_stream.bind(output=None))
                        new_nodes_with_syncs.append(node_to_bsym_map[node])  # node
                    elif len(node_to_streams_map[node]) == 1 and stream_map[node] != 0:  # set new stream
                        stream_p = stream_proxy_map[stream_map[node]]
                        new_nodes_with_syncs.append(sync_with_default_and_set_new_stream.bind(stream_p, output=None))
                        new_nodes_with_syncs.append(node_to_bsym_map[node])  # node
                else:
                    new_nodes_with_syncs.append(node_to_bsym_map[node])  # node
                prev_node = node

        for bsym in new_nodes_with_syncs:
            new_trace.add_bound_symbol(bsym)

        # print(new_trace)
        # return new_trace
        return computation_trace
