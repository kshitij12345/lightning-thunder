from thunder.core.trace import TraceCtx as Trace, from_trace, TraceProvenance
import thunder.core.prims as prims
from thunder.core.prims import PrimIDs
from thunder.extend import OperatorExecutor
import time
import torch
import thunder


class Timer:
    def __init__(self):
        self.start_time_ns = None
        self.end_time_ns = None

    def __enter__(self):
        self.start_time_ns = time.time_ns()
        return self

    def __exit__(self, *args):
        self.end_time_ns = time.time_ns()

    def get_elapsed_time_in_ms(self):
        elapsed_time_ns = self.end_time_ns - self.start_time_ns
        return elapsed_time_ns // 1000000


nvtx_profiler_ex = OperatorExecutor("nvtx_profiler_ex")


# Symbols for profiling.
def nvtx_push_impl(msg):
    torch.cuda.nvtx.range_push(msg)


def nvtx_pop_impl():
    torch.cuda.nvtx.range_pop()


def profile_start_impl():
    torch.cuda.cudart().cudaProfilerStart()


def profile_stop_impl():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


tags = (prims.OpTags.DONT_DCE,)
nvtx_push = nvtx_profiler_ex.register_operator("nvtx_range_push", meta=lambda msg: None, fn=nvtx_push_impl, tags=tags)
nvtx_pop = nvtx_profiler_ex.register_operator("nvtx_range_pop", meta=lambda: None, fn=nvtx_pop_impl, tags=tags)
cuda_profiler_start = nvtx_profiler_ex.register_operator(
    "cuda_profiler_start", meta=lambda: None, fn=profile_start_impl, tags=tags
)
cuda_profiler_stop = nvtx_profiler_ex.register_operator(
    "cuda_profiler_stop", meta=lambda: None, fn=profile_stop_impl, tags=tags
)

NON_COMPUTATION_PRIMS = (
    PrimIDs.ASSERT_TENSOR_METADATA,
    PrimIDs.CHECK_TENSOR_SHAPE_AND_METADATA,
    PrimIDs.CHECK_NONE,
    PrimIDs.CHECK_EMPTY,
    PrimIDs.CHECK_LITERAL_LIKE,
    PrimIDs.CHECK_TYPE,
    PrimIDs.CHECK_INSTANCE,
    PrimIDs.CHECK_NUMBER_TYPE_AND_VALUE,
    PrimIDs.CHECK_BOOL_CONVERSION,
    PrimIDs.CHECK_STRING_VALUE,
    PrimIDs.CHECK_LEN,
    PrimIDs.ASSERT_COMPARE,
    PrimIDs.PYTHON_VARS,
    PrimIDs.UNPACK_FUNCTION_OBJ,
    PrimIDs.UNPACK_CACHE_INFO,
    PrimIDs.UNPACK_ATTR,
    PrimIDs.UNPACK_GETITEM,
    PrimIDs.UNPACK_EMPTY_DICT,
    PrimIDs.UNPACK_ITER,
    PrimIDs.UNPACK_NEXT,
    PrimIDs.UNPACK_KEY,
    PrimIDs.UNPACK_SEQUENCE,
    PrimIDs.UNPACK_TRIVIAL,
    PrimIDs.UNPACK_TUPLE,
    PrimIDs.UNPACK_LIST,
    PrimIDs.UNPACK_DICT_KEY,
    PrimIDs.CONSTRUCT_TUPLE,
    PrimIDs.PACK_SETITEM,
    # TODO: UNPACK_SET
    # Utility prims
    PrimIDs.COMMENT,
    PrimIDs.DEL,
    PrimIDs.PRINT,
)


def nvtx_profile_transform(trace: Trace, **kwargs) -> Trace:
    with Timer() as timer:
        profile_trace = from_trace(trace)

        # Start profiling
        profile_trace.bound_symbols.append(cuda_profiler_start.bind(output=None))
        for bound_symbol in trace.bound_symbols:
            # Synchronize and stop profiling at return.
            if PrimIDs.RETURN == bound_symbol.sym.id:
                profile_trace.bound_symbols.append(cuda_profiler_stop.bind(output=None))
                profile_trace.bound_symbols.append(bound_symbol)
                break

            if bound_symbol.sym.id in NON_COMPUTATION_PRIMS:
                # Just append the symbol.
                profile_trace.bound_symbols.append(bound_symbol)
                continue

            profile_trace.bound_symbols.append(nvtx_push.bind(f"{bound_symbol.python(indent=0)}", output=None))
            profile_trace.bound_symbols.append(bound_symbol)
            profile_trace.bound_symbols.append(nvtx_pop.bind(output=None))

    profile_trace.set_provenance(
        TraceProvenance(f"Profile Transform (took {timer.get_elapsed_time_in_ms()} milliseconds)")
    )
    return profile_trace


def profile_with_nvtx(compiled_model):
    def _fn(*args, **kwargs):
        exec_trace = thunder.last_traces(compiled_model)[-1]
        profile_trace = nvtx_profile_transform(exec_trace)
        cache_rec, inputs, _ = thunder.compile_data(compiled_model).get_computation_and_inputs(*args, **kwargs)
        profile_callable = profile_trace.python_callable()
        return profile_callable(*inputs)

    return _fn
