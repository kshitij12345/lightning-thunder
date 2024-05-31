import torch
import thunder

dim = 4096


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


model = Model().to(device="cuda")
x = torch.randn(4, dim, dim, device="cuda")

# Transform for profiling with NVTX markers.
# This transform adds NVTX markers to all the computation symbols in the execution trace.
# It also marks the start and end of trace with cudaProfilerStart and cudaProfilerEnd to specify the capture range.
from profile_transform import nvtx_profile_transform, nvtx_profiler_ex, profile_with_nvtx

# Ideal UX
# Problem - `transform_for_execution` is called after additional transforms.
#           This leads to reordering of profiling operators when a FusionExecutor is present.
# jmodel = thunder.jit(
#     model,
#     additional_transforms=[
#         nvtx_profile_transform,
#     ],
#     executors=(nvtx_profiler_ex,) + thunder.get_default_executors(),
# )
# o = jmodel(x)

jmodel = thunder.jit(model)
jmodel(x)
# Hacky but works (with FusionExecutors).
# `profile_with_nvtx` grabs the execution trace
# and inserts NVTX marker symbols to the trace and returns
# a python callable with this new trace.
# NOTE: # Model should be called atleast once so that we can grab execution trace.
profiling_model = profile_with_nvtx(jmodel)
profiling_model(x)
