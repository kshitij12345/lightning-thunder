import transformers
import thunder
import time
import torch
import torch.utils.benchmark

def _view_input_as_2d(x):
    shape = x.shape
    return x.view((-1, shape[-1]))

tuple_t = (1000, 1000)

t = torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16)
timer = torch.utils.benchmark.Timer("t", globals={"t": t}, label="Times", description="noop")
m1 = timer.timeit()

timer = torch.utils.benchmark.Timer("t.ndim", globals={"t": t}, label="Times", description="ndim")
m2 = timer.timeit()

timer = torch.utils.benchmark.Timer("t.shape", globals={"t": t}, label="Times", description="shape")
m3 = timer.timeit()

timer = torch.utils.benchmark.Timer("_view_input_as_2d(t)", globals={"t": t, "_view_input_as_2d": _view_input_as_2d}, label="Times", description="view_inp")
m4 = timer.timeit()

timer = torch.utils.benchmark.Timer("len(tuple_t)", globals={"tuple_t": tuple_t}, label="Times", description="Len Tuple")
m5 = timer.timeit()

compare = torch.utils.benchmark.Compare([m1, m2, m3, m4, m5])
compare.print()
