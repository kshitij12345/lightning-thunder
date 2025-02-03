import torch.utils.benchmark
import thunder
import torch

N_ITER = 100
def fn(t0s, a):
    for t0 in t0s:
        t1 = torch.sin(t0); del t0
        t2 = t1.cos()
        for _ in range(N_ITER):
            t2 = t2.asin().sin()

        a = t2 + a

    return a

N_PARALLEL_PATHS = 3
DIM = 1024
t0s = [torch.randn(DIM, DIM, device="cuda") for _ in range(N_PARALLEL_PATHS)]
a = torch.randn(DIM, DIM, device="cuda")

import thunder

jfn = thunder.jit(fn, executors=[thunder.pytorch_executor,])
expected = jfn(t0s, a)

from thunder.transforms.stream_parallelization import StreamParallelization

import time
time.sleep(1)

njfn = thunder.jit(fn, transforms=[StreamParallelization()], executors=[thunder.pytorch_executor,])
actual = njfn(t0s, a)

thunder.last_traces(njfn)[-1].save_trace("stream_trc.py")

torch.testing.assert_close(actual, expected)

import triton
print(triton.testing.do_bench(fn=lambda:jfn(t0s, a)))
print(triton.testing.do_bench(fn=lambda:njfn(t0s, a)))

print(torch.utils.benchmark.Timer("jfn(t0s, a)", globals={"jfn": jfn, "t0s": t0s, "a": a}).blocked_autorange(min_run_time=2))
print(torch.utils.benchmark.Timer("njfn(t0s, a)", globals={"njfn": njfn, "t0s": t0s, "a": a}).blocked_autorange(min_run_time=2))

# print(torch.utils.benchmark.Timer("jfn()", globals={"jfn": lambda: None}).blocked_autorange(min_run_time=2))
# print(torch.utils.benchmark.Timer("njfn()", globals={"njfn": lambda: torch.cuda.Stream()}).blocked_autorange(min_run_time=2))
