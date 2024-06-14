import math
import os
import sys
from typing import ClassVar
import multiprocessing as mp
import sys
import tempfile
from functools import wraps

from thunder.core import devices

import pytest
import torch.distributed as tdist
import torch
import torch.nn as nn

try:
    import expecttest
    import hypothesis
except ImportError:
    raise ImportError(
        "Required packages of `expecttest` and/or `hypothesis` are missing. "
        "Install them with `pip install expecttest hypothesis`"
    )
from torch.testing._internal import common_distributed, common_utils


__all__ = [
    "new_gelu",
    "ToyModel",
    "DataParallelTestCase",
]


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ToyModel(nn.Module):
    """Linear(12, 12) -> gelu -> Linear(12, 8)."""

    N_IN: ClassVar[int] = 12
    N_HIDDEN: ClassVar[int] = 16
    N_OUT: ClassVar[int] = 8
    LAYER_NAMES: ClassVar[tuple[str, ...]] = ("net2", "net1")

    def __init__(self, bias: bool = True):
        super().__init__()
        self.net1 = nn.Linear(ToyModel.N_IN, ToyModel.N_HIDDEN, bias=bias)
        self.net2 = nn.Linear(ToyModel.N_HIDDEN, ToyModel.N_OUT, bias=bias)

    def forward(self, x):
        return self.net2(new_gelu(self.net1(x)))


# note(crcrpar): How to write a test with `DDP`
# Just add a method to :class:`CompileDDPTest`. The class is responsible for
#     - calling `torch.distributed.init_process_group` with NCCL backend
#     - setting rank to each process group / device
# so what you'd need to do is to prepare a model and tensors, wrap the model with DDP, and
# `thunder.jit` the original model or the DDP'd model, and do some computation and/or
# examine the traces of the `thunder.jit`d.
# If you force a test to be run with >2 GPUs for a test, you might want to inherit `CompileDDPTest`
# and modify `world_size` to e.g. `max(torch.cuda.device_count(), 2)`.
# note(crcrpar): Why inheriting `common_distributed.MultiProcessTestCase`?
# When we're quite sure that we would only use `pytest` instead of `unittest`,
# IIUC it's possible to run a test that is dependent on `DistributedDataParallel` and/or
# `torch.distributed` by running the test file with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html),
# but I don't think (a) it's quite intuitive to require `torchrun` explicitly to run a test and
# (b) it's quite friendly to our CI as it's currently simply runs `pytest thunder/tests`.
# I would say it's feasible to write a test with `torch.distributed` by using `torch.multiprocessing`,
# but it would require us to make the function which defines the test logic picklable and would
# lead to boilerplate test functions.
# Ref: https://github.com/NVIDIA/apex/blob/7b2e71b0d4013f8e2f9f1c8dd21980ff1d76f1b6/apex/transformer/testing/distributed_test_base.py#L22
class DataParallelTestCase(common_distributed.MultiProcessTestCase):
    DISTRIBUTED_BACKEND = "nccl"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        torch.cuda.empty_cache()
        super().tearDown()

    # note(crcrpar): This means the world_size is up to two.
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @property
    def init_method(self):
        return f"{common_utils.FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        torch.distributed.init_process_group(
            init_method=self.init_method,
            backend=self.DISTRIBUTED_BACKEND,
            world_size=self.world_size,
            rank=self.rank,
        )

        local_rank = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

        torch.distributed.barrier()
        try:
            self.run_test(test_name, pipe)
        except Exception:
            raise
        finally:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        sys.exit(0)


# Wraps a function so that it becomes one process of several executing the test
#   See test_native_ddp and its helper _test_native_ddp_helper below for an example
#   of how to use this wrapper.
# NOTE This actually requires wrapping a stub, because the test framework manipulates
#   functions in a way that does not allow them to be pickled.
#   The actual logic must be implemented in a helper that can be pickled.
# NOTE Tests wrapped with ddp_wrapper can be invoked directly, but you must invoke them
#   like:
#   if __name__ == '__main__':
#       test_ddp.test_native_ddp_TorchEx_cpu_float32()
class ddp_wrapper:
    def __init__(self, name, fn):
        self.fn = fn
        self.__name__ = name

    def __call__(self, test_stub):
        if not tdist.is_available():
            pytest.skip("This test requires torch.distributed be available")

        # Creates a temporary file for process group discovery
        FILE_SCHEMA: str = "file://"
        if sys.platform == "win32":
            FILE_SCHEMA = "file:///"
        file_name = tempfile.NamedTemporaryFile(delete=False).name
        init_method = f"{FILE_SCHEMA}{file_name}"

        @wraps(test_stub)
        def test_fn(executor, devices, dtype, **kwargs):
            world_size = len(devices)
            input_data = []

            for rank in range(world_size):
                process_data = (init_method, world_size, rank, executor, devices[rank], dtype, kwargs)
                input_data.append(process_data)

            ctx = mp.get_context("spawn")
            pool = ctx.Pool(world_size)

            def callback(result):
                pass

            def error_callback(ex):
                # NOTE: Don't raise the exception here, because it will be
                # raised in the main process. Raising it here will cause a
                # deadlock.
                pass

            # The seconds to wait before the pool tasks complete
            TIMEOUT: int = 30
            try:
                results_future = pool.map_async(self.fn, input_data, 1, callback, error_callback)
                results = results_future.get(TIMEOUT)
            finally:
                pool.close()
                pool.join()

            # Raises the first assertion if any occurred
            root_results = results[0]
            if len(root_results) > 0:
                raise (root_results[0])

        return test_fn


# Configures PyTorch's default process group, must be called at the start of each
#   distributed process
def init_per_process_distributed(
    init_method: str, devicetype: devices.DeviceType, world_size: int, rank: int
) -> tdist.ProcessGroup:
    backend: str
    if devicetype is devices.DeviceType.CUDA:
        backend = "nccl"
    elif devicetype is devices.DeviceType.CPU:
        backend = "gloo"
    else:
        raise ValueError(f"Unknown devicetype {devicetype}")

    tdist.init_process_group(init_method=init_method, backend=backend, world_size=world_size, rank=rank)

    # NOTE _get_default_group is not a public PyTorch function, but there is no
    #   public mechanism to acquire the default process group, which is specified
    #   in operations by setting process_group=None.
    #   Actually acquiring the default ProcessGroup is not typically necessary, but
    #   thunder doesn't like to model primitives with implicit defaults,
    #   so we want to pass the ProcessGroup explicitly
    return tdist.distributed_c10d._get_default_group()
