# torchrun --local-ranks-filter=0 --nnodes 1 --nproc-per-node 2 test_from_local.py
import torch.nn as nn
import torch
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
from torch.distributed.tensor.parallel import parallelize_module, ParallelStyle, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import DTensor
from torch.distributed.tensor._dtensor_spec import TensorMeta, DTensorSpec
from thunder.torch.experimental.dtensor_proxy import DTensorProxy, create_dtensor_proxy_from_proxies
from thunder.core.proxies import AnyProxy
from thunder.torch.experimental.dtensor_torch_and_prims import run_with_fake_tensor, dtensor_from_local_prim, dtensor_redistribute_prim
import os
from thunder.dynamo import thunderfx
import thunder
from thunder.extend import TemporaryExecutor
import torch.distributed as dist
from typing import Optional
from thunder.core.jit_ext import interpreter_needs_wrap, ensure_recursive_proxies, record_source_loc_in_symbol_header, wrap, ProvenanceRecord, PseudoInst

import thunder.examine

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
num_devices = 2
mesh = DeviceMesh("cuda", list(range(num_devices)))

hidden_size = 16

tmp_executor = TemporaryExecutor()
tmp_executor._lookasides[DTensor.from_local] = dtensor_from_local_prim

with torch.device(f"cuda:{LOCAL_RANK}"):
    def parallel_model(x):
        return DTensor.from_local(x, mesh, [Shard(1)])

    i = torch.randn(hidden_size, hidden_size)
    o = parallel_model(i)
    a = thunder.jit(parallel_model, executors=(tmp_executor,))(i)

    print("TESTING EQUALITY")
    torch.testing.assert_close(o, a)

torch.distributed.destroy_process_group()
