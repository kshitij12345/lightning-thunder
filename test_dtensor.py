# torchrun --nnodes 1 --nproc-per-node 2 test_dtensor.py
import torch.nn as nn
import torch
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
import os
from thunder.dynamo import thunderfx
import torch.distributed as dist

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
num_devices = 2
mesh = DeviceMesh("cuda", list(range(num_devices)))

hidden_size = 16

def model(x, w, b):
    return torch.nn.functional.linear(x, w)

# weight = nn.Parameter(distribute_tensor(torch.randn(16, 16), mesh, [Shard(0)]))
# bias = nn.Parameter(distribute_tensor(torch.randn(16), mesh, [Shard(0)]))

# in_dtensor = distribute_tensor(torch.randn(4, 16), mesh, [Replicate()])

# expected = torch.compile(model)(in_dtensor, weight, bias)
# actual = thunderfx(model, nv_enable_matmul=True, nv_enable_linear=True)(in_dtensor, weight, bias)

def model(x):
    return x + 1

in_tensor = torch.randn(num_devices, 4)
mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

print(in_dtensor.shape)

expected = torch.compile(model)(in_dtensor)
actual = thunderfx(model, nv_enable_matmul=True, nv_enable_linear=True)(in_dtensor)

torch.testing.assert_close(actual.to_local(), expected.to_local())
