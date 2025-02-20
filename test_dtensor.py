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


def model(x, w):
    return torch.nn.functional.linear(x, w)


weight = distribute_tensor(torch.randn(16, 16, requires_grad=False), mesh, [Shard(0)])
bias = distribute_tensor(torch.randn(16, requires_grad=False), mesh, [Shard(0)])

in_dtensor = distribute_tensor(torch.randn(4, 16, requires_grad=False), mesh, [Replicate()])

expected = torch.compile(model)(in_dtensor, weight)
tmodel = thunderfx(model)
actual = tmodel(in_dtensor, weight)

# def model(x):
#     return x + 1

# in_tensor = torch.randn(num_devices, 4)
# mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
# in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

# print(in_dtensor.shape)

# expected = torch.compile(model)(in_dtensor)
# actual = thunderfx(model, nv_enable_matmul=True, nv_enable_linear=True)(in_dtensor)

torch.testing.assert_close(actual.to_local(), expected.to_local())

# g_o = distribute_tensor(torch.ones(4, 16), mesh, [Shard(0)])
# expected_g = torch.autograd.grad(expected, (in_dtensor, weight), g_o,)
# actual_g = torch.autograd.grad(actual, (in_dtensor, weight), g_o)

if LOCAL_RANK == 0:
    tmodel.last_traces[-1].save_trace("dtensor_trc.py")
