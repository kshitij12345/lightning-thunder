import torch

import thunder
from thunder.executors.apex_fused_rms_norm_impl import apex_ex

apex_fused_rms_norm_ex = apex_ex

from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction


def fn(x, weight, normalized_shape, eps):
    return FusedRMSNormAffineMixedDtypesFunction.apply(x, weight, normalized_shape, eps)


device = "cuda"
normalized_shape = (3, 3, 3)
x = torch.randn(3, 3, 3, 3, 3, requires_grad=True, device=device)
weight = torch.randn(*normalized_shape, requires_grad=False, device=device)
eps = 1e-7

fn(x, weight, normalized_shape, eps)
jfn = thunder.jit(fn, executors=[apex_fused_rms_norm_ex])
jfn(x, weight, normalized_shape, eps)

print(thunder.last_traces(jfn)[-1])
print(thunder.last_backward_traces(jfn)[-1])
