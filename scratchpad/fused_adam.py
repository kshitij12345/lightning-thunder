import torch
from torch import Tensor
from typing import List, Union, Optional
import math
import thunder
from torch.utils.benchmark import Timer
from thunder.tests.litgpt_model import Config, GPT


def _dispatch_sqrt(x: float):
    if isinstance(x, Tensor):
        return x.sqrt()
    return math.sqrt(x)


def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t = thunder.prims.copy_(step_t + 1, step_t)

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg = thunder.prims.copy_(exp_avg * (beta1) + grad * (1 - beta1), exp_avg)
        # exp_avg = exp_avg * (beta1) + grad * (1 - beta1)

        exp_avg_sq = thunder.prims.copy_((exp_avg_sq * beta2) + (1 - beta2) * grad * grad, exp_avg_sq)
        # exp_avg_sq = (exp_avg_sq * beta2) + (1 - beta2) * grad * grad

        step = step_t
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt) + (eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + (eps)

        param = thunder.prims.copy_(param + (-step_size) * exp_avg / denom, param)


def get_jit_state(jit_params):
    steps = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = []
    for param in jit_params:
        step = torch.tensor(0.0, dtype=torch.float, device=device)
        exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
        exp_avg_sq = torch.zeros_like(param)
        max_exp_avg_sq = None  # torch.zeros_like(param, memory_format=torch.preserve_format)

        steps.append(step)
        exp_avgs.append(exp_avg)
        exp_avg_sqs.append(exp_avg_sq)
        max_exp_avg_sqs.append(max_exp_avg_sq)

    return steps, exp_avgs, exp_avg_sqs, max_exp_avg_sqs


def jit_step(optim_func, jit_params, jit_state):
    jit_grads = [param.grad for param in jit_params]
    steps, exp_avgs, exp_avg_sqs, max_exp_avg_sqs = jit_state
    # with torch.no_grad():
    optim_func(
        jit_params,
        jit_grads,
        exp_avgs=exp_avgs,
        exp_avg_sqs=exp_avg_sqs,
        max_exp_avg_sqs=max_exp_avg_sqs,
        state_steps=steps,
        grad_scale=None,
        found_inf=None,
        amsgrad=False,
        has_complex=False,
        beta1=0.9,
        beta2=0.999,
        lr=0.001,
        weight_decay=0,
        eps=1e-8,
        maximize=False,
        capturable=False,
        differentiable=False,
    )


device = "cuda"
dim = 1024
n_param = 100
params = [torch.randn(dim, dim, device=device, requires_grad=True) for _ in range(n_param)]

# model_name = "open_llama_3b"
# m = GPT(Config.from_name(model_name)).to(device)
# params = list(m.parameters())
# params = params[:50]

with torch.no_grad():
    orig_param = params[0].clone().detach()

with torch.no_grad():
    jit_params = [param.clone().detach() for param in params]

for param in jit_params:
    param.requires_grad_(True)


def computation_and_backward(params):
    result = torch.empty_like(params[0])
    for param in params:
        result = result + param

    result.sum().backward()


def attach_grads(params):
    for param in params:
        param.grad = torch.randn_like(param)


def copy_grads(params, jit_params):
    for param, jit_param in zip(params, jit_params):
        jit_param.grad = param.grad.clone().detach()


adam = torch.optim.Adam(params, fused=True)

# computation_and_backward(params)
# computation_and_backward(jit_params)

attach_grads(params)
copy_grads(params, jit_params)

# 2 iterations
adam.step()
adam.step()

# # thunder.set_execution_callback_file("foo.py")
optim_func = thunder.jit(_single_tensor_adam)
jit_state = get_jit_state(jit_params)

# 2 iterations
jit_step(optim_func, jit_params, jit_state)
jit_step(optim_func, jit_params, jit_state)

torch.testing.assert_close(params, jit_params)

# print(params[0][0], jit_params[0][0])

# Verify that params have changed.
# This should crash
# torch.testing.assert_close(params[0], orig_param)

native_time = Timer(stmt="adam.step()", globals={"adam": adam}).timeit(number=100)
jit_time = Timer(
    stmt="jit_step(optim_func, jit_params, jit_state)",
    globals={"jit_step": jit_step, "jit_params": jit_params, "jit_state": jit_state, "optim_func": optim_func},
).timeit(number=100)

# Sanity after 100 iterations
torch.testing.assert_close(params, jit_params, rtol=1e-4, atol=1e-4)
# print(params[0][0], jit_params[0][0])

print(native_time)
print(jit_time)


exec_trace = thunder.last_traces(optim_func)[-1]

with open("generated_thunder_trace.py", "w") as f:
    f.write(str(exec_trace))

with open("generated_fusion_defintion.py", "w") as f:
    f.write(str(exec_trace.python_ctx()["nvFusion0"].last_used))

with open("generated_kernels.cu", "w") as f:
    f.write(exec_trace.python_ctx()["nvFusion0"].last_used.last_cuda_code())

print("Done")
