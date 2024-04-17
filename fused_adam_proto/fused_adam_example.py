import torch
from torch import Tensor
from typing import List, Union, Optional
import math
import thunder
from torch.utils.benchmark import Timer
from thunder.tests.litgpt_model import Config, GPT
from thunder.core.transforms import value_and_grad


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
        device = param.device
        step = torch.tensor(0.0, dtype=torch.float, device=device)
        exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
        exp_avg_sq = torch.zeros_like(param)
        max_exp_avg_sq = None  # torch.zeros_like(param, memory_format=torch.preserve_format)

        steps.append(step)
        exp_avgs.append(exp_avg)
        exp_avg_sqs.append(exp_avg_sq)
        max_exp_avg_sqs.append(max_exp_avg_sq)

    return steps, exp_avgs, exp_avg_sqs, max_exp_avg_sqs


def jit_step(optim_func, jit_params, jit_grads, jit_state):
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


class ThunderAdam:
    def __init__(self, params) -> None:
        self.params = params
        self.state = get_jit_state(params)

    def step(self, grads):
        jit_step(_single_tensor_adam, self.params, grads, self.state)


def model(x, w1):
    return (x + w1).sum()


inp = torch.randn(1, 1)
w1 = torch.randn(1, 1)

adam = ThunderAdam([w1])


def train_step(x, w1):
    output, grads = value_and_grad(model)(x, w1)
    # Grads are None because of a bug
    # We create our own grads.
    # Tie output here so that value_and_grad computation is not
    # eliminated by DCE
    grad = torch.randn_like(w1) + output
    adam.step(
        [
            grad,
        ]
    )


jit_train_step = thunder.jit(train_step)

print(w1)
for _ in range(100):
    jit_train_step(inp, w1)
print(w1)

print(thunder.last_traces(jit_train_step)[-1])
