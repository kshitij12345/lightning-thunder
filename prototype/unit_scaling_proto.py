import thunder
import torch
from thunder.executors.torchex import ex, TensorProxy
from thunder.core.transforms import get_grad, put_grad
from statistics import geometric_mean


def meta_scale(input, fwd_scale, bwd_scale):
    return TensorProxy(like=input)


def scale_impl(input, fwd_scale, bwd_scale):
    return input * fwd_scale


scale = ex.register_operator("scale", meta=meta_scale, fn=scale_impl)


def scale_grad(input, fwd_scale, bwd_scale):
    scaled_input = scale(input, fwd_scale, bwd_scale)
    g = get_grad(scaled_input)
    put_grad(input, g * bwd_scale)
    return scaled_input


ex.register_implementation(scale, scale, grad_transform=scale_grad)


def foo(x):
    return scale(x, 0.5, 0.5)


jfoo = thunder.jit(foo)

x = torch.ones(3, 3, requires_grad=True)

o = jfoo(x)

o.backward(torch.ones(3, 3) * 2)

with torch.no_grad():
    torch.testing.assert_close(o, x * 0.5)
    torch.testing.assert_close(x.grad, torch.ones_like(x))

# Test for linear


def print_var(x):
    with torch.no_grad():
        print(x.var())


x = torch.randn(32, 32, requires_grad=True)
w = torch.randn(32, 32, requires_grad=True)
g = torch.randn(32, 32)


def linear_wrapped(x, w):
    return torch.nn.functional.linear(x, w)


o = thunder.jit(linear_wrapped)(x, w)
o.backward(g)

print("UNSCALED")
print_var(o)
print_var(x.grad)
print_var(w.grad)

# Clear grads
x.grad = None
w.grad = None


def linear_unit_scaled(x, w):
    fan_out, fan_in = w.shape
    batch_size = x.numel // fan_in

    output_scale = fan_in**-0.5
    grad_input_scale = fan_out**-0.5
    grad_weight_scale = batch_size**-0.5

    output_scale = grad_input_scale = geometric_mean([output_scale, grad_input_scale])

    x_scaled = scale(x, 1, grad_input_scale)
    w_scaled = scale(w, 1, grad_weight_scale)
    return scale(torch.nn.functional.linear(x_scaled, w_scaled), output_scale, 1)


o = thunder.jit(linear_unit_scaled)(x, w)
o.backward(g)

print("SCALED")
print_var(o)
print_var(x.grad)
print_var(w.grad)
