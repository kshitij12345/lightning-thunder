import torch.fx
from thunder.tests.framework import instantiate, NOTHING, DynamoThunderExecutor
from thunder import dtypes
from thunder.dynamo import ThunderCompiler
from thunder import last_traces

import torch
import pytest


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),),
)
def test_basic(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    backend = ThunderCompiler()
    x = torch.ones(2, dtype=dtype, device=device, requires_grad=True)

    @torch.compile(backend=backend, dynamic=dynamic)
    def func(x):
        x = torch.sin(x)
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1

    out = func(x)

    # out should have grad_fn and its name should be ThunderFunctionBackward
    assert out.grad_fn is not None
    assert out.grad_fn.name() == "ThunderFunctionBackward"

    # # We record the GraphModules that was compiled by ThunderCompiler
    # assert len(backend.thunder_to_gm) == 2
    # thunder_func, gm = list(backend.thunder_to_gm.items())[0]
    # assert isinstance(gm, torch.fx.GraphModule)

    # # This shouldn't be empty
    # assert last_traces(thunder_func)


def test_basic_splitter():
    dynamic = False
    x = torch.ones(2, 2, requires_grad=True)

    backend = ThunderCompiler()

    def func(x):
        x = x.exp()
        return torch.sin(x) + torch.cos(x) + torch.linalg.matrix_power(x, 2)

    cfunc = torch.compile(func, backend=backend, dynamic=dynamic)
    expected = torch.compile(func, dynamic=False)(x)
    actual = cfunc(x)

    torch.testing.assert_close(actual, expected)


# def test_splitter_unsupported_ctx():
#     dynamic = False
#     x = torch.ones(2, requires_grad=True)

#     backend = ThunderCompiler()

#     def func(x):
#         x = x + 2
#         with torch.autocast("cpu"):
#             y = torch.sin(x)
#             return torch.matmul(x, y)

#     expected = torch.compile(func, dynamic=False)(x)

#     cfunc = torch.compile(func, backend=backend, dynamic=dynamic)
#     actual = cfunc(x)


#     torch.testing.assert_close(actual, expected)


# def test_splitter_unsupported_ctx_with_graph_break():
#     dynamic = False
#     x = torch.ones(2, 2, requires_grad=True)

#     backend = ThunderCompiler()

#     def func(x):
#         x = x + 2
#         with torch.autocast("cpu"):
#             y = torch.sin(x)
#             torch._dynamo.graph_break()
#             return torch.matmul(x, y)

#     expected = torch.compile(func, dynamic=False)(x)
#     cfunc = torch.compile(func, backend=backend, dynamic=dynamic)
#     actual = cfunc(x)

#     torch.testing.assert_close(actual, expected)
