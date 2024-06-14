import os
import unittest

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.testing import assert_close

import thunder
import thunder.torch as ltorch
from thunder.core import devices
from thunder.distributed import FSDPType
from thunder.tests.framework import TorchExecutor
from thunder.tests.framework import instantiate
from thunder.tests.distributed.helper import ddp_wrapper, init_per_process_distributed

from thunder.executors.transformer_engineex import (
    transformer_engine_ex,
    TE_AVAILABLE,
    te_sync_fp8_meta_bwd,
)

is_fp8_supported: bool = False
# This will be correctly updated below when TE Engine is installed
# and if the current environment doesn't support FP8.
fp8_support_reason: str = ""
if TE_AVAILABLE:
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch.fp8 import check_fp8_support, FP8GlobalStateManager

    is_fp8_supported, fp8_support_reason = check_fp8_support()


def _test_ddp_transformer_engine(input_data):
    # Test Description: We run a dummy training loop for a simple `Linear(Relu(Linear(x)))`
    # model with thunder (using TE executor) and with PyTorch eager + TE
    # and verify that the weights have converged to same value and
    # fp8 meta state is same after `n_iter`.
    init_method, world_size, rank, executor, device, dtype, _unused_kwargs = input_data
    devicetype = devices.device_from_string(device).devicetype
    _unused_dtype = ltorch.to_torch_dtype(dtype)
    init_per_process_distributed(init_method, devicetype, world_size, rank)

    torch.cuda.set_device(rank)

    dim = 256
    n_iter = 10

    class ThunderModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(dim, dim, bias=False)
            self.fc2 = torch.nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.fc2(torch.nn.functional.relu(self.fc1(x)))

    # Weights
    fc1_weight = torch.randn(dim, dim, requires_grad=True).cuda()
    fc2_weight = torch.randn(dim, dim, requires_grad=True).cuda()

    # Inputs (different input on different rank).
    if rank == 0:
        x = torch.arange(dim * dim, dtype=torch.float).view(dim, dim).cuda()
    if rank == 1:
        x = torch.randn(dim, dim).cuda() * 100

    thunder_model = ThunderModel().cuda()
    thunder_model.fc1.weight.data = fc1_weight.clone()
    thunder_model.fc2.weight.data = fc2_weight.clone()

    jit_model = thunder.jit(
        thunder.distributed.ddp(thunder_model),
        executors=[
            transformer_engine_ex,
        ]
        + executor.executors_list(),
    )

    optim = torch.optim.SGD(thunder_model.parameters())

    for _ in range(n_iter):
        o = jit_model(x).sum()
        o.backward()
        optim.step()
        optim.zero_grad()

    # See https://github.com/NVIDIA/TransformerEngine/issues/814
    FP8GlobalStateManager.reset()

    class TEModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = TELinear(dim, dim, bias=False)
            self.fc2 = TELinear(dim, dim, bias=False)

        def forward(self, x):
            return self.fc2(torch.nn.functional.relu(self.fc1(x)))

    te_model = TEModel().cuda()
    te_model.fc1.weight.data = fc1_weight.clone()
    te_model.fc2.weight.data = fc2_weight.clone()

    ddp_model = DDP(te_model)

    optim = torch.optim.SGD(te_model.parameters())

    for _ in range(n_iter):
        with fp8_autocast():
            o = ddp_model(x).sum()

        o.backward()
        optim.step()
        optim.zero_grad()

    thunder_to_te_layer_map = {"te_linear_0": te_model.fc1, "te_linear_1": te_model.fc2}

    fwd_traces = thunder.last_traces(jit_model)

    def is_same_across_ranks(t):
        t_clone = t.clone()
        torch.distributed.all_reduce(t_clone, op=torch.distributed.ReduceOp.AVG)
        assert_close(t, t_clone)

    # Compare the state of the two models.
    comparison_exceptions = []
    for bound_symbol in fwd_traces[-1].bound_symbols:
        if "te_linear" in bound_symbol.sym.name:
            thunder_fp8_meta = bound_symbol._call_ctx[bound_symbol.sym.name].func.fp8_meta
            te_fp8_meta = thunder_to_te_layer_map[bound_symbol.sym.name].fp8_meta
            try:
                # fwd tensor history
                assert_close(thunder_fp8_meta["scaling_fwd"].scale, te_fp8_meta["scaling_fwd"].scale)
                assert_close(thunder_fp8_meta["scaling_fwd"].scale_inv, te_fp8_meta["scaling_fwd"].scale_inv)
                assert_close(thunder_fp8_meta["scaling_fwd"].amax_history, te_fp8_meta["scaling_fwd"].amax_history)
                # bwd tensor history
                assert_close(thunder_fp8_meta["scaling_bwd"].scale, te_fp8_meta["scaling_bwd"].scale)
                assert_close(thunder_fp8_meta["scaling_bwd"].scale_inv, te_fp8_meta["scaling_bwd"].scale_inv)
                assert_close(thunder_fp8_meta["scaling_bwd"].amax_history, te_fp8_meta["scaling_bwd"].amax_history)

                # This has to be on all ranks so that the computation is not blocked
                is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale)
                is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale_inv)
                # NOTE: TE forward tensor meta-data sync
                # Syncing of FP8 meta-data happens in two step in the forward pass.
                # 1. When we enter the fp8_autocast(), all the forward fp8 meta-data
                # in global buffer is synced.
                # See: https://github.com/NVIDIA/TransformerEngine/blob/6a9edc38bf9b941b7d369af5103fa8fe0b121d61/transformer_engine/pytorch/fp8.py#L409-L412
                # 2. Post this, in the forward pass of the module in `prepare_forward`,
                # we read from the global-buffer the synced meta-data.
                # See: https://github.com/NVIDIA/TransformerEngine/blob/6a9edc38bf9b941b7d369af5103fa8fe0b121d61/transformer_engine/pytorch/module/base.py#L539-L545
                # However, at the end of this forward pass, we have seen new inputs and outputs. Their amax are recorded on
                # 0th row of `amax_history` (which will be synced only in the next forward pass).
                # So, here we check that every row except for `0` is same.
                is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].amax_history[1:])
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale)
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale_inv)
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].amax_history)
            except Exception as e:
                # Return exceptions only for rank==0
                if rank == 0:
                    comparison_exceptions.append(e)

    # Compare weights after `n_iters`
    try:
        assert_close(thunder_model.fc1.weight, te_model.fc1.weight)
        assert_close(thunder_model.fc2.weight, te_model.fc2.weight)
    except Exception as e:
        # Return exceptions only for rank==0
        if rank == 0:
            comparison_exceptions.append(e)

    return comparison_exceptions


def _test_ddp_transformer_engine_llama_sanity(input_data):
    # Test Description: We run a dummy training loop for a Transformer Model
    # We run a few iterations to see that TransformerEngine doesn't throw internal assertion
    # due to reordering of forward and backward operators.
    # (This test will fail without `_rearrange_transformer_engine_linear` in `torch_autograd.py`)
    # For more details, see docstring for `_rearrange_transformer_engine_linear` in transformer_engine_ex.py.
    from thunder.tests.llama2_model import Transformer, ModelArgs

    init_method, world_size, rank, executor, device, dtype, _unused_kwargs = input_data
    devicetype = devices.device_from_string(device).devicetype
    _unused_dtype = ltorch.to_torch_dtype(dtype)
    init_per_process_distributed(init_method, devicetype, world_size, rank)

    torch.cuda.set_device(rank)
    # data
    batch_size = 2
    max_seq_len = 32
    vocab_size = 32

    model_args = dict(
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    model.to(device)
    x = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
    y = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
    jit_model = thunder.jit(
        thunder.distributed.ddp(model), executors=(transformer_engine_ex,) + thunder.get_default_executors()
    )

    sanity_exceptions = []
    try:
        for _ in range(5):
            out = jit_model(x, y).sum()
            out.backward()

        bwd_exec_trace = thunder.last_backward_traces(jit_model)[-1]

        # Last symbol of the trace should be `return`
        return_sym_idx = len(bwd_exec_trace.bound_symbols) - 1
        assert thunder.core.prims.PrimIDs.RETURN == bwd_exec_trace.bound_symbols[return_sym_idx].sym.id

        # Verify that the symbol to sync backward
        # fp8 metadata is present in backward trace.
        for idx, bsym in enumerate(bwd_exec_trace.bound_symbols):
            if bsym.sym.id == te_sync_fp8_meta_bwd.id:
                # Verify that `te_sync_fp8_meta_bwd` is before the last symbol of the trace
                # which is `return`
                assert idx < return_sym_idx
                break
        else:
            raise RuntimeError("Backward sync symbol not found.")
    except Exception as e:
        sanity_exceptions.append(e)

    if rank == 0:
        return sanity_exceptions
    return None


def _test_fsdp_transformer_engine(input_data):
    # Test Description: We run a dummy training loop for a simple `Linear(Relu(Linear(x)))`
    # model with thunder (using TE executor) and with PyTorch eager + TE
    # and verify that the weights have converged to same value and
    # fp8 meta state is same after `n_iter`.
    init_method, world_size, rank, executor, device, _unused_dtype, kwargs = input_data
    thunder_fsdp_strategy = kwargs["thunder_fsdp_strategy"]
    devicetype = devices.device_from_string(device).devicetype

    # Setting LOCAL_RANK is necessary for thunder.distributed.fsdp
    with unittest.mock.patch.dict(os.environ, {"LOCAL_RANK": str(rank)}):
        init_per_process_distributed(init_method, devicetype, world_size, rank)
        torch.cuda.set_device(rank)

        dim = 256
        n_iter = 10

        class ThunderModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(dim, dim, bias=False)
                self.fc2 = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.relu(self.fc1(x)))

        # Weights
        fc1_weight = torch.randn(dim, dim, requires_grad=True, device="cuda")
        fc2_weight = torch.randn(dim, dim, requires_grad=True, device="cuda")

        # Inputs (different input on different rank).
        if rank == 0:
            x = torch.arange(dim * dim, dtype=torch.float, device="cuda").view(dim, dim)
        if rank == 1:
            x = torch.randn(dim, dim, device="cuda") * 100

        with torch.device("cuda"):
            thunder_model = ThunderModel()
        thunder_model.fc1.weight.data = fc1_weight.clone()
        thunder_model.fc2.weight.data = fc2_weight.clone()

        jit_model = thunder.jit(
            thunder.distributed.fsdp(thunder_model, sharding_strategy=thunder_fsdp_strategy),
            executors=[
                transformer_engine_ex,
            ]
            + executor.executors_list(),
        )

        optim = torch.optim.SGD(thunder_model.parameters())

        for _ in range(n_iter):
            o = jit_model(x).sum()
            o.backward()
            optim.step()
            optim.zero_grad()

        # See https://github.com/NVIDIA/TransformerEngine/issues/814
        FP8GlobalStateManager.reset()

        class TEModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = TELinear(dim, dim, bias=False)
                self.fc2 = TELinear(dim, dim, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.relu(self.fc1(x)))

        with torch.device("cuda"):
            te_model = TEModel()
        te_model.fc1.weight.data = fc1_weight.clone()
        te_model.fc2.weight.data = fc2_weight.clone()

        fsdp_model = FullyShardedDataParallel(te_model, auto_wrap_policy=always_wrap_policy)

        optim = torch.optim.SGD(te_model.parameters())

        for _ in range(n_iter):
            with fp8_autocast():
                o = fsdp_model(x).sum()

            o.backward()
            optim.step()
            optim.zero_grad()

        thunder_to_te_layer_map = {"te_linear_0": te_model.fc1, "te_linear_1": te_model.fc2}

        fwd_traces = thunder.last_traces(jit_model)

        def is_same_across_ranks(t):
            t_clone = t.clone()
            torch.distributed.all_reduce(t_clone, op=torch.distributed.ReduceOp.AVG)
            assert_close(t, t_clone)

        # Compare the state of the two models.
        comparison_exceptions = []
        for bound_symbol in fwd_traces[-1].bound_symbols:
            if "te_linear" in bound_symbol.sym.name:
                thunder_fp8_meta = bound_symbol._call_ctx[bound_symbol.sym.name].func.fp8_meta
                te_fp8_meta = thunder_to_te_layer_map[bound_symbol.sym.name].fp8_meta
                try:
                    # fwd tensor history
                    assert_close(thunder_fp8_meta["scaling_fwd"].scale, te_fp8_meta["scaling_fwd"].scale)
                    assert_close(thunder_fp8_meta["scaling_fwd"].scale_inv, te_fp8_meta["scaling_fwd"].scale_inv)
                    assert_close(thunder_fp8_meta["scaling_fwd"].amax_history, te_fp8_meta["scaling_fwd"].amax_history)
                    # bwd tensor history
                    assert_close(thunder_fp8_meta["scaling_bwd"].scale, te_fp8_meta["scaling_bwd"].scale)
                    assert_close(thunder_fp8_meta["scaling_bwd"].scale_inv, te_fp8_meta["scaling_bwd"].scale_inv)
                    assert_close(thunder_fp8_meta["scaling_bwd"].amax_history, te_fp8_meta["scaling_bwd"].amax_history)

                    # This has to be on all ranks so that the computation is not blocked
                    is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale)
                    is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale_inv)
                    # See NOTE: TE forward tensor meta-data sync
                    is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].amax_history[1:])
                    is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale)
                    is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale_inv)
                    is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].amax_history)
                except Exception as e:
                    # Return exceptions only for rank==0
                    if rank == 0:
                        comparison_exceptions.append(e)

        # Compare weights after `n_iters`
        shard_size = int(dim / world_size)
        fsdp_te_params = tuple(te_model.parameters())
        try:
            assert_close(thunder_model.fc1.weight, fsdp_te_params[0].view(shard_size, dim))
            assert_close(thunder_model.fc2.weight, fsdp_te_params[1].view(shard_size, dim))
        except Exception as e:
            # Return exceptions only for rank==0
            if rank == 0:
                comparison_exceptions.append(e)

        return comparison_exceptions


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # NOTE: Setting `NVTE_TORCH_COMPILE`
        # It is important to set this flag so that TE doesn't use
        # `torch.compile` to fuse a few operations. This is because
        # `torch.compile` creates a new process and that leads to
        # the error : daemonic processes are not allowed to have children
        # when running the tests.
        # With the setting below, we use `torch.jit` for this test suite
        # See: https://github.com/NVIDIA/TransformerEngine/blob/a38b291b0d1b04847e8ab1df8550df642a03a27d/transformer_engine/pytorch/jit.py#L11-L19
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@ddp_wrapper("test_ddp_transformer_engine", _test_ddp_transformer_engine)
def test_ddp_transformer_engine(executor, devices, dtype):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@ddp_wrapper("test_ddp_transformer_engine_llama_sanity", _test_ddp_transformer_engine_llama_sanity)
def test_ddp_transformer_engine_llama_sanity(executor, devices, dtype):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        # NOTE: ddp_wrapper
        pytest.mark.parametrize(
            "thunder_fsdp_strategy",
            (
                FSDPType.ZERO2,
                FSDPType.ZERO3,
            ),
        ),
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@ddp_wrapper("test_fsdp_transformer_engine", _test_fsdp_transformer_engine)
def test_fsdp_transformer_engine(executor, devices, dtype, thunder_fsdp_strategy):
    pass
