import os
import torch
import torch.distributed as tdist
import thunder
import thunder.distributed
from thunder.tests.litgpt_model import GPT, Config

if __name__ == "__main__":
    tdist.init_process_group(backend="nccl")
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", LOCAL_RANK)
    torch.set_default_device(device)

    # config = Config(block_size=256, padded_vocab_size=32000, n_layer=1, n_head=3, head_size=24, n_embd=144, rotary_percentage=1.0, parallel_residual=False, bias=False, norm_class_name='RMSNorm', mlp_class_name='LLaMAMLP', intermediate_size=384)
    config = Config.from_name("open_llama_3b")
    config.n_layer = 2
    with device:
        model = GPT(config)

    from profile_transform import NvtxProfileTransform

    nvtx_profile_transform = NvtxProfileTransform()
    model = thunder.distributed.fsdp(model)
    model = thunder.jit(model, executors=["torch"], post_optimization_transforms=[nvtx_profile_transform])

    input_ids = torch.randint(1, 30010, (128, 256), dtype=torch.long, device=device)

    if LOCAL_RANK == 0:
        torch.cuda.cudart().cudaProfilerStart()
    logits = model(input_ids)
    logits.sum().backward()

    if LOCAL_RANK == 0:
        # print(torch.cuda.max_memory_allocated())
        pro_trace = thunder.last_prologue_traces(model)[-1]
        with open("pro_trace.py", "w") as f:
            f.write(str(pro_trace))
        trace = thunder.last_traces(model)[-1]
        with open("fwd_trace.py", "w") as f:
            f.write(str(trace))
        bwd_trace = thunder.last_backward_traces(model)[-1]
        with open("bwd_trace.py", "w") as f:
            f.write(str(bwd_trace))

        torch.cuda.cudart().cudaProfilerStop()
