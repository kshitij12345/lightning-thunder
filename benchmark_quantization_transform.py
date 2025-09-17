import transformers
import thunder
import time
from litgpt import Config, GPT
import torch
import torch.utils.benchmark
from quantization_transform import QuantizedLinearTransform, nvfp4_executor
from torchao.quantization import quantize_
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4InferenceConfig,
    NVFP4MMConfig,
)


model_name = "Llama-3-8B"
device = "cuda"
N_LAYER = 10

cfg: Config = Config.from_name(model_name)
cfg.n_layer = N_LAYER

with torch.device(device):
    model = GPT(cfg).to(torch.bfloat16)
model.eval().requires_grad_(False)


def benchmark_model(model, inp, name):
    timer = torch.utils.benchmark.Timer(
        stmt="model(inp)",
        setup="",
        globals={"model": model, "inp": inp},
    )

    measurement = timer.timeit(number=10)
    print(f"{name} Time taken: {measurement}")


inp = torch.randint(0, 255, (1, 2048,), device=device)

benchmark_model(model, inp, "Eager")

def quantization_filter(name, module):
    return isinstance(module, torch.nn.Linear)
    # return "mlp" in name and isinstance(module, torch.nn.Linear)

xforms = [QuantizedLinearTransform(filter_fn=quantization_filter)]
executors = (nvfp4_executor,) + thunder.get_default_executors()

compiled_model = thunder.jit(model, transforms=xforms, executors=executors)

benchmark_model(compiled_model, inp, "Thunder")

trcs = thunder.last_traces(compiled_model)
trcs[-1].save_trace("quantized_trc.py")

mm_config = NVFP4MMConfig.DYNAMIC
config = NVFP4InferenceConfig(
        mm_config=mm_config, use_triton_kernel=False
)

# This mutates the model
quantize_(model, config=NVFP4InferenceConfig())

benchmark_model(model, inp, "Torchao")

cmodel = torch.compile(model)

benchmark_model(cmodel, inp, "TorchCompile + AO")
