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
N_LAYER = 1

cfg: Config = Config.from_name(model_name)
# cfg.n_layer = N_LAYER

with torch.device(device):
    model = GPT(cfg).to(torch.bfloat16)
    # model = model.transformer.h[0].mlp
model.eval().requires_grad_(False)


def benchmark_model(model, inp, name):
    # Warm-up
    # model(inp)
    # torch.cuda.synchronize()

    # torch.cuda.nvtx.range_push(name)
    # model(inp)
    # torch.cuda.nvtx.range_pop()
    timer = torch.utils.benchmark.Timer(
        stmt="model(inp)",
        setup="",
        globals={"model": model, "inp": inp},
        label=f"Llama-3-8B Input Shape {inp.shape}",
        description=name,
    )

    measurement = timer.timeit(number=5)
    print(f"{name} Time taken: {measurement}")
    print()
    return measurement


inp = torch.randint(0, 255, (16, 2048,), device=device)

# class Module(torch.nn.Module):
#     def __init__(self, in_features: int = 64, out_features: int = 256, bias: bool = False):
#         super().__init__()
#         self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.linear(x)
#         return out

# # model = torch.nn.Linear(4096, 6144, device=device).to(torch.bfloat16)
# model = Module(4096, 6144).to(torch.bfloat16).to("cuda")
# model.requires_grad_(False)

# inp = torch.randn(64, 2048, 4096, dtype=torch.bfloat16, device=device)

eager_measurement = benchmark_model(model, inp, "Eager")

compiled_model = thunder.jit(model)

thunder_default_measurement = benchmark_model(compiled_model, inp, "Thunder default")

trcs = thunder.last_traces(compiled_model)
trcs[-1].save_trace("thunder_default_trc.py")

cmodel = torch.compile(model)

torch_compile_measurement = benchmark_model(cmodel, inp, "TorchCompile Default")

def quantization_filter(name, module):
    return isinstance(module, torch.nn.Linear) # and "mlp" in name
    # return "mlp" in name and isinstance(module, torch.nn.Linear)

xforms = [QuantizedLinearTransform(filter_fn=quantization_filter, separate_quantization=True)]
executors = (nvfp4_executor,) + thunder.get_default_executors()
# xforms = []
# executors = None
# executors = () # None

compiled_model = thunder.jit(model, transforms=xforms, executors=executors)

thunder_measurement = benchmark_model(compiled_model, inp, "Thunder + nvFP4")

trcs = thunder.last_traces(compiled_model)
trcs[-1].save_trace("quantized_trc.py")

mm_config = NVFP4MMConfig.DYNAMIC
config = NVFP4InferenceConfig(
        mm_config=mm_config, use_triton_kernel=True
)

# This mutates the model
quantize_(model, config=config)

torchao_measurement = benchmark_model(model, inp, "Torchao")

cmodel = torch.compile(model)

torchao_compile_measurement = benchmark_model(cmodel, inp, "TorchCompile + AO")

compare = torch.utils.benchmark.Compare([eager_measurement, thunder_default_measurement, thunder_measurement,
                                         torch_compile_measurement, torchao_measurement, torchao_compile_measurement])
compare.print()

print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")
