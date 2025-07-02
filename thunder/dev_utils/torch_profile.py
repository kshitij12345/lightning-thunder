import torch
from torch.profiler import ProfilerActivity
from torch._C._autograd import DeviceType

import pandas as pd
from transformers import AutoModelForCausalLM, AutoConfig

def profile_cuda_kernels(model_or_fn, inputs, categorize_kernel, warmup_steps=3, profile_steps=1):
    # Warm-up
    for _ in range(warmup_steps):
        model_or_fn(*inputs)

    with torch.autograd.profiler.profile(use_device="cuda", use_kineto=True, record_shapes=False) as prof:
        for _ in range(profile_steps):
            model_or_fn(*inputs)


    kernel_events = []
    event_list = prof.key_averages()
    for event in event_list:
        if event.device_time_total > 0 and event.device_type == DeviceType.CUDA:
            kernel_events.append({"device_type": event.device_type, "device_time_total": event.device_time_total,
                                  "self_device_time_total": event.self_device_time_total, "name": event.key, "count": event.count,
                                  "kernel_type": categorize_kernel(event.key)})

    return kernel_events

def categorize_kernel(name):
    if "gemm" in name:
        return "gemm"
    elif "flash" in name:
        return "attention"
    elif "elementwise" in name:
        return "elementwise"
    elif "reduce" in name:
        return "reduce"
    else:
        return "other"

def generate_kernel_events_excel(kernel_events, output_path):
    COLUMN_ORDER = ["kernel_type", "name", "device_time_total", "count"]

    df = pd.DataFrame(kernel_events)

    with pd.ExcelWriter(output_path) as writer:
        # Group by preserve the kernel_type as index
        df.groupby("kernel_type", group_keys=True)["device_time_total"].sum().to_excel(writer, sheet_name="kernel_events_summary", index=True)

        df.to_excel(writer, sheet_name="all_kernel_events", index=False, columns=COLUMN_ORDER)

        # For each kernel_type, create a sheet in the excel file.
        for kernel_type in df["kernel_type"].unique():
            df[df["kernel_type"] == kernel_type].to_excel(writer, sheet_name=f"{kernel_type}_events", index=False, columns=COLUMN_ORDER)

model_name = "microsoft/Phi-3.5-mini-instruct"
batch_size = 1
input_length = 1024
output_length = 1024

# Load model configuration
config = AutoConfig.from_pretrained(model_name)

config.num_hidden_layers = 2

with torch.device("cuda"):
    model = AutoModelForCausalLM.from_config(config)

input_ids = torch.randint(0, config.vocab_size, (batch_size, input_length), device="cuda")

kernel_events = profile_cuda_kernels(model, (input_ids,), categorize_kernel)
generate_kernel_events_excel(kernel_events, f"{model_name.replace("/", "_")}_kernel_events.xlsx")
