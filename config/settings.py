import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Path to base model (if downloaded)
    parent_dir = str(Path(__file__).resolve().parent.parent)
    model_path = f"{parent_dir}/models/base"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"

    # KV cache, device and output
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    fp16_window: int =  128
    int8_window: int = 256
    max_new_tokens: int = 150
    attn_sinks: int = 40
