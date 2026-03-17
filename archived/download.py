from huggingface_hub import snapshot_download
import os
import subprocess

# Create models/base directory
# os.makedirs("models/base", exist_ok=True)

# print("Downloading...")
# snapshot_download(
#     repo_id="Qwen/Qwen2.5-0.5B-Instruct",
#     local_dir="models/base",
#     local_dir_use_symlinks=False
# )
# print("Done! Model saved to models/base")


# Step 1: Download original model
os.makedirs("models/base", exist_ok=True)

print("📥 Step 1: Downloading Qwen2.5-0.5B-Instruct...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="models/base",
    local_dir_use_symlinks=False
)

# Step 3: Export to ONNX
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model = ORTModelForCausalLM.from_pretrained(
    "models/base",
    export=True,
    use_cache=True
)
tokenizer = AutoTokenizer.from_pretrained("models/base")

# Save
model.save_pretrained("models/onnx")
tokenizer.save_pretrained("models/onnx")