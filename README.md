# Cacheflow: Stateful LLM with Quantized KV Cache Management

Cacheflow is a lightweight research project that keeps long-running conversations affordable by compressing the attention KV cache on the fly. It maintains a high-quality FP16 window for the most recent tokens and quantizes older tokens to INT8, dramatically reducing memory footprint while preserving context.

## Highlights
- Stateful chat with streaming responses via Gradio.
- Sliding-window KV cache: recent tokens in FP16, older in INT8.
- Real-time telemetry: token counts and memory savings.
- Local-first model use; works offline once the model is downloaded.
- Benchmark script to compare against a standard, uncompressed baseline.


## How It Works
- Decoder (`core/decoder.py`):  
  Streams tokens using `AutoModelForCausalLM` and maintains a global position. New KV entries are appended per layer on each step. A small repetition penalty reduces short loops.
- KV Memory Manager (`core/kv_manager.py`):  
  - Each new token is first stored in an FP16 window.  
  - When FP16 exceeds `fp16_window`, the oldest token is quantized to INT8 and moved to an INT8 window.  
  - When INT8 exceeds `int8_window`, the oldest INT8 token is dropped.  
  - On each forward pass, it reconstructs per-layer KV by concatenating INT8-dequantized tokens followed by the FP16 tail.
- Quantizer (`core/kv_quantizer.py`):  
  Performs per-token linear INT8 quantization with per-token scales and dequantization on retrieval.
- Telemetry (`core/telemetry.py`):  
  Tracks FP16/INT8 token counts, total memory use, and estimated memory saved vs. an all-FP16 cache.
- Chat Engine & UI (`core/chat.py`, `ui/app_ui.py`):  
  A simple turn formatter with a Qwen-style system prompt and a Gradio interface that streams partial responses and live telemetry overlays.

## Overview
<div align="center">

```text
┌───────────────────────────────────────────────────────────────────────┐
│                           KV Memory Manager                           │
├───────────────┬───────────────────────────┬───────────────────────────┤
│   System      │       INT8 Cache          │       FP16 Cache          │
│   Prompt      │    (older tokens,         │    (recent tokens,        │
│   (fixed)     │       1 byte each)        │       2 bytes each)       │
├───────────────┼───────────────────────────┼───────────────────────────┤
│   Always      │       FIFO queue          │       FIFO queue          │
│   preserved   │   compressed when full    │      newest first         │
└───────────────┴───────────────────────────┴───────────────────────────┘
```

</div>

> Note: Implementation stores KV on CPU and casts to the model's dtype when used.

## Configuration
All options live in `config/settings.py`:
- `base_model` — Default: `Qwen/Qwen2.5-0.5B-Instruct`.
- `model_path` — Where the model is stored locally. Default: `models/base` inside the repo.
- `device` — `"cpu"` by default; set to `"cuda"` if you have a GPU.
- `dtype` — Inferred from device (BF16 on CPU, FP16 on CUDA).
- `fp16_window` — Count of recent tokens stored in FP16 (default 128).
- `int8_window` — Count of older tokens stored in INT8 (default 256).
- `max_new_tokens` — Maximum tokens to generate per response (default 150).

> Note: By default the decoder uses `local_files_only=True` (fast/offline after download). To skip pre-download and fetch on demand, set `local_files_only=False` for both the tokenizer and model in `core/decoder.py`. This requires internet access (and an HF token if the model you want to use is private) and may increase first-run latency.

## Trade-offs & Design Philosophy

**Speed vs. Memory**  
Cacheflow introduces additional compute overhead from quantization, dequantization, and cache management operations. This results in slightly higher per-token latency compared to vanilla inference. The trade-off is deliberate: exchange a modest speed penalty for dramatically reduced memory growth, enabling conversations that would otherwise crash from OOM errors.

**No Fine-Tuning Required**  
This system operates entirely at inference time and requires no model modifications, fine-tuning, or retraining. Currently, it works out-of-the-box with all "instruct" variants of Qwen 2 and Qwen 2.5 series available on Hugging Face. Simply point it to your model of choice and the compression happens automatically.

**Plug and Play Architecture**  
Cacheflow wraps around existing models without altering their weights or architecture. The compression is applied to the KV cache during generation, meaning you can switch between compressed and vanilla inference by simply changing a few lines of code.

**What You Gain vs. What You Sacrifice**  
- **Gain:** Fixed memory footprint, infinite conversation length, predictable resource usage
- **Sacrifice:** Moderate increase in per-token latency, small quality degradation from INT8 quantization
- **No sacrifice:** Model accuracy, task performance, or need for specialized training


## Quickstart (Local)
0) Configure (optional)  
   Adjust `config/settings.py` to change base model, model path, device, or cache windows.

1) Install the necessary packages:

    ```bash
    pip install -r requirements.txt
    ``` 

2) Download the base model  
This uses `huggingface_hub.snapshot_download` and saves to `models/base` by default.
   
    ```bash
    python download.py
    ```
    Alternatively (no pre-download): set `local_files_only=False` in `core/decoder.py` for both `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`. The model and tokenizer will be fetched automatically on first run.

3) Launch the Gradio app  
    ```bash
    python app.py
    ```
    The server will be accessible at localhost:7860 by default.


## Benchmarks
This compares the performance of Cacheflow’s compressed KV approach with the standard model downloaded earlier.
A small set of prompts is provided in `benchmark/test_prompts.py` to act as a baseline. You can add more prompts to this file to test against different scenarios.

  ```bash
  python run_benchmark.py
  ```
Results are written to `benchmark/results/` by default.

## Tips & Troubleshooting
- Model not found / load fails:  
  Make sure you ran `python download.py` (locally) or configured a pre-build step (Spaces).
- Memory use grows too fast:  
  Reduce `fp16_window` and/or `int8_window` in `config/settings.py` or via the UI.
- Long, rambling answers:  
  Lower `max_new_tokens` or tweak the decoding temperature/top-k in `decoder.py`.
  