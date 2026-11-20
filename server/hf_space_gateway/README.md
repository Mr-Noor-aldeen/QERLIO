---
title: QERLIO ZeroGPU Gateway
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: apache-2.0
---

# QERLIO ZeroGPU Gateway

Open-Source multi-model OpenAI-compatible inference gateway for **NousResearch/Hermes-4.3-36B** and **ZAQERIS/ZAQERIS-7B** running on Hugging Face ZeroGPU (NVIDIA H100 80GB VRAM).

## Features
- Full OpenAI API Compatibility: `POST /v1/chat/completions` (Streaming & Non-Streaming).
- Model Catalog: `GET /v1/models`.
- Hybrid Reasoning Mode: Emits and formats `<think> ... </think>` blocks.
- Structured Tool Calling: Parses `<tool_call>` schemas.
- Zero Server Expenses: Dynamically allocated on Hugging Face ZeroGPU infrastructure.

## Usage with OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-space-name>.hf.space/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="NousResearch/Hermes-4.3-36B",
    messages=[{"role": "user", "content": "Explain hybrid-mode reasoning."}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```
