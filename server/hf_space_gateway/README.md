---
title: QERLIO Remote API Gateway
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: apache-2.0
---

# QERLIO Remote API Gateway (Template Stub)

OpenAI-compatible FastAPI/Gradio HTTP gateway scaffold (`POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`) with optional bearer-token authentication and a configurable 7-day trial expiration check.

## Scope & Honest Status
- **HTTP Gateway Scaffold Only:** This service provides the HTTP routing, secret token verification (`QERLIO_GATEWAY_SECRET`), and trial expiration timer (`GATEWAY_LAUNCH_TIMESTAMP`).
- **No Model Weights Attached by Default:** Until a live inference engine is connected, `/v1/chat/completions` returns an explicit stub notice (`"simulated": true`, `"usage": null`) and does not execute or impersonate any model.
- **Model Metadata Listed:** Lists `NousResearch/Hermes-4.3-36B` (`ByteDance Seed-OSS-36B-Base` / `SeedOssForCausalLM`, 524,288 context window; GGUF repository: `NousResearch/Hermes-4.3-36B-GGUF`) and `ZAQERIS/ZAQERIS-7B`.

## Usage with OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7860/v1",
    api_key="your-gateway-secret"
)

response = client.chat.completions.create(
    model="NousResearch/Hermes-4.3-36B",
    messages=[{"role": "user", "content": "Ping gateway stub."}],
    stream=False
)
print(response.choices[0].message.content)
```
