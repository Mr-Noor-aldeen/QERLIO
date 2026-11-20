# QERLIO Runtime

[![CI](https://github.com/ZAQERIS/QERLIO/actions/workflows/ci.yml/badge.svg)](https://github.com/ZAQERIS/QERLIO/actions)
[![Release](https://img.shields.io/github/v/release/ZAQERIS/QERLIO?color=blue)](https://github.com/ZAQERIS/QERLIO/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](sdk/python/)
[![Tests](https://img.shields.io/badge/tests-43%2F43%20passing-success.svg)](tests/)
[![Architectures](https://img.shields.io/badge/architectures-200%20supported-blueviolet.svg)](registry/model_architectures.json)
[![Universal Gateway](https://img.shields.io/badge/gateway-OpenAI%20%7C%20Anthropic%20%7C%20Ollama-purple.svg)](server/local_api/)
[![Agent Mode](https://img.shields.io/badge/agents-ReAct%20%7C%20Tri--Agent%20Sandbox-emerald.svg)](core/agent/)

> **Your AI. Your Device. Your Rules.**  
> *Developed by ZAQERIS AI — Independent Research Group*

QERLIO Runtime is a high-efficiency, privacy-first local AI execution platform engineered for edge devices, laptops, and desktop workstations. It provides an adaptive, vendor-neutral abstraction layer over CPU, GPU, and NPU accelerators with guaranteed reference fallbacks.

---

## What's New in v0.2.0

* **200 Model Architectures Supported:** Complete coverage across 10 distinct architectural families (ZAQERIS Native, Dense Transformers, Mixture of Experts, Reasoning & Dialectic, Code Specialists, State Space Models / Mamba, Vision-Language VLMs, Embeddings & Rerankers, Speech & Audio, and Edge/Micro LLMs).
* **Built-in Agent Mode:** Sandboxed ReAct loop with native tool calling support for ZAQERIS-7B (`<tool_call>`), ZQN-12B (`<call name="..." args="..."/>`), and OpenAI/Anthropic function calling formats.
* **Safe Sandbox Tools:** Calculator, Sandboxed Python Evaluator, Workspace File Search, and Synthetic Web Search.
* **Official ZAQERIS Model Connectors:** Direct local inference for `ZAQERIS/ZAQERIS-7B`, `ZAQERIS/ZQN-12B`, and `ZAQERIS/ZAQERIS_AXIOM` with zero cloud server expenses.
* **Enhanced Developer HUD:** Live architecture explorer with category filtering and interactive Agent Studio at `http://127.0.0.1:8787`.

---

## 200 Supported Model Architectures

QERLIO maintains an authoritative registry of 200 validated model architectures in [`registry/model_architectures.json`](file:///c:/Users/rahim/Downloads/QERLIO/registry/model_architectures.json).

| Architectural Family | Total Count | Representative Architectures |
| :--- | :---: | :--- |
| **ZAQERIS Native** | **3** | `zaqeris_axiom` (16-Expert MoE), `zaqion_dialectic` (ZQN-12B), `zaqeris_7b` |
| **Dense Transformers** | **47** | LLaMA 1/2/3/3.1/3.2/3.3, Qwen 1/1.5/2/2.5, Mistral, Gemma 1/2, Phi 1/2/3/4, Yi |
| **Mixture of Experts (MoE)** | **23** | Mixtral 8x7B/8x22B, DeepSeek V2/V3 MoE, Grok-1, DBRX, Arctic, Qwen2-MoE |
| **Reasoning & Dialectic** | **19** | DeepSeek R1/Zero/Distill, QwQ-32B, Marco-o1, Skywork-T1, NuminaMath |
| **Code Specialists** | **23** | Qwen2.5-Coder, Codestral, StarCoder 1/2, DeepSeek Coder V2, CodeGeeX4 |
| **State Space & Hybrid** | **18** | Mamba 1/2, Jamba 1.5, Zamba 1/2, RWKV 4/5/6/7, RecurrentGemma, RetNet |
| **Vision-Language (VLM)** | **25** | LLaVA 1.5/NeXT/OneVision, Qwen2-VL, PaliGemma 1/2, Pixtral 12B, InternVL2 |
| **Embeddings & Rerankers** | **15** | ModernBERT, BGE M3/Reranker, E5 Large, Nomic-BERT, ColBERT v2 |
| **Speech & Audio** | **10** | Whisper v2/v3/Turbo, Moonshine, SeamlessM4T v1/v2, Wav2Vec2 |
| **Edge & Micro LLMs** | **9** | SmolLM 1/2, MobileLLM, OpenELM, Danube 1/2/3, Femto-LLaMA |

Inspect architectures via CLI:
```bash
python tools/qerlio_cli.py architectures count
python tools/qerlio_cli.py architectures list --family "ZAQERIS Native"
python tools/qerlio_cli.py architectures info zaqion_dialectic
```

---

## Agent Mode & Tool Execution

QERLIO features an integrated, sandboxed tool execution engine allowing models to execute computations, search files, and run code safely:

```bash
# Execute safe calculator
python tools/qerlio_cli.py agent execute --tool calculator --args '{"expression": "(12 * 15) + 30"}'

# Execute sandboxed python code
python tools/qerlio_cli.py agent execute --tool python_eval --args '{"code": "result = sum(range(101))"}'

# Run multi-turn agent simulation
python tools/qerlio_cli.py agent run --prompt "Verify theorem steps"
```

---

## Universal Multi-Protocol API Gateway & Diagnostic HUD

Launch the drop-in local server and developer dashboard:
```bash
python tools/qerlio_cli.py server --port 8787
```

* **Live Developer HUD:** Open `http://localhost:8787` in any browser to inspect real-time throughput, P95 TTFT, RAM distribution, the 200 model architectures catalog, and the Agent Studio.
* **OpenAI Protocol:** `POST http://localhost:8787/v1/chat/completions` (SSE Streaming, Function & Tool Calling)
* **Anthropic Claude Protocol:** `POST http://localhost:8787/v1/messages` (with `tool_use`)
* **Ollama Protocol:** `POST http://localhost:8787/api/generate` & `POST http://localhost:8787/api/chat`
* **Architectures Endpoints:** `GET http://localhost:8787/v1/architectures` & `/v1/architectures/{id}`
* **Agent Endpoints:** `POST http://localhost:8787/v1/agents/execute` & `POST http://localhost:8787/v1/agents/run`

---

## Automated Verification Suite

Run the full test suite (43 passing tests):
```bash
python -m pytest tests/ -v
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
