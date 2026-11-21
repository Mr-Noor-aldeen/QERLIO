# QERLIO Runtime

[![CI](https://github.com/ZAQERIS/QERLIO/actions/workflows/ci.yml/badge.svg)](https://github.com/ZAQERIS/QERLIO/actions)
[![Release](https://img.shields.io/github/v/release/ZAQERIS/QERLIO?color=blue)](https://github.com/ZAQERIS/QERLIO/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](sdk/python/)
[![Tests](https://img.shields.io/badge/tests-pytest-success.svg)](tests/)
[![Architectures](https://img.shields.io/badge/architectures-5%20tested%20%7C%20179%20experimental-blueviolet.svg)](registry/model_architectures.json)
[![Universal Gateway](https://img.shields.io/badge/gateway-OpenAI%20%7C%20Anthropic%20%7C%20Ollama-purple.svg)](server/local_api/)
[![Agent Mode](https://img.shields.io/badge/agents-ReAct%20%7C%20Tri--Agent%20Sandbox-emerald.svg)](core/agent/)

> **Your AI. Your Device. Your Rules.**  
> *Developed by ZAQERIS AI — Independent Research Group*

QERLIO Runtime is a local-first AI execution platform and tooling suite providing a from-scratch GGUF v2/v3 native inference engine (Python standard library + NumPy), a C++20 runtime core, chat-template and tool-call parsers, and a multi-protocol local API gateway.

---

## What's New in v0.3.0

* **From-Scratch GGUF v2/v3 Loader & MMAP Streaming (`core/inference/gguf_loader.py`):** Binary parser and zero-copy memory-mapped tensor reader with strict magic/version/bounds validation and row-level dequantized lookup (`get_tensor_rows_f32`).
* **16 GGML Dequantization Kernels (`core/inference/dequant.py`):** Vectorized NumPy + scalar reference decoders matching within `< 1e-6` absolute error for `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ3_S`, and `IQ4_XS`.
* **Native Byte-Level BPE Tokenizer (`core/inference/tokenizer.py`):** Pure-Python GGUF tokenizer supporting NFC normalization, `seed-coder` / `qwen35` / `gpt2` pre-tokenization, special/control token splitting, and exact round-trip encode/decode.
* **`SeedOssForCausalLM` Native Engine (`core/inference/seed_oss_model.py`):** Executes ByteDance Seed-OSS / `NousResearch/Hermes-4.3-36B` checkpoints (`seed_oss` architecture: RMSNorm, GQA with QKV biases, half-split RoPE, `post_attention_norm`, SwiGLU MLP) with zero runtime dependency on external inference libraries. Verified locally on the official `models/hermes-4_3_36b-Q4_K_M.gguf` checkpoint (`21,762,145,216` bytes, SHA-256 `17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08`; initialized via `mmap` streaming in `0.67s`; all 64 layers, `hidden=5120`, `ffn=27648`, `heads=80`, `kv_heads=8`, `head_dim=128`, `vocab=155136`): `Hermes36BAdapter` 33-token chat round-trip passed and full 64-layer forward pass on `'The capital of France is'` (`[0, 1063, 9203, 437, 10341, 506]`) completed in `419.43s` (`finite=True`), predicting top-1 token `' Paris'` (`id=10805`, `logit=16.4356`).
* **`Qwen35ForCausalLM` Hybrid Engine (`core/inference/qwen35_model.py`):** Executes `qwen35` hybrid checkpoints combining Gated DeltaNet recurrent linear-attention layers (`attn_qkv`/`attn_gate`, causal depthwise `ssm_conv1d`, `ssm_beta`/`ssm_alpha`/`ssm_dt`/`ssm_a`, gated RMSNorm `ssm_norm`, `ssm_out`) and Gated Full-Attention layers (`attn_q` + sigmoid gate, Q/K RMSNorm, partial half-split RoPE), with a unified `Qwen35HybridCache` and ChatML support. Verified locally on `models/Qwen3.8-27B-UD-Q4_K_M.gguf` (all 64 trunk layers: 48 Gated DeltaNet + 16 Gated Full-Attention + 248,320 LM head; prompt `'The capital of France is'` predicted top-1 token `' Paris'` (`logit=17.5078`, `prefill=610.40s`) and generated `' Paris.\nThe'` (`[11751, 13, 198, 760]`) across 4 greedy tokens via `Qwen35HybridCache`).
* **KV Cache, Sampler & Streaming (`core/inference/kv_cache.py`, `core/inference/sampler.py`):** Position-aware GQA `KVCache` with upfront memory budget checking (`InsufficientMemoryError`, `ContextLengthExceededError`), greedy/temperature/top-k/top-p/repetition-penalty sampling, stop sequences, and clean mid-stream cancellation.
* **Live Gateway Execution (`server/local_api/server.py`, `tools/qerlio_cli.py`):** `/v1/models/load`, `/v1/models/unload`, `/v1/chat/completions`, and `/v1/messages` execute the loaded native engine with `"simulated": false` when a GGUF model is loaded, and return explicit `"simulated": true` stubs only when no model is loaded. CLI `qerlio hermes chat --model-path <model.gguf>` invokes `load_gguf_engine`.

### Known Limitations (v0.3.0)
* Inference executes on CPU via pure NumPy + Python stdlib; CUDA/Metal/Vulkan GPU acceleration and quantized C++ GEMM kernels are not yet wired into the Python tensor forward pass.
* Only `seed_oss` and `qwen35` (`Qwen3.5`/`Qwen3.8` dense hybrid) text decoder graphs are implemented in `core/inference/`; vision encoders and Mixture-of-Experts (`qwen35moe`, `zaqeris_axiom` weights) are not yet executed locally.
* Memory-mapped multi-gigabyte checkpoints dequantize weight tiles on demand into float32 on CPU, which prioritizes correctness and portability over interactive tok/s latency.

## What Was in v0.2.2

* **Hermes 4.3 36B (`Seed-OSS`) Template & Tag Parser (`core/models/hermes_adapter.py`):** Chat template formatter and output parser for [`NousResearch/Hermes-4.3-36B`](https://huggingface.co/NousResearch/Hermes-4.3-36B) (official GGUF repository: [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF)), built on **ByteDance Seed-OSS-36B-Base** (`SeedOssForCausalLM`, `model_type: seed_oss`, 524,288 context window) with the `Llama-3-Chat` template (`<seed:bos>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`).
* **Reasoning & Tool-Call Parsing:** Supports `thinking=False` by default, optional `thinking=True` system prompt injection, `keep_cots` history filtering, unclosed `<think>` / `<tool_call>` and malformed JSON handling, and conversion of `<tool_call>` blocks into OpenAI `tool_calls` and Anthropic `tool_use` formats.
* **GGUF Metadata KV Parsing & Downloader:** Parses GGUF v2/v3 metadata key-value tables (`general.architecture`, `general.name`, context length, quantization type) in both Python (`parse_gguf_raw`) and C++ (`GGUFParser::parse_header`), and adds `qerlio hermes download` for pulling official GGUF files from `NousResearch/Hermes-4.3-36B-GGUF`.
* **Architecture Catalog (`183` entries: `3` tested, `180` experimental):** Added `seed_36b` (`ByteDance Seed-OSS-36B (Hermes 4.3 36B)`) as `experimental` under *Reasoning & Dialectic* (QERLIO formats prompts, parses tags, and inspects GGUF headers, but does not yet execute `SeedOssForCausalLM` tensor weights locally).
* **Remote Gateway Template Stub (`server/hf_space_gateway/`):** FastAPI/Gradio HTTP gateway template with optional bearer-token authentication and a 7-day trial expiration check (returns explicit stub responses until an inference backend is attached).

## What's New in v0.2.1

* **Accurate architecture registry:** 182 real, publicly available open-weight architectures across 10 families. Entries without downloadable weights, duplicates, and unverifiable names were removed, and plain model names are used throughout.
* **Support tiers:** every registry entry now carries a `support_status` field, either `tested` or `experimental` (see below).
* **Corrected ZAQERIS model metadata:** AXIOM is listed as a sparse MoE trained from scratch with 79.43B total / 12.39B active parameters per token, packed with ZAQERIS-M3 at ~3.16 bits per parameter.
* **Docs cleanup:** removed local machine paths and the dashboard screenshot; links are now relative to the repository.
* **No placeholder metrics:** the HUD and `/v1/telemetry/hud` no longer show fixed throughput, latency, memory or perplexity figures. Unmeasured values are `null` / "—" with a "no live measurement" label. `scripts/capture_hud.py` now saves screenshots outside the repository (`~/.qerlio/hud_captures`, or `QERLIO_HUD_CAPTURE_DIR`).

## What's in v0.2.0

* **Architecture catalog:** a registry of model architecture metadata covering ZAQERIS Native, Dense Transformers, Mixture of Experts, Reasoning, Code Specialists, State Space Models / Mamba, Vision-Language, Embeddings & Rerankers, Speech & Audio, and Edge/Micro LLMs.
* **Built-in Agent Mode:** Sandboxed ReAct loop with native tool calling support for ZAQERIS-7B (`<tool_call>`), ZQN-12B (`<call name="..." args="..."/>`), and OpenAI/Anthropic function calling formats.
* **Safe Sandbox Tools:** Calculator, Sandboxed Python Evaluator, Workspace File Search, and Synthetic Web Search.
* **ZAQERIS Model Connectors:** local endpoints for `ZAQERIS/ZAQERIS-7B`, `ZAQERIS/ZQN-12B`, and `ZAQERIS/ZAQERIS_AXIOM`.
* **Developer HUD:** architecture explorer with family and tier filtering, plus an interactive Agent Studio at `http://127.0.0.1:8787`.

---

## Model Architectures: 5 Tested, 179 Experimental

QERLIO keeps a registry of 184 model architectures in [`registry/model_architectures.json`](registry/model_architectures.json), generated by [`scripts/build_architectures_registry.py`](scripts/build_architectures_registry.py). Each entry has a `support_status`:

* **`tested` (5):** the architecture has a dedicated code path in this repository exercised by the test suite: `seed_36b` (`SeedOssForCausalLM` native forward pass, KV cache, BPE tokenizer, chat/tool calling), `qwen35` (`Qwen35ForCausalLM` hybrid Gated DeltaNet + Gated Full-Attention engine), plus ZAQERIS-7B (`<tool_call>` parsing), ZQN-12B (`<call .../>` parsing), and AXIOM (ZAQERIS-M3 adapter interface).
* **`experimental` (179):** catalog metadata and/or template/header tooling only. These are real, publicly available open-weight models, but QERLIO does not yet load and run their tensor weights end to end.

| Architectural Family | Tested | Experimental | Representative Architectures |
| :--- | :---: | :---: | :--- |
| **ZAQERIS Native** | 3 | 0 | `zaqeris_axiom`, `zqn_12b`, `zaqeris_7b` |
| **Dense Transformer** | 0 | 51 | Llama 1/2/3/3.1/3.2/3.3, Qwen 1/1.5/2/2.5, Mistral 7B, Gemma 1/2, Phi 1/2/3/4, Yi, Falcon, OLMo |
| **Mixture of Experts** | 0 | 19 | Mixtral 8x7B/8x22B, DeepSeek-V2/V3, Grok-1, DBRX, Snowflake Arctic, Qwen2-MoE, OLMoE |
| **Reasoning & Dialectic** | 1 | 16 | `seed_36b` (ByteDance Seed-OSS-36B / Hermes 4.3 36B — **tested**), DeepSeek-R1 / R1-Zero / Distill, QwQ-32B-Preview, Marco-o1, Sky-T1, NuminaMath |
| **Code Specialist** | 0 | 22 | Qwen2.5-Coder, Codestral, StarCoder 1/2, DeepSeek-Coder-V2, CodeGeeX4 |
| **State Space & Hybrid** | 1 | 14 | `qwen35` (Qwen3.5 / Qwen3.8 Hybrid — **tested**), Mamba 1/2, Jamba, Zamba 1/2, RWKV 4/5/6/7, RecurrentGemma, HGRN |
| **Vision-Language** | 0 | 25 | LLaVA 1.5/NeXT/OneVision, Qwen2-VL, PaliGemma 1/2, Pixtral 12B, InternVL2 |
| **Embedding & Reranker** | 0 | 15 | ModernBERT, BGE-M3 / Reranker, E5, nomic-embed-text, ColBERTv2 |
| **Speech & Audio** | 0 | 10 | Whisper large-v3 / turbo, Moonshine, SeamlessM4T v1/v2, wav2vec 2.0 |
| **Edge & Micro** | 0 | 7 | SmolLM 1/2, MobileLLM, OpenELM, H2O-Danube 1/2/3 |

**ZAQERIS models**

| ID | Model | Details |
| :--- | :--- | :--- |
| `zaqeris_axiom` | [AXIOM](https://huggingface.co/ZAQERIS/ZAQERIS_AXIOM) | Sparse MoE trained from scratch: 79.43B total / 12.39B active parameters per token (16 experts per layer, top-2 routing); published as ZAQERIS-M3, ~3.16 bits per parameter. |
| `zqn_12b` | [ZQN-12B](https://huggingface.co/ZAQERIS/ZQN-12B) | 12B model for reasoning and agentic tool use. |
| `zaqeris_7b` | [ZAQERIS-7B](https://huggingface.co/ZAQERIS/ZAQERIS-7B) | 7B model for function calling and tool use, trained from scratch (only the tokenizer is reused). |

Inspect architectures via CLI:
```bash
python tools/qerlio_cli.py architectures count
python tools/qerlio_cli.py architectures list --status tested
python tools/qerlio_cli.py architectures list --family "ZAQERIS Native"
python tools/qerlio_cli.py architectures info seed_36b
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

* **Live Developer HUD:** Open `http://localhost:8787` in any browser to inspect runtime telemetry (throughput, TTFT and model memory are shown only when actually measured; until real inference lands they read "no live measurement"), the architecture catalog with its support tiers, and the Agent Studio.
* **OpenAI Protocol:** `POST http://localhost:8787/v1/chat/completions` (SSE Streaming, Function & Tool Calling)
* **Anthropic Claude Protocol:** `POST http://localhost:8787/v1/messages` (with `tool_use`)
* **Ollama Protocol:** `POST http://localhost:8787/api/generate` & `POST http://localhost:8787/api/chat`
* **Architectures Endpoints:** `GET http://localhost:8787/v1/architectures` (optional `?status=tested|experimental`) & `/v1/architectures/{id}`
* **Agent Endpoints:** `POST http://localhost:8787/v1/agents/execute` & `POST http://localhost:8787/v1/agents/run`

---

## Automated Verification Suite

Run the full test suite:
```bash
python -m pytest tests/ -v
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
