# Changelog

All notable changes to the QERLIO Runtime project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and internal engineering directive `ZAQ-ENG-DIR-2026-11-20`.

---

## [0.2.2] - 2026-11-20

### Added
- **Hermes 4.3 36B (`Seed-OSS`) Chat Template & Tag Parser (`core/models/hermes_adapter.py`):** Prompt formatter and output tag parser for [`NousResearch/Hermes-4.3-36B`](https://huggingface.co/NousResearch/Hermes-4.3-36B) and [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF) (`ByteDance Seed-OSS-36B-Base` / `SeedOssForCausalLM`, `model_type: seed_oss`, 524,288 context length) using the `Llama-3-Chat` template (`<seed:bos>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`).
- **Thinking & Tool-Call Parsing Features:**
  - `thinking` defaults to `False` (`--thinking` flag on CLI enables the reasoning system prompt).
  - `keep_cots` support (`False` by default) to strip or preserve `<think>...</think>` blocks in multi-turn history and parsed output.
  - Edge-case handling for unclosed `<think>` tags, unclosed `<tool_call>` tags, and malformed JSON inside `<tool_call>`.
  - Conversion helpers `to_openai_tool_calls` and `to_anthropic_tool_use` for translating parsed `<tool_call>` blocks into OpenAI and Anthropic protocol schemas.
- **GGUF Metadata KV Parsing:** `parse_gguf_raw` (`tools/qerlio_cli.py`) and `GGUFParser::parse_header` (`core/model_registry/gguf_parser.cpp`) now parse the GGUF v2/v3 metadata key-value section (`general.architecture`, `general.name`, `<arch>.context_length`, `general.file_type` / quantization) and compute real SHA-256 checksums in C++.
- **CLI Subcommands (`qerlio hermes`):**
  - `qerlio hermes chat`: Format prompts locally (`--thinking`, `--keep-cots`) or query an external OpenAI-compatible endpoint.
  - `qerlio hermes test`: Offline template-formatting and tag-parsing self-test (no model inference).
  - `qerlio hermes download`: Download official GGUF files from `NousResearch/Hermes-4.3-36B-GGUF`.
- **Architecture Catalog Entry (`seed_36b`):** Added `seed_36b` (`ByteDance Seed-OSS-36B (Hermes 4.3 36B)`) to `registry/model_architectures.json` under `Reasoning & Dialectic` with `support_status: "experimental"` (183 total: 3 tested, 180 experimental). QERLIO does not yet execute `SeedOssForCausalLM` tensor weights locally.
- **Documentation (`docs/models/HERMES_4_3_36B.md`):** Reference guide covering official `Seed-OSS` metadata, template formatting, tag parsing, and GGUF downloading.
- **Remote Gateway Template Stub (`server/hf_space_gateway/`):** FastAPI/Gradio HTTP gateway scaffold with optional secret-token verification and a 7-day trial expiration check.

### Fixed
- Removed model-impersonating responses and invented tool calls/usage counts from `server/local_api/server.py` and `server/hf_space_gateway/app.py`; stub endpoints now return `"simulated": true` and `"usage": null`.
- Removed unmeasured placeholder performance and memory metrics from `sdk/python/qerlio/runtime.py`, `src/qerlio_abi.cpp`, `core/compression/quant_matrix.py`, and `core/compression/universal_quant.cpp`.

---

## [0.2.1] - 2026-11-20

### Fixed
- **Accurate Architecture Registry:** Filtered catalog down to 182 real, publicly verifiable open-weight architectures across 10 families. Removed 19 duplicate, private, or unreleased entries.
- **Support Tiers:** Introduced explicit `support_status` field (`tested` vs `experimental`). Marked only models with real code paths in the repository as `tested` (`zaqeris_axiom`, `zqn_12b`, `zaqeris_7b`).
- **Corrected ZAQERIS Model Metadata:** AXIOM documented as a sparse MoE (79.43B total / 12.39B active) with ZAQERIS-M3 packed format at ~3.16 bits per parameter (corrected from 3.12 bpp).
- **Removed Fake Telemetry:** Replaced hardcoded throughput, TTFT, and memory numbers in HUD and `/v1/telemetry/hud` with `null` and a clear `no live measurement` label.
- **Repository Cleanliness:** Removed mock UI screenshots from git tracking and configured `capture_hud.py` to save captures outside the repository (`~/.qerlio/hud_captures`). Replaced absolute machine paths with relative links.

---

## [0.2.0] - 2026-11-19

### Added
- **Architecture Catalog:** Registry of model architecture metadata covering 10 architectural families.
- **Built-in Agent Mode:** Sandboxed ReAct loop with native tool calling support for ZAQERIS-7B, ZQN-12B, and OpenAI function calling formats.
- **Safe Sandbox Tools:** Calculator, Sandboxed Python Evaluator, Workspace File Search, and Synthetic Web Search.
- **Universal Multi-API Gateway:** Local API server supporting OpenAI, Anthropic, and Ollama protocol compatibility.
- **Developer HUD:** Web-based architecture explorer and agent studio interface.
