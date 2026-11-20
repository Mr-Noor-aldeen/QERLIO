# Changelog

All notable changes to the QERLIO Runtime project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and internal engineering directive `ZAQ-ENG-DIR-2026-11-20`.

---

## [0.2.2] - 2026-11-20

### Added
- **Hermes 4.3 36B Support:** Full native adapter (`core/models/hermes_adapter.py`) implementing the official ByteDance Seed 36B base model specifications and Llama-3-Chat formatting (`<seed:bos>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`).
- **Hybrid-Mode Reasoning:** Automatic extraction and isolation of `<think> ... </think>` internal monologue blocks from user-facing answers.
- **Structured Tool Calling:** Support for function calling within `<tools>` XML schemas and structured `<tool_call>` extraction and execution in the sandboxed runtime.
- **CLI Subcommands:**
  - `qerlio hermes chat`: Chat and inspect prompt formatting with Hermes 4.3 36B.
  - `qerlio hermes test`: End-to-end automated reasoning and tool extraction conformance test.
  - `qerlio hermes download`: Direct downloader for official GGUF quantized releases (`Q4_K_M`, `Q3_K_M`, etc.).
  - `qerlio hermes run`: Local execution of downloaded GGUF weights via QERLIO Engine.
- **Architecture Catalog Update:** Registered `seed_36b` (ByteDance Seed 36B / Hermes 4.3) under `Reasoning & Dialectic` with `tested` support tier, bringing the catalog to 183 total architectures (4 tested, 179 experimental).
- **Reproduction Guide:** Comprehensive documentation in [`docs/models/HERMES_4_3_36B.md`](docs/models/HERMES_4_3_36B.md) documenting exact download steps, CLI commands, sampling parameters, and recorded outputs.
- **Secure Space Gateway:** Open-source FastAPI/Gradio gateway (`server/hf_space_gateway/`) with bearer authentication and 7-day automatic trial expiration protection.

### Changed
- Explicitly marked simulated mock API responses with `"simulated": true` in `server/local_api/server.py` and `server/hf_space_gateway/app.py` pursuant to Directive Rule 7.
- Updated total architecture count to 183 across CLI, API, registry, and documentation.

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
