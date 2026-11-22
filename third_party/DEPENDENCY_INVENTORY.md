# QERLIO Runtime — Third-Party Dependency Inventory & Licensing Audit

**Policy:** Every third-party library incorporated into or linked with QERLIO must have its source, pinned version, SPDX license identifier, and exact license text recorded in this inventory.

---

## 1. Direct Runtime Dependencies

QERLIO's native inference engine (`core/inference/`) is implemented from scratch using only the Python standard library, NumPy, and standard C11/C++20 (zero runtime dependency on external inference libraries).

| Dependency Name | Upstream Repository / Source | Pinned Version / Commit | SPDX License Identifier | Usage in QERLIO | Compliance Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NumPy** | `https://numpy.org` | >= 1.24 | `BSD-3-Clause` | Python numerical kernels (`core/inference/`) | Only Python third-party runtime dependency of the native engine |
| **nlohmann/json** | `https://github.com/nlohmann/json` | v3.11.3 | `MIT` | Model manifest & JSON configuration parsing (optional C++ tooling) | Header-only integration; license in `third_party/licenses/JSON_LICENSE.txt` |

---

## 2. Research & Sister Dependencies (Isolated)

| Dependency Name | Source Location | Version | License | Usage in QERLIO | Compliance Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **zaqeris-m3-gptq** | Separate ZAQERIS research package (not vendored in this repository) | 0.1.0 | Proprietary / ZAQERIS AI Internal | Research compression adapter (`formats/zaqeris_compression`) | Gated behind experimental feature flag; strictly isolated from open-core runtime |

---

## 3. Offline Reference Tools (Dev/Test Fixture Generation Only — Never Runtime Dependencies)

Per v0.3.0 policy, external inference engines and reference libraries are never imported or linked by `core/inference/` at runtime. They may be run strictly offline inside `tests/reference/` to generate static verification fixtures once.

| Tool / Library | Version | License | Offline Usage |
| :--- | :--- | :--- | :--- |
| **pytest** | >= 7.0 | `MIT` | Test suite execution (`tests/`) |
| **gguf (`gguf-py`)** | 0.19.0 | `MIT` | Offline GGUF metadata, tensor inspection & T2 dequantization reference fixture generation (`tests/reference/`) |
| **tokenizers** | 0.23.3 | `Apache-2.0` | Offline T3 BPE tokenizer reference fixture generation (`tests/reference/`) |
| **transformers** | 5.20.1 | `Apache-2.0` | Offline T4/T5 `SeedOssForCausalLM` / `SeedOssDecoderLayer` reference verification (`tests/reference/`); never imported by `core/inference/` |
| **torch** | 2.11.0+cu128 | `BSD-3-Clause` | Offline tensor execution for `transformers` reference verification (`tests/reference/`); never imported by `core/inference/` |
| **safetensors** | >= 0.4 | `Apache-2.0` | Offline conversion & reference inspection only |
| **pydantic / fastapi / uvicorn** | >= 2.0 | `MIT` / `BSD-3-Clause` | Optional local HTTP API server (`server/local_api/`) |
