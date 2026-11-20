# QERLIO Runtime — Third-Party Dependency Inventory & Licensing Audit

**Policy:** Every third-party library incorporated into or linked with QERLIO must have its source, pinned version, SPDX license identifier, and exact license text recorded in this inventory.

---

## 1. Direct Dependencies

| Dependency Name | Upstream Repository / Source | Pinned Version / Commit | SPDX License Identifier | Usage in QERLIO | Compliance Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **llama.cpp** | `https://github.com/ggml-org/llama.cpp` | To be pinned on submodule pull | `MIT` | Primary GGUF execution backend (`backends/gguf_llama_cpp`) | Full MIT license text and copyright notices preserved in `third_party/licenses/LLAMA_CPP_LICENSE.txt` |
| **ggml** | Sub-component of `llama.cpp` | Pinned with llama.cpp | `MIT` | Tensor computation library | Attribution retained; no modifications to upstream copyright headers |
| **nlohmann/json** | `https://github.com/nlohmann/json` | v3.11.3 | `MIT` | Model manifest & JSON configuration parsing | Header-only integration; license in `third_party/licenses/JSON_LICENSE.txt` |

---

## 2. Research & Sister Dependencies (Isolated)

| Dependency Name | Source Location | Version | License | Usage in QERLIO | Compliance Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **zaqeris-m3-gptq** | Separate ZAQERIS research package (not vendored in this repository) | 0.1.0 | Proprietary / ZAQERIS AI Internal | Research compression adapter (`formats/zaqeris_compression`) | Gated behind experimental feature flag; strictly isolated from open-core runtime |

---

## 3. Tooling & Testing Dependencies (Build/Dev Only)

| Tool / Library | Version in Host | License | Purpose |
| :--- | :--- | :--- | :--- |
| **pytest** | 9.1.1 | MIT | Test suite execution for Python tools |
| **gguf-py** | 0.19.0 | MIT | Reference GGUF header parsing & validation tests |
| **safetensors** | 0.8.0 | Apache-2.0 | Conversion & integrity checks |
| **pydantic** | 2.14.1 | MIT | Manifest schema validation |
