# QERLIO Runtime — Platform, Backend, and Architecture Support Matrix

**Version:** 1.0 (Phase 0 Baseline)  
**Rule:** Any platform, device, or backend that has not been empirically verified on actual physical hardware must be classified as `UNKNOWN` or `EXPERIMENTAL`. Never mark a feature as `SUPPORTED` without reproducible test logs.

---

## 1. Operating System & Platform Matrix

| Platform / OS | Target ABI / Architecture | Backend Pathway | Current Status | Notes / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Windows 11** | `x86_64` (AVX2, FMA) | CPU (Reference) | `PLANNED` | Development host environment verified (Intel Core i7-14650HX). |
| **Windows 11** | `x86_64` (CUDA 13.1) | NVIDIA CUDA (RTX 5060) | `PLANNED` | Hardware present; native build toolchain required. |
| **Android 10+ (API 29+)** | `arm64-v8a` / `arm64-v9a` | CPU (NEON / DotProd) | `PLANNED` | Primary mobile target baseline. |
| **Android 10+ (API 29+)** | `arm64-v8a` (Vulkan 1.1+) | GPU (Vulkan) | `EXPERIMENTAL` | Requires physical device testing across Adreno/Mali GPUs. |
| **Android (Qualcomm QNN)**| `arm64-v8a` (Hexagon NPU) | QNN Backend | `UNKNOWN` | Requires Snapdragon Hardware + Qualcomm AI Engine Direct SDK. |
| **Android (MediaTek)** | `arm64-v8a` (APU) | NeuroPilot Backend | `UNKNOWN` | Pending MediaTek hardware verification and SDK access. |
| **Android (NNAPI)** | `arm64-v8a` | NNAPI | `DEPRECATED` | Deprecated in Android 15. Not supported as primary target. |
| **macOS / iOS** | `arm64` (Apple Silicon) | Metal / Core ML | `UNKNOWN` | Host is Windows; macOS CI runner required. |
| **Linux (Ubuntu 22.04+)**| `x86_64` / `aarch64` | CPU / CUDA | `PLANNED` | Standard CI environment target. |

---

## 2. Model Architecture Support Matrix (GGUF Backend)

| Architecture Family | Typical Models | GGUF Compatibility | Test Status | Verified Quantizations |
| :--- | :--- | :--- | :--- | :--- |
| **Llama / Llama 3** | Llama-3-8B, Llama-3.2-1B/3B | Supported by GGUF specification | `PLANNED` | Q4_K_M, Q8_0 (Pending smoke test) |
| **Qwen / Qwen 2.5** | Qwen2.5-0.5B, 1.5B, 7B | Supported by GGUF specification | `PLANNED` | Q4_K_M, Q8_0 (Pending smoke test) |
| **Mistral** | Mistral-7B, Nemo-12B | Supported by GGUF specification | `PLANNED` | Q4_K_M, Q8_0 (Pending smoke test) |
| **Gemma / Gemma 2** | Gemma-2-2B, Gemma-2-9B | Supported by GGUF specification | `PLANNED` | Q4_K_M (Pending sliding window verification) |
| **ZAQERIS ZQN / AXIOM** | ZQN-12B, AXIOM-1.1 | Custom Research / Hybrid | `EXPERIMENTAL` | Gated research adapter required (see ADR-004) |

---

## 3. Quantization Support Classification

| Quantization Format | Target Backend | Performance Profile | Status |
| :--- | :--- | :--- | :--- |
| **Q4_K_M** | CPU / GPU | Balanced speed, memory, and perplexity (recommended default) | `PLANNED` |
| **Q4_0** | CPU (NEON / AVX2) | High throughput on older architectures, slight perplexity penalty | `PLANNED` |
| **Q5_K_M** | CPU / GPU | High quality for reasoning workloads with moderate memory footprint | `PLANNED` |
| **Q8_0** | CPU / GPU | Reference precision baseline for fidelity comparison | `PLANNED` |
| **F16 / BF16** | GPU / CPU | Uncompressed baseline; high memory bandwidth required | `PLANNED` |
| **ZAQERIS M3-GPTQ** | ZAQERIS Custom Adapter | Specialized research quantization | `RESEARCH` (Isolated in `formats/zaqeris_compression`) |
