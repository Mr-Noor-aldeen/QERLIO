> **Historical Archive Note (v0.3.1):** Phase 0 environment/discovery snapshot (`2026-11-19`). Note that `gguf`, `transformers`, `tokenizers`, and `torch` are permitted strictly offline under `tests/reference/` and are not runtime dependencies of `core/inference/`, and `ADR-002` (`llama.cpp`) was superseded in v0.3.0 by the from-scratch native engine.

# QERLIO Runtime — Initial Repository & Environment Audit

**Date:** 2026-11-19  
**Auditor:** Principal Systems Architect / QERLIO Core Team  
**Workspace:** QERLIO repository root  
**Phase:** Phase 0 (Discovery, Baseline & Environment Assessment)  
**Status:** COMPLETE WITH DOCUMENTED BLOCKERS  

---

## 1. Repository Status

- **Repository Initial State:** Empty workspace (the repository root contained 0 files at startup).
- **Git Version Control:** Not initialized (`git` is currently not present in system PATH).
- **Existing Code/Scaffolding:** None existed prior to Phase 0 initialization.
- **Related Assets in Host Environment:**
  - A separate ZAQERIS research workspace provides the `zaqeris-m3-gptq` compression package.
  - Python wheel `zaqeris-m3-gptq 0.1.0` installed in active Python environment.

---

## 2. Host Software Environment

| Subsystem | Specification / Detected Value | Status / Evidence |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 | `VERIFIED` |
| **Host Hardware** | Intentionally not published in this repository | `N/A` |
| **Python Runtime** | Python 3.12.10 (64-bit) | `VERIFIED` (`python --version`) |
| **PyTorch & ML Libs** | PyTorch 2.11.0, `gguf 0.19.0`, `safetensors 0.8.0`, `transformers 5.20.1`, `tokenizers 0.23.3` | `VERIFIED` (active in site-packages) |
| **Node.js Runtime** | Node.js v22.14.0 / npm 10.9.2 | `VERIFIED` |
| **Package Manager** | Windows Package Manager (`winget`) v1.29.380 | `VERIFIED` |
| **Attached Android Devices** | None detected / No ADB daemon in PATH | `NOT AVAILABLE` |
| **Apple Metal / macOS** | Non-Apple Host | `NOT APPLICABLE` |

---

## 3. Toolchain & Compiler Availability (Blocker Assessment)

| Tool | Status | Impact | Remediation Plan |
| :--- | :--- | :--- | :--- |
| **CMake** | `VERIFIED` (v4.4.4) | Build configuration ready | Verified at `C:\Program Files\CMake\bin\cmake.exe` |
| **Ninja** | `VERIFIED` (v1.13.2) | Fast build generator ready | Verified at `...\WinGet\Packages\Ninja-build.Ninja...\ninja.exe` |
| **C++ Compiler (Clang)** | `INSTALLED` (v23.1.6) | Gated by Host Application Control | Windows Application Control policy blocks `clang.exe`; requires administrator whitelist or MSVC Build Tools |
| **Git** | `MISSING` from PATH | Version tracking and submodule checkout disabled | Available via `winget install Git.Git` |
| **Android NDK / SDK** | `MISSING` from PATH | Android cross-compilation blocked | Scheduled in Phase 5 after core C ABI is validated |

### Blocker Classification (Rule #20)
> **Rule #20 Policy:** "If a requirement cannot be implemented because hardware, SDK, credentials, license permission, or a toolchain is unavailable, create a precise blocker report and a safe stub interface only if the architecture needs it. Never fake the integration."

- **Current Blocker:** Direct C++ compilation on the local Windows host is temporarily blocked by the absence of `cmake` and a C++20 compiler (`clang-cl`, `gcc`, or `cl.exe`) in PATH.
- **Python / Tooling Environment:** Python 3.12 with `gguf`, `safetensors`, `torch`, and `pytest` is fully operational and capable of driving model inspection, manifest verification, JSON schema validation, and test harness execution.
- **Action Required for Native Compilation:** Install `Git.Git`, `Kitware.CMake`, and `LLVM.LLVM` (or Visual Studio C++ Build Tools).

---

## 4. Architectural Baseline Decisions (ADR Summary)

1. **ADR-001 (Core & ABI):** Native Core written in C++20, strictly isolated behind a flat C ABI (`qerlio.h`). No C++ exceptions or STL types across the boundary.
2. **ADR-002 (GGUF Backend):** `llama.cpp` selected as reference GGUF execution engine via an isolated C++ backend adapter (`backends/gguf_llama_cpp`).
3. **ADR-003 (Android Strategy):** JNI wrapper over stable C ABI. NNAPI rejected as primary future path (deprecated in Android 15); focus on CPU (ARM64 NEON) + Vulkan/vendor accelerators.
4. **ADR-004 (Model Integrity):** Zero execution of remote code. Models require SHA-256 validation, strict header bounds checking, and signed/verified manifests.
5. **ADR-005 (Local API Security):** Loopback only (`127.0.0.1`), token-authenticated, strict CORS/CSRF defense.
6. **ADR-006 (Privacy & Telemetry):** Telemetry disabled by default. Network access disabled in inference runtime. No prompt/conversation data in default logs.
7. **ADR-007 (Error Taxonomy & Cancellation):** First-class atomic cancellation tokens and structured error codes.

---

## 5. Phase 0 Audit Conclusion & Next Steps

Phase 0 discovery confirms an active Python environment with `gguf` and `torch`, and zero pre-existing codebase conflicts. The specification architecture is scaffolded with complete directory structure, ADRs, risk register, threat model, and reference C ABI header.
