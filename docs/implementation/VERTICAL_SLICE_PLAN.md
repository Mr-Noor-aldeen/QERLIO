# QERLIO Runtime — Phase 1 Vertical Slice Execution Plan

**Objective:** Implement the first verifiable, end-to-end vertical slice of QERLIO:
1. Pure C ABI interface implementation (`src/qerlio.cpp`).
2. Robust, bounded GGUF metadata reader & SHA-256 verifier (`core/model_registry/gguf_verifier.cpp`).
3. CLI inspection tool (`tools/model-inspect`).
4. Unit and conformance tests (`tests/`).

---

## 1. Scope and Boundaries

- **In Scope for Slice 1:**
  - Complete memory lifecycle (`create` -> `load` -> `session` -> `cancel` -> `destroy`).
  - Strict GGUF header parsing with bounds validation (preventing buffer overflows and integer wraps).
  - SHA-256 cryptographic verification before loading.
  - Mock/reference inference stream simulating token generation with atomic cancellation.
  - Python-based verification test suite using the host's Python 3.12 + `gguf` library.
  
- **Deferred to Subsequent Slices:**
  - Full tensor matrix multiplication kernels (delegated to pinned `llama.cpp` integration in Slice 2).
  - Multi-backend GPU/Vulkan offloading (Slice 3).
  - Android JNI bindings (Slice 4).

---

## 2. Directory Layout for Slice 1

```text
QERLIO/
├── CMakeLists.txt
├── include/
│   └── qerlio/
│       └── qerlio.h
├── src/
│   ├── engine.cpp
│   ├── model.cpp
│   ├── session.cpp
│   └── qerlio_abi.cpp
├── core/
│   ├── model_registry/
│   │   ├── gguf_parser.hpp
│   │   └── gguf_parser.cpp
│   └── memory/
│       └── memory_budget.hpp
├── tools/
│   └── qerlio_cli.py
└── tests/
    ├── test_abi_lifecycle.py
    ├── test_manifest_validation.py
    └── test_malformed_gguf.py
```

---

## 3. Host Environment Path Forward

Since Windows C++ compilers (`cl.exe`, `clang`, `gcc`) and `cmake` are currently missing from PATH, Phase 1 executes two synchronized parallel tracks:
1. **Production C++ Core Codebase:** Written in standards-compliant C++20 with `CMakeLists.txt` so it is ready for compilation on CI runners and local developer machines once compilers are installed.
2. **Operational Python Verification Harness:** Uses the existing local Python 3.12 environment (which already includes `gguf`, `safetensors`, `torch`, `pydantic`, and `pytest`) to immediately execute manifest validation, GGUF structural integrity checks, and conformance testing on real model files without waiting for compiler installation.
