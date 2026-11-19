# ADR-002: Selection of llama.cpp as Initial GGUF Execution Backend

## Status
Accepted

## Context
Writing an optimized inference engine from scratch (including matrix multiplication kernels, attention mechanisms, tensor operations for dozens of quantization formats, and architecture variants for Llama, Qwen, Mistral, Gemma) would require months or years of effort and introduce severe correctness risks. The project requires a production-grade, highly tested baseline for GGUF model execution.

## Decision
1. **Reference Backend:** We adopt `llama.cpp` as the first execution backend for GGUF models.
2. **Pinned Integration:** Upstream `llama.cpp` will be pinned to a specific, audited release tag or commit hash (not a moving HEAD).
3. **Strict Isolation:** `llama.cpp` is encapsulated entirely within `backends/gguf_llama_cpp/`. Neither the public headers (`include/qerlio/`) nor the core orchestrator will include `llama.h` directly.
4. **License & Attribution:** The MIT license and all copyright notices of `llama.cpp` / GGML are preserved in `third_party/licenses/` and reported in the SBOM.
5. **No Blind Copying:** Only the required backend integration adapter is built. QERLIO retains ownership of the adaptive scheduler, memory planning, security gating, model registry, and client APIs.

## Consequences
- **Positive:** Immediate support for standard GGUF quantizations (Q4_K_M, Q8_0, etc.) on CPU and GPU; proven correctness baseline; active community maintenance.
- **Negative:** Dependency on upstream library changes; need to maintain an abstraction adapter layer when upstream updates its internal APIs.
