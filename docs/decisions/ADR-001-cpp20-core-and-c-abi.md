# ADR-001: C++20 Native Core and Stable C ABI Boundary

## Status
Accepted

## Context
QERLIO Runtime must execute with high efficiency across Android, iOS, Windows, Linux, and macOS. It must be embeddable into applications written in Kotlin, Swift, Python, C#, or Go without exposing complex C++ standard library types, ABI incompatibilities (name mangling, std::string layout differences, compiler version mismatches), or uncaught C++ exceptions.

## Decision
1. **Core Language:** The internal implementation of QERLIO Core will be written in standard modern **C++20**.
2. **Public API Boundary:** The public interface exposed to all foreign language wrappers (Kotlin/JNI, Swift, C#, Python ctypes/CFFI) is a pure, flat **C ABI** (`include/qerlio/qerlio.h`).
3. **Handle-Based Opaque Types:** All native core objects (`qerlio_engine_t`, `qerlio_model_t`, `qerlio_session_t`, etc.) are exposed as opaque pointer handles. Internal layouts are never leaked to client headers.
4. **No Exceptions across ABI:** All C API functions catch internal exceptions and return structured `qerlio_status_t` error codes.
5. **Memory Ownership:** Memory allocated by QERLIO must be freed by QERLIO deallocation functions (`qerlio_*_destroy` or `qerlio_free_string`). Caller buffers are borrowed and never freed by QERLIO.
6. **Thread Safety:** Engine initialization/destruction is synchronized. Sessions are thread-safe for cancellation while generation is streaming on a worker thread.

## Consequences
- **Positive:** Maximum portable compatibility; frictionless JNI/Swift integration; stable ABI that does not break when internal C++ classes evolve.
- **Negative:** Requires manual boilerplate mapping between internal C++ RAII objects and external C handle wrappers.
