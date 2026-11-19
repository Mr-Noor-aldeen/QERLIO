# ADR-007: Structured Error Taxonomy and First-Class Cancellation Semantics

## Status
Accepted

## Context
In local AI runtimes on mobile and desktop devices, user-initiated generation cancellations are frequent (e.g., clicking "Stop Generating", closing an application tab, or system suspend). Without first-class atomic cancellation, generation threads can spin, leak memory, cause race conditions, or hang the UI thread. Additionally, cryptic integer error codes or ad-hoc strings make cross-language wrappers difficult to maintain.

## Decision
1. **Atomic Cancellation Tokens:** Every generation session binds to an atomic `std::atomic<bool>` cancellation flag managed via `qerlio_cancellation_token_t`.
2. **Cooperative Interruptibility:** Inference loops must inspect the cancellation token before evaluating each chunk or token. When cancelled:
   - Token streaming halts immediately (< 50ms latency target).
   - KV cache state is cleanly updated or reset.
   - The stream emits `QERLIO_STATUS_CANCELLED`.
   - No deadlock or resource leak occurs.
3. **Structured Error Taxonomy:** All C API functions return a typed `qerlio_status_t` enum:
   - `QERLIO_OK = 0`
   - `QERLIO_ERR_INVALID_ARGUMENT = 1`
   - `QERLIO_ERR_UNSUPPORTED_PLATFORM = 2`
   - `QERLIO_ERR_UNSUPPORTED_ARCHITECTURE = 3`
   - `QERLIO_ERR_UNSUPPORTED_QUANTIZATION = 4`
   - `QERLIO_ERR_MODEL_CORRUPT = 5`
   - `QERLIO_ERR_MODEL_HASH_MISMATCH = 6`
   - `QERLIO_ERR_INSUFFICIENT_MEMORY = 7`
   - `QERLIO_ERR_BACKEND_INIT_FAILED = 8`
   - `QERLIO_ERR_OPERATOR_UNSUPPORTED = 9`
   - `QERLIO_ERR_CANCELLED = 10`
   - `QERLIO_ERR_TIMEOUT = 11`
   - `QERLIO_ERR_PERMISSION_DENIED = 12`
   - `QERLIO_ERR_NETWORK_DISABLED = 13`
   - `QERLIO_ERR_IO = 14`
   - `QERLIO_ERR_INTERNAL = 15`
4. **Thread-Safe Extended Diagnostics:** An API `qerlio_get_last_error(qerlio_engine_t* engine, char* buffer, size_t buffer_size)` retrieves thread-local detailed diagnostic error descriptions without exposing raw exceptions.

## Consequences
- **Positive:** Deterministic cancellation; UI remains responsive; language wrappers easily map status codes to native exceptions (e.g., Kotlin exceptions).
- **Negative:** Requires inference kernels and token loops to frequently poll cancellation state.
