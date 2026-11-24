# QERLIO Runtime — System Architecture & Component Design

**Owner:** ZAQERIS AI Research Group  
**Document Version:** 1.0  
**Date:** 2026-11-19  

---

## 1. High-Level Architectural Diagram

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Application Layer"]
        CLI["qerlio CLI Tool"]
        APP_AND["Android Kotlin App / SDK"]
        APP_DSK["Desktop / Local Web UI"]
        PY_TOOL["Python Research & Benchmark Tools"]
    end

    subgraph InterfaceLayer ["Stable Interface Layer"]
        C_ABI["Flat C ABI (qerlio.h)"]
        JNI_BRIDGE["Android JNI Bridge"]
    end

    subgraph CoreEngine ["QERLIO Core Engine (C++20)"]
        ENG_MGR["Engine Manager & Lifecycle"]
        SESS_MGR["Session & Generation Controller"]
        CANCEL_TOK["Atomic Cancellation Token"]
        MEM_PLAN["Memory Planner & Budgeter"]
        SCHED["Adaptive Device Scheduler"]
        SAFE_GATE["Security & Tool Permission Gateway"]
        MOD_REG["Model Registry & Hash Verifier"]
    end

    subgraph BackendLayer ["Hardware Abstraction & Backend Adapters"]
        ROUTER["Backend Router & Fallback Handler"]
        GGUF_BACKEND["From-Scratch Native GGUF Engine (Python/NumPy + C11 SIMD)"]
        VULKAN_BACKEND["Vulkan GPU Adapter (Planned)"]
        CUDA_BACKEND["CUDA GPU Adapter (Planned)"]
        NPU_BACKEND["Vendor NPU Adapter (Planned)"]
        CPU_REF["Portable CPU Reference Baseline"]
    end

    subgraph StorageLayer ["Local Storage & Formats"]
        MODELS_DIR["Verified GGUF Model Files"]
        MANIFEST_DB["Model Manifests & Checksums"]
        LOCAL_LOGS["Local Anonymized Diagnostics"]
    end

    CLI --> C_ABI
    APP_DSK --> C_ABI
    APP_AND --> JNI_BRIDGE --> C_ABI
    PY_TOOL --> C_ABI

    C_ABI --> ENG_MGR
    C_ABI --> SESS_MGR
    SESS_MGR --> CANCEL_TOK
    SESS_MGR --> MEM_PLAN
    SESS_MGR --> SCHED
    ENG_MGR --> MOD_REG

    MOD_REG --> StorageLayer
    SCHED --> ROUTER

    ROUTER --> GGUF_BACKEND
    ROUTER --> VULKAN_BACKEND
    ROUTER --> CUDA_BACKEND
    ROUTER --> NPU_BACKEND
    ROUTER --> CPU_REF
```

---

## 2. Component Responsibilities

### 2.1 Engine Manager (`qerlio_engine_t`)
- Manages global state, logger initialization, and device discovery.
- Thread-safe coordinator for active models and sessions.
- Enforces single-instance or bounded multi-instance memory limits.

### 2.2 Model Registry & Verifier (`qerlio_model_t`)
- Reads GGUF metadata headers without loading full tensors into memory.
- Verifies SHA-256 against expected hash.
- Validates tensor counts, quantization schemes, and architecture compatibility.
- Enforces strict memory budgets before tensor allocation.

### 2.3 Session Manager & Cancellation Token (`qerlio_session_t`)
- Allocates thread-safe execution context for generation.
- Binds an atomic cancellation flag checked before each token step.
- Manages KV Cache lifecycle, context window rolling, and prompt evaluation.
- Emits chunks via user-supplied token callback.

### 2.4 Backend Router & Fallback Controller
- Evaluates device capabilities (CPU vector extensions, available memory, driver versions).
- Directs model execution to the optimal verified backend.
- Automatically and gracefully falls back to CPU reference execution if GPU/NPU initialization fails.

---

## 3. Data Flow during Streaming Generation

```mermaid
sequenceDiagram
    autonumber
    actor User as Client Application
    participant Engine as QERLIO C ABI
    participant Session as Session Manager
    participant Memory as Memory Planner
    participant Backend as Backend Router
    participant Device as Hardware (CPU/GPU)

    User->>Engine: qerlio_generate_stream(session, prompt, config, callback)
    Engine->>Session: Prepare generation context
    Session->>Memory: Check available budget for prompt + KV cache
    Memory-->>Session: Budget approved
    Session->>Backend: Initialize decode loop
    
    loop Token Generation Loop
        Backend->>Device: Compute next token logits
        Device-->>Backend: Logits tensor
        Backend->>Session: Sampled token string
        Session->>User: Invoke callback(token_chunk)
        User-->>Session: Return true (continue) or false (cancel)
        opt If Cancelled or Stop Token
            Session->>Backend: Abort remaining steps
        end
    end

    Session-->>Engine: Compute final execution metrics
    Engine-->>User: Return QERLIO_OK / QERLIO_ERR_CANCELLED
```
