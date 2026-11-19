# QERLIO Runtime

> **Your AI. Your Device. Your Rules.**  
> *Developed by ZAQERIS AI — Research Group*

QERLIO Runtime is a high-efficiency, privacy-first local AI execution platform engineered for mobile devices, laptops, and desktop workstations. It provides an adaptive, vendor-neutral abstraction layer over CPU, GPU, and NPU accelerators with guaranteed reference fallbacks.

---

## Key Architectural Principles

1. **Local-First & Offline by Default:** Zero cloud dependencies for core execution. No telemetry or prompt logging without explicit user opt-in.
2. **Correctness Before Speed:** High-performance kernels must pass strict numerical parity and stability benchmarks.
3. **Graceful Degradation:** Automatic fallback to CPU if an accelerator or operator fails.
4. **Stable Language Boundary:** Pure C ABI (`include/qerlio/qerlio.h`) enabling seamless Kotlin/JNI, Swift, and Python bindings without leaking C++ internals.
5. **Memory and Thermal Protection:** Real-time context budgeting and thermal-aware throttling to protect battery health and prevent OOM terminations.

---

## Repository Structure

```text
QERLIO/
├── CMakeLists.txt              # Top-level build configuration (C++20)
├── include/qerlio/
│   └── qerlio.h                # Authoritative C ABI contract
├── src/
│   └── qerlio_abi.cpp          # Core engine implementation
├── core/
│   ├── engine/                 # Engine lifecycle and threading
│   ├── model_registry/         # GGUF header parser & SHA-256 verifier
│   └── session/                # Generation session & atomic cancellation
├── tools/
│   └── qerlio_cli.py           # CLI diagnostic harness & model inspector
├── tests/
│   ├── test_gguf_integrity.py  # Fuzz and integrity test suite
│   └── test_manifest_schema.py # Model manifest validation
├── docs/
│   ├── SPECIFICATION_v1.0.md   # Master Research & Engineering Specification
│   ├── architecture/           # Mermaid diagrams & system architecture
│   ├── decisions/              # Architecture Decision Records (ADR 001 - 007)
│   ├── security/               # STRIDE threat model & trust boundaries
│   ├── backend-matrix/         # Supported platform & quantization matrix
│   └── implementation/         # Audit reports & vertical slice roadmap
└── third_party/
    └── DEPENDENCY_INVENTORY.md # SPDX licensing & dependency audit
```

---

## Universal Multi-Protocol API Gateway & Diagnostic HUD

Launch the drop-in local server and developer dashboard with a single command:
```bash
python tools/qerlio_cli.py server --port 8787
```

* **Live Developer HUD:** Open `http://localhost:8787` in any browser to inspect real-time throughput, P95 TTFT, RAM distribution, thermal telemetry, and the 28-scheme quantization matrix.
* **OpenAI Protocol:** `POST http://localhost:8787/v1/chat/completions` (SSE Streaming & Non-streaming)
* **Anthropic Claude Protocol:** `POST http://localhost:8787/v1/messages`
* **Ollama Protocol:** `POST http://localhost:8787/api/generate` & `POST http://localhost:8787/api/chat`

| Compression Family | Supported Schemes | Key Benefit |
| :--- | :--- | :--- |
| **Standard GGUF** | `F16`, `Q8_0`, `Q5_0`, `Q5_1`, `Q4_0`, `Q4_1` | General compatibility baseline |
| **k-quants (SOTA)** | `Q2_K` to `Q6_K` (`Q4_K_M`, `Q5_K_M`) | Optimal quality-to-size Pareto frontier |
| **i-quants (Sub-3bit)** | `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ3_S`, `IQ4_XS` | Extreme compression (< 2 bits/weight) |
| **ZAQERIS RVQ** | `M3_GPTQ` | Residual Vector Quantization with weight cache |
| **Hardware Accelerated**| `FP8_E4M3`, `FP8_E5M2` | High-throughput tensor core acceleration |
| **KV Cache Precision** | `KV_Q8_0`, `KV_Q4_0` + Sink-Aware Pruning | Up to 85% RAM savings on 16k context |

### OpenAI Example:
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="qerlio")
res = client.chat.completions.create(
    model="qerlio-adaptive-balanced",
    messages=[{"role": "user", "content": "Explain local AI."}],
    stream=True
)
for chunk in res:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Anthropic Example:
```bash
curl -X POST http://127.0.0.1:8787/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "qerlio-adaptive-balanced", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Ollama Example:
```bash
curl -X POST http://127.0.0.1:8787/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qerlio:latest", "prompt": "Hello from Ollama client!"}'
```

### 2. System Diagnostics
Inspect local CPU, RAM, and GPU accelerator status:
```bash
python tools/qerlio_cli.py doctor
```

### 3. Inspect a GGUF Model Header
Safely inspect GGUF metadata without executing binary code:
```bash
python tools/qerlio_cli.py inspect path/to/model.gguf
```

### 4. Run the Reproducible Benchmark Lab
```bash
python tools/qerlio_cli.py benchmark path/to/model.gguf --tokens 64 --runs 3
```

### 5. Run the Test Suite (23 Passing Tests)
```bash
python -m pytest tests/ -v
```

---

## Project Status

- **Current Phase:** Phase 0 Completed / Phase 1 Scaffolded.
- **Audit Report:** [REPOSITORY_AUDIT.md](file:///c:/Users/rahim/Downloads/QERLIO/docs/implementation/REPOSITORY_AUDIT.md)
- **License Notice:** See [DEPENDENCY_INVENTORY.md](file:///c:/Users/rahim/Downloads/QERLIO/third_party/DEPENDENCY_INVENTORY.md).
