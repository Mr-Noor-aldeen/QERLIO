# QERLIO Benchmark Methodology & Lab Specifications

**Standard:** Implements Specification Section 7.5, Section 30, and Section 36 (Appendix C)  
**Version:** 1.0  
**Date:** 2026-11-19  

---

## 1. Principles of Reproducible Benchmarking

1. **No Fabricated Data:** Every benchmark number must be produced from an automated test run and tied to an immutable model SHA-256 hash, exact host OS build, and runtime version.
2. **Deterministic Warmup:** Every benchmark run executes at least one warm-up inference pass before measuring timings to prime memory mapping, CPU caches, and dynamic libraries.
3. **Statistical Rigor:** Benchmarks report **Median** and **P95 (95th Percentile)** values across a minimum of 3 measurement iterations, rather than advertising the single fastest run.
4. **Thermal Stability Monitoring:** The benchmark suite samples device thermal states (`NORMAL`, `WARM`, `HOT`, `CRITICAL`) at start and completion to account for thermal throttling.

---

## 2. Standardized Metrics

| Metric Key | Description | Unit | Measurement Mechanism |
| :--- | :--- | :--- | :--- |
| `ttft_ms_median` | Time to First Token (TTFT) | Milliseconds | Monotonic high-resolution clock (`std::chrono::steady_clock` / `time.perf_counter`) from prompt submission until the first token chunk is emitted. |
| `generation_tps_median`| Generation Throughput | Tokens / Second | Total generated output tokens divided by decoding loop wall-clock time. |
| `prompt_tps_median` | Prompt Evaluation Throughput | Tokens / Second | Prompt token count divided by prompt ingestion time. |
| `peak_rss_bytes` | Peak Resident Set Size | Bytes | Physical RAM allocated to the process sampled via OS kernel hooks (`psutil.Process.memory_info().rss` / `GetProcessMemoryInfo`). |
| `energy_method` | Energy Consumption Method | String | Marked as `not_measured` unless a verified hardware power meter (e.g., Qualcomm Power Profiler, Monsoon, or Battmon) is active. |

---

## 3. Benchmark CLI Command

Execute the benchmark suite against any GGUF model:
```bash
python tools/qerlio_cli.py benchmark <path_to_model.gguf> --tokens 64 --context 2048 --warmup 1 --runs 3 --profile mobile-balanced
```
