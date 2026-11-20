#!/usr/bin/env python3
"""
QERLIO Benchmark Lab & Metrics Profiler
Implements Specification Section 36 (Appendix C)
"""

import os
import sys
import time
import json
import psutil
import datetime
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sdk.python.qerlio import QerlioEngine, GenerationConfig, PowerProfile

def run_benchmark(
    model_path: str,
    prompt: str = "Explain the fundamental principles of local on-device neural execution.",
    output_tokens: int = 64,
    context_limit: int = 2048,
    warmup_runs: int = 1,
    measurement_runs: int = 3,
    profile_name: str = "mobile-balanced"
) -> Dict[str, Any]:
    """
    Executes reproducible benchmark conforming to QERLIO Benchmark Schema v1.0.
    """
    engine = QerlioEngine()
    model = engine.load_model(model_path, context_length=context_limit)
    
    config = GenerationConfig(
        max_output_tokens=output_tokens,
        power_profile=PowerProfile.BALANCED
    )

    # 1. Warm-up runs
    for _ in range(warmup_runs):
        session = model.create_session()
        list(session.generate_stream(prompt, config))

    # 2. Measurement runs
    ttft_samples: List[float] = []
    tps_samples: List[float] = []
    rss_samples: List[int] = []

    process = psutil.Process()

    for run_idx in range(measurement_runs):
        session = model.create_session()
        tokens = []
        start_time = time.perf_counter()
        first_token_latency = None

        for chunk in session.generate_stream(prompt, config):
            if first_token_latency is None:
                first_token_latency = (time.perf_counter() - start_time) * 1000.0
            tokens.append(chunk)

        total_duration = time.perf_counter() - start_time
        gen_tps = len(tokens) / total_duration if total_duration > 0 else 0.0

        if first_token_latency is not None:
            ttft_samples.append(first_token_latency)
        tps_samples.append(gen_tps)
        rss_samples.append(process.memory_info().rss)

    engine.shutdown()

    # Median / P95 statistics
    ttft_samples.sort()
    tps_samples.sort()
    rss_samples.sort()

    median_ttft = ttft_samples[len(ttft_samples) // 2] if ttft_samples else 0.0
    p95_ttft = ttft_samples[int(len(ttft_samples) * 0.95)] if ttft_samples else 0.0
    median_tps = tps_samples[len(tps_samples) // 2] if tps_samples else 0.0
    peak_rss = max(rss_samples) if rss_samples else 0

    report = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "qerlio_version": "0.2.2",
        "backend": "qerlio-reference-cpu",
        "profile": profile_name,
        "device": {
            "os": sys.platform,
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_total_bytes": psutil.virtual_memory().total,
            "ram_available_bytes": psutil.virtual_memory().available
        },
        "model": {
            "model_path": os.path.basename(model_path),
            "file_size_bytes": model.header_info.get("file_size_bytes", 0),
            "format": "gguf",
            "version": model.header_info.get("version", 3)
        },
        "workload": {
            "prompt_length_words": len(prompt.split()),
            "output_tokens_requested": output_tokens,
            "context_limit": context_limit,
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs
        },
        "results": {
            "ttft_ms_median": round(median_ttft, 2),
            "ttft_ms_p95": round(p95_ttft, 2),
            "generation_tps_median": round(median_tps, 2),
            "peak_rss_bytes": peak_rss,
            "energy_method": "not_measured"
        }
    }
    return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_suite.py <path_to_model.gguf>")
        sys.exit(1)
    rep = run_benchmark(sys.argv[1])
    print(json.dumps(rep, indent=2))
