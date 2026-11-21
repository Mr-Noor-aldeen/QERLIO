#!/usr/bin/env python3
"""
QERLIO Benchmark Harness
Reports null / no_live_measurement when running against the reference stub backend.
"""

import os
import sys
import json
import datetime
from typing import Dict, Any

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
    Executes the benchmark lifecycle harness conforming to QERLIO Benchmark Schema v1.0.
    Because the reference CPU backend is a lifecycle stub without live neural execution,
    throughput and latency metrics are reported as None ('no_live_measurement').
    """
    engine = QerlioEngine()
    model = engine.load_model(model_path, context_length=context_limit)

    config = GenerationConfig(
        max_output_tokens=output_tokens,
        power_profile=PowerProfile.BALANCED
    )

    for _ in range(warmup_runs):
        session = model.create_session()
        list(session.generate_stream(prompt, config))

    for _ in range(measurement_runs):
        session = model.create_session()
        list(session.generate_stream(prompt, config))

    engine.shutdown()

    report = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "qerlio_version": "0.2.2",
        "backend": "qerlio-reference-cpu",
        "simulated": True,
        "profile": profile_name,
        "model": {
            "model_path": os.path.basename(model_path),
            "file_size_bytes": model.header_info.get("file_size_bytes", 0),
            "format": "gguf",
            "version": model.header_info.get("version", 3),
            "architecture": model.header_info.get("architecture")
        },
        "workload": {
            "prompt_length_words": len(prompt.split()),
            "output_tokens_requested": output_tokens,
            "context_limit": context_limit,
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs
        },
        "results": {
            "measurement_status": "no_live_measurement",
            "ttft_ms_median": None,
            "ttft_ms_p95": None,
            "generation_tps_median": None,
            "peak_rss_bytes": None,
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
