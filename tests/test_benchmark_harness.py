import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.benchmark.benchmark_suite import run_benchmark
from tests.test_gguf_integrity import create_synthetic_gguf

def test_benchmark_execution(tmp_path):
    model_file = tmp_path / "bench_model.gguf"
    create_synthetic_gguf(str(model_file), magic=b"GGUF", version=3, tensor_count=64, kv_count=16)

    report = run_benchmark(
        str(model_file),
        prompt="Test benchmark prompt",
        output_tokens=8,
        context_limit=1024,
        warmup_runs=1,
        measurement_runs=2,
        profile_name="mobile-entry"
    )

    assert report["schema_version"] == "1.0"
    assert report["backend"] == "qerlio-reference-cpu"
    assert report["profile"] == "mobile-entry"
    assert "results" in report
    assert report["results"]["generation_tps_median"] > 0
    assert report["results"]["peak_rss_bytes"] > 0
    assert report["device"]["cpu_cores_physical"] > 0
