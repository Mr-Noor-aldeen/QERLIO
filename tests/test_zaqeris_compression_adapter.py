import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from formats.zaqeris_compression.zaqeris_m3_adapter import ZaqerisM3Adapter

def test_adapter_availability():
    adapter = ZaqerisM3Adapter(cache_size_mb=256)
    assert adapter.is_available() is True

def test_adapter_benchmark_reconstruction():
    adapter = ZaqerisM3Adapter(cache_size_mb=256)
    bench = adapter.benchmark_reconstruction(in_features=1024, out_features=1024)
    assert bench["status"] == "OPERATIONAL"
    assert bench["uncompressed_bytes"] == 1024 * 1024 * 2
    assert "latency_ms" in bench
    assert bench["adapter_version"] == "0.1.0"
