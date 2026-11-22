"""Tests for QERLIO's from-scratch C SIMD GEMM backend, sequential tensor streamer,
and batched `SeedOssEngine.forward_batch()` equivalence.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from core.inference.dequant import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    compute_tensor_bytes,
    dequantize_bytes,
)
from core.inference.gguf_loader import GGUFLoader
from core.inference.native_gemm import NativeGemmBackend, SequentialTensorStreamer
from core.inference.seed_oss_model import SeedOssEngine
from tests.test_t1_t2_gguf_and_dequant import _make_random_quant_blocks
from tests.test_t4_to_t10_engine import (
    _create_deterministic_seed_oss_weights,
    write_reference_gguf_v3,
)


def test_native_gemm_backend_parity_and_validation() -> None:
    backend = NativeGemmBackend.get_default()
    if backend is None:
        # Clean fallback when no C compiler or shared library exists on host
        status = NativeGemmBackend.probe_status()
        assert "Unavailable" in status
        return

    rng = np.random.Generator(np.random.PCG64(9001))
    m, n, k = 6, 41, 512

    for qtype in (GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0):
        assert backend.supports(qtype, k)
        blk_sz = 32 if qtype == GGML_TYPE_Q8_0 else 256
        n_blocks_total = (n * k) // blk_sz
        raw = _make_random_quant_blocks(qtype, n_blocks=n_blocks_total, seed=70 + qtype)
        raw_u8 = np.frombuffer(raw, dtype=np.uint8)
        w_f32 = dequantize_bytes(raw_u8, qtype, n * k).reshape(n, k)

        for x_shape in ((k,), (1, k), (m, k)):
            x = rng.standard_normal(x_shape, dtype=np.float32) * np.float32(0.02)
            bias = rng.standard_normal(n, dtype=np.float32) * np.float32(0.05)

            y_ref = np.matmul(x, w_f32.T).astype(np.float32) + bias
            y_out = np.empty((*x.shape[:-1], n), dtype=np.float32)
            row_bytes = compute_tensor_bytes(qtype, k)

            # Execute in two non-uniform chunks matching streaming chunk semantics
            split = 17
            backend.matmul_raw_chunk(
                x, int(raw_u8.ctypes.data), y_out, qtype, n, k, 0, split, bias=bias
            )
            backend.matmul_raw_chunk(
                x,
                int(raw_u8[split * row_bytes :].ctypes.data),
                y_out,
                qtype,
                n,
                k,
                split,
                n,
                bias=bias,
            )
            np.testing.assert_allclose(y_out, y_ref, atol=1e-4, rtol=1e-5)

    # F32 kernel check
    assert backend.supports(GGML_TYPE_F32, k)
    x_f32 = rng.standard_normal((m, k), dtype=np.float32)
    w_raw = rng.standard_normal((n, k), dtype=np.float32)
    y_out_f32 = np.empty((m, n), dtype=np.float32)
    backend.matmul_raw_chunk(x_f32, int(w_raw.ctypes.data), y_out_f32, GGML_TYPE_F32, n, k, 0, n)
    np.testing.assert_allclose(y_out_f32, np.matmul(x_f32, w_raw.T), atol=1e-4, rtol=1e-5)

    assert not backend.supports(GGML_TYPE_Q4_K, 128)


@pytest.fixture(scope="module")
def synthetic_seed_oss_gguf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("native_gemm_gguf")
    gguf_path = tmp_dir / "native_gemm_test.gguf"
    model_data = _create_deterministic_seed_oss_weights()
    write_reference_gguf_v3(gguf_path, model_data)
    return gguf_path


def test_sequential_tensor_streamer_exact_bytes_hit_and_miss(
    synthetic_seed_oss_gguf_path: Path,
) -> None:
    gguf_path = synthetic_seed_oss_gguf_path

    with GGUFLoader(gguf_path, header_only=False) as loader:
        streamer = SequentialTensorStreamer(gguf_path, initial_buf_bytes=4096)
        try:
            t_q = loader.get_tensor_info("blk.0.attn_q.weight")
            t_k = loader.get_tensor_info("blk.0.attn_k.weight")
            t_v = loader.get_tensor_info("blk.0.attn_v.weight")

            off_q = loader.data_offset + t_q.offset
            off_k = loader.data_offset + t_k.offset
            off_v = loader.data_offset + t_v.offset

            # Read Q and prefetch K (prefetch hit on next call)
            arr_q, _ = streamer.acquire(off_q, t_q.n_bytes, prefetch=(off_k, t_k.n_bytes))
            raw_q = loader.get_tensor_raw("blk.0.attn_q.weight")
            assert bytes(arr_q) == bytes(raw_q)
            raw_q.release()

            # Read K (prefetch hit) and prefetch Q, then intentionally request V (prefetch miss)
            arr_k, _ = streamer.acquire(off_k, t_k.n_bytes, prefetch=(off_q, t_q.n_bytes))
            raw_k = loader.get_tensor_raw("blk.0.attn_k.weight")
            assert bytes(arr_k) == bytes(raw_k)
            raw_k.release()

            arr_v, _ = streamer.acquire(off_v, t_v.n_bytes)
            raw_v = loader.get_tensor_raw("blk.0.attn_v.weight")
            assert bytes(arr_v) == bytes(raw_v)
            raw_v.release()
        finally:
            streamer.close()


def test_seed_oss_forward_batch_matches_sequential_forward(
    synthetic_seed_oss_gguf_path: Path,
) -> None:
    gguf_path = synthetic_seed_oss_gguf_path

    prompts = [
        [0, 291, 48, 287],
        [0, 120, 45, 77, 19, 200],
        [0, 318],
    ]

    for preload in (True, False):
        with SeedOssEngine(gguf_path, preload_weights=preload, header_only=False) as eng:
            # Independent single-sequence baseline with KV caches
            seq_caches = [eng.create_kv_cache(max_seq_len=32) for _ in prompts]
            seq_prefill_logits = [
                eng.forward(p, kv_cache=c, last_token_only=True)
                for p, c in zip(prompts, seq_caches)
            ]

            # Batched multi-sequence prefill (varying sequence lengths!)
            batch_caches = [eng.create_kv_cache(max_seq_len=32) for _ in prompts]
            batch_prefill_logits = eng.forward_batch(
                prompts, kv_caches=batch_caches, last_token_only=True
            )

            assert len(batch_prefill_logits) == len(prompts)
            for l_seq, l_bat, cs, cb in zip(
                seq_prefill_logits, batch_prefill_logits, seq_caches, batch_caches
            ):
                assert cs.seq_len == cb.seq_len
                np.testing.assert_allclose(l_bat, l_seq, atol=1e-4, rtol=1e-5)

            # Next decode step batched vs sequential
            next_toks = [int(np.argmax(x)) for x in batch_prefill_logits]
            seq_step_logits = [
                eng.forward([tok], kv_cache=c, last_token_only=True)
                for tok, c in zip(next_toks, seq_caches)
            ]
            batch_step_logits = eng.forward_batch(
                [[tok] for tok in next_toks], kv_caches=batch_caches, last_token_only=True
            )
            for l_seq, l_bat in zip(seq_step_logits, batch_step_logits):
                np.testing.assert_allclose(l_bat, l_seq, atol=1e-4, rtol=1e-5)


def test_real_36b_transformers_reference_fixture_integrity() -> None:
    """Verify committed offline HF transformers real 36B T4/T5 parity fixture schema & parity summary."""
    import json

    ref_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "hermes_4_3_36b_transformers_reference.json"
    )
    assert ref_path.is_file(), f"Missing HF transformers 36B fixture: {ref_path}"
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    assert data["summary"]["t4_top1_matches"] == "5/5"
    assert data["summary"]["t5_exact_token_matches"] == "320/320"
    assert len(data["prompts"]) == 5
    for p in data["prompts"]:
        assert len(p["greedy_64_token_ids"]) == 64
        assert p["t5_exact_token_matches"] == 64
        assert p["top5_overlap_with_qerlio"] >= 4
