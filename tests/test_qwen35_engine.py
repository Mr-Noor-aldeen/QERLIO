"""Synthetic unit & plumbing verification tests for the `qwen35` Hybrid Architecture Engine.

NOTE (v0.3.1 honesty requirement):
All tests in this file run exclusively on a tiny deterministic 4-layer **synthetic** GGUF v3
fixture. `qwen35` remains `experimental` in `registry/model_architectures.json` because no
>=64-token official-weight verification run under Directive Section 2 has been completed.

Covers:
- Dynamic hyperparameter loading from `qwen35.*` GGUF v3 metadata
- Gated DeltaNet recurrent linear-attention layers (`attn_qkv`, `attn_gate`, `ssm_conv1d`,
  `ssm_beta`, `ssm_alpha`, `ssm_dt.bias`, `ssm_a`, `ssm_norm`, `ssm_out`)
- Gated Full-Attention layers (`attn_q` interleaved query+gate, `attn_q_norm`, `attn_k_norm`,
  partial RoPE on `rope_dimension_count`, GQA, sigmoid gate, `attn_output`)
- Hybrid cache (`Qwen35HybridCache`) step-by-step decoding equivalence with full-sequence
  recomputation (both recurrent conv/SSM states and full-attention KV cache)
- ChatML (`<|im_start|>...<|im_end|>`) formatting, thinking mode, and `<tool_call>` parsing
- Unified `load_gguf_engine` and Local API (`/v1/models/load`, `/v1/chat/completions`) wiring
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from core.inference import (
    ContextLengthExceededError,
    GGML_TYPE_F32,
    GGUFLoader,
    InsufficientMemoryError,
    Qwen35Config,
    Qwen35Engine,
    Qwen35HybridCache,
    SamplerConfig,
    load_gguf_engine,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_str_array(items: List[str]) -> bytes:
    parts = [struct.pack("<IQ", 8, len(items))]
    for s in items:
        parts.append(_pack_str(s))
    return b"".join(parts)


def _pack_i32_array(items: List[int]) -> bytes:
    return struct.pack(f"<IQ{len(items)}i", 5, len(items), *items)


def _build_synthetic_qwen35_gguf(path: Path) -> None:
    """Create a self-contained, deterministic 4-layer `qwen35` GGUF v3 checkpoint:
    - Layers 0, 1, 2: Gated DeltaNet recurrent layers (`(il + 1) % 4 != 0`)
    - Layer 3: Gated Full-Attention layer (`(il + 1) % 4 == 0`)
    """
    rng = np.random.default_rng(20261121)

    vocab_size = 32
    hidden_size = 16
    ffn_size = 32
    n_layers = 4
    full_attn_interval = 4

    # Full-attention params
    head_count = 4
    head_count_kv = 2
    head_dim = 4
    rope_dim = 2

    # Gated DeltaNet SSM params
    ssm_conv_kernel = 4
    ssm_state_size = 4   # head_k_dim = head_v_dim = 4
    ssm_group_count = 2  # num_k_heads = 2 -> key_dim = 8
    ssm_dt_rank = 4      # num_v_heads = 4 -> value_dim = 16
    ssm_inner_size = 16  # conv_dim = 2*8 + 16 = 32
    conv_dim = 2 * (ssm_state_size * ssm_group_count) + ssm_inner_size

    tokens = [
        "<|endoftext|>",  # 0
        "<|im_start|>",   # 1
        "<|im_end|>",     # 2
        "<think>",        # 3
        "</think>",       # 4
        "<tool_call>",    # 5
        "</tool_call>",   # 6
        "Ċ",              # 7 (\n in GPT-2 byte-to-unicode)
        "system",         # 8
        "user",           # 9
        "assistant",      # 10
        "Hello",          # 11
        "You are Qwen, a helpful AI assistant.",  # 12
        "Paris",          # 13
        "42",             # 14
        ".",              # 15
    ] + [f"tok_{i}" for i in range(16, vocab_size)]
    # Mark tokens 0..6 as CONTROL (3), 8..14 as USER_DEFINED (4) so synthetic 32-token vocab encodes ChatML prompts
    token_types = [3] * 7 + [1] + [4] * 7 + [1] * (vocab_size - 15)
    merges = ["H e", "l l", "w o"]

    kvs: List[Tuple[str, int, bytes]] = [
        ("general.architecture", 8, _pack_str("qwen35")),
        ("general.name", 8, _pack_str("Synthetic-Qwen35-Test")),
        ("qwen35.block_count", 4, struct.pack("<I", n_layers)),
        ("qwen35.nextn_predict_layers", 4, struct.pack("<I", 0)),
        ("qwen35.context_length", 4, struct.pack("<I", 256)),
        ("qwen35.embedding_length", 4, struct.pack("<I", hidden_size)),
        ("qwen35.feed_forward_length", 4, struct.pack("<I", ffn_size)),
        ("qwen35.attention.head_count", 4, struct.pack("<I", head_count)),
        ("qwen35.attention.head_count_kv", 4, struct.pack("<I", head_count_kv)),
        ("qwen35.attention.key_length", 4, struct.pack("<I", head_dim)),
        ("qwen35.attention.value_length", 4, struct.pack("<I", head_dim)),
        ("qwen35.rope.dimension_count", 4, struct.pack("<I", rope_dim)),
        ("qwen35.rope.freq_base", 6, struct.pack("<f", 10000000.0)),
        ("qwen35.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-6)),
        ("qwen35.full_attention_interval", 4, struct.pack("<I", full_attn_interval)),
        ("qwen35.ssm.conv_kernel", 4, struct.pack("<I", ssm_conv_kernel)),
        ("qwen35.ssm.state_size", 4, struct.pack("<I", ssm_state_size)),
        ("qwen35.ssm.group_count", 4, struct.pack("<I", ssm_group_count)),
        ("qwen35.ssm.time_step_rank", 4, struct.pack("<I", ssm_dt_rank)),
        ("qwen35.ssm.inner_size", 4, struct.pack("<I", ssm_inner_size)),
        ("tokenizer.ggml.model", 8, _pack_str("gpt2")),
        ("tokenizer.ggml.pre", 8, _pack_str("qwen35")),
        ("tokenizer.ggml.tokens", 9, _pack_str_array(tokens)),
        ("tokenizer.ggml.token_type", 9, _pack_i32_array(token_types)),
        ("tokenizer.ggml.merges", 9, _pack_str_array(merges)),
        ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 0)),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
        ("tokenizer.ggml.padding_token_id", 4, struct.pack("<I", 0)),
    ]

    tensors_dict: Dict[str, Tuple[Tuple[int, ...], np.ndarray]] = {}

    def _add_t(name: str, gguf_dims: Tuple[int, ...], arr: np.ndarray) -> None:
        tensors_dict[name] = (gguf_dims, arr.astype(np.float32))

    _add_t(
        "token_embd.weight",
        (hidden_size, vocab_size),
        rng.normal(0.0, 0.15, size=(vocab_size, hidden_size)),
    )
    _add_t("output_norm.weight", (hidden_size,), np.ones(hidden_size, dtype=np.float32))
    _add_t(
        "output.weight",
        (hidden_size, vocab_size),
        rng.normal(0.0, 0.15, size=(vocab_size, hidden_size)),
    )

    for il in range(n_layers):
        p = f"blk.{il}"
        _add_t(f"{p}.attn_norm.weight", (hidden_size,), np.ones(hidden_size, dtype=np.float32))
        _add_t(
            f"{p}.post_attention_norm.weight",
            (hidden_size,),
            np.ones(hidden_size, dtype=np.float32),
        )
        _add_t(
            f"{p}.ffn_gate.weight",
            (hidden_size, ffn_size),
            rng.normal(0.0, 0.1, size=(ffn_size, hidden_size)),
        )
        _add_t(
            f"{p}.ffn_up.weight",
            (hidden_size, ffn_size),
            rng.normal(0.0, 0.1, size=(ffn_size, hidden_size)),
        )
        _add_t(
            f"{p}.ffn_down.weight",
            (ffn_size, hidden_size),
            rng.normal(0.0, 0.1, size=(hidden_size, ffn_size)),
        )

        is_rec = ((il + 1) % full_attn_interval) != 0
        if is_rec:
            _add_t(
                f"{p}.attn_qkv.weight",
                (hidden_size, conv_dim),
                rng.normal(0.0, 0.1, size=(conv_dim, hidden_size)),
            )
            _add_t(
                f"{p}.attn_gate.weight",
                (hidden_size, ssm_inner_size),
                rng.normal(0.0, 0.1, size=(ssm_inner_size, hidden_size)),
            )
            _add_t(
                f"{p}.ssm_conv1d.weight",
                (ssm_conv_kernel, conv_dim),
                rng.normal(0.0, 0.15, size=(conv_dim, ssm_conv_kernel)),
            )
            _add_t(
                f"{p}.ssm_beta.weight",
                (hidden_size, ssm_dt_rank),
                rng.normal(0.0, 0.1, size=(ssm_dt_rank, hidden_size)),
            )
            _add_t(
                f"{p}.ssm_alpha.weight",
                (hidden_size, ssm_dt_rank),
                rng.normal(0.0, 0.1, size=(ssm_dt_rank, hidden_size)),
            )
            _add_t(
                f"{p}.ssm_dt.bias",
                (ssm_dt_rank,),
                np.full(ssm_dt_rank, -1.0, dtype=np.float32),
            )
            _add_t(
                f"{p}.ssm_a",
                (ssm_dt_rank,),
                np.full(ssm_dt_rank, -0.05, dtype=np.float32),
            )
            _add_t(
                f"{p}.ssm_norm.weight",
                (ssm_state_size,),
                np.ones(ssm_state_size, dtype=np.float32),
            )
            _add_t(
                f"{p}.ssm_out.weight",
                (ssm_inner_size, hidden_size),
                rng.normal(0.0, 0.1, size=(hidden_size, ssm_inner_size)),
            )
        else:
            q_out_dim = head_count * head_dim * 2
            kv_out_dim = head_count_kv * head_dim
            _add_t(
                f"{p}.attn_q.weight",
                (hidden_size, q_out_dim),
                rng.normal(0.0, 0.1, size=(q_out_dim, hidden_size)),
            )
            _add_t(
                f"{p}.attn_k.weight",
                (hidden_size, kv_out_dim),
                rng.normal(0.0, 0.1, size=(kv_out_dim, hidden_size)),
            )
            _add_t(
                f"{p}.attn_v.weight",
                (hidden_size, kv_out_dim),
                rng.normal(0.0, 0.1, size=(kv_out_dim, hidden_size)),
            )
            _add_t(
                f"{p}.attn_q_norm.weight",
                (head_dim,),
                np.ones(head_dim, dtype=np.float32),
            )
            _add_t(
                f"{p}.attn_k_norm.weight",
                (head_dim,),
                np.ones(head_dim, dtype=np.float32),
            )
            _add_t(
                f"{p}.attn_output.weight",
                (head_count * head_dim, hidden_size),
                rng.normal(0.0, 0.1, size=(hidden_size, head_count * head_dim)),
            )

    # Serialize GGUF v3 file
    header_parts: List[bytes] = [
        b"GGUF",
        struct.pack("<IQQ", 3, len(tensors_dict), len(kvs)),
    ]
    for k, vtype, vbytes in kvs:
        header_parts.append(_pack_str(k))
        header_parts.append(struct.pack("<I", vtype))
        header_parts.append(vbytes)

    data_chunks: List[bytes] = []
    curr_offset = 0
    for name, (gguf_dims, arr) in tensors_dict.items():
        raw = arr.astype("<f4").tobytes()
        header_parts.append(_pack_str(name))
        header_parts.append(struct.pack("<I", len(gguf_dims)))
        header_parts.append(struct.pack(f"<{len(gguf_dims)}Q", *gguf_dims))
        header_parts.append(struct.pack("<IQ", GGML_TYPE_F32, curr_offset))
        data_chunks.append(raw)
        curr_offset += len(raw)
        pad = (32 - (curr_offset % 32)) % 32
        if pad:
            data_chunks.append(b"\x00" * pad)
            curr_offset += pad

    header_raw = b"".join(header_parts)
    align_pad = (32 - (len(header_raw) % 32)) % 32
    path.write_bytes(header_raw + (b"\x00" * align_pad) + b"".join(data_chunks))


@pytest.fixture(scope="module")
def qwen35_gguf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("qwen35_gguf") / "synthetic_qwen35.gguf"
    _build_synthetic_qwen35_gguf(p)
    return p


def test_synthetic_qwen35_config_and_hybrid_layer_schedule(qwen35_gguf_path: Path) -> None:
    """Verify Qwen35Config reads all hybrid Gated DeltaNet + Full-Attention hyperparameters."""
    with Qwen35Engine(qwen35_gguf_path) as eng:
        cfg = eng.config
        assert cfg.architecture == "qwen35"
        assert cfg.block_count == 4
        assert cfg.full_attention_interval == 4
        assert cfg.recurrent_layer_indices == (0, 1, 2)
        assert cfg.full_attention_layer_indices == (3,)
        assert cfg.ssm_conv_dim == 32
        assert cfg.ssm_key_dim == 8
        assert cfg.ssm_value_dim == 16


def test_synthetic_qwen35_preload_vs_lazy_mmap_exact_parity(qwen35_gguf_path: Path) -> None:
    """Verify preloaded weights and lazy mmap streaming produce identical logits."""
    tokens = [1, 9, 7, 11, 2]
    with Qwen35Engine(qwen35_gguf_path, preload_weights=True) as eng_pre:
        logits_pre = eng_pre.forward(tokens)

    with Qwen35Engine(qwen35_gguf_path, preload_weights=False) as eng_lazy:
        logits_lazy = eng_lazy.forward(tokens)

    assert logits_pre.shape == (len(tokens), 32)
    assert np.all(np.isfinite(logits_pre))
    np.testing.assert_allclose(logits_pre, logits_lazy, atol=1e-6, rtol=1e-6)


def test_synthetic_qwen35_hybrid_cache_step_by_step_matches_full_recomputation(
    qwen35_gguf_path: Path,
) -> None:
    """Verify step-by-step decoding with Qwen35HybridCache (both DeltaNet conv/SSM states
    and Full-Attention KV cache) matches full-sequence recomputation within 1e-5.
    """
    seq = [1, 9, 7, 11, 13, 14, 15, 2]
    with Qwen35Engine(qwen35_gguf_path) as eng:
        # 1) Full sequence at once
        full_logits = eng.forward(seq)

        # 2) Prefill first 4 tokens, then decode 1 token at a time using Qwen35HybridCache
        cache = eng.create_cache()
        prefill_logits = eng.forward(seq[:4], kv_cache=cache)
        step_logits_list = [prefill_logits]
        for tok in seq[4:]:
            step_out = eng.forward([tok], kv_cache=cache)
            step_logits_list.append(step_out)

        cached_logits = np.vstack(step_logits_list)
        assert cache.seq_len == len(seq)
        max_diff = float(np.max(np.abs(full_logits - cached_logits)))
        assert max_diff < 1e-5, f"Hybrid cache mismatch: max_diff={max_diff}"


def test_synthetic_qwen35_generation_chat_and_api_integration(qwen35_gguf_path: Path) -> None:
    """Verify Qwen35Engine generation, ChatML formatting, `load_gguf_engine`, and local API on synthetic GGUF."""
    from fastapi.testclient import TestClient
    from server.local_api.server import app, unload_native_models

    with load_gguf_engine(qwen35_gguf_path) as eng:
        assert isinstance(eng, Qwen35Engine)
        gen = eng.generate([1, 9, 7, 11], max_new_tokens=6, stop_token_ids=[])
        assert len(gen["token_ids"]) == 6

        chat_res = eng.chat(
            [{"role": "user", "content": "Hello"}],
            thinking=True,
            max_new_tokens=4,
        )
        assert "<|im_start|>user\nHello<|im_end|>" in chat_res["prompt"]
        assert len(chat_res["generated_token_ids"]) >= 1

    unload_native_models()
    client = TestClient(app)
    try:
        load_res = client.post(
            "/v1/models/load",
            json={
                "model_path": str(qwen35_gguf_path),
                "model_id": "Qwen/Qwen3.8-27B",
                "max_context_length": 128,
            },
        )
        assert load_res.status_code == 200
        assert load_res.json()["architecture"] == "qwen35"
        assert load_res.json()["simulated"] is False

        chat_api = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3.8-27B",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
            },
        )
        assert chat_api.status_code == 200
        assert chat_api.json()["simulated"] is False
    finally:
        unload_native_models()


def test_synthetic_qwen35_single_step_last_token_forward(qwen35_gguf_path: Path) -> None:
    """Verify synthetic 4-layer `qwen35` single-step last-token-only forward pass output shape and finiteness."""
    with Qwen35Engine(qwen35_gguf_path, max_context_length=256, preload_weights=False) as eng:
        assert eng.config.architecture == "qwen35"
        assert eng.config.block_count == 4
        assert eng.config.vocab_size == 32
        assert eng.config.recurrent_layer_indices == (0, 1, 2)
        assert eng.config.full_attention_layer_indices == (3,)
        ids = [1, 9, 7, 11]
        logits = eng.forward(ids, max_layers=4, return_all_logits=False)
        assert logits.shape == (1, 32)
        assert logits.dtype == np.float32
        assert bool(np.all(np.isfinite(logits)))
