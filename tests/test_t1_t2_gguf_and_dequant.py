"""Tests T1 & T2: GGUF v3 Full Loader & Dequantization Kernels.

- T1: Verifies GGUF v2/v3 binary parsing, all 771 tensor descriptors and 27 metadata KVs
  of official `hermes-4_3_36b-Q4_K_M.gguf` against `tests/fixtures/hermes_4_3_36b_q4_k_m_metadata.json`,
  full memory-mapped loading, row-level dequantization, and strict rejection of corrupt/truncated files.
- T2: Verifies vectorized NumPy dequantization (`dequantize_bytes`) against the independent pure-Python
  scalar reference (`dequantize_block_scalar_reference`) within `< 1e-6` absolute error across all
  supported GGML quantization types (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
  `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`), plus real `Q6_K` blocks from `output.weight` in Hermes 4.3 36B.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from core.inference.dequant import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_INFO,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    compute_tensor_bytes,
    dequantize_block_scalar_reference,
    dequantize_bytes,
    get_block_size,
    get_type_size,
)
from core.inference.gguf_loader import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUFLoader,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "hermes_4_3_36b_q4_k_m_metadata.json"
HERMES_Q6K_FIXTURE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "hermes_output_weight_rows0_3_q6k.bin"
)
HERMES_PREFIX_BIN_CANDIDATES = [
    Path("C:/Users/rahim/.gemini/antigravity/brain/dc714e6b-287c-4524-a387-3f332fb5705d/scratch/hermes_prefix.bin"),
    REPO_ROOT / "tests" / "fixtures" / "hermes_prefix.bin",
]


def _find_hermes_prefix_bin() -> Path | None:
    for cand in HERMES_PREFIX_BIN_CANDIDATES:
        if cand.exists() and cand.stat().st_size >= 6_290_368:
            return cand
    return None


def _build_hermes_header_prefix_from_fixture(out_dir: Path) -> Path:
    """Reconstruct a GGUF v3 binary prefix matching `hermes_4_3_36b_q4_k_m_metadata.json`
    plus real Q6_K `output.weight` rows from `hermes_output_weight_rows0_3_q6k.bin` for
    headless CI environments where the 21.7 GB model file is absent.
    """
    fixture = json.loads(HERMES_FIXTURE_PATH.read_text(encoding="utf-8"))
    kvs: Dict[str, Any] = fixture["metadata_kvs"]
    merges_meta: Dict[str, Any] = kvs["tokenizer.ggml.merges"]

    pre_parts: List[bytes] = [
        GGUF_MAGIC,
        struct.pack("<IQQ", 3, fixture["tensor_count"], fixture["kv_count"]),
    ]
    post_kv_parts: List[bytes] = []
    seen_merges = False

    for key, val in kvs.items():
        if key == "tokenizer.ggml.merges":
            pre_parts.append(_pack_gguf_string(key))
            pre_parts.append(struct.pack("<IIQ", 9, 8, int(merges_meta["count"])))
            seen_merges = True
            continue
        dest = post_kv_parts if seen_merges else pre_parts
        dest.append(_pack_gguf_string(key))
        if key == "tokenizer.ggml.tokens":
            count = int(val["count"])
            tokens = (
                list(val["first_5"])
                + [f"tok_{i}" for i in range(5, count - 5)]
                + list(val["last_5"])
            )
            dest.append(struct.pack("<IIQ", 9, 8, count))
            for tok in tokens:
                dest.append(_pack_gguf_string(tok))
        elif key == "tokenizer.ggml.token_type":
            count = int(val["count"])
            ttypes = (
                [int(x) for x in val["first_5"]]
                + ([1] * (count - 10))
                + [int(x) for x in val["last_5"]]
            )
            dest.append(struct.pack("<IIQ", 9, 5, count))
            dest.append(struct.pack(f"<{count}i", *ttypes))
        elif isinstance(val, str):
            dest.append(struct.pack("<I", 8))
            dest.append(_pack_gguf_string(val))
        elif isinstance(val, float):
            dest.append(struct.pack("<If", 6, val))
        elif isinstance(val, int):
            dest.append(struct.pack("<II", 4, val))
        else:
            raise TypeError(f"Unexpected KV type in fixture for {key}: {type(val)}")

    tensor_parts: List[bytes] = []
    for t in fixture["tensors"]:
        tensor_parts.append(_pack_gguf_string(t["name"]))
        dims = t["dims"]
        tensor_parts.append(struct.pack("<I", len(dims)))
        tensor_parts.append(struct.pack(f"<{len(dims)}Q", *dims))
        tensor_parts.append(struct.pack("<IQ", int(t["type_id"]), int(t["offset"])))

    pre_bytes = b"".join(pre_parts)
    merges_first = b"".join(_pack_gguf_string(m) for m in merges_meta["first_5"])
    merges_last = b"".join(_pack_gguf_string(m) for m in merges_meta["last_5"])
    post_bytes = b"".join(post_kv_parts) + b"".join(tensor_parts)

    target_header_end = int(fixture["header_end_offset"])
    filler_count = int(merges_meta["count"]) - 10
    fixed_bytes_total = (
        len(pre_bytes)
        + len(merges_first)
        + len(merges_last)
        + len(post_bytes)
        + filler_count * 8
    )
    remaining_payload_bytes = target_header_end - fixed_bytes_total
    if remaining_payload_bytes < filler_count:
        raise ValueError(
            f"Cannot pack exact header_end_offset={target_header_end}: remaining={remaining_payload_bytes}"
        )

    base_len, extra = divmod(remaining_payload_bytes, filler_count)
    filler_chunks = [
        _pack_gguf_string("a" * (base_len + (1 if i < extra else 0)))
        for i in range(filler_count)
    ]

    full_header = (
        pre_bytes
        + merges_first
        + b"".join(filler_chunks)
        + merges_last
        + post_bytes
    )
    assert len(full_header) == target_header_end

    pad_len = int(fixture["data_offset"]) - len(full_header)
    q6k_rows = HERMES_Q6K_FIXTURE_PATH.read_bytes()
    out_path = out_dir / "hermes_reconstructed_prefix.bin"
    out_path.write_bytes(full_header + (b"\x00" * pad_len) + q6k_rows)
    return out_path


def _get_hermes_prefix_bin(tmp_path: Path) -> Path:
    real_prefix = _find_hermes_prefix_bin()
    if real_prefix is not None:
        return real_prefix
    return _build_hermes_header_prefix_from_fixture(tmp_path)


def _pack_gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _build_synthetic_gguf_bytes(
    *,
    magic: bytes = GGUF_MAGIC,
    version: int = 3,
    kvs: List[Tuple[str, int, bytes]] | None = None,
    tensors: List[Tuple[str, Tuple[int, ...], int, int, bytes]] | None = None,
    alignment: int = 32,
    truncate_data_bytes: int = 0,
) -> bytes:
    """Construct a complete, byte-exact GGUF binary buffer for testing."""
    kvs = kvs or []
    tensors = tensors or []
    parts: List[bytes] = [
        magic,
        struct.pack("<IQQ", version, len(tensors), len(kvs)),
    ]
    for key, vtype, val_bytes in kvs:
        parts.append(_pack_gguf_string(key))
        parts.append(struct.pack("<I", vtype))
        parts.append(val_bytes)

    for name, dims, type_id, offset, _ in tensors:
        parts.append(_pack_gguf_string(name))
        parts.append(struct.pack("<I", len(dims)))
        parts.append(struct.pack(f"<{len(dims)}Q", *dims))
        parts.append(struct.pack("<IQ", type_id, offset))

    header = b"".join(parts)
    data_offset = ((len(header) + alignment - 1) // alignment) * alignment
    pad = b"\x00" * (data_offset - len(header))

    total_data_len = 0
    for _, _, _, offset, raw_data in tensors:
        total_data_len = max(total_data_len, offset + len(raw_data))
    data_buf = bytearray(total_data_len)
    for _, _, _, offset, raw_data in tensors:
        data_buf[offset : offset + len(raw_data)] = raw_data

    full = header + pad + bytes(data_buf)
    if truncate_data_bytes > 0:
        full = full[:-truncate_data_bytes]
    return full


def _make_random_quant_blocks(ggml_type: int, n_blocks: int, seed: int = 42) -> bytes:
    """Create deterministic quantization blocks with realistic non-NaN/Inf fp16 scales."""
    rng = np.random.default_rng(seed + ggml_type * 101)
    _, type_size = GGML_TYPE_INFO[ggml_type]

    if ggml_type == GGML_TYPE_F32:
        vals = rng.uniform(-5.0, 5.0, size=n_blocks).astype("<f4")
        return vals.tobytes()

    if ggml_type == GGML_TYPE_F16:
        vals = rng.uniform(-5.0, 5.0, size=n_blocks).astype("<f2")
        return vals.tobytes()

    if ggml_type == GGML_TYPE_BF16:
        vals = rng.uniform(-5.0, 5.0, size=n_blocks).astype(np.float32)
        u32 = vals.view(np.uint32)
        u16 = (u32 >> np.uint32(16)).astype("<u2")
        return u16.tobytes()

    raw = bytearray(rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8).tobytes())
    scales_a = rng.uniform(0.005, 0.25, size=n_blocks).astype("<f2")
    scales_b = rng.uniform(0.005, 0.25, size=n_blocks).astype("<f2")

    for i in range(n_blocks):
        base = i * type_size
        sa = struct.pack("<e", float(scales_a[i]))
        sb = struct.pack("<e", float(scales_b[i]))
        if ggml_type in (
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q8_0,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ4_XS,
        ):
            raw[base : base + 2] = sa
        elif ggml_type in (GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K):
            raw[base : base + 2] = sa
            raw[base + 2 : base + 4] = sb
        elif ggml_type == GGML_TYPE_Q2_K:
            raw[base + 80 : base + 82] = sa
            raw[base + 82 : base + 84] = sb
        elif ggml_type == GGML_TYPE_Q3_K:
            raw[base + 108 : base + 110] = sa
        elif ggml_type == GGML_TYPE_Q6_K:
            raw[base + 208 : base + 210] = sa
    return bytes(raw)


# ==============================================================================
# TEST SUITE T1: GGUF v3 Binary Parser, Hermes 4.3 36B Fixture & Corrupt Handling
# ==============================================================================


def test_t1_hermes_4_3_36b_metadata_fixture_structure() -> None:
    """Verify the saved Hermes 4.3 36B Q4_K_M metadata fixture is complete and accurate."""
    assert HERMES_FIXTURE_PATH.exists(), f"Missing fixture: {HERMES_FIXTURE_PATH}"
    fixture = json.loads(HERMES_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["repo_id"] == "NousResearch/Hermes-4.3-36B-GGUF"
    assert fixture["filename"] == "hermes-4_3_36b-Q4_K_M.gguf"
    assert fixture["magic"] == "GGUF"
    assert fixture["version"] == 3
    assert fixture["tensor_count"] == 771
    assert fixture["kv_count"] == 27
    assert fixture["alignment"] == 32
    assert fixture["header_end_offset"] == 6290352
    assert fixture["data_offset"] == 6290368

    # Architecture parameters (ByteDance Seed-OSS 36B base for Hermes 4.3 36B)
    assert fixture["architecture"] == "seed_oss"
    assert fixture["block_count"] == 64
    assert fixture["context_length"] == 524288
    assert fixture["embedding_length"] == 5120
    assert fixture["feed_forward_length"] == 27648
    assert fixture["attention.head_count"] == 80
    assert fixture["attention.head_count_kv"] == 8
    assert fixture["attention.key_length"] == 128
    assert fixture["attention.value_length"] == 128
    assert fixture["rope.freq_base"] == 10000000.0
    assert pytest.approx(fixture["attention.layer_norm_rms_epsilon"], rel=1e-6) == 1e-6
    assert fixture["bos_token_id"] == 0
    assert fixture["eos_token_id"] == 155127
    assert fixture["padding_token_id"] == 1
    assert fixture["vocab_size"] == 155136
    assert fixture["merges_count"] == 154737
    assert fixture["file_type"] == 15
    assert fixture["quantization_version"] == 2

    assert len(fixture["metadata_kvs"]) == 27
    assert len(fixture["tensors"]) == 771
    assert fixture["tensor_type_counts"] == {"Q6_K": 65, "F32": 321, "Q4_K": 385}

    # Check key tensors
    tensors_by_name = {t["name"]: t for t in fixture["tensors"]}
    assert tensors_by_name["output.weight"] == {
        "name": "output.weight",
        "n_dims": 2,
        "dims": [5120, 155136],
        "type_id": 14,
        "type": "Q6_K",
        "offset": 0,
    }
    assert tensors_by_name["token_embd.weight"] == {
        "name": "token_embd.weight",
        "n_dims": 2,
        "dims": [5120, 155136],
        "type_id": 12,
        "type": "Q4_K",
        "offset": 651591680,
    }
    assert tensors_by_name["blk.0.attn_q.weight"]["dims"] == [5120, 10240]
    assert tensors_by_name["blk.0.attn_k.weight"]["dims"] == [5120, 1024]
    assert tensors_by_name["blk.0.attn_v.weight"]["dims"] == [5120, 1024]
    assert tensors_by_name["blk.0.attn_output.weight"]["dims"] == [10240, 5120]
    assert tensors_by_name["blk.63.ffn_down.weight"]["dims"] == [27648, 5120]


def test_t1_gguf_loader_matches_hermes_prefix_bin_100_percent(tmp_path: Path) -> None:
    """Verify GGUFLoader parses the Hermes 4.3 36B GGUF prefix (and reconstructed CI fixture) identically to the metadata dump."""
    reconstructed = _build_hermes_header_prefix_from_fixture(tmp_path)
    targets = [reconstructed]
    real_prefix = _find_hermes_prefix_bin()
    if real_prefix is not None:
        targets.append(real_prefix)

    fixture = json.loads(HERMES_FIXTURE_PATH.read_text(encoding="utf-8"))
    for prefix_bin in targets:
        with GGUFLoader(prefix_bin, header_only=True) as loader:
            summary = loader.extract_architecture_summary()
            for key, expected_val in summary.items():
                assert fixture[key] == expected_val, f"Mismatch on summary field {key} ({prefix_bin})"

            assert len(loader.metadata) == 27
            assert set(loader.metadata.keys()) == set(fixture["metadata_kvs"].keys())
            assert len(loader.tensor_list) == 771

            for idx, (parsed_tensor, expected_tensor) in enumerate(
                zip(loader.tensor_list, fixture["tensors"])
            ):
                assert parsed_tensor.to_dict() == expected_tensor, f"Tensor mismatch at index {idx}"


def test_t1_gguf_loader_full_synthetic_file_and_row_lookup(tmp_path: Path) -> None:
    """Verify GGUFLoader on a complete multi-tensor GGUF v3 file with mmap & row slicing."""
    q4k_raw = _make_random_quant_blocks(GGML_TYPE_Q4_K, n_blocks=4, seed=10)  # 4 * 256 = 1024 elements -> shape (256, 4) in GGUF -> (4, 256) in NumPy
    f32_raw = _make_random_quant_blocks(GGML_TYPE_F32, n_blocks=256, seed=20)

    kvs = [
        ("general.architecture", 8, _pack_gguf_string("seed_oss")),
        ("seed_oss.block_count", 4, struct.pack("<I", 2)),
        ("seed_oss.embedding_length", 4, struct.pack("<I", 256)),
    ]
    off0 = 0
    off1 = len(q4k_raw)
    tensors = [
        ("token_embd.weight", (256, 4), GGML_TYPE_Q4_K, off0, q4k_raw),
        ("output_norm.weight", (256,), GGML_TYPE_F32, off1, f32_raw),
    ]
    gguf_bytes = _build_synthetic_gguf_bytes(kvs=kvs, tensors=tensors)
    gguf_file = tmp_path / "synthetic_valid.gguf"
    gguf_file.write_bytes(gguf_bytes)

    with GGUFLoader(gguf_file, header_only=False) as loader:
        assert loader.version == 3
        assert loader.tensor_count == 2
        assert loader.kv_count == 3
        assert loader.metadata["general.architecture"] == "seed_oss"
        assert loader.metadata["seed_oss.block_count"] == 2

        emb_info = loader.get_tensor_info("token_embd.weight")
        assert emb_info.dims == (256, 4)
        assert emb_info.numpy_shape == (4, 256)

        full_emb = loader.get_tensor_f32("token_embd.weight")
        assert full_emb.shape == (4, 256)
        assert full_emb.dtype == np.float32

        # Row lookup should match rows [3, 1] of full_emb exactly
        sub_rows = loader.get_tensor_rows_f32("token_embd.weight", [3, 1])
        assert sub_rows.shape == (2, 256)
        np.testing.assert_allclose(sub_rows[0], full_emb[3], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(sub_rows[1], full_emb[1], atol=0.0, rtol=0.0)

        norm_w = loader.get_tensor_f32("output_norm.weight")
        assert norm_w.shape == (256,)
        expected_norm = np.frombuffer(f32_raw, dtype="<f4")
        np.testing.assert_allclose(norm_w, expected_norm, atol=0.0, rtol=0.0)


def test_t1_gguf_loader_rejects_corrupt_and_truncated_files(tmp_path: Path) -> None:
    """Verify GGUFLoader raises clear errors on corrupt magic, bad version, truncated header, or truncated tensor data."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        GGUFLoader(tmp_path / "does_not_exist.gguf")

    # 2. Too small file
    tiny = tmp_path / "tiny.gguf"
    tiny.write_bytes(b"GGUF\x03\x00")
    with pytest.raises(ValueError, match="too small"):
        GGUFLoader(tiny)

    # 3. Bad magic
    bad_magic = tmp_path / "bad_magic.gguf"
    bad_magic.write_bytes(_build_synthetic_gguf_bytes(magic=b"BAD!"))
    with pytest.raises(ValueError, match="Invalid GGUF magic"):
        GGUFLoader(bad_magic)

    # 4. Unsupported version
    bad_ver = tmp_path / "bad_version.gguf"
    bad_ver.write_bytes(_build_synthetic_gguf_bytes(version=99))
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        GGUFLoader(bad_ver)

    # 5. Truncated header (KV claims 5000-byte string in a 60-byte file)
    trunc_hdr = tmp_path / "trunc_hdr.gguf"
    raw_hdr = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 5000) + b"abc"
    trunc_hdr.write_bytes(raw_hdr)
    with pytest.raises(ValueError, match="Corrupt or excessive string length"):
        GGUFLoader(trunc_hdr)

    # 6. Truncated tensor data payload (header_only=False must reject!)
    f32_raw = _make_random_quant_blocks(GGML_TYPE_F32, n_blocks=64, seed=7)
    trunc_data = tmp_path / "trunc_data.gguf"
    trunc_data.write_bytes(
        _build_synthetic_gguf_bytes(
            tensors=[("weight", (64,), GGML_TYPE_F32, 0, f32_raw)],
            truncate_data_bytes=12,
        )
    )
    with pytest.raises(ValueError, match="Truncated GGUF tensor data"):
        GGUFLoader(trunc_data, header_only=False)

    # 7. Real/reconstructed prefix file opened with header_only=False must raise ValueError because full 21.7GB tensors are truncated
    prefix_bin = _get_hermes_prefix_bin(tmp_path)
    with pytest.raises(ValueError, match="Truncated GGUF tensor data"):
        GGUFLoader(prefix_bin, header_only=False)


# ==============================================================================
# TEST SUITE T2: Dequantization Kernels Accuracy (< 1e-6 vs Scalar Reference)
# ==============================================================================


@pytest.mark.parametrize(
    "ggml_type,n_blocks",
    [
        (GGML_TYPE_F32, 64),
        (GGML_TYPE_F16, 64),
        (GGML_TYPE_BF16, 64),
        (GGML_TYPE_Q4_0, 16),
        (GGML_TYPE_Q4_1, 16),
        (GGML_TYPE_Q5_0, 16),
        (GGML_TYPE_Q5_1, 16),
        (GGML_TYPE_Q8_0, 16),
        (GGML_TYPE_Q2_K, 8),
        (GGML_TYPE_Q3_K, 8),
        (GGML_TYPE_Q4_K, 12),
        (GGML_TYPE_Q5_K, 8),
        (GGML_TYPE_Q6_K, 12),
        (GGML_TYPE_IQ4_NL, 16),
        (GGML_TYPE_IQ3_S, 12),
        (GGML_TYPE_IQ4_XS, 12),
    ],
)
def test_t2_vectorized_dequant_matches_scalar_reference(ggml_type: int, n_blocks: int) -> None:
    """Verify vectorized `dequantize_bytes` matches `dequantize_block_scalar_reference` within 1e-6."""
    block_size = get_block_size(ggml_type)
    type_size = get_type_size(ggml_type)
    num_elements = n_blocks * block_size
    raw_bytes = _make_random_quant_blocks(ggml_type, n_blocks=n_blocks, seed=12345)

    vec_out = dequantize_bytes(raw_bytes, ggml_type, num_elements)
    assert vec_out.shape == (num_elements,)
    assert vec_out.dtype == np.float32
    assert np.all(np.isfinite(vec_out))

    scalar_all: List[float] = []
    for b_idx in range(n_blocks):
        block_slice = raw_bytes[b_idx * type_size : (b_idx + 1) * type_size]
        scalar_all.extend(dequantize_block_scalar_reference(block_slice, ggml_type))

    scalar_arr = np.array(scalar_all, dtype=np.float32)
    max_abs_err = float(np.max(np.abs(vec_out - scalar_arr)))
    assert max_abs_err < 1e-6, (
        f"Dequantization error {max_abs_err} >= 1e-6 for type_id={ggml_type}"
    )


def test_t2_real_hermes_output_weight_q6_k_blocks(tmp_path: Path) -> None:
    """Dequantize real Q6_K super-blocks from `output.weight` at offset 6290368."""
    prefix_bin = _get_hermes_prefix_bin(tmp_path)

    with GGUFLoader(prefix_bin, header_only=True) as loader:
        assert loader.data_offset == 6290368
        out_info = loader.get_tensor_info("output.weight")
        assert out_info.type_id == GGML_TYPE_Q6_K
        assert out_info.offset == 0
        assert out_info.dims == (5120, 155136)

        # Dequantize the first 4 rows of `output.weight` (each row is 5120 elements = 20 Q6_K blocks = 4200 bytes)
        rows_f32 = loader.get_tensor_rows_f32("output.weight", [0, 1, 2, 3])
        assert rows_f32.shape == (4, 5120)
        assert rows_f32.dtype == np.float32
        assert np.all(np.isfinite(rows_f32))
        assert float(np.std(rows_f32)) > 1e-4

    # Cross-check the first 20 Q6_K blocks (row 0 = 5120 elements) against the pure-Python scalar oracle
    n_blocks = 20
    type_size = get_type_size(GGML_TYPE_Q6_K)  # 210 bytes
    with open(prefix_bin, "rb") as f:
        f.seek(6290368)
        raw_row0 = f.read(n_blocks * type_size)

    vec_row0 = dequantize_bytes(raw_row0, GGML_TYPE_Q6_K, n_blocks * 256)
    scalar_row0: List[float] = []
    for b_idx in range(n_blocks):
        blk = raw_row0[b_idx * type_size : (b_idx + 1) * type_size]
        scalar_row0.extend(dequantize_block_scalar_reference(blk, GGML_TYPE_Q6_K))

    scalar_arr = np.array(scalar_row0, dtype=np.float32)
    max_abs_err = float(np.max(np.abs(vec_row0 - scalar_arr)))
    assert max_abs_err < 1e-6
    np.testing.assert_allclose(rows_f32[0], vec_row0, atol=0.0, rtol=0.0)


def test_t2_dequant_validation_errors() -> None:
    """Verify dequantize_bytes and compute_tensor_bytes reject invalid sizes or unknown types."""
    with pytest.raises(ValueError, match="Unsupported GGML type"):
        get_block_size(999)

    with pytest.raises(ValueError, match="not divisible by block size"):
        compute_tensor_bytes(GGML_TYPE_Q4_K, 100)

    with pytest.raises(ValueError, match="Byte buffer length mismatch"):
        dequantize_bytes(b"\x00" * 10, GGML_TYPE_Q4_K, 256)
