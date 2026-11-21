import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import struct
import hashlib
import tempfile
from tools.qerlio_cli import parse_gguf_raw

def create_synthetic_gguf(file_path, magic=b"GGUF", version=3, tensor_count=10, kv_count=5, extra_bytes=b""):
    with open(file_path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", tensor_count))
        f.write(struct.pack("<Q", kv_count))
        if extra_bytes:
            f.write(extra_bytes)

def test_valid_synthetic_gguf(tmp_path):
    model_file = tmp_path / "valid_model.gguf"
    create_synthetic_gguf(str(model_file), magic=b"GGUF", version=3, tensor_count=42, kv_count=8)
    
    info = parse_gguf_raw(str(model_file))
    assert info["magic"] == "GGUF"
    assert info["version"] == 3
    assert info["tensor_count"] == 42
    assert info["metadata_kv_count"] == 8
    assert info["file_size_bytes"] == 24

def test_invalid_magic_rejected(tmp_path):
    model_file = tmp_path / "invalid_magic.gguf"
    create_synthetic_gguf(str(model_file), magic=b"FAKE", version=3)
    
    with pytest.raises(ValueError, match="Invalid GGUF magic"):
        parse_gguf_raw(str(model_file))

def test_unsupported_version_rejected(tmp_path):
    model_file = tmp_path / "unsupported_version.gguf"
    create_synthetic_gguf(str(model_file), magic=b"GGUF", version=99)
    
    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        parse_gguf_raw(str(model_file))

def test_truncated_file_rejected(tmp_path):
    model_file = tmp_path / "truncated.gguf"
    with open(model_file, "wb") as f:
        f.write(b"GGUF\x03\x00")  # Only 6 bytes
        
    with pytest.raises(ValueError, match="File is too small"):
        parse_gguf_raw(str(model_file))

def test_sha256_verification(tmp_path):
    model_file = tmp_path / "sample.gguf"
    create_synthetic_gguf(str(model_file))
    
    with open(model_file, "rb") as f:
        expected = hashlib.sha256(f.read()).hexdigest()
        
    # Re-verify matching
    with open(model_file, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    assert expected == actual


def _gguf_kv_str(key: str, val: str) -> bytes:
    kb = key.encode("utf-8")
    vb = val.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _gguf_kv_u32(key: str, val: int) -> bytes:
    kb = key.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", val)


def test_gguf_metadata_kv_parsing_seed_oss(tmp_path):
    model_file = tmp_path / "hermes_4_3_36b.gguf"
    kv_payload = b"".join([
        _gguf_kv_str("general.architecture", "seed_oss"),
        _gguf_kv_str("general.name", "Hermes-4.3-36B"),
        _gguf_kv_u32("general.file_type", 15),  # Q4_K_M
        _gguf_kv_u32("seed_oss.context_length", 524288),
        _gguf_kv_u32("seed_oss.embedding_length", 5120),
        _gguf_kv_u32("seed_oss.block_count", 64),
    ])
    create_synthetic_gguf(
        str(model_file),
        magic=b"GGUF",
        version=3,
        tensor_count=580,
        kv_count=6,
        extra_bytes=kv_payload
    )

    info = parse_gguf_raw(str(model_file))
    assert info["architecture"] == "seed_oss"
    assert info["name"] == "Hermes-4.3-36B"
    assert info["quantization_type"] == "Q4_K_M"
    assert info["context_length"] == 524288
    assert info["embedding_length"] == 5120
    assert info["block_count"] == 64
