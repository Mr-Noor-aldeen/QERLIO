"""Test T3 — Native BPE Tokenizer Accuracy & Round-Trip Verification.

Verifies `core.inference.tokenizer.GGUFTokenizer` against the 64 reference cases
generated offline from `NousResearch/Hermes-4.3-36B` (`tests/fixtures/hermes_tokenizer_reference.json`).
Ensures zero prohibited runtime imports (`transformers`, `tokenizers`, `tiktoken`, `gguf`, `torch`),
exact token-for-token encode match, and exact byte-for-byte decode round-trip match.
"""

from __future__ import annotations

import ast
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.inference.tokenizer import (
    GGUFTokenizer,
    _bytes_to_unicode,
    _unicode_to_bytes,
    read_gguf_tokenizer_metadata,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_SOURCE_PATH = REPO_ROOT / "core" / "inference" / "tokenizer.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "hermes_tokenizer_reference.json"

CANDIDATE_GGUF_PATHS = [
    REPO_ROOT / "models" / "hermes-4_3_36b-Q4_K_M.gguf",
    *(
        [Path(os.environ["QERLIO_HERMES_GGUF_PATH"])]
        if os.environ.get("QERLIO_HERMES_GGUF_PATH")
        else []
    ),
    REPO_ROOT / "tests" / "fixtures" / "hermes_prefix.bin",
]

PROHIBITED_MODULES = {"transformers", "tokenizers", "tiktoken", "gguf", "torch"}


def _load_reference_fixture() -> Dict[str, Any]:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _pack_gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _build_gguf_prefix_bytes_from_fixture(fixture: Dict[str, Any]) -> bytes:
    """Builds a valid GGUF v3 binary header from the reference fixture's fallback tables."""
    vocab_size = int(fixture["vocab_size"])
    tokens = [f"<|unused_{i}|>" for i in range(vocab_size)]
    token_types = [1] * vocab_size

    sparse_tokens: Dict[str, List[Any]] = fixture["gguf_fallback_tables"]["sparse_tokens"]
    for str_idx, (tok_str, ttype) in sparse_tokens.items():
        idx = int(str_idx)
        tokens[idx] = tok_str
        token_types[idx] = int(ttype)

    merges: List[str] = fixture["gguf_fallback_tables"]["ordered_merges"]

    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, 0, 7))  # version=3, tensor_count=0, kv_count=7

    # 1. tokenizer.ggml.model = "gpt2"
    buf.extend(_pack_gguf_string("tokenizer.ggml.model"))
    buf.extend(struct.pack("<I", 8))
    buf.extend(_pack_gguf_string(fixture.get("tokenizer_model", "gpt2")))

    # 2. tokenizer.ggml.pre = "seed-coder"
    buf.extend(_pack_gguf_string("tokenizer.ggml.pre"))
    buf.extend(struct.pack("<I", 8))
    buf.extend(_pack_gguf_string(fixture.get("tokenizer_pre", "seed-coder")))

    # 3. tokenizer.ggml.tokens (ARRAY of STRING)
    buf.extend(_pack_gguf_string("tokenizer.ggml.tokens"))
    buf.extend(struct.pack("<IIQ", 9, 8, len(tokens)))
    for t in tokens:
        buf.extend(_pack_gguf_string(t))

    # 4. tokenizer.ggml.token_type (ARRAY of INT32)
    buf.extend(_pack_gguf_string("tokenizer.ggml.token_type"))
    buf.extend(struct.pack("<IIQ", 9, 5, len(token_types)))
    buf.extend(struct.pack(f"<{len(token_types)}i", *token_types))

    # 5. tokenizer.ggml.merges (ARRAY of STRING)
    buf.extend(_pack_gguf_string("tokenizer.ggml.merges"))
    buf.extend(struct.pack("<IIQ", 9, 8, len(merges)))
    for m in merges:
        buf.extend(_pack_gguf_string(m))

    # 6. tokenizer.ggml.bos_token_id (UINT32)
    buf.extend(_pack_gguf_string("tokenizer.ggml.bos_token_id"))
    buf.extend(struct.pack("<II", 4, int(fixture.get("bos_token_id", 0))))

    # 7. tokenizer.ggml.eos_token_id (UINT32)
    buf.extend(_pack_gguf_string("tokenizer.ggml.eos_token_id"))
    buf.extend(struct.pack("<II", 4, int(fixture.get("eos_token_id", 155127))))

    return bytes(buf)


@pytest.fixture(scope="module")
def reference_fixture() -> Dict[str, Any]:
    assert FIXTURE_PATH.exists(), f"Missing reference fixture at {FIXTURE_PATH}"
    data = _load_reference_fixture()
    assert data.get("model_repo") == "NousResearch/Hermes-4.3-36B"
    assert len(data.get("cases", [])) >= 55
    return data


@pytest.fixture(scope="module")
def tokenizer(reference_fixture: Dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> GGUFTokenizer:
    """Build deterministic GGUF prefix from the committed Hermes tokenizer reference fixture.
    (Real-weight full-file T3 check runs in opt-in `tests/test_real_hermes_36b_optin.py`.)
    """
    tmp_dir = tmp_path_factory.mktemp("gguf_tokenizer_prefix")
    prefix_gguf = tmp_dir / "hermes_tokenizer_prefix.gguf"
    prefix_gguf.write_bytes(_build_gguf_prefix_bytes_from_fixture(reference_fixture))
    return GGUFTokenizer.from_gguf_file(str(prefix_gguf))


def test_t3_zero_prohibited_imports() -> None:
    """Verifies that `core/inference/tokenizer.py` uses only the Python standard library."""
    source = TOKENIZER_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TOKENIZER_SOURCE_PATH))

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])

    forbidden = imported_roots.intersection(PROHIBITED_MODULES)
    assert not forbidden, f"Prohibited imports found in tokenizer.py: {forbidden}"


def test_t3_byte_to_unicode_bijection() -> None:
    """Verifies GPT-2 byte-to-unicode table is an exact 256-entry bijection."""
    b2u = _bytes_to_unicode()
    u2b = _unicode_to_bytes()
    assert len(b2u) == 256
    assert len(u2b) == 256
    for b in range(256):
        ch = b2u[b]
        assert isinstance(ch, str) and len(ch) == 1
        assert u2b[ch] == b
    # Space (0x20 = 32) maps to 'Ġ' (U+0120 = 288)
    assert b2u[32] == "Ġ"
    assert u2b["Ġ"] == 32


def test_t3_vocabulary_and_special_tokens(tokenizer: GGUFTokenizer) -> None:
    """Verifies Hermes 4.3 36B vocabulary dimensions and special/added token IDs."""
    assert tokenizer.vocab_size == 155136
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 155127
    assert tokenizer.pre_type == "seed-coder"

    expected_specials = {
        "<seed:bos>": 0,
        "<seed:pad>": 1,
        "<seed:eos>": 2,
        "<seed:think>": 3,
        "</seed:think>": 4,
        "<seed:cot_budget_reflect>": 5,
        "</seed:cot_budget_reflect>": 6,
        "<tool_call>": 155121,
        "</tool_call>": 155122,
        "<tool_response>": 155123,
        "</tool_response>": 155124,
        "<|start_header_id|>": 155125,
        "<|end_header_id|>": 155126,
        "<|eot_id|>": 155127,
        "<think>": 155128,
        "</think>": 155129,
    }
    for tok_str, expected_id in expected_specials.items():
        assert tokenizer.id_to_token[expected_id] == tok_str
        assert tokenizer.added_token_to_id[tok_str] == expected_id
        assert tokenizer.encode(tok_str) == [expected_id]


@pytest.mark.parametrize(
    "case",
    _load_reference_fixture()["cases"],
    ids=lambda c: c["id"],
)
def test_t3_reference_encode_and_roundtrip(tokenizer: GGUFTokenizer, case: Dict[str, Any]) -> None:
    """Verifies exact token-for-token encoding and byte-for-byte decode round-trip on all 64 cases."""
    text = case["text"]
    expected_ids = case["token_ids"]

    encoded_ids = tokenizer.encode(text)
    assert encoded_ids == expected_ids, (
        f"Token ID mismatch on case '{case['id']}':\n"
        f"  expected ({len(expected_ids)}): {expected_ids}\n"
        f"  actual   ({len(encoded_ids)}): {encoded_ids}"
    )

    decoded_text = tokenizer.decode(encoded_ids)
    assert decoded_text == text, (
        f"Round-trip decode mismatch on case '{case['id']}':\n"
        f"  expected: {text!r}\n"
        f"  actual:   {decoded_text!r}"
    )


def test_t3_add_bos_and_skip_special_tokens(tokenizer: GGUFTokenizer) -> None:
    """Verifies `add_bos=True`, `parse_special=False`, and `skip_special_tokens=True` options."""
    ids_no_bos = tokenizer.encode("Hello, world!", add_bos=False)
    ids_with_bos = tokenizer.encode("Hello, world!", add_bos=True)
    assert ids_with_bos == [0] + ids_no_bos

    assert tokenizer.decode(ids_with_bos, skip_special_tokens=False) == "<seed:bos>Hello, world!"
    assert tokenizer.decode(ids_with_bos, skip_special_tokens=True) == "Hello, world!"

    # When parse_special=False, `<|eot_id|>` is tokenized into ordinary subword tokens instead of [155127]
    unparsed_ids = tokenizer.encode("<|eot_id|>", parse_special=False)
    assert unparsed_ids != [155127]
    assert len(unparsed_ids) > 1
    assert tokenizer.decode(unparsed_ids) == "<|eot_id|>"


def test_t3_gguf_binary_prefix_builder_fallback(
    reference_fixture: Dict[str, Any], tmp_path: Path
) -> None:
    """Verifies that `GGUFTokenizer.from_gguf_file` parses a synthesized binary GGUF prefix identically."""
    gguf_bytes = _build_gguf_prefix_bytes_from_fixture(reference_fixture)
    gguf_path = tmp_path / "synthetic_hermes_prefix.gguf"
    gguf_path.write_bytes(gguf_bytes)

    metadata = read_gguf_tokenizer_metadata(str(gguf_path))
    assert metadata["tokenizer.ggml.model"] == "gpt2"
    assert metadata["tokenizer.ggml.pre"] == "seed-coder"

    fallback_tok = GGUFTokenizer.from_gguf_file(str(gguf_path))
    for case in reference_fixture["cases"]:
        ids = fallback_tok.encode(case["text"])
        assert ids == case["token_ids"], f"Fallback GGUF mismatch on {case['id']}"
        assert fallback_tok.decode(ids) == case["text"]
