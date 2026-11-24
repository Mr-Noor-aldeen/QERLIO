"""Opt-in end-to-end verification on the official 21.76 GB Hermes-4.3-36B Q4_K_M GGUF.

Per Directive v0.3.1 (Section 4 F4c):
- Disabled in default CI runs unless BOTH `QERLIO_RUN_REAL_HERMES=1` is set AND the
  official `hermes-4_3_36b-Q4_K_M.gguf` file is present on disk.
- NEVER falls back to a synthetic/toy GGUF. If either condition is absent, skips cleanly
  with an explicit reason.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.inference.gguf_loader import GGUFLoader
from core.inference.seed_oss_model import SeedOssEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_FILENAME = "hermes-4_3_36b-Q4_K_M.gguf"
OFFICIAL_SIZE_BYTES = 21_762_145_216
OFFICIAL_SHA256 = "17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08"


def _resolve_official_gguf_path() -> Path:
    env_path = os.environ.get("QERLIO_HERMES_GGUF_PATH")
    if env_path:
        return Path(env_path)
    return REPO_ROOT / "models" / OFFICIAL_FILENAME


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def test_optin_real_hermes_4_3_36b_official_gguf() -> None:
    if os.environ.get("QERLIO_RUN_REAL_HERMES") != "1":
        pytest.skip("Opt-in real Hermes-4.3-36B test skipped (QERLIO_RUN_REAL_HERMES!=1).")

    gguf_path = _resolve_official_gguf_path()
    if not gguf_path.is_file():
        pytest.skip(
            f"Opt-in real Hermes-4.3-36B test skipped (official GGUF absent at {gguf_path.name})."
        )

    assert gguf_path.stat().st_size == OFFICIAL_SIZE_BYTES, (
        f"Unexpected file size for {gguf_path.name}: {gguf_path.stat().st_size} != {OFFICIAL_SIZE_BYTES}"
    )
    actual_sha = _sha256_file(gguf_path)
    assert actual_sha == OFFICIAL_SHA256, (
        f"Official GGUF SHA-256 mismatch: {actual_sha} != {OFFICIAL_SHA256}"
    )

    meta_fixture = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "hermes_4_3_36b_q4_k_m_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    with GGUFLoader(gguf_path, header_only=False) as loader:
        assert len(loader.metadata) == 27
        assert len(loader.tensor_list) == 771
        for parsed_t, expected_t in zip(loader.tensor_list, meta_fixture["tensors"]):
            assert parsed_t.to_dict() == expected_t

    tok_fixture = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "hermes_tokenizer_reference.json").read_text(
            encoding="utf-8"
        )
    )
    ref_fixture = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "hermes_4_3_36b_transformers_reference.json").read_text(
            encoding="utf-8"
        )
    )
    p0_ref = ref_fixture["prompts"][0]

    with SeedOssEngine(gguf_path, max_context_length=512, preload_weights=False, header_only=False) as eng:
        assert eng.tokenizer is not None
        for case in tok_fixture["cases"]:
            enc = eng.tokenizer.encode(case["text"])
            assert enc == case["token_ids"], f"Real GGUF tokenizer mismatch on {case['id']}"
            assert eng.tokenizer.decode(enc, skip_special_tokens=False) == case["text"]

        prompt_ids = eng.tokenizer.encode(p0_ref["prompt_text"], add_bos=True, parse_special=False)
        assert prompt_ids == p0_ref["prompt_token_ids"]

        logits = eng.forward(prompt_ids, last_token_only=True)
        assert logits.shape == (155136,)
        assert bool(np.all(np.isfinite(logits)))

        top1_id = int(np.argmax(logits))
        assert top1_id == p0_ref["top1_token_id"] == 10805
        assert eng.tokenizer.decode([top1_id], skip_special_tokens=False) == " Paris"

        q_top5 = {int(x) for x in np.argsort(logits)[-5:]}
        r_top5 = {int(item["token_id"]) for item in p0_ref["top20"][:5]}
        assert len(q_top5 & r_top5) == 5

        ref_top20_ids = [int(item["token_id"]) for item in p0_ref["top20"]]
        ref_top20_vals = np.asarray([float(item["logit"]) for item in p0_ref["top20"]], dtype=np.float32)
        max_diff = float(np.max(np.abs(logits[ref_top20_ids] - ref_top20_vals)))
        assert max_diff < 1e-4, f"Real 36B top-20 logit diff {max_diff} >= 1e-4"
