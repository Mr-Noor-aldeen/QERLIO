import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.model_registry.architecture_catalog import ArchitectureCatalog
from server.local_api.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "registry" / "model_architectures.json"
ZAQERIS_NATIVE_IDS = {"zaqeris_axiom", "zqn_12b", "zaqeris_7b"}
VALID_STATUSES = {"tested", "experimental"}


def _raw_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_architectures_count_matches_metadata_and_ids_unique():
    cat = ArchitectureCatalog()
    raw = _raw_registry()
    all_archs = cat.list_all()
    assert cat.total_count == len(all_archs) == raw["total_count"]
    ids = [a["id"] for a in raw["architectures"]]
    assert len(ids) == len(set(ids))


def test_support_status_tiers():
    raw = _raw_registry()
    archs = raw["architectures"]
    for item in archs:
        assert item.get("support_status") in VALID_STATUSES, item["id"]
    tested = {a["id"] for a in archs if a["support_status"] == "tested"}
    experimental = [a for a in archs if a["support_status"] == "experimental"]
    assert raw["tested_count"] == len(tested)
    assert raw["experimental_count"] == len(experimental)
    assert raw["tested_count"] + raw["experimental_count"] == raw["total_count"]
    # Only architectures with a code path exercised in this repo may be marked tested.
    assert tested == ZAQERIS_NATIVE_IDS
    seed = next(a for a in archs if a["id"] == "seed_36b")
    assert seed["support_status"] == "experimental"
    assert "Seed-OSS" in seed["name"]
    assert seed["huggingface_repo"] == "NousResearch/Hermes-4.3-36B"
    assert seed["gguf_repo"] == "NousResearch/Hermes-4.3-36B-GGUF"


def test_architectures_schema_completeness():
    cat = ArchitectureCatalog()
    required_keys = {
        "id", "name", "family", "developer", "attention_type",
        "positional_embedding", "activation", "norm_type",
        "supported_quantizations", "native_tools_supported",
        "parameters_range", "primary_use", "support_status"
    }
    for item in cat.list_all():
        for k in required_keys:
            assert k in item
            assert item[k] is not None
        assert isinstance(item["supported_quantizations"], list)
        assert len(item["supported_quantizations"]) > 0


def test_zaqeris_native_models_present():
    cat = ArchitectureCatalog()
    zaqeris_items = cat.list_by_family("ZAQERIS Native")
    assert {z["id"] for z in zaqeris_items} == ZAQERIS_NATIVE_IDS
    axiom = cat.get("zaqeris_axiom")
    assert "79.43B" in axiom["parameters_range"]
    assert "12.39B" in axiom["parameters_range"]


def test_no_placeholder_or_unreleased_entries():
    names = " ".join(a["name"].lower() + " " + a["developer"].lower() for a in _raw_registry()["architectures"])
    for banned in ("chinchilla", "yi-lightning", "femto", "pico qwen", "edge research consortium", "stripemamba"):
        assert banned not in names


def test_architectures_search_and_family_filter():
    cat = ArchitectureCatalog()
    families = cat.list_families()
    assert len(families) >= 8
    assert "Dense Transformer" in families
    assert "Mixture of Experts" in families
    assert "Reasoning & Dialectic" in families

    qwen_matches = cat.search("qwen")
    assert len(qwen_matches) >= 5


def test_architectures_api_endpoints():
    raw = _raw_registry()
    res = client.get("/v1/architectures")
    assert res.status_code == 200
    data = res.json()
    assert data["global_supported_count"] == raw["total_count"]
    assert data["tested_count"] == raw["tested_count"]
    assert data["experimental_count"] == raw["experimental_count"]
    assert len(data["data"]) == raw["total_count"]

    res_single = client.get("/v1/architectures/zqn_12b")
    assert res_single.status_code == 200
    single_data = res_single.json()
    assert single_data["id"] == "zqn_12b"
    assert single_data["native_tools_supported"] is True
    assert single_data["support_status"] == "tested"

    res_tested = client.get("/v1/architectures", params={"status": "tested"})
    assert res_tested.status_code == 200
    assert {a["id"] for a in res_tested.json()["data"]} == ZAQERIS_NATIVE_IDS

    res_404 = client.get("/v1/architectures/non_existent_arch_xyz")
    assert res_404.status_code == 404
