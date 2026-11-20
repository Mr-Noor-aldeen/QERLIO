import pytest
from core.model_registry.architecture_catalog import ArchitectureCatalog
from server.local_api.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_architectures_total_count_and_uniqueness():
    cat = ArchitectureCatalog()
    assert cat.total_count == 200
    all_archs = cat.list_all()
    assert len(all_archs) == 200
    ids = [a["id"] for a in all_archs]
    assert len(ids) == len(set(ids))

def test_architectures_schema_completeness():
    cat = ArchitectureCatalog()
    required_keys = {
        "id", "name", "family", "developer", "attention_type",
        "positional_embedding", "activation", "norm_type",
        "supported_quantizations", "native_tools_supported",
        "parameters_range", "primary_use"
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
    assert len(zaqeris_items) == 3
    zaqeris_ids = {z["id"] for z in zaqeris_items}
    assert zaqeris_ids == {"zaqeris_axiom", "zaqion_dialectic", "zaqeris_7b"}

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
    res = client.get("/v1/architectures")
    assert res.status_code == 200
    data = res.json()
    assert data["global_supported_count"] == 200
    assert len(data["data"]) == 200

    res_single = client.get("/v1/architectures/zaqion_dialectic")
    assert res_single.status_code == 200
    single_data = res_single.json()
    assert single_data["id"] == "zaqion_dialectic"
    assert single_data["native_tools_supported"] is True

    res_404 = client.get("/v1/architectures/non_existent_arch_xyz")
    assert res_404.status_code == 404
