import os
import sys
import json
import io
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from core.models.hermes_adapter import (
    Hermes36BAdapter,
    THINKING_SYSTEM_PROMPT,
    STANDARD_SYSTEM_PROMPT,
    DEFAULT_GENERATION_CONFIG,
)
from core.remote.hf_remote_client import HFRemoteClient
from server.hf_space_gateway.app import app as space_gateway_app


def test_hermes_adapter_base_specs():
    adapter = Hermes36BAdapter(thinking=True)
    assert adapter.base_model == "ByteDance Seed 36B"
    assert adapter.bos_token == "<seed:bos>"
    assert adapter.eos_token == "<|eot_id|>"
    assert DEFAULT_GENERATION_CONFIG["temperature"] == 0.6
    assert DEFAULT_GENERATION_CONFIG["top_p"] == 0.95
    assert DEFAULT_GENERATION_CONFIG["top_k"] == 20


def test_hermes_chat_prompt_standard():
    adapter = Hermes36BAdapter(thinking=False)
    messages = [
        {"role": "user", "content": "What is the capital of Egypt?"}
    ]
    prompt = adapter.format_chat_prompt(messages)
    assert prompt.startswith("<seed:bos>")
    assert STANDARD_SYSTEM_PROMPT in prompt
    assert "<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of Egypt?<|eot_id|>" in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_hermes_chat_prompt_thinking():
    adapter = Hermes36BAdapter(thinking=True)
    messages = [
        {"role": "user", "content": "Derive the gradient of cross-entropy loss."}
    ]
    prompt = adapter.format_chat_prompt(messages)
    assert prompt.startswith("<seed:bos>")
    assert THINKING_SYSTEM_PROMPT in prompt
    assert "<think> </think>" in prompt
    assert "Derive the gradient of cross-entropy loss." in prompt


def test_hermes_chat_prompt_with_tools():
    adapter = Hermes36BAdapter(thinking=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Fetch current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }
    ]
    messages = [{"role": "user", "content": "How is the weather in Cairo?"}]
    prompt = adapter.format_chat_prompt(messages, tools=tools)
    assert "<tools>" in prompt
    assert "lookup_weather" in prompt
    assert "</tools>" in prompt
    assert "<tool_call>" in prompt


def test_hermes_parse_generation_with_thinking():
    adapter = Hermes36BAdapter()
    raw = (
        "<think>\nThe user wants to know the capital of Egypt. "
        "Egypt's capital is Cairo.</think>\n"
        "The capital of Egypt is Cairo."
    )
    parsed = adapter.parse_generation(raw)
    assert parsed["has_thinking"] is True
    assert "The user wants to know the capital of Egypt." in parsed["thinking"]
    assert parsed["final_content"] == "The capital of Egypt is Cairo."
    assert parsed["has_tool_calls"] is False
    assert parsed["tool_calls"] == []


def test_hermes_parse_generation_with_tool_call():
    adapter = Hermes36BAdapter()
    raw = (
        "<think>\nNeed to evaluate 42 * 100.</think>\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"expr": "42 * 100"}}\n</tool_call>'
    )
    parsed = adapter.parse_generation(raw)
    assert parsed["has_thinking"] is True
    assert "Need to evaluate 42 * 100." in parsed["thinking"]
    assert parsed["has_tool_calls"] is True
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "calculator"
    assert parsed["tool_calls"][0]["arguments"] == {"expr": "42 * 100"}


def test_hermes_parse_generation_plain():
    adapter = Hermes36BAdapter()
    raw = "Direct answer without special tags."
    parsed = adapter.parse_generation(raw)
    assert parsed["has_thinking"] is False
    assert parsed["thinking"] is None
    assert parsed["has_tool_calls"] is False
    assert parsed["final_content"] == "Direct answer without special tags."


def test_hf_remote_client_mock_completion():
    client = HFRemoteClient(base_url="https://mock-space.hf.space/v1", token="hf_test_123")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer hf_test_123"

    mock_resp_data = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>\nTesting thinking.\n</think>\nMock answer."
                }
            }
        ]
    }
    mock_bytes = json.dumps(mock_resp_data).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_bytes
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.generate(
            model="NousResearch/Hermes-4.3-36B",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert res["thinking"] == "Testing thinking."
        assert res["content"] == "Mock answer."
        assert res["model"] == "NousResearch/Hermes-4.3-36B"


def test_hf_remote_client_mock_streaming():
    client = HFRemoteClient(base_url="https://mock-space.hf.space/v1")
    sse_lines = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        b'data: {"choices": [{"delta": {"content": " World"}}]}\n',
        b'data: [DONE]\n'
    ]

    mock_resp = MagicMock()
    mock_resp.__iter__.return_value = iter(sse_lines)
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        chunks = list(client.stream_generate(
            model="NousResearch/Hermes-4.3-36B",
            messages=[{"role": "user", "content": "Hi"}]
        ))
        assert chunks == ["Hello", " World"]


def test_space_gateway_api():
    test_client = TestClient(space_gateway_app)

    health_res = test_client.get("/healthz")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "healthy"
    assert "NousResearch/Hermes-4.3-36B" in health_data["models"]
    assert "ZAQERIS/ZAQERIS-7B" in health_data["models"]

    models_res = test_client.get("/v1/models")
    assert models_res.status_code == 200
    models_data = models_res.json()
    model_ids = [m["id"] for m in models_data["data"]]
    assert "NousResearch/Hermes-4.3-36B" in model_ids
    assert "ZAQERIS/ZAQERIS-7B" in model_ids

    chat_res = test_client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": "Test prompt"}],
        "stream": False
    })
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "choices" in chat_data
    content = chat_data["choices"][0]["message"]["content"]
    assert "<think>" in content
    assert "ByteDance Seed 36B" in content

    tools_res = test_client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": "Compute 42 * 2"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculates math expressions"
                }
            }
        ]
    })
    assert tools_res.status_code == 200
    tools_data = tools_res.json()
    assert tools_data["choices"][0]["finish_reason"] == "tool_calls"
    assert len(tools_data["choices"][0]["message"]["tool_calls"]) == 1


def test_space_gateway_secret_auth_and_expiry():
    test_client = TestClient(space_gateway_app)

    with patch.dict(os.environ, {"QERLIO_GATEWAY_SECRET": "secret_zaq_999"}):
        # 1. Missing secret key -> 401
        res_unauth = test_client.get("/v1/models")
        assert res_unauth.status_code == 401

        # 2. Invalid secret key -> 401
        res_wrong = test_client.post(
            "/v1/chat/completions",
            json={"model": "NousResearch/Hermes-4.3-36B", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong_token"}
        )
        assert res_wrong.status_code == 401

        # 3. Valid secret key -> 200
        res_ok = test_client.post(
            "/v1/chat/completions",
            json={"model": "NousResearch/Hermes-4.3-36B", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret_zaq_999"}
        )
        assert res_ok.status_code == 200

    # 4. Expired trial after 7 days -> 403
    past_timestamp = str(1000000000.0)  # Very old timestamp
    with patch.dict(os.environ, {"GATEWAY_LAUNCH_TIMESTAMP": past_timestamp}):
        res_expired = test_client.get("/v1/models")
        assert res_expired.status_code == 403
        assert "expired" in res_expired.json()["detail"].lower()


def test_simulated_flag_on_mock_endpoints():
    from server.local_api.server import app as local_app
    local_client = TestClient(local_app)

    res_local = local_client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": "Test prompt"}]
    })
    assert res_local.status_code == 200
    assert res_local.json().get("simulated") is True

    space_client = TestClient(space_gateway_app)
    res_space = space_client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": "Test prompt"}]
    })
    assert res_space.status_code == 200
    assert res_space.json().get("simulated") is True
