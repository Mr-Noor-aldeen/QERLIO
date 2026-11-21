import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, Generator, List, Optional
from core.models.hermes_adapter import Hermes36BAdapter, DEFAULT_GENERATION_CONFIG


class HFRemoteClient:
    """
    HTTP client for OpenAI-compatible remote chat completion endpoints.

    Note on chat templating:
    - When calling `/chat/completions` (`generate` and `stream_generate`), the remote
      server applies its own tokenizer chat template to the `messages` array. For
      Hermes/Seed models, this client uses `Hermes36BAdapter.prepare_messages` to
      inject the appropriate system prompt (`thinking=True` vs `thinking=False`) and
      filter historical chain-of-thought blocks (`keep_cots`), and uses
      `Hermes36BAdapter.parse_generation` to parse `<think>` and `<tool_call>` tags
      from the returned content.
    - To apply the raw `<seed:bos>` / `Llama-3-Chat` string template locally, call
      `format_prompt(...)`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        thinking: bool = False,
        keep_cots: bool = False
    ):
        self.base_url = (base_url or os.environ.get("QERLIO_REMOTE_URL") or "https://router.huggingface.co/v1").rstrip("/")
        self.token = token or os.environ.get("HF_TOKEN") or ""
        self.hermes_adapter = Hermes36BAdapter(thinking=thinking, keep_cots=keep_cots)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "QERLIO-Runtime/0.3.0"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @staticmethod
    def _is_hermes_or_seed(model: str) -> bool:
        lowered = model.lower()
        return "hermes" in lowered or "seed" in lowered

    def format_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[bool] = None,
        keep_cots: Optional[bool] = None
    ) -> str:
        adapter = Hermes36BAdapter(
            thinking=self.hermes_adapter.thinking if thinking is None else thinking,
            keep_cots=self.hermes_adapter.keep_cots if keep_cots is None else keep_cots
        )
        return adapter.format_chat_prompt(messages, tools=tools)

    def ping(self) -> Dict[str, Any]:
        endpoints = [f"{self.base_url}/models", f"{self.base_url}/healthz", self.base_url]
        for ep in endpoints:
            try:
                req = urllib.request.Request(ep, headers=self._headers(), method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status in (200, 204):
                        return {"connected": True, "endpoint": ep, "status_code": resp.status}
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    return {"connected": False, "endpoint": ep, "status_code": e.code, "error": str(e)}
            except Exception:
                continue
        return {"connected": False, "endpoint": self.base_url, "error": "Endpoint unreachable"}

    def generate(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = DEFAULT_GENERATION_CONFIG["temperature"],
        top_p: float = DEFAULT_GENERATION_CONFIG["top_p"],
        max_tokens: int = 512,
        thinking: bool = False,
        keep_cots: bool = False
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
        adapter = Hermes36BAdapter(thinking=thinking, keep_cots=keep_cots)
        outgoing_messages = (
            adapter.prepare_messages(messages)
            if self._is_hermes_or_seed(model)
            else messages
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": outgoing_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                choice = res_json.get("choices", [{}])[0]
                message = choice.get("message", {})
                raw_content = message.get("content") or ""

                if self._is_hermes_or_seed(model):
                    parsed = adapter.parse_generation(raw_content)
                    return {
                        "model": model,
                        "raw_response": res_json,
                        "thinking": parsed["thinking"],
                        "unclosed_thinking": parsed["unclosed_thinking"],
                        "tool_calls": parsed["tool_calls"] or message.get("tool_calls", []),
                        "openai_tool_calls": parsed["openai_tool_calls"] or message.get("tool_calls", []),
                        "anthropic_tool_use": parsed["anthropic_tool_use"],
                        "content": parsed["final_content"]
                    }

                return {
                    "model": model,
                    "raw_response": res_json,
                    "thinking": None,
                    "unclosed_thinking": False,
                    "tool_calls": message.get("tool_calls", []),
                    "openai_tool_calls": message.get("tool_calls", []),
                    "anthropic_tool_use": [],
                    "content": raw_content
                }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Remote inference failed HTTP {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Remote connection error: {e}") from e

    def stream_generate(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = DEFAULT_GENERATION_CONFIG["temperature"],
        top_p: float = DEFAULT_GENERATION_CONFIG["top_p"],
        max_tokens: int = 512,
        thinking: bool = False,
        keep_cots: bool = False
    ) -> Generator[str, None, None]:
        endpoint = f"{self.base_url}/chat/completions"
        adapter = Hermes36BAdapter(thinking=thinking, keep_cots=keep_cots)
        outgoing_messages = (
            adapter.prepare_messages(messages)
            if self._is_hermes_or_seed(model)
            else messages
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": outgoing_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=self._headers(), method="POST")

        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str.startswith(":"):
                    continue
                if line_str == "data: [DONE]":
                    break
                if line_str.startswith("data: "):
                    chunk = json.loads(line_str[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        yield content
