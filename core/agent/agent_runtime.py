import json
import re
from typing import Any, Callable, Dict, List, Optional
from core.agent.tools import ToolRegistry

class AgentRuntime:
    def __init__(self, tool_registry: Optional[ToolRegistry] = None, max_iterations: int = 5):
        self.tools = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []

        qwen_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(qwen_pattern, text, re.DOTALL):
            try:
                payload = json.loads(match.group(1))
                calls.append({
                    "name": payload.get("name"),
                    "arguments": payload.get("arguments", {})
                })
            except Exception:
                continue

        zqn_pattern = r'<call\s+name="([^"]+)"\s+args=(\'(\{.*?\})\'|"(\{.*?\})")\s*/>'
        for match in re.finditer(zqn_pattern, text, re.DOTALL):
            name = match.group(1)
            raw_args = match.group(3) or match.group(4) or "{}"
            try:
                args = json.loads(raw_args)
                calls.append({"name": name, "arguments": args})
            except Exception:
                continue

        zqn_alt_pattern = r'<call\s+name="([^"]+)"\s+args="([^"]+)"\s*/>'
        for match in re.finditer(zqn_alt_pattern, text):
            name = match.group(1)
            raw_args = match.group(2)
            try:
                args = json.loads(raw_args.replace("&quot;", '"'))
                if not any(c["name"] == name and c["arguments"] == args for c in calls):
                    calls.append({"name": name, "arguments": args})
            except Exception:
                continue

        return calls

    def format_tool_response(self, tool_name: str, result: Dict[str, Any], model_flavor: str = "zaqeris") -> str:
        payload = json.dumps(result, ensure_ascii=False)
        return f"<tool_response>\n{payload}\n</tool_response>"

    def run_step(self, generated_text: str) -> Dict[str, Any]:
        calls = self.parse_tool_calls(generated_text)
        if not calls:
            return {
                "has_tool_calls": False,
                "tool_calls": [],
                "results": [],
                "next_prompt_suffix": ""
            }

        results = []
        for call in calls:
            res = self.tools.execute(call["name"], call["arguments"])
            results.append({
                "call": call,
                "execution": res
            })

        response_text = "\n".join(
            self.format_tool_response(r["call"]["name"], r["execution"])
            for r in results
        )

        return {
            "has_tool_calls": True,
            "tool_calls": calls,
            "results": results,
            "next_prompt_suffix": response_text
        }
