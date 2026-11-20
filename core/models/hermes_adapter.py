import json
import re
from typing import Any, Dict, List, Optional, Tuple

THINKING_SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider "
    "the problem and deliberate with yourself via systematic reasoning processes to help come to a "
    "correct solution prior to answering. You should enclose your thoughts and internal monologue "
    "inside <think> </think> tags, and then provide your solution or response to the problem."
)

STANDARD_SYSTEM_PROMPT = "You are Hermes, created by Nous Research."

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "add_generation_prompt": True
}

class Hermes36BAdapter:
    def __init__(self, thinking: bool = True, keep_cots: bool = False):
        self.thinking = thinking
        self.keep_cots = keep_cots
        self.base_model = "ByteDance Seed 36B"
        self.bos_token = "<seed:bos>"
        self.eos_token = "<|eot_id|>"

    def format_chat_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True
    ) -> str:
        prompt_parts: List[str] = [self.bos_token]
        system_content = THINKING_SYSTEM_PROMPT if self.thinking else STANDARD_SYSTEM_PROMPT

        has_custom_system = len(messages) > 0 and messages[0].get("role") == "system"
        if has_custom_system:
            system_content = messages[0]["content"]

        if tools and len(tools) > 0:
            prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n")
            prompt_parts.append(system_content)
            prompt_parts.append(
                "\n\n# Tools\n\nYou are a function calling AI model. You may call one or more "
                "functions to assist with the user query.\n\nYou are provided with function "
                "signatures within <tools></tools> XML tags:\n<tools>"
            )
            for t in tools:
                prompt_parts.append(f"\n{json.dumps(t, ensure_ascii=False)}")
            prompt_parts.append(
                "\n</tools>\n\nFor each function call, return a json object with function name and "
                "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
                '{"name": "<function-name>", "arguments": <args-json-object>}\n</tool_call><|eot_id|>'
            )
        else:
            prompt_parts.append(
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>"
            )

        start_idx = 1 if has_custom_system else 0
        for i in range(start_idx, len(messages)):
            m = messages[i]
            role = m.get("role", "user")
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls")

            if role in ("user", "system"):
                prompt_parts.append(
                    f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                )
            elif role == "assistant":
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n")
                if content:
                    prompt_parts.append(f"\n{content}")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", tc)
                        name = fn.get("name", "")
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        args_str = json.dumps(args, ensure_ascii=False)
                        prompt_parts.append(
                            f'\n<tool_call>\n{{"name": "{name}", "arguments": {args_str}}}\n</tool_call>'
                        )
                prompt_parts.append("<|eot_id|>")
            elif role == "tool":
                prev_is_tool = i > 0 and messages[i - 1].get("role") == "tool"
                if not prev_is_tool:
                    prompt_parts.append("<|start_header_id|>user<|end_header_id|>\n")
                prompt_parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
                next_is_tool = i + 1 < len(messages) and messages[i + 1].get("role") == "tool"
                if not next_is_tool:
                    prompt_parts.append("<|eot_id|>")

        if add_generation_prompt:
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(prompt_parts)

    def parse_generation(self, raw_output: str) -> Dict[str, Any]:
        thinking_text: Optional[str] = None
        final_answer = raw_output

        think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            final_answer = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

        tool_calls: List[Dict[str, Any]] = []
        call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for m in re.finditer(call_pattern, final_answer, re.DOTALL):
            try:
                payload = json.loads(m.group(1))
                tool_calls.append({
                    "name": payload.get("name"),
                    "arguments": payload.get("arguments", {})
                })
            except Exception:
                continue

        cleaned_text = re.sub(call_pattern, "", final_answer, flags=re.DOTALL).strip()

        return {
            "thinking": thinking_text,
            "has_thinking": thinking_text is not None,
            "tool_calls": tool_calls,
            "has_tool_calls": len(tool_calls) > 0,
            "final_content": cleaned_text if cleaned_text else (None if tool_calls else final_answer),
            "raw_output": raw_output
        }
