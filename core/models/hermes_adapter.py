import json
import re
from typing import Any, Dict, List, Optional

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
    """
    Chat template formatter and output tag parser for NousResearch/Hermes-4.3-36B
    (built on ByteDance Seed-OSS-36B-Base / SeedOssForCausalLM).
    Note: This class formats prompts and parses <think> and <tool_call> tags;
    it does not load or execute model weights.
    """

    def __init__(self, thinking: bool = False, keep_cots: bool = False):
        self.thinking = thinking
        self.keep_cots = keep_cots
        self.base_model = "ByteDance Seed-OSS-36B-Base"
        self.architecture_class = "SeedOssForCausalLM"
        self.model_type = "seed_oss"
        self.context_length = 524288
        self.hf_repo = "NousResearch/Hermes-4.3-36B"
        self.gguf_repo = "NousResearch/Hermes-4.3-36B-GGUF"
        self.bos_token = "<seed:bos>"
        self.eos_token = "<|eot_id|>"

    def _clean_assistant_history_content(self, content: str, reasoning_content: Optional[str] = None) -> str:
        if self.keep_cots:
            if reasoning_content and "<think>" not in content:
                if content:
                    return f"<think>\n{reasoning_content.strip()}\n</think>\n{content}"
                return f"<think>\n{reasoning_content.strip()}\n</think>"
            return content
        cleaned = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepares a list of chat messages by injecting the appropriate Hermes system prompt
        (thinking vs standard) when no custom system message is present, and handling
        historical assistant chain-of-thought blocks according to keep_cots.
        """
        prepared: List[Dict[str, Any]] = []
        has_custom_system = len(messages) > 0 and messages[0].get("role") == "system"
        if not has_custom_system:
            sys_prompt = THINKING_SYSTEM_PROMPT if self.thinking else STANDARD_SYSTEM_PROMPT
            prepared.append({"role": "system", "content": sys_prompt})

        for m in messages:
            msg_copy = dict(m)
            if msg_copy.get("role") == "assistant":
                raw_content = msg_copy.get("content") or ""
                reasoning_content = msg_copy.get("reasoning_content")
                msg_copy["content"] = self._clean_assistant_history_content(raw_content, reasoning_content)
            prepared.append(msg_copy)
        return prepared

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
                cleaned_content = self._clean_assistant_history_content(
                    content, m.get("reasoning_content")
                )
                prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
                if cleaned_content:
                    prompt_parts.append(f"\n{cleaned_content}")
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

    @staticmethod
    def to_openai_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts parsed Hermes tool calls into OpenAI chat completion tool_calls format.
        """
        openai_calls: List[Dict[str, Any]] = []
        for idx, tc in enumerate(tool_calls):
            args = tc.get("arguments", {})
            args_json = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            openai_calls.append({
                "id": tc.get("id") or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": args_json
                }
            })
        return openai_calls

    @staticmethod
    def to_anthropic_tool_use(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts parsed Hermes tool calls into Anthropic Messages API tool_use content blocks.
        """
        anthropic_blocks: List[Dict[str, Any]] = []
        for idx, tc in enumerate(tool_calls):
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            anthropic_blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{idx}",
                "name": tc["name"],
                "input": args
            })
        return anthropic_blocks

    def parse_generation(self, raw_output: str) -> Dict[str, Any]:
        thinking_text: Optional[str] = None
        unclosed_thinking = False
        working_text = raw_output

        think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            working_text = re.sub(r"<think>.*?</think>", "", raw_output, count=1, flags=re.DOTALL).strip()
        elif "<think>" in raw_output:
            unclosed_match = re.search(r"<think>(.*)$", raw_output, re.DOTALL)
            if unclosed_match:
                thinking_text = unclosed_match.group(1).strip()
                unclosed_thinking = True
                working_text = re.sub(r"<think>.*$", "", raw_output, flags=re.DOTALL).strip()

        tool_calls: List[Dict[str, Any]] = []
        malformed_tool_calls: List[str] = []

        closed_call_pattern = r"<tool_call>(.*?)</tool_call>"
        for m in re.finditer(closed_call_pattern, working_text, re.DOTALL):
            inner = m.group(1).strip()
            try:
                payload = json.loads(inner)
                if not isinstance(payload, dict):
                    malformed_tool_calls.append(inner)
                    continue
                name = payload.get("name")
                if not isinstance(name, str) or not name.strip():
                    malformed_tool_calls.append(inner)
                    continue
                arguments = payload.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        malformed_tool_calls.append(inner)
                        continue
                if not isinstance(arguments, dict):
                    malformed_tool_calls.append(inner)
                    continue
                tool_calls.append({
                    "name": name.strip(),
                    "arguments": arguments
                })
            except Exception:
                malformed_tool_calls.append(inner)

        cleaned_text = re.sub(closed_call_pattern, "", working_text, flags=re.DOTALL).strip()

        if "<tool_call>" in cleaned_text:
            unclosed_tc = re.search(r"<tool_call>(.*)$", cleaned_text, re.DOTALL)
            if unclosed_tc:
                malformed_tool_calls.append(unclosed_tc.group(1).strip())
                cleaned_text = re.sub(r"<tool_call>.*$", "", cleaned_text, flags=re.DOTALL).strip()

        if self.keep_cots and thinking_text is not None:
            cot_block = f"<think>\n{thinking_text}\n</think>"
            final_content: Optional[str] = f"{cot_block}\n{cleaned_text}".strip() if cleaned_text else cot_block
        else:
            if cleaned_text:
                final_content = cleaned_text
            elif tool_calls or thinking_text is not None or malformed_tool_calls:
                final_content = None
            else:
                final_content = working_text

        openai_tool_calls = self.to_openai_tool_calls(tool_calls)
        anthropic_tool_use = self.to_anthropic_tool_use(tool_calls)

        return {
            "thinking": thinking_text,
            "has_thinking": thinking_text is not None,
            "unclosed_thinking": unclosed_thinking,
            "tool_calls": tool_calls,
            "has_tool_calls": len(tool_calls) > 0,
            "malformed_tool_calls": malformed_tool_calls,
            "openai_tool_calls": openai_tool_calls,
            "anthropic_tool_use": anthropic_tool_use,
            "final_content": final_content,
            "raw_output": raw_output
        }
