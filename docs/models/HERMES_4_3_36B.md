# NousResearch/Hermes-4.3-36B — Template, Parser & GGUF Tooling Reference

> **Support Tier in Registry:** `experimental` (`seed_36b`)  
> **Scope in QERLIO v0.2.2:** Chat template formatter (`Llama-3-Chat` / `<seed:bos>`), `<think>` and `<tool_call>` tag parser (with OpenAI & Anthropic conversion), GGUF metadata inspector, and GGUF downloader. QERLIO does **not** yet execute `SeedOssForCausalLM` model weights locally.

---

## 1. Official Model Metadata

`NousResearch/Hermes-4.3-36B` (model card title: **Hermes 4.3 - Seed 36B**) is an open-weight hybrid-mode reasoning and tool-use model published by **Nous Research**.

* **Base Model:** **ByteDance Seed-OSS-36B-Base** (not Meta Llama).
* **Transformers Architecture Class:** `SeedOssForCausalLM` (`model_type: seed_oss`).
* **Parameters:** 36B.
* **Context Length:** **524,288 tokens** (`max_position_embeddings: 524288`).
* **Chat Template:** `Llama-3-Chat` format with Seed BOS token:
  - BOS Token: `<seed:bos>`
  - Role Header: `<|start_header_id|>{role}<|end_header_id|>\n\n`
  - Turn Terminator: `<|eot_id|>`
* **Official Repositories:**
  - BF16 Safetensors: [`NousResearch/Hermes-4.3-36B`](https://huggingface.co/NousResearch/Hermes-4.3-36B)
  - GGUF Checkpoints: [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF) (`hermes-4_3_36b-Q3_K_M.gguf`, `hermes-4_3_36b-Q4_K_M.gguf`, `hermes-4_3_36b-Q5_K_M.gguf`, `hermes-4_3_36b-Q6_K.gguf`, `hermes-4_3_36b-Q8_0.gguf`, `hermes-4_3_36b.gguf`)

---

## 2. Recommended Sampling & Parser Configuration (From Model Card)

* **Temperature:** `0.6`
* **Top-P:** `0.95`
* **Top-K:** `20`
* **add_generation_prompt:** `True`
* **vLLM Tool Parser:** `--enable-auto-tool-choice --tool-call-parser hermes`
* **SGLang Tool Parser:** `--tool-call-parser qwen25`

---

## 3. What QERLIO Currently Implements (`core/models/hermes_adapter.py`)

### A. Prompt Formatting & `thinking` / `keep_cots` Flags
* **`thinking=False` (Default):** Uses the standard system prompt (`"You are Hermes, created by Nous Research."`) unless a custom system message is provided.
* **`thinking=True`:** Prepends the official deep-thinking system prompt instructing the model to enclose its internal deliberation inside `<think> </think>` tags prior to answering.
* **`keep_cots=False` (Default):** Strips `<think>...</think>` blocks from historical `assistant` turns when formatting multi-turn prompts and removes `<think>...</think>` from `parsed["final_content"]`.
* **`keep_cots=True`:** Preserves `<think>...</think>` blocks (and `reasoning_content` fields) in historical `assistant` turns and retains the `<think>` block in `parsed["final_content"]`.

### B. Tag Parsing & Edge-Case Handling (`parse_generation`)
* **Closed & Unclosed `<think>` Blocks:** Extracts the reasoning text into `parsed["thinking"]`. If `<think>` is opened without a closing `</think>`, sets `parsed["unclosed_thinking"] = True` and strips the unclosed block from `final_content` (when `keep_cots=False`).
* **Structured `<tool_call>` Parsing:** Parses valid `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` blocks into `parsed["tool_calls"]`. Unclosed `<tool_call>` tags or invalid JSON payloads are recorded in `parsed["malformed_tool_calls"]` without raising unhandled exceptions.
* **Protocol Conversions:**
  - `parsed["openai_tool_calls"]` / `Hermes36BAdapter.to_openai_tool_calls(...)`: formats calls as OpenAI `[{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json-str>"}}]`.
  - `parsed["anthropic_tool_use"]` / `Hermes36BAdapter.to_anthropic_tool_use(...)`: formats calls as Anthropic `[{"type": "tool_use", "id": ..., "name": ..., "input": {...}}]`.

---

## 4. CLI Usage (Formatting, Parser Self-Test & GGUF Download)

### A. Format a Chat Prompt Locally
By default, `thinking` is disabled (`False`):
```bash
python tools/qerlio_cli.py hermes chat --prompt "Explain dialectic synthesis."
```
To enable the thinking system prompt (`thinking=True`) and optionally keep chain-of-thought blocks (`keep_cots=True`):
```bash
python tools/qerlio_cli.py hermes chat --prompt "Explain dialectic synthesis." --thinking --keep-cots
```

### B. Run Offline Template & Tag-Parser Self-Test
This command tests prompt formatting, `<think>` and `<tool_call>` extraction, and OpenAI/Anthropic format conversion on an offline sample string (it does **not** load or execute model weights):
```bash
python tools/qerlio_cli.py hermes test
```

### C. Download Official GGUF Weights
Download quantized GGUF files from [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF):
```bash
python tools/qerlio_cli.py hermes download --quant Q4_K_M --output-dir models
```

### D. Run Unit Tests for the Adapter & Parser
```bash
python -m pytest tests/test_hermes_36b.py -v
```
