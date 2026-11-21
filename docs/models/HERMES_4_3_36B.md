# NousResearch/Hermes-4.3-36B — Native Engine, Template & GGUF Reference

> **Support Tier in Registry:** `tested` (`seed_36b`)  
> **Scope in QERLIO v0.3.0:** From-scratch native GGUF v3 inference engine (`core/inference/seed_oss_model.py`, `gguf_loader.py`, `dequant.py`, `tokenizer.py`, `kv_cache.py`, `sampler.py`) using only the Python standard library and NumPy, integrated with `Hermes36BAdapter` (`Llama-3-Chat` / `<seed:bos>`, `<think>` / `<tool_call>` parsing) and the local API gateway (`/v1/models/load`, `/v1/chat/completions`, `/v1/messages`).

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

### A. Run Local Inference with the Native Engine
Load a local GGUF checkpoint directly via CLI or via the local API server (`POST /v1/models/load`):
```bash
python tools/qerlio_cli.py hermes chat --model-path models/hermes-4_3_36b-Q4_K_M.gguf --prompt "Explain dialectic synthesis." --thinking
```

### B. Format a Chat Prompt Locally
By default, `thinking` is disabled (`False`):
```bash
python tools/qerlio_cli.py hermes chat --prompt "Explain dialectic synthesis."
```
To enable the thinking system prompt (`thinking=True`) and optionally keep chain-of-thought blocks (`keep_cots=True`):
```bash
python tools/qerlio_cli.py hermes chat --prompt "Explain dialectic synthesis." --thinking --keep-cots
```

### C. Run Offline Template & Tag-Parser Self-Test
This command tests prompt formatting, `<think>` and `<tool_call>` extraction, and OpenAI/Anthropic format conversion on an offline sample string (it does **not** load or execute model weights):
```bash
python tools/qerlio_cli.py hermes test
```

### D. Download Official GGUF Weights
Download quantized GGUF files from [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF):
```bash
python tools/qerlio_cli.py hermes download --quant Q4_K_M --output-dir models
```

### E. Run Native Inference & Adapter Verification Tests
```bash
python -m pytest tests/test_hermes_36b.py tests/test_t1_t2_gguf_and_dequant.py tests/test_t3_tokenizer.py tests/test_t4_to_t10_engine.py -v
```

---

## 5. Track B SHA-256 & Live 64-Layer Verification

Official GGUF artifact verified locally (gitignored under `models/`):

- **Artifact:** `NousResearch/Hermes-4.3-36B-GGUF` (`hermes-4_3_36b-Q4_K_M.gguf`)
- **Size:** `21,762,145,216` bytes
- **SHA-256:** `17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08`
- **GGUF Header & Tensors (`SeedOssEngine(..., preload_weights=False, header_only=False)`):**
  - `architecture=seed_oss`, `name='Hermes 4_3_36b'`
  - `kv_count=27`, `tensor_count=771`, `layers=64`, `hidden=5120`, `ffn=27648`, `heads=80`, `kv_heads=8`, `head_dim=128`, `vocab=155136`
  - Engine `mmap` initialization: `0.67s`
- **Tokenizer & Adapter Check:** `Hermes36BAdapter(thinking=False, keep_cots=False)` 33-token chat prompt exact encode/decode round-trip passed.
- **Full 64-Layer Forward Pass (`'The capital of France is'` -> `[0, 1063, 9203, 437, 10341, 506]`):**
  - Checkpoints every 8 layers: layer 8 (`60.6s`), 16 (`110.6s`), 24 (`161.3s`), 32 (`203.8s`), 40 (`253.2s`), 48 (`300.2s`), 56 (`358.2s`), 64 (`408.8s`) — all `finite=True`
  - Total elapsed including LM head: `419.43s` (`logits shape=(155136,)`, `dtype=float32`, `finite=True`)
  - Top-3 predicted next tokens: `#1 ' Paris'` (`id=10805`, `logit=16.4356`), `#2 ' a'` (`id=387`, `logit=13.9003`), `#3 ' the'` (`id=405`, `logit=13.0227`)
