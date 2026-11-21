#!/usr/bin/env python3

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import struct
import hashlib
import json
import argparse
import platform
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

from core.model_registry.architecture_catalog import ArchitectureCatalog
from core.agent.agent_runtime import AgentRuntime
from core.agent.tools import ToolRegistry
from core.models.hermes_adapter import Hermes36BAdapter, DEFAULT_GENERATION_CONFIG
from core.remote.hf_remote_client import HFRemoteClient

_GGUF_FILE_TYPE_MAP = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    32: "BF16",
}


def cmd_doctor(args):
    print("==================================================")
    print("        QERLIO Runtime System Diagnostics        ")
    print("==================================================")
    print(f"OS Platform:          {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture:         {platform.machine()} ({platform.architecture()[0]})")
    print(f"Python Runtime:       {platform.python_version()} ({sys.executable})")

    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"CPU Cores:            {cpu_count_physical} physical, {cpu_count_logical} logical")

    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024 ** 3)
    avail_ram_gb = ram.available / (1024 ** 3)
    print(f"Physical Memory:      {total_ram_gb:.2f} GB total, {avail_ram_gb:.2f} GB available")

    if TORCH_AVAILABLE:
        print(f"PyTorch Version:      {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available:       {cuda_avail}")
    else:
        print("PyTorch:              Not installed in active environment")

    print(f"GGUF Python Support:  {'Installed' if GGUF_AVAILABLE else 'Not installed'}")
    print("==================================================")
    return 0


def _read_gguf_string(f) -> str:
    raw_len = f.read(8)
    if len(raw_len) < 8:
        raise EOFError("Unexpected EOF reading GGUF string length")
    str_len = struct.unpack("<Q", raw_len)[0]
    if str_len > 65536:
        raise ValueError(f"GGUF string length exceeds safety bound: {str_len}")
    data = f.read(str_len)
    if len(data) < str_len:
        raise EOFError("Unexpected EOF reading GGUF string payload")
    return data.decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int):
    if vtype == 0:    # UINT8
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == 1:  # INT8
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == 2:  # UINT16
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == 3:  # INT16
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == 4:  # UINT32
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5:  # INT32
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6:  # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == 7:  # BOOL
        return bool(struct.unpack("<B", f.read(1))[0])
    elif vtype == 8:  # STRING
        return _read_gguf_string(f)
    elif vtype == 9:  # ARRAY
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if count > 1000000:
            raise ValueError(f"GGUF array count exceeds safety bound: {count}")
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    elif vtype == 10: # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 11: # INT64
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == 12: # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def parse_gguf_raw(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size < 24:
        raise ValueError(f"File is too small to be a valid GGUF binary: {file_size} bytes")

    metadata = {}
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: expected b'GGUF', got {magic}")

        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        if file_size > 24 and metadata_kv_count <= 50000:
            for _ in range(metadata_kv_count):
                if f.tell() >= file_size:
                    break
                try:
                    key = _read_gguf_string(f)
                    vtype_bytes = f.read(4)
                    if len(vtype_bytes) < 4:
                        break
                    vtype = struct.unpack("<I", vtype_bytes)[0]
                    val = _read_gguf_value(f, vtype)
                    if not isinstance(val, list) or len(val) <= 32:
                        metadata[key] = val
                except Exception:
                    break

    arch = metadata.get("general.architecture")
    quant_type = metadata.get("general.quantization_type")
    if not quant_type and "general.file_type" in metadata:
        quant_type = _GGUF_FILE_TYPE_MAP.get(metadata["general.file_type"])

    context_length = None
    embedding_length = None
    block_count = None
    if arch:
        context_length = metadata.get(f"{arch}.context_length")
        embedding_length = metadata.get(f"{arch}.embedding_length")
        block_count = metadata.get(f"{arch}.block_count")

    return {
        "file_path": os.path.abspath(file_path),
        "file_size_bytes": file_size,
        "magic": "GGUF",
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
        "architecture": arch,
        "name": metadata.get("general.name"),
        "quantization_type": quant_type,
        "context_length": context_length,
        "embedding_length": embedding_length,
        "block_count": block_count,
        "metadata": metadata,
    }


def cmd_models_inspect(args):
    try:
        info = parse_gguf_raw(args.path)
        print(json.dumps(info, indent=2))
        return 0
    except Exception as e:
        print(f"Error inspecting model: {e}", file=sys.stderr)
        return 1


def cmd_models_verify(args):
    target_path = args.path
    expected_hash = args.expected_sha256.lower()

    if not os.path.isfile(target_path):
        print(f"Error: file not found: {target_path}", file=sys.stderr)
        return 1

    print(f"Verifying SHA-256 for: {target_path} ...")
    h = hashlib.sha256()
    with open(target_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)

    actual_hash = h.hexdigest().lower()
    print(f"Actual SHA-256:   {actual_hash}")
    print(f"Expected SHA-256: {expected_hash}")

    if actual_hash == expected_hash:
        print("[SUCCESS] Checksum matches.")
        return 0
    else:
        print("[FAILURE] Checksum mismatch!", file=sys.stderr)
        return 2


def cmd_benchmark(args):
    from tools.benchmark.benchmark_suite import run_benchmark
    rep = run_benchmark(
        model_path=args.path,
        output_tokens=args.tokens,
        context_limit=args.context,
        warmup_runs=args.warmup,
        measurement_runs=args.runs,
        profile_name=args.profile
    )
    print(json.dumps(rep, indent=2))
    return 0


def cmd_architectures_list(args):
    cat = ArchitectureCatalog()
    if args.status:
        items = cat.list_by_status(args.status)
    elif args.family:
        items = cat.list_by_family(args.family)
    elif args.search:
        items = cat.search(args.search)
    else:
        items = cat.list_all()

    print(f"{'ID':<25} | {'Status':<12} | {'Family':<22} | {'Developer':<20} | {'Name'}")
    print("-" * 105)
    for it in items:
        status_tag = it.get("support_status", "experimental")
        print(f"{it['id']:<25} | {status_tag:<12} | {it['family']:<22} | {it['developer']:<20} | {it['name']}")
    print("-" * 105)
    print(f"Displayed: {len(items)} / {cat.total_count} catalogued ({cat.tested_count} tested, {cat.experimental_count} experimental)")
    return 0


def cmd_architectures_count(args):
    cat = ArchitectureCatalog()
    print(f"Architectures: {cat.total_count} catalogued ({cat.tested_count} tested, {cat.experimental_count} experimental)")
    return 0


def cmd_architectures_info(args):
    cat = ArchitectureCatalog()
    info = cat.get(args.id)
    if not info:
        print(f"Architecture '{args.id}' not found in registry.", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_agent_execute(args):
    tools = ToolRegistry()
    raw_args = json.loads(args.args) if args.args else {}
    res = tools.execute(args.tool, raw_args)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res.get("success") else 1


def cmd_agent_run(args):
    runtime = AgentRuntime()
    print(f"Agent prompt: {args.prompt}")
    simulated_agent_thought = f'<think>\nEvaluating: {args.prompt}\n</think>\n<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "42 * 2"}}}}\n</tool_call>'
    step_res = runtime.run_step(simulated_agent_thought)
    print(json.dumps(step_res, indent=2, ensure_ascii=False))
    return 0


def cmd_hermes_chat(args):
    keep_cots = getattr(args, "keep_cots", False)
    adapter = Hermes36BAdapter(thinking=args.thinking, keep_cots=keep_cots)
    client = HFRemoteClient(base_url=args.remote_url, thinking=args.thinking, keep_cots=keep_cots)
    messages = [{"role": "user", "content": args.prompt}]

    if args.remote_url:
        print(f"Connecting to remote endpoint: {args.remote_url}")
        res = client.generate(
            model="NousResearch/Hermes-4.3-36B",
            messages=messages,
            thinking=args.thinking,
            keep_cots=keep_cots
        )
        if res.get("thinking"):
            print(f"[Thinking / Internal Monologue]:\n{res['thinking']}\n")
        print(f"[Answer]:\n{res.get('content') or res.get('raw_response')}")
        return 0

    prompt_formatted = adapter.format_chat_prompt(messages)
    print("Formatted Hermes 4.3 36B Prompt (Llama-3-Chat / Seed-OSS-36B-Base):")
    print("-" * 60)
    print(prompt_formatted)
    print("-" * 60)
    return 0


def cmd_hermes_test(args):
    default_adapter = Hermes36BAdapter()
    adapter = Hermes36BAdapter(thinking=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Safe math evaluator",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
            }
        }
    ]
    messages = [
        {"role": "user", "content": "Calculate the value of 15 * 8 + 40"}
    ]
    adapter.format_chat_prompt(messages, tools=tools)

    sample_tagged_text = (
        "<think>\nThe user wants to compute 15 * 8 + 40. "
        "First 15 * 8 = 120. Then 120 + 40 = 160. "
        "I will call the calculator tool to verify.</think>\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "15 * 8 + 40"}}\n</tool_call>'
    )
    parsed = adapter.parse_generation(sample_tagged_text)

    print("Hermes 4.3 36B Offline Template & Tag Parser Self-Test (No Model Inference):")
    print(f"- Base Model: {adapter.base_model} ({adapter.architecture_class})")
    print(f"- Context Length: {adapter.context_length}")
    print(f"- Default Thinking Mode: {default_adapter.thinking}")
    print(f"- BOS/EOS: {adapter.bos_token} / {adapter.eos_token}")
    print(f"- Has Thinking (sample with thinking=True): {parsed['has_thinking']}")
    print(f"- Extracted Thinking: {parsed['thinking']}")
    print(f"- Has Tool Calls: {parsed['has_tool_calls']}")
    print(f"- Extracted Tool Call: {parsed['tool_calls']}")
    print(f"- OpenAI tool_calls Count: {len(parsed['openai_tool_calls'])}")
    print(f"- Anthropic tool_use Count: {len(parsed['anthropic_tool_use'])}")

    if (
        default_adapter.thinking is False
        and adapter.context_length == 524288
        and parsed["has_thinking"]
        and parsed["has_tool_calls"]
        and len(parsed["openai_tool_calls"]) == 1
        and len(parsed["anthropic_tool_use"]) == 1
    ):
        print("[TEMPLATE_PARSER_OK] Offline template formatting and tag parsing verified (no model weights executed).")
        return 0
    return 1


def cmd_hermes_download(args):
    print("==================================================")
    print(" QERLIO: Downloading Hermes 4.3 36B Official GGUF")
    print("==================================================")
    repo_id = "NousResearch/Hermes-4.3-36B-GGUF"
    quant_file = f"hermes-4_3_36b-{args.quant}.gguf"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {repo_id}")
    print(f"Quantization: {args.quant} ({quant_file})")
    print(f"Target Directory: {output_dir.resolve()}")

    try:
        from huggingface_hub import hf_hub_download
        print("Initiating download from Hugging Face...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=quant_file,
            local_dir=str(output_dir),
            token=os.environ.get("HF_TOKEN")
        )
        print(f"[SUCCESS] Download completed: {path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to download GGUF from {repo_id}: {e}")
        print("Note: You can also download directly via CLI:")
        print(f"  hf download {repo_id} {quant_file} --local-dir {output_dir}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="QERLIO Runtime CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_doctor = subparsers.add_parser("doctor", help="Inspect environment")
    p_doctor.set_defaults(func=cmd_doctor)

    p_inspect = subparsers.add_parser("inspect", help="Inspect GGUF model")
    p_inspect.add_argument("path", help="Path to GGUF model file")
    p_inspect.set_defaults(func=cmd_models_inspect)

    p_verify = subparsers.add_parser("verify", help="Verify GGUF hash")
    p_verify.add_argument("path", help="Path to GGUF model file")
    p_verify.add_argument("expected_sha256", help="Expected SHA-256")
    p_verify.set_defaults(func=cmd_models_verify)

    p_bench = subparsers.add_parser("benchmark", help="Benchmark model")
    p_bench.add_argument("path", help="Path to GGUF model")
    p_bench.add_argument("--tokens", type=int, default=64)
    p_bench.add_argument("--context", type=int, default=2048)
    p_bench.add_argument("--warmup", type=int, default=1)
    p_bench.add_argument("--runs", type=int, default=3)
    p_bench.add_argument("--profile", default="mobile-balanced")
    p_bench.set_defaults(func=cmd_benchmark)

    p_server = subparsers.add_parser("server", help="Start local API server")
    p_server.add_argument("--host", default="127.0.0.1")
    p_server.add_argument("--port", type=int, default=8787)
    p_server.set_defaults(func=lambda a: __import__("server.local_api.server", fromlist=["start_server"]).start_server(a.host, a.port))

    p_arch = subparsers.add_parser("architectures", help="Explore supported model architectures")
    arch_sub = p_arch.add_subparsers(dest="arch_subcommand", required=True)

    p_arch_list = arch_sub.add_parser("list", help="List catalogued architectures")
    p_arch_list.add_argument("--family", help="Filter by family")
    p_arch_list.add_argument("--search", help="Search keyword")
    p_arch_list.add_argument("--status", choices=["tested", "experimental"], help="Filter by support status")
    p_arch_list.set_defaults(func=cmd_architectures_list)

    p_arch_count = arch_sub.add_parser("count", help="Show architecture counts by support tier")
    p_arch_count.set_defaults(func=cmd_architectures_count)

    p_arch_info = arch_sub.add_parser("info", help="Get architecture details")
    p_arch_info.add_argument("id", help="Architecture ID")
    p_arch_info.set_defaults(func=cmd_architectures_info)

    p_agent = subparsers.add_parser("agent", help="Agent Mode operations")
    agent_sub = p_agent.add_subparsers(dest="agent_subcommand", required=True)

    p_agent_exec = agent_sub.add_parser("execute", help="Execute an agent sandbox tool")
    p_agent_exec.add_argument("--tool", required=True, help="Tool name")
    p_agent_exec.add_argument("--args", default="{}", help="Tool arguments JSON string")
    p_agent_exec.set_defaults(func=cmd_agent_execute)

    p_agent_run = agent_sub.add_parser("run", help="Run multi-turn agent step")
    p_agent_run.add_argument("--prompt", required=True, help="Prompt text")
    p_agent_run.set_defaults(func=cmd_agent_run)

    p_hermes = subparsers.add_parser("hermes", help="Hermes 4.3 36B template, parser & GGUF download tools")
    hermes_sub = p_hermes.add_subparsers(dest="hermes_subcommand", required=True)

    p_h_chat = hermes_sub.add_parser("chat", help="Format prompt or query a remote endpoint for Hermes 4.3 36B")
    p_h_chat.add_argument("--prompt", required=True, help="User prompt")
    p_h_chat.add_argument("--thinking", action="store_true", default=False, help="Enable hybrid reasoning (<think>)")
    p_h_chat.add_argument("--keep-cots", action="store_true", default=False, help="Keep <think> blocks in history and parsed output")
    p_h_chat.add_argument("--remote-url", default=None, help="Optional remote OpenAI-compatible API URL")
    p_h_chat.set_defaults(func=cmd_hermes_chat)

    p_h_test = hermes_sub.add_parser("test", help="Run offline chat-template and tag-parser self-test (no model inference)")
    p_h_test.set_defaults(func=cmd_hermes_test)

    p_h_download = hermes_sub.add_parser("download", help="Download official Hermes 4.3 36B GGUF weights")
    p_h_download.add_argument("--quant", default="Q4_K_M", choices=["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"], help="Quantization scheme")
    p_h_download.add_argument("--output-dir", default="models", help="Destination folder")
    p_h_download.set_defaults(func=cmd_hermes_download)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
