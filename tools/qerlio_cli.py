#!/usr/bin/env python3

import sys
import os

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
        if cuda_avail:
            print(f"GPU Device Count:     {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                dev_name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  GPU [{i}]:            {dev_name} ({mem_gb:.2f} GB VRAM)")
    else:
        print("PyTorch:              Not installed in active environment")
        
    print(f"GGUF Python Support:  {'Installed' if GGUF_AVAILABLE else 'Not installed'}")
    print("==================================================")
    return 0

def parse_gguf_raw(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    file_size = os.path.getsize(file_path)
    if file_size < 24:
        raise ValueError(f"File is too small to be a valid GGUF binary: {file_size} bytes")
        
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: expected b'GGUF', got {magic}")
            
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
            
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
        
    return {
        "file_path": os.path.abspath(file_path),
        "file_size_bytes": file_size,
        "magic": "GGUF",
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count
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
    adapter = Hermes36BAdapter(thinking=args.thinking)
    client = HFRemoteClient(base_url=args.remote_url)
    messages = [{"role": "user", "content": args.prompt}]
    
    if args.remote_url:
        print(f"Connecting to remote endpoint: {args.remote_url}")
        res = client.generate(
            model="NousResearch/Hermes-4.3-36B",
            messages=messages,
            thinking=args.thinking
        )
        if res.get("thinking"):
            print(f"[Thinking / Internal Monologue]:\n{res['thinking']}\n")
        print(f"[Answer]:\n{res.get('content') or res.get('raw_response')}")
        return 0

    prompt_formatted = adapter.format_chat_prompt(messages)
    print("Formatted Hermes 4.3 36B Prompt (Llama-3-Chat / Seed 36B):")
    print("-" * 60)
    print(prompt_formatted)
    print("-" * 60)
    return 0

def cmd_hermes_test(args):
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
    formatted = adapter.format_chat_prompt(messages, tools=tools)
    
    simulated_output = (
        "<think>\nThe user wants to compute 15 * 8 + 40. "
        "First 15 * 8 = 120. Then 120 + 40 = 160. "
        "I will call the calculator tool to verify.</think>\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "15 * 8 + 40"}}\n</tool_call>'
    )
    parsed = adapter.parse_generation(simulated_output)
    
    print("Hermes 4.3 36B End-to-End Conformance Test:")
    print(f"- Base Model: {adapter.base_model}")
    print(f"- BOS/EOS: {adapter.bos_token} / {adapter.eos_token}")
    print(f"- Has Thinking: {parsed['has_thinking']}")
    print(f"- Extracted Thinking: {parsed['thinking']}")
    print(f"- Has Tool Calls: {parsed['has_tool_calls']}")
    print(f"- Extracted Tool Call: {parsed['tool_calls']}")
    
    if parsed["has_thinking"] and parsed["has_tool_calls"] and len(parsed["tool_calls"]) == 1:
        print("[SUCCESS] Hermes 4.3 36B reasoning & tool extraction passed.")
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
        print(f"Initiating download from Hugging Face...")
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

def cmd_hermes_run(args):
    print("==================================================")
    print(" QERLIO: Running Hermes 4.3 36B Local GGUF Engine")
    print("==================================================")
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model file not found at: {model_path}")
        print("Download first via: qerlio hermes download --quant Q4_K_M")
        return 1

    adapter = Hermes36BAdapter(thinking=args.thinking)
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = adapter.format_chat_prompt(messages)

    try:
        from sdk.python.qerlio import QerlioEngine, GenerationConfig
        engine = QerlioEngine()
        model = engine.load_model(str(model_path), context_length=args.context_length)
        session = model.create_session()
        
        print(f"Model: {model_path.name}")
        print(f"Base: ByteDance Seed 36B / Llama-3-Chat")
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        
        config = GenerationConfig(max_output_tokens=args.max_tokens, temperature=0.6, top_p=0.95)
        raw_output = []
        for chunk in session.generate_stream(formatted_prompt, config=config):
            raw_output.append(chunk)
            print(chunk, end="", flush=True)
        print("\n" + "-" * 60)
        
        full_text = "".join(raw_output)
        parsed = adapter.parse_generation(full_text)
        if parsed.get("thinking"):
            print(f"[Extracted Thinking]:\n{parsed['thinking']}")
        if parsed.get("tool_calls"):
            print(f"[Extracted Tool Calls]:\n{json.dumps(parsed['tool_calls'], indent=2)}")
        return 0
    except Exception as e:
        print(f"[ERROR] Engine execution failure: {e}")
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
    p_bench.add_argument("path", help="Path to GGUF model file")
    p_bench.add_argument("--tokens", type=int, default=64, help="Tokens")
    p_bench.add_argument("--context", type=int, default=2048, help="Context limit")
    p_bench.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    p_bench.add_argument("--runs", type=int, default=3, help="Measurement runs")
    p_bench.add_argument("--profile", default="mobile-balanced", help="Profile")
    p_bench.set_defaults(func=cmd_benchmark)

    p_server = subparsers.add_parser("server", help="Start local API server")
    p_server.add_argument("--host", default="127.0.0.1", help="Host")
    p_server.add_argument("--port", type=int, default=8787, help="Port")
    p_server.set_defaults(func=lambda args: __import__("server.local_api.server", fromlist=["start_server"]).start_server(args.host, args.port))

    p_arch = subparsers.add_parser("architectures", help="Model architectures registry")
    arch_sub = p_arch.add_subparsers(dest="arch_subcommand", required=True)
    
    p_arch_list = arch_sub.add_parser("list", help="List catalogued architectures")
    p_arch_list.add_argument("--family", default=None, help="Filter by family")
    p_arch_list.add_argument("--search", default=None, help="Search by query")
    p_arch_list.add_argument("--status", default=None, choices=["tested", "experimental"], help="Filter by support tier")
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

    p_hermes = subparsers.add_parser("hermes", help="Hermes 4.3 36B operations")
    hermes_sub = p_hermes.add_subparsers(dest="hermes_subcommand", required=True)

    p_h_chat = hermes_sub.add_parser("chat", help="Chat with Hermes 4.3 36B")
    p_h_chat.add_argument("--prompt", required=True, help="User prompt")
    p_h_chat.add_argument("--thinking", action="store_true", default=True, help="Enable hybrid reasoning")
    p_h_chat.add_argument("--remote-url", default=None, help="Remote Space or Router API URL")
    p_h_chat.set_defaults(func=cmd_hermes_chat)

    p_h_test = hermes_sub.add_parser("test", help="Run end-to-end conformance test for Hermes 4.3 36B")
    p_h_test.add_argument("--remote-url", default=None, help="Remote API URL")
    p_h_test.set_defaults(func=cmd_hermes_test)

    p_h_download = hermes_sub.add_parser("download", help="Download official Hermes 4.3 36B GGUF weights")
    p_h_download.add_argument("--quant", default="Q4_K_M", choices=["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"], help="Quantization scheme")
    p_h_download.add_argument("--output-dir", default="models", help="Destination folder")
    p_h_download.set_defaults(func=cmd_hermes_download)

    p_h_run = hermes_sub.add_parser("run", help="Run Hermes 4.3 36B locally using QERLIO GGUF engine")
    p_h_run.add_argument("--model-path", required=True, help="Path to local GGUF weights")
    p_h_run.add_argument("--prompt", required=True, help="Prompt text")
    p_h_run.add_argument("--thinking", action="store_true", default=True, help="Enable hybrid reasoning")
    p_h_run.add_argument("--context-length", type=int, default=4096, help="Context length")
    p_h_run.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    p_h_run.set_defaults(func=cmd_hermes_run)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
