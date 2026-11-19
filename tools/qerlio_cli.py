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

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
