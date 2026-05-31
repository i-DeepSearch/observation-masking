#!/usr/bin/env python
import sys
# -*- coding: utf-8 -*-
"""
Standalone vLLM OpenAI-compatible server
Usage:
    python deploy_vllm_service.py --model openai/gpt-oss-20b --port 8001 --tensor_parallel_size 1
    python deploy_vllm_service.py --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B --port 8001 --tensor_parallel_size 2
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Start vLLM OpenAI-compatible API server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=None, help="Max model length")
    parser.add_argument("--max_num_seqs", type=int, default=None, help="Max number of concurrent sequences")
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype passed to vLLM, e.g. auto, bfloat16")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method passed to vLLM, e.g. fp8")
    parser.add_argument("--kv_cache_dtype", type=str, default=None, help="KV cache dtype passed to vLLM, e.g. fp8_e4m3")
    parser.add_argument("--served_model_name", type=str, default=None, help="Served model name exposed by OpenAI API")
    parser.add_argument("--calculate_kv_scales", action="store_true", help="Let vLLM calculate FP8 KV cache scales on the fly")
    parser.add_argument("--enable_expert_parallel", action="store_true", help="Enable expert parallelism for MoE models")
    parser.add_argument("--enable_auto_tool_choice", action="store_true", help="Enable vLLM auto tool choice")
    parser.add_argument("--tool_call_parser", type=str, default=None, help="Tool call parser passed to vLLM")
    parser.add_argument("--reasoning_parser", type=str, default=None, help="Reasoning parser passed to vLLM")
    parser.add_argument("--compilation_config", type=str, default=None, help="Compilation config JSON passed to vLLM")
    parser.add_argument("--block_size", type=int, default=None, help="Token block size for paged KV cache, e.g. 256")
    parser.add_argument("--tokenizer_mode", type=str, default=None, help="Tokenizer mode, e.g. deepseek_v4")
    args = parser.parse_args()

    # Build vLLM command
    cmd_parts = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--enable-prefix-caching",
    ]

    if args.max_model_len:
        cmd_parts.extend(["--max-model-len", str(args.max_model_len)])

    if args.max_num_seqs:
        cmd_parts.extend(["--max-num-seqs", str(args.max_num_seqs)])

    if args.dtype:
        cmd_parts.extend(["--dtype", args.dtype])

    if args.quantization:
        cmd_parts.extend(["--quantization", args.quantization])

    if args.kv_cache_dtype:
        cmd_parts.extend(["--kv-cache-dtype", args.kv_cache_dtype])

    if args.served_model_name:
        cmd_parts.extend(["--served-model-name", args.served_model_name])

    if args.calculate_kv_scales:
        cmd_parts.append("--calculate-kv-scales")

    if args.enable_expert_parallel:
        cmd_parts.append("--enable-expert-parallel")

    if args.enable_auto_tool_choice:
        cmd_parts.append("--enable-auto-tool-choice")

    if args.tool_call_parser:
        cmd_parts.extend(["--tool-call-parser", args.tool_call_parser])

    if args.reasoning_parser:
        cmd_parts.extend(["--reasoning-parser", args.reasoning_parser])

    if args.compilation_config:
        cmd_parts.extend(["--compilation-config", args.compilation_config])

    if args.block_size:
        cmd_parts.extend(["--block-size", str(args.block_size)])

    if args.tokenizer_mode:
        cmd_parts.extend(["--tokenizer-mode", args.tokenizer_mode])

    # Print info
    print("=" * 80)
    print("Starting vLLM OpenAI-compatible API server")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    if args.dtype:
        print(f"Dtype: {args.dtype}")
    if args.quantization:
        print(f"Quantization: {args.quantization}")
    if args.kv_cache_dtype:
        print(f"KV Cache Dtype: {args.kv_cache_dtype}")
    if args.calculate_kv_scales:
        print("Calculate KV Scales: enabled")
    if args.enable_expert_parallel:
        print("Expert Parallel: enabled")
    if args.enable_auto_tool_choice:
        print("Auto Tool Choice: enabled")
    if args.tool_call_parser:
        print(f"Tool Call Parser: {args.tool_call_parser}")
    if args.reasoning_parser:
        print(f"Reasoning Parser: {args.reasoning_parser}")
    if args.compilation_config:
        print(f"Compilation Config: {args.compilation_config}")
    print("=" * 80)
    print("\nServer will be available at:")
    print(f"  http://{args.host}:{args.port}/v1")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    print()

    # Execute
    import subprocess
    try:
        subprocess.run(cmd_parts, check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")


if __name__ == "__main__":
    main()
