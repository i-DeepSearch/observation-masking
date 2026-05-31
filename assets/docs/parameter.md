# Script Parameters Reference

## scripts/start_search_service.sh

Start local search service for BrowseComp-Plus benchmark.

```bash
bash scripts/start_search_service.sh [search_mode] [port] [cuda_visible_devices]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `search_mode` | `dense` | `bm25`, `dense`, or `agentir` |
| `port` | `8000` | Port for search service |
| `cuda_visible_devices` | `0` | GPU ID for dense/AgentIR searcher (ignored for BM25) |

**Important**: Dense and AgentIR search need enough GPU memory; budget about 15GB.

**Examples:**
```bash
# BM25 search (CPU-based, lightweight)
bash scripts/start_search_service.sh bm25 8000

# Dense search (GPU-based, better quality)
bash scripts/start_search_service.sh dense 8000 0

# AgentIR search (GPU-based, reasoning-aware dense search)
bash scripts/start_search_service.sh agentir 8000 0
```

---

## Model Launcher Scripts

The following launchers share the same positional argument order:

- `scripts/start_deepseek_flash.sh`
- `scripts/start_gptoss_servers.sh`
- `scripts/start_nemotron_servers.sh`
- `scripts/start_qwen_servers.sh`

```bash
bash scripts/<launcher>.sh \
  [base_port] \
  [cuda_visible_devices] \
  [model_path] \
  [gpu_memory_utilization] \
  [tensor_parallel_size] \
  [max_num_seqs] \
  [max_model_len]
```

| Parameter | Description |
|-----------|-------------|
| `base_port` | First serving port. Replicas are exposed at `base_port`, `base_port + 1`, ... |
| `cuda_visible_devices` | Comma-separated GPU IDs assigned to the launcher |
| `model_path` | Hugging Face repo ID or local model path |
| `gpu_memory_utilization` | vLLM GPU memory utilization. Use one value for all replicas or comma-separated values per replica |
| `tensor_parallel_size` | Number of GPUs per replica. The launcher starts `num_replicas = num_gpus / tensor_parallel_size` services |
| `max_num_seqs` | Optional vLLM maximum concurrent sequences per replica |
| `max_model_len` | Optional maximum model context length |

Model-specific deployment commands and recommended values live in [`deployment.md`](deployment.md).

---

## run_agent.sh

Main script for running the agent on benchmarks.

```bash
bash run_agent.sh \
  [output_dir] \
  [base_port] \
  [num_servers] \
  [dataset_name] \
  [browser_backend] \
  [model_path] \
  [archive_turns] \
  [external_url] \
  [parallel_tool_calls]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `results/browsecomp-plus/qwen3.5-9b-bm25-auto` | Output directory for result JSONL files |
| `base_port` | `8010` | First local vLLM server port. `run_agent.sh` builds URLs as `http://localhost:{base_port + i}/v1` for `i = 0 ... num_servers - 1` |
| `num_servers` | `8` | Number of local vLLM server URLs to build. This should match the launcher replica count: `num_gpus / tensor_parallel_size`; ignored when `external_url` is set |
| `dataset_name` | `browsecomp_plus` | Dataset key. `browsecomp_plus` uses local BrowseComp-Plus parquet files; other datasets load through the unified HuggingFace loader |
| `browser_backend` | `local` | `local` or `serper` |
| `model_path` | `Qwen/Qwen3.5-9B` | Model name or path |
| `archive_turns` | `4` | Observation masking window. Browser results older than this many assistant turns are archived; use a large value such as `10000` to effectively disable it |
| `external_url` | empty | Optional comma-separated OpenAI-compatible endpoint list. When set, it bypasses local port construction |
| `parallel_tool_calls` | `on` | Set to `off` to disable model-specific multi-tool-call prompting and concurrent `browser.search` execution |

Set `SEARCH_URL` to use a local search service on a non-default port:

```bash
SEARCH_URL=http://localhost:8003 bash run_agent.sh results/<exp> 8010 3 browsecomp_plus local <model_path>
```

If you want to set `parallel_tool_calls` while leaving `external_url` empty, pass `""` for the eighth argument:

```bash
bash run_agent.sh results/<exp> 8010 3 browsecomp_plus local <model_path> 4 "" on
```

**Examples:**
```bash
# BrowseComp-Plus with AgentIR search on port 8003
SEARCH_URL=http://localhost:8003 bash run_agent.sh results/<exp> 8010 3 browsecomp_plus local <model_path> 4 "" on

# BrowseComp-ZH with Serper API
bash run_agent.sh results/<exp> 8010 7 browsecomp-zh serper <model_path> 4 "" on
```

---

## eval.py

Evaluate generated JSONL result files and write `evaluated.jsonl` under the same result directory.

```bash
python eval.py \
  --input_dir [result_dir] \
  --model_name_or_path [tokenizer_or_model_path] \
  --judge_model [judge_model]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | required | Directory containing result `*.jsonl` files from `run_agent.sh`; `evaluated.jsonl` is excluded automatically |
| `--model_name_or_path` | auto-inferred | HuggingFace tokenizer/model path used for context-length plots. If omitted, `eval.py` infers it from `--input_dir` and falls back to `Qwen/Qwen3.5-9B` |
| `--judge_model` | `None` | Optional judge model override for `LLMJudge`, for example `gpt-5.2` through LiteLLM |

**Examples:**
```bash
python eval.py \
  --input_dir results/<dataset>/<exp> \
  --model_name_or_path <model_path>

python eval.py \
  --input_dir results/<dataset>/<exp> \
  --judge_model <judge_model>
```

---

## scripts/stop_servers.sh

Stop all running vLLM servers.

```bash
bash scripts/stop_servers.sh
```

This script reads PIDs from `logs/server_pids.txt` and terminates all servers.
