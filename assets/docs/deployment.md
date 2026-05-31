# Model Deployment

This document collects the model-serving recipes that are too detailed for the main README. Multi-replica launchers split the selected GPUs by tensor-parallel groups: `num_replicas = num_gpus / tensor_parallel_size`. They expose OpenAI-compatible vLLM endpoints at consecutive ports: `base_port`, `base_port + 1`, ... `base_port + num_replicas - 1`.

## Recommended 8-GPU Recipes

| Model | Launcher | Recommended 8-GPU command | Result |
|-------|----------|---------------------------|--------|
| Qwen3.5-9B | `scripts/start_qwen_servers.sh` | `bash scripts/start_qwen_servers.sh 8010 "1,2,3,4,5,6,7" "Qwen/Qwen3.5-9B" "0.95,0.95,0.95,0.95,0.95,0.95,0.95" 1 32` | 7 replicas, TP=1, ports `8010-8016` |
| Qwen3.5/3.6-35B-A3B | `scripts/start_qwen_servers.sh` | `bash scripts/start_qwen_servers.sh 8010 "0,1,2,3,4,5,6,7" "Qwen/Qwen3.5-35B-A3B" "0.90" 2 32` | 4 replicas, TP=2, ports `8010-8013` |
| GPT-OSS-20B | `scripts/start_gptoss_servers.sh` | `bash scripts/start_gptoss_servers.sh 8010 "2,3,4,5,6,7" "openai/gpt-oss-20b" "0.95,0.95,0.95" 2 32` | 3 replicas, TP=2, ports `8010-8012` |
| GPT-OSS-120B | `scripts/start_gptoss_servers.sh` | `bash scripts/start_gptoss_servers.sh 8010 "0,1,2,3,4,5,6,7" "openai/gpt-oss-120b" "0.95,0.95" 4 32` | 2 replicas, TP=4, ports `8010-8011` |
| DeepSeek-V4-Flash | `scripts/start_deepseek_flash.sh` | `bash scripts/start_deepseek_flash.sh 8010 "0,1,2,3,4,5,6,7" "deepseek-ai/DeepSeek-V4-Flash" "0.92" 8 64` | 1 replica, TP=8, port `8010` |
| NVIDIA Nemotron 3 Nano 30B-A3B | `scripts/start_nemotron_servers.sh` | `bash scripts/start_nemotron_servers.sh 8010 "0,1,2,3,4,5,6,7" "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" "0.90" 2 32` | 4 replicas, TP=2, ports `8010-8013` |

Adjust `gpu_memory_utilization` downward if vLLM fails during KV-cache allocation. Increase `tensor_parallel_size` when a model does not fit in the per-GPU memory budget.

## Environment Setup

### Qwen3.5 / Qwen3.6

The Qwen launcher uses a dedicated `qwen35` environment:

```bash
uv venv qwen35 --python 3.12
uv pip install vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --python qwen35/bin/python
```

### DeepSeek-V4-Flash

DeepSeek uses a dedicated `ds` environment and optionally installs DeepGEMM:

```bash
uv venv ds --python 3.12 --seed
uv pip install vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --python ds/bin/python

VIRTUAL_ENV="$PWD/ds" PATH="$PWD/ds/bin:$PATH" \
  bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)
```

To reuse an existing environment:

```bash
DS_LINK=/path/to/ds bash scripts/start_deepseek_flash.sh
```

To refresh DeepGEMM:

```bash
DS_INSTALL_DEEPGEMM=1 bash scripts/start_deepseek_flash.sh
```

## Launcher Parameters

All model launchers use the same positional argument order:

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

| Parameter | Meaning |
|-----------|---------|
| `base_port` | First serving port. Replicas use `base_port`, `base_port + 1`, ... |
| `cuda_visible_devices` | Physical GPU IDs assigned to this launch. |
| `model_path` | Hugging Face repo ID or local model path. |
| `gpu_memory_utilization` | Single value for all replicas or comma-separated values per replica. |
| `tensor_parallel_size` | Number of GPUs per replica. `num_replicas = num_gpus / tensor_parallel_size`. |
| `max_num_seqs` | Optional vLLM maximum concurrent sequences per replica. |
| `max_model_len` | Optional maximum model context length. |

The supported launchers are:

- `scripts/start_qwen_servers.sh`
- `scripts/start_gptoss_servers.sh`
- `scripts/start_nemotron_servers.sh`
- `scripts/start_deepseek_flash.sh`

### Qwen3.5 / Qwen3.6

```bash
bash scripts/start_qwen_servers.sh \
  8010 \
  "1,2,3,4,5,6,7" \
  "Qwen/Qwen3.5-9B" \
  "0.95,0.95,0.95,0.95,0.95,0.95,0.95" \
  1 \
  32
```

### GPT-OSS

```bash
bash scripts/start_gptoss_servers.sh \
  8010 \
  "2,3,4,5,6,7" \
  "openai/gpt-oss-20b" \
  "0.95,0.95,0.95" \
  2 \
  32
```

### NVIDIA Nemotron 3

```bash
bash scripts/start_nemotron_servers.sh \
  8010 \
  "0,1,2,3,4,5,6,7" \
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
  "0.90" \
  2 \
  32
```

### DeepSeek-V4-Flash

```bash
bash scripts/start_deepseek_flash.sh \
  8010 \
  "0,1,2,3,4,5,6,7" \
  "deepseek-ai/DeepSeek-V4-Flash" \
  "0.92" \
  8 \
  64
```

DeepSeek also supports `DS_LINK`, `DS_CUDA_COMPAT_DIR`, and `DS_INSTALL_DEEPGEMM` environment variables for its dedicated runtime environment.

## Mapping Server Count to `run_agent.sh`

When running the agent, set `num_servers` to the number of launched replicas:

| Serving Recipe | Ports | `run_agent.sh` `base_port` | `run_agent.sh` `num_servers` |
|----------------|-------|-----------------------------|------------------------------|
| Qwen3.5-9B TP=1 on the last 7 GPUs | `8010-8016` | `8010` | `7` |
| Qwen3.5/3.6-35B TP=2 on 8 GPUs | `8010-8013` | `8010` | `4` |
| GPT-OSS-20B TP=2 on the last 6 GPUs | `8010-8012` | `8010` | `3` |
| GPT-OSS-120B TP=4 on 8 GPUs | `8010-8011` | `8010` | `2` |
| DeepSeek-V4-Flash TP=8 on 8 GPUs | `8010` | `8010` | `1` |
| Nemotron 3 TP=2 on 8 GPUs | `8010-8013` | `8010` | `4` |
