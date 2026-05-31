#!/bin/bash
# Deploy DeepSeek-V4-Flash using a dedicated ds venv.
#
# Usage:
#   ./scripts/start_deepseek_flash.sh [base_port] [cuda_visible_devices] [model_path] [gpu_memory_utilization] [tensor_parallel_size] [max_num_seqs] [max_model_len]
#
# Override defaults with env vars before calling or pass positional args:
#   BASE_PORT, CUDA_DEVICES, MODEL, GPU_MEM_UTIL, TP_SIZE, MAX_NUM_SEQS, MAX_MODEL_LEN
#   DS_LINK, DS_CUDA_COMPAT_DIR, DS_INSTALL_DEEPGEMM

# vLLM config
BASE_PORT=${1:-${BASE_PORT:-8010}}
CUDA_DEVICES=${2:-${CUDA_DEVICES:-"0,1,2,3,4,5,6,7"}}
MODEL=${3:-${MODEL:-"deepseek-ai/DeepSeek-V4-Flash"}}
GPU_MEM_UTIL=${4:-${GPU_MEM_UTIL:-"0.92"}}
TP_SIZE=${5:-${TP_SIZE:-8}}
MAX_NUM_SEQS=${6:-${MAX_NUM_SEQS:-64}}
MAX_MODEL_LEN=${7:-${MAX_MODEL_LEN:-""}}
DS_INSTALL_DEEPGEMM=${DS_INSTALL_DEEPGEMM:-0}

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/cache_env.sh"
mkdir -p logs

LOCAL_DS_LINK="$PROJECT_ROOT/ds"
if [ -z "${DS_LINK:-}" ]; then
    if [ -x "$LOCAL_DS_LINK/bin/python" ]; then
        DS_LINK="$LOCAL_DS_LINK"
    else
        DS_LINK="$LOCAL_DS_LINK"
    fi
fi

PYTHON="$DS_LINK/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "Error: DeepSeek ds venv not found at $DS_LINK"
    echo "Create it with:"
    echo "  uv venv ds --python 3.12 --seed"
    echo "  uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly --python ds/bin/python"
    echo "Or set DS_LINK=/path/to/ds before running this script."
    exit 1
fi

LOCAL_CUDA_COMPAT_DIR="$PROJECT_ROOT/.cuda-compat-13.0"
if [ -z "${DS_CUDA_COMPAT_DIR:-}" ]; then
    if [ -f "$LOCAL_CUDA_COMPAT_DIR/libcuda.so.1" ]; then
        DS_CUDA_COMPAT_DIR="$LOCAL_CUDA_COMPAT_DIR"
    else
        DS_CUDA_COMPAT_DIR=""
    fi
fi

SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PY_CUDA_LIBS=""
append_cuda_lib() {
    if [ -d "$1" ]; then
        if [ -n "$PY_CUDA_LIBS" ]; then
            PY_CUDA_LIBS="$PY_CUDA_LIBS:$1"
        else
            PY_CUDA_LIBS="$1"
        fi
    fi
}

append_cuda_lib "$DS_CUDA_COMPAT_DIR"

for cuda_home in "$SITE_PACKAGES"/nvidia/cu*; do
    if [ -d "$cuda_home" ]; then
        export CUDA_HOME=${CUDA_HOME:-"$cuda_home"}
        export CUDA_PATH=${CUDA_PATH:-"$CUDA_HOME"}
        export PATH="$CUDA_HOME/bin:$PATH"
        append_cuda_lib "$CUDA_HOME/lib"
        append_cuda_lib "$CUDA_HOME/lib64"
        break
    fi
done

for lib_dir in "$SITE_PACKAGES"/nvidia/*/lib "$SITE_PACKAGES"/nvidia/*/lib64; do
    append_cuda_lib "$lib_dir"
done

if [ -n "$PY_CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="$PY_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$PY_CUDA_LIBS${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

if [ "$DS_INSTALL_DEEPGEMM" = "1" ]; then
    echo "Installing/updating DeepGEMM for DeepSeek into ds..."
    VIRTUAL_ENV="$DS_LINK" PATH="$DS_LINK/bin:$PATH" \
        bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)
fi

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

if [ $((TOTAL_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "Error: Total GPUs ($TOTAL_GPUS) must be divisible by TP_SIZE ($TP_SIZE)"
    exit 1
fi

NUM_SERVERS=$((TOTAL_GPUS / TP_SIZE))

IFS=',' read -ra GPU_MEM_UTIL_ARRAY <<< "$GPU_MEM_UTIL"
if [ ${#GPU_MEM_UTIL_ARRAY[@]} -eq 1 ]; then
    for i in $(seq 1 $((NUM_SERVERS-1))); do
        GPU_MEM_UTIL_ARRAY+=("${GPU_MEM_UTIL_ARRAY[0]}")
    done
fi
if [ ${#GPU_MEM_UTIL_ARRAY[@]} -ne $NUM_SERVERS ]; then
    echo "Error: gpu_memory_utilization has ${#GPU_MEM_UTIL_ARRAY[@]} value(s) but NUM_SERVERS=$NUM_SERVERS"
    echo "Provide either a single value or exactly $NUM_SERVERS comma-separated values."
    exit 1
fi

echo "=========================================="
echo " DeepSeek-V4-Flash vLLM Servers"
echo "=========================================="
echo " Model      : $MODEL"
echo " Python     : $PYTHON"
echo " ds env     : $DS_LINK"
if [ -n "$DS_CUDA_COMPAT_DIR" ]; then
    echo " CUDA compat: $DS_CUDA_COMPAT_DIR"
fi
echo " Total GPUs : $TOTAL_GPUS (${GPU_ARRAY[@]})"
echo " TP size    : $TP_SIZE"
echo " Servers    : $NUM_SERVERS"
echo " Base port  : $BASE_PORT"
echo " Mem util   : ${GPU_MEM_UTIL_ARRAY[@]}"
echo " Max seqs   : $MAX_NUM_SEQS"
[ -n "$MAX_MODEL_LEN" ] && echo " Max ctx    : $MAX_MODEL_LEN"
echo "=========================================="
echo ""

PIDS=()
for i in $(seq 0 $((NUM_SERVERS-1))); do
    START_IDX=$((i * TP_SIZE))
    END_IDX=$((START_IDX + TP_SIZE - 1))

    SERVER_GPUS=""
    for j in $(seq $START_IDX $END_IDX); do
        if [ -n "$SERVER_GPUS" ]; then
            SERVER_GPUS="$SERVER_GPUS,${GPU_ARRAY[$j]}"
        else
            SERVER_GPUS="${GPU_ARRAY[$j]}"
        fi
    done

    PORT=$((BASE_PORT + i))
    LOG_FILE="logs/vllm_deepseek_flash_${PORT}.log"
    SERVER_MEM_UTIL="${GPU_MEM_UTIL_ARRAY[$i]}"
    EXTRA_ARGS=()
    [ -n "$MAX_MODEL_LEN" ] && EXTRA_ARGS+=(--max_model_len "$MAX_MODEL_LEN")

    echo "Starting Server $((i+1))/$NUM_SERVERS:"
    echo "  - GPUs: $SERVER_GPUS"
    echo "  - Port: $PORT"
    echo "  - Mem Util: $SERVER_MEM_UTIL"
    echo "  - Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$SERVER_GPUS "$PYTHON" scripts/deploy_vllm_service.py \
        --model "$MODEL" \
        --port "$PORT" \
        --tensor_parallel_size "$TP_SIZE" \
        --gpu_memory_utilization "$SERVER_MEM_UTIL" \
        --max_num_seqs "$MAX_NUM_SEQS" \
        --kv_cache_dtype fp8 \
        --block_size 256 \
        --tokenizer_mode deepseek_v4 \
        --enable_expert_parallel \
        --enable_auto_tool_choice \
        --tool_call_parser deepseek_v4 \
        --reasoning_parser deepseek_v4 \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "  - PID: $PID"
    echo ""

    sleep 2
done

echo "${PIDS[@]}" > logs/deepseek_flash_vllm.pid
echo "${PIDS[@]}" > logs/deepseek_flash_vllm_pids.txt

echo "=========================================="
echo "All $NUM_SERVERS servers started!"
echo "=========================================="
echo "Server URLs:"
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    echo "  - http://localhost:${PORT}/v1"
done
echo "PIDs saved to logs/deepseek_flash_vllm_pids.txt"
echo "Stop: kill \$(cat logs/deepseek_flash_vllm_pids.txt)"
echo "=========================================="

wait -n
echo ""
echo "One server exited. Stopping all servers..."
kill ${PIDS[@]} 2>/dev/null
wait
echo "All servers stopped."
