#!/bin/bash
# Start multiple vLLM servers using qwen35 venv.
# Usage: ./scripts/start_qwen_servers.sh [base_port] [cuda_visible_devices] [model_path] [gpu_memory_utilization] [tensor_parallel_size] [max_num_seqs] [max_model_len]

BASE_PORT=${1:-${BASE_PORT:-8010}}
CUDA_VISIBLE_DEVICES_ARG=${2:-${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}}
MODEL=${3:-${MODEL:-"Qwen/Qwen3.5-9B"}}
GPU_MEM_UTIL=${4:-${GPU_MEM_UTIL:-0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95}}
TP_SIZE=${5:-${TP_SIZE:-1}}
MAX_NUM_SEQS=${6:-${MAX_NUM_SEQS:-""}}
MAX_MODEL_LEN=${7:-${MAX_MODEL_LEN:-""}}

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOCAL_QWEN35_LINK="$PROJECT_ROOT/qwen35"
if [ -z "${QWEN35_LINK:-}" ]; then
    if [ -x "$LOCAL_QWEN35_LINK/bin/python" ]; then
        QWEN35_LINK="$LOCAL_QWEN35_LINK"
    else
        QWEN35_LINK="$LOCAL_QWEN35_LINK"
    fi
fi

PYTHON="$QWEN35_LINK/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "Error: qwen35 venv not found at $QWEN35_LINK"
    echo "Set QWEN35_LINK=/path/to/qwen35 or create $LOCAL_QWEN35_LINK."
    exit 1
fi

LOCAL_CUDA_COMPAT_DIR="$PROJECT_ROOT/.cuda-compat-13.0"
if [ -z "${QWEN35_CUDA_COMPAT_DIR:-}" ]; then
    if [ -f "$LOCAL_CUDA_COMPAT_DIR/libcuda.so.1" ]; then
        QWEN35_CUDA_COMPAT_DIR="$LOCAL_CUDA_COMPAT_DIR"
    else
        QWEN35_CUDA_COMPAT_DIR=""
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

append_cuda_lib "$QWEN35_CUDA_COMPAT_DIR"

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

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES_ARG"
TOTAL_GPUS=${#GPU_ARRAY[@]}

if [ $((TOTAL_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "Error: Total GPUs ($TOTAL_GPUS) must be divisible by TP_SIZE ($TP_SIZE)"
    exit 1
fi

NUM_SERVERS=$((TOTAL_GPUS / TP_SIZE))

echo "=========================================="
echo "Starting Qwen3.5-35B-A3B vLLM Servers"
echo "=========================================="
echo "Model: $MODEL"
echo "Python: $PYTHON"
echo "qwen35: $QWEN35_LINK"
if [ -n "$QWEN35_CUDA_COMPAT_DIR" ]; then
    echo "CUDA Compat: $QWEN35_CUDA_COMPAT_DIR"
fi
echo "Total GPUs: $TOTAL_GPUS (${GPU_ARRAY[@]})"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Number of Servers: $NUM_SERVERS"
echo "Base Port: $BASE_PORT"
echo "GPU Memory Util: $GPU_MEM_UTIL"
[ -n "$MAX_NUM_SEQS" ] && echo "Max Seqs: $MAX_NUM_SEQS"
[ -n "$MAX_MODEL_LEN" ] && echo "Max Model Len: $MAX_MODEL_LEN"
echo "=========================================="
echo ""

mkdir -p logs

IFS=',' read -ra GPU_MEM_UTIL_ARRAY <<< "$GPU_MEM_UTIL"

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
    LOG_FILE="logs/vllm_qwen35_${PORT}.log"
    EXTRA_ARGS=()
    [ -n "$MAX_NUM_SEQS" ] && EXTRA_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
    [ -n "$MAX_MODEL_LEN" ] && EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")

    # Pick per-server mem util, fall back to first value
    if [ ${#GPU_MEM_UTIL_ARRAY[@]} -gt 1 ] && [ -n "${GPU_MEM_UTIL_ARRAY[$i]}" ]; then
        SERVER_MEM_UTIL=${GPU_MEM_UTIL_ARRAY[$i]}
    else
        SERVER_MEM_UTIL=${GPU_MEM_UTIL_ARRAY[0]}
    fi

    echo "Starting Server $((i+1))/$NUM_SERVERS:"
    echo "  - GPUs: $SERVER_GPUS"
    echo "  - Port: $PORT"
    echo "  - Mem Util: $SERVER_MEM_UTIL"
    echo "  - Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$SERVER_GPUS "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $SERVER_MEM_UTIL \
        --trust-remote-code \
        --enable-prefix-caching \
        --served-model-name "$MODEL" \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "  - PID: $PID"
    echo ""

    sleep 2
done

echo "=========================================="
echo "All $NUM_SERVERS servers started!"
echo "=========================================="
echo ""
echo "Server URLs:"
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    echo "  - http://localhost:${PORT}/v1"
done
echo ""

echo "${PIDS[@]}" > logs/qwen35_server_pids.txt
echo "PIDs saved to logs/qwen35_server_pids.txt"
echo "To stop: kill \$(cat logs/qwen35_server_pids.txt)"
echo "=========================================="

wait -n
echo ""
echo "One server exited. Stopping all servers..."
kill ${PIDS[@]} 2>/dev/null
wait
echo "All servers stopped."
