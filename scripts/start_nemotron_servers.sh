#!/bin/bash
# Start multiple vLLM servers, one per TP group.
# Usage: ./scripts/start_nemotron_servers.sh [base_port] [cuda_visible_devices] [model_path] [gpu_memory_utilization] [tensor_parallel_size] [max_num_seqs] [max_model_len]

BASE_PORT=${1:-${BASE_PORT:-8010}}
CUDA_VISIBLE_DEVICES=${2:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
MODEL=${3:-${MODEL:-"OpenResearcher/OpenResearcher-30B-A3B"}}
GPU_MEM_UTIL=${4:-${GPU_MEM_UTIL:-0.9,0.9,0.9,0.9}}
TP_SIZE=${5:-${TP_SIZE:-2}}
MAX_NUM_SEQS=${6:-${MAX_NUM_SEQS:-""}}
MAX_MODEL_LEN=${7:-${MAX_MODEL_LEN:-""}}

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Detect available GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
else
    # Auto-detect all GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    GPU_ARRAY=($(seq 0 $((NUM_GPUS-1))))
fi

TOTAL_GPUS=${#GPU_ARRAY[@]}

# Calculate number of servers
if [ $((TOTAL_GPUS % TP_SIZE)) -ne 0 ]; then
    echo "Error: Total GPUs ($TOTAL_GPUS) must be divisible by TP_SIZE ($TP_SIZE)"
    exit 1
fi

NUM_SERVERS=$((TOTAL_GPUS / TP_SIZE))

echo "=========================================="
echo "Starting Multiple vLLM Servers"
echo "=========================================="
echo "Model: $MODEL"
echo "Total GPUs: $TOTAL_GPUS (${GPU_ARRAY[@]})"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Number of Servers: $NUM_SERVERS"
echo "Base Port: $BASE_PORT"
echo "GPU Memory Util: $GPU_MEM_UTIL"
[ -n "$MAX_NUM_SEQS" ] && echo "Max Seqs: $MAX_NUM_SEQS"
[ -n "$MAX_MODEL_LEN" ] && echo "Max Model Len: $MAX_MODEL_LEN"
echo "=========================================="

IFS=',' read -ra GPU_MEM_UTIL_ARRAY <<< "$GPU_MEM_UTIL"
echo ""

# Create log directory
mkdir -p logs

# Start each server in background
PIDS=()
for i in $(seq 0 $((NUM_SERVERS-1))); do
    # Get GPU IDs for this server
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
    LOG_FILE="logs/vllm_server_${PORT}.log"
    EXTRA_ARGS=()
    [ -n "$MAX_NUM_SEQS" ] && EXTRA_ARGS+=(--max_num_seqs "$MAX_NUM_SEQS")
    [ -n "$MAX_MODEL_LEN" ] && EXTRA_ARGS+=(--max_model_len "$MAX_MODEL_LEN")

    # Get GPU memory util for this server
    if [ ${#GPU_MEM_UTIL_ARRAY[@]} -eq 1 ]; then
        SERVER_MEM_UTIL=${GPU_MEM_UTIL_ARRAY[0]}
    else
        SERVER_MEM_UTIL=${GPU_MEM_UTIL_ARRAY[$i]}
        if [ -z "$SERVER_MEM_UTIL" ]; then
            SERVER_MEM_UTIL=${GPU_MEM_UTIL_ARRAY[0]}
        fi
    fi

    echo "Starting Server $((i+1))/$NUM_SERVERS:"
    echo "  - GPUs: $SERVER_GPUS"
    echo "  - Port: $PORT"
    echo "  - Mem Util: $SERVER_MEM_UTIL"
    echo "  - Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$SERVER_GPUS python scripts/deploy_vllm_service.py \
        --model "$MODEL" \
        --port $PORT \
        --tensor_parallel_size $TP_SIZE \
        --gpu_memory_utilization $SERVER_MEM_UTIL \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "  - PID: $PID"
    echo ""

    # Wait a bit before starting next server
    sleep 2
done

echo "=========================================="
echo "All servers started!"
echo "=========================================="
echo ""
echo "Server URLs:"
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    echo "  - http://localhost:${PORT}/v1"
done
echo ""
echo "To stop all servers:"
echo "  kill ${PIDS[@]}"
echo ""
echo "Or save PIDs to file:"
echo "  echo '${PIDS[@]}' > logs/server_pids.txt"
echo ""

# Save PIDs
echo "${PIDS[@]}" > logs/server_pids.txt
echo "PIDs saved to logs/server_pids.txt"
echo ""
echo "To stop all servers later:"
echo "  kill \$(cat logs/server_pids.txt)"
echo ""
echo "Press Ctrl+C to stop all servers now, or close this terminal to keep them running."
echo "=========================================="

# Wait for any server to exit
wait -n

# If one exits, kill all others
echo ""
echo "One server exited. Stopping all servers..."
kill ${PIDS[@]} 2>/dev/null
wait
echo "All servers stopped."
