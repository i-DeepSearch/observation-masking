#!/bin/bash
# Start multiple vLLM servers for GPT-OSS models, one per TP group.
# Usage: ./scripts/start_gptoss_servers.sh [base_port] [cuda_visible_devices] [model_path] [gpu_memory_utilization] [tensor_parallel_size] [max_num_seqs] [max_model_len]
#
# Arguments:
#   base_port             : Starting port number (default: 8010)
#   cuda_visible_devices  : Comma-separated GPU IDs to use (default: auto-detect all)
#   model_path            : Path or HuggingFace repo for the model
#   gpu_memory_utilization: GPU memory utilization ratio, single value or comma-separated per server
#                           e.g. "0.9" applies to all servers; "0.9,0.8,0.7" sets per-server ratio
#   tensor_parallel_size  : TP size per server (default: 4, recommended for 120B on H100 80GB)
#   max_num_seqs          : vLLM max concurrent sequences per server
#   max_model_len         : Optional max model length

BASE_PORT=${1:-${BASE_PORT:-8010}}
CUDA_VISIBLE_DEVICES_ARG=${2:-${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}}
MODEL=${3:-${MODEL:-"openai/gpt-oss-120b"}}
GPU_MEM_UTIL_ARG=${4:-${GPU_MEM_UTIL:-"0.95,0.95"}}  # Single value or comma-separated list per server
TP_SIZE=${5:-${TP_SIZE:-4}}
MAX_NUM_SEQS=${6:-${MAX_NUM_SEQS:-32}}
MAX_MODEL_LEN=${7:-${MAX_MODEL_LEN:-""}}

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Detect available GPUs
if [ -n "$CUDA_VISIBLE_DEVICES_ARG" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES_ARG"
else
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

# Parse per-server gpu_memory_utilization
IFS=',' read -ra MEM_UTIL_ARRAY <<< "$GPU_MEM_UTIL_ARG"
# If only one value provided, replicate it for all servers
if [ ${#MEM_UTIL_ARRAY[@]} -eq 1 ]; then
    for i in $(seq 1 $((NUM_SERVERS-1))); do
        MEM_UTIL_ARRAY+=("${MEM_UTIL_ARRAY[0]}")
    done
fi
if [ ${#MEM_UTIL_ARRAY[@]} -ne $NUM_SERVERS ]; then
    echo "Error: gpu_mem_util has ${#MEM_UTIL_ARRAY[@]} value(s) but NUM_SERVERS=$NUM_SERVERS"
    echo "Provide either a single value or exactly $NUM_SERVERS comma-separated values."
    exit 1
fi

echo "=========================================="
echo "Starting GPTOSS-120B vLLM Servers (H100)"
echo "=========================================="
echo "Model: $MODEL"
echo "Total GPUs: $TOTAL_GPUS (${GPU_ARRAY[@]})"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Number of Servers: $NUM_SERVERS"
echo "Base Port: $BASE_PORT"
echo "GPU Mem Util per server: ${MEM_UTIL_ARRAY[@]}"
echo "Max Seqs: $MAX_NUM_SEQS"
[ -n "$MAX_MODEL_LEN" ] && echo "Max Model Len: $MAX_MODEL_LEN"
echo "=========================================="
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
    LOG_FILE="logs/vllm_gptoss_server_${PORT}.log"
    MEM_UTIL="${MEM_UTIL_ARRAY[$i]}"
    EXTRA_ARGS=()
    [ -n "$MAX_MODEL_LEN" ] && EXTRA_ARGS+=(--max_model_len "$MAX_MODEL_LEN")

    echo "Starting Server $((i+1))/$NUM_SERVERS:"
    echo "  - GPUs: $SERVER_GPUS"
    echo "  - Port: $PORT"
    echo "  - GPU Mem Util: $MEM_UTIL"
    echo "  - Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$SERVER_GPUS python scripts/deploy_vllm_service.py \
        --model "$MODEL" \
        --port $PORT \
        --tensor_parallel_size $TP_SIZE \
        --gpu_memory_utilization $MEM_UTIL \
        --max_num_seqs $MAX_NUM_SEQS \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "  - PID: $PID"
    echo ""

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
echo "PIDs: ${PIDS[@]}"
echo "${PIDS[@]}" > logs/gptoss_server_pids.txt
echo "PIDs saved to logs/gptoss_server_pids.txt"
echo ""
echo "To stop all servers:"
echo "  kill \$(cat logs/gptoss_server_pids.txt)"
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
