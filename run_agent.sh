#!/bin/bash
# Quick start script for running agent with multiple vLLM servers
# Usage: ./run_agent.sh [output_dir] [base_port] [num_servers] [dataset_name] [browser_backend] [model_path] [archive_turns] [external_url] [parallel_tool_calls]
#
# external_url (arg 8): if set, skip local port building and use this URL directly.
#   Single URL:  https://your-ngrok.ngrok-free.dev/v1
#   Multi URL:   http://host1/v1,http://host2/v1
#   When using external_url, num_servers is ignored (worker count = number of comma-separated URLs).

OUTPUT_DIR=${1:-"results/browsecomp-plus/qwen3.5-9b-bm25-auto"}
BASE_PORT=${2:-8010}
NUM_SERVERS=${3:-8}
DATASET_NAME=${4:-"browsecomp_plus"}
BROWSER_BACKEND=${5:-"local"}
MODEL=${6:-"Qwen/Qwen3.5-9B"}
# Auto-archive threshold: browser results older than this many assistant turns get archived
# Default 10000 effectively disables it; pass e.g. 4 to enable aggressive observation masking.
ARCHIVE_TURNS=${7:-"4"}
# External model URL (optional). When set, bypasses local vLLM port construction entirely.
EXTERNAL_URL=${8:-""}
# Parallel tool calls: "off" disables model-specific multi-tool-call prompting and concurrent browser.search execution.
PARALLEL_TOOL_CALLS=${9:-"on"}

# Search service URL for local BrowseComp-Plus backends. Override, for example:
#   SEARCH_URL=http://localhost:8003 bash run_agent.sh ...
SEARCH_URL=${SEARCH_URL:-"http://localhost:8000"}

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build comma-separated server URLs
if [ -n "$EXTERNAL_URL" ]; then
    SERVER_URLS="$EXTERNAL_URL"
else
    SERVER_URLS=""
    for i in $(seq 0 $((NUM_SERVERS-1))); do
        PORT=$((BASE_PORT + i))
        URL="http://localhost:${PORT}/v1"
        if [ -n "$SERVER_URLS" ]; then
            SERVER_URLS="${SERVER_URLS},${URL}"
        else
            SERVER_URLS="${URL}"
        fi
    done
fi

echo "=========================================="
echo "Starting Agent with Multiple vLLM Servers"
echo "=========================================="
echo "Model: $MODEL"
if [ -n "$EXTERNAL_URL" ]; then
    echo "Mode: External URL"
    echo "Server URL(s): $EXTERNAL_URL"
else
    echo "Number of Servers: $NUM_SERVERS"
    echo "Server URLs:"
    for i in $(seq 0 $((NUM_SERVERS-1))); do
        PORT=$((BASE_PORT + i))
        echo "  - http://localhost:${PORT}/v1"
    done
fi
echo "Search Service: $SEARCH_URL"
echo "Dataset: $DATASET_NAME"
echo "Browser Backend: $BROWSER_BACKEND"
echo "Output Directory: $OUTPUT_DIR"
echo "Archive after turns: ${ARCHIVE_TURNS:-4 (default)}"
echo "Parallel tool calls: $PARALLEL_TOOL_CALLS"
echo "=========================================="
echo ""

# Check if using browsecomp-plus dataset (needs local data path)
if [ "$DATASET_NAME" = "browsecomp_plus" ]; then
    DATA_PATH="${SCRIPT_DIR}/Tevatron/browsecomp-plus/data/*.parquet"
    echo "Using local BrowseComp-Plus dataset: $DATA_PATH"
    echo ""

    python deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL" \
        --search_url "$SEARCH_URL" \
        --dataset_name "$DATASET_NAME" \
        --data_path "$DATA_PATH" \
        --browser_backend "$BROWSER_BACKEND" \
        --reasoning_effort high \
        --vllm_server_url "$SERVER_URLS" \
        --max_concurrency_per_worker 32 \
        $( [ "$PARALLEL_TOOL_CALLS" = "off" ] && echo "--disable_parallel_tool_calls" ) \
        $( [ -n "$ARCHIVE_TURNS" ] && echo "--force_archive_after_turns $ARCHIVE_TURNS" )
else
    # HuggingFace datasets or OpenAI BrowseComp (no local data_path needed)
    echo "Using dataset: $DATASET_NAME"
    echo "Available datasets: browsecomp, gaia, xbench"
    echo ""

    python deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL" \
        --search_url "$SEARCH_URL" \
        --dataset_name "$DATASET_NAME" \
        --browser_backend "$BROWSER_BACKEND" \
        --reasoning_effort high \
        --vllm_server_url "$SERVER_URLS" \
        --max_concurrency_per_worker 32 \
        $( [ "$PARALLEL_TOOL_CALLS" = "off" ] && echo "--disable_parallel_tool_calls" ) \
        $( [ -n "$ARCHIVE_TURNS" ] && echo "--force_archive_after_turns $ARCHIVE_TURNS" )
fi
