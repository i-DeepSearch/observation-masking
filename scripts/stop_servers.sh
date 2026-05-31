#!/bin/bash
# Stop all running vLLM servers
# Usage: ./scripts/stop_servers.sh

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f logs/server_pids.txt ]; then
    echo "Stopping all vLLM servers..."
    PIDS=$(cat logs/server_pids.txt)
    kill $PIDS 2>/dev/null

    # Wait for processes to exit
    sleep 2

    # Force kill if still running
    kill -9 $PIDS 2>/dev/null

    echo "All servers stopped."
    rm logs/server_pids.txt
else
    echo "No server PIDs found. Servers may not be running."
    echo "Trying to find and kill all vLLM processes..."

    # Find and kill all python processes running vllm
    pkill -f "vllm.entrypoints.openai.api_server"

    echo "Done."
fi
