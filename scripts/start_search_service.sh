#!/bin/bash

# Start search service with proper environment setup
# Usage: ./scripts/start_search_service.sh [bm25|dense|agentir] [port] [cuda_device]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parameters
SEARCH_MODE="${1:-agentir}"
SEARCH_MODE="${SEARCH_MODE,,}"
if [ -n "${2:-}" ]; then
    PORT="$2"
else
    PORT="8000"
fi
export CUDA_VISIBLE_DEVICES=${3:-0}
# Set GPU_IDS to 0 because CUDA_VISIBLE_DEVICES restricts the view to only the selected GPUs,
# so the application sees them starting from index 0.
export GPU_IDS=0

case "$SEARCH_MODE" in
    bm25|dense|agentir)
        ;;
    *)
        echo -e "${RED}Error: Invalid search mode. Use 'bm25', 'dense', or 'agentir'${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Starting Search Service${NC}"
echo -e "${GREEN}================================${NC}"
echo "Search Mode: ${SEARCH_MODE}"
echo "Port: ${PORT}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/cache_env.sh"

configure_java() {
    local project_jdk_home="${PROJECT_ROOT}/.jdk/jdk-21"
    local java_version
    local java_major
    local java_bin
    local javac_bin

    if [ -x "${project_jdk_home}/bin/java" ] && [ -x "${project_jdk_home}/bin/javac" ]; then
        export JAVA_HOME="${project_jdk_home}"
        export PATH="${JAVA_HOME}/bin:${PATH}"
    elif [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/java" ] && [ -x "${JAVA_HOME}/bin/javac" ]; then
        export PATH="${JAVA_HOME}/bin:${PATH}"
    fi

    java_bin="$(command -v java || true)"
    javac_bin="$(command -v javac || true)"
    if [ -z "${java_bin}" ] || [ -z "${javac_bin}" ]; then
        echo -e "${RED}Error: Java JDK 21 is not available in this shell.${NC}"
        echo "Run ./setup.sh, or source ${PROJECT_ROOT}/.jdk/env.sh before starting the service."
        exit 1
    fi

    java_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    java_major="${java_version%%.*}"
    if [[ "${java_version}" == 1.* ]]; then
        java_major="${java_version#1.}"
        java_major="${java_major%%.*}"
    fi
    if [ "${java_major}" != "21" ]; then
        echo -e "${RED}Error: OpenJDK 21 is required, but found Java ${java_version:-unknown}.${NC}"
        echo "Run ./setup.sh, or source ${PROJECT_ROOT}/.jdk/env.sh before starting the service."
        exit 1
    fi

    echo "JAVA_HOME: ${JAVA_HOME:-$(dirname "$(dirname "${java_bin}")")}"
    echo "Java: ${java_version}"
    echo "javac path: ${javac_bin}"
    echo ""
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run ./setup.sh first${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Verify we're using the right Python
PYTHON_VERSION=$(python --version)
echo "Using: $PYTHON_VERSION"
echo "Python path: $(which python)"
echo ""

if [ "$SEARCH_MODE" = "agentir" ]; then
    AGENTIR_INDEX_DIR="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/agentir-4b/AgentIR_browsecomp-plus"
    INDEX_FILES=$(ls "${AGENTIR_INDEX_DIR}"/*.pkl 2>/dev/null | wc -l)
    if [ "$INDEX_FILES" -eq 0 ]; then
        echo -e "${YELLOW}AgentIR index not found. Downloading via setup.sh...${NC}"
        bash "${PROJECT_ROOT}/setup.sh"
    fi
fi

configure_java

# Set common environment variables
export LUCENE_EXTRA_DIR="${PROJECT_ROOT}/tevatron"
export CORPUS_PARQUET_PATH="${PROJECT_ROOT}/Tevatron/browsecomp-plus-corpus/data/*.parquet"

echo "LUCENE_EXTRA_DIR: ${LUCENE_EXTRA_DIR}"
echo "CORPUS_PARQUET_PATH: ${CORPUS_PARQUET_PATH}"

# Check if Lucene JARs exist
if [ ! -f "${LUCENE_EXTRA_DIR}/lucene-highlighter-9.9.1.jar" ]; then
    echo -e "${RED}Error: Lucene JARs not found in ${LUCENE_EXTRA_DIR}${NC}"
    echo "Please run ./setup.sh to download them"
    exit 1
fi

# Check if corpus exists
CORPUS_COUNT=$(ls ${PROJECT_ROOT}/Tevatron/browsecomp-plus-corpus/data/*.parquet 2>/dev/null | wc -l)
if [ "$CORPUS_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: Corpus not found${NC}"
    echo "Please run ./setup.sh to download the corpus"
    exit 1
fi

# Configure searcher-specific settings
unset REASONING_AWARE_SEARCH

if [ "$SEARCH_MODE" = "bm25" ]; then
    export LUCENE_INDEX_DIR="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/bm25"
    export SEARCHER_TYPE="bm25"

    if [ ! -d "$LUCENE_INDEX_DIR" ]; then
        echo -e "${RED}Error: BM25 index not found at $LUCENE_INDEX_DIR${NC}"
        echo "Please run ./setup.sh to download the index"
        exit 1
    fi

    echo "LUCENE_INDEX_DIR: ${LUCENE_INDEX_DIR}"
    echo "SEARCHER_TYPE: ${SEARCHER_TYPE}"

elif [ "$SEARCH_MODE" = "dense" ]; then
    export DENSE_INDEX_PATH="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl"
    export DENSE_MODEL_NAME="Qwen/Qwen3-Embedding-8B"
    export SEARCHER_TYPE="dense"

    # Check if index files exist
    INDEX_FILES=$(ls ${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl 2>/dev/null | wc -l)
    if [ "$INDEX_FILES" -eq 0 ]; then
        echo -e "${RED}Error: Dense index not found${NC}"
        echo "Please run ./setup.sh to download the index"
        exit 1
    fi

    echo "DENSE_INDEX_PATH: ${DENSE_INDEX_PATH}"
    echo "DENSE_MODEL_NAME: ${DENSE_MODEL_NAME}"
    echo "SEARCHER_TYPE: ${SEARCHER_TYPE}"

elif [ "$SEARCH_MODE" = "agentir" ]; then
    AGENTIR_INDEX_DIR="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/agentir-4b/AgentIR_browsecomp-plus"
    export DENSE_INDEX_PATH="${AGENTIR_INDEX_DIR}/*.pkl"
    export DENSE_MODEL_NAME="Tevatron/AgentIR-4B"
    export SEARCHER_TYPE="dense"
    export REASONING_AWARE_SEARCH="true"

    INDEX_FILES=$(ls "${AGENTIR_INDEX_DIR}"/*.pkl 2>/dev/null | wc -l)
    if [ "$INDEX_FILES" -eq 0 ]; then
        echo -e "${RED}Error: AgentIR index not found at ${AGENTIR_INDEX_DIR}${NC}"
        echo "Please run ./setup.sh to download the index"
        exit 1
    fi

    echo "DENSE_INDEX_PATH: ${DENSE_INDEX_PATH}"
    echo "DENSE_MODEL_NAME: ${DENSE_MODEL_NAME}"
    echo "SEARCHER_TYPE: ${SEARCHER_TYPE}"
    echo "REASONING_AWARE_SEARCH: ${REASONING_AWARE_SEARCH}"

else
    echo -e "${RED}Error: Invalid search mode. Use 'bm25', 'dense', or 'agentir'${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting uvicorn server...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Start uvicorn
uvicorn scripts.deploy_search_service:app --host 0.0.0.0 --port ${PORT}
