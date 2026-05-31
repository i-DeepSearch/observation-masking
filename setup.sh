#!/bin/bash

set -e  # Exit on error

echo "================================"
echo "GPT-OSS-DeepResearch-Eval Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Keep large build/download caches off small home directories by default.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/scripts/cache_env.sh"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# Check if uv is installed
echo -e "\n${YELLOW}[1/6] Checking uv installation...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed. Please install it first:${NC}"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}✓ uv is installed${NC}"

# Create virtual environment with Python 3.12
echo -e "\n${YELLOW}[2/6] Creating Python 3.12 virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    uv venv --python 3.12
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}[3/6] Installing Python packages...${NC}"
source .venv/bin/activate

# Install Python dependencies
uv pip install -e .
echo -e "${GREEN}✓ Python packages installed${NC}"

# Install OpenJDK 21
echo -e "\n${YELLOW}[4/6] Installing OpenJDK 21...${NC}"
JDK_ROOT="$PWD/.jdk"
JDK_HOME="$JDK_ROOT/jdk-21"

download_file() {
    local output="$1"
    shift
    local tmp_output="${output}.tmp"
    local url

    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        echo -e "${RED}Neither curl nor wget is installed; cannot download $output.${NC}"
        exit 1
    fi

    for url in "$@"; do
        rm -f "$tmp_output"
        echo "Downloading $(basename "$output") from $url"
        if command -v curl &> /dev/null; then
            if curl -L --fail --show-error --connect-timeout 20 --max-time 300 --retry 3 --retry-delay 2 -o "$tmp_output" "$url"; then
                mv "$tmp_output" "$output"
                return 0
            fi
        elif wget --timeout=20 --read-timeout=60 --tries=3 --progress=dot:giga -O "$tmp_output" "$url"; then
            mv "$tmp_output" "$output"
            return 0
        fi
    done

    rm -f "$tmp_output"
    echo -e "${RED}Failed to download $output from all known URLs.${NC}"
    exit 1
}

java_version() {
    "$1" -version 2>&1 | awk -F '"' '/version/ {print $2}'
}

java_major_version() {
    local version="$1"
    local major="${version%%.*}"
    if [[ "$version" == 1.* ]]; then
        local rest="${version#1.}"
        major="${rest%%.*}"
    fi
    echo "$major"
}

use_java_21() {
    local java_bin="$1"
    local version
    local major
    local resolved_java
    local java_home

    version=$(java_version "$java_bin")
    major=$(java_major_version "$version")
    if [ "$major" != "21" ]; then
        return 1
    fi

    resolved_java=$(readlink -f "$java_bin" 2>/dev/null || printf '%s' "$java_bin")
    java_home="$(cd "$(dirname "$resolved_java")/.." && pwd)"
    if [ ! -x "$java_home/bin/javac" ]; then
        echo "Found Java 21 at $java_home, but javac is missing."
        return 1
    fi

    export JAVA_HOME="$java_home"
    export PATH="$JAVA_HOME/bin:$PATH"

    echo -e "${GREEN}✓ Using OpenJDK 21: $version ($JAVA_HOME)${NC}"
    if [[ "$JAVA_HOME" == "$JDK_ROOT/"* ]]; then
        printf 'export JAVA_HOME="%s"\nexport PATH="$JAVA_HOME/bin:$PATH"\n' "$JAVA_HOME" > "$JDK_ROOT/env.sh"
        echo "For future shells, run: source $JDK_ROOT/env.sh"
    fi
    return 0
}

download_jdk_21() {
    local machine
    local adoptium_arch
    local jdk_url
    local jdk_tarball
    local jdk_tmp

    machine=$(uname -m)
    case "$machine" in
        x86_64|amd64)
            adoptium_arch="x64"
            ;;
        aarch64|arm64)
            adoptium_arch="aarch64"
            ;;
        *)
            echo -e "${RED}Unsupported CPU architecture for automatic JDK download: $machine${NC}"
            exit 1
            ;;
    esac

    mkdir -p "$JDK_ROOT"
    jdk_url="https://api.adoptium.net/v3/binary/latest/21/ga/linux/${adoptium_arch}/jdk/hotspot/normal/eclipse"
    jdk_tarball="$JDK_ROOT/openjdk-21-${adoptium_arch}.tar.gz"
    jdk_tmp="$JDK_ROOT/jdk-21.tmp"

    echo -e "${YELLOW}Downloading OpenJDK 21 to $JDK_HOME (without root permissions)...${NC}"
    download_file "$jdk_tarball" "$jdk_url"

    rm -rf "$jdk_tmp"
    mkdir -p "$jdk_tmp"
    tar -xzf "$jdk_tarball" -C "$jdk_tmp" --strip-components=1
    rm -rf "$JDK_HOME"
    mv "$jdk_tmp" "$JDK_HOME"
    rm -f "$jdk_tarball"
}

FOUND_JAVA_21=0
if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
    if use_java_21 "$JAVA_HOME/bin/java"; then
        FOUND_JAVA_21=1
    fi
fi

if [ "$FOUND_JAVA_21" -eq 0 ] && command -v java &> /dev/null; then
    EXISTING_JAVA_VERSION=$(java_version "$(command -v java)")
    if use_java_21 "$(command -v java)"; then
        FOUND_JAVA_21=1
    else
        echo "Found Java ${EXISTING_JAVA_VERSION:-unknown}, but OpenJDK 21 is required."
    fi
fi

if [ "$FOUND_JAVA_21" -eq 0 ] && [ -x "$JDK_HOME/bin/java" ]; then
    if use_java_21 "$JDK_HOME/bin/java"; then
        FOUND_JAVA_21=1
    fi
fi

if [ "$FOUND_JAVA_21" -eq 0 ]; then
    download_jdk_21
    use_java_21 "$JDK_HOME/bin/java"
fi

# Clone and install tevatron
echo -e "\n${YELLOW}[5/6] Installing tevatron...${NC}"
if [ -d "tevatron" ]; then
    echo -e "${YELLOW}tevatron directory already exists. Skipping clone.${NC}"
    cd tevatron
    uv pip install -e .
    cd ..
else
    git clone https://github.com/texttron/tevatron.git
    cd tevatron
    uv pip install -e .
    cd ..
    echo -e "${GREEN}✓ tevatron installed${NC}"
fi

# Check Lucene JARs (already in tevatron/)
echo -e "\n${YELLOW}[6/9] Checking Lucene highlighter JARs...${NC}"
if [ -f "tevatron/lucene-highlighter-9.9.1.jar" ]; then
    echo -e "${GREEN}✓ Lucene JARs found in tevatron/${NC}"
else
    echo -e "${YELLOW}Downloading Lucene highlighter JARs to tevatron/...${NC}"
    cd tevatron
    LUCENE_VERSION="9.9.1"
    wget -q "https://repo1.maven.org/maven2/org/apache/lucene/lucene-highlighter/${LUCENE_VERSION}/lucene-highlighter-${LUCENE_VERSION}.jar"
    wget -q "https://repo1.maven.org/maven2/org/apache/lucene/lucene-queries/${LUCENE_VERSION}/lucene-queries-${LUCENE_VERSION}.jar"
    wget -q "https://repo1.maven.org/maven2/org/apache/lucene/lucene-memory/${LUCENE_VERSION}/lucene-memory-${LUCENE_VERSION}.jar"
    cd ..
    echo -e "${GREEN}✓ Lucene JARs downloaded${NC}"
fi

# Check huggingface-cli installation
echo -e "\n${YELLOW}[7/9] Checking huggingface-cli installation...${NC}"
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}huggingface-cli not found. Installing...${NC}"
    uv pip install huggingface_hub[cli]
    echo -e "${GREEN}✓ huggingface-cli installed${NC}"
else
    echo -e "${GREEN}✓ huggingface-cli is already installed${NC}"
fi

# Download test dataset (queries and answers)
echo -e "\n${YELLOW}[8/11] Downloading test dataset from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus" ]; then
    echo -e "${YELLOW}Test dataset already exists. Skipping download.${NC}"
else
    mkdir -p Tevatron
    echo -e "${YELLOW}Downloading Tevatron/browsecomp-plus (test queries and answers)...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus --repo-type=dataset --local-dir ./Tevatron/browsecomp-plus
    echo -e "${GREEN}✓ Test dataset downloaded${NC}"
fi

# Download corpus
echo -e "\n${YELLOW}[9/11] Downloading corpus from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus-corpus" ]; then
    echo -e "${YELLOW}Corpus already exists. Skipping download.${NC}"
else
    mkdir -p Tevatron
    echo -e "${YELLOW}Downloading Tevatron/browsecomp-plus-corpus...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus-corpus --repo-type=dataset --local-dir ./Tevatron/browsecomp-plus-corpus
    echo -e "${GREEN}✓ Corpus downloaded${NC}"
fi

# Download indexes
echo -e "\n${YELLOW}[10/12] Downloading BM25 index from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus-indexes/bm25" ]; then
    echo -e "${YELLOW}BM25 index already exists. Skipping.${NC}"
else
    mkdir -p Tevatron/browsecomp-plus-indexes
    echo -e "${YELLOW}Downloading BM25 index (~2.1GB)...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="bm25/*" --local-dir ./Tevatron/browsecomp-plus-indexes
    echo -e "${GREEN}✓ BM25 index downloaded${NC}"
fi

# Download Qwen3-Embedding-8B index
echo -e "\n${YELLOW}[11/12] Downloading Qwen3-Embedding-8B index from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b" ]; then
    echo -e "${YELLOW}Qwen3-Embedding-8B index already exists. Skipping.${NC}"
else
    mkdir -p Tevatron/browsecomp-plus-indexes
    echo -e "${YELLOW}Downloading Qwen3-Embedding-8B index (~1.6GB, this may take a while)...${NC}"
    huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-8b/*" --local-dir ./Tevatron/browsecomp-plus-indexes
    echo -e "${GREEN}✓ Qwen3-Embedding-8B index downloaded${NC}"
fi

# Download AgentIR-4B index
echo -e "\n${YELLOW}[12/12] Downloading AgentIR-4B index from Hugging Face...${NC}"
if [ -d "Tevatron/browsecomp-plus-indexes/agentir-4b/AgentIR_browsecomp-plus" ] && [ "$(ls Tevatron/browsecomp-plus-indexes/agentir-4b/AgentIR_browsecomp-plus/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
    echo -e "${YELLOW}AgentIR-4B index already exists. Skipping.${NC}"
else
    mkdir -p Tevatron/browsecomp-plus-indexes/agentir-4b
    echo -e "${YELLOW}Downloading Tevatron/AgentIR-indexes...${NC}"
    huggingface-cli download Tevatron/AgentIR-indexes --repo-type=dataset --local-dir ./Tevatron/browsecomp-plus-indexes/agentir-4b
    echo -e "${GREEN}✓ AgentIR-4B index downloaded${NC}"
fi

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}================================${NC}"
