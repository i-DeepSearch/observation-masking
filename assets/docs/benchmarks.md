# Evaluation Benchmarks

This document provides an overview of the benchmarks used for evaluation in this project.

## Available Benchmarks

| Benchmark | Dataset Key | Size | Language | Search Backend | Description |
|-----------|-------------|------|----------|----------------|-------------|
| **[BrowseComp-Plus](https://arxiv.org/abs/2508.06600)** | `browsecomp_plus` | 830 | EN | local | Deep-research benchmark isolating retriever and LLM agent effects |
| **[BrowseComp-ZH](https://arxiv.org/abs/2504.19314)** | `browsecomp-zh` | 289 | ZH | serper | Benchmarking web browsing ability of LLMs in Chinese |
| **[GAIA-text](https://arxiv.org/abs/2311.12983)** | `gaia` | 103 | EN | serper | Text-only subset of GAIA benchmark (dev split) |
| **[xbench-DeepSearch](https://huggingface.co/datasets/xbench/DeepSearch)** | `xbench` | 100 | ZH | serper | DeepSearch benchmark with encrypted test cases |


## Usage

To run evaluations on these benchmarks, use the dataset keys listed in the "Dataset Key" column:

```bash
# BrowseComp-Plus (requires local search service)
bash scripts/start_search_service.sh bm25 8000
# or: bash scripts/start_search_service.sh dense 8000 0
# or: bash scripts/start_search_service.sh agentir 8000 0
bash run_agent.sh results/bc 8001 2 browsecomp_plus local <model>

# Other listed benchmarks (using Serper API)
bash run_agent.sh results/browsecomp-zh 8001 2 browsecomp-zh serper <model>
bash run_agent.sh results/gaia 8001 2 gaia serper <model>
bash run_agent.sh results/xbench 8001 2 xbench serper <model>
```

## Search Backend Notes

- **local**: Required for BrowseComp-Plus. Uses local BM25, Dense, or AgentIR search service.
- **serper**: Uses Serper Google Search API. Requires `SERPER_API_KEY` in `.env`.
