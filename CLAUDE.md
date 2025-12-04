# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tetra V1** - Scientific knowledge graph agent for drug discovery hypothesis generation.

- **Language**: Python 3.10+ with `uv` package manager
- **Framework**: Google ADK (Agent Development Kit) + Gemini 2.0 Flash
- **Architecture**: Seven-phase pipeline: STRING expansion → PubMed mining → Relationship extraction → Graph merge → ML link prediction

## Quick Reference

### Build & Run
```bash
uv sync                                    # Install dependencies
uv run python main.py                      # Interactive CLI agent
uv run python main.py --query "..."        # Single query mode
uv run streamlit run frontend/app.py       # Streamlit UI (port 8501)
uv run adk api_server kg_agent/ --port 8080  # ADK API server
```

### Testing
```bash
uv run pytest tests/ -v                    # Run all tests
uv run pytest tests/ -v --asyncio-mode=auto  # With async support
uv run pytest tests/test_merge.py -v       # Single test file
uv run pytest tests/ --cov=. --cov-report=term-missing  # With coverage
```

### Pipeline & Scripts
```bash
./scripts/test_kg_pipeline.sh              # Full integration test
uv run python scripts/benchmark_miner_scale.py  # Benchmark batched miner
uv run python scripts/test_miner_provenance.py  # Test provenance validation
```

## Architecture

### Seven-Phase Pipeline
```
Phase 1: STRING Network Expansion (seed proteins → PPIs)
Phase 2: Query Construction Agent (LLM builds PubMed query)
Phase 3: PubMed Fetch + PubTator NER
Phase 4A: Co-occurrence graph (parallel, Python-only)
Phase 4B: LLM Relationship Mining (parallel, semaphore-limited)
Phase 5: Merge Agent (graph fusion + confidence scoring)
Phase 6: ML Link Prediction (Node2Vec + LogReg)
Phase 7: Summary + Cost Report
```

### ADK Tool Pattern

Tools use module-level globals for ADK compatibility:
```python
# In agent/adk_orchestrator.py or kg_agent/agent.py
_tools_instance: Optional[AgentTools] = None

def set_tools(tools: AgentTools) -> None:
    global _tools_instance
    _tools_instance = tools

# Tool functions discovered by ADK via reflection
async def get_string_network(seed_proteins: list[str], min_score: int = 700) -> dict:
    """Docstring becomes schema for ADK agent."""
    return await get_tools().get_string_network(seed_proteins, min_score)
```

## Module Structure

```
tetra_v1/
├── main.py                    # CLI entry point
├── agent/
│   ├── adk_orchestrator.py    # ADK LlmAgent wrapper + tool functions
│   ├── tools.py               # AgentTools class (clients wrapper)
│   └── graph_agent.py         # ReAct-based GraphRAG Q&A
├── pipeline/                  # Seven-phase components
│   ├── config.py, metrics.py, merge.py, ...
├── clients/                   # Async API clients (httpx)
│   ├── string_client.py, pubmed_client.py
├── extraction/                # Relationship extraction
│   ├── batched_litellm_miner.py  # Main extractor
│   ├── config.toml            # Schema + prompts
├── models/knowledge_graph.py  # NetworkX graph + algorithms
├── ml/link_predictor.py       # Node2Vec + LogReg
├── kg_agent/agent.py          # ADK-compatible agent module
├── frontend/app.py            # Streamlit UI
└── scripts/                   # Utility scripts
```

## Critical Patterns

### 0. Reusable Scripts
Use re-usable & parameterized scripts. If they do not already exist, write them or modify existing ones. Do not write inline python on the command line unless doing ad hoc data processing (e.g., parsing JSON from an API call).

### 1. Async/Await Everywhere
All I/O operations must be non-blocking:
```python
# Correct
async with httpx.AsyncClient() as client:
    response = await client.get(url)

# Never use
requests.get(url)  # Blocks entire pipeline
time.sleep(1)      # Use asyncio.sleep()
```

### 2. Entity Normalization
Normalize IDs before graph operations:
```python
entity_id.upper().replace("-", "")  # "hcrtr-1" → "HCRTR1"
```

### 3. Token Tracking
Every LLM call tracks tokens for cost accounting via `TokenUsage` dataclass.

### 4. Pydantic Models
All data structures use Pydantic v2 with validation.

## Configuration

**Environment variables** (`.env`):
- `GOOGLE_API_KEY` - Required for Gemini
- `NCBI_API_KEY` - Optional (higher PubMed rate limit)
- `OPENROUTER_API_KEY` - For Cerebras/OpenRouter extractors
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` - For tracing

**TOML config** (`extraction/config.toml`): Schema and prompt definitions.

## Key Files to Understand

- `agent/tools.py` - All tool implementations (STRING, PubMed, extraction, graph building)
- `extraction/batched_litellm_miner.py` - Batched relationship extraction with provenance
- `models/knowledge_graph.py` - Graph operations and algorithms
- `kg_agent/agent.py` - ADK agent wrapper for API server

## Troubleshooting

**"Tools not initialized"**: Call `set_tools(tools)` before agent.run()

**Slow PubMed**: Set `NCBI_API_KEY` for higher rate limits

**High LLM costs**: Check `token_breakdown` in PipelineReport, reduce `pubmed_max_results`

**No edges after merge**: Check confidence thresholds (default 0.4)
