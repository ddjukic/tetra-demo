# Tetra V1 Architecture - Multi-Agent Pipeline

## Overview

This document describes the multi-agent pipeline architecture for building and querying scientific knowledge graphs. The system uses Google ADK for agent orchestration with Langfuse for observability.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT: Seed Terms                              │
│                    (e.g., "orexin", "HCRTR1", "HCRTR2")                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: STRING NETWORK EXPANSION                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STRING Client                                                       │   │
│  │  - Fetch known PPIs for seed proteins                               │   │
│  │  - Get interaction partners (network expansion)                      │   │
│  │  - Min confidence: 700 (high confidence)                            │   │
│  │  - Output: Expanded entity list + STRING interactions               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Metrics: tokens=0, api_calls=N, latency_ms, entities_found               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: QUERY CONSTRUCTION AGENT                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LLM Agent (gemini-2.0-flash-exp)                                   │   │
│  │  - Input: Expanded entities from STRING                             │   │
│  │  - Task: Construct optimal PubMed query                             │   │
│  │  - Output: PubMed query string + query strategy explanation         │   │
│  │  - Uses MeSH terms, Boolean operators, date filters                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Metrics: prompt_tokens, completion_tokens, latency_ms                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: PUBMED FETCH + PUBTATOR NER                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PubMed Client                                                       │   │
│  │  - Search with constructed query                                     │   │
│  │  - Fetch abstracts (batched, single API call)                       │   │
│  │  - Get PubTator NER annotations (genes, diseases, chemicals)        │   │
│  │  - Max results: configurable (default 50)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Metrics: tokens=0, api_calls=2, papers_found, entities_annotated         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │      PARALLEL FAN-OUT          │
                     ▼                                ▼
┌────────────────────────────────────┐  ┌────────────────────────────────────┐
│   PHASE 4A: CO-OCCURRENCE GRAPH    │  │   PHASE 4B: LLM RELATIONSHIP      │
│  ┌──────────────────────────────┐  │  │              MINING               │
│  │  Pure Python (Fast)          │  │  │  ┌──────────────────────────────┐ │
│  │  - Count entity co-mentions  │  │  │  │  LLM Agent (parallel calls)  │ │
│  │  - Weight by frequency       │  │  │  │  - Extract typed relations   │ │
│  │  - PMI scoring               │  │  │  │  - Semaphore: max_concurrent │ │
│  │  - Output: co-occurrence     │  │  │  │  - Retry with exp backoff    │ │
│  │    edges with weights        │  │  │  │  - Output: relationships     │ │
│  └──────────────────────────────┘  │  │  │    with evidence + type      │ │
│  Metrics: latency_ms only         │  │  │  └──────────────────────────────┘ │
└────────────────────────────────────┘  │  Metrics: prompt_tokens,          │
                     │                   │  completion_tokens per chunk,     │
                     │                   │  total_cost, retries_used         │
                     │                   └────────────────────────────────────┘
                     │                                │
                     └────────────────┬───────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 5: MERGE AGENT                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Graph Merge (Thread-Safe)                                          │   │
│  │  - Combine STRING interactions                                       │   │
│  │  - Merge co-occurrence edges                                         │   │
│  │  - Add LLM-extracted relationships                                   │   │
│  │  - Deduplicate with evidence aggregation                            │   │
│  │  - Calculate confidence scores                                       │   │
│  │  - Output: Unified KnowledgeGraph                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Metrics: nodes_merged, edges_merged, duplicates_resolved                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 6: ML LINK PREDICTION                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LinkPredictor (Trained Model)                                       │   │
│  │  - Extract graph features                                            │   │
│  │  - Predict novel interactions                                        │   │
│  │  - Score: 0-1 probability                                           │   │
│  │  - Filter by min_ml_score threshold                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Metrics: predictions_made, avg_score, latency_ms                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 7: NOTIFICATION + SUMMARY                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Summary Agent                                                       │   │
│  │  - Generate human-readable summary                                   │   │
│  │  - Key findings: top entities, novel predictions, communities       │   │
│  │  - Token usage breakdown                                             │   │
│  │  - Cost estimation                                                   │   │
│  │  - Pipeline timing report                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Output: PipelineReport to user                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAPHRAG Q&A AGENT (ReAct)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LlmAgent with PlanReActPlanner                                      │   │
│  │  Tools:                                                              │   │
│  │    - get_graph_summary()                                             │   │
│  │    - query_evidence(p1, p2)                                          │   │
│  │    - find_path(source, target)                                       │   │
│  │    - compute_centrality(method)                                      │   │
│  │    - detect_communities()                                            │   │
│  │    - run_diamond(seeds)                                              │   │
│  │    - calculate_proximity(drug, disease)                              │   │
│  │    - predict_synergy(t1, t2, disease)                                │   │
│  │    - get_predictions(min_score)                                      │   │
│  │    - generate_hypothesis(p1, p2)                                     │   │
│  │  ReAct: Thought → Action → Observation loop                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Conversation history preserved via InMemorySessionService                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. STRING FIRST

STRING provides a curated, high-confidence protein interaction network. By starting with STRING:
- We get a **theoretical framework** of known biology
- STRING entities serve as the **NER vocabulary** for PubMed extraction
- Ensures we focus on biologically relevant proteins

### 2. Sequential + Parallel Hybrid

The pipeline is **sequential at the phase level** but **parallel within phases**:
- Phases must complete in order (STRING → Query → PubMed → Extract → Merge)
- Relationship mining is parallelized with `asyncio.Semaphore` (max 5 concurrent)
- Co-occurrence and LLM extraction run in parallel (Phase 4A || Phase 4B)

### 3. Token Tracking at Every Step

Every LLM call tracks:
```python
@dataclass
class TokenUsage:
    phase: str
    step: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    cost_usd: float  # Estimated
```

### 4. Langfuse Instrumentation

OpenTelemetry-based tracing via `GoogleADKInstrumentor`:
- Every tool call becomes a span
- LLM requests/responses captured
- Custom attributes for cost tracking
- Session-scoped traces for conversation continuity

### 5. ADK for Retries/Rate Limits

Built-in retry with exponential backoff:
```python
@dataclass
class MiningConfig:
    max_concurrent: int = 5
    max_retries: int = 3
    base_delay: float = 1.0
    retry_on_codes: tuple = (429, 503, 500, 502)
```

### 6. ReAct Planning for Q&A

The GraphRAG agent uses `PlanReActPlanner` for flexible reasoning:
- **Thought**: Reason about what information is needed
- **Action**: Call appropriate tool
- **Observation**: Process tool result
- **Repeat**: Until question is answered

## Data Models

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    # STRING
    string_min_score: int = 700
    string_max_partners: int = 30

    # PubMed
    pubmed_max_results: int = 50
    pubmed_date_filter: str | None = None  # e.g., "2020:2024"

    # Relationship Mining
    mining_max_concurrent: int = 5
    mining_max_retries: int = 3
    mining_model: str = "gemini-2.0-flash-exp"

    # ML Link Prediction
    ml_min_score: float = 0.7
    ml_max_predictions: int = 20

    # Observability
    langfuse_enabled: bool = True
    langfuse_session_id: str | None = None
```

### PipelineReport

```python
@dataclass
class PipelineReport:
    # Timing
    total_duration_s: float
    phase_timings: dict[str, float]

    # Token Usage
    total_tokens: int
    token_breakdown: dict[str, TokenUsage]
    estimated_cost_usd: float

    # Results
    nodes_created: int
    edges_created: int
    papers_processed: int
    relationships_extracted: int
    predictions_made: int

    # Key Findings
    top_entities: list[tuple[str, float]]  # (entity, centrality)
    novel_predictions: list[dict]
    communities: list[list[str]]
```

## Implementation Phases

### Phase 1: Langfuse Instrumentation Setup
- Copy `_setup_langfuse_tracing()` from tetra_v0
- Add `GoogleADKInstrumentor` initialization
- Create `observability/tracing.py` module

### Phase 2: Pipeline Infrastructure
- Create `pipeline/config.py` with `PipelineConfig`
- Create `pipeline/metrics.py` with `TokenUsage`, `PipelineReport`
- Add cost estimation functions

### Phase 3: STRING-First Pipeline
- Create `pipeline/string_expansion.py`
- Fetch STRING network for seed proteins
- Get interaction partners for network expansion

### Phase 4: Query Construction Agent
- Create `pipeline/query_agent.py`
- LLM agent that builds optimal PubMed query from STRING entities
- Output: validated PubMed query string

### Phase 5: Parallel Fan-Out
- Create `pipeline/parallel_extraction.py`
- Co-occurrence graph (fast, Python-only)
- LLM relationship mining (parallel with semaphore)
- Use `asyncio.gather()` for parallel execution

### Phase 6: Merge Agent + Graph Construction
- Create `pipeline/merge.py`
- Thread-safe graph merging
- Evidence aggregation
- Confidence scoring

### Phase 7: GraphRAG Q&A Agent
- Adapt from tetra_v0's `GraphAgentManager`
- Use `PlanReActPlanner` for flexible reasoning
- Tool access to all GDS algorithms

### Phase 8: Frontend Integration
- Replace "Copy Prompt" with "Build Knowledge Graph" button
- Real-time progress updates via Streamlit
- Results display with graph visualization

## File Structure

```
tetra_v1/
├── agent/
│   ├── adk_orchestrator.py      # Main ADK agent (existing)
│   ├── tools.py                 # Tool implementations (existing)
│   ├── query_agent.py           # NEW: Query construction agent
│   └── graph_agent.py           # NEW: GraphRAG Q&A agent (from v0)
├── pipeline/
│   ├── __init__.py
│   ├── config.py                # NEW: PipelineConfig
│   ├── metrics.py               # NEW: TokenUsage, PipelineReport
│   ├── string_expansion.py      # NEW: STRING-first logic
│   ├── parallel_extraction.py   # NEW: Parallel fan-out
│   └── merge.py                 # NEW: Graph merging
├── observability/
│   ├── __init__.py
│   └── tracing.py               # NEW: Langfuse setup
├── models/
│   └── knowledge_graph.py       # Existing + GDS algorithms
├── clients/
│   ├── pubmed_client.py         # Existing
│   └── string_client.py         # Existing
├── extraction/
│   ├── relationship_extractor.py # Existing
│   └── relationship_inferrer.py  # Existing
├── ml/
│   └── link_predictor.py        # Existing
└── frontend/
    └── app.py                   # Updated with Build button
```

## Cost Estimation

Based on Gemini 2.0 Flash pricing (input: $0.075/1M tokens, output: $0.30/1M tokens):

| Phase | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|-------|-------------------|--------------------|-----------|
| Query Construction | 2,000 | 500 | $0.0003 |
| Relationship Mining (50 papers) | 150,000 | 30,000 | $0.020 |
| Summary Generation | 5,000 | 2,000 | $0.001 |
| **Total per pipeline run** | ~157,000 | ~32,500 | **~$0.022** |

Plus Q&A queries: ~$0.001-0.005 per query depending on tool usage.
