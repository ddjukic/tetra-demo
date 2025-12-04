# Scientific Knowledge Graph Agent - Design Document

> **Version:** 1.0
> **Last Updated:** 2025-12-02
> **Purpose:** Drug discovery hypothesis generation through graph ML and LLM synthesis

---

## 1. System Overview

A tool that enables drug discovery scientists to rapidly explore biological systems, synthesize literature evidence, and generate testable hypotheses for novel protein interactions.

### 1.1 Core Value Proposition

| Capability | How It Works |
|------------|--------------|
| **Novel Interaction Prediction** | Node2Vec embeddings + LogReg trained on STRING physical interactions |
| **Evidence Synthesis** | PubMed/PubTator NER + LLM relationship extraction |
| **Hypothesis Generation** | LLM infers relationship types for ML-predicted edges using graph context |
| **Interactive Exploration** | Single orchestrator agent with specialized tools |

### 1.2 High-Level Architecture

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                        │
│   (OpenAI function-calling with system prompt + tools)       │
└─────────────────────────────────────────────────────────────┘
    │
    ├─────────────────┬─────────────────┬─────────────────┐
    ▼                 ▼                 ▼                 ▼
┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌───────────┐
│ STRING  │   │   PUBMED    │   │    GRAPH    │   │    ML     │
│ TOOLS   │   │   TOOLS     │   │    TOOLS    │   │  PREDICT  │
└─────────┘   └─────────────┘   └─────────────┘   └───────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              PRE-TRAINED LINK PREDICTOR                      │
│   Node2Vec embeddings (128d) + LogReg edge classifier        │
│   Trained once on STRING physical interactions (~600K edges) │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Design Decisions

### 2.1 Decision: Node2Vec + Logistic Regression over GNN

**Context:** Multiple approaches exist for link prediction (GCN, GraphSAGE, TransE, etc.)

**Decision:** Use Node2Vec (random walk embeddings) + Logistic Regression

**Rationale:**
| Factor | Node2Vec | GCN/GraphSAGE |
|--------|----------|---------------|
| **Training complexity** | Simple, CPU-friendly | Requires GPU, batching |
| **No node features needed** | Works with pure structure | Needs node features |
| **Interpretability** | Embeddings are vectors, easy to inspect | Black box |
| **STRING graph is sparse** | Random walks handle sparsity well | Message passing struggles |
| **Demo suitability** | Train in 30-60 min on CPU | Hours, GPU required |

**Expected Performance:** 0.80-0.85 AUC on held-out STRING edges (based on literature)

### 2.2 Decision: Single Orchestrator Agent (Not Multi-Agent)

**Context:** Could use multi-agent architecture (query agent → algorithm agent → synthesis agent)

**Decision:** Single LLM agent with function-calling tools

**Rationale:**
- Simpler to debug and maintain
- Function calling provides structured algorithm dispatch
- Conversation memory is straightforward
- Sufficient for query complexity in demo scope
- Avoids agent-to-agent communication overhead

### 2.3 Decision: OpenAI API (Not Google ADK)

**Context:** v0 used Google ADK. User wants fast LLM providers.

**Decision:** Use OpenAI Python SDK with async client

**Rationale:**
- Faster inference (GPT-4o-mini, GPT-4o)
- Excellent function-calling support
- Structured output with Pydantic
- Simpler than ADK for single-agent use case
- Can switch to other providers (OpenRouter) easily

### 2.4 Decision: In-Memory Graph + Pickle Persistence

**Context:** Could use Neo4j, SQLite, or pure in-memory

**Decision:** NetworkX MultiDiGraph with pickle/JSON persistence

**Rationale:**
- Transparent algorithm implementation
- Sufficient for demo scale (~10K nodes)
- No external dependencies
- Easy to inspect and debug
- Migration path to Neo4j documented if needed

### 2.5 Decision: Async HTTP Clients Throughout

**Context:** Need to call STRING API, PubMed API, OpenAI API

**Decision:** Use `httpx.AsyncClient` for all external calls

**Rationale:**
- Parallel API calls via `asyncio.gather()`
- Consistent async pattern
- Connection pooling
- Timeout handling

---

## 3. Component Design

### 3.1 Link Predictor (`ml/link_predictor.py`)

**Purpose:** Pre-trained ML model for predicting novel protein interactions

**Training Data:**
- STRING physical interactions (human, score >= 700)
- ~600K high-confidence edges
- ~20K proteins

**Architecture:**
```
STRING Graph (NetworkX)
        │
        ▼
    Node2Vec
    (128d embeddings, 80 walk length, 10 walks/node)
        │
        ▼
    Edge Features
    (Hadamard product: emb1 * emb2)
        │
        ▼
    Logistic Regression
    (binary: edge exists or not)
        │
        ▼
    Probability Score [0, 1]
```

**Negative Sampling:** Random non-edges (same count as positive edges)

**Evaluation:** Train/test split (80/20), report AUC and Average Precision

**Persistence:** Pickle file containing:
- `embeddings`: Dict[str, np.ndarray]
- `classifier`: LogisticRegression
- `protein_to_gene`: Dict[str, str]
- `gene_to_protein`: Dict[str, str]

### 3.2 Knowledge Graph (`models/knowledge_graph.py`)

**Purpose:** Evidence-backed graph structure

**Node Types:**
- Proteins/Genes (primary)
- Diseases, Chemicals (from PubTator)

**Edge Schema:**
```python
{
    "source": str,           # Gene symbol
    "target": str,           # Gene symbol
    "relation_type": str,    # "activates", "inhibits", "binds_to", etc.
    "evidence": [            # List of evidence items
        {
            "source_type": str,    # "literature", "string", "ml_predicted"
            "source_id": str,      # PMID, "STRING", etc.
            "confidence": float,
            "text_snippet": str    # Optional
        }
    ],
    "ml_score": float,       # Link predictor probability (if applicable)
    "inferred_type": str,    # LLM-inferred relationship type
    "reasoning": str         # LLM reasoning for inference
}
```

**Relationship Types:**
- `activates`, `inhibits`, `regulates`
- `binds_to`, `interacts_with`
- `associated_with`, `cooccurs_with`
- `hypothesized` (ML-predicted, no literature)

### 3.3 STRING Client (`clients/string_client.py`)

**Purpose:** Fetch protein interaction data from STRING API

**Endpoints Used:**
| Endpoint | Purpose |
|----------|---------|
| `/api/json/network` | Get interaction network for seed proteins |
| `/api/json/interaction_partners` | Get neighbors of proteins |
| `/api/json/functional_annotation` | Get GO/KEGG annotations |

**Rate Limiting:** Caller identity required, reasonable request rate

### 3.4 PubMed Client (`clients/pubmed_client.py`)

**Purpose:** Search literature and get NER annotations

**APIs Used:**
| API | Purpose |
|-----|---------|
| E-utilities esearch | Search PubMed, get PMIDs |
| E-utilities efetch | Fetch article details (title, abstract) |
| PubTator3 API | Pre-computed NER (genes, diseases, chemicals) |

**Batching:**
- esearch: up to 100 results per call
- efetch: up to 200 PMIDs per call
- PubTator: batches of PMIDs

### 3.5 Relationship Extractor (`extraction/relationship_extractor.py`)

**Purpose:** Extract typed relationships from abstracts using LLM

**Prompt Strategy:**
1. Pass abstract text
2. Pass co-occurring entity pairs (from PubTator)
3. Ask for relationship classification

**Output Schema:**
```python
[
    {
        "entity1": str,
        "entity2": str,
        "relationship": str,  # One of valid types
        "confidence": float,  # 0-1
        "evidence_text": str  # Supporting quote
    }
]
```

**Concurrency:** Semaphore-controlled (5 concurrent calls)

### 3.6 Relationship Inferrer (`extraction/relationship_inferrer.py`)

**Purpose:** Infer relationship types for ML-predicted edges

**When Used:** For high-score predictions with no literature evidence

**Input:**
- Protein pair
- ML score
- Graph neighborhood context (known interactions)
- Functional annotations

**Output:**
```python
{
    "hypothesized_relationship": str,
    "confidence": str,  # "LOW", "MEDIUM", "HIGH"
    "reasoning": str,
    "validation_experiments": [str]
}
```

### 3.7 Agent Tools (`agent/tools.py`)

**12 Tools Available:**

| Category | Tool | Purpose |
|----------|------|---------|
| **STRING** | `get_string_network` | Fetch known interactions |
| **STRING** | `get_string_partners` | Get interaction partners |
| **PubMed** | `search_literature` | Search PubMed |
| **PubMed** | `get_entity_annotations` | Get PubTator NER |
| **Extract** | `extract_relationships` | LLM extraction from abstracts |
| **Graph** | `build_knowledge_graph` | Construct graph from data |
| **Graph** | `get_graph_summary` | Graph statistics |
| **Graph** | `get_protein_neighborhood` | Local graph view |
| **ML** | `predict_novel_links` | Run link predictor |
| **ML** | `infer_novel_relationships` | LLM inference for predictions |
| **Query** | `query_evidence` | Get evidence for edge |
| **Query** | `find_path` | Shortest path between proteins |

### 3.8 Orchestrator Agent (`agent/orchestrator.py`)

**Purpose:** Main conversational agent

**System Prompt Focus:**
1. Interpret user query intent
2. Plan tool execution sequence
3. Call tools as needed
4. Synthesize results into actionable insights

**Typical Flow (Exploration Query):**
```
1. get_string_network(seed_proteins)
2. search_literature(query)
3. get_entity_annotations(pmids)
4. extract_relationships(articles, annotations)
5. build_knowledge_graph(...)
6. predict_novel_links()
7. infer_novel_relationships(top_predictions)
8. Synthesize findings
```

---

## 4. Data Flow

### 4.1 Training Pipeline (Offline, One-Time)

```
STRING Downloads
├── 9606.protein.physical.links.detailed.v12.0.txt.gz
├── 9606.protein.info.v12.0.txt.gz
└── 9606.protein.aliases.v12.0.txt.gz
        │
        ▼
┌─────────────────────────────────────────┐
│         Load & Filter (score >= 700)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│         NetworkX Graph (~20K nodes)     │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│    Node2Vec Training (30-60 min CPU)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│    Logistic Regression + Evaluation     │
└─────────────────────────────────────────┘
        │
        ▼
    link_predictor.pkl
```

### 4.2 Runtime Pipeline (Per Query)

```
User Query: "Explore the orexin system"
        │
        ▼
┌─────────────────────────────────────────┐
│           Orchestrator Agent            │
│   1. Identify seed proteins (HCRTR1,    │
│      HCRTR2, HCRT)                      │
│   2. Call STRING API                    │
│   3. Search PubMed                      │
│   4. Extract relationships              │
│   5. Build graph                        │
│   6. Run link prediction                │
│   7. Infer novel relationships          │
│   8. Synthesize hypothesis              │
└─────────────────────────────────────────┘
        │
        ▼
    Evidence-Backed Hypothesis
    + Validation Suggestions
```

---

## 5. Project Structure

```
tetra_v1/
├── DESIGN.md                    # This document
├── pyproject.toml               # UV dependencies
├── .env.example                 # Environment variables template
│
├── data/
│   └── string/                  # Downloaded STRING files
│       ├── 9606.protein.physical.links.detailed.v12.0.txt.gz
│       ├── 9606.protein.info.v12.0.txt.gz
│       └── 9606.protein.aliases.v12.0.txt.gz
│
├── models/
│   ├── __init__.py
│   ├── knowledge_graph.py       # KG data structures
│   └── link_predictor.pkl       # Trained model (generated)
│
├── ml/
│   ├── __init__.py
│   └── link_predictor.py        # Node2Vec + LogReg
│
├── clients/
│   ├── __init__.py
│   ├── string_client.py         # STRING API
│   └── pubmed_client.py         # PubMed + PubTator
│
├── extraction/
│   ├── __init__.py
│   ├── relationship_extractor.py
│   └── relationship_inferrer.py
│
├── agent/
│   ├── __init__.py
│   ├── tools.py                 # Tool implementations
│   └── orchestrator.py          # Main agent
│
├── scripts/
│   ├── download_string.py       # Download STRING data
│   └── train_link_predictor.py  # Training script
│
├── frontend/
│   └── app.py                   # Streamlit UI
│
├── tests/
│   ├── test_link_predictor.py
│   ├── test_clients.py
│   └── test_extraction.py
│
├── main.py                      # CLI entry point
├── demo.py                      # Demo script
├── Dockerfile
└── docker-compose.yaml
```

---

## 6. Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Graph Embeddings** | node2vec | Standard, CPU-friendly |
| **ML Classifier** | scikit-learn | Simple, interpretable |
| **Graph Storage** | NetworkX | Transparent, sufficient scale |
| **LLM** | OpenAI (gpt-4o-mini) | Fast, good function calling |
| **HTTP Client** | httpx | Async, connection pooling |
| **API Framework** | FastAPI | Optional, for Streamlit backend |
| **Frontend** | Streamlit | Rapid prototyping |
| **Visualization** | PyVis | Interactive graphs |

---

## 7. Configuration

### 7.1 Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - Chunking
PUBMED_MAX_RESULTS=50
MINING_MAX_CONCURRENT=5

# Optional - ML
MIN_STRING_SCORE=700
NODE2VEC_DIMENSIONS=128
NODE2VEC_WALK_LENGTH=80
NODE2VEC_NUM_WALKS=10
```

### 7.2 Link Predictor Config

```python
LinkPredictorConfig(
    embedding_dim=128,
    walk_length=80,
    num_walks=10,
    p=1.0,  # Return parameter
    q=1.0,  # In-out parameter
    min_score=700,  # STRING score threshold
)
```

---

## 8. Expected Outcomes

### 8.1 Demo Query

**Input:** "Explore the orexin system for potential novel drug targets"

**Expected Output:**
```
Based on analysis of the orexin signaling system:

**Known Interactions (STRING):**
- HCRTR1 interacts with HCRTR2 (score: 0.92)
- HCRT binds to HCRTR1, HCRTR2 (score: 0.95)
- G-protein coupling: GNAI1, GNAQ (score: 0.88)

**Literature Evidence (12 papers):**
- HCRTR2 activates NTRK1 signaling (PMID: 12345678)
- Orexin system regulates sleep-wake cycle (PMID: 87654321)

**Novel Predictions (ML score > 0.7):**
1. HCRTR2 - NTRK1: score 0.87 (no direct literature)
   Hypothesized: regulatory relationship
   Reasoning: Both involved in neuronal signaling,
   share 3 common neighbors
   Validation: co-IP assay, proximity ligation

2. HCRT - BDNF: score 0.82 (1 indirect paper)
   Hypothesized: associated_with
   Validation: expression correlation, knockdown study

**Recommended Next Steps:**
1. Validate HCRTR2-NTRK1 interaction experimentally
2. Investigate BDNF as downstream effector
3. Consider dual-target approach for narcolepsy
```

### 8.2 Performance Targets

| Metric | Target |
|--------|--------|
| Link predictor AUC | > 0.80 |
| Query response time | < 30s |
| Concurrent users | ~10 (demo) |
| Graph size | ~1K-10K nodes |

---

## 9. Simplifications for Demo

1. **No persistent sessions** - Each conversation is stateless
2. **No user authentication** - Single-user demo
3. **No incremental updates** - Full graph rebuild per session
4. **Limited error recovery** - Fail fast with clear errors
5. **CPU training only** - Node2Vec on CPU is sufficient
6. **No caching** - Fresh API calls each time

---

## 10. Batched Extraction & Provenance Validation

### 10.1 Architecture

The batched extraction pipeline (`extraction/batched_litellm_miner.py`) processes multiple abstracts per LLM call for efficiency:

```
Articles + Annotations
        │
        ▼
┌─────────────────────────────────────────┐
│     Token-Aware Chunking (~5K tokens)   │
│     - Greedy bin-packing                │
│     - Min 3 chunks for parallelization  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│     Parallel LLM Extraction             │
│     - [PMID: X] markers in prompt       │
│     - Structured JSON output            │
│     - Entity list (not pairs)           │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│     Provenance Validation               │
│     - PMID attribution check            │
│     - Evidence sentence matching        │
└─────────────────────────────────────────┘
```

### 10.2 Provenance Validation

Each extracted relationship undergoes two validation checks:

| Check | Method | Threshold |
|-------|--------|-----------|
| **PMID Attribution** | Verify PMID exists in chunk's PMID list | Exact match |
| **Evidence Matching** | Fuzzy match evidence text against abstract | 0.7 similarity |

**Validation Result Fields:**
- `pmid_valid`: Boolean - PMID attribution correct
- `evidence_valid`: Boolean - Evidence found in abstract
- `evidence_similarity`: Float 0.0-1.0 - Match quality

### 10.3 Known Issue: Evidence Paraphrasing

**Problem:** LLMs sometimes paraphrase or summarize evidence rather than quoting verbatim, even when explicitly instructed to provide exact quotes.

**Observed Behavior - Context Scaling Study:**

| Scale | Abstracts/Chunk | PMID Accuracy | Evidence Verbatim | Paraphrased |
|-------|-----------------|---------------|-------------------|-------------|
| Small (15 abs) | ~5 | 100% | 87.8% | 12.2% |
| Large (100 abs) Run 1 | ~17 | 100% | 68.4% | 31.6% |
| Large (100 abs) Run 2 | ~17 | 100% | 53.9% | 46.1% |
| Large (100 abs) Run 3 | ~17 | 100% | 63.6% | 36.4% |
| **Large Average** | ~17 | **100%** | **62.0%** | **38.0%** |

**Key Finding:** Evidence paraphrasing increases ~3x (12% → 38%) when chunk size increases from ~5 to ~17 abstracts, while PMID attribution remains perfect.

**Example Failures:**
```
Evidence returned: "TAK-861 ... activates OX2R with a half-maximal e..."
Similarity score: 0.44 (below 0.7 threshold)
Actual text was paraphrased/summarized by model
```

**Impact:**
- **Attribution is reliable** - PMID linkage can be trusted at any scale
- **Evidence quality degrades with context size** - More abstracts per chunk → more paraphrasing
- **Chunk size is an optimization parameter** - Smaller chunks improve verbatim extraction

**Potential Mitigations (Not Yet Implemented):**
1. **Lower threshold** - Accept 0.5+ similarity for "good enough" evidence
2. **Two-pass extraction** - First extract relationships, then retrieve exact quotes
3. **Sentence-level retrieval** - Use embedding similarity to find best matching sentence
4. **Stronger prompting** - XML tags, examples of verbatim extraction
5. **Post-hoc correction agent** - Second LLM pass to fix paraphrased evidence

**Current Recommendation:**
- Use `evidence_similarity` score to rank confidence
- Flag low-similarity evidence for manual review
- Trust PMID attribution; treat evidence as "supporting context" not "exact quote"

### 10.4 Chunk Size Optimization

Based on the scaling study finding that smaller chunks improve verbatim extraction, we tested different chunk sizes:

| Chunk Size | Tokens | Abstracts/Chunk | Chunks | Validation Rate | Evidence Failures |
|------------|--------|-----------------|--------|-----------------|-------------------|
| Large (default) | 5000 | ~15-17 | 6 | 62.0% | ~37 |
| Medium | 3200 | ~10-12 | 9 | 71.4% | 40 |
| Small | 2500 | ~8-9 | 11 | **75.5%** | 34 |

**Key Findings:**
1. **Smaller chunks = higher validation rate** (75.5% at 2500 vs 62% at 5000 tokens)
2. **More relationships extracted** with smaller chunks (139-140 vs 98 avg)
3. **PMID attribution remains 100% accurate** across all chunk sizes
4. **Trade-off: More API calls** with smaller chunks (11 vs 6 chunks)

**Evidence Statistics Per Abstract (100 abstracts):**

| Chunk Size | Rels per PMID | Evidence per PMID |
|------------|---------------|-------------------|
| 2500 tokens | 2.04 ± 1.49 | 1.62 ± 0.90 |
| 3200 tokens | 2.06 ± 1.43 | 1.63 ± 1.24 |

**Recommended Settings:**
- Use `--chunk-tokens 2500` for highest validation rate
- Wall clock time is similar (~7s) due to more parallel chunks
- Override via `BatchedLiteLLMMiner(chunk_tokens=2500)`

### 10.5 Sentence-Index Solution

**The Breakthrough:** Instead of asking the LLM to extract/quote evidence text (which causes paraphrasing), have it reference sentence indices and extract verbatim programmatically.

**How It Works:**

```
1. Pre-processing: Number sentences in each abstract
   "HCRTR2 activates PKC. This leads to..."
   → "[1] HCRTR2 activates PKC. [2] This leads to..."

2. LLM Output: Sentence indices instead of text
   {
     "entity1": "HCRTR2",
     "entity2": "PKC",
     "relationship": "ACTIVATES",
     "evidence_sentence_indices": [1],  // ← Indices, not text
     "pmid": "12345678"
   }

3. Post-processing: Extract verbatim by index
   evidence_text = sentences[1-1]  // → "HCRTR2 activates PKC."
```

**Results Comparison (100 abstracts):**

| Approach | Validation Rate | PMID Accuracy | Evidence Failures |
|----------|-----------------|---------------|-------------------|
| Text extraction (old) | 62.0% | 100% | ~37 |
| Sentence indices (new) | **97.7%** | 100% | 2 |

**Improvement:** +35.7 percentage points (62% → 97.7%)

**Why It Works:**
- LLM just outputs integers (e.g., `[1, 2]`), no text reproduction
- Eliminates paraphrasing entirely
- Reduces output tokens
- Validation becomes simple bounds checking
- 100% verbatim evidence by construction

**Failure Mode (2.3% Loss):**
The only failures are **hallucinated sentence indices** - model cites indices that don't exist:
```
PMID 38295907: Model cited [2, 6, 8] but abstract only has 4 sentences
- 2 relationships failed (same PMID, same invalid indices)
- Root cause: Index hallucination, not paraphrasing
- Easily detectable via bounds checking
```

This is a fundamentally different (and rarer) failure mode than paraphrasing - the model occasionally "sees" more sentences than exist, likely due to context mixing across abstracts in the chunk.

**Implementation Files:**
- `extraction/config.toml` - Updated prompts with numbered sentence format
- `extraction/config_loader.py` - Schema uses `evidence_sentence_indices: [int]`
- `extraction/batched_litellm_miner.py` - `number_sentences()`, index-based validation

### 10.6 Configuration

Batched extraction settings in `extraction/config.toml`:

```toml
[BATCHED]
TARGET_TOKENS_PER_CHUNK = 5000
MIN_CHUNKS = 3
MAX_CONCURRENT = 5
MAX_RETRIES = 3
RETRY_DELAY_MS = 1000
MIN_CONFIDENCE = 0.5
MAX_TOKENS = 8192  # Higher for multi-abstract output
```

---

## 11. Google ADK Integration & Optimizations

### 11.1 Architecture Decision: Google ADK over OpenAI API

**Context:** Original design used OpenAI Python SDK. Decision made to switch to Google ADK (Agent Development Kit) for better Gemini integration.

**Decision:** Use Google ADK with `LlmAgent` and `PlanReActPlanner`

**Rationale:**
| Factor | OpenAI SDK | Google ADK |
|--------|-----------|------------|
| **Model access** | GPT-4 only | Native Gemini 2.5 Flash/Pro |
| **Agent framework** | Manual tool orchestration | Built-in planning + reasoning |
| **Tool execution** | Custom implementation | Native function tool support |
| **Session management** | Manual | `InMemorySessionService` |

**Key Components:**
- `LlmAgent` - Core agent with instruction and tools
- `PlanReActPlanner` - Planning and reasoning strategy
- `Runner` - Async execution with event streaming
- `InMemorySessionService` - Conversation session management

### 11.2 ADK Response Accumulation Fix

**Problem:** ADK spreads response text across multiple events, not just the final one. Naive implementation only captured final event, resulting in empty responses.

**Symptom:** "Agent did not produce a final response" for complex multi-tool queries.

**Root Cause:**
```python
# WRONG: Only looking at final event
if event.is_final_response():
    if event.content and event.content.parts:
        # Final event often has empty parts!
        ...
```

**Solution:** Accumulate text from ALL events during iteration:
```python
accumulated_text = []

async for event in self.runner.run_async(...):
    # Accumulate text from ALL events
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, 'text') and part.text and part.text.strip():
                accumulated_text.append(part.text)

    if event.is_final_response():
        final_response = "\n".join(accumulated_text)
        break
```

**File:** `agent/adk_orchestrator.py:403-432`

### 11.3 Extraction Performance Optimization

**Problem:** Initial pipeline used `RelationshipExtractor` (sync Gemini SDK with `run_in_executor`), taking ~7s per article.

**Impact:**
- 5 articles: 70.3s (including agent overhead)
- 10 articles: Would be ~100-140s

**Solution:** Replace with `BatchedLiteLLMMiner`:
- Token-aware chunking for batch processing
- LiteLLM multi-provider support
- Sentence-index based evidence extraction (97.7% validation)
- Parallel chunk processing

**Performance After Fix:**
| Scale | Old Time | New Time | Improvement |
|-------|----------|----------|-------------|
| 5 articles | 70.3s | ~28s | 2.5x faster |
| 10 articles | ~140s | 56.5s | 2.5x faster |

**Files Changed:**
- `agent/tools.py:30-54` - Updated imports and constructor
- `agent/tools.py:624-667` - Simplified `extract_relationships()` to use miner
- `main.py:34,172,181` - Use `create_batched_miner()` factory

### 11.4 Updated Tool Architecture

**Before (Slow):**
```
AgentTools.__init__(
    relationship_extractor: RelationshipExtractor  # sync Gemini, 7s/article
)
```

**After (Fast):**
```
AgentTools.__init__(
    relationship_miner: BatchedLiteLLMMiner  # async LiteLLM, batched
)
```

**Key Difference:** `extract_relationships()` now delegates to `relationship_miner.run()` which:
1. Chunks abstracts by token count
2. Processes chunks in parallel (semaphore-controlled)
3. Returns validated relationships with provenance

### 11.5 Planner Removal Optimization

**Problem:** `PlanReActPlanner` adds multiple LLM reasoning cycles per tool call.

**Solution:** Remove planner entirely - let the model do native function calling.

```python
# Before (slow)
agent = LlmAgent(..., planner=PlanReActPlanner())

# After (faster)
agent = LlmAgent(...)  # No planner
```

**Result:** Simple STRING query dropped from ~15s to ~4.7s.

### 11.6 Critical Bottleneck: LLM Context Processing

**Discovery:** The real bottleneck is NOT tool execution but LLM processing of tool outputs.

**Server Log Evidence (10 articles pipeline):**
```
21:54:38,918 - LLM call sent (after extraction tool completed)
21:56:18,806 - Response received (100 seconds later!)
```

**Root Cause:** ADK passes ALL tool output through the orchestrating LLM. When `extract_relationships` returns 50+ relationships with evidence sentences, the model takes **100+ seconds** to process that context.

**Timeline Breakdown:**
| Component | Duration | Notes |
|-----------|----------|-------|
| STRING network | ~0.5s | API call |
| PubMed search | ~1.5s | API call |
| PubTator annotations | ~0.5s | API call |
| Relationship extraction | ~8-10s | Batched LLM (fast!) |
| **LLM context processing** | **~100s** | **BOTTLENECK** |
| Total | ~107s | |

**Key Insight:** The extraction tools are fast. The problem is passing large JSON payloads through the orchestrating LLM for "reasoning".

### 11.7 Architecture Problem

Current flow (slow):
```
User Query → LLM → Tool Call → [Tool executes, returns JSON]
                             → LLM processes entire JSON (~100s)
                             → LLM decides next tool
                             → ... repeat
```

The LLM sees and "thinks about" every relationship, every PMID, every evidence sentence. This is wasteful - the LLM doesn't need to see raw data to orchestrate.

**Solution Direction:** State-based data flow where LLM only sees summaries.

### 11.6 Configuration

**ADK Orchestrator Settings:**
```python
ADKOrchestrator(
    tools=tools,
    model="gemini-2.5-flash",  # Fast, cost-effective
    app_name="tetra_kg_agent",
)
```

**Batched Miner Settings (via `main.py`):**
```python
relationship_miner = create_batched_miner(
    extractor_name="gemini",  # Uses gemini/gemini-2.5-flash via LiteLLM
)
```

### 11.8 Next Step: State-Based Data Flow

**Solution direction:** Use ADK's `ToolContext.state` to pass data between tools without LLM mediation. Tools write full data to state, return only summaries to LLM.

**Status:** In progress - validating approach.

---

## 12. Future Enhancements (Post-Demo)

1. **Entity Resolution** - UMLS/UniProt canonical IDs
2. **GNN Upgrade** - GraphSAGE for better predictions
3. **Neo4j Migration** - For larger graphs
4. **Multi-model Routing** - Cheaper models for simple queries
5. **Streaming Responses** - Real-time agent output
6. **Validation Framework** - Silver standard evaluation

---

*Document maintained in `/tetra_v1/DESIGN.md`*
