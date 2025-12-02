# TODO - Tetra V1 Optimizations & Refinements

## 2024-12-02 22:58 UTC - Architecture Optimizations

### 1. Ensemble Node2Vec with Multiple p/q Parameters - COMPLETED

**Problem:** Currently only one node2vec model trained with a single set of p & q parameters.

**Status:** IMPLEMENTED (2025-12-03)

**Background:**
- `p` (return parameter): Controls likelihood of returning to previous node
  - High p = explore outward, less likely to return
  - Low p = allow backtracking, stay local
- `q` (in-out parameter): Controls BFS vs DFS behavior
  - q > 1: BFS-like, captures **homophily** (local neighborhood clustering)
  - q < 1: DFS-like, captures **structural equivalence** (global roles like hubs)

**Implementation:**
- Created `ml/ensemble_predictor.py` with `EnsembleLinkPredictor`
- Trained 2 additional node2vec models with different p/q parameters

**Training Results:**

| Model | p | q | Behavior | ROC-AUC | Avg Precision |
|-------|---|---|----------|---------|---------------|
| Structural | 1.0 | 0.5 | DFS-like, captures hub roles | 0.9866 | 0.9860 |
| Homophily | 1.0 | 2.0 | BFS-like, captures local clusters | 0.9890 | 0.9883 |
| Original | 1.0 | 1.0 | Balanced | 0.9888 | 0.9881 |

**Key Features:**
1. EnsembleLinkPredictor class with weighted/majority voting
2. estimate_graph_homophily() function for structural analysis
3. interpret_disagreement() for explaining model consensus/divergence
4. Automatic model loading from models/ directory

**Files Created:**
- `/Users/dejandukic/dejan_dev/tetra/tetra_v1/ml/ensemble_predictor.py`
- `/Users/dejandukic/dejan_dev/tetra/tetra_v1/models/link_predictor_structural.pkl`
- `/Users/dejandukic/dejan_dev/tetra/tetra_v1/models/link_predictor_homophily.pkl`

**Usage:**
```python
from ml.ensemble_predictor import EnsembleLinkPredictor

ensemble = EnsembleLinkPredictor()
result = ensemble.predict_single("BRCA1", "TP53")
# result['ensemble_score'] - weighted average score
# result['predictions'] - individual model predictions
# result['interpretation'] - explanation of agreement/disagreement
```

**Note on Node2Vec + GNN compatibility:**
- Random walk embeddings "emulate" conv-GNN aggregation
- Using Node2Vec as input to GNN is redundant (double aggregation)
- Current approach (Node2Vec → dot product) is correct
- If adding GNN later, use raw features (expression, GO terms) not Node2Vec embeddings
- GAT (attention-based) may add value over Node2Vec; vanilla GCN won't

**Priority:** COMPLETED
**Effort:** 1 day

---

### 2. Batched Abstract Mining (Token-Aware Chunking) - COMPLETED

**Problem:** Currently 1 LLM invocation per abstract - inefficient and hits rate limits.

**Status:** IMPLEMENTED (2025-12-03)

**Implementation:**
- Created `pipeline/batched_mining.py` with `BatchedMiningOrchestrator`
- Created `scripts/test_batched_mining.py` for scale testing

**Key Features:**
1. Token-aware chunking with configurable target (~5K tokens per chunk)
2. Minimum 3 chunks enforced for parallelization
3. Structured JSON output with Gemini response_schema
4. Exponential backoff retry on rate limits (429, 503)
5. Evidence sentences preserved verbatim from abstracts
6. Relationship deduplication by (source, target, type)

**Test Results (2025-12-03, orexin sleep regulation query):**

| Abstracts | Chunks | Tokens Used | Relationships | Duration | Throughput |
|-----------|--------|-------------|---------------|----------|------------|
| 20        | 3      | 14,495      | 78            | 25.6s    | 0.8/s      |
| 50        | 3      | 31,275      | 141           | 48.7s    | 1.0/s      |
| 100       | 6      | 59,048      | 244           | 48.6s    | 2.1/s      |

**Key Observations:**
- Throughput scales with parallelization: 2.6x improvement from 20 to 100 abstracts
- Duration stays constant (~50s) due to parallel chunk processing
- Relationship extraction quality is good (meaningful orexin/sleep/narcolepsy edges)
- Token efficiency: ~280 tokens per abstract average

**Files Created:**
- `/Users/dejandukic/dejan_dev/tetra/tetra_v1/pipeline/batched_mining.py`
- `/Users/dejandukic/dejan_dev/tetra/tetra_v1/scripts/test_batched_mining.py`

**Usage:**
```python
from pipeline.batched_mining import BatchedMiningOrchestrator, BatchedMiningConfig

config = BatchedMiningConfig(
    target_tokens_per_chunk=5000,
    min_chunks=3,
    max_concurrent=5,
)
orchestrator = BatchedMiningOrchestrator(config)
result = await orchestrator.run(articles, annotations)
# result['relationships'] - list of ExtractedRelationship
# result['statistics'] - mining metrics
```

**Priority:** COMPLETED
**Effort:** 1 day

---

## Testing & Validation Findings

### Pipeline Integration Test (2025-12-02 23:54 UTC)

**All core pipeline modules tested successfully:**

1. **PipelineConfig**: Loaded correctly from environment
   - Model: gemini-2.0-flash-exp
   - PubMed max results: 50
   - Status: PASS

2. **PipelineReport**: Phase tracking working
   - Created and tracked metrics for phases
   - Status: PASS

3. **STRING Expansion**: Successfully expanded BRCA1 protein
   - Input: ["BRCA1"] (1 seed protein)
   - Output: 31 expanded proteins, 241 interactions
   - API connectivity: OK
   - Status: PASS

4. **Query Construction**: PubMed query agent functional
   - Successfully constructed compound query for 3 proteins
   - Query: (BRCA1 OR TP53 OR PTEN) AND (DNA damage OR repair) AND humans[MeSH Terms] AND 2019:2024[pdat]
   - Tokens used: 337
   - Status: PASS

5. **Observability**: Langfuse tracing enabled
   - Credentials configured correctly
   - Tracing initialized successfully
   - Status: PASS

**Summary:** All pipeline modules functioning correctly. No errors detected. Ready for integration testing with full workflows.

---

## Backlog

- [ ] Add timeouts to LLM API calls (query_agent.py)
- [ ] Remove `async` from merge_to_knowledge_graph() (no awaits)
- [ ] Add PUBMED_API_KEY validation in frontend
- [ ] Improve type hints in app.py
- [ ] Add error handling for StringClient initialization
- [ ] Integrate batched_mining.py into parallel_extraction.py pipeline
- [ ] Add tiktoken as optional dependency for accurate token counting
- [ ] Consider increasing target_tokens_per_chunk to 10K-15K for better efficiency
- [ ] Add Langfuse tracing to batched mining operations
