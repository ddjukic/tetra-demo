# TODO - Tetra V1 Optimizations & Refinements

## 2024-12-02 22:58 UTC - Architecture Optimizations

### 1. Ensemble Node2Vec with Multiple p/q Parameters

**Problem:** Currently only one node2vec model trained with a single set of p & q parameters.

**Background:**
- `p` (return parameter): Controls likelihood of returning to previous node
  - High p = explore outward, less likely to return
  - Low p = allow backtracking, stay local
- `q` (in-out parameter): Controls BFS vs DFS behavior
  - q > 1: BFS-like, captures **homophily** (local neighborhood clustering)
  - q < 1: DFS-like, captures **structural equivalence** (global roles like hubs)

**Optimization:**
1. Train at least 2-3 node2vec models with different p/q settings:
   - Model A: p=1, q=0.5 (DFS-like, structural equivalence)
   - Model B: p=1, q=2.0 (BFS-like, homophily)
   - Model C: p=0.5, q=1.0 (balanced)

2. Create ensemble scoring:
   ```python
   def ensemble_predict(node1, node2, models, weights=None):
       scores = [model.predict_proba(node1, node2) for model in models]
       # Weighted average or voting
       return aggregate(scores, weights)
   ```

3. Agent-guided interpretation:
   - Agent analyzes which model's prediction is more confident
   - Interprets based on graph structure (homophilic vs structural)
   - Provides reasoning: "Model A (structural) suggests these proteins share hub-like roles..."

4. Add homophily estimation algorithm:
   ```python
   def estimate_homophily(graph, attribute='type'):
       """
       Calculate edge homophily ratio.
       H = (edges connecting same-type nodes) / (total edges)
       H > 0.5 suggests homophily, H < 0.5 suggests heterophily
       """
   ```

**Note on Node2Vec + GNN compatibility:**
- Random walk embeddings "emulate" conv-GNN aggregation
- Using Node2Vec as input to GNN is redundant (double aggregation)
- Current approach (Node2Vec → dot product) is correct
- If adding GNN later, use raw features (expression, GO terms) not Node2Vec embeddings
- GAT (attention-based) may add value over Node2Vec; vanilla GCN won't

**Priority:** Medium
**Effort:** 2-3 days

---

### 2. Batched Abstract Mining (Token-Aware Chunking)

**Problem:** Currently 1 LLM invocation per abstract - inefficient and hits rate limits.

**Current state:**
- ~50 abstracts → 50 API calls
- Redundant context (NER entities repeated in every prompt)
- High latency, rate limit risk
- Not DRY

**Optimization:**
1. Batch abstracts by token count:
   - Target: ~10K tokens per agent call (conservative)
   - Or: ~32K tokens per call (aggressive, like v0 design)
   - Minimum 3 parallel agents even if all fit in one prompt

2. Token-aware chunking algorithm:
   ```python
   def chunk_abstracts_by_tokens(articles, target_tokens=10000, min_chunks=3):
       """
       Group abstracts into chunks targeting ~target_tokens each.
       Ensure at least min_chunks for parallelization.
       """
       chunks = []
       current_chunk = []
       current_tokens = 0

       for article in articles:
           article_tokens = count_tokens(article['abstract'])
           if current_tokens + article_tokens > target_tokens and current_chunk:
               chunks.append(current_chunk)
               current_chunk = []
               current_tokens = 0
           current_chunk.append(article)
           current_tokens += article_tokens

       if current_chunk:
           chunks.append(current_chunk)

       # Ensure minimum chunks for parallelization
       while len(chunks) < min_chunks:
           # Split largest chunk
           ...

       return chunks
   ```

3. Shared entity context:
   - Extract unique entities across ALL abstracts first
   - Pass entity list once per chunk (not per abstract)
   - Reduces prompt tokens significantly

4. Parallel execution with semaphore:
   ```python
   async def mine_relationships_batched(articles, annotations, config):
       chunks = chunk_abstracts_by_tokens(articles, target_tokens=10000)
       unique_entities = extract_unique_entities(annotations)

       semaphore = asyncio.Semaphore(config.mining_max_concurrent)
       tasks = [
           extract_from_chunk(chunk, unique_entities, semaphore)
           for chunk in chunks
       ]
       results = await asyncio.gather(*tasks)
       return merge_results(results)
   ```

**Statistics collected (2025-12-02):**
- [x] Fetched 200 orexin papers (2010-2024) - 194 with abstracts

**Token Distribution (cl100k_base encoding):**
| Metric | Characters | Tokens |
|--------|------------|--------|
| Min    | 117        | 23     |
| Max    | 3,581      | 901    |
| Mean   | 1,304      | 273    |
| Median | 1,305      | 262    |
| **Total** | —       | **52,980** |

**Batching Recommendation:**
- At 10K tokens/batch: ~6 batches needed (conservative)
- At 32K tokens/batch: ~2 batches needed (aggressive)
- **Recommendation:** Use 10K tokens/batch with min 3 parallel workers
- All 50 abstracts (~13K tokens) could fit in 2 batches

**Priority:** High (performance critical)
**Effort:** 1-2 days

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
