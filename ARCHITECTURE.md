# Tetra V1 Architecture

> **Last Updated**: December 2024
> **Status**: Near-complete research prototype

This document describes the architecture for building and querying scientific knowledge graphs. The system uses Google ADK for agent orchestration, LiteLLM for multi-provider extraction, and Node2Vec+LogReg for link prediction.

---

## Conceptual Architecture (High-Level)

```mermaid
flowchart TB
    subgraph Input["User Input"]
        seeds["Seed Terms<br/>(HCRTR1, HCRTR2, orexin)"]
    end

    subgraph DataSources["Data Sources"]
        string["STRING DB<br/>Curated PPIs<br/>(API only)"]
        query_llm["Query Construction<br/>⚡ LLM (Gemini)"]
        pubmed["PubMed/PubTator<br/>Literature + NER<br/>(API only)"]
    end

    subgraph Processing["Processing Pipeline"]
        extraction["Relationship Extraction<br/>⚡ LLM (LiteLLM)<br/>+ Co-occurrence (Python)"]
        grounding["Entity Grounding<br/>INDRA/Gilda → HGNC<br/>(API only)"]
        merge["Evidence Merge<br/>Multi-source Fusion<br/>(Deterministic)"]
    end

    subgraph ML["Machine Learning"]
        embeddings["Node2Vec<br/>Graph Embeddings"]
        predictor["Link Predictor<br/>LogReg Classifier"]
    end

    subgraph Output["Output"]
        kg["Knowledge Graph<br/>NetworkX + Provenance"]
        predictions["Novel Predictions<br/>Scored Hypotheses"]
    end

    seeds --> string
    string --> query_llm
    query_llm --> pubmed
    string --> merge
    pubmed --> extraction
    extraction --> merge
    merge --> grounding
    grounding --> kg
    kg --> embeddings
    embeddings --> predictor
    predictor --> predictions
```

---

## Detailed Architecture (Component-Level)

```mermaid
flowchart TB
    subgraph UserLayer["User Interface Layer"]
        cli["main.py<br/>CLI Entry Point"]
        streamlit["frontend/app.py<br/>Streamlit UI"]
    end

    subgraph AgentLayer["Agent Layer (Google ADK)"]
        orchestrator["ADKOrchestrator<br/>agent/adk_orchestrator.py"]
        data_fetch["DataFetchAgent<br/>agent/data_fetch_agent.py"]
        tools_module["Module Functions<br/>(ADK Discovery)"]
        tools_class["AgentTools Class<br/>agent/tools.py"]

        orchestrator --> tools_module
        orchestrator --> data_fetch
        tools_module -->|"get_tools()"| tools_class
    end

    subgraph ClientLayer["API Client Layer"]
        string_client["StringClient<br/>clients/string_client.py"]
        pubmed_client["PubMedClient<br/>clients/pubmed_client.py"]
        gilda_client["GildaClient<br/>clients/gilda_client.py"]
    end

    subgraph PipelineLayer["Pipeline Layer"]
        query_agent["QueryAgent ⚡LLM<br/>pipeline/query_agent.py"]
    end

    subgraph ExtractionLayer["Extraction Layer"]
        batched_miner["BatchedLiteLLMMiner ⚡LLM<br/>extraction/batched_litellm_miner.py"]
        config_loader["ConfigLoader<br/>extraction/config_loader.py"]
        inferrer["RelationshipInferrer ⚡LLM<br/>extraction/relationship_inferrer.py"]

        batched_miner --> config_loader
    end

    subgraph MLLayer["ML Layer"]
        link_predictor["LinkPredictor<br/>ml/link_predictor.py"]
        hard_neg["HardNegativeSampler<br/>ml/hard_negative_sampling.py"]
        node2vec["Node2Vec<br/>(gensim)"]

        link_predictor --> hard_neg
        link_predictor --> node2vec
    end

    subgraph DataLayer["Data Layer"]
        kg["KnowledgeGraph<br/>models/knowledge_graph.py"]
        networkx["NetworkX<br/>MultiDiGraph"]

        kg --> networkx
    end

    subgraph ExternalAPIs["External APIs"]
        string_api["STRING API<br/>string-db.org"]
        pubmed_api["PubMed E-utilities<br/>+ PubTator3"]
        gilda_api["INDRA/Gilda<br/>grounding.indra.bio"]
        llm_api["LLM Providers<br/>Gemini/Cerebras/OpenRouter"]
    end

    cli --> orchestrator
    streamlit --> orchestrator

    data_fetch --> string_client
    data_fetch --> pubmed_client
    data_fetch --> query_agent

    tools_class --> string_client
    tools_class --> pubmed_client
    tools_class --> gilda_client
    tools_class --> batched_miner
    tools_class --> link_predictor
    tools_class --> kg

    string_client --> string_api
    pubmed_client --> pubmed_api
    gilda_client --> gilda_api
    query_agent --> llm_api
    batched_miner --> llm_api
    inferrer --> llm_api
```

---

## Seven-Phase Pipeline

```mermaid
flowchart LR
    subgraph Phase1["Phase 1"]
        p1["STRING<br/>Network<br/>Expansion"]
    end

    subgraph Phase2["Phase 2"]
        p2["Query<br/>Construction<br/>(LLM)"]
    end

    subgraph Phase3["Phase 3"]
        p3["PubMed<br/>Fetch +<br/>PubTator NER"]
    end

    subgraph Phase4["Phase 4 (Parallel)"]
        p4a["4A: Co-occurrence<br/>(Python)"]
        p4b["4B: LLM Mining<br/>(Batched)"]
    end

    subgraph Phase5["Phase 5"]
        p5["Graph Merge<br/>+ Grounding<br/>+ Dedup"]
    end

    subgraph Phase6["Phase 6"]
        p6["ML Link<br/>Prediction"]
    end

    subgraph Phase7["Phase 7"]
        p7["Summary<br/>+ Report"]
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4
    p4a --> Phase5
    p4b --> Phase5
    Phase5 --> Phase6 --> Phase7
```

**LLM Usage Summary:**
| Phase | Component | Uses LLM? | Details |
|-------|-----------|-----------|---------|
| 1 | STRING Expansion | ❌ No | REST API to string-db.org |
| 2 | Query Construction | ⚡ **Yes** | Gemini via `google.generativeai` |
| 3 | PubMed Fetch | ❌ No | NCBI E-utilities + PubTator3 APIs |
| 4A | Co-occurrence | ❌ No | Pure Python (entity pairs in same PMID) |
| 4B | Relationship Mining | ⚡ **Yes** | LiteLLM (Cerebras/Gemini/GPT-4) |
| 5 | Graph Merge | ❌ No | Deterministic fusion + INDRA/Gilda API |
| 6 | Link Prediction | ❌ No | ML model (Node2Vec + LogReg) |
| 7 | Summary | ❌ No | Aggregation only |

---

## ADK Two-Layer Tool Pattern

```mermaid
flowchart TB
    subgraph ADKFramework["Google ADK Framework"]
        discovery["Tool Discovery<br/>(Reflection)"]
        gemini["Gemini LLM<br/>(Function Calling)"]
    end

    subgraph Layer1["Layer 1: Module Functions"]
        mod_global["_tools_instance: Optional[AgentTools]"]
        set_tools["set_tools(tools)"]
        get_tools["get_tools() → AgentTools"]

        func1["async def get_string_network(...)"]
        func2["async def extract_relationships(...)"]
        func3["def predict_novel_links(...)"]
    end

    subgraph Layer2["Layer 2: AgentTools Class"]
        class_init["AgentTools.__init__(<br/>link_predictor,<br/>string_client,<br/>pubmed_client,<br/>relationship_miner,<br/>relationship_inferrer)"]

        state["Session State:<br/>self.current_graph"]

        impl1["async def get_string_network(...)"]
        impl2["async def extract_relationships(...)"]
        impl3["def predict_novel_links(...)"]
    end

    discovery -->|"Introspect"| func1
    discovery -->|"Introspect"| func2
    discovery -->|"Introspect"| func3

    gemini -->|"Invoke"| func1
    gemini -->|"Invoke"| func2
    gemini -->|"Invoke"| func3

    func1 -->|"get_tools()"| impl1
    func2 -->|"get_tools()"| impl2
    func3 -->|"get_tools()"| impl3

    set_tools --> mod_global
    get_tools --> mod_global

    class_init --> state
    impl1 --> state
    impl2 --> state
    impl3 --> state
```

---

## Batched Extraction Pipeline

```mermaid
flowchart TB
    subgraph Input["Input"]
        articles["100 PubMed<br/>Abstracts"]
        annotations["PubTator<br/>Annotations"]
    end

    subgraph Chunking["Token-Aware Chunking"]
        tokenize["Count Tokens<br/>(tiktoken)"]
        binpack["Greedy<br/>Bin-Packing"]
        chunks["8-16 Chunks<br/>(~5K tokens each)"]
    end

    subgraph Parallel["Parallel Mining"]
        sem["asyncio.Semaphore(5)"]
        chunk1["Chunk 1"]
        chunk2["Chunk 2"]
        chunk3["Chunk N"]
        llm["LiteLLM<br/>Multi-Provider"]
    end

    subgraph Validation["Provenance Validation"]
        pmid_check["PMID<br/>Attribution"]
        index_check["Sentence Index<br/>Bounds Check"]
    end

    subgraph Output["Output"]
        valid["Valid<br/>Relationships<br/>(97.7%)"]
        stats["Statistics<br/>+ Metrics"]
    end

    articles --> tokenize
    annotations --> tokenize
    tokenize --> binpack
    binpack --> chunks

    chunks --> sem
    sem --> chunk1
    sem --> chunk2
    sem --> chunk3

    chunk1 --> llm
    chunk2 --> llm
    chunk3 --> llm

    llm --> pmid_check
    pmid_check --> index_check
    index_check --> valid
    index_check --> stats
```

---

## Evidence Integration Flow

```mermaid
flowchart TB
    subgraph Sources["Evidence Sources"]
        string["STRING DB<br/>weight=0.9"]
        llm["LLM Extraction<br/>weight=0.7"]
        cooc["Co-occurrence<br/>weight=0.5"]
    end

    subgraph Processing["Processing"]
        collect["Collect<br/>Edge Candidates"]
        group["Group by<br/>(src, tgt, type)"]
        merge["Merge Evidence<br/>Arrays"]
        dedup["Deduplicate<br/>by PMID"]
    end

    subgraph Scoring["Confidence Scoring"]
        base["max(source_weights)"]
        multi["+ multi_source_boost<br/>(0.15 per source)"]
        evidence["+ evidence_boost<br/>(0.1 per citation)"]
        cap["min(total, 1.0)"]
    end

    subgraph Output["Output"]
        edge["Edge with<br/>Confidence + Evidence"]
    end

    string --> collect
    llm --> collect
    cooc --> collect

    collect --> group
    group --> merge
    merge --> dedup

    dedup --> base
    base --> multi
    multi --> evidence
    evidence --> cap
    cap --> edge
```

---

## Entity Grounding & Deduplication

```mermaid
flowchart TB
    subgraph Input["Raw Entities"]
        e1["LEP"]
        e2["Leptin"]
        e3["ob protein"]
        e4["TNF"]
    end

    subgraph Grounding["INDRA/Gilda Grounding"]
        api["POST /ground_multi"]
        cache["In-Memory Cache"]
        select["Select Best Result<br/>(prefer HGNC)"]
    end

    subgraph Results["Grounding Results"]
        r1["LEP → HGNC:6553"]
        r2["Leptin → HGNC:6553"]
        r3["ob → HGNC:6553"]
        r4["TNF → HGNC:11892"]
    end

    subgraph Dedup["HGNC Deduplication"]
        group["Group by<br/>HGNC ID"]
        canonical["Select<br/>Canonical Name"]
        remap["Remap Edges<br/>+ Merge Evidence"]
    end

    subgraph Output["Output"]
        final["LEP (aliases: Leptin, ob)<br/>TNF"]
    end

    e1 --> api
    e2 --> api
    e3 --> api
    e4 --> api

    api --> cache
    cache --> select

    select --> r1
    select --> r2
    select --> r3
    select --> r4

    r1 --> group
    r2 --> group
    r3 --> group
    r4 --> group

    group --> canonical
    canonical --> remap
    remap --> final
```

---

## ML Link Prediction Pipeline

```mermaid
flowchart TB
    subgraph Training["Offline Training (One-Time)"]
        string_data["STRING Physical<br/>Interactions"]
        split["Split Edges<br/>80/20 BEFORE<br/>Embeddings"]
        train_graph["Training Graph<br/>(only train edges)"]
        node2vec["Node2Vec<br/>Training"]
        embeddings["128D<br/>Embeddings"]
        hadamard["Hadamard<br/>Product"]
        logreg["Logistic<br/>Regression"]
    end

    subgraph Evaluation["Hard Negative Evaluation"]
        hard_neg["Hard Negatives<br/>(2-hop + degree-matched)"]
        eval["Evaluate on<br/>Test Edges"]
        auc["ROC-AUC: 0.78<br/>(honest)"]
    end

    subgraph Inference["Runtime Inference"]
        candidates["Candidate<br/>Pairs"]
        predict["predict_proba()"]
        filter["Filter by<br/>min_score"]
        novel["Novel<br/>Predictions"]
    end

    string_data --> split
    split --> train_graph
    train_graph --> node2vec
    node2vec --> embeddings
    embeddings --> hadamard
    hadamard --> logreg

    logreg --> hard_neg
    hard_neg --> eval
    eval --> auc

    candidates --> predict
    embeddings --> predict
    logreg --> predict
    predict --> filter
    filter --> novel
```

---

## Key Design Decisions

### 1. Node2Vec + LogReg over GNN

| Factor | Node2Vec + LogReg | GCN/GraphSAGE |
|--------|-------------------|---------------|
| Training complexity | CPU-friendly, 30-60 min | Requires GPU, hours |
| Node features | None needed | Requires features |
| Interpretability | Embeddings inspectable | Black box |
| STRING sparsity | Random walks handle well | Message passing struggles |

### 2. Sentence-Index Provenance

| Approach | Validation Rate | Failure Mode |
|----------|-----------------|--------------|
| Text extraction | 62% | Paraphrasing (38%) |
| Sentence indices | 97.7% | Index hallucination (2.3%) |

### 3. Hard Negative Evaluation

| Strategy | ROC-AUC | Interpretation |
|----------|---------|----------------|
| Random negatives | 0.99 | Inflated (degree bias) |
| Combined hard | 0.78 | Honest evaluation |

### 4. Multi-Source Confidence

```
Confidence = max(source_weights)
           + 0.15 × (num_sources - 1)
           + min(0.1 × (evidence_count - 1), 0.2)
```

---

## File Structure

```
tetra_v1/
├── main.py                         # CLI entry point
├── DESIGN.md                       # Design decisions & rationale
├── ARCHITECTURE.md                 # This document
├── DEMO.md                         # Interview preparation guide
│
├── agent/
│   ├── adk_orchestrator.py         # ADK agent + module functions
│   ├── data_fetch_agent.py         # Data fetching orchestrator (STRING + PubMed)
│   ├── tools.py                    # AgentTools class (stateful)
│   └── graph_agent.py              # GraphRAG Q&A agent
│
├── clients/
│   ├── string_client.py            # STRING API (async httpx)
│   ├── pubmed_client.py            # PubMed + PubTator (async)
│   └── gilda_client.py             # INDRA/Gilda grounding (async)
│
├── extraction/
│   ├── batched_litellm_miner.py    # ⚡LLM Batched relationship extraction (LiteLLM)
│   ├── config.toml                 # Prompts + schema definitions
│   ├── config_loader.py            # Provider routing
│   └── relationship_inferrer.py    # ⚡LLM relationship inference
│
├── models/
│   └── knowledge_graph.py          # NetworkX graph + algorithms
│
├── ml/
│   ├── link_predictor.py           # Node2Vec + LogReg
│   └── hard_negative_sampling.py   # Evaluation strategies
│
├── pipeline/
│   ├── config.py                   # PipelineConfig
│   ├── metrics.py                  # TokenUsage, statistics
│   ├── merge.py                    # Evidence aggregation
│   └── query_agent.py              # ⚡LLM PubMed query construction (Gemini)
│
├── frontend/
│   └── app.py                      # Streamlit UI
│
├── scripts/
│   ├── train_link_predictor.py     # Training pipeline
│   ├── benchmark_miner_scale.py    # Performance testing
│   └── test_miner_provenance.py    # Provenance validation
│
└── tests/
    ├── test_batched_litellm_miner.py
    ├── test_link_predictor.py
    └── test_knowledge_graph.py
```

---

## Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=AIza...              # Gemini access

# Optional
NCBI_API_KEY=...                    # Higher PubMed rate limits
OPENROUTER_API_KEY=sk-or-...        # Cerebras/OpenRouter access
LANGFUSE_PUBLIC_KEY=...             # Observability
LANGFUSE_SECRET_KEY=...
```

### Extraction Config (`extraction/config.toml`)

```toml
[BATCHED]
TARGET_TOKENS_PER_CHUNK = 5000
MIN_CHUNKS = 3
MAX_CONCURRENT = 5
MAX_RETRIES = 3
RETRY_DELAY_MS = 1000
MIN_CONFIDENCE = 0.5
MAX_TOKENS = 8192

[EXTRACTORS.cerebras]
MODEL = "openrouter/openai/gpt-oss-120b"
TEMPERATURE = 0.1

[EXTRACTORS.gemini]
MODEL = "gemini/gemini-2.5-flash"
TEMPERATURE = 0.1
```

---

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Link predictor AUC (hard neg) | > 0.75 | 0.78-0.86 |
| Provenance validation | > 90% | 97.7% |
| Extraction throughput | > 2000 tok/s | 3,000+ tok/s |
| Entity grounding | > 70% | 75-80% |
| Query response time | < 30s | ~15-25s |

---

## Cost Estimation

Based on Gemini 2.5 Flash pricing:

| Operation | Input Tokens | Output Tokens | Est. Cost |
|-----------|--------------|---------------|-----------|
| Query construction | 2,000 | 500 | $0.0003 |
| Relationship mining (50 papers) | 150,000 | 30,000 | $0.020 |
| Inference (5 predictions) | 10,000 | 2,000 | $0.002 |
| **Total per pipeline run** | ~162,000 | ~32,500 | **~$0.022** |

---

## Future Enhancements

1. **GraphSAGE Upgrade**: Better predictions with learned node features
2. **Neo4j Migration**: For production scale (millions of nodes)
3. **Streaming Responses**: Real-time agent output
4. **Fine-tuned Extraction**: Domain-specific model instead of prompting
5. **Multi-species Support**: Extend HGNC to include model organisms
