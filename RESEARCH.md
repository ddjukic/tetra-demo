# Research Notes

## ROC-AUC Investigation - Link Prediction Validation Analysis
### Date: 2025-12-03

---

## Executive Summary

**Verdict: The 0.99 ROC-AUC is LIKELY LEGITIMATE but MISLEADING.**

The high ROC-AUC scores (~0.99) are technically correct given the validation setup, but they do NOT reflect real-world link prediction performance. The problem is an "easy" version of link prediction due to **random negative sampling bias** combined with **high-confidence STRING edge filtering**.

---

## 1. Validation Methodology Analysis

### What the Code Does Correctly

After careful analysis of `/Users/dejandukic/dejan_dev/tetra/tetra_v1/ml/link_predictor.py`, the implementation follows proper validation principles:

**Edge Splitting BEFORE Embedding Training:**
```python
# Step 1: Split edges FIRST
print(f"\nStep 1: Splitting {len(self._all_edges)} edges (test_size={test_size})")
edges_array = np.array([(e[0], e[1]) for e in self._all_edges])
indices = np.arange(len(edges_array))
np.random.shuffle(indices)

n_test = int(len(indices) * test_size)
test_indices = indices[:n_test]
train_indices = indices[n_test:]
```

**Training Graph Built from Train Edges Only:**
```python
# Step 2: Build training graph (only train edges)
self.train_graph = nx.Graph()
train_edge_set = set()
for p1, p2 in train_edges:
    self.train_graph.add_edge(p1, p2)
    train_edge_set.add(tuple(sorted([p1, p2])))
```

**Node2Vec Trained on Training Graph Only:**
```python
# Step 3: Train Node2Vec on training graph ONLY
self.node2vec_model = self.train_embeddings(self.train_graph)
```

**Metrics Computed on Test Set:**
```python
# Step 7: Evaluate on truly held-out test set
y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
```

**Conclusion:** There is NO direct data leakage. Test edges are truly held out from embedding training. The ROC-AUC IS computed on the test set.

---

## 2. The Real Problem: Random Negative Sampling Bias

### The Critical Issue

The negative sampling strategy creates an artificially easy classification problem:

```python
# Sample negatives for training (same count as train positives)
train_negatives = []
while len(train_negatives) < len(train_edges):
    i, j = np.random.randint(0, len(train_nodes), size=2)
    if i != j:
        n1, n2 = train_nodes[i], train_nodes[j]
        edge = tuple(sorted([n1, n2]))
        if edge not in self.edge_set and edge not in seen_negatives:
            train_negatives.append((n1, n2))
```

**Why This Creates Easy Negatives:**

1. **Uniform Random Sampling**: Negatives are sampled uniformly from all non-edge pairs
2. **Sparse Network**: In a network with N nodes, there are O(N^2) possible pairs but only O(N) edges
3. **Most Random Pairs Are Trivially Negative**: Two randomly chosen proteins have essentially zero biological relationship
4. **Degree Bias**: Random positive edges favor high-degree nodes, while random negative edges have lower average degree

### Literature Evidence

Recent research confirms this is a well-known problem:

> "The common edge sampling procedure in the link prediction task has an implicit bias toward high-degree nodes and produces a highly skewed evaluation that favors methods overly dependent on node degree, to the extent that a **null link prediction method based solely on node degree can yield nearly optimal performance**."
> - [Implicit degree bias in the link prediction task (arXiv 2024)](https://arxiv.org/html/2405.14985v1)

> "Despite Noise-RF achieving a remarkable AUC value of **0.993**, it exhibited a pronounced bias. In the NPInter 4.0 dataset, nearly 98.9% of pairs in the positive set exhibited degrees exceeding 8, whereas 96.1% of randomly sampled negative pairs had degrees below 8."
> - [BMC Biology 2025](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02231-w)

> "Previous studies have predominantly centered on improving model performance through novel and efficient ML approaches, often resulting in **overoptimistic predictive estimates**."
> - [PNAS 2025: Bias-aware training and evaluation](https://www.pnas.org/doi/10.1073/pnas.2416646122)

---

## 3. STRING Database Filtering Effect

### High-Confidence Edge Selection

The implementation filters edges by combined score >= 700:

```python
if combined_score >= self.min_score:  # min_score=700
    self._all_edges.append((protein1, protein2, combined_score))
```

**STRING Score Interpretation:**
- Score > 900: Highest confidence (robust evidence)
- Score 700-900: High confidence
- Score 400-700: Medium confidence
- Score < 400: Low confidence

**Impact:** By using only high-confidence edges (score >= 700), we are training on the "easiest" positive examples - interactions with strong experimental support. This creates a sharper boundary between positives and random negatives.

---

## 4. Literature Benchmark Comparison

### What Other Papers Report

| Method | Dataset | ROC-AUC | Notes |
|--------|---------|---------|-------|
| Node2Vec (StellarGraph demo) | General graph | 0.912 | Proper validation |
| Node2Vec (various PPI) | PPI networks | 0.82-0.87 | Standard benchmarks |
| cGAN (BMC 2022) | PPI | 0.907 | Compared to Node2Vec 0.819 |
| TSAW (2024) | Social networks | 0.864 | Best random walk variant |
| Node2Vec original paper | Facebook | 0.98 | Social network (easier) |

### Key Observation

**Legitimate Node2Vec PPI link prediction scores typically range from 0.80-0.92 with proper evaluation.**

Scores approaching 0.99 are possible but usually indicate:
1. Random negative sampling (easy problem)
2. High-confidence edge filtering (clean positives)
3. Dense, well-connected graphs
4. Degree bias in evaluation

---

## 5. Why This Specific Problem Is "Easy"

### Multiple Factors Combine

1. **Random Negative Sampling**: Creates trivially distinguishable negative pairs
2. **High-Confidence Filtering**: Only the most reliable edges are used
3. **Community Structure**: PPI networks have strong modular structure that Node2Vec captures well
4. **Degree Distribution**: High-degree hub proteins appear disproportionately in positive edges
5. **Hadamard Product Features**: The feature representation (element-wise product of embeddings) can exploit degree patterns

### What the Model Likely Learns

The classifier is likely learning:
- **Degree-based features**: High-degree nodes appear in positive pairs
- **Community proximity**: Nodes in the same community co-occur in positives
- **NOT**: Fine-grained biological interaction signatures

---

## 6. Potential Issues Identified

### Issue 1: Random Negative Sampling (HIGH SEVERITY)

**Problem:** Negatives are uniformly random, creating an unrealistically easy task.

**Evidence:**
```python
i, j = np.random.randint(0, len(train_nodes), size=2)
```

**Impact:** Inflates ROC-AUC by 0.05-0.15 compared to realistic evaluation.

### Issue 2: No Degree Correction (MEDIUM SEVERITY)

**Problem:** No correction for degree bias in positive/negative sampling.

**Evidence:** Code samples edges uniformly without degree-aware stratification.

### Issue 3: Test Negatives From Training Nodes Only (LOW SEVERITY)

**Problem:** Test negatives use `train_nodes`, limiting to nodes seen during embedding.

**Evidence:**
```python
# Use only nodes that appear in training graph (so we have embeddings)
test_negatives = []
while len(test_negatives) < len(test_edges):
    i, j = np.random.randint(0, len(train_nodes), size=2)
```

**Note:** This is actually necessary to compute features, but could be documented better.

### Issue 4: No Inductive Evaluation (LOW SEVERITY)

**Problem:** All test nodes have embeddings from training graph.

**Real-world impact:** Cannot evaluate prediction for entirely new proteins.

---

## 7. Recommendations

### For More Realistic Evaluation

1. **Implement Hard Negative Sampling**
   - Sample negatives from same pathway/GO category but non-interacting
   - Use distance-based sampling (2-hop neighbors that don't interact)
   - Match degree distribution between positive and negative samples

2. **Add Degree-Corrected Evaluation**
   - Sample negative edges with same degree distribution as positives
   - Report metrics stratified by node degree

3. **Implement Inductive Evaluation**
   - Hold out some proteins entirely (not just edges)
   - Test on proteins not seen during embedding training

4. **Use Cross-Validation Across Edge Types**
   - Stratify by STRING evidence type (experimental, text-mining, etc.)

### For Current Use

If the current model is being used in production:
- **It IS useful** for ranking protein pairs by interaction likelihood
- **Calibration is unreliable** - don't interpret scores as probabilities
- **Novel predictions need validation** - ML score alone is insufficient
- **Compare against baselines** - how much better than degree-based predictor?

---

## 8. Confidence Assessment

### Is the 0.99 ROC-AUC Legitimate?

| Aspect | Assessment |
|--------|------------|
| Technical correctness | YES - No data leakage in edge splitting |
| Test set evaluation | YES - Metrics computed on held-out edges |
| Realistic performance estimate | NO - Random negatives inflate scores |
| Useful for applications | PARTIALLY - Rankings useful, calibration unreliable |

### Bottom Line

The 0.99 ROC-AUC is **technically legitimate** given the evaluation setup, but **not representative** of real-world link prediction difficulty. With proper hard negative sampling and degree-corrected evaluation, expect scores in the **0.80-0.90 range**.

**This is a common pitfall in the field.** Recent papers (2024-2025) have extensively documented this bias, and the community is moving toward more rigorous evaluation standards.

---

## 9. References

1. [Implicit degree bias in the link prediction task (arXiv 2024)](https://arxiv.org/html/2405.14985v1)
2. [Negative sampling strategies impact the prediction of scale-free biomolecular network interactions (BMC Biology 2025)](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02231-w)
3. [Bias-aware training and evaluation of link prediction algorithms (PNAS 2025)](https://www.pnas.org/doi/10.1073/pnas.2416646122)
4. [Assessment of community efforts to advance network-based PPI prediction (Nature Comms 2023)](https://www.nature.com/articles/s41467-023-37079-7)
5. [StellarGraph Node2Vec Link Prediction Demo](https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/node2vec-link-prediction.html)
6. [STRING Database Confidence Scores](https://www.blopig.com/blog/2017/01/confidence-scores-in-string/)
7. [Revisiting the negative example sampling problem for PPI (PMC 2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3198576/)
8. [Topology-Driven Negative Sampling Enhances Generalizability in PPI Prediction (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.04.27.591478v2.full)

---
---

## Fraunhofer SCAI Alzheimer's Knowledge Graph - Benchmark Opportunity
### Date: 2025-12-03

---

## Executive Summary

The Fraunhofer SCAI Institute has developed one of the largest and most comprehensive Alzheimer's disease knowledge graphs through multiple initiatives: the **AETIONOMY project**, **NeuroMMSig database**, and **Human Brain Pharmacome**. Led by Dr. Alpha Tom Kodamullil and Prof. Martin Hofmann-Apitius, this work represents years of manual curation and represents an excellent benchmark opportunity for validating Tetra V1's ability to rapidly recapitulate curated biomedical knowledge.

**Key Opportunity**: Demonstrate that Tetra V1 can recover in minutes what took a large team years to curate manually.

---

## 1. Primary Publications and Resources

### Main Papers

1. **Kodamullil, A.T., et al. (2015)**
   "Computable cause-and-effect models of healthy and Alzheimer's disease states and their mechanistic differential analysis"
   *Alzheimer's & Dementia*, 11(11), 1329-1339
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/25849034/) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1552526015000837)

2. **Domingo-Fernández, D., et al. (2017)**
   "Multimodal mechanistic signatures for neurodegenerative diseases (NeuroMMSig): A web server for mechanism enrichment"
   *Bioinformatics*, 33(22), 3679-3681
   [Oxford Academic](https://academic.oup.com/bioinformatics/article/33/22/3679/3884654)

### Web Resources

- **NeuroMMSig Database**: https://neurommsig.scai.fraunhofer.de/
- **NeuroMMSig v2.0**: https://neurommsig-v2.scai.fraunhofer.de/
- **GitHub Repository**: https://github.com/neurommsig/neurommsig-knowledge
- **Human Brain Pharmacome**: https://pharmacome.github.io/overview/
- **BEL Commons**: https://bel-commons.scai.fraunhofer.de
- **Fraunhofer SCAI Applied Semantics**: https://www.scai.fraunhofer.de/en/business-research-areas/bioinformatics/fields-of-research/Applied-Semantics.html

---

## 2. Knowledge Graph Statistics and Characteristics

### Alzheimer's Disease BEL Model (2015)
- **Nodes**: 4,052 nodes
- **Edges**: 9,926 edges (causal relationships)
- **Language**: Biological Expression Language (BEL)
- **Coverage**: Disease-associated genes, protein-protein interactions, miRNAs, bioprocesses, pathways, clinical readouts

### NeuroMMSig Database
- **Alzheimer's Mechanisms**: 124-126 manually curated mechanistic subgraphs
- **Parkinson's Mechanisms**: 65 mechanistic subgraphs
- **Entity Types**:
  - Genes (HGNC nomenclature)
  - Proteins
  - SNPs (genetic variants)
  - miRNAs
  - Imaging features
  - Bioprocesses/pathways
  - Drugs and compounds
  - Clinical trial associations
  - Side effects
- **Enrichment**: Multimodal data including imaging features, variant information, and drug associations
- **Threshold**: Only mechanisms with >20 nodes are used for enrichment analysis

### Project Timeline
- **AETIONOMY Project**: 2014-2019 (5-year IMI-funded initiative)
- **Manual Curation**: Years of effort by bioinformatics team at Fraunhofer SCAI
- **Funding**: Innovative Medicines Initiative (IMI), Fraunhofer MAVO program

---

## 3. Methodology and Data Sources

### Biological Expression Language (BEL)
The knowledge graphs are encoded in BEL, which captures cause-and-effect relationships as triplets:
- Subject-Predicate-Object format
- Examples: "Amyloid-beta increases apoptosis", "APP protein decreases neuron survival"
- Enables computable mechanistic models

### Data Sources
1. **Scientific Literature**: Manual extraction from research articles and reviews
2. **Canonical Pathways**: Well-established biological pathways
3. **Databases**: Integration of knowledge from various biomedical databases
4. **Genomic Data**: Complex genomic datasets from large AD patient cohorts
5. **Clinical Data**: Evidence from mouse, rat, and human studies

### Entity Types Captured
- **Proteins**: APP, tau/MAPT, APOE, PSEN1, PSEN2, NGFR, neurotrophic receptors
- **Genes**: Disease-associated genes with HGNC nomenclature
- **Bioprocesses**: Apoptosis, neurotrophin signaling, synaptic function, proteostasis
- **Pathways**: Amyloid cascade, tau hyperphosphorylation, neuroinflammation
- **Molecular**: miRNAs, SNPs, epigenetic factors
- **Drugs**: FDA-approved compounds, experimental therapeutics
- **Phenotypes**: Clinical readouts, imaging features

---

## 4. Key Scientific Insights from Their Work

### Core Alzheimer's Mechanisms (From NeuroMMSig)

1. **Amyloid Cascade Pathway**
   - APP processing by β and γ secretases
   - Aβ aggregation and fibril formation
   - Oxidative stress induction
   - Inflammatory cascade activation

2. **Tau Pathology**
   - Tau hyperphosphorylation
   - Neurofibrillary tangle (NFT) formation
   - MAPT gene isoform imbalance (4R vs 3R tau)
   - Microtubule destabilization

3. **Neuroinflammation**
   - Microglial activation
   - Complement system involvement
   - NF-κB pathway activation
   - Cytokine release and neuronal death

4. **Aβ-Tau Crosstalk**
   - Aβ oligomers trigger tau hyperphosphorylation
   - GSK3 kinase involvement
   - ABL1 kinase dysregulation
   - Synergistic impairment of synaptic genes

5. **Synaptic Dysfunction**
   - Long-term potentiation (LTP) inhibition
   - Nicotinic receptor signaling disruption
   - Neurotrophic signaling impairment (NGFR pathway)
   - Exosome-mediated tau propagation

### Landmark Relationships
- **Amyloid-beta associates with NGFR → inhibits neuron survival → neuron death**
- **APP processing → Aβ formation → tau hyperphosphorylation → NFT formation**
- **Aβ oligomers → microglial activation → neuroinflammation → synaptic loss**
- **APOE ε4 allele → increased AD risk (strongest genetic risk factor)**

---

## 5. Publicly Available Resources

### GitHub Repository
**Repository**: https://github.com/neurommsig/neurommsig-knowledge
- **License**: CC BY 4.0 (for BEL scripts), MIT (for Python code)
- **Installation**: `pip install neurommsig-knowledge`
- **Access Method**:
```python
from neurommsig_knowledge import repository
from pybel import union

graphs = repository.get_graphs()
graph = union(graphs.values())
```

### Web Interfaces
- **NeuroMMSig Server**: Mechanism enrichment analysis for multi-scale data
- **BEL Commons**: Web application for exploring BEL networks
- **Human Brain Pharmacome**: Drug repurposing predictions

### Data Format
- **Primary Format**: BEL (Biological Expression Language)
- **Analysis Tools**: PyBEL (Python library for BEL graph manipulation)
- **Export Formats**: Compatible with network analysis tools

---

## 6. Validation Strategy for Tetra V1

### Objective
Demonstrate that Tetra V1 can recapitulate the manually curated Alzheimer's knowledge graph in a fraction of the time using autonomous agentic literature mining.

### Validation Approach

#### Phase 1: Core Protein Recovery (Seed-Based)
**Target**: Recover the 4 core AD genes and their immediate interactors

**Seed Proteins**:
1. **APP** (Amyloid Precursor Protein) - HGNC:620
2. **PSEN1** (Presenilin 1) - HGNC:9508
3. **MAPT** (Tau protein) - HGNC:6893
4. **APOE** (Apolipoprotein E) - HGNC:613

**Tetra Query**:
```
Query: "Map the protein interaction network and functional relationships for the following Alzheimer's disease genes: APP, PSEN1, MAPT, and APOE. Include direct interactors, regulatory relationships, and pathway associations."
```

**Expected Recovery**:
- Direct protein-protein interactions (should overlap with BEL edges)
- Regulatory relationships (increases/decreases)
- Pathway memberships
- Disease associations

**Success Metrics**:
- **Edge Recall**: % of BEL edges recovered by Tetra
- **Node Recall**: % of proteins/genes in NeuroMMSig subgraphs recovered
- **Relationship Type Coverage**: Capture activation, inhibition, binding, modification

#### Phase 2: Mechanism Recovery
**Target**: Recover the 124 mechanistic subgraphs identified in NeuroMMSig

**Tetra Queries** (stratified by mechanism):
1. **Amyloid Cascade**: "Describe the molecular mechanisms of amyloid-beta production from APP, including secretase processing, aggregation, and downstream effects"
2. **Tau Pathology**: "Map the mechanisms of tau hyperphosphorylation, including kinases involved, NFT formation, and effects on microtubules"
3. **Neuroinflammation**: "Identify the neuroinflammatory pathways in Alzheimer's, including microglial activation, complement system, and cytokine signaling"
4. **Synaptic Dysfunction**: "Map synaptic dysfunction mechanisms in AD, including LTP inhibition, neurotrophic signaling, and receptor involvement"
5. **Genetic Variants**: "Identify AD-associated genetic variants in APP, PSEN1, PSEN2, MAPT, and APOE with functional consequences"

**Success Metrics**:
- **Mechanism Coverage**: % of 124 NeuroMMSig mechanisms represented
- **Granularity Match**: Do we capture same level of detail?
- **Novel Insights**: Do we discover relationships not in NeuroMMSig?

#### Phase 3: Drug Repurposing Candidates
**Target**: Compare drug-target predictions with Human Brain Pharmacome

**Tetra Query**:
```
Query: "Identify FDA-approved drugs and their targets that could modulate Alzheimer's disease pathways, including amyloid processing, tau pathology, neuroinflammation, and synaptic function"
```

**Success Metrics**:
- **Drug Overlap**: % of drugs identified by both systems
- **Novel Candidates**: Drugs found by Tetra but not in Pharmacome (potential discoveries)
- **Mechanism Rationale**: Can we explain why each drug might work?

#### Phase 4: Comorbidity Analysis
**Target**: Validate shared mechanisms between AD and Type 2 Diabetes (T2DM)

Kodamullil et al. (2020) published analysis of AD-T2DM comorbidity. We can test if Tetra independently discovers these connections.

**Tetra Query**:
```
Query: "Identify shared molecular mechanisms and pathways between Alzheimer's disease and Type 2 Diabetes Mellitus"
```

**Success Metrics**:
- **Shared Pathway Recovery**: Do we find the same shared pathways?
- **Novel Comorbidity Links**: Do we identify additional comorbidities?

---

## 7. Quantitative Success Criteria

### Tier 1: Core Validation (Minimum Viable Benchmark)
- **Edge Recall ≥ 60%**: Recover at least 60% of direct relationships in AD BEL model
- **Core Protein Coverage = 100%**: All APP, PSEN1, MAPT, APOE interactions captured
- **Mechanism Coverage ≥ 50%**: At least 62 of 124 NeuroMMSig mechanisms represented

### Tier 2: Strong Validation (Compelling Demonstration)
- **Edge Recall ≥ 75%**: Recover 75% of BEL relationships
- **Mechanism Coverage ≥ 70%**: At least 87 of 124 mechanisms
- **Novel Discovery Rate ≥ 10%**: Find at least 10% new relationships not in NeuroMMSig
- **Drug Candidate Overlap ≥ 60%**: 60% agreement on repurposing candidates

### Tier 3: Exceptional Validation (Publishable Result)
- **Edge Recall ≥ 85%**: Near-complete recovery of curated knowledge
- **Mechanism Coverage ≥ 85%**: Comprehensive mechanism representation
- **Novel Discovery Rate ≥ 20%**: Substantial new insights
- **Literature Recency Advantage**: Capture relationships from papers published after 2019 (post-AETIONOMY)

### Speed Benchmark
- **Tetra Execution Time**: Target < 30 minutes for full analysis
- **Manual Curation Time**: 5 years (AETIONOMY project: 2014-2019)
- **Speedup Factor**: ~87,600x faster (5 years → 30 minutes)

---

## 8. Specific Tetra Queries for Demonstration

### Query Set 1: Breadth - Core AD Biology
```
1. "Map the complete amyloid precursor protein (APP) processing pathway, including all secretases, cleavage products, and downstream signaling events in Alzheimer's disease"

2. "Identify all kinases involved in tau hyperphosphorylation and their regulatory mechanisms in Alzheimer's pathology"

3. "Map the neuroinflammatory cascade in Alzheimer's disease, from Aβ oligomer recognition to neuronal death"

4. "Describe the mechanisms linking APOE ε4 genotype to Alzheimer's disease risk and progression"
```

### Query Set 2: Depth - Specific Mechanisms
```
5. "How does amyloid-beta interact with the nerve growth factor receptor (NGFR) pathway to induce neuronal apoptosis?"

6. "What is the role of microglial exosomes in tau protein propagation between neurons?"

7. "Describe the crosstalk between amyloid-beta and tau pathologies, including shared kinases and signaling pathways"

8. "Map the synaptic dysfunction mechanisms induced by Aβ oligomers, including LTP inhibition and receptor modulation"
```

### Query Set 3: Translational - Drug Discovery
```
9. "Identify all FDA-approved drugs that modulate GSK3 kinase activity and could potentially reduce tau hyperphosphorylation"

10. "Find drug repurposing candidates that target both amyloid production and neuroinflammation pathways"

11. "What experimental therapeutics targeting alpha-7 nicotinic receptors have been tested in Alzheimer's disease?"
```

### Query Set 4: Integration - Systems Biology
```
12. "Build a multi-scale model of Alzheimer's disease pathogenesis linking genetic variants (APP, PSEN1, APOE) to molecular pathways (amyloid, tau, inflammation) to clinical phenotypes (cognitive decline, neurodegeneration)"

13. "Identify shared molecular mechanisms between Alzheimer's disease and Parkinson's disease based on pathway overlap"

14. "Map the relationship between metabolic dysfunction (insulin signaling, glucose metabolism) and Alzheimer's disease pathology"
```

---

## 9. Evaluation Methodology

### Data Collection
1. **Extract NeuroMMSig BEL Graphs**: Use PyBEL to parse all AD mechanisms from GitHub repo
2. **Define Ground Truth**: Create reference dataset of:
   - All unique proteins/genes (nodes)
   - All unique relationships (edges)
   - All mechanistic subgraphs (pathways)
3. **Run Tetra Queries**: Execute all 14 queries and collect results
4. **Structure Tetra Output**: Extract entities and relationships from Tetra's knowledge graph

### Comparison Analysis
1. **Entity-Level**:
   - Precision: % of Tetra entities that appear in NeuroMMSig
   - Recall: % of NeuroMMSig entities recovered by Tetra
   - F1 Score: Harmonic mean of precision and recall

2. **Relationship-Level**:
   - Edge Precision: % of Tetra relationships validated in NeuroMMSig
   - Edge Recall: % of NeuroMMSig relationships found by Tetra
   - Relationship Type Accuracy: Do we get the directionality/type correct?

3. **Mechanism-Level**:
   - Mechanism Coverage: % of 124 mechanisms represented in Tetra output
   - Mechanism Completeness: Average % of entities per mechanism recovered
   - Novel Mechanism Discovery: Count of coherent mechanisms not in NeuroMMSig

4. **Temporal Advantage**:
   - Post-2019 Literature: Count relationships from papers published after AETIONOMY
   - Citation Analysis: % of recent high-impact AD papers referenced

### Qualitative Assessment
1. **Expert Review**: Have AD researcher assess biological validity of Tetra outputs
2. **Coherence**: Do the mechanisms form biologically plausible narratives?
3. **Novelty**: Are novel insights actually interesting/publishable?
4. **Clinical Relevance**: Do drug repurposing suggestions have therapeutic rationale?

---

## 10. Expected Outcomes

### Conservative Estimate
- **Entity Recall**: 65-75% of NeuroMMSig proteins/genes
- **Edge Recall**: 55-65% of BEL relationships
- **Mechanism Coverage**: 60-70% of mechanisms represented
- **Execution Time**: 20-40 minutes
- **Novel Insights**: 50-100 new relationships from recent literature

### Optimistic Estimate
- **Entity Recall**: 80-90%
- **Edge Recall**: 70-80%
- **Mechanism Coverage**: 75-85%
- **Execution Time**: 15-30 minutes
- **Novel Insights**: 100-200 new relationships
- **Literature Advantage**: 2019-2025 papers provide 20-30% new edges

### Competitive Advantages of Tetra
1. **Speed**: Minutes vs. years
2. **Recency**: Captures latest literature automatically
3. **Flexibility**: Easy to expand to new diseases/mechanisms
4. **Scalability**: Can process multiple diseases in parallel
5. **Transparency**: Provides source citations for all relationships
6. **Adaptability**: Can incorporate new data sources without retraining

---

## 11. Demo Narrative Structure

### Story Arc
1. **The Challenge**: "It took a team at Fraunhofer SCAI 5 years to manually curate the largest Alzheimer's knowledge graph"
2. **The Opportunity**: "Can our agentic system recapitulate this knowledge in minutes?"
3. **The Approach**: "We'll use the exact same seed proteins and mechanisms as our validation benchmark"
4. **The Execution**: [Live demo of Tetra running queries]
5. **The Results**: "Here's what we recovered..." [Show overlap statistics]
6. **The Advantage**: "And here are novel insights from papers published in the last 5 years that weren't available during AETIONOMY"
7. **The Impact**: "This demonstrates that autonomous AI agents can accelerate biomedical knowledge synthesis by 87,000x"

### Visualization Ideas
- **Network Overlap Graph**: Show NeuroMMSig (blue) vs Tetra (red) with overlap (purple)
- **Mechanism Heatmap**: 124 mechanisms × 2 systems showing coverage
- **Timeline**: 2014-2019 manual curation vs 2025 30-min automated extraction
- **Novel Discovery Highlight**: Call out specific interesting new relationships
- **Drug Repurposing Table**: Side-by-side comparison of candidates

---

## 12. Implementation Checklist

### Data Preparation
- [ ] Clone neurommsig-knowledge GitHub repository
- [ ] Install PyBEL and dependencies
- [ ] Extract all AD BEL graphs into structured format
- [ ] Create reference entity and relationship lists
- [ ] Catalog all 124 mechanistic subgraphs

### Tetra Execution
- [ ] Prepare 14 validation queries
- [ ] Set up Tetra to log all entity extractions
- [ ] Run queries and collect outputs
- [ ] Structure Tetra results in comparable format
- [ ] Extract literature citations from Tetra outputs

### Analysis
- [ ] Calculate precision/recall/F1 for entities
- [ ] Calculate precision/recall/F1 for relationships
- [ ] Compute mechanism coverage metrics
- [ ] Identify novel relationships with post-2019 citations
- [ ] Generate comparison visualizations

### Reporting
- [ ] Write technical validation report
- [ ] Create demo presentation slides
- [ ] Prepare network visualizations
- [ ] Document any interesting novel discoveries
- [ ] Draft potential publication abstract

---

## 13. Related Benchmark Opportunities

### Other AD Knowledge Graphs (For Extended Validation)

1. **AlzKB (Alzheimer's Knowledge Base) - 2024**
   - 118,902 entities
   - 1,309,527 relationships
   - 22 diverse data sources
   - Focus on drug discovery
   - [JMIR Publication](https://www.jmir.org/2024/1/e46777)

2. **ADKG (Alzheimer's Disease Knowledge Graph) - 2024**
   - 3,199,276 entity mentions
   - 633,733 triplets
   - ADERC corpus: 800 PubMed abstracts, 20,886 entities, 4,935 relationships
   - GPT-4 augmented annotations
   - [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11245034/)

3. **Alzforum Mutation Database**
   - Curated variants in APP, PSEN1, PSEN2, APOE, MAPT, SORL1, TREM2
   - Gold standard for pathogenic variant classification
   - [Alzforum](https://www.alzforum.org/mutations)

### Potential Multi-Disease Validation
- **Parkinson's Disease**: NeuroMMSig has 65 PD mechanisms (similar validation possible)
- **AD-T2DM Comorbidity**: Published analysis available for comparison
- **AD-PD Overlap**: Test cross-disease mechanism discovery

---

## 14. References and Sources

### Primary Sources
- [Fraunhofer SCAI Alzheimer's Research](https://www.scai.fraunhofer.de/en/about-us/staff/kodamullil.html)
- [Dr. Alpha Tom Kodamullil Profile](https://www.izb.fraunhofer.de/en/press/news-18-03-2025.html)
- [NeuroMMSig About Page](https://neurommsig.scai.fraunhofer.de/pathways/about)
- [Human Brain Pharmacome Overview](https://pharmacome.github.io/overview/)
- [AETIONOMY Completed IMI Projects](https://www.imi-neuronet.org/completed-projects/)

### Key Publications
- Kodamullil et al. (2015) - Alzheimer's & Dementia
- Domingo-Fernández et al. (2017) - Bioinformatics
- Kodamullil et al. (2020) - JAD (AD-T2DM comorbidity)
- Hoyt et al. (2019) - Database (PyBEL re-curation)

### Public Resources
- [NeuroMMSig GitHub](https://github.com/neurommsig/neurommsig-knowledge)
- [PyBEL Library](https://github.com/pybel/pybel)
- [BEL Commons](https://bel-commons.scai.fraunhofer.de)

### Recent Reviews
- [Unified Framework for AD Knowledge Graphs (2025)](https://www.mdpi.com/2076-3425/15/5/523)
- [AlzKB Publication (2024)](https://www.jmir.org/2024/1/e46777)
- [ADKG Publication (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11245034/)
- [Tau and neuroinflammation in AD (2023)](https://link.springer.com/article/10.1186/s12974-023-02853-3)
- [Amyloid-β Pathway in AD (2021)](https://www.nature.com/articles/s41380-021-01249-0)

---

## 15. Next Steps

### Immediate (This Week)
1. Set up PyBEL environment and extract NeuroMMSig data
2. Review BEL graph structure and entity/relationship schema
3. Prepare Tetra query templates and validation scripts
4. Define specific success metrics and thresholds

### Short-term (Next 2 Weeks)
1. Execute initial Tetra queries on AD core proteins
2. Perform preliminary overlap analysis
3. Identify gaps and refine queries
4. Collect preliminary statistics for demo

### Medium-term (Next Month)
1. Complete full validation analysis
2. Identify and investigate novel discoveries
3. Prepare comprehensive validation report
4. Create demo presentation materials

### Long-term (Next Quarter)
1. Consider publication of validation results
2. Expand to PD and other neurodegenerative diseases
3. Develop automated benchmark pipeline
4. Integrate NeuroMMSig as ongoing validation suite

---

## 16. Contact Information

### Fraunhofer SCAI Researchers
- **Dr. Alpha Tom Kodamullil**: Group Leader, Applied Semantics
  - Email: alpha.tom.kodamullil@scai.fraunhofer.de (likely)
  - Recently awarded €60,000 Early Career Grant for 2025-2026
  - Could be potential collaborator/validator

- **Prof. Martin Hofmann-Apitius**: Former supervisor, AETIONOMY lead
  - Scientific Coordination Board member for IMI Neuronet

### Potential Outreach
If validation results are strong, consider:
1. Sharing results with Kodamullil team for feedback
2. Proposing collaboration on novel discovery validation
3. Joint publication comparing manual vs. automated curation
4. Contributing novel relationships back to NeuroMMSig

---

**Last Updated**: 2025-12-03
**Prepared By**: Research analysis for Tetra V1 validation strategy
**Status**: Ready for implementation
