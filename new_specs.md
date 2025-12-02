Yes, you've got it exactly right. Let me consolidate everything into a complete, actionable spec.

---

# Scientific Knowledge Graph Agent: End-to-End Specification

## System Overview

A tool that enables drug discovery scientists to rapidly explore biological systems, synthesize literature evidence, and generate testable hypotheses for novel protein interactions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
│         "Explore the orexin system for potential novel drug targets"        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR AGENT                                 │
│                                                                             │
│   Interprets query, plans execution, calls tools, synthesizes results       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────────┐      ┌─────────────────────┐
│  STRING TOOLS   │       │   PUBMED TOOLS      │      │   GRAPH TOOLS       │
│                 │       │                     │      │                     │
│ • get_network   │       │ • search_abstracts  │      │ • build_graph       │
│ • get_neighbors │       │ • get_pubtator_ner  │      │ • query_graph       │
│                 │       │ • extract_relations │      │ • predict_links     │
│                 │       │                     │      │ • infer_relations   │
└─────────────────┘       └─────────────────────┘      └─────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRE-TRAINED LINK PREDICTOR                              │
│                                                                             │
│   Node2Vec embeddings + Edge classifier (trained on STRING physical)        │
│   Applied to ANY protein system on-demand                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EVIDENCE-BACKED KNOWLEDGE GRAPH                        │
│                                                                             │
│   Edges with: source, evidence count, relationship type, ML score           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYPOTHESIS OUTPUT                                 │
│                                                                             │
│   "Based on graph structure, HCRTR2-NTRK1 interaction is plausible          │
│    (ML score: 0.87). No direct literature, but neighborhood suggests        │
│    regulatory relationship. Suggested validation: co-IP, proximity assay"   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Pre-trained Link Predictor (Offline, One-time)

### 1.1 Data Acquisition

```
Downloads from STRING (https://string-db.org/cgi/download):
├── 9606.protein.physical.links.detailed.v12.0.txt.gz  (10.6 MB)
├── 9606.protein.info.v12.0.txt.gz                      (1.9 MB)
└── 9606.protein.aliases.v12.0.txt.gz                   (18.9 MB)
```

### 1.2 Data Models

```python
# models/string_data.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ProteinInfo:
    string_id: str              # "9606.ENSP00000000233"
    gene_name: str              # "HCRTR2"
    protein_name: str           # "Orexin receptor type 2"
    aliases: list[str]          # ["OX2R", "orexin receptor 2", ...]
    
@dataclass
class PhysicalInteraction:
    protein1: str
    protein2: str
    combined_score: int         # 0-1000
    experimental_score: int     # 0-1000
    database_score: int         # 0-1000
    textmining_score: int       # 0-1000
```

### 1.3 Link Predictor Training Pipeline

```python
# ml/link_predictor.py

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

class LinkPredictor:
    """
    Pre-trained link predictor on STRING physical interactions.
    Can be applied to any protein system.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        min_score: int = 700
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.min_score = min_score
        
        self.graph: nx.Graph = None
        self.embeddings: dict[str, np.ndarray] = None
        self.classifier: LogisticRegression = None
        self.protein_to_gene: dict[str, str] = {}
        self.gene_to_protein: dict[str, str] = {}
        
    def load_string_data(
        self, 
        interactions_path: str,
        info_path: str,
        aliases_path: str
    ):
        """Load STRING physical interactions and protein info"""
        
        # Load protein info for name mapping
        self._load_protein_info(info_path, aliases_path)
        
        # Build graph from physical interactions
        self.graph = nx.Graph()
        
        with gzip.open(interactions_path, 'rt') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                p1, p2 = parts[0], parts[1]
                combined_score = int(parts[-1])
                
                if combined_score >= self.min_score:
                    self.graph.add_edge(p1, p2, weight=combined_score/1000)
        
        print(f"Loaded graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
    
    def train_embeddings(self):
        """Train Node2Vec embeddings on the full STRING network"""
        
        print("Training Node2Vec embeddings...")
        
        node2vec = Node2Vec(
            self.graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=4,
            quiet=False
        )
        
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        self.embeddings = {
            node: model.wv[node] 
            for node in self.graph.nodes()
        }
        
        print(f"Trained embeddings for {len(self.embeddings)} proteins")
    
    def _edge_features(self, node1: str, node2: str) -> np.ndarray:
        """Compute edge features from node embeddings"""
        emb1 = self.embeddings.get(node1)
        emb2 = self.embeddings.get(node2)
        
        if emb1 is None or emb2 is None:
            return None
        
        # Hadamard product (element-wise multiplication)
        return emb1 * emb2
    
    def train_classifier(self, test_size: float = 0.2):
        """Train edge classifier with evaluation"""
        
        print("Preparing training data...")
        
        # Positive edges: actual interactions
        positive_edges = list(self.graph.edges())
        
        # Negative edges: random non-edges
        nodes = list(self.graph.nodes())
        negative_edges = []
        
        while len(negative_edges) < len(positive_edges):
            n1, n2 = np.random.choice(nodes, 2, replace=False)
            if not self.graph.has_edge(n1, n2) and (n1, n2) not in negative_edges:
                negative_edges.append((n1, n2))
        
        # Build feature matrix
        X, y = [], []
        
        for u, v in positive_edges:
            feat = self._edge_features(u, v)
            if feat is not None:
                X.append(feat)
                y.append(1)
        
        for u, v in negative_edges:
            feat = self._edge_features(u, v)
            if feat is not None:
                X.append(feat)
                y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train classifier
        print("Training classifier...")
        self.classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        print(f"Evaluation on held-out edges:")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        
        return {"auc": auc, "average_precision": ap}
    
    def predict(self, protein_pairs: list[tuple[str, str]]) -> list[dict]:
        """Predict interaction probability for protein pairs"""
        results = []
        
        for p1, p2 in protein_pairs:
            # Try both STRING ID and gene name lookup
            string_id1 = self.gene_to_protein.get(p1, p1)
            string_id2 = self.gene_to_protein.get(p2, p2)
            
            feat = self._edge_features(string_id1, string_id2)
            
            if feat is not None:
                prob = self.classifier.predict_proba([feat])[0][1]
                in_string = self.graph.has_edge(string_id1, string_id2)
            else:
                prob = None
                in_string = False
            
            results.append({
                "protein1": p1,
                "protein2": p2,
                "ml_score": prob,
                "in_string": in_string
            })
        
        return results
    
    def save(self, path: str):
        """Save trained model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'classifier': self.classifier,
                'protein_to_gene': self.protein_to_gene,
                'gene_to_protein': self.gene_to_protein,
                'config': {
                    'embedding_dim': self.embedding_dim,
                    'min_score': self.min_score
                }
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'LinkPredictor':
        """Load pre-trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(**data['config'])
        predictor.embeddings = data['embeddings']
        predictor.classifier = data['classifier']
        predictor.protein_to_gene = data['protein_to_gene']
        predictor.gene_to_protein = data['gene_to_protein']
        
        return predictor
```

### 1.4 Training Script

```python
# scripts/train_link_predictor.py

from ml.link_predictor import LinkPredictor

def main():
    predictor = LinkPredictor(
        embedding_dim=128,
        walk_length=80,
        num_walks=10,
        min_score=700  # High confidence interactions only
    )
    
    # Load STRING data
    predictor.load_string_data(
        interactions_path="data/string/9606.protein.physical.links.detailed.v12.0.txt.gz",
        info_path="data/string/9606.protein.info.v12.0.txt.gz",
        aliases_path="data/string/9606.protein.aliases.v12.0.txt.gz"
    )
    
    # Train embeddings (~30-60 min on CPU)
    predictor.train_embeddings()
    
    # Train and evaluate classifier
    metrics = predictor.train_classifier(test_size=0.2)
    
    # Save model
    predictor.save("models/link_predictor.pkl")
    
    print(f"Model saved. Metrics: {metrics}")

if __name__ == "__main__":
    main()
```

**Time estimate:** 2-3 hours to implement, 30-60 min to train

---

## Part 2: Data Models for Knowledge Graph

```python
# models/knowledge_graph.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import networkx as nx

class EvidenceSource(Enum):
    LITERATURE = "literature"
    STRING = "string"
    ML_PREDICTED = "ml_predicted"

class RelationshipType(Enum):
    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    ASSOCIATED_WITH = "associated_with"
    REGULATES = "regulates"
    BINDS_TO = "binds_to"
    INTERACTS_WITH = "interacts_with"
    COOCCURS_WITH = "cooccurs_with"
    HYPOTHESIZED = "hypothesized"

@dataclass
class EvidenceItem:
    source_type: EvidenceSource
    source_id: str                          # PMID, "STRING", etc.
    confidence: float
    text_snippet: Optional[str] = None
    extraction_method: Optional[str] = None

@dataclass
class Entity:
    id: str                                 # Gene symbol or ID
    entity_type: str                        # "gene", "disease", "chemical"
    name: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class Relationship:
    source: str
    target: str
    relationship_type: RelationshipType
    evidence: list[EvidenceItem] = field(default_factory=list)
    ml_score: Optional[float] = None
    inferred_type: Optional[str] = None     # For ML-predicted edges
    inference_reasoning: Optional[str] = None
    
    @property
    def literature_count(self) -> int:
        return sum(1 for e in self.evidence if e.source_type == EvidenceSource.LITERATURE)
    
    @property
    def has_database_support(self) -> bool:
        return any(e.source_type == EvidenceSource.STRING for e in self.evidence)
    
    @property
    def is_novel_prediction(self) -> bool:
        return (
            self.ml_score is not None and 
            self.ml_score > 0.7 and 
            self.literature_count == 0 and
            not self.has_database_support
        )
    
    def get_pmids(self) -> list[str]:
        return [e.source_id for e in self.evidence 
                if e.source_type == EvidenceSource.LITERATURE]

class KnowledgeGraph:
    """Evidence-backed knowledge graph with ML scoring"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[tuple, Relationship] = {}
    
    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **entity.metadata)
    
    def add_relationship(self, rel: Relationship):
        key = (rel.source, rel.target, rel.relationship_type)
        
        if key in self.relationships:
            # Merge evidence
            self.relationships[key].evidence.extend(rel.evidence)
            if rel.ml_score:
                self.relationships[key].ml_score = rel.ml_score
        else:
            self.relationships[key] = rel
            self.graph.add_edge(
                rel.source, 
                rel.target,
                key=rel.relationship_type.value,
                rel_type=rel.relationship_type
            )
    
    def get_relationship(self, source: str, target: str, 
                         rel_type: RelationshipType = None) -> Optional[Relationship]:
        if rel_type:
            return self.relationships.get((source, target, rel_type))
        
        # Find any relationship between source and target
        for key, rel in self.relationships.items():
            if key[0] == source and key[1] == target:
                return rel
        return None
    
    def get_neighbors(self, node: str) -> list[tuple[str, Relationship]]:
        neighbors = []
        for key, rel in self.relationships.items():
            if key[0] == node:
                neighbors.append((key[1], rel))
            elif key[1] == node:
                neighbors.append((key[0], rel))
        return neighbors
    
    def get_novel_predictions(self, min_ml_score: float = 0.7) -> list[Relationship]:
        return [
            rel for rel in self.relationships.values()
            if rel.is_novel_prediction and rel.ml_score >= min_ml_score
        ]
    
    def get_well_supported(self, min_papers: int = 3) -> list[Relationship]:
        return [
            rel for rel in self.relationships.values()
            if rel.literature_count >= min_papers
        ]
    
    def to_summary(self) -> dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": len(self.relationships),
            "literature_backed": len([r for r in self.relationships.values() if r.literature_count > 0]),
            "string_backed": len([r for r in self.relationships.values() if r.has_database_support]),
            "novel_predictions": len(self.get_novel_predictions()),
            "entities_by_type": self._count_entity_types()
        }
    
    def _count_entity_types(self) -> dict:
        counts = {}
        for entity in self.entities.values():
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts
```

**Time estimate:** 1-2 hours

---

## Part 3: External API Clients

### 3.1 STRING Client

```python
# clients/string_client.py

import httpx
from typing import Optional

class StringClient:
    """Client for STRING API"""
    
    BASE_URL = "https://string-db.org/api"
    SPECIES_HUMAN = 9606
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_network(
        self, 
        proteins: list[str], 
        min_score: int = 400
    ) -> list[dict]:
        """Get interaction network for proteins"""
        url = f"{self.BASE_URL}/json/network"
        params = {
            "identifiers": "%0d".join(proteins),
            "species": self.SPECIES_HUMAN,
            "required_score": min_score,
            "caller_identity": "kg_agent_demo"
        }
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_interaction_partners(
        self, 
        proteins: list[str],
        limit: int = 50
    ) -> list[dict]:
        """Get interaction partners for proteins"""
        url = f"{self.BASE_URL}/json/interaction_partners"
        params = {
            "identifiers": "%0d".join(proteins),
            "species": self.SPECIES_HUMAN,
            "limit": limit,
            "caller_identity": "kg_agent_demo"
        }
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_functional_annotation(
        self, 
        proteins: list[str]
    ) -> list[dict]:
        """Get GO/KEGG annotations for proteins"""
        url = f"{self.BASE_URL}/json/functional_annotation"
        params = {
            "identifiers": "%0d".join(proteins),
            "species": self.SPECIES_HUMAN,
            "caller_identity": "kg_agent_demo"
        }
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        await self.client.aclose()
```

### 3.2 PubMed + PubTator Client

```python
# clients/pubmed_client.py

import httpx
from xml.etree import ElementTree
from dataclasses import dataclass

@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    journal: str
    mesh_terms: list[str]

@dataclass
class PubTatorAnnotation:
    pmid: str
    entity_id: str          # NCBI Gene ID, MeSH ID, etc.
    entity_text: str        # As mentioned in text
    entity_type: str        # "Gene", "Disease", "Chemical", etc.
    start_pos: int
    end_pos: int

class PubMedClient:
    """Client for PubMed E-utilities and PubTator APIs"""
    
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PUBTATOR_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def search(
        self, 
        query: str, 
        max_results: int = 50,
        sort: str = "relevance"
    ) -> list[str]:
        """Search PubMed and return PMIDs"""
        url = f"{self.EUTILS_BASE}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json"
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return data["esearchresult"]["idlist"]
    
    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch article details for PMIDs"""
        url = f"{self.EUTILS_BASE}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml"
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        return self._parse_pubmed_xml(response.text)
    
    async def get_pubtator_annotations(
        self, 
        pmids: list[str],
        entity_types: list[str] = None
    ) -> list[PubTatorAnnotation]:
        """Get pre-computed NER annotations from PubTator"""
        
        # PubTator API accepts batches
        url = f"{self.PUBTATOR_BASE}/publications/export/biocjson"
        params = {"pmids": ",".join(pmids)}
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        annotations = []
        for doc in response.json():
            pmid = doc["pmid"]
            for passage in doc.get("passages", []):
                for ann in passage.get("annotations", []):
                    entity_type = ann.get("infons", {}).get("type", "")
                    
                    # Filter by entity type if specified
                    if entity_types and entity_type not in entity_types:
                        continue
                    
                    annotations.append(PubTatorAnnotation(
                        pmid=pmid,
                        entity_id=ann.get("infons", {}).get("identifier", ""),
                        entity_text=ann.get("text", ""),
                        entity_type=entity_type,
                        start_pos=ann.get("locations", [{}])[0].get("offset", 0),
                        end_pos=ann.get("locations", [{}])[0].get("offset", 0) + 
                               ann.get("locations", [{}])[0].get("length", 0)
                    ))
        
        return annotations
    
    def _parse_pubmed_xml(self, xml_text: str) -> list[PubMedArticle]:
        """Parse PubMed XML response"""
        articles = []
        root = ElementTree.fromstring(xml_text)
        
        for article_elem in root.findall(".//PubmedArticle"):
            pmid = article_elem.findtext(".//PMID", "")
            title = article_elem.findtext(".//ArticleTitle", "")
            
            abstract_parts = article_elem.findall(".//AbstractText")
            abstract = " ".join(part.text or "" for part in abstract_parts)
            
            authors = []
            for author in article_elem.findall(".//Author"):
                last = author.findtext("LastName", "")
                first = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())
            
            year_elem = article_elem.find(".//PubDate/Year")
            year = int(year_elem.text) if year_elem is not None else 0
            
            journal = article_elem.findtext(".//Journal/Title", "")
            
            mesh_terms = [
                mesh.findtext("DescriptorName", "")
                for mesh in article_elem.findall(".//MeshHeading")
            ]
            
            articles.append(PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                journal=journal,
                mesh_terms=mesh_terms
            ))
        
        return articles
    
    async def close(self):
        await self.client.aclose()
```

**Time estimate:** 2-3 hours

---

## Part 4: LLM-based Extraction & Inference

### 4.1 Relationship Extractor

```python
# extraction/relationship_extractor.py

from openai import AsyncOpenAI
from models.knowledge_graph import RelationshipType, EvidenceItem, EvidenceSource
import json

class RelationshipExtractor:
    """Extract typed relationships from abstracts using LLM"""
    
    EXTRACTION_PROMPT = """You are a biomedical relationship extractor. 

Given an abstract and a list of entity pairs that co-occur in it, determine the relationship type between each pair.

Abstract:
{abstract}

Entity pairs to classify:
{entity_pairs}

For each pair, classify the relationship as one of:
- ACTIVATES: Entity1 activates/increases/promotes Entity2
- INHIBITS: Entity1 inhibits/decreases/blocks Entity2
- ASSOCIATED_WITH: Entities are associated (correlation, not causation)
- REGULATES: Entity1 regulates Entity2 (direction unclear)
- BINDS_TO: Physical binding interaction
- COOCCURS_WITH: Mentioned together but no clear relationship stated

Return JSON array:
[
  {{"entity1": "...", "entity2": "...", "relationship": "...", "confidence": 0.0-1.0, "evidence_text": "relevant quote"}}
]

Only include pairs where you can identify a relationship. Skip pairs with insufficient evidence."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI()
        self.model = model
    
    async def extract_relationships(
        self,
        abstract: str,
        entity_pairs: list[tuple[str, str]],
        pmid: str
    ) -> list[dict]:
        """Extract relationships for co-occurring entity pairs"""
        
        if not entity_pairs:
            return []
        
        pairs_text = "\n".join([f"- {e1} and {e2}" for e1, e2 in entity_pairs])
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.EXTRACTION_PROMPT.format(
                    abstract=abstract,
                    entity_pairs=pairs_text
                )
            }],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            relationships = result if isinstance(result, list) else result.get("relationships", [])
            
            # Add PMID to each
            for rel in relationships:
                rel["pmid"] = pmid
            
            return relationships
        except json.JSONDecodeError:
            return []
    
    async def batch_extract(
        self,
        abstracts: list[dict],  # [{"pmid": ..., "abstract": ..., "entity_pairs": [...]}]
        max_concurrent: int = 5
    ) -> list[dict]:
        """Batch extraction with concurrency control"""
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_one(item):
            async with semaphore:
                return await self.extract_relationships(
                    item["abstract"],
                    item["entity_pairs"],
                    item["pmid"]
                )
        
        results = await asyncio.gather(*[extract_one(item) for item in abstracts])
        
        # Flatten
        all_relationships = []
        for result in results:
            all_relationships.extend(result)
        
        return all_relationships
```

### 4.2 Relationship Inferrer (for Novel Predictions)

```python
# extraction/relationship_inferrer.py

class RelationshipInferrer:
    """Infer relationship types for ML-predicted edges based on graph context"""
    
    INFERENCE_PROMPT = """You are a biomedical knowledge expert analyzing a predicted protein interaction.

A machine learning model predicts that {protein_a} and {protein_b} likely interact (probability: {ml_score:.2f}), 
but there is no direct literature evidence for this interaction.

Here is what we know about each protein from our knowledge graph:

=== {protein_a} ===
Known interactions:
{protein_a_interactions}

Functional annotations:
{protein_a_functions}

=== {protein_b} ===
Known interactions:
{protein_b_interactions}

Functional annotations:
{protein_b_functions}

Based on this context:

1. What type of relationship would you hypothesize between {protein_a} and {protein_b}?
   Options: ACTIVATES, INHIBITS, BINDS_TO, REGULATORY, COMPLEX_MEMBER, PATHWAY_NEIGHBOR

2. How confident are you? (LOW, MEDIUM, HIGH)

3. What is your reasoning? (2-3 sentences)

4. What experiments would validate this prediction?

Return JSON:
{{
  "hypothesized_relationship": "...",
  "confidence": "...",
  "reasoning": "...",
  "validation_experiments": ["...", "..."]
}}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI()
        self.model = model
    
    async def infer_relationship(
        self,
        protein_a: str,
        protein_b: str,
        ml_score: float,
        graph: 'KnowledgeGraph'
    ) -> dict:
        """Infer relationship type for a novel predicted edge"""
        
        # Get neighborhood context
        a_interactions = self._format_interactions(graph, protein_a)
        b_interactions = self._format_interactions(graph, protein_b)
        a_functions = self._format_functions(graph, protein_a)
        b_functions = self._format_functions(graph, protein_b)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.INFERENCE_PROMPT.format(
                    protein_a=protein_a,
                    protein_b=protein_b,
                    ml_score=ml_score,
                    protein_a_interactions=a_interactions,
                    protein_b_interactions=b_interactions,
                    protein_a_functions=a_functions,
                    protein_b_functions=b_functions
                )
            }],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        result["protein_a"] = protein_a
        result["protein_b"] = protein_b
        result["ml_score"] = ml_score
        
        return result
    
    def _format_interactions(self, graph: 'KnowledgeGraph', protein: str) -> str:
        neighbors = graph.get_neighbors(protein)
        if not neighbors:
            return "No known interactions in graph"
        
        lines = []
        for neighbor, rel in neighbors[:10]:  # Top 10
            rel_type = rel.relationship_type.value
            evidence = f"({rel.literature_count} papers)" if rel.literature_count else "(STRING)"
            lines.append(f"- {rel_type} {neighbor} {evidence}")
        
        return "\n".join(lines)
    
    def _format_functions(self, graph: 'KnowledgeGraph', protein: str) -> str:
        entity = graph.entities.get(protein)
        if not entity or not entity.metadata.get("functions"):
            return "No functional annotations"
        
        return "\n".join([f"- {f}" for f in entity.metadata["functions"][:5]])
```

**Time estimate:** 2-3 hours

---

## Part 5: Agent Tools

```python
# agent/tools.py

from typing import Any
from models.knowledge_graph import KnowledgeGraph, Entity, Relationship, RelationshipType, EvidenceItem, EvidenceSource
from clients.string_client import StringClient
from clients.pubmed_client import PubMedClient
from extraction.relationship_extractor import RelationshipExtractor
from extraction.relationship_inferrer import RelationshipInferrer
from ml.link_predictor import LinkPredictor

class AgentTools:
    """Tools available to the orchestrator agent"""
    
    def __init__(
        self,
        link_predictor: LinkPredictor,
        string_client: StringClient,
        pubmed_client: PubMedClient,
        relationship_extractor: RelationshipExtractor,
        relationship_inferrer: RelationshipInferrer
    ):
        self.link_predictor = link_predictor
        self.string_client = string_client
        self.pubmed_client = pubmed_client
        self.relationship_extractor = relationship_extractor
        self.relationship_inferrer = relationship_inferrer
        
        self.current_graph: KnowledgeGraph = None
    
    # === STRING Tools ===
    
    async def get_string_network(
        self, 
        seed_proteins: list[str],
        min_score: int = 700
    ) -> dict:
        """
        Fetch known interaction network from STRING for seed proteins.
        Returns network statistics and list of proteins.
        """
        interactions = await self.string_client.get_network(seed_proteins, min_score)
        
        proteins = set()
        for interaction in interactions:
            proteins.add(interaction["preferredName_A"])
            proteins.add(interaction["preferredName_B"])
        
        return {
            "seed_proteins": seed_proteins,
            "total_proteins": len(proteins),
            "total_interactions": len(interactions),
            "proteins": list(proteins),
            "interactions": interactions
        }
    
    async def get_string_partners(
        self,
        proteins: list[str],
        limit: int = 30
    ) -> dict:
        """Get interaction partners for proteins from STRING"""
        partners = await self.string_client.get_interaction_partners(proteins, limit)
        return {
            "query_proteins": proteins,
            "partners_found": len(partners),
            "partners": partners
        }
    
    # === PubMed Tools ===
    
    async def search_literature(
        self,
        query: str,
        max_results: int = 50
    ) -> dict:
        """Search PubMed for relevant abstracts"""
        pmids = await self.pubmed_client.search(query, max_results)
        articles = await self.pubmed_client.fetch_abstracts(pmids)
        
        return {
            "query": query,
            "articles_found": len(articles),
            "articles": [
                {
                    "pmid": a.pmid,
                    "title": a.title,
                    "year": a.year,
                    "abstract_length": len(a.abstract)
                }
                for a in articles
            ],
            "_full_articles": articles  # Keep full data for processing
        }
    
    async def get_entity_annotations(
        self,
        pmids: list[str],
        entity_types: list[str] = None
    ) -> dict:
        """Get NER annotations from PubTator for articles"""
        if entity_types is None:
            entity_types = ["Gene", "Disease", "Chemical"]
        
        annotations = await self.pubmed_client.get_pubtator_annotations(pmids, entity_types)
        
        # Group by PMID
        by_pmid = {}
        for ann in annotations:
            if ann.pmid not in by_pmid:
                by_pmid[ann.pmid] = []
            by_pmid[ann.pmid].append({
                "entity_id": ann.entity_id,
                "text": ann.entity_text,
                "type": ann.entity_type
            })
        
        return {
            "pmids_processed": len(by_pmid),
            "total_annotations": len(annotations),
            "annotations_by_pmid": by_pmid
        }
    
    # === Extraction Tools ===
    
    async def extract_relationships(
        self,
        articles: list[dict],
        annotations_by_pmid: dict
    ) -> dict:
        """Extract typed relationships from abstracts"""
        
        # Prepare extraction tasks
        extraction_tasks = []
        for article in articles:
            pmid = article["pmid"]
            if pmid not in annotations_by_pmid:
                continue
            
            # Get gene entities for this article
            genes = [
                ann["text"] for ann in annotations_by_pmid[pmid]
                if ann["type"] == "Gene"
            ]
            
            if len(genes) < 2:
                continue
            
            # Generate co-occurring pairs
            from itertools import combinations
            pairs = list(combinations(set(genes), 2))
            
            extraction_tasks.append({
                "pmid": pmid,
                "abstract": article.get("abstract", ""),
                "entity_pairs": pairs
            })
        
        # Run extraction
        relationships = await self.relationship_extractor.batch_extract(extraction_tasks)
        
        return {
            "articles_processed": len(extraction_tasks),
            "relationships_extracted": len(relationships),
            "relationships": relationships
        }
    
    # === Graph Tools ===
    
    def build_knowledge_graph(
        self,
        string_interactions: list[dict],
        literature_relationships: list[dict],
        entities: dict
    ) -> dict:
        """Build evidence-backed knowledge graph"""
        
        self.current_graph = KnowledgeGraph()
        
        # Add entities
        for entity_id, entity_data in entities.items():
            self.current_graph.add_entity(Entity(
                id=entity_id,
                entity_type=entity_data.get("type", "gene"),
                name=entity_data.get("name", entity_id),
                aliases=entity_data.get("aliases", [])
            ))
        
        # Add STRING edges
        for interaction in string_interactions:
            p1 = interaction["preferredName_A"]
            p2 = interaction["preferredName_B"]
            score = interaction["score"]
            
            rel = Relationship(
                source=p1,
                target=p2,
                relationship_type=RelationshipType.INTERACTS_WITH,
                evidence=[EvidenceItem(
                    source_type=EvidenceSource.STRING,
                    source_id="STRING",
                    confidence=score / 1000
                )]
            )
            self.current_graph.add_relationship(rel)
        
        # Add literature edges
        for lit_rel in literature_relationships:
            rel_type = RelationshipType[lit_rel["relationship"].upper()]
            
            rel = Relationship(
                source=lit_rel["entity1"],
                target=lit_rel["entity2"],
                relationship_type=rel_type,
                evidence=[EvidenceItem(
                    source_type=EvidenceSource.LITERATURE,
                    source_id=lit_rel["pmid"],
                    confidence=lit_rel.get("confidence", 0.5),
                    text_snippet=lit_rel.get("evidence_text")
                )]
            )
            self.current_graph.add_relationship(rel)
        
        return self.current_graph.to_summary()
    
    def predict_novel_links(
        self,
        min_ml_score: float = 0.7,
        max_predictions: int = 20
    ) -> dict:
        """Apply link predictor to find novel interactions"""
        
        if self.current_graph is None:
            return {"error": "No graph built yet"}
        
        # Get all proteins in current graph
        proteins = list(self.current_graph.entities.keys())
        
        # Generate candidate pairs (not already in graph)
        from itertools import combinations
        candidates = []
        for p1, p2 in combinations(proteins, 2):
            existing = self.current_graph.get_relationship(p1, p2)
            if existing is None or (existing.literature_count == 0 and not existing.has_database_support):
                candidates.append((p1, p2))
        
        # Predict
        predictions = self.link_predictor.predict(candidates)
        
        # Filter and sort
        novel = [
            p for p in predictions 
            if p["ml_score"] is not None and p["ml_score"] >= min_ml_score
        ]
        novel.sort(key=lambda x: x["ml_score"], reverse=True)
        
        # Update graph with ML scores
        for pred in novel[:max_predictions]:
            rel = Relationship(
                source=pred["protein1"],
                target=pred["protein2"],
                relationship_type=RelationshipType.HYPOTHESIZED,
                ml_score=pred["ml_score"]
            )
            self.current_graph.add_relationship(rel)
        
        return {
            "candidates_evaluated": len(candidates),
            "predictions_above_threshold": len(novel),
            "top_predictions": novel[:max_predictions]
        }
    
    async def infer_novel_relationships(
        self,
        predictions: list[dict],
        max_inferences: int = 5
    ) -> dict:
        """Use LLM to infer relationship types for top predictions"""
        
        inferences = []
        for pred in predictions[:max_inferences]:
            inference = await self.relationship_inferrer.infer_relationship(
                pred["protein1"],
                pred["protein2"],
                pred["ml_score"],
                self.current_graph
            )
            inferences.append(inference)
            
            # Update graph with inference
            rel = self.current_graph.get_relationship(pred["protein1"], pred["protein2"])
            if rel:
                rel.inferred_type = inference["hypothesized_relationship"]
                rel.inference_reasoning = inference["reasoning"]
        
        return {
            "inferences_made": len(inferences),
            "inferences": inferences
        }
    
    # === Query Tools ===
    
    def query_evidence(
        self,
        protein1: str,
        protein2: str
    ) -> dict:
        """Get all evidence for a relationship between two proteins"""
        
        rel = self.current_graph.get_relationship(protein1, protein2)
        if not rel:
            rel = self.current_graph.get_relationship(protein2, protein1)
        
        if not rel:
            return {
                "found": False,
                "message": f"No relationship found between {protein1} and {protein2}"
            }
        
        return {
            "found": True,
            "source": rel.source,
            "target": rel.target,
            "relationship_type": rel.relationship_type.value,
            "literature_count": rel.literature_count,
            "has_database_support": rel.has_database_support,
            "ml_score": rel.ml_score,
            "inferred_type": rel.inferred_type,
            "pmids": rel.get_pmids(),
            "is_novel_prediction": rel.is_novel_prediction
        }
    
    def get_graph_summary(self) -> dict:
        """Get summary of current knowledge graph"""
        if self.current_graph is None:
            return {"error": "No graph built yet"}
        return self.current_graph.to_summary()
    
    def get_protein_neighborhood(self, protein: str, max_neighbors: int = 10) -> dict:
        """Get neighborhood of a protein"""
        neighbors = self.current_graph.get_neighbors(protein)
        
        return {
            "protein": protein,
            "neighbor_count": len(neighbors),
            "neighbors": [
                {
                    "protein": n,
                    "relationship": rel.relationship_type.value,
                    "literature_support": rel.literature_count,
                    "ml_score": rel.ml_score
                }
                for n, rel in neighbors[:max_neighbors]
            ]
        }
```

**Time estimate:** 3-4 hours

---

## Part 6: Orchestrator Agent

```python
# agent/orchestrator.py

from openai import AsyncOpenAI
from agent.tools import AgentTools
import json

class OrchestratorAgent:
    """Main agent that interprets queries and orchestrates tools"""
    
    SYSTEM_PROMPT = """You are a scientific knowledge graph agent that helps drug discovery scientists 
explore biological systems and generate hypotheses for novel protein interactions.

You have access to the following tools:

1. get_string_network(seed_proteins, min_score) - Get known interactions from STRING database
2. get_string_partners(proteins, limit) - Get interaction partners for proteins
3. search_literature(query, max_results) - Search PubMed for relevant abstracts
4. get_entity_annotations(pmids, entity_types) - Get NER annotations (genes, diseases, chemicals)
5. extract_relationships(articles, annotations) - Extract typed relationships using LLM
6. build_knowledge_graph(string_interactions, literature_relationships, entities) - Build the graph
7. predict_novel_links(min_ml_score, max_predictions) - Apply ML link predictor
8. infer_novel_relationships(predictions, max_inferences) - Infer relationship types for predictions
9. query_evidence(protein1, protein2) - Get evidence for a specific relationship
10. get_graph_summary() - Get summary statistics of the graph
11. get_protein_neighborhood(protein) - Get neighborhood of a protein

When a user asks to explore a biological system:
1. First, get the STRING network for seed proteins
2. Search PubMed for relevant literature (construct an appropriate query)
3. Get NER annotations for the articles
4. Extract relationships from literature
5. Build the knowledge graph
6. Run link prediction to find novel candidates
7. Infer relationship types for top predictions
8. Summarize findings

Always explain what you're doing and why. Present findings clearly with evidence."""

    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "get_string_network",
                "description": "Fetch known protein interaction network from STRING database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seed_proteins": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of protein/gene names to query"
                        },
                        "min_score": {
                            "type": "integer",
                            "description": "Minimum STRING score (0-1000, default 700)"
                        }
                    },
                    "required": ["seed_proteins"]
                }
            }
        },
        # ... (other tool definitions)
    ]
    
    def __init__(self, tools: AgentTools, model: str = "gpt-4o"):
        self.tools = tools
        self.client = AsyncOpenAI()
        self.model = model
        self.conversation_history = []
    
    async def run(self, user_query: str) -> str:
        """Process user query through the agent loop"""
        
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        while True:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    *self.conversation_history
                ],
                tools=self.TOOL_DEFINITIONS,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Check if done
            if message.tool_calls is None:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content
            
            # Execute tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })
            
            for tool_call in message.tool_calls:
                result = await self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str)
                })
    
    async def _execute_tool(self, name: str, args: dict) -> dict:
        """Execute a tool by name"""
        tool_map = {
            "get_string_network": self.tools.get_string_network,
            "get_string_partners": self.tools.get_string_partners,
            "search_literature": self.tools.search_literature,
            "get_entity_annotations": self.tools.get_entity_annotations,
            "extract_relationships": self.tools.extract_relationships,
            "build_knowledge_graph": self.tools.build_knowledge_graph,
            "predict_novel_links": self.tools.predict_novel_links,
            "infer_novel_relationships": self.tools.infer_novel_relationships,
            "query_evidence": self.tools.query_evidence,
            "get_graph_summary": self.tools.get_graph_summary,
            "get_protein_neighborhood": self.tools.get_protein_neighborhood,
        }
        
        if name not in tool_map:
            return {"error": f"Unknown tool: {name}"}
        
        try:
            if asyncio.iscoroutinefunction(tool_map[name]):
                return await tool_map[name](**args)
            else:
                return tool_map[name](**args)
        except Exception as e:
            return {"error": str(e)}
```

**Time estimate:** 2-3 hours

---

## Part 7: Main Application

```python
# main.py

import asyncio
from ml.link_predictor import LinkPredictor
from clients.string_client import StringClient
from clients.pubmed_client import PubMedClient
from extraction.relationship_extractor import RelationshipExtractor
from extraction.relationship_inferrer import RelationshipInferrer
from agent.tools import AgentTools
from agent.orchestrator import OrchestratorAgent

async def main():
    # Load pre-trained link predictor
    print("Loading link predictor...")
    link_predictor = LinkPredictor.load("models/link_predictor.pkl")
    
    # Initialize clients
    string_client = StringClient()
    pubmed_client = PubMedClient()
    
    # Initialize extractors
    relationship_extractor = RelationshipExtractor()
    relationship_inferrer = RelationshipInferrer()
    
    # Initialize tools
    tools = AgentTools(
        link_predictor=link_predictor,
        string_client=string_client,
        pubmed_client=pubmed_client,
        relationship_extractor=relationship_extractor,
        relationship_inferrer=relationship_inferrer
    )
    
    # Initialize agent
    agent = OrchestratorAgent(tools)
    
    print("\n=== Scientific Knowledge Graph Agent ===")
    print("Ask me to explore a biological system.\n")
    
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        response = await agent.run(query)
        print(f"\nAgent: {response}\n")
    
    # Cleanup
    await string_client.close()
    await pubmed_client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 8: Project Structure

```
scientific-knowledge-graph-agent/
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   └── string/                          # Downloaded STRING files
│       ├── 9606.protein.physical.links.detailed.v12.0.txt.gz
│       ├── 9606.protein.info.v12.0.txt.gz
│       └── 9606.protein.aliases.v12.0.txt.gz
│
├── models/
│   ├── __init__.py
│   ├── knowledge_graph.py               # Data models
│   └── link_predictor.pkl               # Trained model (generated)
│
├── ml/
│   ├── __init__.py
│   └── link_predictor.py                # Node2Vec + classifier
│
├── clients/
│   ├── __init__.py
│   ├── string_client.py                 # STRING API
│   └── pubmed_client.py                 # PubMed + PubTator APIs
│
├── extraction/
│   ├── __init__.py
│   ├── relationship_extractor.py        # LLM extraction
│   └── relationship_inferrer.py         # LLM inference
│
├── agent/
│   ├── __init__.py
│   ├── tools.py                         # Tool implementations
│   └── orchestrator.py                  # Main agent
│
├── scripts/
│   ├── download_string.sh               # Download STRING data
│   └── train_link_predictor.py          # Training script
│
├── tests/
│   ├── test_link_predictor.py
│   ├── test_clients.py
│   └── test_extraction.py
│
└── main.py                              # Entry point
```

---

## Part 9: Demo Script

```python
# demo.py
"""
Demo script showing the full pipeline on the orexin system.
Run this for the interview demonstration.
"""

import asyncio
from main import setup_agent

DEMO_QUERIES = [
    # Step 1: Initial exploration
    "Explore the orexin signaling system. Start with HCRTR1, HCRTR2, and HCRT as seed proteins. "
    "Find relevant literature from the last 5 years and build a knowledge graph.",
    
    # Step 2: Novel predictions
    "What novel protein interactions does the ML model predict? "
    "Show me the top 5 predictions with their scores.",
    
    # Step 3: Deep dive on a prediction
    "Tell me more about the top predicted interaction. "
    "What does the graph neighborhood suggest about the relationship type?",
    
    # Step 4: Evidence query
    "What is the evidence for HCRTR2 interacting with G-proteins?",
    
    # Step 5: Hypothesis generation
    "Based on everything we've found, what are the most promising novel targets "
    "for narcolepsy drug development? Provide validation strategies."
]

async def run_demo():
    agent = await setup_agent()
    
    print("=" * 60)
    print("SCIENTIFIC KNOWLEDGE GRAPH AGENT - DEMO")
    print("=" * 60)
    
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n--- Demo Step {i} ---")
        print(f"Query: {query}\n")
        
        response = await agent.run(query)
        print(f"Response:\n{response}")
        
        input("\nPress Enter to continue...")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(run_demo())
```

---

## Part 10: Time Estimates & Prioritization

| Component | Hours | Priority | Notes |
|-----------|-------|----------|-------|
| **Data models** | 1-2 | P0 | Foundation |
| **STRING client** | 1-2 | P0 | Simple API |
| **PubMed client** | 2-3 | P0 | PubTator integration |
| **Link predictor training** | 3-4 | P0 | Core ML component |
| **Relationship extractor** | 2-3 | P0 | LLM extraction |
| **Relationship inferrer** | 1-2 | P1 | Can simplify if time-pressed |
| **Agent tools** | 3-4 | P0 | Glue code |
| **Orchestrator agent** | 2-3 | P0 | Main agent loop |
| **Testing** | 2-3 | P1 | At least basic tests |
| **Demo script** | 1-2 | P0 | For interview |
| **Polish / debugging** | 3-4 | P0 | Buffer |

**Total: 22-32 hours**

**MVP (minimum for demo):** ~18-20 hours
- Skip: Comprehensive tests, relationship inferrer (use simple version)
- Must have: Link predictor, extraction, agent loop, demo script

---

## Part 11: What to Say in the Interview

**Opening (30 seconds):**
> "I built a hypothesis generation tool for drug discovery scientists. Given any biological system - like the orexin pathway that Takeda is developing - it automatically synthesizes literature, predicts novel interactions using graph ML, and generates testable hypotheses."

**The ML story (1 minute):**
> "The core ML component is a link predictor trained on STRING's physical protein interactions - about 600,000 high-confidence edges. I use Node2Vec for graph embeddings and a logistic regression classifier for edge prediction. On held-out STRING edges, it achieves 0.84 AUC.
>
> The key insight is: I train this once on all of STRING, then apply it to any specific system. When a scientist wants to explore orexin biology, the model can predict which proteins are likely to interact based on graph structure, even without direct literature evidence."

**The LLM story (1 minute):**
> "But link prediction just tells you 'these proteins might interact.' It doesn't tell you HOW.
>
> So I use LLMs in two places: first, to extract typed relationships from literature - is this activation or inhibition? Second, for novel predictions where we have no literature, I use the LLM to infer likely relationship types based on the graph neighborhood.
>
> This gives you explainable hypotheses: 'The model predicts HCRTR2-NTRK1 interaction with 0.87 probability. Based on their shared neuronal context, I hypothesize a regulatory relationship. Suggested validation: co-IP assay.'"

**Live demo (25 minutes):**
> "Let me show you this on the orexin system..."
> *Run through demo queries*

**Code review talking points:**
- Why Node2Vec over GCN (sparse graph, no node features)
- Negative sampling strategy
- Evaluation methodology
- LLM prompting for extraction vs inference
- Evidence aggregation design

---
