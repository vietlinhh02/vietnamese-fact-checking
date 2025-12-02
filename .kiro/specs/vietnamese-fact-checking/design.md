# Design Document: Vietnamese Autonomous Fact-Checking System

## Overview

The Vietnamese Autonomous Fact-Checking System is a research-oriented platform designed to verify factual claims in Vietnamese text using an agent-based architecture. The system implements the ReAct (Reasoning and Acting) framework, where a Large Language Model (LLM) orchestrates a dynamic loop of reasoning, action execution, and observation processing to gather multi-step evidence, construct dynamic reasoning graphs, and generate faithful, traceable explanations.

The system is optimized for Google Colab Pro environment, leveraging free-tier APIs and open-source models (PhoBERT, XLM-RoBERTa, open-source LLMs) to minimize costs while maintaining research-grade performance. It supports cross-lingual fact-checking by utilizing both Vietnamese and English knowledge sources.

**Key Design Principles:**
- **Cost Optimization**: Use free APIs (Google Search API free tier, Gemini API, Groq API) and open-source models
- **Resource Efficiency**: Implement checkpointing, quantization, and batch processing for Colab Pro constraints
- **Transparency**: Maintain complete reasoning traces and provide source citations
- **Modularity**: Design components that can be independently evaluated and improved
- **Cross-lingual Capability**: Support Vietnamese-English evidence comparison without translation loss

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Layer                               │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ Claim Detection  │────────▶│ Claim Extraction │             │
│  │   (PhoBERT)      │         │   & Validation   │             │
│  └──────────────────┘         └──────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Core (ReAct Loop)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LLM Controller (Gemini/Groq/Llama)                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │  │
│  │  │  Reasoning │─▶│   Action   │─▶│ Observation│─┐       │  │
│  │  │   Module   │  │  Executor  │  │  Processor │ │       │  │
│  │  └────────────┘  └────────────┘  └────────────┘ │       │  │
│  │       ▲                                          │       │  │
│  │       └──────────────────────────────────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Tool Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Web Search   │  │ Web Crawler  │  │ Credibility  │         │
│  │ (Google API) │  │ (BeautifulSoup│  │  Analyzer    │         │
│  │              │  │  + Selenium)  │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Reasoning Layer                                │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ Stance Detection │         │ Graph Builder    │             │
│  │   (XLM-R)        │────────▶│ (NER + RE)       │             │
│  └──────────────────┘         └──────────────────┘             │
│                                        │                         │
│                                        ▼                         │
│                              ┌──────────────────┐               │
│                              │ GNN Reasoner     │               │
│                              │ (Graph Conv)     │               │
│                              └──────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Output Layer                                 │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ Verdict Predictor│────────▶│ RAG Explainer    │             │
│  │   (Classifier)   │         │ (LLM + Evidence) │             │
│  └──────────────────┘         └──────────────────┘             │
│                                        │                         │
│                                        ▼                         │
│                              ┌──────────────────┐               │
│                              │ Verification     │               │
│                              │ (Generate-Verify)│               │
│                              └──────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Claim Detection**: Input text → PhoBERT classifier → Extracted claims
2. **Agent Initialization**: Claim → LLM generates initial reasoning plan
3. **ReAct Loop**:
   - **Reason**: LLM analyzes current state and decides next step
   - **Act**: Execute tool (Search, Crawl, API call)
   - **Observe**: Process results, update memory, build graph
   - Repeat until sufficient evidence collected
4. **Graph Construction**: Evidence → NER/RE → Dynamic reasoning graph
5. **Stance Detection**: Claim-Evidence pairs → XLM-R → Stance labels
6. **GNN Reasoning**: Graph → GNN → Claim representation
7. **Verdict Prediction**: Claim representation → Classifier → Verdict + confidence
8. **Explanation Generation**: Evidence + Verdict → RAG → Explanation with citations
9. **Verification**: Explanation → Claim extraction → Re-verification → Quality score

## Components and Interfaces

### 1. Claim Detection Module

**Purpose**: Identify and extract verifiable factual claims from Vietnamese text.

**Model**: Fine-tuned PhoBERT-base (135M parameters)
- Base model: `vinai/phobert-base`
- Fine-tuning: Binary classification (claim vs non-claim)
- Training data: Vietnamese news articles with manual annotations

**Input Interface**:
```python
def detect_claims(text: str) -> List[Claim]:
    """
    Args:
        text: Vietnamese text input
    Returns:
        List of Claim objects with text, context, and confidence
    """
```

**Output Interface**:
```python
@dataclass
class Claim:
    text: str              # The claim sentence
    context: str           # Surrounding sentences
    confidence: float      # Classification confidence [0, 1]
    sentence_type: str     # "factual_claim", "opinion", "question", "command"
    start_idx: int         # Character position in original text
    end_idx: int
```

**Implementation Details**:
- Use sliding window approach for long documents
- Apply Vietnamese word segmentation (VnCoreNLP or pyvi)
- Batch processing for efficiency on Colab GPU

### 2. Agent Core (ReAct Framework)

**Purpose**: Orchestrate the fact-checking process through iterative reasoning and action.

**LLM Options** (in order of preference for free tier):
1. **Gemini 1.5 Flash** (Google AI Studio API - free tier: 15 RPM)
2. **Groq API** (Llama 3.1 70B - free tier: 30 RPM)
3. **Local Llama 3.1 8B** (quantized 4-bit for Colab)

**Agent State**:
```python
@dataclass
class AgentState:
    claim: Claim
    reasoning_trace: List[ReasoningStep]
    working_memory: Dict[str, Any]
    collected_evidence: List[Evidence]
    reasoning_graph: KnowledgeGraph
    iteration_count: int
    max_iterations: int = 10
```

**ReAct Loop Interface**:
```python
class ReActAgent:
    def reason(self, state: AgentState) -> Thought:
        """Generate reasoning about next step"""
        
    def act(self, thought: Thought) -> Action:
        """Decide on action based on reasoning"""
        
    def observe(self, action_result: ActionResult) -> Observation:
        """Process action results and update state"""
        
    def should_terminate(self, state: AgentState) -> bool:
        """Decide if enough evidence collected"""
```

**Tool Interface**:
```python
class Tool(ABC):
    @abstractmethod
    def execute(self, params: Dict) -> ActionResult:
        """Execute tool with parameters"""
        
class SearchTool(Tool):
    """Web search using Google Custom Search API"""
    
class CrawlTool(Tool):
    """Crawl and extract content from URLs"""
    
class CredibilityTool(Tool):
    """Analyze source credibility"""
```

**Prompt Template** (for reasoning):
```
You are a fact-checking agent. Your task is to verify the claim: "{claim}"

Current state:
- Evidence collected: {num_evidence} pieces
- Reasoning steps taken: {reasoning_trace}
- Working memory: {memory_summary}

Think step-by-step about what to do next:
1. What information do I still need?
2. What is the best source to find this information?
3. What search query or action should I take?

Output your reasoning as:
Thought: [Your step-by-step thinking]
Action: [search/crawl/api_call]
Action Input: [specific parameters]
```

### 3. Web Crawler and Content Extractor

**Purpose**: Fetch and extract clean content from Vietnamese news websites.

**Target Sources** (state-managed, credible):
- VnExpress (vnexpress.net)
- VTV News (vtv.vn)
- VOV (vov.vn)
- Tuổi Trẻ (tuoitre.vn)
- Thanh Niên (thanhnien.vn)
- Báo Chính phủ (baochinhphu.vn)

**Technology Stack**:
- **Static content**: `requests` + `BeautifulSoup4`
- **Dynamic content**: `selenium` with headless Chrome (for JavaScript-heavy sites)
- **Content extraction**: `trafilatura` library (optimized for news articles)

**Interface**:
```python
@dataclass
class WebContent:
    url: str
    title: str
    author: Optional[str]
    publish_date: Optional[datetime]
    main_text: str
    metadata: Dict[str, Any]
    extraction_success: bool
    
def crawl_url(url: str, render_js: bool = False) -> WebContent:
    """
    Args:
        url: Target URL
        render_js: Whether to use Selenium for JavaScript rendering
    Returns:
        Extracted web content
    """
```

**Content Cleaning Pipeline**:
1. Fetch HTML (with or without JS rendering)
2. Parse DOM tree
3. Remove boilerplate (ads, navigation, footer) using trafilatura
4. Extract metadata from meta tags and structured data
5. Normalize text (remove extra whitespace, fix encoding)
6. Validate extraction quality (minimum text length, presence of title)

### 4. Cross-lingual Search Module

**Purpose**: Generate and execute search queries in both Vietnamese and English.

**Translation Strategy**:
- **Option 1**: Google Translate API (free tier: 500K characters/month)
- **Option 2**: MarianMT models (`Helsinki-NLP/opus-mt-vi-en`) - fully local
- **Hybrid**: Use MarianMT by default, fallback to Google Translate for complex queries

**Interface**:
```python
@dataclass
class SearchQuery:
    text: str
    language: str  # "vi" or "en"
    source_claim: str
    
@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    language: str
    rank: int
    
def generate_queries(claim: Claim) -> List[SearchQuery]:
    """Generate Vietnamese and English search queries"""
    
def execute_search(query: SearchQuery) -> List[SearchResult]:
    """Execute search using Google Custom Search API"""
```

**Query Generation Strategies**:
1. **Direct translation**: Translate claim to English
2. **Entity-focused**: Extract key entities and search for them
3. **Question reformulation**: Convert claim to question form
4. **Decomposition**: Break complex claims into sub-queries

**Rate Limiting**:
- Implement exponential backoff for API calls
- Cache search results locally (SQLite database)
- Batch queries when possible

### 5. Source Credibility Analyzer

**Purpose**: Compute credibility scores for evidence sources.

**Credibility Features**:
```python
@dataclass
class CredibilityFeatures:
    # Domain features
    domain: str
    tld: str  # .vn, .com, .org
    domain_age_days: Optional[int]
    uses_https: bool
    
    # Content features
    has_author: bool
    has_publish_date: bool
    article_length: int
    writing_quality_score: float  # Grammar, spelling
    
    # External signals
    is_state_managed: bool  # Vietnamese government-managed
    mbfc_rating: Optional[str]  # From Media Bias Fact Check API
    
@dataclass
class CredibilityScore:
    overall_score: float  # [0, 1]
    feature_scores: Dict[str, float]
    confidence: float
    explanation: str
```

**Scoring Model**:
- **Rule-based baseline**: Weighted sum of features
  - State-managed Vietnamese sources: +0.4
  - HTTPS: +0.1
  - Has author + date: +0.1
  - Domain age > 1 year: +0.1
  - Good writing quality: +0.2
  - MBFC "High" rating: +0.3

- **ML model** (optional): Train logistic regression on labeled data
  - Features: All credibility features
  - Labels: Manual annotations of source reliability

**Interface**:
```python
def analyze_credibility(source: WebContent) -> CredibilityScore:
    """Compute credibility score for a source"""
```

### 6. Stance Detection Module

**Purpose**: Determine relationship between claim and evidence (Support/Refute/Neutral).

**Model Architecture**:
- **Base model**: XLM-RoBERTa-base (270M parameters)
  - Pretrained: `xlm-roberta-base`
  - Supports 100 languages including Vietnamese and English
  
**Fine-tuning Strategy**:
1. **Data**: Create Vietnamese-English stance dataset
   - Translate FEVER/SNLI to Vietnamese
   - Collect Vietnamese claim-evidence pairs
   - Augment with back-translation
   
2. **Training**:
   - Input: `[CLS] claim [SEP] evidence [SEP]`
   - Output: 3-class classification (Support/Refute/Neutral)
   - Loss: Cross-entropy
   - Optimizer: AdamW with linear warmup
   - Mixed precision (FP16) for memory efficiency

**Interface**:
```python
@dataclass
class StanceResult:
    stance: str  # "support", "refute", "neutral"
    confidence_scores: Dict[str, float]  # Probabilities for each class
    claim_lang: str
    evidence_lang: str
    
def detect_stance(claim: str, evidence: str) -> StanceResult:
    """
    Args:
        claim: Claim text (Vietnamese)
        evidence: Evidence text (Vietnamese or English)
    Returns:
        Stance classification result
    """
```

**Optimization for Colab**:
- Use 8-bit quantization (`bitsandbytes` library)
- Batch size: 8-16 depending on GPU memory
- Gradient accumulation if needed

### 7. Dynamic Reasoning Graph Builder

**Purpose**: Construct knowledge graph from collected evidence.

**Graph Schema**:
```python
@dataclass
class GraphNode:
    id: str
    type: str  # "entity", "claim", "evidence"
    text: str
    attributes: Dict[str, Any]
    embedding: Optional[np.ndarray]
    
@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str  # "supports", "refutes", "mentions", "related_to"
    weight: float
    evidence_source: str
    
class KnowledgeGraph:
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    
    def add_node(self, node: GraphNode) -> None:
        """Add or merge node"""
        
    def add_edge(self, edge: GraphEdge) -> None:
        """Add edge between nodes"""
        
    def merge_duplicate_entities(self) -> None:
        """Merge nodes representing same entity"""
        
    def to_dgl_graph(self) -> dgl.DGLGraph:
        """Convert to DGL format for GNN"""
```

**NER and RE Pipeline**:
1. **Named Entity Recognition**:
   - Model: Fine-tuned PhoBERT for Vietnamese NER
   - Entities: PERSON, ORGANIZATION, LOCATION, DATE, NUMBER
   - For English: Use spaCy `en_core_web_sm`

2. **Relation Extraction**:
   - Model: Fine-tuned XLM-R for relation classification
   - Relations: Common semantic relations (works_for, located_in, etc.)
   - Fallback: Dependency parsing for simple relations

3. **Entity Linking**:
   - Use string similarity (Levenshtein distance) for merging
   - Consider context embeddings for disambiguation

**Interface**:
```python
def extract_entities(text: str, lang: str) -> List[Entity]:
    """Extract named entities from text"""
    
def extract_relations(text: str, entities: List[Entity]) -> List[Relation]:
    """Extract relations between entities"""
    
def build_graph(evidence_list: List[Evidence]) -> KnowledgeGraph:
    """Build reasoning graph from evidence"""
```

## Data Models

### Core Data Structures

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np

@dataclass
class Claim:
    """Represents a factual claim to be verified"""
    id: str
    text: str
    context: str
    confidence: float
    sentence_type: str
    start_idx: int
    end_idx: int
    language: str = "vi"
    
@dataclass
class Evidence:
    """Represents a piece of evidence"""
    id: str
    text: str
    source_url: str
    source_title: str
    source_author: Optional[str]
    publish_date: Optional[datetime]
    credibility_score: float
    language: str
    stance: Optional[str]  # Filled after stance detection
    stance_confidence: Optional[float]
    
@dataclass
class ReasoningStep:
    """One step in the ReAct loop"""
    iteration: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: datetime
    
@dataclass
class Verdict:
    """Final verdict on claim"""
    claim_id: str
    label: str  # "supported", "refuted", "not_enough_info"
    confidence_scores: Dict[str, float]
    supporting_evidence: List[str]  # Evidence IDs
    refuting_evidence: List[str]
    explanation: str
    reasoning_trace: List[ReasoningStep]
    quality_score: float
    
@dataclass
class FactCheckResult:
    """Complete fact-checking result"""
    claim: Claim
    verdict: Verdict
    evidence: List[Evidence]
    reasoning_graph: KnowledgeGraph
    metadata: Dict[str, Any]
```

### Database Schema (SQLite for caching)

```sql
-- Cache for search results
CREATE TABLE search_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    language TEXT NOT NULL,
    results JSON NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Cache for crawled content
CREATE TABLE content_cache (
    url_hash TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    content JSON NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Credibility scores cache
CREATE TABLE credibility_cache (
    domain TEXT PRIMARY KEY,
    score REAL NOT NULL,
    features JSON NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Experiment tracking
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    config JSON NOT NULL,
    results JSON NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Claim Detection Completeness
*For any* Vietnamese text containing N factual claims, the claim detection module should identify all N claims with confidence scores, and no non-claim sentences should be misclassified as claims with high confidence (>0.7).
**Validates: Requirements 1.1, 1.2, 1.4**

### Property 2: Claim Extraction Context Preservation
*For any* detected claim, the extracted output should contain both the claim text and its surrounding context, where context includes at least one sentence before and after the claim.
**Validates: Requirements 1.3**

### Property 3: ReAct Loop Structure Invariant
*For any* execution of the ReAct loop, each iteration should follow the sequence: Reasoning → Action → Observation, and the reasoning trace should contain all three components for every iteration.
**Validates: Requirements 2.1, 2.2, 2.3, 2.6**

### Property 4: Agent Memory Monotonicity
*For any* ReAct loop execution, the working memory should only grow or stay the same (never lose information), and each observation should result in at least one update to the memory state.
**Validates: Requirements 2.4**

### Property 5: Evidence Collection Termination
*For any* claim verification task, the ReAct loop should terminate within a maximum number of iterations (default: 10) or when the agent determines sufficient evidence has been collected.
**Validates: Requirements 2.5**

### Property 6: Source Whitelist Compliance
*For any* evidence collected through crawling, the source URL should belong to the approved list of Vietnamese state-managed news sources (VnExpress, VTV, VOV, Tuổi Trẻ, Thanh Niên, Báo Chính phủ).
**Validates: Requirements 3.1**

### Property 7: Content Extraction Purity
*For any* crawled webpage, the extracted main text should not contain boilerplate elements (ads, navigation menus, footers), verified by checking for common boilerplate patterns and HTML tags.
**Validates: Requirements 3.3**

### Property 8: Metadata Completeness
*For any* successfully extracted article, the output should contain at minimum: URL, title, main text, and extraction timestamp. Author and publish date are optional but should be extracted if available.
**Validates: Requirements 3.4**

### Property 9: Bilingual Query Generation
*For any* Vietnamese claim, the search query generation should produce at least one Vietnamese query and one English query, where the English query preserves the semantic meaning of the original claim.
**Validates: Requirements 4.1, 4.2**

### Property 10: Multilingual Evidence Collection
*For any* search execution, the collected evidence should include sources from both Vietnamese and English languages (if available), with at least one source per language when search results exist.
**Validates: Requirements 4.3**

### Property 11: API Rate Limit Compliance
*For any* sequence of API calls, the system should never exceed the rate limits of free-tier APIs (e.g., 15 RPM for Gemini, 100 queries/day for Google Search), verified by tracking call timestamps and implementing delays.
**Validates: Requirements 4.4**

### Property 12: Credibility Score Existence
*For any* piece of evidence collected, there should exist an associated credibility score in the range [0, 1], computed from source features and external signals.
**Validates: Requirements 5.1, 5.5**

### Property 13: State-Managed Source Priority
*For any* two evidence sources where one is Vietnamese state-managed and the other is not, the state-managed source should receive a higher credibility score (difference ≥ 0.2).
**Validates: Requirements 5.2**

### Property 14: Credibility Feature Coverage
*For any* credibility score computation, the system should evaluate at minimum: domain TLD, HTTPS usage, and presence of author/date metadata.
**Validates: Requirements 5.3**

### Property 15: Stance Classification Completeness
*For any* claim-evidence pair, the stance detection should output exactly one of three classes (Support, Refute, Neutral) along with confidence scores for all three classes that sum to 1.0.
**Validates: Requirements 6.3, 6.4**

### Property 16: Graph Node Extraction
*For any* piece of evidence text, the graph builder should extract at least one entity or proposition as a graph node, unless the text is too short (<10 words).
**Validates: Requirements 7.1**

### Property 17: Graph Edge Existence
*For any* reasoning graph with N entities (N ≥ 2), there should exist at least one edge connecting entities, representing relationships extracted from evidence.
**Validates: Requirements 7.2**

### Property 18: Graph Monotonic Growth
*For any* sequence of evidence additions, the number of nodes and edges in the reasoning graph should be non-decreasing (monotonically increasing or staying the same).
**Validates: Requirements 7.3**

### Property 19: Entity Uniqueness
*For any* reasoning graph, there should be no duplicate entity nodes representing the same real-world entity (verified by checking for identical or highly similar entity names after normalization).
**Validates: Requirements 7.4**

### Property 20: Contradiction Preservation
*For any* reasoning graph containing contradictory information (evidence A supports claim, evidence B refutes claim), both pieces of evidence should be preserved as separate nodes with their source attribution and credibility scores.
**Validates: Requirements 7.5**

### Property 21: GNN Output Format
*For any* reasoning graph processed by the GNN, the output should include a vector representation for the claim node with dimensionality matching the model's hidden size.
**Validates: Requirements 8.3**

### Property 22: Verdict Classification
*For any* claim verification, the final verdict should be exactly one of three classes: "Supported", "Refuted", or "Not Enough Info", with confidence scores for each class.
**Validates: Requirements 8.4, 8.5**

### Property 23: RAG Grounding
*For any* generated explanation, every factual statement should be traceable to at least one piece of collected evidence, with no information introduced that wasn't in the evidence set.
**Validates: Requirements 9.1**

### Property 24: Citation Completeness
*For any* factual claim made in the explanation, there should exist at least one citation with a source URL pointing to the evidence that supports that claim.
**Validates: Requirements 9.2**

### Property 25: Reasoning Trace Inclusion
*For any* final explanation, the output should include the complete reasoning trace showing all ReAct loop iterations, search queries executed, and evidence collected.
**Validates: Requirements 9.4**

### Property 26: Contradiction Presentation
*For any* verdict where contradictory evidence exists (both supporting and refuting evidence with credibility > 0.5), the explanation should explicitly mention both sides with their respective credibility scores.
**Validates: Requirements 9.5**

### Property 27: Self-Verification Execution
*For any* generated explanation, the system should extract factual claims from the explanation and perform verification searches for each claim.
**Validates: Requirements 10.1, 10.2**

### Property 28: Hallucination Detection Response
*For any* verification that fails (cannot find supporting evidence for a claim in the explanation), the system should either flag the explanation as uncertain or revise the problematic claim.
**Validates: Requirements 10.3**

### Property 29: Quality Score Output
*For any* completed fact-check, the final output should include a quality score in the range [0, 1] indicating the reliability of the explanation based on verification results.
**Validates: Requirements 10.4**

### Property 30: Hallucination Removal
*For any* detected hallucination in the explanation (claim that cannot be verified against collected evidence), the hallucinated content should be removed or marked as unverified in the final output.
**Validates: Requirements 10.5**

### Property 31: Dataset Evidence Association
*For any* claim in the constructed dataset, there should exist at least one associated evidence item collected from credible sources.
**Validates: Requirements 11.2**

### Property 32: Dataset Schema Compliance
*For any* exported dataset, each record should contain all required fields: claim text, evidence list, label (Supported/Refuted/NEI), and metadata (source URLs, timestamps).
**Validates: Requirements 11.5**

### Property 33: Checkpoint Periodicity
*For any* training or inference run on Colab, checkpoints should be saved at intervals of 30 minutes or less to handle potential session timeouts.
**Validates: Requirements 12.1**

### Property 34: Checkpoint Recovery
*For any* interrupted session, the system should be able to resume from the most recent checkpoint without losing progress or requiring re-computation of completed steps.
**Validates: Requirements 12.4**

### Property 35: Evaluation Metrics Completeness
*For any* evaluation run, the system should compute and output all standard metrics: Accuracy, Precision, Recall, and F1-Score for each class and overall.
**Validates: Requirements 13.1**

### Property 36: Ablation Study Support
*For any* system configuration, individual components (e.g., GNN, stance detection, credibility scoring) should be able to be disabled independently to measure their contribution to performance.
**Validates: Requirements 13.2**

### Property 37: Statistical Significance Testing
*For any* comparison between two methods, the performance report should include statistical significance tests (e.g., paired t-test, McNemar's test) with p-values.
**Validates: Requirements 13.3**

### Property 38: Experiment Reproducibility
*For any* experiment run, the system should log the complete configuration (hyperparameters, random seeds, model versions) to enable exact reproduction of results.
**Validates: Requirements 13.5**

### Property 39: Demo Real-time Progress
*For any* claim submitted through the demo interface, the system should display progress updates showing the current ReAct loop iteration, action being executed, and evidence collected so far.
**Validates: Requirements 14.2**

### Property 40: Demo Output Completeness
*For any* completed verification in the demo, the output should display: verdict label, confidence score, list of evidence sources with URLs, and the full explanation text.
**Validates: Requirements 14.3**

## Error Handling

### Error Categories and Strategies

#### 1. API Errors
- **Rate Limit Exceeded**: Implement exponential backoff, use cached results, switch to alternative API
- **API Timeout**: Retry with increased timeout, fallback to alternative source
- **Invalid API Response**: Log error, skip source, continue with other sources
- **Quota Exhausted**: Switch to local models (MarianMT for translation, local LLM for generation)

#### 2. Crawling Errors
- **HTTP Errors (404, 403, 500)**: Log error, try alternative URLs from search results
- **JavaScript Rendering Failure**: Fallback to static HTML parsing, skip if content unavailable
- **Content Extraction Failure**: Use alternative extraction method (readability, newspaper3k), mark as low quality
- **Encoding Issues**: Try multiple encodings (UTF-8, Windows-1252), use chardet for detection

#### 3. Model Errors
- **OOM (Out of Memory)**: Reduce batch size, use gradient checkpointing, apply quantization
- **Model Loading Failure**: Download model again, use smaller model variant
- **Inference Timeout**: Reduce max sequence length, use faster model
- **NaN/Inf in Outputs**: Clip gradients, reduce learning rate, check input data

#### 4. Data Errors
- **Empty Claim**: Return error message, skip processing
- **No Evidence Found**: Return "Not Enough Info" verdict with explanation
- **All Sources Unreliable**: Flag low confidence, use best available evidence
- **Contradictory Evidence**: Present both sides, compute confidence based on credibility

#### 5. System Errors
- **Colab Session Timeout**: Auto-save checkpoint, resume from checkpoint on reconnect
- **Disk Space Full**: Clean cache, compress checkpoints, use Google Drive for storage
- **Network Disconnection**: Queue operations, retry when connection restored

### Error Recovery Mechanisms

```python
class ErrorHandler:
    def __init__(self):
        self.retry_config = {
            'max_retries': 3,
            'backoff_factor': 2,
            'timeout': 30
        }
        
    def handle_api_error(self, error: Exception, context: Dict) -> Any:
        """Handle API-related errors with retry logic"""
        if isinstance(error, RateLimitError):
            time.sleep(self.retry_config['backoff_factor'] ** context['attempt'])
            return self.retry_with_backoff(context)
        elif isinstance(error, QuotaExceededError):
            return self.fallback_to_local_model(context)
        else:
            logger.error(f"API error: {error}")
            return None
            
    def handle_crawl_error(self, error: Exception, url: str) -> Optional[WebContent]:
        """Handle crawling errors with fallback strategies"""
        if isinstance(error, HTTPError):
            logger.warning(f"HTTP error for {url}: {error}")
            return None
        elif isinstance(error, JavaScriptError):
            return self.fallback_static_parse(url)
        else:
            logger.error(f"Crawl error for {url}: {error}")
            return None
```

## Testing Strategy

### Unit Testing

Unit tests will verify individual components in isolation:

1. **Claim Detection Tests**:
   - Test with sentences containing clear factual claims
   - Test with opinions, questions, commands (should be rejected)
   - Test with edge cases (very short/long sentences, special characters)

2. **Crawling Tests**:
   - Test with static HTML pages
   - Test with JavaScript-heavy pages
   - Test boilerplate removal accuracy
   - Test metadata extraction

3. **Stance Detection Tests**:
   - Test with clear support/refute examples
   - Test with neutral evidence
   - Test cross-lingual pairs (Vietnamese claim + English evidence)

4. **Graph Building Tests**:
   - Test entity extraction accuracy
   - Test relation extraction
   - Test entity merging logic
   - Test contradiction handling

5. **RAG Generation Tests**:
   - Test citation insertion
   - Test grounding to evidence
   - Test handling of missing evidence

### Property-Based Testing

Property-based tests will verify universal properties across many randomly generated inputs using the **Hypothesis** library for Python.

**Configuration**: Each property test should run a minimum of 100 iterations with varied inputs.

**Test Framework**: Hypothesis (Python property-based testing library)

#### PBT 1: Claim Detection Completeness
**Feature: vietnamese-fact-checking, Property 1: Claim Detection Completeness**
```python
@given(st.text(min_size=50, max_size=500))
@settings(max_examples=100)
def test_claim_detection_completeness(text):
    """Test that all claims are detected and non-claims are rejected"""
    results = detect_claims(text)
    # Property: High-confidence results should be valid claims
    for claim in results:
        if claim.confidence > 0.7:
            assert claim.sentence_type == "factual_claim"
```

#### PBT 2: Claim Context Preservation
**Feature: vietnamese-fact-checking, Property 2: Claim Extraction Context Preservation**
```python
@given(st.text(min_size=100))
def test_context_preservation(text):
    """Test that extracted claims include surrounding context"""
    claims = detect_claims(text)
    for claim in claims:
        assert len(claim.context) > len(claim.text)
        assert claim.text in claim.context
```

#### PBT 3: ReAct Loop Structure
**Feature: vietnamese-fact-checking, Property 3: ReAct Loop Structure Invariant**
```python
@given(st.text(min_size=20, max_size=200))
def test_react_loop_structure(claim_text):
    """Test that ReAct loop maintains proper structure"""
    agent = ReActAgent()
    result = agent.verify_claim(Claim(text=claim_text))
    
    for step in result.reasoning_trace:
        assert step.thought is not None
        assert step.action is not None
        assert step.observation is not None
```

#### PBT 4: Agent Memory Monotonicity
**Feature: vietnamese-fact-checking, Property 4: Agent Memory Monotonicity**
```python
@given(st.lists(st.text(min_size=10), min_size=1, max_size=10))
def test_memory_monotonicity(observations):
    """Test that agent memory only grows"""
    agent = ReActAgent()
    memory_sizes = []
    
    for obs in observations:
        agent.observe(obs)
        memory_sizes.append(len(agent.working_memory))
    
    # Memory should be non-decreasing
    assert all(memory_sizes[i] <= memory_sizes[i+1] 
               for i in range(len(memory_sizes)-1))
```

#### PBT 5: Evidence Collection Termination
**Feature: vietnamese-fact-checking, Property 5: Evidence Collection Termination**
```python
@given(st.text(min_size=20))
def test_termination(claim_text):
    """Test that ReAct loop terminates within max iterations"""
    agent = ReActAgent(max_iterations=10)
    result = agent.verify_claim(Claim(text=claim_text))
    
    assert len(result.reasoning_trace) <= 10
```

#### PBT 6: Source Whitelist Compliance
**Feature: vietnamese-fact-checking, Property 6: Source Whitelist Compliance**
```python
@given(st.lists(st.text(min_size=50), min_size=1, max_size=5))
def test_source_whitelist(claims):
    """Test that all crawled sources are from approved list"""
    approved_domains = ['vnexpress.net', 'vtv.vn', 'vov.vn', 
                       'tuoitre.vn', 'thanhnien.vn', 'baochinhphu.vn']
    
    for claim_text in claims:
        evidence = collect_evidence(claim_text)
        for ev in evidence:
            domain = urlparse(ev.source_url).netloc
            assert any(approved in domain for approved in approved_domains)
```

#### PBT 7: Content Extraction Purity
**Feature: vietnamese-fact-checking, Property 7: Content Extraction Purity**
```python
@given(st.sampled_from(['https://vnexpress.net', 'https://vtv.vn']))
def test_content_purity(base_url):
    """Test that extracted content has no boilerplate"""
    content = crawl_url(base_url + '/sample-article')
    
    # Should not contain common boilerplate patterns
    boilerplate_patterns = ['Quảng cáo', 'Đăng nhập', 'Đăng ký', 
                           'Liên hệ', 'Bản quyền']
    for pattern in boilerplate_patterns:
        assert pattern not in content.main_text
```

#### PBT 8: Bilingual Query Generation
**Feature: vietnamese-fact-checking, Property 9: Bilingual Query Generation**
```python
@given(st.text(min_size=20, max_size=200))
def test_bilingual_queries(claim_text):
    """Test that both Vietnamese and English queries are generated"""
    queries = generate_queries(Claim(text=claim_text, language='vi'))
    
    languages = [q.language for q in queries]
    assert 'vi' in languages
    assert 'en' in languages
```

#### PBT 9: API Rate Limit Compliance
**Feature: vietnamese-fact-checking, Property 11: API Rate Limit Compliance**
```python
@given(st.lists(st.text(min_size=10), min_size=20, max_size=50))
def test_rate_limit_compliance(queries):
    """Test that API calls respect rate limits"""
    start_time = time.time()
    
    for query in queries:
        execute_search(query)
    
    elapsed = time.time() - start_time
    calls_per_minute = len(queries) / (elapsed / 60)
    
    # Should not exceed 15 calls per minute for Gemini
    assert calls_per_minute <= 15
```

#### PBT 10: Credibility Score Range
**Feature: vietnamese-fact-checking, Property 12: Credibility Score Existence**
```python
@given(st.builds(WebContent))
def test_credibility_score_range(content):
    """Test that credibility scores are in valid range"""
    score = analyze_credibility(content)
    
    assert 0 <= score.overall_score <= 1
    assert score.confidence >= 0
```

#### PBT 11: State-Managed Source Priority
**Feature: vietnamese-fact-checking, Property 13: State-Managed Source Priority**
```python
@given(st.sampled_from(['vnexpress.net', 'random-blog.com']))
def test_state_managed_priority(domain):
    """Test that state-managed sources get higher scores"""
    state_content = WebContent(url=f'https://vnexpress.net/article')
    other_content = WebContent(url=f'https://random-blog.com/post')
    
    state_score = analyze_credibility(state_content).overall_score
    other_score = analyze_credibility(other_content).overall_score
    
    assert state_score - other_score >= 0.2
```

#### PBT 12: Stance Classification Completeness
**Feature: vietnamese-fact-checking, Property 15: Stance Classification Completeness**
```python
@given(st.text(min_size=20), st.text(min_size=20))
def test_stance_completeness(claim, evidence):
    """Test that stance detection outputs valid classification"""
    result = detect_stance(claim, evidence)
    
    assert result.stance in ['support', 'refute', 'neutral']
    assert abs(sum(result.confidence_scores.values()) - 1.0) < 0.01
```

#### PBT 13: Graph Monotonic Growth
**Feature: vietnamese-fact-checking, Property 18: Graph Monotonic Growth**
```python
@given(st.lists(st.builds(Evidence), min_size=2, max_size=10))
def test_graph_monotonic_growth(evidence_list):
    """Test that graph grows monotonically"""
    graph = KnowledgeGraph()
    node_counts = []
    
    for evidence in evidence_list:
        graph.add_evidence(evidence)
        node_counts.append(len(graph.nodes))
    
    # Node count should be non-decreasing
    assert all(node_counts[i] <= node_counts[i+1] 
               for i in range(len(node_counts)-1))
```

#### PBT 14: RAG Grounding
**Feature: vietnamese-fact-checking, Property 23: RAG Grounding**
```python
@given(st.builds(Claim), st.lists(st.builds(Evidence), min_size=1))
def test_rag_grounding(claim, evidence_list):
    """Test that explanation is grounded in evidence"""
    explanation = generate_explanation(claim, evidence_list)
    
    # Extract facts from explanation
    explanation_facts = extract_facts(explanation)
    evidence_texts = [e.text for e in evidence_list]
    
    # Each fact should be traceable to evidence
    for fact in explanation_facts:
        assert any(is_supported_by(fact, ev_text) 
                  for ev_text in evidence_texts)
```

#### PBT 15: Citation Completeness
**Feature: vietnamese-fact-checking, Property 24: Citation Completeness**
```python
@given(st.builds(Verdict))
def test_citation_completeness(verdict):
    """Test that all claims in explanation have citations"""
    claims_in_explanation = extract_claims(verdict.explanation)
    citations = extract_citations(verdict.explanation)
    
    # Number of citations should be >= number of factual claims
    assert len(citations) >= len(claims_in_explanation)
```

### Integration Testing

Integration tests will verify end-to-end workflows:

1. **Full Pipeline Test**: Input claim → Output verdict with explanation
2. **Cross-lingual Test**: Vietnamese claim → English evidence → Correct verdict
3. **Multi-source Test**: Claim requiring evidence from multiple sources
4. **Contradiction Test**: Claim with conflicting evidence → Balanced explanation
5. **Error Recovery Test**: Simulate API failures → System continues with fallbacks

### Performance Testing

1. **Latency**: Measure end-to-end time for single claim verification
2. **Throughput**: Measure claims processed per hour on Colab Pro
3. **Memory Usage**: Monitor GPU/RAM usage during inference
4. **API Quota**: Track API calls per claim to ensure staying within limits

### Evaluation Metrics

1. **Claim Detection**: Precision, Recall, F1 on labeled Vietnamese claims
2. **Stance Detection**: Accuracy on Vietnamese-English pairs
3. **Verdict Prediction**: Accuracy, Macro-F1 on test set
4. **Explanation Quality**: ROUGE scores against human-written explanations, citation accuracy
5. **End-to-End**: Accuracy on full fact-checking task compared to human judgments

