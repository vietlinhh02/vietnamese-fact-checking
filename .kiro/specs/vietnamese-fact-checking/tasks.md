# Implementation Plan

- [x] 1. Setup project structure and environment for Colab Pro





  - Create directory structure: `src/`, `data/`, `models/`, `experiments/`, `notebooks/`, `tests/`
  - Setup requirements.txt with core dependencies (transformers, torch, dgl, hypothesis, beautifulsoup4, selenium, trafilatura, gradio)
  - Configure Google Drive mounting for persistent storage (`/content/drive/MyDrive/vietnamese-fact-checking/`)
  - Implement checkpoint manager with 30-minute auto-save to prevent session timeout data loss
  - Setup logging system with file and console handlers
  - Create configuration management (YAML/JSON) for hyperparameters and API keys
  - _Requirements: 12.1, 12.4_

- [x] 2. Implement data models and core interfaces





  - [x] 2.1 Create data classes for Claim, Evidence, Verdict, ReasoningStep


    - Define dataclasses with type hints and validation
    - Implement serialization/deserialization methods (to_dict, from_dict)
    - Add utility methods for data manipulation
    - _Requirements: All (foundational)_
  

  - [x] 2.2 Write property test for data model serialization

    - **Property 2: Claim Extraction Context Preservation**
    - **Validates: Requirements 1.3**
  


  - [x] 2.3 Implement SQLite caching layer





    - Create database schema for search_cache, content_cache, credibility_cache
    - Implement cache manager with TTL (time-to-live) support
    - Add cache hit/miss logging for performance monitoring


    - _Requirements: 4.4, 12.3_
  

  - [x] 2.4 Write unit tests for cache operations


    - Test cache insertion, retrieval, expiration
    - Test concurrent access handling
    - _Requirements: 4.4_

- [x] 3. Build web crawling and content extraction module




  - [x] 3.1 Implement static HTML crawler with BeautifulSoup


    - Create crawler for approved Vietnamese news sources (VnExpress, VTV, VOV, Tuổi Trẻ, Thanh Niên)
    - Implement robots.txt compliance checking
    - Add request rate limiting and user-agent rotation
    - _Requirements: 3.1, 3.5_
  
  - [x] 3.2 Implement dynamic content crawler with Selenium


    - Setup headless Chrome driver for JavaScript rendering
    - Implement wait strategies for dynamic content loading
    - Add fallback to static parsing if Selenium fails
    - _Requirements: 3.2_

  
  - [x] 3.3 Implement content extraction with trafilatura





    - Extract main article text, removing boilerplate (ads, menus, footers)
    - Extract metadata (title, author, publish date, URL)
    - Implement quality validation (minimum text length, presence of title)


    - _Requirements: 3.3, 3.4_
  


  - [ ] 3.4 Write property test for source whitelist compliance
    - **Property 6: Source Whitelist Compliance**
    - **Validates: Requirements 3.1**


  
  - [ ] 3.5 Write property test for content extraction purity
    - **Property 7: Content Extraction Purity**
    - **Validates: Requirements 3.3**
  
  - [ ] 3.6 Write unit tests for crawling error handling
    - Test HTTP errors (404, 403, 500)
    - Test encoding issues
    - Test timeout handling
    - _Requirements: 3.5_

- [x] 4. Checkpoint - Verify crawling module works





  - Ensure all tests pass, ask the user if questions arise.
-

- [x] 5. Implement source credibility analyzer




  - [x] 5.1 Create credibility feature extractor


    - Extract domain features (TLD, domain age, HTTPS)
    - Extract content features (author, date, article length, writing quality)
    - Implement state-managed source detection for Vietnamese outlets
    - _Requirements: 5.2, 5.3_
  
  - [x] 5.2 Implement rule-based credibility scoring

    - Create weighted scoring formula (state-managed +0.4, HTTPS +0.1, etc.)
    - Compute overall credibility score [0, 1]
    - Generate explanation for score
    - _Requirements: 5.1, 5.5_
  
  - [x] 5.3 Integrate Media Bias Fact Check API (optional, free tier)


    - Implement API client with rate limiting
    - Merge MBFC ratings into credibility score
    - Handle API failures gracefully
    - _Requirements: 5.4_
  
  - [x] 5.4 Write property test for credibility score range


    - **Property 12: Credibility Score Existence**
    - **Validates: Requirements 5.1, 5.5**
  
  - [x] 5.5 Write property test for state-managed source priority

    - **Property 13: State-Managed Source Priority**
    - **Validates: Requirements 5.2**
- [x] 6. Build claim detection module with PhoBERT



- [ ] 6. Build claim detection module with PhoBERT

  - [x] 6.1 Prepare training data for claim detection


    - Collect Vietnamese news articles
    - Manually annotate sentences as claim/non-claim (or use weak supervision)
    - Split into train/val/test sets (70/15/15)
    - _Requirements: 1.1, 1.2_
  
  - [x] 6.2 Fine-tune PhoBERT for claim classification


    - Load pretrained `vinai/phobert-base` model
    - Add classification head for binary classification
    - Implement training loop with AdamW optimizer and linear warmup
    - Use mixed precision (FP16) for memory efficiency
    - Save best checkpoint based on validation F1
    - _Requirements: 1.1, 1.2, 1.5_
  
  - [x] 6.3 Implement claim extraction pipeline


    - Apply Vietnamese word segmentation (pyvi or VnCoreNLP)
    - Use sliding window for long documents
    - Extract claim with surrounding context
    - Batch process for GPU efficiency
    - _Requirements: 1.3, 1.4_
  
  - [x] 6.4 Write property test for claim detection completeness


    - **Property 1: Claim Detection Completeness**
    - **Validates: Requirements 1.1, 1.2, 1.4**
  
  - [x] 6.5 Write property test for context preservation


    - **Property 2: Claim Extraction Context Preservation**
    - **Validates: Requirements 1.3**

- [x] 7. Implement cross-lingual search module


  - [x] 7.1 Implement translation service


    - Setup MarianMT model (`Helsinki-NLP/opus-mt-vi-en`) for local translation
    - Implement Google Translate API client as fallback (free tier: 500K chars/month)
    - Add translation caching to reduce API calls
    - _Requirements: 4.2, 4.5_
  
  - [x] 7.2 Implement search query generator


    - Generate Vietnamese queries from claim (direct, entity-focused, question form)
    - Generate English queries via translation
    - Implement query decomposition for complex claims
    - _Requirements: 4.1, 4.2_
  


  - [x] 7.3 Implement Exa Search API client
    - Setup Exa API client with API key management
    - Implement search with 'auto' or 'neural' mode for intelligent results
    - Support content extraction (text, highlights, summary) directly from search
    - Cache results in SQLite
    - _Requirements: 4.3, 4.4_
  
  - [x] 7.4 Write property test for bilingual query generation


    - **Property 9: Bilingual Query Generation**
    - **Validates: Requirements 4.1, 4.2**
  
  - [x] 7.5 Write property test for multilingual evidence collection
    - **Property 10: Multilingual Evidence Collection**
    - **Validates: Requirements 4.3**
  
  - [x] 7.6 Write property test for API rate limit compliance
    - **Property 11: API Rate Limit Compliance**
    - **Validates: Requirements 4.4**

- [ ] 8. Checkpoint - Verify search and translation work



  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Build stance detection module with XLM-RoBERTa






  - [x] 9.1 Prepare cross-lingual stance detection dataset


    - Translate FEVER/SNLI dataset to Vietnamese using MarianMT
    - Collect Vietnamese claim-evidence pairs from news
    - Augment with back-translation
    - Create train/val/test splits
    - _Requirements: 6.1, 6.2, 6.5_
  
  - [x] 9.2 Fine-tune XLM-RoBERTa for stance detection



    - Load pretrained `xlm-roberta-base` model
    - Add 3-class classification head (Support/Refute/Neutral)
    - Implement training with cross-entropy loss
    - Use 8-bit quantization for memory efficiency
    - Apply gradient accumulation if needed
    - _Requirements: 6.1, 6.2_
  
  - [x] 9.3 Implement stance detection inference


    - Create inference pipeline for claim-evidence pairs
    - Support both Vietnamese-Vietnamese and Vietnamese-English pairs
    - Output stance label and confidence scores
    - Batch process for efficiency
    - _Requirements: 6.3, 6.4_
  
  - [x] 9.4 Write property test for stance classification completeness


    - **Property 15: Stance Classification Completeness**
    - **Validates: Requirements 6.3, 6.4**

- [-] 10. Implement NER and relation extraction for graph building



  - [x] 10.1 Setup NER models


    - Fine-tune PhoBERT for Vietnamese NER (PERSON, ORG, LOC, DATE, NUMBER)
    - Use spaCy `en_core_web_sm` for English NER
    - Implement entity extraction pipeline
    - _Requirements: 7.1_
  
  - [x] 10.2 Implement relation extraction


    - Fine-tune XLM-R for relation classification (works_for, located_in, etc.)
    - Use dependency parsing as fallback for simple relations
    - Extract relations between entities
    - _Requirements: 7.2_
  
  - [x] 10.3 Build knowledge graph constructor






    - Implement KnowledgeGraph class with node/edge management
    - Add entity linking and merging logic (string similarity + embeddings)
    - Handle contradictions by preserving both with source attribution
    - Convert to DGL format for GNN processing
    - _Requirements: 7.3, 7.4, 7.5, 7.6_
  
  - [ ] 10.4 Write property test for graph node extraction
    - **Property 16: Graph Node Extraction**
    - **Validates: Requirements 7.1**
  
  - [ ] 10.5 Write property test for graph monotonic growth
    - **Property 18: Graph Monotonic Growth**
    - **Validates: Requirements 7.3**
  
  - [ ] 10.6 Write property test for entity uniqueness
    - **Property 19: Entity Uniqueness**
    - **Validates: Requirements 7.4**
  
  - [ ] 10.7 Write property test for contradiction preservation
    - **Property 20: Contradiction Preservation**
    - **Validates: Requirements 7.5**

- [ ] 11. Implement GNN-based verdict predictor
  - [ ] 11.1 Design GNN architecture
    - Implement Graph Convolutional Network (GCN) with 2-3 layers
    - Add node feature initialization (embeddings from XLM-R)
    - Implement message passing and aggregation
    - Add readout layer for claim node representation
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [ ] 11.2 Train GNN on reasoning graphs
    - Prepare training data (graphs with ground truth verdicts)
    - Implement training loop with cross-entropy loss
    - Use DGL for efficient graph operations
    - Save best model checkpoint
    - _Requirements: 8.1, 8.2_
  
  - [ ] 11.3 Implement verdict prediction
    - Extract claim node representation from GNN
    - Apply classifier to predict verdict (Supported/Refuted/NEI)
    - Output confidence scores for each class
    - _Requirements: 8.4, 8.5_
  
  - [ ] 11.4 Write property test for GNN output format
    - **Property 21: GNN Output Format**
    - **Validates: Requirements 8.3**
  
  - [ ] 11.5 Write property test for verdict classification
    - **Property 22: Verdict Classification**
    - **Validates: Requirements 8.4, 8.5**

- [ ] 12. Checkpoint - Verify reasoning pipeline works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Build ReAct agent core
  - [ ] 13.1 Implement LLM controller
    - Setup Gemini 1.5 Flash API client (free tier: 15 RPM)
    - Setup Groq API client as alternative (Llama 3.1 70B, free tier: 30 RPM)
    - Implement local Llama 3.1 8B with 4-bit quantization as fallback
    - Add prompt templates for reasoning, action, observation
    - _Requirements: 2.1, 2.2_
  
  - [ ] 13.2 Implement ReAct loop orchestrator
    - Create AgentState class to track reasoning trace and working memory
    - Implement reason() method to generate thoughts
    - Implement act() method to decide on actions
    - Implement observe() method to process results
    - Implement termination logic (max iterations or sufficient evidence)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  
  - [ ] 13.3 Implement tool executor
    - Create Tool interface and concrete implementations (SearchTool (Exa), CrawlTool, CredibilityTool)
    - Implement tool parameter parsing from LLM output
    - Add error handling and retry logic
    - _Requirements: 2.2, 2.3_
  
  - [ ] 13.4 Write property test for ReAct loop structure
    - **Property 3: ReAct Loop Structure Invariant**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
  
  - [ ] 13.5 Write property test for agent memory monotonicity
    - **Property 4: Agent Memory Monotonicity**
    - **Validates: Requirements 2.4**
  
  - [ ] 13.6 Write property test for evidence collection termination
    - **Property 5: Evidence Collection Termination**
    - **Validates: Requirements 2.5**

- [ ] 14. Implement RAG-based explanation generator
  - [ ] 14.1 Build evidence retriever
    - Implement relevance scoring for evidence (stance + credibility)
    - Select top-k most relevant evidence pieces
    - Format evidence with source citations
    - _Requirements: 9.1_
  
  - [ ] 14.2 Implement RAG generation
    - Create prompt template with claim, verdict, and evidence
    - Use LLM (Gemini/Groq/Llama) to generate explanation
    - Ensure all facts are grounded in provided evidence
    - Insert inline citations with source URLs
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ] 14.3 Add reasoning trace to explanation
    - Format ReAct loop trace for human readability
    - Include all search queries, actions, and key observations
    - _Requirements: 9.4_
  
  - [ ] 14.4 Handle contradictory evidence
    - Detect when both supporting and refuting evidence exist
    - Present both sides with credibility scores
    - Explain uncertainty in verdict
    - _Requirements: 9.5_
  
  - [ ] 14.5 Write property test for RAG grounding
    - **Property 23: RAG Grounding**
    - **Validates: Requirements 9.1**
  
  - [ ] 14.6 Write property test for citation completeness
    - **Property 24: Citation Completeness**
    - **Validates: Requirements 9.2**
  
  - [ ] 14.7 Write property test for reasoning trace inclusion
    - **Property 25: Reasoning Trace Inclusion**
    - **Validates: Requirements 9.4**
  
  - [ ] 14.8 Write property test for contradiction presentation
    - **Property 26: Contradiction Presentation**
    - **Validates: Requirements 9.5**

- [ ] 15. Implement self-verification module
  - [ ] 15.1 Build claim extractor for explanations
    - Parse generated explanation to extract factual claims
    - Use claim detection model or rule-based extraction
    - _Requirements: 10.1_
  
  - [ ] 15.2 Implement verification loop
    - For each extracted claim, perform quick search
    - Check if claim is supported by collected evidence
    - Flag unsupported claims as potential hallucinations
    - _Requirements: 10.2, 10.3_
  
  - [ ] 15.3 Implement quality scoring
    - Compute quality score based on verification results
    - Score = (verified_claims / total_claims)
    - Output score with explanation
    - _Requirements: 10.4_
  
  - [ ] 15.4 Implement hallucination correction
    - Remove or revise claims that fail verification
    - Mark uncertain claims with caveats
    - Regenerate explanation if needed
    - _Requirements: 10.5_
  
  - [ ] 15.5 Write property test for self-verification execution
    - **Property 27: Self-Verification Execution**
    - **Validates: Requirements 10.1, 10.2**
  
  - [ ] 15.6 Write property test for quality score output
    - **Property 29: Quality Score Output**
    - **Validates: Requirements 10.4**

- [ ] 16. Checkpoint - Verify end-to-end pipeline works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Build dataset construction pipeline
  - [ ] 17.1 Implement claim collector
    - Crawl Vietnamese news articles
    - Extract claims using claim detection model
    - Store claims with metadata
    - _Requirements: 11.1_
  
  - [ ] 17.2 Implement evidence gatherer
    - For each claim, use ReAct agent to collect evidence
    - Store evidence with credibility scores
    - _Requirements: 11.2_
  
  - [ ] 17.3 Implement auto-labeling
    - Use full pipeline to generate initial labels
    - Compute confidence scores
    - Flag low-confidence examples for human review
    - _Requirements: 11.3_
  
  - [ ] 17.4 Create annotation interface (optional)
    - Build simple Gradio interface for label verification
    - Allow annotators to correct labels and add notes
    - _Requirements: 11.4_
  
  - [ ] 17.5 Implement dataset export
    - Export to JSONL format with all required fields
    - Include claim, evidence list, label, metadata
    - Split into train/val/test sets
    - _Requirements: 11.5_
  
  - [ ] 17.6 Write property test for dataset evidence association
    - **Property 31: Dataset Evidence Association**
    - **Validates: Requirements 11.2**
  
  - [ ] 17.7 Write property test for dataset schema compliance
    - **Property 32: Dataset Schema Compliance**
    - **Validates: Requirements 11.5**

- [ ] 18. Implement evaluation and comparison framework
  - [ ] 18.1 Build evaluation pipeline
    - Load test dataset
    - Run full pipeline on each claim
    - Compute metrics (Accuracy, Precision, Recall, F1)
    - Generate detailed performance report
    - _Requirements: 13.1_
  
  - [ ] 18.2 Implement ablation study support
    - Add configuration flags to enable/disable components
    - Run experiments with different configurations
    - Compare performance across configurations
    - _Requirements: 13.2_
  
  - [ ] 18.3 Add statistical significance testing
    - Implement paired t-test and McNemar's test
    - Compute p-values for method comparisons
    - Include in performance reports
    - _Requirements: 13.3_
  
  - [ ] 18.4 Create visualization tools
    - Generate plots (confusion matrix, precision-recall curves)
    - Create tables for paper publication
    - Export results in LaTeX format
    - _Requirements: 13.4_
  
  - [ ] 18.5 Implement experiment tracking
    - Log all hyperparameters and random seeds
    - Save complete configuration for reproducibility
    - Track experiments in SQLite database
    - _Requirements: 13.5_
  
  - [ ] 18.6 Write property test for evaluation metrics completeness
    - **Property 35: Evaluation Metrics Completeness**
    - **Validates: Requirements 13.1**
  
  - [ ] 18.7 Write property test for experiment reproducibility
    - **Property 38: Experiment Reproducibility**
    - **Validates: Requirements 13.5**

- [ ] 19. Build interactive demo system
  - [ ] 19.1 Create Gradio interface
    - Design UI with claim input textbox
    - Add submit button and progress indicators
    - Display results in organized sections
    - _Requirements: 14.1_
  
  - [ ] 19.2 Implement real-time progress display
    - Show current ReAct loop iteration
    - Display action being executed
    - Show evidence collected so far
    - _Requirements: 14.2_
  
  - [ ] 19.3 Implement result visualization
    - Display verdict with confidence score
    - Show evidence sources with clickable URLs
    - Render full explanation with citations
    - Visualize reasoning graph with interactive exploration
    - _Requirements: 14.3, 14.4_
  
  - [ ] 19.4 Deploy demo
    - Test demo in Colab notebook
    - Optionally deploy to Hugging Face Spaces (free tier)
    - Add example claims for users to try
    - _Requirements: 14.5_
  
  - [ ] 19.5 Write property test for demo real-time progress
    - **Property 39: Demo Real-time Progress**
    - **Validates: Requirements 14.2**
  
  - [ ] 19.6 Write property test for demo output completeness
    - **Property 40: Demo Output Completeness**
    - **Validates: Requirements 14.3**

- [ ] 20. Optimize for Colab Pro constraints
  - [ ] 20.1 Implement checkpoint system
    - Auto-save every 30 minutes
    - Save to Google Drive for persistence
    - Implement resume from checkpoint
    - _Requirements: 12.1, 12.4_
  
  - [ ] 20.2 Optimize memory usage
    - Apply 8-bit quantization to all large models
    - Use gradient checkpointing for training
    - Implement batch processing with optimal batch sizes
    - Clear GPU cache between operations
    - _Requirements: 12.2, 12.3, 12.5_
  
  - [ ] 20.3 Write property test for checkpoint periodicity
    - **Property 33: Checkpoint Periodicity**
    - **Validates: Requirements 12.1**
  
  - [ ] 20.4 Write property test for checkpoint recovery
    - **Property 34: Checkpoint Recovery**
    - **Validates: Requirements 12.4**

- [ ] 21. Final checkpoint - Complete system integration test
  - Run end-to-end test with real Vietnamese claims
  - Verify all components work together
  - Test demo system with multiple users
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 22. Documentation and paper preparation
  - Write comprehensive README with setup instructions
  - Document API usage and configuration
  - Create Jupyter notebooks with examples
  - Prepare experimental results for paper
  - Write methodology and results sections
  - Create figures and tables for publication