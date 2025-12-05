# Vietnamese Autonomous Fact-Checking System

A research project building an automated fact-checking system for Vietnamese, utilizing an Agent-based architecture combined with advanced AI models and Knowledge Graphs.

## 🚀 Achievements

The system has completed its Core Infrastructure and key AI modules:

### 1. AI & NLP Modules
*   **Claim Detection**:
    *   Uses **PhoBERT** (fine-tuned) to classify sentences: Factual Claim (verifiable) vs Opinion/Question.
    *   Complete training pipeline built on Kaggle (`notebooks/train_phobert_kaggle.ipynb`).
*   **Stance Detection**:
    *   Uses **XLM-RoBERTa** for Cross-lingual capabilities.
    *   Supports stance matching between Claim (Vietnamese) and Evidence (Vietnamese/English).
    *   3-label classification: Support, Refute, Neutral.
*   **Named Entity Recognition (NER)**:
    *   Hybrid System: Combines Rule-based (for Vietnamese) and spaCy (for English).
    *   Supports extraction of: PERSON, ORG, LOC, DATE, NUMBER.
*   **Translation Service**:
    *   Integrates **MarianMT** (Helsinki-NLP) running locally for high speed.
    *   Automatic fallback to Google Translate API when needed.
    *   Smart caching mechanism to reduce computational load.

### 2. Search & Retrieval
*   **Semantic Search**: Integrates **Exa Search API** for neural semantic search.
*   **Web Crawler**:
    *   **Static**: Uses `trafilatura` and `BeautifulSoup` for mainstream news sites.
    *   **Dynamic**: Integrates `Selenium` (Headless Chrome) to handle dynamic websites (JS-heavy).
    *   Complies with `robots.txt` and Rate Limiting mechanisms.
*   **Query Generation**: Automatically generates diverse queries (Direct, Entity-focused, Question-based) to optimize search results.

### 3. Reasoning & Knowledge Graph
*   **Knowledge Graph Builder**:
    *   Builds dynamic knowledge graphs from collected evidence.
    *   Nodes: Claim, Entity, Evidence.
    *   Edges: Relations (supports, refutes, mentions, works_for, etc.).
    *   Supports export to DGL (Deep Graph Library) format for GNNs.
*   **Credibility Analysis**: Evaluates source credibility based on domain whitelists and Media Bias/Fact Check database.

### 4. Infrastructure
*   **Caching System**: Redis-based caching for Search, Content, Translation, and Credibility Scores (customizable TTL).
*   **Configuration**: Centralized configuration management (`src/config.py`) for the entire system.
*   **Testing**: High test coverage with `pytest` and `hypothesis` (Property-based testing).

---

## 🧠 AI Focus

The system is designed around core AI components:

1.  **PhoBERT (VinAI)**: Backbone for Vietnamese processing, fine-tuned for specific classification tasks.
2.  **XLM-RoBERTa (Facebook AI)**: Cross-lingual bridge, allowing the system to leverage vast English knowledge sources to verify Vietnamese information.
3.  **Graph Neural Networks (GNN)**: (In development) Uses graph structures to model complex relationships between entities and evidence, helping to make more accurate verdicts.
4.  **ReAct Framework**: Agent architecture allowing the system to "Reason" and "Act" iteratively to find missing information.

---

## 🛠️ Future Work

To finalize the system towards a real-world product, focus is needed on the following items:

1.  **Fine-tune PhoBERT for NER**: Replace the current rule-based NER module with a specialized Deep Learning model for Vietnamese to increase accuracy.
2.  **Finalize ReAct Agent**:
    *   Integrate LLM (Gemini Pro / GPT-4) as the "brain" coordinating tools (Search, Crawl, Verify).
    *   Realize the loop: *Thought -> Action -> Observation -> Thought*.
3.  **Expand Dataset**:
    *   Collect more labeled data for Claim Detection and Stance Detection (specific to Vietnamese news).
    *   Enhance data against new types of fake news (Deepfake text, propaganda).
4.  **Deployment & Interface**:
    *   Build Backend API (FastAPI).
    *   Develop Dashboard to visualize Knowledge Graph and reasoning process.
    *   Dockerize for easy deployment.

---

## 📂 Project Structure

```
vietnamese-fact-checking/
├── src/                    # Main source code
│   ├── claim_detector.py       # Claim detection model (PhoBERT)
│   ├── stance_detector.py      # Stance detection model (XLM-R)
│   ├── graph_builder.py        # Knowledge Graph builder
│   ├── web_crawler.py          # Data collection (Static/Dynamic)
│   ├── exa_search_client.py    # Semantic search
│   ├── translation_service.py  # Multilingual translation
│   ├── cache_manager.py        # Redis Cache management
│   └── config.py               # System configuration
├── notebooks/              # Model training notebooks
├── scripts/                # Utility scripts (data prep, testing)
├── tests/                  # Unit tests & Property-based tests
└── data/                   # Training & testing data
```

## 🔧 Installation & Usage

1.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configuration**:
    *   Install Redis.
    *   Create `.env` file and fill in API Keys (Exa, OpenAI/Gemini, etc.).
3.  **Run Tests**:
    ```bash
    pytest tests/
    ```

---
*Scientific Research Project 2024-2025*
