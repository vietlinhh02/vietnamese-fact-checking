# Requirements Document

## Introduction

Hệ thống kiểm chứng thông tin tự chủ cho tiếng Việt (Vietnamese Autonomous Fact-Checking System) là một công cụ nghiên cứu khoa học nhằm xác minh tính xác thực của các tuyên bố bằng tiếng Việt. Hệ thống sử dụng kiến trúc tác tử tự chủ (agent-based architecture) với LLM làm bộ não trung tâm, thực hiện vòng lặp Suy luận-Hành động-Quan sát (ReAct framework) để thu thập bằng chứng đa bước, xây dựng biểu đồ suy luận động, và tạo ra các giải thích trung thực có khả năng truy vết.

Hệ thống được thiết kế để chạy trên Google Colab Pro, tối ưu chi phí bằng cách sử dụng các mô hình open-source (PhoBERT, XLM-RoBERTa) và API miễn phí, đồng thời có khả năng kiểm chứng chéo ngôn ngữ (cross-lingual) bằng cách tận dụng nguồn dữ liệu tiếng Anh.

## Glossary

- **Hệ thống (System)**: Hệ thống kiểm chứng thông tin tự chủ cho tiếng Việt
- **Tuyên bố (Claim)**: Một phát biểu khẳng định có thể kiểm chứng được tính đúng sai
- **Bằng chứng (Evidence)**: Thông tin từ các nguồn đáng tin cậy được sử dụng để xác minh tuyên bố
- **Tác tử (Agent)**: Thành phần LLM điều phối vòng lặp suy luận-hành động-quan sát
- **Biểu đồ Suy luận (Reasoning Graph)**: Cấu trúc đồ thị tri thức chứa các thực thể, mệnh đề và mối quan hệ logic
- **Lập trường (Stance)**: Mối quan hệ giữa bằng chứng và tuyên bố (ủng hộ, phản đối, trung lập)
- **Phán quyết (Verdict)**: Kết luận cuối cùng về tính xác thực của tuyên bố (Đúng, Sai, Không đủ thông tin)
- **RAG (Retrieval-Augmented Generation)**: Kỹ thuật sinh văn bản dựa trên bằng chứng đã thu thập
- **XLM-R**: Mô hình ngôn ngữ đa ngôn ngữ Cross-lingual RoBERTa
- **PhoBERT**: Mô hình BERT được huấn luyện trước cho tiếng Việt
- **GNN (Graph Neural Network)**: Mạng nơ-ron đồ thị
- **NER (Named Entity Recognition)**: Nhận dạng thực thể có tên
- **RE (Relation Extraction)**: Trích xuất quan hệ giữa các thực thể
- **Nguồn tin uy tín (Credible Source)**: Các trang báo điện tử được quản lý bởi nhà nước Việt Nam

## Requirements

### Requirement 1: Phát hiện Tuyên bố

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống tự động phát hiện các tuyên bố cần kiểm chứng từ văn bản tiếng Việt, để có thể lọc ra những phát biểu có thể xác minh được.

#### Acceptance Criteria

1. WHEN the System receives Vietnamese text input THEN the System SHALL identify sentences containing verifiable factual claims
2. WHEN analyzing a sentence THEN the System SHALL distinguish between factual claims and opinions, questions, or commands
3. WHEN detecting claims THEN the System SHALL extract claim text with surrounding context for verification
4. WHEN multiple claims exist in one text THEN the System SHALL identify and separate each individual claim
5. WHERE the input contains mixed content THEN the System SHALL classify each sentence as claim or non-claim with confidence scores

### Requirement 2: Thu thập Bằng chứng qua Vòng lặp ReAct

**User Story:** Là một nhà nghiên cứu, tôi muốn tác tử LLM tự động thu thập bằng chứng thông qua vòng lặp suy luận-hành động-quan sát, để có thể tìm kiếm thông tin đa bước một cách thông minh.

#### Acceptance Criteria

1. WHEN the Agent receives a claim THEN the System SHALL generate reasoning thoughts to decompose the verification task into logical steps
2. WHEN reasoning is complete THEN the Agent SHALL decide on specific actions using available tools (Search, API_call)
3. WHEN an action is executed THEN the System SHALL return observations to the Agent for further reasoning
4. WHEN observations are received THEN the Agent SHALL integrate information into working memory and adjust the search strategy
5. WHEN sufficient evidence is collected THEN the Agent SHALL terminate the reasoning loop and proceed to verdict generation
6. WHILE collecting evidence THEN the System SHALL maintain a trace of all reasoning-action-observation cycles for explainability

### Requirement 3: Crawl và Trích xuất Nội dung Web

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống crawl và trích xuất nội dung từ các trang báo uy tín của Việt Nam, để có nguồn bằng chứng đáng tin cậy cho việc kiểm chứng.

#### Acceptance Criteria

1. WHEN the System needs evidence THEN the System SHALL crawl content from Vietnamese state-managed news sources (VnExpress, VTV, VOV, Tuổi Trẻ, Thanh Niên)
2. WHEN crawling a webpage THEN the System SHALL handle dynamic content by rendering JavaScript before extraction
3. WHEN extracting content THEN the System SHALL remove boilerplate elements (ads, menus, footers) and retain only main article text
4. WHEN parsing HTML THEN the System SHALL extract article metadata (title, author, publish date, URL)
5. WHEN extraction fails THEN the System SHALL log errors and continue with alternative sources

### Requirement 4: Tìm kiếm Đa ngôn ngữ

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống có khả năng tìm kiếm bằng chứng bằng cả tiếng Việt và tiếng Anh, để tận dụng kho tri thức toàn cầu cho việc kiểm chứng.

#### Acceptance Criteria

1. WHEN the Agent generates search queries THEN the System SHALL create queries in both Vietnamese and English
2. WHEN searching in English THEN the System SHALL translate Vietnamese claims to English while preserving semantic meaning
3. WHEN search results are returned THEN the System SHALL collect evidence from both Vietnamese and English sources
4. WHEN using free APIs THEN the System SHALL implement rate limiting and caching to stay within quota limits
5. WHERE translation is needed THEN the System SHALL use free translation APIs (Google Translate API free tier or MarianMT models)

### Requirement 5: Đánh giá Độ tin cậy Nguồn tin

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống đánh giá độ tin cậy của các nguồn cung cấp bằng chứng, để có thể cân nhắc trọng số thông tin trong quá trình suy luận.

#### Acceptance Criteria

1. WHEN evidence is collected from a source THEN the System SHALL compute a credibility score for that source
2. WHEN analyzing Vietnamese sources THEN the System SHALL assign higher credibility to state-managed news outlets
3. WHEN analyzing domain features THEN the System SHALL evaluate TLD, domain age, HTTPS usage, and writing style
4. WHERE external credibility APIs are available THEN the System SHALL integrate free-tier API data (Media Bias Fact Check API)
5. WHEN credibility scores are computed THEN the System SHALL store scores with evidence for downstream reasoning

### Requirement 6: Xác định Lập trường Xuyên ngôn ngữ

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống xác định lập trường của bằng chứng đối với tuyên bố mà không cần dịch, để bảo toàn ngữ nghĩa gốc và tránh lỗi dịch máy.

#### Acceptance Criteria

1. WHEN comparing Vietnamese claim with English evidence THEN the System SHALL use XLM-RoBERTa model for direct cross-lingual stance detection
2. WHEN comparing Vietnamese claim with Vietnamese evidence THEN the System SHALL use fine-tuned PhoBERT or XLM-R model
3. WHEN stance detection is performed THEN the System SHALL classify relationship as Support, Refute, or Neutral
4. WHEN stance is determined THEN the System SHALL output confidence scores for each stance class
5. WHERE training data is limited THEN the System SHALL use few-shot learning or data augmentation techniques

### Requirement 7: Xây dựng Biểu đồ Suy luận Động

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống xây dựng biểu đồ tri thức động trong quá trình thu thập bằng chứng, để cấu trúc hóa thông tin và hỗ trợ suy luận phức tạp.

#### Acceptance Criteria

1. WHEN evidence is collected THEN the System SHALL extract named entities (persons, organizations, locations) as graph nodes
2. WHEN entities are extracted THEN the System SHALL identify relationships between entities as graph edges
3. WHEN new information arrives THEN the System SHALL incrementally add nodes and edges to the reasoning graph
4. WHEN duplicate entities are detected THEN the System SHALL merge them into single nodes with consolidated information
5. WHEN contradictions are found THEN the System SHALL represent conflicting information with source attribution and credibility scores
6. WHEN the graph is complete THEN the System SHALL output a structured knowledge graph for GNN processing

### Requirement 8: Dự đoán Tính xác thực bằng GNN

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống sử dụng mạng nơ-ron đồ thị để suy luận trên biểu đồ tri thức, để đưa ra phán quyết chính xác dựa trên toàn bộ bằng chứng và mối quan hệ logic.

#### Acceptance Criteria

1. WHEN the reasoning graph is constructed THEN the System SHALL input the graph into a Graph Neural Network model
2. WHEN GNN processes the graph THEN the System SHALL propagate information between nodes through multiple layers
3. WHEN message passing is complete THEN the System SHALL extract the final representation of the claim node
4. WHEN classification is performed THEN the System SHALL predict verdict as Supported, Refuted, or Not Enough Info
5. WHEN verdict is generated THEN the System SHALL output confidence scores for each verdict class

### Requirement 9: Tạo Giải thích Trung thực với RAG

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống tạo ra lời giải thích trung thực có trích dẫn nguồn, để người dùng có thể hiểu và xác minh quá trình suy luận.

#### Acceptance Criteria

1. WHEN generating explanation THEN the System SHALL use Retrieval-Augmented Generation to ground output in collected evidence
2. WHEN creating explanation text THEN the System SHALL cite specific sources with URLs for each claim made
3. WHEN using LLM for generation THEN the System SHALL use free-tier APIs (Gemini API, Groq API) or open-source models (Llama, Mistral)
4. WHEN explanation is complete THEN the System SHALL include reasoning trace showing all search steps and evidence considered
5. WHEN contradictory evidence exists THEN the System SHALL present both sides with source credibility information

### Requirement 10: Xác minh và Kiểm tra Chất lượng

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống tự động xác minh lại các tuyên bố trong lời giải thích, để đảm bảo không có thông tin bịa đặt (hallucination).

#### Acceptance Criteria

1. WHEN explanation is generated THEN the System SHALL extract factual claims from the explanation text
2. WHEN claims are extracted THEN the System SHALL perform quick verification searches for each claim
3. IF verification fails for any claim THEN the System SHALL flag the explanation as uncertain or revise it
4. WHEN verification is complete THEN the System SHALL output a quality score indicating explanation reliability
5. WHERE LLM hallucination is detected THEN the System SHALL remove or correct the hallucinated content

### Requirement 11: Xây dựng Dataset Tiếng Việt

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống hỗ trợ xây dựng dataset kiểm chứng thông tin cho tiếng Việt, để có dữ liệu huấn luyện và đánh giá mô hình.

#### Acceptance Criteria

1. WHEN building dataset THEN the System SHALL crawl claims from Vietnamese news articles and social media
2. WHEN claims are collected THEN the System SHALL automatically gather candidate evidence from credible sources
3. WHEN evidence is gathered THEN the System SHALL use the Agent to propose initial labels (Supported/Refuted/NEI)
4. WHERE human annotation is available THEN the System SHALL support annotation interface for label verification
5. WHEN dataset is complete THEN the System SHALL export data in standard format (JSONL) with claim, evidence, label, and metadata

### Requirement 12: Tối ưu hóa cho Colab Pro

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống được tối ưu để chạy hiệu quả trên Google Colab Pro, để tận dụng tài nguyên GPU miễn phí và quản lý session timeout.

#### Acceptance Criteria

1. WHEN running on Colab THEN the System SHALL implement checkpoint saving every 30 minutes to handle session timeouts
2. WHEN using GPU THEN the System SHALL batch process claims to maximize GPU utilization
3. WHEN memory is limited THEN the System SHALL use gradient checkpointing and mixed precision training for large models
4. WHEN session disconnects THEN the System SHALL automatically resume from last checkpoint upon reconnection
5. WHERE models are large THEN the System SHALL use quantization (4-bit, 8-bit) to reduce memory footprint

### Requirement 13: Đánh giá và So sánh Phương pháp

**User Story:** Là một nhà nghiên cứu, tôi muốn hệ thống hỗ trợ đánh giá và so sánh các phương pháp khác nhau, để có thể xuất bản kết quả nghiên cứu khoa học.

#### Acceptance Criteria

1. WHEN evaluation is performed THEN the System SHALL compute standard metrics (Accuracy, Precision, Recall, F1-Score)
2. WHEN comparing methods THEN the System SHALL support ablation studies by enabling/disabling specific components
3. WHEN results are generated THEN the System SHALL output detailed performance reports with statistical significance tests
4. WHEN visualizing results THEN the System SHALL create plots and tables suitable for academic papers
5. WHERE multiple runs are needed THEN the System SHALL support experiment tracking with different hyperparameters and random seeds

### Requirement 14: Tạo Demo System

**User Story:** Là một nhà nghiên cứu, tôi muốn có một demo system tương tác, để có thể trình bày kết quả nghiên cứu và kiểm chứng các claim mẫu.

#### Acceptance Criteria

1. WHEN demo is launched THEN the System SHALL provide a web interface (Gradio or Streamlit) for claim input
2. WHEN user submits a claim THEN the System SHALL display real-time progress of the reasoning-action-observation loop
3. WHEN verification is complete THEN the System SHALL show verdict, confidence score, evidence sources, and explanation
4. WHEN displaying results THEN the System SHALL visualize the reasoning graph with interactive exploration
5. WHERE demo is deployed THEN the System SHALL run within Colab notebook or be deployable to free hosting (Hugging Face Spaces)
