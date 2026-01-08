"""Mock LLM provider for testing RAG explanation generator with predefined data."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from .llm_controller import LLMProvider, LLMResponse
except ImportError:
    from llm_controller import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class MockExplanationTemplate:
    """Template for generating mock explanations."""
    
    supported_template: str = """Tuyên bố '{claim}' được hỗ trợ bởi bằng chứng với độ tin cậy {confidence:.2f}.

{evidence_summary}

{reasoning_summary}

Sources:
{sources}"""
    
    refuted_template: str = """Tuyên bố '{claim}' bị bác bỏ bởi bằng chứng với độ tin cậy {confidence:.2f}.

{evidence_summary}

{reasoning_summary}

Sources:
{sources}"""
    
    not_enough_info_template: str = """Tuyên bố '{claim}' không có đủ thông tin để xác minh với độ tin cậy {confidence:.2f}.

{evidence_summary}

{reasoning_summary}

Sources:
{sources}"""


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that generates explanations using predefined templates and test data."""
    
    def __init__(self, test_data_path: Optional[str] = None):
        """Initialize mock LLM provider.
        
        Args:
            test_data_path: Path to test data JSON file
        """
        self.test_data_path = test_data_path or "data/rag_test_dataset.json"
        self.templates = MockExplanationTemplate()
        self.test_data = self._load_test_data()
        
        logger.info(f"Initialized MockLLMProvider with {len(self.test_data.get('test_cases', []))} test cases")
    
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data from JSON file."""
        try:
            data_path = Path(self.test_data_path)
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Test data file not found: {data_path}")
                return {"test_cases": [], "mock_explanations": {}}
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {"test_cases": [], "mock_explanations": {}}
    
    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs
    ) -> LLMResponse:
        """Generate mock explanation based on input messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens (ignored in mock)
            temperature: Temperature (ignored in mock)
            **kwargs: Additional parameters (ignored in mock)
            
        Returns:
            LLMResponse with generated explanation
        """
        start_time = time.time()
        
        try:
            # Extract the user prompt
            user_message = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                raise ValueError("No user message found in input")
            
            # Generate explanation based on prompt content
            explanation = self._generate_explanation_from_prompt(user_message)
            
            # Calculate mock usage
            prompt_tokens = len(user_message.split()) * 1.3
            completion_tokens = len(explanation.split()) * 1.3
            
            usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens)
            }
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=explanation,
                usage=usage,
                model="mock-llm-v1",
                provider="mock",
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Mock LLM generation failed: {e}")
            # Return a basic fallback explanation
            explanation = "Không thể tạo giải thích chi tiết. Vui lòng kiểm tra lại dữ liệu đầu vào."
            
            return LLMResponse(
                content=explanation,
                usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                model="mock-llm-v1",
                provider="mock",
                latency=0.1
            )
    
    def _generate_explanation_from_prompt(self, prompt: str) -> str:
        """Generate explanation based on prompt content.
        
        Args:
            prompt: The RAG prompt containing claim, verdict, and evidence
            
        Returns:
            Generated explanation string
        """
        # Extract key information from prompt
        claim_text = self._extract_claim_from_prompt(prompt)
        verdict_label = self._extract_verdict_from_prompt(prompt)
        confidence_scores = self._extract_confidence_from_prompt(prompt)
        evidence_info = self._extract_evidence_from_prompt(prompt)
        
        # Check if we have a predefined explanation for this claim
        predefined_explanation = self._get_predefined_explanation(claim_text)
        if predefined_explanation:
            return predefined_explanation
        
        # Generate explanation using templates
        return self._generate_template_explanation(
            claim_text, verdict_label, confidence_scores, evidence_info
        )
    
    def _extract_claim_from_prompt(self, prompt: str) -> str:
        """Extract claim text from RAG prompt."""
        lines = prompt.split('\n')
        for line in lines:
            if line.startswith('CLAIM TO VERIFY:'):
                return line.replace('CLAIM TO VERIFY:', '').strip()
        return "Unknown claim"
    
    def _extract_verdict_from_prompt(self, prompt: str) -> str:
        """Extract verdict label from RAG prompt."""
        lines = prompt.split('\n')
        for line in lines:
            if line.startswith('VERDICT:'):
                verdict = line.replace('VERDICT:', '').strip().lower()
                if 'supported' in verdict:
                    return 'supported'
                elif 'refuted' in verdict:
                    return 'refuted'
                elif 'not enough info' in verdict or 'not_enough_info' in verdict:
                    return 'not_enough_info'
        return 'supported'
    
    def _extract_confidence_from_prompt(self, prompt: str) -> Dict[str, float]:
        """Extract confidence scores from RAG prompt."""
        confidence_scores = {"supported": 0.5, "refuted": 0.3, "not_enough_info": 0.2}
        
        lines = prompt.split('\n')
        for i, line in enumerate(lines):
            if 'CONFIDENCE SCORES:' in line:
                # Parse the next few lines for confidence scores
                for j in range(1, 4):
                    if i + j < len(lines):
                        score_line = lines[i + j]
                        if 'Supported:' in score_line:
                            try:
                                confidence_scores['supported'] = float(score_line.split(':')[1].strip())
                            except:
                                pass
                        elif 'Refuted:' in score_line:
                            try:
                                confidence_scores['refuted'] = float(score_line.split(':')[1].strip())
                            except:
                                pass
                        elif 'Not Enough Info:' in score_line:
                            try:
                                confidence_scores['not_enough_info'] = float(score_line.split(':')[1].strip())
                            except:
                                pass
                break
        
        return confidence_scores
    
    def _extract_evidence_from_prompt(self, prompt: str) -> List[Dict[str, str]]:
        """Extract evidence information from RAG prompt."""
        evidence_list = []
        lines = prompt.split('\n')
        
        in_evidence_section = False
        current_evidence = {}
        
        for line in lines:
            if line.startswith('EVIDENCE:'):
                in_evidence_section = True
                continue
            elif line.startswith('REASONING TRACE:') or line.startswith('CONTRADICTORY EVIDENCE:'):
                in_evidence_section = False
                if current_evidence:
                    evidence_list.append(current_evidence)
                    current_evidence = {}
                continue
            
            if in_evidence_section and line.strip():
                if line.startswith('[') and ']' in line:
                    # New evidence piece
                    if current_evidence:
                        evidence_list.append(current_evidence)
                    
                    # Extract evidence text
                    evidence_text = line.split(']', 1)[1].strip()
                    current_evidence = {"text": evidence_text}
                elif line.strip().startswith('Source:'):
                    # Extract source info
                    source_info = line.replace('Source:', '').strip()
                    if '(' in source_info and ')' in source_info:
                        title = source_info.split('(')[0].strip()
                        url = source_info.split('(')[1].replace(')', '').strip()
                        current_evidence["source_title"] = title
                        current_evidence["source_url"] = url
        
        # Add the last evidence if exists
        if current_evidence:
            evidence_list.append(current_evidence)
        
        return evidence_list
    
    def _get_predefined_explanation(self, claim_text: str) -> Optional[str]:
        """Get predefined explanation for known claims."""
        mock_explanations = self.test_data.get("mock_explanations", {})
        
        # Check for exact matches or partial matches
        for key, explanation in mock_explanations.items():
            test_case = self._get_test_case_by_id(key)
            if test_case and claim_text in test_case.get("claim", {}).get("text", ""):
                return explanation
        
        return None
    
    def _get_test_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get test case by ID."""
        for test_case in self.test_data.get("test_cases", []):
            if test_case.get("id") == case_id:
                return test_case
        return None
    
    def _generate_template_explanation(
        self,
        claim_text: str,
        verdict_label: str,
        confidence_scores: Dict[str, float],
        evidence_info: List[Dict[str, str]]
    ) -> str:
        """Generate explanation using templates."""
        
        # Get the appropriate template
        if verdict_label == "supported":
            template = self.templates.supported_template
        elif verdict_label == "refuted":
            template = self.templates.refuted_template
        else:
            template = self.templates.not_enough_info_template
        
        # Get confidence for this verdict
        confidence = confidence_scores.get(verdict_label, 0.5)
        
        # Generate evidence summary
        evidence_summary = self._generate_evidence_summary(evidence_info, verdict_label)
        
        # Generate reasoning summary
        reasoning_summary = "Quá trình xác minh bao gồm việc tìm kiếm và phân tích các nguồn thông tin đáng tin cậy."
        
        # Generate sources list
        sources = self._generate_sources_list(evidence_info)
        
        # Fill template
        explanation = template.format(
            claim=claim_text,
            confidence=confidence,
            evidence_summary=evidence_summary,
            reasoning_summary=reasoning_summary,
            sources=sources
        )
        
        return explanation
    
    def _generate_evidence_summary(self, evidence_info: List[Dict[str, str]], verdict_label: str) -> str:
        """Generate evidence summary based on verdict."""
        if not evidence_info:
            return "Không có bằng chứng cụ thể được cung cấp."
        
        summary_parts = []
        for i, evidence in enumerate(evidence_info[:3], 1):  # Top 3 evidence
            text = evidence.get("text", "")[:150] + "..." if len(evidence.get("text", "")) > 150 else evidence.get("text", "")
            summary_parts.append(f"[{i}] {text}")
        
        return "\n".join(summary_parts)
    
    def _generate_sources_list(self, evidence_info: List[Dict[str, str]]) -> str:
        """Generate sources list for citation."""
        if not evidence_info:
            return "Không có nguồn được cung cấp."
        
        sources = []
        for i, evidence in enumerate(evidence_info[:3], 1):
            title = evidence.get("source_title", f"Nguồn {i}")
            url = evidence.get("source_url", "")
            sources.append(f"[{i}] {title} - {url}")
        
        return "\n".join(sources)


def create_mock_llm_controller(test_data_path: Optional[str] = None):
    """Create LLM controller with mock provider for testing.
    
    Args:
        test_data_path: Path to test data JSON file
        
    Returns:
        LLMController configured with mock provider
    """
    from llm_controller import LLMController
    
    # Create mock controller
    controller = LLMController(providers=[])  # No real providers
    
    # Add mock provider
    mock_provider = MockLLMProvider(test_data_path)
    controller.providers["mock"] = mock_provider
    controller.provider_order = ["mock"]
    
    logger.info("Created LLM controller with mock provider")
    return controller


if __name__ == "__main__":
    # Test mock provider
    import logging
    logging.basicConfig(level=logging.INFO)
    
    provider = MockLLMProvider()
    
    # Test message
    messages = [
        {"role": "system", "content": "You are a fact-checking assistant."},
        {"role": "user", "content": """
CLAIM TO VERIFY: Việt Nam có 63 tỉnh thành phố trực thuộc trung ương

VERDICT: SUPPORTED
CONFIDENCE SCORES:
- Supported: 0.88
- Refuted: 0.08
- Not Enough Info: 0.04

EVIDENCE:
[1] Theo Hiến pháp năm 2013, Việt Nam được chia thành 63 đơn vị hành chính cấp tỉnh
    Source: Tổ chức hành chính địa phương Việt Nam (https://baochinhphu.vn/hanh-chinh-dia-phuong)

Please generate a comprehensive explanation.
"""}
    ]
    
    response = provider.generate(messages)
    print("Generated explanation:")
    print(response.content)
    print(f"\nUsage: {response.usage}")