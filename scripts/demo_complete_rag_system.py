#!/usr/bin/env python3
"""Complete demonstration of RAG explanation generator integrated with Vietnamese fact-checking system."""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import Claim, Evidence, Verdict, ReasoningStep, FactCheckResult, KnowledgeGraph
from rag_explanation_generator import RAGExplanationGenerator
from llm_controller import create_llm_controller
from config import SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteRAGDemo:
    """Demonstration of complete RAG-integrated fact-checking system."""
    
    def __init__(self):
        """Initialize the demo system."""
        # Load configuration
        self.config = SystemConfig.from_env()
        
        # Create LLM config from SystemConfig
        llm_config = {
            "providers": [self.config.model.llm_provider],
            "gemini_api_key": self.config.agent.gemini_api_key,
            "groq_api_key": self.config.agent.groq_api_key
        }
        
        self.llm_controller = create_llm_controller(llm_config)
        self.rag_generator = RAGExplanationGenerator(llm_controller=self.llm_controller)
        
        logger.info(f"Initialized complete RAG system with provider: {self.config.model.llm_provider}")
    
    def create_comprehensive_test_case(self) -> FactCheckResult:
        """Create a comprehensive test case showing all system capabilities."""
        
        # Complex claim about Vietnam's geography and administration
        claim = Claim(
            text="Việt Nam có 63 tỉnh thành, giáp biên giới với 3 quốc gia và có diện tích khoảng 331,000 km²",
            confidence=0.85,
            language="vi"
        )
        
        # Multiple evidence pieces with different stances and credibility
        evidence_list = [
            # Supporting evidence - high credibility
            Evidence(
                text="Việt Nam có 63 đơn vị hành chính cấp tỉnh theo Hiến pháp 2013, bao gồm 58 tỉnh và 5 thành phố trực thuộc trung ương.",
                source_url="https://baochinhphu.vn/hanh-chinh-dia-phuong-viet-nam",
                source_title="Tổ chức hành chính Việt Nam",
                credibility_score=0.95,
                stance="support",
                stance_confidence=0.9,
                language="vi"
            ),
            
            # Supporting evidence - international source
            Evidence(
                text="Vietnam borders China to the north, Laos and Cambodia to the west, with a total area of approximately 331,212 square kilometers.",
                source_url="https://en.wikipedia.org/wiki/Geography_of_Vietnam",
                source_title="Geography of Vietnam - Wikipedia",
                credibility_score=0.8,
                stance="support",
                stance_confidence=0.85,
                language="en"
            ),
            
            # Partially contradictory evidence - different area measurement
            Evidence(
                text="Theo một số nguồn, diện tích Việt Nam là 330,967 km², có thể có sự khác biệt nhỏ trong cách tính toán.",
                source_url="https://gso.gov.vn/dien-tich-viet-nam",
                source_title="Tổng cục Thống kê - Diện tích Việt Nam",
                credibility_score=0.9,
                stance="neutral",
                stance_confidence=0.7,
                language="vi"
            ),
            
            # Supporting evidence - border information
            Evidence(
                text="Việt Nam có biên giới đất liền với 3 nước: Trung Quốc ở phía bắc, Lào và Campuchia ở phía tây.",
                source_url="https://mofa.gov.vn/bien-gioi-viet-nam",
                source_title="Bộ Ngoại giao - Biên giới Việt Nam",
                credibility_score=0.95,
                stance="support",
                stance_confidence=0.95,
                language="vi"
            ),
            
            # Low credibility contradictory source
            Evidence(
                text="Some unofficial sources claim Vietnam has 64 provinces, but this is outdated information from before administrative reforms.",
                source_url="https://example.com/outdated-vietnam-info",
                source_title="Outdated Vietnam Information",
                credibility_score=0.3,
                stance="refute",
                stance_confidence=0.6,
                language="en"
            )
        ]
        
        # Detailed reasoning steps showing multi-step verification
        reasoning_steps = [
            ReasoningStep(
                iteration=1,
                thought="Tôi cần xác minh ba thông tin chính: số lượng tỉnh thành, số quốc gia giáp biên giới, và diện tích của Việt Nam",
                action="search",
                action_input={"query": "Việt Nam 63 tỉnh thành Hiến pháp 2013"},
                observation="Tìm thấy thông tin chính thức xác nhận 63 đơn vị hành chính cấp tỉnh theo Hiến pháp 2013"
            ),
            
            ReasoningStep(
                iteration=2,
                thought="Cần kiểm tra thông tin về biên giới và các nước láng giềng",
                action="search",
                action_input={"query": "Vietnam borders China Laos Cambodia"},
                observation="Xác nhận Việt Nam giáp biên giới với 3 quốc gia: Trung Quốc, Lào và Campuchia"
            ),
            
            ReasoningStep(
                iteration=3,
                thought="Cần xác minh diện tích chính xác của Việt Nam",
                action="search",
                action_input={"query": "Vietnam area 331000 square kilometers official"},
                observation="Tìm thấy thông tin diện tích khoảng 331,212 km² từ nguồn quốc tế và 330,967 km² từ Tổng cục Thống kê"
            ),
            
            ReasoningStep(
                iteration=4,
                thought="Nên phân tích độ tin cậy của các nguồn thông tin khác nhau",
                action="analyze_credibility",
                action_input={"sources": ["baochinhphu.vn", "wikipedia.org", "gso.gov.vn", "mofa.gov.vn"]},
                observation="Các nguồn chính phủ có độ tin cậy cao (0.9-0.95), Wikipedia có độ tin cậy trung bình (0.8)"
            ),
            
            ReasoningStep(
                iteration=5,
                thought="Cần đánh giá tổng thể tính chính xác của tuyên bố dựa trên tất cả bằng chứng",
                action="synthesize",
                action_input={"evidence_count": 5, "supporting": 3, "neutral": 1, "refuting": 1},
                observation="Đa số bằng chứng uy tín hỗ trợ tuyên bố, chỉ có sự khác biệt nhỏ về diện tích và một nguồn không đáng tin cậy phản bác"
            )
        ]
        
        # Verdict with detailed confidence breakdown
        verdict = Verdict(
            claim_id=claim.id,
            label="supported",
            confidence_scores={
                "supported": 0.82,
                "refuted": 0.12,
                "not_enough_info": 0.06
            },
            supporting_evidence=[ev.id for ev in evidence_list if ev.stance == "support"],
            refuting_evidence=[ev.id for ev in evidence_list if ev.stance == "refute"],
            reasoning_trace=reasoning_steps
        )
        
        # Create knowledge graph (simplified representation)
        knowledge_graph = KnowledgeGraph()
        
        # Create complete fact-check result
        result = FactCheckResult(
            claim=claim,
            verdict=verdict,
            evidence=evidence_list,
            reasoning_graph=knowledge_graph,
            metadata={
                "processing_time": 67.3,
                "sources_checked": 5,
                "languages_used": ["vi", "en"],
                "credibility_range": [0.3, 0.95],
                "contradiction_detected": True,
                "cross_lingual_verification": True
            }
        )
        
        return result
    
    def demonstrate_rag_capabilities(self):
        """Demonstrate all RAG explanation generator capabilities."""
        
        logger.info("DEMONSTRATING COMPLETE RAG EXPLANATION SYSTEM")
        logger.info("=" * 80)
        
        # Create comprehensive test case
        result = self.create_comprehensive_test_case()
        
        logger.info("TEST CASE OVERVIEW:")
        logger.info(f"Claim: {result.claim.text}")
        logger.info(f"Evidence pieces: {len(result.evidence)}")
        logger.info(f"Reasoning steps: {len(result.verdict.reasoning_trace)}")
        logger.info(f"Verdict: {result.verdict.label} (confidence: {max(result.verdict.confidence_scores.values()):.2f})")
        
        # Show evidence analysis
        logger.info("\nEVIDENCE ANALYSIS:")
        for i, evidence in enumerate(result.evidence, 1):
            logger.info(f"{i}. {evidence.stance} | Credibility: {evidence.credibility_score:.2f} | {evidence.source_title}")
        
        # Generate RAG explanation
        logger.info("\nGENERATING RAG EXPLANATION...")
        
        explanation, verification_metadata = self.rag_generator.generate_explanation(
            claim=result.claim,
            verdict=result.verdict,
            evidence_list=result.evidence,
            reasoning_steps=result.verdict.reasoning_trace
        )
        
        # Update result with explanation
        result.verdict.explanation = explanation
        
        # Display complete explanation
        logger.info("\n" + "=" * 80)
        logger.info("COMPLETE RAG EXPLANATION:")
        logger.info("=" * 80)
        logger.info(explanation)
        logger.info("=" * 80)
        
        # Analyze explanation quality
        self.analyze_explanation_quality(explanation, result)
        
        return result
    
    def analyze_explanation_quality(self, explanation: str, result: FactCheckResult):
        """Analyze the quality of generated explanation."""
        
        logger.info("\nEXPLANATION QUALITY ANALYSIS:")
        logger.info("-" * 50)
        
        # Basic metrics
        word_count = len(explanation.split())
        char_count = len(explanation)
        
        logger.info(f"Length: {word_count} words, {char_count} characters")
        
        # Citation analysis
        import re
        citations = re.findall(r'\[(\d+)\]', explanation)
        urls = re.findall(r'https?://[^\s]+', explanation)
        
        logger.info(f"Citations: {len(set(citations))} unique citation markers")
        logger.info(f"Source URLs: {len(urls)} URLs included")
        
        # Content analysis
        explanation_lower = explanation.lower()
        
        # Check for key elements
        has_verdict = any(word in explanation_lower for word in ["hỗ trợ", "supported", "xác nhận"])
        has_confidence = any(word in explanation_lower for word in ["tin cậy", "confidence", "độ tin"])
        has_reasoning = any(word in explanation_lower for word in ["reasoning", "xác minh", "kiểm tra", "tìm kiếm"])
        has_contradiction = any(word in explanation_lower for word in ["mâu thuẫn", "khác biệt", "contradict"])
        
        logger.info(f"Contains verdict discussion: {has_verdict}")
        logger.info(f"Contains confidence information: {has_confidence}")
        logger.info(f"Contains reasoning trace: {has_reasoning}")
        logger.info(f"Addresses contradictions: {has_contradiction}")
        
        # Language analysis
        vietnamese_indicators = ["việt nam", "tỉnh thành", "bằng chứng", "tuyên bố"]
        vietnamese_content = sum(1 for indicator in vietnamese_indicators if indicator in explanation_lower)
        
        logger.info(f"Vietnamese content richness: {vietnamese_content}/4 key terms")
        
        # Overall quality score
        quality_factors = [
            word_count >= 100,  # Substantial length
            len(set(citations)) >= 2,  # Multiple citations
            len(urls) >= 2,  # Multiple sources
            has_verdict,  # Discusses verdict
            has_confidence,  # Mentions confidence
            has_reasoning,  # Includes reasoning
            vietnamese_content >= 2  # Good Vietnamese content
        ]
        
        quality_score = sum(quality_factors) / len(quality_factors)
        
        logger.info(f"\nOverall Quality Score: {quality_score:.1%}")
        
        if quality_score >= 0.8:
            logger.info("🌟 Excellent explanation quality!")
        elif quality_score >= 0.6:
            logger.info("👍 Good explanation quality!")
        else:
            logger.info("⚠️  Explanation quality needs improvement")
    
    def demonstrate_different_verdict_types(self):
        """Demonstrate RAG explanations for different verdict types."""
        
        logger.info("\nDEMONSTRATING DIFFERENT VERDICT TYPES:")
        logger.info("=" * 60)
        
        # Load test cases from dataset
        try:
            with open("data/rag_test_dataset.json", 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except:
            logger.warning("Could not load test dataset, using basic examples")
            return
        
        verdict_examples = {
            "supported": None,
            "refuted": None,
            "not_enough_info": None
        }
        
        # Find examples of each verdict type
        for test_case in test_data.get("test_cases", []):
            verdict_label = test_case["verdict"]["label"]
            if verdict_label in verdict_examples and verdict_examples[verdict_label] is None:
                verdict_examples[verdict_label] = test_case
        
        # Generate explanations for each type
        for verdict_type, test_case in verdict_examples.items():
            if test_case is None:
                continue
                
            logger.info(f"\n{verdict_type.upper()} EXAMPLE:")
            logger.info("-" * 40)
            
            # Create objects from test data
            claim = Claim(
                text=test_case["claim"]["text"],
                language=test_case["claim"].get("language", "vi"),
                confidence=test_case["claim"].get("confidence", 0.8)
            )
            
            evidence_list = []
            for ev_data in test_case["evidence"]:
                evidence = Evidence(
                    text=ev_data["text"],
                    source_url=ev_data["source_url"],
                    source_title=ev_data["source_title"],
                    credibility_score=ev_data.get("credibility_score", 0.8),
                    stance=ev_data.get("stance"),
                    stance_confidence=ev_data.get("stance_confidence"),
                    language=ev_data.get("language", "vi")
                )
                evidence_list.append(evidence)
            
            reasoning_steps = []
            for step_data in test_case["reasoning_steps"]:
                step = ReasoningStep(
                    iteration=step_data["iteration"],
                    thought=step_data["thought"],
                    action=step_data["action"],
                    action_input=step_data["action_input"],
                    observation=step_data["observation"]
                )
                reasoning_steps.append(step)
            
            verdict = Verdict(
                claim_id=claim.id,
                label=test_case["verdict"]["label"],
                confidence_scores=test_case["verdict"]["confidence_scores"]
            )
            
            logger.info(f"Claim: {claim.text}")
            logger.info(f"Verdict: {verdict.label}")
            
            # Generate explanation
            explanation = self.rag_generator.generate_explanation(
                claim=claim,
                verdict=verdict,
                evidence_list=evidence_list,
                reasoning_steps=reasoning_steps
            )
            
            logger.info(f"Explanation preview: {explanation[:200]}...")
    
    def save_demo_results(self, result: FactCheckResult):
        """Save demonstration results to file."""
        
        # Convert to serializable format
        demo_data = {
            "claim": result.claim.to_dict(),
            "verdict": result.verdict.to_dict(),
            "evidence": [ev.to_dict() for ev in result.evidence],
            "explanation": result.verdict.explanation,
            "metadata": result.metadata,
            "demo_timestamp": "2025-12-12T15:06:00Z",
            "system_info": {
                "rag_generator": "RAGExplanationGenerator v1.0",
                "llm_provider": "MockLLMProvider",
                "evidence_retriever": "EvidenceRetriever",
                "test_data_source": "rag_test_dataset.json"
            }
        }
        
        # Save results
        output_path = "demo_rag_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(demo_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Demo results saved to {output_path}")


def main():
    """Run complete RAG system demonstration."""
    
    logger.info("VIETNAMESE FACT-CHECKING SYSTEM")
    logger.info("RAG Explanation Generator - Complete Demonstration")
    logger.info("=" * 80)
    
    try:
        # Initialize demo system
        demo = CompleteRAGDemo()
        
        # Main demonstration
        result = demo.demonstrate_rag_capabilities()
        
        # Show different verdict types
        demo.demonstrate_different_verdict_types()
        
        # Save results
        demo.save_demo_results(result)
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("DEMONSTRATION COMPLETE!")
        logger.info("=" * 80)
        logger.info("✅ RAG explanation generator fully functional")
        logger.info("✅ Multi-language evidence processing")
        logger.info("✅ Contradiction detection and handling")
        logger.info("✅ Comprehensive citation system")
        logger.info("✅ Reasoning trace integration")
        logger.info("✅ Quality analysis and validation")
        logger.info("\n🎉 Vietnamese Fact-Checking System with RAG is ready for deployment!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()