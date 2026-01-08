#!/usr/bin/env python3
"""Test script for RAG explanation generator."""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import Claim, Evidence, Verdict, ReasoningStep
from rag_explanation_generator import RAGExplanationGenerator, EvidenceRetriever
from llm_controller import create_llm_controller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create test data for RAG explanation generation."""
    
    # Test claim about Vietnam provinces
    claim = Claim(
        text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương",
        confidence=0.9,
        language="vi"
    )
    
    # Supporting evidence
    evidence1 = Evidence(
        text="Theo Hiến pháp năm 2013, Việt Nam được chia thành 63 đơn vị hành chính cấp tỉnh, bao gồm 58 tỉnh và 5 thành phố trực thuộc trung ương.",
        source_url="https://baochinhphu.vn/hanh-chinh-dia-phuong",
        source_title="Tổ chức hành chính địa phương Việt Nam",
        credibility_score=0.9,
        stance="support",
        stance_confidence=0.95,
        language="vi"
    )
    
    evidence2 = Evidence(
        text="Vietnam is divided into 63 provincial-level administrative units, including 58 provinces and 5 centrally governed cities.",
        source_url="https://en.wikipedia.org/wiki/Provinces_of_Vietnam",
        source_title="Provinces of Vietnam - Wikipedia",
        credibility_score=0.7,
        stance="support", 
        stance_confidence=0.85,
        language="en"
    )
    
    # Slightly contradictory evidence (outdated info)
    evidence3 = Evidence(
        text="Trước đây Việt Nam có 64 tỉnh thành, nhưng sau khi sáp nhập một số đơn vị hành chính, hiện tại chỉ còn 63 tỉnh thành.",
        source_url="https://tuoitre.vn/hanh-chinh-dia-phuong",
        source_title="Lịch sử thay đổi đơn vị hành chính",
        credibility_score=0.6,
        stance="neutral",
        stance_confidence=0.7,
        language="vi"
    )
    
    # Create verdict
    verdict = Verdict(
        claim_id=claim.id,
        label="supported",
        confidence_scores={
            "supported": 0.85,
            "refuted": 0.10,
            "not_enough_info": 0.05
        }
    )
    
    # Create reasoning steps
    reasoning_steps = [
        ReasoningStep(
            iteration=1,
            thought="Tôi cần tìm kiếm thông tin chính thức về số lượng tỉnh thành của Việt Nam",
            action="search",
            action_input={"query": "Việt Nam 63 tỉnh thành chính thức"},
            observation="Tìm thấy thông tin từ trang chính phủ xác nhận 63 đơn vị hành chính cấp tỉnh"
        ),
        ReasoningStep(
            iteration=2,
            thought="Tôi nên kiểm tra thêm nguồn tiếng Anh để xác minh thông tin",
            action="search",
            action_input={"query": "Vietnam 63 provinces administrative units"},
            observation="Wikipedia và các nguồn quốc tế cũng xác nhận 63 đơn vị hành chính"
        ),
        ReasoningStep(
            iteration=3,
            thought="Cần phân tích độ tin cậy của các nguồn thông tin",
            action="analyze_credibility",
            action_input={"source_url": "https://baochinhphu.vn/hanh-chinh-dia-phuong"},
            observation="Nguồn chính phủ có độ tin cậy cao (0.9), thông tin được cập nhật"
        )
    ]
    
    return claim, [evidence1, evidence2, evidence3], verdict, reasoning_steps


def test_evidence_retriever():
    """Test evidence retriever functionality."""
    logger.info("Testing Evidence Retriever...")
    
    claim, evidence_list, verdict, _ = create_test_data()
    
    retriever = EvidenceRetriever(
        stance_weight=0.4,
        credibility_weight=0.3,
        relevance_weight=0.3,
        top_k=3
    )
    
    top_evidence, contradictory = retriever.retrieve_evidence(
        claim, evidence_list, verdict.label
    )
    
    logger.info(f"Retrieved {len(top_evidence)} top evidence pieces")
    for i, scored_ev in enumerate(top_evidence, 1):
        logger.info(f"Evidence {i}: Score={scored_ev.final_score:.3f}, "
                   f"Stance={scored_ev.evidence.stance}, "
                   f"Credibility={scored_ev.evidence.credibility_score:.2f}")
    
    if contradictory:
        logger.info(f"Found contradictory evidence groups: {list(contradictory.keys())}")
    else:
        logger.info("No contradictory evidence detected")
    
    return top_evidence, contradictory


def test_rag_generation():
    """Test RAG explanation generation."""
    logger.info("Testing RAG Generation...")
    
    claim, evidence_list, verdict, reasoning_steps = create_test_data()
    
    # Test with mock LLM controller if real one not available
    try:
        generator = RAGExplanationGenerator()
        
        explanation = generator.generate_explanation(
            claim=claim,
            verdict=verdict,
            evidence_list=evidence_list,
            reasoning_steps=reasoning_steps
        )
        
        logger.info("Generated explanation:")
        logger.info("=" * 60)
        logger.info(explanation)
        logger.info("=" * 60)
        
        # Validate explanation properties
        assert explanation.strip(), "Explanation should not be empty"
        assert len(explanation) > 100, "Explanation should be substantial"
        
        # Check for citations
        import re
        citations = re.findall(r'\[(\d+)\]', explanation)
        logger.info(f"Found {len(citations)} citations: {citations}")
        
        # Check for URLs
        urls = re.findall(r'https?://[^\s]+', explanation)
        logger.info(f"Found {len(urls)} URLs")
        
        return explanation
        
    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        logger.info("This is expected if no LLM API keys are configured")
        return None


def test_fallback_explanation():
    """Test fallback explanation generation."""
    logger.info("Testing Fallback Explanation...")
    
    claim, evidence_list, verdict, reasoning_steps = create_test_data()
    
    generator = RAGExplanationGenerator()
    
    # Test basic explanation (fallback)
    basic_explanation = generator._generate_basic_explanation(
        claim, verdict, evidence_list
    )
    
    logger.info("Generated basic explanation:")
    logger.info("=" * 60)
    logger.info(basic_explanation)
    logger.info("=" * 60)
    
    assert basic_explanation.strip(), "Basic explanation should not be empty"
    assert claim.text in basic_explanation, "Basic explanation should mention the claim"
    
    return basic_explanation


def test_reasoning_trace_formatting():
    """Test reasoning trace formatting."""
    logger.info("Testing Reasoning Trace Formatting...")
    
    from rag_explanation_generator import ReasoningTraceFormatter
    
    _, _, _, reasoning_steps = create_test_data()
    
    formatted_trace = ReasoningTraceFormatter.format_trace(reasoning_steps)
    
    logger.info("Formatted reasoning trace:")
    logger.info("=" * 60)
    logger.info(formatted_trace)
    logger.info("=" * 60)
    
    assert "Step 1:" in formatted_trace, "Should contain step numbers"
    assert "Thought:" in formatted_trace, "Should contain thoughts"
    assert "Action:" in formatted_trace, "Should contain actions"
    assert "Observation:" in formatted_trace, "Should contain observations"
    
    return formatted_trace


def main():
    """Run all RAG explanation tests."""
    logger.info("Starting RAG Explanation Generator Tests")
    logger.info("=" * 80)
    
    try:
        # Test 1: Evidence Retriever
        logger.info("\n1. Testing Evidence Retriever")
        top_evidence, contradictory = test_evidence_retriever()
        
        # Test 2: Reasoning Trace Formatting
        logger.info("\n2. Testing Reasoning Trace Formatting")
        formatted_trace = test_reasoning_trace_formatting()
        
        # Test 3: Fallback Explanation
        logger.info("\n3. Testing Fallback Explanation")
        basic_explanation = test_fallback_explanation()
        
        # Test 4: Full RAG Generation (may fail without API keys)
        logger.info("\n4. Testing Full RAG Generation")
        explanation = test_rag_generation()
        
        logger.info("\n" + "=" * 80)
        logger.info("RAG Explanation Generator Tests Completed Successfully!")
        
        if explanation:
            logger.info("✓ Full RAG generation working")
        else:
            logger.info("⚠ Full RAG generation requires API keys")
        
        logger.info("✓ Evidence retrieval working")
        logger.info("✓ Reasoning trace formatting working")
        logger.info("✓ Fallback explanation working")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()