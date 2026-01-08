#!/usr/bin/env python3
"""Test script for GNN verdict predictor inference."""

import os
import sys
import logging
import argparse
import json
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gnn_verdict_predictor import VerdictPredictor
from src.data_models import Claim, Evidence, KnowledgeGraph
from src.graph_builder import GraphBuilder
from src.logging_config import setup_logging

# Setup logging
logger = logging.getLogger(__name__)


def create_test_case() -> tuple:
    """Create a test case with claim, evidence, and reasoning graph.
    
    Returns:
        Tuple of (claim, evidence_list, expected_verdict)
    """
    # Test claim about Vietnam joining ASEAN
    claim = Claim(
        text="Việt Nam gia nhập ASEAN vào năm 1995",
        id="test_claim_001",
        language="vi",
        confidence=0.9
    )
    
    # Supporting evidence
    evidence_list = [
        Evidence(
            text="Việt Nam chính thức trở thành thành viên thứ 7 của ASEAN vào ngày 28 tháng 7 năm 1995 tại Manila, Philippines.",
            source_url="https://vnexpress.net/viet-nam-gia-nhap-asean-1995",
            source_title="Việt Nam gia nhập ASEAN năm 1995",
            credibility_score=0.9,
            language="vi",
            stance="support",
            stance_confidence=0.95,
            id="evidence_001"
        ),
        Evidence(
            text="ASEAN welcomed Vietnam as its seventh member on July 28, 1995, marking a significant milestone in regional cooperation.",
            source_url="https://asean.org/vietnam-membership-1995",
            source_title="Vietnam joins ASEAN in 1995",
            credibility_score=0.95,
            language="en",
            stance="support", 
            stance_confidence=0.9,
            id="evidence_002"
        ),
        Evidence(
            text="Trước khi gia nhập ASEAN, Việt Nam đã có nhiều năm chuẩn bị và đàm phán với các nước thành viên.",
            source_url="https://tuoitre.vn/asean-viet-nam-history",
            source_title="Lịch sử gia nhập ASEAN của Việt Nam",
            credibility_score=0.8,
            language="vi",
            stance="neutral",
            stance_confidence=0.7,
            id="evidence_003"
        )
    ]
    
    expected_verdict = "supported"
    
    return claim, evidence_list, expected_verdict


def create_refuted_test_case() -> tuple:
    """Create a test case that should be refuted.
    
    Returns:
        Tuple of (claim, evidence_list, expected_verdict)
    """
    claim = Claim(
        text="Hà Nội là thủ đô của Thái Lan",
        id="test_claim_002", 
        language="vi",
        confidence=0.8
    )
    
    evidence_list = [
        Evidence(
            text="Bangkok là thủ đô và thành phố lớn nhất của Thái Lan với dân số hơn 8 triệu người.",
            source_url="https://vnexpress.net/bangkok-thu-do-thai-lan",
            source_title="Bangkok - Thủ đô Thái Lan",
            credibility_score=0.9,
            language="vi",
            stance="refute",
            stance_confidence=0.95,
            id="evidence_004"
        ),
        Evidence(
            text="Hà Nội là thủ đô của Việt Nam, được thành lập từ năm 1010 dưới thời Lý Thái Tổ.",
            source_url="https://baochinhphu.vn/ha-noi-thu-do-viet-nam",
            source_title="Hà Nội - Thủ đô nghìn năm văn hiến",
            credibility_score=0.95,
            language="vi",
            stance="refute",
            stance_confidence=0.9,
            id="evidence_005"
        )
    ]
    
    expected_verdict = "refuted"
    
    return claim, evidence_list, expected_verdict


def create_nei_test_case() -> tuple:
    """Create a test case with not enough information.
    
    Returns:
        Tuple of (claim, evidence_list, expected_verdict)
    """
    claim = Claim(
        text="Dân số Việt Nam sẽ đạt 100 triệu người vào năm 2030",
        id="test_claim_003",
        language="vi", 
        confidence=0.7
    )
    
    evidence_list = [
        Evidence(
            text="Theo Tổng cục Thống kê, dân số Việt Nam hiện tại là khoảng 97.3 triệu người.",
            source_url="https://gso.gov.vn/dan-so-viet-nam-2023",
            source_title="Dân số Việt Nam năm 2023",
            credibility_score=0.9,
            language="vi",
            stance="neutral",
            stance_confidence=0.6,
            id="evidence_006"
        ),
        Evidence(
            text="Tốc độ tăng dân số của Việt Nam đang có xu hướng chậm lại trong những năm gần đây.",
            source_url="https://tuoitre.vn/tang-dan-so-cham-lai",
            source_title="Tăng dân số chậm lại",
            credibility_score=0.8,
            language="vi",
            stance="neutral",
            stance_confidence=0.5,
            id="evidence_007"
        )
    ]
    
    expected_verdict = "not_enough_info"
    
    return claim, evidence_list, expected_verdict


def test_inference(predictor: VerdictPredictor, claim: Claim, evidence_list: List[Evidence], expected: str) -> Dict[str, Any]:
    """Test inference on a single case.
    
    Args:
        predictor: Trained verdict predictor
        claim: Claim to verify
        evidence_list: List of evidence
        expected: Expected verdict
        
    Returns:
        Test result dictionary
    """
    logger.info(f"Testing claim: {claim.text}")
    logger.info(f"Expected verdict: {expected}")
    
    try:
        # Build reasoning graph
        builder = GraphBuilder()
        graph = builder.build_graph(evidence_list)
        
        # Add claim node
        claim_node_id = builder.add_claim_node(graph, claim.text, claim.id)
        
        # Handle contradictions
        contradictions = builder.handle_contradictions(graph, claim_node_id)
        
        logger.info(f"Built graph with {graph.node_count()} nodes and {graph.edge_count()} edges")
        logger.info(f"Supporting evidence: {len(contradictions['supporting'])}")
        logger.info(f"Refuting evidence: {len(contradictions['refuting'])}")
        
        # Predict verdict
        verdict = predictor.predict_verdict(graph, claim)
        
        # Log results
        logger.info(f"Predicted verdict: {verdict.label}")
        logger.info(f"Confidence scores: {verdict.confidence_scores}")
        logger.info(f"Quality score: {verdict.quality_score:.3f}")
        
        # Check if prediction matches expected
        correct = verdict.label == expected
        logger.info(f"Prediction correct: {correct}")
        
        return {
            "claim_text": claim.text,
            "expected": expected,
            "predicted": verdict.label,
            "correct": correct,
            "confidence_scores": verdict.confidence_scores,
            "quality_score": verdict.quality_score,
            "graph_stats": {
                "nodes": graph.node_count(),
                "edges": graph.edge_count(),
                "supporting_evidence": len(contradictions['supporting']),
                "refuting_evidence": len(contradictions['refuting'])
            }
        }
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {
            "claim_text": claim.text,
            "expected": expected,
            "predicted": "error",
            "correct": False,
            "error": str(e)
        }


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test GNN verdict predictor inference")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting GNN inference testing")
    
    # Initialize predictor
    predictor = VerdictPredictor(model_path=args.model, device=args.device)
    
    # Create test cases
    test_cases = [
        create_test_case(),
        create_refuted_test_case(), 
        create_nei_test_case()
    ]
    
    # Run tests
    results = []
    correct_predictions = 0
    
    for i, (claim, evidence_list, expected) in enumerate(test_cases, 1):
        logger.info(f"\n--- Test Case {i} ---")
        result = test_inference(predictor, claim, evidence_list, expected)
        results.append(result)
        
        if result.get("correct", False):
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / len(test_cases)
    logger.info(f"\nOverall accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_cases)})")
    
    # Save results
    if args.output:
        output_data = {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_cases": len(test_cases),
            "results": results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved results to {args.output}")
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    for i, result in enumerate(results, 1):
        status = "✓" if result.get("correct", False) else "✗"
        predicted = result.get("predicted", "error")
        expected = result.get("expected", "unknown")
        print(f"Test {i}: {status} Predicted: {predicted}, Expected: {expected}")


if __name__ == "__main__":
    main()