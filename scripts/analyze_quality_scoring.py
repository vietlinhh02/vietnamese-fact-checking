#!/usr/bin/env python3
"""Analyze quality scoring algorithm."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_verification import QualityScorer
from src.data_models import Claim, Evidence
from dataclasses import dataclass
from typing import List

@dataclass
class MockVerificationResult:
    """Mock verification result for testing."""
    claim: Claim
    is_verified: bool
    supporting_evidence: List[Evidence]
    confidence: float
    verification_method: str
    explanation: str

def analyze_scoring_algorithm():
    """Analyze the quality scoring algorithm step by step."""
    
    print("=" * 70)
    print("ANALYZING QUALITY SCORING ALGORITHM")
    print("=" * 70)
    
    # Create mock results matching the debug case
    mock_results = [
        MockVerificationResult(
            claim=Claim(text="Claim 1", context="", confidence=0.8, sentence_type="factual_claim", start_idx=0, end_idx=10, language="vi"),
            is_verified=True,
            supporting_evidence=[Evidence(text="Evidence 1", source_url="url1", source_title="title1", credibility_score=0.9, language="vi")],
            confidence=1.0,
            verification_method="evidence_match",
            explanation="Verified"
        ),
        MockVerificationResult(
            claim=Claim(text="Claim 2", context="", confidence=0.8, sentence_type="factual_claim", start_idx=0, end_idx=10, language="vi"),
            is_verified=True,
            supporting_evidence=[Evidence(text="Evidence 2", source_url="url2", source_title="title2", credibility_score=0.9, language="vi")],
            confidence=0.987,
            verification_method="evidence_match",
            explanation="Verified"
        ),
        MockVerificationResult(
            claim=Claim(text="Claim 3", context="", confidence=0.8, sentence_type="factual_claim", start_idx=0, end_idx=10, language="vi"),
            is_verified=False,
            supporting_evidence=[],
            confidence=0.0,
            verification_method="evidence_match",
            explanation="Not verified"
        ),
        MockVerificationResult(
            claim=Claim(text="Claim 4", context="", confidence=0.8, sentence_type="factual_claim", start_idx=0, end_idx=10, language="vi"),
            is_verified=False,
            supporting_evidence=[],
            confidence=0.0,
            verification_method="evidence_match",
            explanation="Not verified"
        )
    ]
    
    # Initialize scorer with default settings
    scorer = QualityScorer()
    
    print(f"Scorer settings:")
    print(f"  min_verification_rate: {scorer.min_verification_rate}")
    print(f"  confidence_weight: {scorer.confidence_weight}")
    print(f"  method_weights: {scorer.method_weights}")
    
    # Manual calculation
    print(f"\n--- MANUAL CALCULATION ---")
    
    # Step 1: Calculate weighted sum
    total_weight = 0.0
    weighted_sum = 0.0
    
    for i, result in enumerate(mock_results, 1):
        method_weight = scorer.method_weights.get(result.verification_method, 0.5)
        
        if result.is_verified:
            claim_score = method_weight * result.confidence
        else:
            claim_score = 0.0
        
        weighted_sum += claim_score
        total_weight += method_weight
        
        print(f"Claim {i}: verified={result.is_verified}, confidence={result.confidence:.3f}")
        print(f"  method_weight={method_weight}, claim_score={claim_score:.3f}")
    
    print(f"\nWeighted sum: {weighted_sum:.6f}")
    print(f"Total weight: {total_weight:.6f}")
    
    # Step 2: Base score
    base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    print(f"Base score: {base_score:.6f}")
    
    # Step 3: Confidence adjustment
    avg_confidence = sum(r.confidence for r in mock_results) / len(mock_results)
    confidence_adjustment = avg_confidence * scorer.confidence_weight
    print(f"Average confidence: {avg_confidence:.6f}")
    print(f"Confidence adjustment: {confidence_adjustment:.6f}")
    
    # Step 4: Final score
    final_score = base_score * (1 - scorer.confidence_weight) + confidence_adjustment
    print(f"Final score: {final_score:.6f}")
    
    # Compare with actual scorer
    quality_score = scorer.compute_quality_score(mock_results)
    print(f"Scorer result: {quality_score.overall_score:.6f}")
    
    # Test different confidence weights
    print(f"\n--- TESTING DIFFERENT CONFIDENCE WEIGHTS ---")
    
    for conf_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        test_scorer = QualityScorer(confidence_weight=conf_weight)
        test_quality = test_scorer.compute_quality_score(mock_results)
        
        # Determine quality level
        if test_quality.overall_score >= 0.8:
            level = "HIGH"
        elif test_quality.overall_score >= 0.5:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        print(f"Confidence weight {conf_weight:.1f}: score={test_quality.overall_score:.3f}, level={level}")
    
    # Test simpler scoring
    print(f"\n--- SIMPLE VERIFICATION RATE ---")
    simple_rate = sum(1 for r in mock_results if r.is_verified) / len(mock_results)
    print(f"Simple verification rate: {simple_rate:.3f}")
    
    if simple_rate >= 0.8:
        simple_level = "HIGH"
    elif simple_rate >= 0.5:
        simple_level = "MEDIUM"
    else:
        simple_level = "LOW"
    
    print(f"Simple level: {simple_level}")

if __name__ == "__main__":
    analyze_scoring_algorithm()