#!/usr/bin/env python3
"""Test quality threshold logic."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_verification import SelfVerificationOutputFormatter
from src.data_models import Evidence, Claim
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class MockQualityScore:
    """Mock quality score for testing."""
    overall_score: float
    verified_claims: int
    total_claims: int
    verification_rate: float
    confidence_scores: Dict[str, float]
    flagged_claims: List[Any]
    explanation: str

def test_quality_thresholds():
    """Test quality level determination for different scores."""
    
    print("=" * 60)
    print("TESTING QUALITY THRESHOLD LOGIC")
    print("=" * 60)
    
    test_scores = [
        (1.0, "HIGH"),
        (0.9, "HIGH"), 
        (0.8, "HIGH"),
        (0.79, "MEDIUM"),
        (0.7, "MEDIUM"),
        (0.6, "MEDIUM"),
        (0.5, "MEDIUM"),
        (0.49, "LOW"),
        (0.3, "LOW"),
        (0.0, "LOW")
    ]
    
    formatter = SelfVerificationOutputFormatter()
    
    for score, expected_level in test_scores:
        # Create mock quality score
        mock_quality = MockQualityScore(
            overall_score=score,
            verified_claims=1,
            total_claims=1,
            verification_rate=1.0,
            confidence_scores={"evidence_match": score},
            flagged_claims=[],
            explanation="Test explanation"
        )
        
        # Create structured output
        structured = formatter.to_structured_output(
            quality_score=mock_quality,
            verification_results=[],
            correction_applied=False,
            correction_strategy="none",
            original_length=100,
            corrected_length=100
        )
        
        actual_level = structured['quality_assessment']['quality_level']
        
        # Check if correct
        is_correct = actual_level == expected_level
        status = "✅" if is_correct else "❌"
        
        print(f"{status} Score: {score:.2f} -> Expected: {expected_level}, Got: {actual_level}")
        
        if not is_correct:
            print(f"   ERROR: Threshold logic issue!")

if __name__ == "__main__":
    test_quality_thresholds()