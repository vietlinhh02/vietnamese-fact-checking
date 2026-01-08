#!/usr/bin/env python3
"""Debug the medium quality test case."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.self_verification import SelfVerificationModule, SelfVerificationOutputFormatter
from src.data_models import Evidence

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def debug_medium_quality():
    """Debug the medium quality test case that failed."""
    
    print("=" * 60)
    print("DEBUGGING MEDIUM QUALITY TEST CASE")
    print("=" * 60)
    
    # Exact same test case that failed
    explanation = """
    Việt Nam có 63 tỉnh thành phố trực thuộc trung ương theo quy định hiện hành.
    GDP của Việt Nam năm 2023 đạt 430,1 tỷ USD theo báo cáo chính thức.
    Tỷ lệ biết chữ của Việt Nam đạt 99,9% theo thống kê mới nhất.
    Việt Nam có 100 sân bay quốc tế trên toàn quốc.
    """.strip()
    
    evidence_list = [
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương theo quy định hiện hành",
            source_url="https://chinhphu.vn/provinces",
            source_title="Danh sách tỉnh thành Việt Nam",
            credibility_score=0.95,
            language="vi"
        ),
        Evidence(
            text="GDP của Việt Nam năm 2023 đạt 430,1 tỷ USD theo báo cáo của Tổng cục Thống kê",
            source_url="https://gso.gov.vn/gdp-2023",
            source_title="Báo cáo GDP Việt Nam 2023",
            credibility_score=0.92,
            language="vi"
        ),
        Evidence(
            text="Việt Nam là một quốc gia Đông Nam Á với nền kinh tế đang phát triển",
            source_url="https://worldbank.org/vietnam",
            source_title="Vietnam Economic Overview",
            credibility_score=0.80,
            language="vi"
        )
    ]
    
    print(f"Explanation:")
    print(f'"{explanation}"')
    print(f"\nEvidence pieces: {len(evidence_list)}")
    
    # Run verification
    verifier = SelfVerificationModule()
    quality_score, verification_results = verifier.verify_explanation(explanation, evidence_list)
    
    print(f"\n--- DETAILED ANALYSIS ---")
    print(f"Overall Score: {quality_score.overall_score}")
    print(f"Verification Rate: {quality_score.verification_rate}")
    print(f"Verified Claims: {quality_score.verified_claims}")
    print(f"Total Claims: {quality_score.total_claims}")
    print(f"Flagged Claims: {len(quality_score.flagged_claims)}")
    
    # Manual threshold check
    if quality_score.overall_score >= 0.8:
        expected_level = "HIGH"
    elif quality_score.overall_score >= 0.5:
        expected_level = "MEDIUM"
    else:
        expected_level = "LOW"
    
    print(f"Manual threshold check: {expected_level}")
    
    # Create structured output
    formatter = SelfVerificationOutputFormatter()
    structured_output = formatter.to_structured_output(
        quality_score=quality_score,
        verification_results=verification_results,
        correction_applied=False,
        correction_strategy="none",
        original_length=len(explanation),
        corrected_length=len(explanation)
    )
    
    actual_level = structured_output['quality_assessment']['quality_level']
    print(f"Structured output level: {actual_level}")
    
    # Check each claim
    print(f"\n--- CLAIM ANALYSIS ---")
    for i, result in enumerate(verification_results, 1):
        status = "✓ VERIFIED" if result.is_verified else "✗ UNVERIFIED"
        print(f"{i}. {status}")
        print(f"   Claim: {result.claim.text[:60]}...")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Method: {result.verification_method}")
        print(f"   Evidence count: {len(result.supporting_evidence)}")
    
    # Check if there's a precision issue
    print(f"\n--- PRECISION CHECK ---")
    print(f"Exact score: {quality_score.overall_score}")
    print(f"Score >= 0.5: {quality_score.overall_score >= 0.5}")
    print(f"Score >= 0.8: {quality_score.overall_score >= 0.8}")
    
    return quality_score, verification_results, structured_output

if __name__ == "__main__":
    debug_medium_quality()