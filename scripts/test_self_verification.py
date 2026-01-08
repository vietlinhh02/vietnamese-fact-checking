#!/usr/bin/env python3
"""Test script for self-verification module."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.self_verification import SelfVerificationModule
from src.data_models import Evidence

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_self_verification():
    """Test the self-verification module with sample data."""
    
    # Sample explanation with potential hallucinations
    explanation = """
    Tuyên bố về dân số Việt Nam được hỗ trợ bởi bằng chứng từ Tổng cục Thống kê. 
    Theo số liệu chính thức năm 2023, Việt Nam có dân số khoảng 98 triệu người.
    
    Ngoài ra, GDP của Việt Nam năm 2023 đạt mức 430 tỷ USD, tăng 5.05% so với năm trước.
    Đây là mức tăng trưởng ấn tượng trong bối cảnh kinh tế thế giới khó khăn.
    
    Việt Nam hiện có 65 tỉnh thành phố trực thuộc trung ương, bao gồm 58 tỉnh và 5 thành phố trực thuộc trung ương.
    Thủ đô Hà Nội là trung tâm chính trị và văn hóa của đất nước.
    
    Theo một số nguồn tin, tỷ lệ biết chữ của Việt Nam đạt 99.8% vào năm 2023.
    """
    
    # Sample evidence (some supporting, some missing)
    evidence_list = [
        Evidence(
            text="Dân số Việt Nam năm 2023 ước tính khoảng 98,5 triệu người theo Tổng cục Thống kê",
            source_url="https://gso.gov.vn/population-2023",
            source_title="Thống kê dân số Việt Nam 2023",
            credibility_score=0.95,
            language="vi"
        ),
        Evidence(
            text="GDP Việt Nam năm 2023 đạt 430,1 tỷ USD, tăng trưởng 5,05% so với năm 2022",
            source_url="https://gso.gov.vn/gdp-2023",
            source_title="Báo cáo GDP Việt Nam 2023",
            credibility_score=0.92,
            language="vi"
        ),
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương, gồm 58 tỉnh và 5 thành phố trực thuộc trung ương",
            source_url="https://chinhphu.vn/provinces-list",
            source_title="Danh sách tỉnh thành Việt Nam",
            credibility_score=0.98,
            language="vi"
        ),
        Evidence(
            text="Hà Nội là thủ đô và trung tâm chính trị của Việt Nam",
            source_url="https://hanoi.gov.vn/about",
            source_title="Giới thiệu về Hà Nội",
            credibility_score=0.99,
            language="vi"
        )
    ]
    
    print("=" * 60)
    print("TESTING SELF-VERIFICATION MODULE")
    print("=" * 60)
    
    # Initialize self-verification module
    verifier = SelfVerificationModule()
    
    print("\n1. ORIGINAL EXPLANATION:")
    print("-" * 40)
    print(explanation)
    
    print(f"\n2. AVAILABLE EVIDENCE ({len(evidence_list)} pieces):")
    print("-" * 40)
    for i, ev in enumerate(evidence_list, 1):
        print(f"{i}. {ev.text[:80]}...")
        print(f"   Source: {ev.source_title} (credibility: {ev.credibility_score:.2f})")
    
    # Run self-verification
    print("\n3. RUNNING SELF-VERIFICATION...")
    print("-" * 40)
    
    quality_score, verification_results = verifier.verify_explanation(explanation, evidence_list)
    
    print(f"\n4. VERIFICATION RESULTS:")
    print("-" * 40)
    print(f"Overall Quality Score: {quality_score.overall_score:.2f}/1.00")
    print(f"Verification Rate: {quality_score.verification_rate:.1%}")
    print(f"Claims Verified: {quality_score.verified_claims}/{quality_score.total_claims}")
    
    if quality_score.flagged_claims:
        print(f"Flagged Claims: {len(quality_score.flagged_claims)}")
    
    print(f"\n5. DETAILED CLAIM ANALYSIS:")
    print("-" * 40)
    
    for i, result in enumerate(verification_results, 1):
        status = "✓ VERIFIED" if result.is_verified else "✗ UNVERIFIED"
        print(f"\n{i}. {result.claim.text}")
        print(f"   Status: {status}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Method: {result.verification_method}")
        print(f"   Explanation: {result.explanation}")
        
        if result.supporting_evidence:
            print(f"   Supporting Evidence: {len(result.supporting_evidence)} pieces")
    
    print(f"\n6. QUALITY ASSESSMENT:")
    print("-" * 40)
    print(quality_score.explanation)
    
    # Test correction strategies
    print(f"\n7. HALLUCINATION CORRECTION:")
    print("-" * 40)
    
    strategies = ["flag", "revise", "remove", "adaptive"]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.upper()}")
        corrected = verifier.correct_hallucinations(
            explanation, verification_results, quality_score, strategy
        )
        print(f"Length change: {len(explanation)} -> {len(corrected)} chars")
        
        if strategy == "flag":
            # Show flagged version
            print("Sample (first 200 chars):")
            print(corrected[:200] + "..." if len(corrected) > 200 else corrected)
    
    # Test regeneration recommendation
    print(f"\n8. REGENERATION RECOMMENDATION:")
    print("-" * 40)
    
    should_regenerate = verifier.should_regenerate_explanation(quality_score)
    print(f"Should regenerate explanation: {should_regenerate}")
    print(f"Threshold: Quality score < 0.3")
    
    # Generate correction summary
    print(f"\n9. CORRECTION SUMMARY:")
    print("-" * 40)
    summary = verifier.get_correction_summary(verification_results, quality_score)
    print(summary)
    
    print("\n" + "=" * 60)
    print("SELF-VERIFICATION TEST COMPLETE")
    print("=" * 60)
    
    return quality_score, verification_results


def test_edge_cases():
    """Test edge cases for self-verification."""
    
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    verifier = SelfVerificationModule()
    
    # Test 1: Empty explanation
    print("\n1. Empty explanation:")
    quality_score, results = verifier.verify_explanation("", [])
    print(f"   Quality score: {quality_score.overall_score:.2f}")
    print(f"   Claims found: {quality_score.total_claims}")
    
    # Test 2: No factual claims
    print("\n2. Opinion-only text:")
    opinion_text = "Tôi nghĩ rằng Việt Nam rất đẹp. Bạn có đồng ý không? Hãy đến thăm Việt Nam."
    quality_score, results = verifier.verify_explanation(opinion_text, [])
    print(f"   Quality score: {quality_score.overall_score:.2f}")
    print(f"   Claims found: {quality_score.total_claims}")
    
    # Test 3: All claims verified
    print("\n3. All claims verified:")
    verified_text = "Việt Nam có 63 tỉnh thành."
    verified_evidence = [
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương",
            source_url="https://example.com",
            source_title="Test",
            credibility_score=0.9,
            language="vi"
        )
    ]
    quality_score, results = verifier.verify_explanation(verified_text, verified_evidence)
    print(f"   Quality score: {quality_score.overall_score:.2f}")
    print(f"   Verification rate: {quality_score.verification_rate:.1%}")
    
    # Test 4: No evidence available
    print("\n4. No evidence available:")
    claim_text = "Việt Nam có 100 triệu dân vào năm 2025."
    quality_score, results = verifier.verify_explanation(claim_text, [])
    print(f"   Quality score: {quality_score.overall_score:.2f}")
    print(f"   Flagged claims: {len(quality_score.flagged_claims)}")


if __name__ == "__main__":
    try:
        # Run main test
        quality_score, verification_results = test_self_verification()
        
        # Run edge case tests
        test_edge_cases()
        
        print(f"\n✓ All tests completed successfully!")
        print(f"Final quality score: {quality_score.overall_score:.2f}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)