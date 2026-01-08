#!/usr/bin/env python3
"""Test script for RAG explanation generator with self-verification."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.rag_explanation_generator import RAGExplanationGenerator
from src.data_models import Claim, Evidence, Verdict, ReasoningStep
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_rag_with_self_verification():
    """Test RAG explanation generator with self-verification enabled."""
    
    print("=" * 70)
    print("TESTING RAG EXPLANATION GENERATOR WITH SELF-VERIFICATION")
    print("=" * 70)
    
    # Create test data
    claim = Claim(
        text="Việt Nam có 63 tỉnh thành phố và GDP đạt 430 tỷ USD năm 2023",
        context="Thông tin về Việt Nam",
        confidence=0.9,
        sentence_type="factual_claim",
        start_idx=0,
        end_idx=100,
        language="vi"
    )
    
    # Evidence list (some supporting, some missing)
    evidence_list = [
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương theo quy định hiện hành",
            source_url="https://chinhphu.vn/provinces",
            source_title="Danh sách tỉnh thành Việt Nam",
            source_author="Chính phủ Việt Nam",
            publish_date=datetime(2023, 1, 1),
            credibility_score=0.95,
            language="vi",
            stance="support",
            stance_confidence=0.9
        ),
        Evidence(
            text="GDP của Việt Nam năm 2023 đạt 430,1 tỷ USD theo báo cáo của Tổng cục Thống kê",
            source_url="https://gso.gov.vn/gdp-2023",
            source_title="Báo cáo GDP Việt Nam 2023",
            source_author="Tổng cục Thống kê",
            publish_date=datetime(2023, 12, 31),
            credibility_score=0.92,
            language="vi",
            stance="support",
            stance_confidence=0.85
        ),
        Evidence(
            text="Việt Nam là một quốc gia Đông Nam Á với nền kinh tế đang phát triển",
            source_url="https://worldbank.org/vietnam",
            source_title="Vietnam Economic Overview",
            credibility_score=0.8,
            language="en",
            stance="neutral",
            stance_confidence=0.7
        )
    ]
    
    # Verdict
    verdict = Verdict(
        claim_id=claim.id,
        label="supported",
        confidence_scores={
            "supported": 0.85,
            "refuted": 0.10,
            "not_enough_info": 0.05
        },
        supporting_evidence=[ev.id for ev in evidence_list[:2]],
        refuting_evidence=[],
        explanation="",
        reasoning_trace=[],
        quality_score=0.0
    )
    
    # Reasoning steps
    reasoning_steps = [
        ReasoningStep(
            iteration=1,
            thought="I need to verify information about Vietnam's provinces and GDP",
            action="search",
            action_input={"query": "Vietnam provinces GDP 2023"},
            observation="Found information about 63 provinces and GDP data",
            timestamp=datetime.now()
        ),
        ReasoningStep(
            iteration=2,
            thought="Let me search for more specific GDP information",
            action="search",
            action_input={"query": "Vietnam GDP 430 billion USD 2023"},
            observation="Confirmed GDP figure from official statistics",
            timestamp=datetime.now()
        )
    ]
    
    print(f"\n1. TEST CLAIM:")
    print(f"   {claim.text}")
    
    print(f"\n2. AVAILABLE EVIDENCE ({len(evidence_list)} pieces):")
    for i, ev in enumerate(evidence_list, 1):
        print(f"   {i}. {ev.text[:60]}...")
        print(f"      Source: {ev.source_title} (credibility: {ev.credibility_score:.2f})")
    
    print(f"\n3. VERDICT: {verdict.label.upper()}")
    print(f"   Confidence: {verdict.confidence_scores}")
    
    # Test with self-verification enabled
    print(f"\n4. GENERATING EXPLANATION WITH SELF-VERIFICATION...")
    print("-" * 50)
    
    generator = RAGExplanationGenerator(enable_self_verification=True)
    
    explanation, verification_metadata = generator.generate_explanation(
        claim=claim,
        verdict=verdict,
        evidence_list=evidence_list,
        reasoning_steps=reasoning_steps
    )
    
    print(f"\n5. GENERATED EXPLANATION:")
    print("-" * 50)
    print(explanation)
    
    print(f"\n6. SELF-VERIFICATION RESULTS:")
    print("-" * 50)
    
    if verification_metadata:
        if "error" in verification_metadata:
            print(f"   ✗ Verification failed: {verification_metadata['error']}")
        else:
            # Handle both structured and legacy formats
            if 'quality_assessment' in verification_metadata:
                # Structured format
                quality_assessment = verification_metadata['quality_assessment']
                print(f"   Quality Score: {quality_assessment['overall_score']:.2f}/1.00")
                print(f"   Verification Rate: {quality_assessment['verification_rate']:.1%}")
                print(f"   Claims Verified: {quality_assessment['verified_claims']}/{quality_assessment['total_claims']}")
                print(f"   Flagged Claims: {quality_assessment['flagged_claims']}")
                print(f"   Quality Level: {quality_assessment['quality_level']}")
                print(f"   Correction Applied: {verification_metadata['correction_applied']}")
                
                if 'confidence_scores' in quality_assessment and quality_assessment['confidence_scores']:
                    print(f"   Verification Methods:")
                    for method, confidence in quality_assessment['confidence_scores'].items():
                        print(f"     - {method}: {confidence:.2f}")
            else:
                # Legacy format
                print(f"   Quality Score: {verification_metadata['quality_score']:.2f}/1.00")
                print(f"   Verification Rate: {verification_metadata['verification_rate']:.1%}")
                print(f"   Claims Verified: {verification_metadata['verified_claims']}/{verification_metadata['total_claims']}")
                print(f"   Flagged Claims: {verification_metadata['flagged_claims']}")
                print(f"   Correction Applied: {verification_metadata['correction_applied']}")
                
                if 'confidence_scores' in verification_metadata and verification_metadata['confidence_scores']:
                    print(f"   Verification Methods:")
                    for method, confidence in verification_metadata['confidence_scores'].items():
                        print(f"     - {method}: {confidence:.2f}")
    else:
        print("   No verification metadata available")
    
    # Test with self-verification disabled
    print(f"\n7. COMPARISON: EXPLANATION WITHOUT SELF-VERIFICATION...")
    print("-" * 50)
    
    generator_no_verify = RAGExplanationGenerator(enable_self_verification=False)
    
    explanation_no_verify, metadata_no_verify = generator_no_verify.generate_explanation(
        claim=claim,
        verdict=verdict,
        evidence_list=evidence_list,
        reasoning_steps=reasoning_steps
    )
    
    print(f"Length with verification: {len(explanation)} chars")
    print(f"Length without verification: {len(explanation_no_verify)} chars")
    print(f"Difference: {len(explanation) - len(explanation_no_verify)} chars")
    
    # Show first 300 chars of non-verified version
    print(f"\nFirst 300 chars (no verification):")
    print(explanation_no_verify[:300] + "..." if len(explanation_no_verify) > 300 else explanation_no_verify)
    
    return explanation, verification_metadata


def test_hallucination_detection():
    """Test hallucination detection with fabricated claims."""
    
    print(f"\n" + "=" * 70)
    print("TESTING HALLUCINATION DETECTION")
    print("=" * 70)
    
    # Create claim with some fabricated information
    claim = Claim(
        text="Việt Nam có 70 tỉnh thành và dân số 120 triệu người",
        context="Thông tin về Việt Nam",
        confidence=0.8,
        sentence_type="factual_claim",
        start_idx=0,
        end_idx=100,
        language="vi"
    )
    
    # Limited evidence that doesn't support the fabricated claims
    evidence_list = [
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương",
            source_url="https://chinhphu.vn/provinces",
            source_title="Danh sách tỉnh thành",
            credibility_score=0.95,
            language="vi"
        ),
        Evidence(
            text="Dân số Việt Nam khoảng 98 triệu người năm 2023",
            source_url="https://gso.gov.vn/population",
            source_title="Thống kê dân số",
            credibility_score=0.9,
            language="vi"
        )
    ]
    
    verdict = Verdict(
        claim_id=claim.id,
        label="refuted",
        confidence_scores={"supported": 0.2, "refuted": 0.7, "not_enough_info": 0.1},
        supporting_evidence=[],
        refuting_evidence=[ev.id for ev in evidence_list],
        explanation="",
        reasoning_trace=[],
        quality_score=0.0
    )
    
    print(f"\n1. FABRICATED CLAIM:")
    print(f"   {claim.text}")
    
    print(f"\n2. CONTRADICTING EVIDENCE:")
    for i, ev in enumerate(evidence_list, 1):
        print(f"   {i}. {ev.text}")
    
    generator = RAGExplanationGenerator(enable_self_verification=True)
    
    explanation, verification_metadata = generator.generate_explanation(
        claim=claim,
        verdict=verdict,
        evidence_list=evidence_list,
        reasoning_steps=[]
    )
    
    print(f"\n3. EXPLANATION WITH HALLUCINATION DETECTION:")
    print("-" * 50)
    print(explanation)
    
    print(f"\n4. HALLUCINATION DETECTION RESULTS:")
    print("-" * 50)
    
    if verification_metadata and "error" not in verification_metadata:
        # Handle both structured and legacy formats
        if 'quality_assessment' in verification_metadata:
            # Structured format
            quality_assessment = verification_metadata['quality_assessment']
            quality_score = quality_assessment['overall_score']
            verification_rate = quality_assessment['verification_rate']
            flagged_claims = quality_assessment['flagged_claims']
        else:
            # Legacy format
            quality_score = verification_metadata['quality_score']
            verification_rate = verification_metadata['verification_rate']
            flagged_claims = verification_metadata['flagged_claims']
        
        print(f"   Quality Score: {quality_score:.2f}/1.00")
        print(f"   Verification Rate: {verification_rate:.1%}")
        print(f"   Flagged Claims: {flagged_claims}")
        
        if quality_score < 0.5:
            print(f"   ⚠ LOW QUALITY: Potential hallucinations detected")
        elif quality_score < 0.8:
            print(f"   ⚠ MEDIUM QUALITY: Some claims need verification")
        else:
            print(f"   ✓ HIGH QUALITY: Most claims verified")
    
    return explanation, verification_metadata


if __name__ == "__main__":
    try:
        # Test 1: Normal RAG with self-verification
        explanation1, metadata1 = test_rag_with_self_verification()
        
        # Test 2: Hallucination detection
        explanation2, metadata2 = test_hallucination_detection()
        
        print(f"\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"Test 1 - Quality Score: {metadata1.get('quality_score', 'N/A') if metadata1 else 'N/A'}")
        print(f"Test 2 - Quality Score: {metadata2.get('quality_score', 'N/A') if metadata2 else 'N/A'}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)