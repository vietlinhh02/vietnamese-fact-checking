#!/usr/bin/env python3
"""Demo script showcasing self-verification output functions with Gemini API."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from src.self_verification import (
    SelfVerificationModule, 
    SelfVerificationOutputFormatter,
    print_verification_results
)
from src.rag_explanation_generator import RAGExplanationGenerator
from src.data_models import Claim, Evidence, Verdict, ReasoningStep
from src.config import SystemConfig
from src.llm_controller import create_llm_controller
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_output_functions():
    """Demonstrate all self-verification output functions."""
    
    print("=" * 80)
    print("SELF-VERIFICATION OUTPUT FUNCTIONS DEMO")
    print("=" * 80)
    
    # Create test data with mixed quality
    explanation = """
    Bằng chứng chính: [1] Dân số Việt Nam khoảng 98 triệu người năm 2023... (Nguồn: Thống kê dân số)
    [2] Việt Nam có 63 tỉnh thành phố trực thuộc trung ương... (Nguồn: Chính phủ)
    [3] GDP Việt Nam đạt 430 tỷ USD năm 2023... (Nguồn: Tổng cục Thống kê)
    
    Tuy nhiên, có một số thông tin chưa được xác minh:
    [4] Tỷ lệ biết chữ đạt 99.9% năm 2023... (Nguồn: Không rõ)
    [5] Việt Nam có 100 sân bay quốc tế... (Nguồn: Không xác thực)
    """
    
    # Evidence list (some supporting, some missing)
    evidence_list = [
        Evidence(
            text="Dân số Việt Nam năm 2023 ước tính khoảng 98,2 triệu người theo Tổng cục Thống kê",
            source_url="https://gso.gov.vn/population-2023",
            source_title="Thống kê dân số Việt Nam 2023",
            credibility_score=0.95,
            language="vi"
        ),
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương theo quy định hiện hành",
            source_url="https://chinhphu.vn/provinces",
            source_title="Danh sách tỉnh thành Việt Nam",
            credibility_score=0.98,
            language="vi"
        ),
        Evidence(
            text="GDP Việt Nam năm 2023 đạt 430,1 tỷ USD, tăng trưởng 5,05% so với năm 2022",
            source_url="https://gso.gov.vn/gdp-2023",
            source_title="Báo cáo GDP Việt Nam 2023",
            credibility_score=0.92,
            language="vi"
        )
        # Note: No evidence for literacy rate or airports - should be flagged
    ]
    
    print(f"\n1. SAMPLE EXPLANATION TO VERIFY:")
    print("-" * 50)
    print(explanation)
    
    print(f"\n2. AVAILABLE EVIDENCE ({len(evidence_list)} pieces):")
    print("-" * 50)
    for i, ev in enumerate(evidence_list, 1):
        print(f"{i}. {ev.text[:70]}...")
        print(f"   Source: {ev.source_title} (credibility: {ev.credibility_score:.2f})")
    
    # Run self-verification
    print(f"\n3. RUNNING SELF-VERIFICATION...")
    print("-" * 50)
    
    verifier = SelfVerificationModule()
    quality_score, verification_results = verifier.verify_explanation(explanation, evidence_list)
    
    # Demonstrate different output formats
    print(f"\n4. OUTPUT FORMAT DEMONSTRATIONS:")
    print("=" * 50)
    
    # Format 1: Console Output (default)
    print(f"\n📊 CONSOLE OUTPUT FORMAT:")
    console_output = print_verification_results(quality_score, verification_results, "console")
    print(console_output)
    
    # Format 2: Summary Output
    print(f"\n📋 SUMMARY OUTPUT FORMAT:")
    summary_output = print_verification_results(quality_score, verification_results, "summary")
    print(summary_output)
    
    # Format 3: Detailed Output
    print(f"\n📝 DETAILED OUTPUT FORMAT:")
    detailed_output = print_verification_results(quality_score, verification_results, "detailed")
    print(detailed_output)
    
    # Format 4: JSON Output
    print(f"\n🔧 JSON OUTPUT FORMAT:")
    json_output = print_verification_results(quality_score, verification_results, "json")
    print(json_output)
    
    # Format 5: Correction Report
    print(f"\n🔧 CORRECTION REPORT:")
    print("-" * 50)
    
    corrected_explanation = verifier.correct_hallucinations(
        explanation, verification_results, quality_score, "adaptive"
    )
    
    formatter = SelfVerificationOutputFormatter()
    correction_report = formatter.format_correction_report(
        explanation, corrected_explanation, quality_score, "adaptive"
    )
    print(correction_report)
    
    return quality_score, verification_results


def demo_rag_with_formatted_output():
    """Demonstrate RAG explanation generator with formatted self-verification output."""
    
    print("\n" + "=" * 80)
    print("RAG EXPLANATION GENERATOR WITH FORMATTED OUTPUT")
    print("=" * 80)
    
    # Create test claim
    claim = Claim(
        text="Việt Nam có 63 tỉnh thành phố và GDP đạt 430 tỷ USD năm 2023",
        context="Thông tin về Việt Nam",
        confidence=0.9,
        sentence_type="factual_claim",
        start_idx=0,
        end_idx=100,
        language="vi"
    )
    
    # Evidence list
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
        supporting_evidence=[ev.id for ev in evidence_list],
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
        )
    ]
    
    print(f"\n1. GENERATING EXPLANATION WITH SELF-VERIFICATION...")
    print("-" * 50)
    
    # Load config and create controller
    config = SystemConfig.from_env()
    print(f"Loaded configuration for provider: {config.model.llm_provider}")
    
    llm_config = {
        "providers": [config.model.llm_provider],
        "gemini_api_key": config.agent.gemini_api_key,
        "local_model": config.model.llm_model
    }
    llm_controller = create_llm_controller(llm_config)
    
    generator = RAGExplanationGenerator(
        llm_controller=llm_controller,
        enable_self_verification=True
    )
    
    explanation, verification_metadata = generator.generate_explanation(
        claim=claim,
        verdict=verdict,
        evidence_list=evidence_list,
        reasoning_steps=reasoning_steps
    )
    
    print(f"\n2. GENERATED EXPLANATION:")
    print("-" * 50)
    print(explanation)
    
    print(f"\n3. VERIFICATION METADATA (JSON FORMAT):")
    print("-" * 50)
    if verification_metadata:
        print(json.dumps(verification_metadata, indent=2, ensure_ascii=False))
    else:
        print("No verification metadata available")
    
    return explanation, verification_metadata


def demo_hallucination_detection_output():
    """Demonstrate hallucination detection with formatted output."""
    
    print("\n" + "=" * 80)
    print("HALLUCINATION DETECTION WITH FORMATTED OUTPUT")
    print("=" * 80)
    
    # Create explanation with fabricated information
    fabricated_explanation = """
    Việt Nam có 70 tỉnh thành phố và dân số 120 triệu người năm 2023.
    Ngoài ra, Việt Nam có 50 sân bay quốc tế và xuất khẩu 100 triệu tấn gạo mỗi năm.
    Thủ đô Hà Nội có dân số 15 triệu người và diện tích 5000 km².
    """
    
    # Limited evidence that contradicts the fabricated claims
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
    
    print(f"\n1. FABRICATED EXPLANATION:")
    print("-" * 50)
    print(fabricated_explanation)
    
    print(f"\n2. CONTRADICTING EVIDENCE:")
    print("-" * 50)
    for i, ev in enumerate(evidence_list, 1):
        print(f"{i}. {ev.text}")
        print(f"   Source: {ev.source_title}")
    
    print(f"\n3. RUNNING HALLUCINATION DETECTION...")
    print("-" * 50)
    
    verifier = SelfVerificationModule()
    quality_score, verification_results = verifier.verify_explanation(
        fabricated_explanation, evidence_list
    )
    
    print(f"\n4. HALLUCINATION DETECTION RESULTS:")
    print("-" * 50)
    
    # Use console format for clear presentation
    detection_output = print_verification_results(
        quality_score, verification_results, "console"
    )
    print(detection_output)
    
    # Show correction strategies
    print(f"\n5. CORRECTION STRATEGIES:")
    print("-" * 50)
    
    strategies = ["flag", "revise", "remove", "adaptive"]
    
    for strategy in strategies:
        print(f"\n🔧 Strategy: {strategy.upper()}")
        corrected = verifier.correct_hallucinations(
            fabricated_explanation, verification_results, quality_score, strategy
        )
        
        # Show length change and sample
        length_change = len(corrected) - len(fabricated_explanation)
        print(f"   Length change: {length_change:+d} characters")
        print(f"   Sample (first 100 chars): {corrected[:100]}...")
    
    return quality_score, verification_results


if __name__ == "__main__":
    try:
        print("🚀 Starting Self-Verification Output Functions Demo...")
        
        # Demo 1: Basic output functions
        quality1, results1 = demo_output_functions()
        
        # Demo 2: RAG integration with formatted output
        explanation2, metadata2 = demo_rag_with_formatted_output()
        
        # Demo 3: Hallucination detection with formatted output
        quality3, results3 = demo_hallucination_detection_output()
        
        print(f"\n" + "=" * 80)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Final summary
        print(f"\nFINAL SUMMARY:")
        print(f"Demo 1 - Quality Score: {quality1.overall_score:.2f}")
        print(f"Demo 2 - Quality Score: {metadata2.get('quality_score', 'N/A') if metadata2 else 'N/A'}")
        print(f"Demo 3 - Quality Score: {quality3.overall_score:.2f}")
        
        print(f"\n✅ Self-verification output functions are working perfectly with Gemini API!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)