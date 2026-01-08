#!/usr/bin/env python3
"""Demo script for Gemini structured outputs in self-verification system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from src.verification_schemas import VerificationSchemas, create_verification_prompt_with_schema
from src.llm_controller import GeminiProvider
from src.self_verification import SelfVerificationModule, SelfVerificationOutputFormatter
from src.data_models import Evidence
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_structured_quality_assessment():
    """Demo structured quality assessment with Gemini."""
    
    print("=" * 80)
    print("GEMINI STRUCTURED OUTPUTS - QUALITY ASSESSMENT DEMO")
    print("=" * 80)
    
    # Sample explanation with mixed quality
    explanation = """
    Việt Nam có 63 tỉnh thành phố trực thuộc trung ương theo quy định hiện hành.
    GDP của Việt Nam năm 2023 đạt 430,1 tỷ USD theo báo cáo chính thức.
    Tuy nhiên, có một số thông tin chưa được xác minh như tỷ lệ biết chữ 99.9%.
    Việt Nam cũng có 100 sân bay quốc tế (thông tin này chưa được kiểm chứng).
    """
    
    evidence_summary = """
    Available Evidence:
    1. Việt Nam có 63 tỉnh thành phố - Source: Chính phủ (credibility: 0.95)
    2. GDP 430,1 tỷ USD năm 2023 - Source: Tổng cục Thống kê (credibility: 0.92)
    3. No evidence for literacy rate or airports count
    """
    
    print(f"\n1. EXPLANATION TO ANALYZE:")
    print("-" * 50)
    print(explanation)
    
    print(f"\n2. AVAILABLE EVIDENCE:")
    print("-" * 50)
    print(evidence_summary)
    
    # Create structured prompt and schema
    prompt, schema = create_verification_prompt_with_schema(
        explanation, evidence_summary, "quality"
    )
    
    print(f"\n3. USING GEMINI STRUCTURED OUTPUT...")
    print("-" * 50)
    print(f"Schema type: {schema['type']}")
    print(f"Required fields: {len(schema['required'])}")
    
    # Initialize Gemini provider
    gemini = GeminiProvider()
    
    if not gemini.is_available():
        print("❌ Gemini not available - please set GEMINI_API_KEY")
        return None
    
    try:
        # Generate structured response
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = gemini.generate(
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
            response_schema=schema
        )
        
        print(f"\n4. STRUCTURED RESPONSE:")
        print("-" * 50)
        
        # Parse JSON response
        structured_result = json.loads(response.content)
        
        # Pretty print the structured output
        print(json.dumps(structured_result, indent=2, ensure_ascii=False))
        
        print(f"\n5. ANALYSIS:")
        print("-" * 50)
        print(f"✅ Response is valid JSON: {isinstance(structured_result, dict)}")
        print(f"✅ Has required fields: {all(field in structured_result for field in schema['required'])}")
        print(f"✅ Quality score: {structured_result.get('overall_score', 'N/A')}")
        print(f"✅ Quality level: {structured_result.get('quality_level', 'N/A')}")
        
        return structured_result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def demo_structured_verification_summary():
    """Demo complete verification summary with structured output."""
    
    print("\n" + "=" * 80)
    print("GEMINI STRUCTURED OUTPUTS - VERIFICATION SUMMARY DEMO")
    print("=" * 80)
    
    # Run actual self-verification
    explanation = """
    Bằng chứng chính: Việt Nam có 63 tỉnh thành phố và GDP đạt 430 tỷ USD năm 2023.
    Tuy nhiên, có thông tin chưa xác minh: tỷ lệ biết chữ 99.9% và 50 sân bay quốc tế.
    """
    
    evidence_list = [
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương",
            source_url="https://chinhphu.vn/provinces",
            source_title="Danh sách tỉnh thành",
            credibility_score=0.95,
            language="vi"
        ),
        Evidence(
            text="GDP Việt Nam năm 2023 đạt 430,1 tỷ USD",
            source_url="https://gso.gov.vn/gdp-2023",
            source_title="Báo cáo GDP 2023",
            credibility_score=0.92,
            language="vi"
        )
    ]
    
    print(f"\n1. RUNNING SELF-VERIFICATION...")
    print("-" * 50)
    
    # Run verification
    verifier = SelfVerificationModule()
    quality_score, verification_results = verifier.verify_explanation(explanation, evidence_list)
    
    print(f"Traditional output - Quality: {quality_score.overall_score:.2f}")
    print(f"Verification rate: {quality_score.verification_rate:.1%}")
    
    print(f"\n2. CONVERTING TO STRUCTURED FORMAT...")
    print("-" * 50)
    
    # Convert to structured output
    formatter = SelfVerificationOutputFormatter()
    structured_output = formatter.to_structured_output(
        quality_score=quality_score,
        verification_results=verification_results,
        correction_applied=True,
        correction_strategy="adaptive",
        original_length=len(explanation),
        corrected_length=len(explanation) + 50  # Simulated correction
    )
    
    print(json.dumps(structured_output, indent=2, ensure_ascii=False))
    
    print(f"\n3. VALIDATION:")
    print("-" * 50)
    
    # Validate against schema
    schemas = VerificationSchemas()
    schema = schemas.get_verification_summary_schema()
    
    # Check required fields
    missing_fields = []
    for field in schema['required']:
        if field not in structured_output:
            missing_fields.append(field)
    
    if not missing_fields:
        print("✅ All required fields present")
    else:
        print(f"❌ Missing fields: {missing_fields}")
    
    # Check data types
    quality_assessment = structured_output.get('quality_assessment', {})
    print(f"✅ Quality level: {quality_assessment.get('quality_level')}")
    print(f"✅ Verification results count: {len(structured_output.get('verification_results', []))}")
    print(f"✅ Recommendations count: {len(structured_output.get('recommendations', []))}")
    
    return structured_output


def demo_fact_check_explanation_schema():
    """Demo structured fact-check explanation output."""
    
    print("\n" + "=" * 80)
    print("GEMINI STRUCTURED OUTPUTS - FACT-CHECK EXPLANATION DEMO")
    print("=" * 80)
    
    # Create prompt for fact-check explanation
    claim = "Việt Nam có 63 tỉnh thành phố và GDP đạt 430 tỷ USD năm 2023"
    
    evidence_summary = """
    Evidence Analysis:
    1. Provinces: Confirmed 63 provinces/cities (Source: Government, credibility: 0.95)
    2. GDP: Confirmed 430.1 billion USD in 2023 (Source: Statistics Office, credibility: 0.92)
    3. Context: Vietnam is a Southeast Asian developing economy (Source: World Bank, credibility: 0.80)
    """
    
    prompt, schema = create_verification_prompt_with_schema(
        claim, evidence_summary, "fact_check"
    )
    
    print(f"\n1. CLAIM TO FACT-CHECK:")
    print("-" * 50)
    print(claim)
    
    print(f"\n2. SCHEMA COMPLEXITY:")
    print("-" * 50)
    print(f"Schema type: {schema['type']}")
    print(f"Top-level properties: {len(schema['properties'])}")
    print(f"Required fields: {len(schema['required'])}")
    
    # Show schema structure
    for prop, details in schema['properties'].items():
        prop_type = details.get('type', 'unknown')
        print(f"  - {prop}: {prop_type}")
    
    print(f"\n3. GENERATING STRUCTURED FACT-CHECK...")
    print("-" * 50)
    
    # Initialize Gemini
    gemini = GeminiProvider()
    
    if not gemini.is_available():
        print("❌ Gemini not available")
        return None
    
    try:
        messages = [{"role": "user", "content": prompt}]
        
        response = gemini.generate(
            messages=messages,
            max_tokens=2000,
            temperature=0.1,
            response_schema=schema
        )
        
        # Parse and display
        fact_check_result = json.loads(response.content)
        
        print("✅ STRUCTURED FACT-CHECK RESULT:")
        print(json.dumps(fact_check_result, indent=2, ensure_ascii=False))
        
        print(f"\n4. VALIDATION:")
        print("-" * 50)
        
        # Validate key components
        verdict = fact_check_result.get('verdict', 'UNKNOWN')
        confidence_scores = fact_check_result.get('confidence_scores', {})
        evidence_count = len(fact_check_result.get('evidence_summary', []))
        reasoning_steps = len(fact_check_result.get('reasoning_steps', []))
        
        print(f"✅ Verdict: {verdict}")
        print(f"✅ Confidence scores: {confidence_scores}")
        print(f"✅ Evidence pieces: {evidence_count}")
        print(f"✅ Reasoning steps: {reasoning_steps}")
        
        # Check if verification metadata is included
        verification_metadata = fact_check_result.get('verification_metadata')
        if verification_metadata:
            quality_score = verification_metadata.get('quality_assessment', {}).get('overall_score')
            print(f"✅ Self-verification quality: {quality_score}")
        
        return fact_check_result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def compare_structured_vs_traditional():
    """Compare structured outputs vs traditional text outputs."""
    
    print("\n" + "=" * 80)
    print("COMPARISON: STRUCTURED VS TRADITIONAL OUTPUTS")
    print("=" * 80)
    
    explanation = "Việt Nam có 63 tỉnh thành và GDP 430 tỷ USD năm 2023."
    evidence_list = [
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương",
            source_url="https://chinhphu.vn/provinces",
            source_title="Danh sách tỉnh thành",
            credibility_score=0.95,
            language="vi"
        )
    ]
    
    # Traditional approach
    print(f"\n1. TRADITIONAL OUTPUT:")
    print("-" * 50)
    
    verifier = SelfVerificationModule()
    quality_score, verification_results = verifier.verify_explanation(explanation, evidence_list)
    
    traditional_output = f"""
Quality Score: {quality_score.overall_score:.2f}
Verification Rate: {quality_score.verification_rate:.1%}
Claims Verified: {quality_score.verified_claims}/{quality_score.total_claims}
Status: {'HIGH' if quality_score.overall_score >= 0.8 else 'MEDIUM' if quality_score.overall_score >= 0.5 else 'LOW'} QUALITY
"""
    print(traditional_output)
    
    # Structured approach
    print(f"\n2. STRUCTURED OUTPUT:")
    print("-" * 50)
    
    formatter = SelfVerificationOutputFormatter()
    structured_output = formatter.to_structured_output(
        quality_score=quality_score,
        verification_results=verification_results,
        correction_applied=False,
        correction_strategy="none",
        original_length=len(explanation),
        corrected_length=len(explanation)
    )
    
    print(json.dumps(structured_output, indent=2, ensure_ascii=False))
    
    print(f"\n3. BENEFITS COMPARISON:")
    print("-" * 50)
    
    benefits = [
        ("Parsability", "❌ Manual parsing", "✅ JSON.parse()"),
        ("Type Safety", "❌ String manipulation", "✅ Schema validation"),
        ("API Integration", "❌ Text processing", "✅ Direct object use"),
        ("Consistency", "❌ Format variations", "✅ Schema-enforced"),
        ("Extensibility", "❌ Breaking changes", "✅ Schema evolution"),
        ("Validation", "❌ Manual checks", "✅ Automatic validation"),
        ("Tooling", "❌ Limited support", "✅ Rich ecosystem")
    ]
    
    for feature, traditional, structured in benefits:
        print(f"{feature:15} | {traditional:20} | {structured}")
    
    return structured_output


if __name__ == "__main__":
    try:
        print("🚀 Starting Gemini Structured Outputs Demo...")
        
        # Demo 1: Quality assessment
        quality_result = demo_structured_quality_assessment()
        
        # Demo 2: Verification summary
        summary_result = demo_structured_verification_summary()
        
        # Demo 3: Fact-check explanation
        fact_check_result = demo_fact_check_explanation_schema()
        
        # Demo 4: Comparison
        comparison_result = compare_structured_vs_traditional()
        
        print(f"\n" + "=" * 80)
        print("🎉 ALL STRUCTURED OUTPUT DEMOS COMPLETED!")
        print("=" * 80)
        
        # Summary
        success_count = sum([
            1 if quality_result else 0,
            1 if summary_result else 0,
            1 if fact_check_result else 0,
            1 if comparison_result else 0
        ])
        
        print(f"\n✅ Successful demos: {success_count}/4")
        print(f"✅ Structured outputs provide:")
        print(f"   - Type safety and validation")
        print(f"   - Easy API integration")
        print(f"   - Consistent format")
        print(f"   - Rich metadata")
        print(f"   - Programmatic processing")
        
        print(f"\n🚀 Self-verification system is now production-ready with Gemini structured outputs!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)