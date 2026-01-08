#!/usr/bin/env python3
"""Demo structured outputs using prompt engineering with Gemini."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from src.llm_controller import GeminiProvider
from src.self_verification import SelfVerificationModule, SelfVerificationOutputFormatter
from src.data_models import Evidence

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_structured_verification_prompt(explanation: str, evidence_summary: str) -> str:
    """Create prompt for structured verification output using prompt engineering."""
    
    prompt = f"""
Analyze the following explanation for factual accuracy and provide a structured JSON response.

EXPLANATION TO VERIFY:
{explanation}

AVAILABLE EVIDENCE:
{evidence_summary}

Please provide your analysis in the following JSON format:

{{
  "quality_assessment": {{
    "overall_score": <number between 0 and 1>,
    "verification_rate": <number between 0 and 1>,
    "verified_claims": <integer>,
    "total_claims": <integer>,
    "flagged_claims": <integer>,
    "quality_level": "<HIGH|MEDIUM|LOW>",
    "explanation": "<detailed explanation in Vietnamese>"
  }},
  "verification_results": [
    {{
      "claim_text": "<extracted claim text>",
      "is_verified": <true|false>,
      "confidence": <number between 0 and 1>,
      "verification_method": "<evidence_match|search_verification|no_verification>",
      "explanation": "<explanation for this claim>",
      "supporting_evidence_count": <integer>
    }}
  ],
  "recommendations": [
    "<recommendation 1>",
    "<recommendation 2>"
  ],
  "correction_strategy": "<flag|revise|remove|adaptive>",
  "hallucination_detected": <true|false>
}}

Instructions:
1. Extract all factual claims from the explanation
2. Verify each claim against the available evidence
3. Calculate overall quality score based on verification rate and confidence
4. Provide specific recommendations for improvement
5. Detect potential hallucinations (unverified claims with high confidence)
6. Respond ONLY with valid JSON, no additional text

Focus on Vietnamese fact-checking context and provide explanations in Vietnamese.
"""
    
    return prompt


def demo_structured_prompt_engineering():
    """Demo structured outputs using prompt engineering."""
    
    print("=" * 80)
    print("STRUCTURED OUTPUTS VIA PROMPT ENGINEERING")
    print("=" * 80)
    
    # Test case 1: Mixed quality explanation
    explanation1 = """
    Việt Nam có 63 tỉnh thành phố trực thuộc trung ương theo quy định hiện hành.
    GDP của Việt Nam năm 2023 đạt 430,1 tỷ USD theo báo cáo chính thức.
    Tuy nhiên, có một số thông tin chưa được xác minh như tỷ lệ biết chữ 99.9%.
    Việt Nam cũng có 100 sân bay quốc tế (thông tin này chưa được kiểm chứng).
    """
    
    evidence_summary1 = """
    Available Evidence:
    1. Việt Nam có 63 tỉnh thành phố - Source: Chính phủ Việt Nam (credibility: 0.95)
    2. GDP 430,1 tỷ USD năm 2023 - Source: Tổng cục Thống kê (credibility: 0.92)
    3. No evidence available for literacy rate (99.9%)
    4. No evidence available for number of international airports (100)
    """
    
    print(f"\n1. TEST CASE 1: MIXED QUALITY EXPLANATION")
    print("-" * 50)
    print(f"Explanation: {explanation1[:100]}...")
    
    # Create structured prompt
    prompt1 = create_structured_verification_prompt(explanation1, evidence_summary1)
    
    # Initialize Gemini
    gemini = GeminiProvider()
    
    if not gemini.is_available():
        print("❌ Gemini not available - please set GEMINI_API_KEY")
        return None
    
    try:
        # Generate structured response
        messages = [{"role": "user", "content": prompt1}]
        
        response = gemini.generate(
            messages=messages,
            max_tokens=1500,
            temperature=0.1
        )
        
        print(f"\n2. GEMINI STRUCTURED RESPONSE:")
        print("-" * 50)
        
        # Parse JSON response
        try:
            structured_result = json.loads(response.content)
            print(json.dumps(structured_result, indent=2, ensure_ascii=False))
            
            # Validate structure
            print(f"\n3. VALIDATION:")
            print("-" * 50)
            
            quality_assessment = structured_result.get('quality_assessment', {})
            verification_results = structured_result.get('verification_results', [])
            recommendations = structured_result.get('recommendations', [])
            
            print(f"✅ Quality score: {quality_assessment.get('overall_score', 'N/A')}")
            print(f"✅ Quality level: {quality_assessment.get('quality_level', 'N/A')}")
            print(f"✅ Claims analyzed: {len(verification_results)}")
            print(f"✅ Recommendations: {len(recommendations)}")
            print(f"✅ Hallucination detected: {structured_result.get('hallucination_detected', 'N/A')}")
            
            return structured_result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {response.content[:500]}...")
            return None
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return None


def demo_fabricated_claims_detection():
    """Demo detection of fabricated claims."""
    
    print(f"\n" + "=" * 80)
    print("FABRICATED CLAIMS DETECTION")
    print("=" * 80)
    
    # Test case 2: Fabricated claims
    explanation2 = """
    Việt Nam có 70 tỉnh thành phố và dân số 120 triệu người năm 2023.
    Ngoài ra, Việt Nam có 50 sân bay quốc tế và xuất khẩu 100 triệu tấn gạo mỗi năm.
    Thủ đô Hà Nội có dân số 15 triệu người và diện tích 5000 km².
    """
    
    evidence_summary2 = """
    Available Evidence:
    1. Việt Nam có 63 tỉnh thành phố (NOT 70) - Source: Chính phủ (credibility: 0.95)
    2. Dân số Việt Nam khoảng 98 triệu người (NOT 120) - Source: Thống kê dân số (credibility: 0.90)
    3. No evidence for 50 international airports
    4. No evidence for 100 million tons rice export
    5. No evidence for Hanoi population of 15 million or area of 5000 km²
    """
    
    print(f"\n1. FABRICATED CLAIMS TEST:")
    print("-" * 50)
    print(f"Explanation: {explanation2[:100]}...")
    
    # Create prompt
    prompt2 = create_structured_verification_prompt(explanation2, evidence_summary2)
    
    gemini = GeminiProvider()
    
    try:
        messages = [{"role": "user", "content": prompt2}]
        
        response = gemini.generate(
            messages=messages,
            max_tokens=1500,
            temperature=0.1
        )
        
        print(f"\n2. HALLUCINATION DETECTION RESULT:")
        print("-" * 50)
        
        structured_result = json.loads(response.content)
        print(json.dumps(structured_result, indent=2, ensure_ascii=False))
        
        # Analyze hallucination detection
        print(f"\n3. HALLUCINATION ANALYSIS:")
        print("-" * 50)
        
        quality_assessment = structured_result.get('quality_assessment', {})
        verification_results = structured_result.get('verification_results', [])
        hallucination_detected = structured_result.get('hallucination_detected', False)
        
        overall_score = quality_assessment.get('overall_score', 0)
        flagged_claims = quality_assessment.get('flagged_claims', 0)
        
        print(f"✅ Hallucination detected: {hallucination_detected}")
        print(f"✅ Overall quality: {overall_score:.2f} ({'LOW' if overall_score < 0.5 else 'MEDIUM' if overall_score < 0.8 else 'HIGH'})")
        print(f"✅ Flagged claims: {flagged_claims}")
        
        # Count unverified claims
        unverified_claims = [r for r in verification_results if not r.get('is_verified', True)]
        print(f"✅ Unverified claims: {len(unverified_claims)}")
        
        if unverified_claims:
            print(f"\n   Unverified claims:")
            for i, claim in enumerate(unverified_claims[:3], 1):
                print(f"   {i}. {claim.get('claim_text', '')[:60]}...")
        
        return structured_result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def compare_with_traditional_verification():
    """Compare structured prompt output with traditional verification."""
    
    print(f"\n" + "=" * 80)
    print("COMPARISON: STRUCTURED PROMPT vs TRADITIONAL")
    print("=" * 80)
    
    explanation = """
    Việt Nam có 63 tỉnh thành phố và GDP đạt 430 tỷ USD năm 2023.
    Tuy nhiên, có thông tin chưa xác minh về tỷ lệ biết chữ 99.9%.
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
    
    print(f"\n1. TRADITIONAL SELF-VERIFICATION:")
    print("-" * 50)
    
    # Traditional approach
    verifier = SelfVerificationModule()
    quality_score, verification_results = verifier.verify_explanation(explanation, evidence_list)
    
    formatter = SelfVerificationOutputFormatter()
    traditional_output = formatter.to_structured_output(
        quality_score=quality_score,
        verification_results=verification_results,
        correction_applied=False,
        correction_strategy="none",
        original_length=len(explanation),
        corrected_length=len(explanation)
    )
    
    print(f"Quality Score: {quality_score.overall_score:.2f}")
    print(f"Verification Rate: {quality_score.verification_rate:.1%}")
    print(f"Claims: {quality_score.verified_claims}/{quality_score.total_claims}")
    
    print(f"\n2. STRUCTURED PROMPT APPROACH:")
    print("-" * 50)
    
    # Structured prompt approach
    evidence_summary = """
    Available Evidence:
    1. Việt Nam có 63 tỉnh thành phố - Source: Chính phủ (credibility: 0.95)
    2. GDP 430,1 tỷ USD năm 2023 - Source: Tổng cục Thống kê (credibility: 0.92)
    3. No evidence for literacy rate 99.9%
    """
    
    prompt = create_structured_verification_prompt(explanation, evidence_summary)
    
    gemini = GeminiProvider()
    
    try:
        messages = [{"role": "user", "content": prompt}]
        response = gemini.generate(messages=messages, max_tokens=1000, temperature=0.1)
        
        structured_result = json.loads(response.content)
        
        gemini_quality = structured_result.get('quality_assessment', {})
        gemini_score = gemini_quality.get('overall_score', 0)
        gemini_rate = gemini_quality.get('verification_rate', 0)
        gemini_claims = f"{gemini_quality.get('verified_claims', 0)}/{gemini_quality.get('total_claims', 0)}"
        
        print(f"Quality Score: {gemini_score:.2f}")
        print(f"Verification Rate: {gemini_rate:.1%}")
        print(f"Claims: {gemini_claims}")
        
        print(f"\n3. COMPARISON ANALYSIS:")
        print("-" * 50)
        
        score_diff = abs(quality_score.overall_score - gemini_score)
        rate_diff = abs(quality_score.verification_rate - gemini_rate)
        
        print(f"Score difference: {score_diff:.3f}")
        print(f"Rate difference: {rate_diff:.3f}")
        
        if score_diff < 0.1 and rate_diff < 0.1:
            print("✅ Results are consistent between approaches")
        else:
            print("⚠ Results differ - may need calibration")
        
        print(f"\n4. ADVANTAGES:")
        print("-" * 50)
        print("Traditional approach:")
        print("  ✅ Deterministic and consistent")
        print("  ✅ Fast processing")
        print("  ✅ No API costs")
        
        print("\nStructured prompt approach:")
        print("  ✅ More nuanced analysis")
        print("  ✅ Better natural language understanding")
        print("  ✅ Flexible reasoning")
        print("  ❌ API costs and latency")
        
        return {
            "traditional": traditional_output,
            "structured_prompt": structured_result,
            "score_difference": score_diff,
            "rate_difference": rate_diff
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    try:
        print("🚀 Starting Structured Prompt Engineering Demo...")
        
        # Demo 1: Mixed quality explanation
        result1 = demo_structured_prompt_engineering()
        
        # Demo 2: Fabricated claims detection
        result2 = demo_fabricated_claims_detection()
        
        # Demo 3: Comparison with traditional
        result3 = compare_with_traditional_verification()
        
        print(f"\n" + "=" * 80)
        print("🎉 STRUCTURED PROMPT ENGINEERING DEMO COMPLETED!")
        print("=" * 80)
        
        success_count = sum([1 if r else 0 for r in [result1, result2, result3]])
        print(f"\n✅ Successful demos: {success_count}/3")
        
        if success_count > 0:
            print(f"\n🚀 Key Benefits of Structured Prompt Engineering:")
            print(f"   ✅ Consistent JSON output format")
            print(f"   ✅ Rich analysis with natural language understanding")
            print(f"   ✅ Flexible reasoning and explanation")
            print(f"   ✅ Hallucination detection capabilities")
            print(f"   ✅ Vietnamese language support")
            print(f"   ✅ Production-ready structured outputs")
        
        print(f"\n🎯 Self-verification system now supports both:")
        print(f"   1. Traditional rule-based verification (fast, deterministic)")
        print(f"   2. LLM-powered structured analysis (nuanced, flexible)")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)