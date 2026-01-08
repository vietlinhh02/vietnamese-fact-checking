"""Property-based tests for self-verification module."""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List
import logging

from src.self_verification import (
    SelfVerificationModule, ExplanationClaimExtractor, ClaimVerifier, 
    QualityScorer, VerificationResult, QualityScore
)
from src.data_models import Claim, Evidence

logger = logging.getLogger(__name__)


# Test data generators
@st.composite
def generate_claim(draw):
    """Generate a random claim."""
    text = draw(st.text(min_size=10, max_size=200))
    assume(len(text.strip()) > 5)  # Ensure meaningful text
    
    return Claim(
        text=text.strip(),
        context=draw(st.text(min_size=20, max_size=300)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        sentence_type="factual_claim",
        start_idx=0,
        end_idx=len(text),
        language="vi"
    )


@st.composite
def generate_evidence(draw):
    """Generate random evidence."""
    text = draw(st.text(min_size=20, max_size=500))
    assume(len(text.strip()) > 10)
    
    return Evidence(
        text=text.strip(),
        source_url=f"https://example.com/{draw(st.integers(min_value=1, max_value=1000))}",
        source_title=draw(st.text(min_size=5, max_size=100)),
        credibility_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        language="vi"
    )


@st.composite
def generate_explanation(draw):
    """Generate random explanation text."""
    sentences = draw(st.lists(st.text(min_size=10, max_size=100), min_size=2, max_size=10))
    explanation = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
    assume(len(explanation) > 20)
    return explanation


class TestSelfVerificationProperties:
    """Property-based tests for self-verification module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = SelfVerificationModule()
    
    @given(generate_explanation())
    @settings(max_examples=100)
    def test_property_27_self_verification_execution(self, explanation):
        """
        **Feature: vietnamese-fact-checking, Property 27: Self-Verification Execution**
        **Validates: Requirements 10.1, 10.2**
        
        For any generated explanation, the system should extract factual claims 
        from the explanation and perform verification searches for each claim.
        """
        # Generate some evidence for context
        evidence_list = [
            Evidence(
                text="Sample evidence text for verification",
                source_url="https://example.com/evidence",
                source_title="Sample Evidence",
                credibility_score=0.8,
                language="vi"
            )
        ]
        
        # Run self-verification
        quality_score, verification_results = self.verifier.verify_explanation(
            explanation, evidence_list
        )
        
        # Property: Self-verification should always execute and return results
        assert isinstance(quality_score, QualityScore)
        assert isinstance(verification_results, list)
        
        # Property: Quality score should be in valid range
        assert 0.0 <= quality_score.overall_score <= 1.0
        assert quality_score.total_claims >= 0
        assert quality_score.verified_claims >= 0
        assert quality_score.verified_claims <= quality_score.total_claims
        
        # Property: If claims are extracted, verification results should exist
        if quality_score.total_claims > 0:
            assert len(verification_results) == quality_score.total_claims
            
            # Each verification result should have required fields
            for result in verification_results:
                assert isinstance(result, VerificationResult)
                assert isinstance(result.claim, Claim)
                assert isinstance(result.is_verified, bool)
                assert 0.0 <= result.confidence <= 1.0
                assert result.verification_method is not None
        
        # Property: Verification rate should be consistent
        if quality_score.total_claims > 0:
            expected_rate = quality_score.verified_claims / quality_score.total_claims
            assert abs(quality_score.verification_rate - expected_rate) < 0.001
    
    @given(st.lists(generate_claim(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_verification_loop_completeness(self, claims):
        """Test that verification loop processes all claims."""
        evidence_list = [
            Evidence(
                text="Test evidence for verification",
                source_url="https://test.com/evidence",
                source_title="Test Evidence",
                credibility_score=0.7,
                language="vi"
            )
        ]
        
        # Run verification loop
        results = self.verifier.claim_verifier.run_verification_loop(claims, evidence_list)
        
        # Property: All claims should be processed
        assert len(results) == len(claims)
        
        # Property: Each result should correspond to a claim
        for i, result in enumerate(results):
            assert result.claim.text == claims[i].text
            assert isinstance(result.is_verified, bool)
            assert 0.0 <= result.confidence <= 1.0
    
    @given(st.lists(generate_evidence(), min_size=0, max_size=3))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_29_quality_score_output(self, evidence_list):
        """
        **Feature: vietnamese-fact-checking, Property 29: Quality Score Output**
        **Validates: Requirements 10.4**
        
        For any completed fact-check, the final output should include a quality 
        score in the range [0, 1] indicating the reliability of the explanation 
        based on verification results.
        """
        # Create a sample explanation
        explanation = "Việt Nam có 63 tỉnh thành. Dân số khoảng 98 triệu người. GDP đạt 400 tỷ USD."
        
        # Run verification
        quality_score, verification_results = self.verifier.verify_explanation(
            explanation, evidence_list
        )
        
        # Property: Quality score must be in valid range [0, 1]
        assert 0.0 <= quality_score.overall_score <= 1.0
        
        # Property: Quality score must have all required fields
        assert hasattr(quality_score, 'overall_score')
        assert hasattr(quality_score, 'verified_claims')
        assert hasattr(quality_score, 'total_claims')
        assert hasattr(quality_score, 'verification_rate')
        assert hasattr(quality_score, 'confidence_scores')
        assert hasattr(quality_score, 'flagged_claims')
        assert hasattr(quality_score, 'explanation')
        
        # Property: Verification rate should be in [0, 1]
        assert 0.0 <= quality_score.verification_rate <= 1.0
        
        # Property: Counts should be non-negative and consistent
        assert quality_score.verified_claims >= 0
        assert quality_score.total_claims >= 0
        assert quality_score.verified_claims <= quality_score.total_claims
        
        # Property: Flagged claims should be subset of total claims
        assert len(quality_score.flagged_claims) <= quality_score.total_claims
        
        # Property: If no claims extracted, verification rate should be 1.0 (no hallucinations)
        if quality_score.total_claims == 0:
            assert quality_score.verification_rate == 1.0
    
    @given(generate_explanation(), st.lists(generate_evidence(), min_size=0, max_size=3))
    @settings(max_examples=30)
    def test_hallucination_detection_consistency(self, explanation, evidence_list):
        """Test that hallucination detection is consistent."""
        quality_score, verification_results = self.verifier.verify_explanation(
            explanation, evidence_list
        )
        
        # Property: Flagged claims should match unverified claims
        unverified_claims = [r.claim for r in verification_results if not r.is_verified]
        assert len(quality_score.flagged_claims) == len(unverified_claims)
        
        # Property: Verified claims count should match verification results
        verified_count = sum(1 for r in verification_results if r.is_verified)
        assert quality_score.verified_claims == verified_count
    
    @given(generate_explanation())
    @settings(max_examples=30)
    def test_correction_preserves_structure(self, explanation):
        """Test that correction preserves explanation structure."""
        evidence_list = []  # Empty evidence to force corrections
        
        quality_score, verification_results = self.verifier.verify_explanation(
            explanation, evidence_list
        )
        
        # Test different correction strategies
        for strategy in ["remove", "flag", "revise"]:
            corrected = self.verifier.correct_hallucinations(
                explanation, verification_results, quality_score, strategy
            )
            
            # Property: Corrected explanation should be a string
            assert isinstance(corrected, str)
            
            # Property: Corrected explanation should not be empty (unless original was very short)
            if len(explanation.strip()) > 20:
                assert len(corrected.strip()) > 0
    
    @given(st.lists(generate_claim(), min_size=1, max_size=5))
    @settings(max_examples=30)
    def test_quality_scorer_monotonicity(self, claims):
        """Test that quality scorer behaves monotonically with verification results."""
        # Create verification results with different verification rates
        all_verified = [
            VerificationResult(
                claim=claim,
                is_verified=True,
                supporting_evidence=[],
                confidence=0.8,
                verification_method="evidence_match",
                explanation="Verified"
            ) for claim in claims
        ]
        
        all_unverified = [
            VerificationResult(
                claim=claim,
                is_verified=False,
                supporting_evidence=[],
                confidence=0.0,
                verification_method="no_verification",
                explanation="Not verified"
            ) for claim in claims
        ]
        
        # Compute quality scores
        scorer = QualityScorer()
        score_all_verified = scorer.compute_quality_score(all_verified)
        score_all_unverified = scorer.compute_quality_score(all_unverified)
        
        # Property: All verified should have higher score than all unverified
        assert score_all_verified.overall_score >= score_all_unverified.overall_score
        
        # Property: All verified should have 100% verification rate
        assert score_all_verified.verification_rate == 1.0
        
        # Property: All unverified should have 0% verification rate
        assert score_all_unverified.verification_rate == 0.0
    
    @given(generate_explanation())
    @settings(max_examples=20)
    def test_claim_extraction_deterministic(self, explanation):
        """Test that claim extraction is deterministic."""
        extractor = ExplanationClaimExtractor()
        
        # Extract claims multiple times
        claims1 = extractor.extract_claims(explanation)
        claims2 = extractor.extract_claims(explanation)
        
        # Property: Should extract same number of claims
        assert len(claims1) == len(claims2)
        
        # Property: Claims should have same text (order might differ)
        texts1 = {claim.text for claim in claims1}
        texts2 = {claim.text for claim in claims2}
        assert texts1 == texts2
    
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_empty_explanation_handling(self, short_text):
        """Test handling of empty or very short explanations."""
        # Test with very short text that might not contain claims
        quality_score, verification_results = self.verifier.verify_explanation(
            short_text, []
        )
        
        # Property: Should handle gracefully without errors
        assert isinstance(quality_score, QualityScore)
        assert isinstance(verification_results, list)
        assert quality_score.overall_score >= 0.0
        
        # Property: If no claims found, should have high quality (no hallucinations)
        if quality_score.total_claims == 0:
            assert quality_score.overall_score >= 0.8


if __name__ == "__main__":
    # Run property tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run a few test cases manually
    test_instance = TestSelfVerificationProperties()
    test_instance.setup_method()
    
    # Test with sample explanation
    sample_explanation = "Việt Nam có 63 tỉnh thành phố. Dân số đạt 98 triệu người vào năm 2023."
    sample_evidence = [
        Evidence(
            text="Việt Nam có 63 tỉnh thành phố trực thuộc trung ương",
            source_url="https://example.com/provinces",
            source_title="Danh sách tỉnh thành",
            credibility_score=0.9,
            language="vi"
        )
    ]
    
    quality_score, results = test_instance.verifier.verify_explanation(
        sample_explanation, sample_evidence
    )
    
    print(f"Quality Score: {quality_score.overall_score:.2f}")
    print(f"Verification Rate: {quality_score.verification_rate:.1%}")
    print(f"Claims: {quality_score.verified_claims}/{quality_score.total_claims}")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.claim.text[:50]}... -> {result.is_verified}")