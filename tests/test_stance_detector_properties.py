"""
Property-based tests for stance detection module.

**Feature: vietnamese-fact-checking, Property 15: Stance Classification Completeness**
**Validates: Requirements 6.3, 6.4**
"""

import pytest
from hypothesis import given, settings, strategies as st
from src.stance_detector import StanceDetector, StanceResult
import os
from pathlib import Path


# Check if model exists
MODEL_PATH = "models/xlmr_stance/best_model"
MODEL_EXISTS = Path(MODEL_PATH).exists()


@pytest.fixture(scope="module")
def stance_detector():
    """Load stance detector once for all tests."""
    if not MODEL_EXISTS:
        pytest.skip(f"Model not found at {MODEL_PATH}")
    
    return StanceDetector(model_path=MODEL_PATH)


# Strategy for generating Vietnamese text
vietnamese_text = st.text(
    alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'),
        whitelist_characters='àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ'
    ),
    min_size=20,
    max_size=200
)


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
@given(claim=vietnamese_text, evidence=vietnamese_text)
@settings(max_examples=100, deadline=5000)
def test_stance_classification_completeness(stance_detector, claim, evidence):
    """
    **Feature: vietnamese-fact-checking, Property 15: Stance Classification Completeness**
    **Validates: Requirements 6.3, 6.4**
    
    Property: For any claim-evidence pair, the stance detection should output 
    exactly one of three classes (Support, Refute, Neutral) along with 
    confidence scores for all three classes that sum to 1.0.
    """
    # Skip empty or whitespace-only inputs
    if not claim.strip() or not evidence.strip():
        return
    
    # Run stance detection
    result = stance_detector.detect_stance(
        claim=claim,
        evidence=evidence,
        claim_lang="vi",
        evidence_lang="vi"
    )
    
    # Property 1: Result should be a StanceResult object
    assert isinstance(result, StanceResult), \
        "Result should be a StanceResult object"
    
    # Property 2: Stance should be one of three valid classes
    valid_stances = {"support", "refute", "neutral"}
    assert result.stance in valid_stances, \
        f"Stance must be one of {valid_stances}, got {result.stance}"
    
    # Property 3: Confidence scores should exist for all three classes
    assert "support" in result.confidence_scores, \
        "Confidence scores must include 'support'"
    assert "refute" in result.confidence_scores, \
        "Confidence scores must include 'refute'"
    assert "neutral" in result.confidence_scores, \
        "Confidence scores must include 'neutral'"
    
    # Property 4: All confidence scores should be in [0, 1]
    for stance, score in result.confidence_scores.items():
        assert 0.0 <= score <= 1.0, \
            f"Confidence score for {stance} must be in [0, 1], got {score}"
    
    # Property 5: Confidence scores should sum to approximately 1.0
    total_confidence = sum(result.confidence_scores.values())
    assert abs(total_confidence - 1.0) < 0.01, \
        f"Confidence scores must sum to 1.0, got {total_confidence}"
    
    # Property 6: Language fields should be set correctly
    assert result.claim_lang == "vi", \
        f"Claim language should be 'vi', got {result.claim_lang}"
    assert result.evidence_lang == "vi", \
        f"Evidence language should be 'vi', got {result.evidence_lang}"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
@given(claim=vietnamese_text, evidence=st.text(min_size=20, max_size=200))
@settings(max_examples=50, deadline=5000)
def test_cross_lingual_stance_detection(stance_detector, claim, evidence):
    """
    Test that stance detection works for Vietnamese-English pairs.
    
    Property: For any Vietnamese claim and English evidence, the system
    should still produce valid stance classification.
    """
    # Skip empty or whitespace-only inputs
    if not claim.strip() or not evidence.strip():
        return
    
    # Run cross-lingual stance detection
    result = stance_detector.detect_stance(
        claim=claim,
        evidence=evidence,
        claim_lang="vi",
        evidence_lang="en"
    )
    
    # Same properties should hold for cross-lingual pairs
    valid_stances = {"support", "refute", "neutral"}
    assert result.stance in valid_stances
    
    assert len(result.confidence_scores) == 3
    
    total_confidence = sum(result.confidence_scores.values())
    assert abs(total_confidence - 1.0) < 0.01
    
    assert result.claim_lang == "vi"
    assert result.evidence_lang == "en"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
def test_batch_stance_detection_consistency(stance_detector):
    """
    Test that batch detection produces consistent results.
    
    Property: For any list of claim-evidence pairs, batch detection
    should produce the same results as individual detection.
    """
    # Test pairs
    pairs = [
        ("Việt Nam có dân số 98 triệu người", "Dân số Việt Nam là 98 triệu"),
        ("Thủ đô là Hà Nội", "Hà Nội là thủ đô"),
        ("GDP đạt 430 tỷ USD", "Việt Nam có nền kinh tế phát triển")
    ]
    
    # Individual detection
    individual_results = []
    for claim, evidence in pairs:
        result = stance_detector.detect_stance(claim, evidence)
        individual_results.append(result)
    
    # Batch detection
    batch_results = stance_detector.batch_detect_stance(pairs)
    
    # Results should match
    assert len(batch_results) == len(individual_results)
    
    for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
        # Stance should be the same
        assert individual.stance == batch.stance, \
            f"Pair {i}: Individual stance {individual.stance} != batch stance {batch.stance}"
        
        # Confidence scores should be very close (allowing for small numerical differences)
        for stance in ["support", "refute", "neutral"]:
            diff = abs(individual.confidence_scores[stance] - batch.confidence_scores[stance])
            assert diff < 0.01, \
                f"Pair {i}: Confidence for {stance} differs by {diff}"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
def test_stance_confidence_ordering(stance_detector):
    """
    Test that the predicted stance has the highest confidence score.
    
    Property: For any stance detection result, the confidence score
    of the predicted stance should be >= all other confidence scores.
    """
    test_cases = [
        ("Việt Nam có dân số 98 triệu người", "Dân số Việt Nam đạt 98 triệu người"),
        ("Thủ đô là Hà Nội", "Thủ đô là Thành phố Hồ Chí Minh"),
        ("GDP năm 2023 đạt 430 tỷ USD", "Việt Nam là quốc gia Đông Nam Á")
    ]
    
    for claim, evidence in test_cases:
        result = stance_detector.detect_stance(claim, evidence)
        
        predicted_confidence = result.confidence_scores[result.stance]
        
        # Predicted stance should have highest (or tied for highest) confidence
        for stance, score in result.confidence_scores.items():
            assert predicted_confidence >= score, \
                f"Predicted stance {result.stance} has confidence {predicted_confidence}, " \
                f"but {stance} has higher confidence {score}"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
@given(
    pairs=st.lists(
        st.tuples(vietnamese_text, vietnamese_text),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=20, deadline=10000)
def test_batch_detection_completeness(stance_detector, pairs):
    """
    Test that batch detection returns results for all input pairs.
    
    Property: For any list of N claim-evidence pairs, batch detection
    should return exactly N results, each with valid stance classification.
    """
    # Filter out empty pairs
    valid_pairs = [(c, e) for c, e in pairs if c.strip() and e.strip()]
    
    if not valid_pairs:
        return
    
    # Run batch detection
    results = stance_detector.batch_detect_stance(valid_pairs)
    
    # Should return same number of results as input pairs
    assert len(results) == len(valid_pairs), \
        f"Expected {len(valid_pairs)} results, got {len(results)}"
    
    # Each result should be valid
    valid_stances = {"support", "refute", "neutral"}
    for i, result in enumerate(results):
        assert isinstance(result, StanceResult), \
            f"Result {i} should be a StanceResult object"
        
        assert result.stance in valid_stances, \
            f"Result {i}: Invalid stance {result.stance}"
        
        assert len(result.confidence_scores) == 3, \
            f"Result {i}: Should have 3 confidence scores"
        
        total = sum(result.confidence_scores.values())
        assert abs(total - 1.0) < 0.01, \
            f"Result {i}: Confidence scores sum to {total}, not 1.0"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
