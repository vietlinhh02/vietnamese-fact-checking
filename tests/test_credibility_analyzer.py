"""Unit tests for credibility analyzer module."""

import pytest
from src.credibility_analyzer import (
    CredibilityAnalyzer,
    CredibilityFeatureExtractor,
    RuleBasedCredibilityScorer,
    CredibilityFeatures,
    CredibilityScore,
    MediaBiasFactCheckAPI,
    analyze_credibility
)
from src.web_crawler import WebContent


class TestCredibilityFeatureExtractor:
    """Unit tests for CredibilityFeatureExtractor."""
    
    def test_extract_features_from_state_managed_source(self):
        """Test feature extraction from state-managed Vietnamese source."""
        content = WebContent(
            url="https://vnexpress.net/article-123",
            title="Test Article",
            main_text="This is a test article with sufficient content. " * 20,
            author="Nguyen Van A",
            publish_date="2024-01-15T10:30:00",
            extraction_success=True
        )
        
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        assert features.domain == "vnexpress.net"
        assert features.tld == "net"
        assert features.uses_https is True
        assert features.is_state_managed is True
        assert features.has_author is True
        assert features.has_publish_date is True
        assert features.article_length > 0
        assert 0 <= features.writing_quality_score <= 1
    
    def test_extract_features_from_non_state_source(self):
        """Test feature extraction from non-state source."""
        content = WebContent(
            url="http://example.com/post-456",
            title="Blog Post",
            main_text="Short content",
            extraction_success=True
        )
        
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        assert features.domain == "example.com"
        assert features.tld == "com"
        assert features.uses_https is False
        assert features.is_state_managed is False
        assert features.has_author is False
        assert features.has_publish_date is False
    
    def test_extract_features_handles_www_prefix(self):
        """Test that www. prefix is removed from domain."""
        content = WebContent(
            url="https://www.vnexpress.net/article-123",
            title="Test",
            main_text="Content",
            extraction_success=True
        )
        
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        assert features.domain == "vnexpress.net"
        assert features.is_state_managed is True
    
    def test_writing_quality_score_calculation(self):
        """Test writing quality score calculation."""
        extractor = CredibilityFeatureExtractor()
        
        # Good quality text
        good_text = "This is a well-written article. It has proper punctuation. " \
                   "The sentences are of reasonable length. Each paragraph is well-structured.\n\n" \
                   "This is another paragraph with good content."
        
        good_score = extractor._compute_writing_quality(good_text)
        assert 0 < good_score <= 1
        
        # Poor quality text
        poor_text = "no caps no punctuation just words"
        poor_score = extractor._compute_writing_quality(poor_text)
        assert 0 <= poor_score < good_score
        
        # Empty text
        empty_score = extractor._compute_writing_quality("")
        assert empty_score == 0.0


class TestRuleBasedCredibilityScorer:
    """Unit tests for RuleBasedCredibilityScorer."""
    
    def test_score_state_managed_source(self):
        """Test scoring of state-managed source."""
        features = CredibilityFeatures(
            domain="vnexpress.net",
            tld="net",
            uses_https=True,
            is_state_managed=True,
            has_author=True,
            has_publish_date=True,
            article_length=1000,
            writing_quality_score=0.8
        )
        
        scorer = RuleBasedCredibilityScorer()
        score = scorer.compute_score(features)
        
        assert 0 <= score.overall_score <= 1
        assert score.overall_score >= 0.4  # At least state_managed weight
        assert 'state_managed' in score.feature_scores
        assert score.feature_scores['state_managed'] > 0
    
    def test_score_non_state_source(self):
        """Test scoring of non-state source."""
        features = CredibilityFeatures(
            domain="example.com",
            tld="com",
            uses_https=False,
            is_state_managed=False,
            has_author=False,
            has_publish_date=False,
            article_length=100,
            writing_quality_score=0.2
        )
        
        scorer = RuleBasedCredibilityScorer()
        score = scorer.compute_score(features)
        
        assert 0 <= score.overall_score <= 1
        assert score.overall_score < 0.4  # Less than state_managed weight
    
    def test_https_contributes_to_score(self):
        """Test that HTTPS usage contributes to score."""
        features_https = CredibilityFeatures(
            domain="example.com",
            tld="com",
            uses_https=True,
            is_state_managed=False,
            has_author=False,
            has_publish_date=False,
            article_length=500,
            writing_quality_score=0.5
        )
        
        features_no_https = CredibilityFeatures(
            domain="example.com",
            tld="com",
            uses_https=False,
            is_state_managed=False,
            has_author=False,
            has_publish_date=False,
            article_length=500,
            writing_quality_score=0.5
        )
        
        scorer = RuleBasedCredibilityScorer()
        score_https = scorer.compute_score(features_https)
        score_no_https = scorer.compute_score(features_no_https)
        
        assert score_https.overall_score > score_no_https.overall_score
    
    def test_vn_tld_contributes_to_score(self):
        """Test that .vn TLD contributes to score."""
        features_vn = CredibilityFeatures(
            domain="example.vn",
            tld="vn",
            uses_https=True,
            is_state_managed=False,
            has_author=False,
            has_publish_date=False,
            article_length=500,
            writing_quality_score=0.5
        )
        
        features_com = CredibilityFeatures(
            domain="example.com",
            tld="com",
            uses_https=True,
            is_state_managed=False,
            has_author=False,
            has_publish_date=False,
            article_length=500,
            writing_quality_score=0.5
        )
        
        scorer = RuleBasedCredibilityScorer()
        score_vn = scorer.compute_score(features_vn)
        score_com = scorer.compute_score(features_com)
        
        assert score_vn.overall_score > score_com.overall_score
    
    def test_explanation_generation(self):
        """Test that explanation is generated."""
        features = CredibilityFeatures(
            domain="vnexpress.net",
            tld="net",
            uses_https=True,
            is_state_managed=True,
            has_author=True,
            has_publish_date=True,
            article_length=1000,
            writing_quality_score=0.8
        )
        
        scorer = RuleBasedCredibilityScorer()
        score = scorer.compute_score(features)
        
        assert len(score.explanation) > 0
        assert "State-managed" in score.explanation or "state-managed" in score.explanation.lower()


class TestCredibilityAnalyzer:
    """Unit tests for CredibilityAnalyzer."""
    
    def test_analyze_credibility_end_to_end(self):
        """Test end-to-end credibility analysis."""
        content = WebContent(
            url="https://vnexpress.net/article-123",
            title="Test Article",
            main_text="This is a comprehensive test article with good content. " * 30,
            author="Nguyen Van A",
            publish_date="2024-01-15T10:30:00",
            extraction_success=True
        )
        
        analyzer = CredibilityAnalyzer()
        score = analyzer.analyze_credibility(content)
        
        assert isinstance(score, CredibilityScore)
        assert 0 <= score.overall_score <= 1
        assert len(score.feature_scores) > 0
        assert len(score.explanation) > 0
    
    def test_convenience_function(self):
        """Test convenience function."""
        content = WebContent(
            url="https://vtv.vn/article-456",
            title="News Article",
            main_text="Content here. " * 50,
            extraction_success=True
        )
        
        score = analyze_credibility(content)
        
        assert isinstance(score, CredibilityScore)
        assert 0 <= score.overall_score <= 1


class TestMediaBiasFactCheckAPI:
    """Unit tests for MediaBiasFactCheckAPI."""
    
    def test_mbfc_api_without_key(self):
        """Test MBFC API without API key."""
        api = MediaBiasFactCheckAPI()
        rating = api.get_rating("example.com")
        
        # Without API key, should return None
        assert rating is None
    
    def test_mbfc_api_caching(self):
        """Test that MBFC API caches results."""
        api = MediaBiasFactCheckAPI()
        
        # First call
        rating1 = api.get_rating("example.com")
        
        # Second call should use cache
        rating2 = api.get_rating("example.com")
        
        assert rating1 == rating2
        assert "example.com" in api.cache


class TestCredibilityScore:
    """Unit tests for CredibilityScore dataclass."""
    
    def test_valid_credibility_score(self):
        """Test creating valid credibility score."""
        score = CredibilityScore(
            overall_score=0.75,
            feature_scores={'test': 0.5},
            confidence=0.9,
            explanation="Test explanation"
        )
        
        assert score.overall_score == 0.75
        assert score.confidence == 0.9
    
    def test_invalid_overall_score_raises_error(self):
        """Test that invalid overall score raises error."""
        with pytest.raises(ValueError):
            CredibilityScore(
                overall_score=1.5,  # Invalid: > 1
                feature_scores={},
                confidence=1.0,
                explanation=""
            )
        
        with pytest.raises(ValueError):
            CredibilityScore(
                overall_score=-0.1,  # Invalid: < 0
                feature_scores={},
                confidence=1.0,
                explanation=""
            )
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError):
            CredibilityScore(
                overall_score=0.5,
                feature_scores={},
                confidence=1.5,  # Invalid: > 1
                explanation=""
            )


class TestCredibilityFeatures:
    """Unit tests for CredibilityFeatures dataclass."""
    
    def test_create_credibility_features(self):
        """Test creating credibility features."""
        features = CredibilityFeatures(
            domain="example.com",
            tld="com",
            uses_https=True,
            has_author=True,
            has_publish_date=True,
            article_length=1000,
            writing_quality_score=0.8,
            is_state_managed=False
        )
        
        assert features.domain == "example.com"
        assert features.tld == "com"
        assert features.uses_https is True
        assert features.is_state_managed is False
    
    def test_features_to_dict(self):
        """Test converting features to dictionary."""
        features = CredibilityFeatures(
            domain="example.com",
            tld="com",
            uses_https=True,
            has_author=False,
            has_publish_date=False,
            article_length=500,
            writing_quality_score=0.5,
            is_state_managed=False
        )
        
        features_dict = features.to_dict()
        
        assert isinstance(features_dict, dict)
        assert features_dict['domain'] == "example.com"
        assert features_dict['uses_https'] is True
