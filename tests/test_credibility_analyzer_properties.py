"""Property-based tests for credibility analyzer module."""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from src.credibility_analyzer import (
    CredibilityAnalyzer,
    CredibilityFeatureExtractor,
    RuleBasedCredibilityScorer,
    CredibilityFeatures,
    analyze_credibility
)
from src.web_crawler import WebContent


# Custom strategies for generating test data
@composite
def web_content_strategy(draw):
    """Generate random WebContent objects for testing."""
    # Generate domain from approved sources or random domains
    approved_sources = [
        "vnexpress.net",
        "vtv.vn",
        "vov.vn",
        "tuoitre.vn",
        "thanhnien.vn",
        "baochinhphu.vn"
    ]
    
    other_domains = [
        "example.com",
        "random-blog.net",
        "news-site.org",
        "info.vn",
        "test.com.vn"
    ]
    
    # Mix of approved and non-approved sources
    all_domains = approved_sources + other_domains
    domain = draw(st.sampled_from(all_domains))
    
    # Generate URL with or without HTTPS
    protocol = draw(st.sampled_from(['https', 'http']))
    url = f"{protocol}://{domain}/article-{draw(st.integers(min_value=1, max_value=10000))}"
    
    # Generate title
    title = draw(st.text(min_size=10, max_size=200))
    
    # Generate main text
    main_text = draw(st.text(min_size=100, max_size=5000))
    
    # Generate optional author
    has_author = draw(st.booleans())
    author = draw(st.text(min_size=5, max_size=50)) if has_author else None
    
    # Generate optional publish date
    has_date = draw(st.booleans())
    publish_date = "2024-01-15T10:30:00" if has_date else None
    
    return WebContent(
        url=url,
        title=title,
        main_text=main_text,
        author=author,
        publish_date=publish_date,
        extraction_success=True
    )


@composite
def credibility_features_strategy(draw):
    """Generate random CredibilityFeatures objects for testing."""
    approved_sources = [
        "vnexpress.net",
        "vtv.vn",
        "vov.vn",
        "tuoitre.vn",
        "thanhnien.vn",
        "baochinhphu.vn"
    ]
    
    other_domains = [
        "example.com",
        "random-blog.net",
        "news-site.org"
    ]
    
    all_domains = approved_sources + other_domains
    domain = draw(st.sampled_from(all_domains))
    
    is_state_managed = domain in approved_sources
    
    tld = domain.split('.')[-1]
    
    return CredibilityFeatures(
        domain=domain,
        tld=tld,
        domain_age_days=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=7300))),
        uses_https=draw(st.booleans()),
        has_author=draw(st.booleans()),
        has_publish_date=draw(st.booleans()),
        article_length=draw(st.integers(min_value=0, max_value=10000)),
        writing_quality_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        is_state_managed=is_state_managed,
        mbfc_rating=draw(st.one_of(
            st.none(),
            st.sampled_from(['Very High', 'High', 'Mostly Factual', 'Mixed', 'Low', 'Very Low'])
        ))
    )


class TestCredibilityScoreRange:
    """
    Property 12: Credibility Score Existence
    
    **Feature: vietnamese-fact-checking, Property 12: Credibility Score Existence**
    **Validates: Requirements 5.1, 5.5**
    
    For any piece of evidence collected, there should exist an associated credibility score
    in the range [0, 1], computed from source features and external signals.
    """
    
    @given(web_content_strategy())
    @settings(max_examples=100)
    def test_credibility_score_exists_and_in_range(self, content):
        """Test that credibility scores exist and are in valid range [0, 1]."""
        # Analyze credibility
        score = analyze_credibility(content)
        
        # Property: Score must exist and be in [0, 1]
        assert score is not None, "Credibility score must exist"
        assert score.overall_score is not None, "Overall score must exist"
        assert 0 <= score.overall_score <= 1, \
            f"Overall score must be in [0, 1], got {score.overall_score}"
        
        # Property: Confidence must be in [0, 1]
        assert 0 <= score.confidence <= 1, \
            f"Confidence must be in [0, 1], got {score.confidence}"
    
    @given(credibility_features_strategy())
    @settings(max_examples=100)
    def test_scorer_produces_valid_range(self, features):
        """Test that scorer always produces scores in valid range."""
        scorer = RuleBasedCredibilityScorer()
        score = scorer.compute_score(features)
        
        # Property: Score must be in [0, 1]
        assert 0 <= score.overall_score <= 1, \
            f"Score must be in [0, 1], got {score.overall_score}"
        
        # Property: All feature scores must be non-negative
        for feature_name, feature_score in score.feature_scores.items():
            assert feature_score >= 0, \
                f"Feature score for {feature_name} must be non-negative, got {feature_score}"
    
    @given(web_content_strategy())
    @settings(max_examples=100)
    def test_feature_extraction_always_succeeds(self, content):
        """Test that feature extraction always produces valid features."""
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        # Property: Features must exist
        assert features is not None, "Features must exist"
        assert features.domain is not None, "Domain must exist"
        assert features.tld is not None, "TLD must exist"
        
        # Property: Boolean features must be boolean
        assert isinstance(features.uses_https, bool)
        assert isinstance(features.has_author, bool)
        assert isinstance(features.has_publish_date, bool)
        assert isinstance(features.is_state_managed, bool)
        
        # Property: Numeric features must be in valid ranges
        assert features.article_length >= 0, "Article length must be non-negative"
        assert 0 <= features.writing_quality_score <= 1, \
            f"Writing quality score must be in [0, 1], got {features.writing_quality_score}"
    
    @given(web_content_strategy())
    @settings(max_examples=100)
    def test_explanation_exists(self, content):
        """Test that credibility score always includes an explanation."""
        score = analyze_credibility(content)
        
        # Property: Explanation must exist
        assert score.explanation is not None, "Explanation must exist"
        assert isinstance(score.explanation, str), "Explanation must be a string"


class TestStateManagedSourcePriority:
    """
    Property 13: State-Managed Source Priority
    
    **Feature: vietnamese-fact-checking, Property 13: State-Managed Source Priority**
    **Validates: Requirements 5.2**
    
    For any two evidence sources where one is Vietnamese state-managed and the other is not,
    the state-managed source should receive a higher credibility score (difference >= 0.2).
    """
    
    @given(
        st.sampled_from([
            "vnexpress.net",
            "vtv.vn",
            "vov.vn",
            "tuoitre.vn",
            "thanhnien.vn",
            "baochinhphu.vn"
        ]),
        st.sampled_from([
            "example.com",
            "random-blog.net",
            "news-site.org",
            "test.info"
        ])
    )
    @settings(max_examples=100)
    def test_state_managed_higher_score(self, state_domain, other_domain):
        """Test that state-managed sources get higher scores than non-state sources."""
        # Create content from state-managed source
        state_content = WebContent(
            url=f"https://{state_domain}/article-1",
            title="Test Article",
            main_text="This is a test article with sufficient length to pass quality checks. " * 10,
            author="Test Author",
            publish_date="2024-01-15T10:30:00",
            extraction_success=True
        )
        
        # Create content from non-state source with same features
        other_content = WebContent(
            url=f"https://{other_domain}/article-1",
            title="Test Article",
            main_text="This is a test article with sufficient length to pass quality checks. " * 10,
            author="Test Author",
            publish_date="2024-01-15T10:30:00",
            extraction_success=True
        )
        
        # Analyze credibility
        state_score = analyze_credibility(state_content)
        other_score = analyze_credibility(other_content)
        
        # Property: State-managed source should have higher score
        score_difference = state_score.overall_score - other_score.overall_score
        assert score_difference >= 0.2, \
            f"State-managed source should have at least 0.2 higher score, got difference of {score_difference:.3f}"
    
    @given(
        st.sampled_from([
            "vnexpress.net",
            "vtv.vn",
            "vov.vn",
            "tuoitre.vn",
            "thanhnien.vn",
            "baochinhphu.vn"
        ])
    )
    @settings(max_examples=100)
    def test_state_managed_detection(self, state_domain):
        """Test that state-managed sources are correctly identified."""
        content = WebContent(
            url=f"https://{state_domain}/article-1",
            title="Test Article",
            main_text="Test content",
            extraction_success=True
        )
        
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        # Property: State-managed sources must be detected
        assert features.is_state_managed, \
            f"Domain {state_domain} should be detected as state-managed"
    
    @given(
        st.sampled_from([
            "example.com",
            "random-blog.net",
            "news-site.org",
            "test.info",
            "blog.wordpress.com"
        ])
    )
    @settings(max_examples=100)
    def test_non_state_managed_detection(self, other_domain):
        """Test that non-state sources are correctly identified."""
        content = WebContent(
            url=f"https://{other_domain}/article-1",
            title="Test Article",
            main_text="Test content",
            extraction_success=True
        )
        
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        # Property: Non-state sources must not be detected as state-managed
        assert not features.is_state_managed, \
            f"Domain {other_domain} should not be detected as state-managed"
    
    @given(credibility_features_strategy())
    @settings(max_examples=100)
    def test_state_managed_feature_contributes_to_score(self, features):
        """Test that state-managed feature contributes positively to score."""
        scorer = RuleBasedCredibilityScorer()
        
        # Compute score with current features
        score = scorer.compute_score(features)
        
        # If state-managed, check that it contributes to the score
        if features.is_state_managed:
            assert 'state_managed' in score.feature_scores
            assert score.feature_scores['state_managed'] > 0, \
                "State-managed feature should contribute positively to score"
        else:
            # If not state-managed, the contribution should be 0
            if 'state_managed' in score.feature_scores:
                assert score.feature_scores['state_managed'] == 0, \
                    "Non-state-managed sources should have 0 contribution from state_managed feature"


class TestCredibilityFeatureConsistency:
    """Additional property tests for feature consistency."""
    
    @given(web_content_strategy())
    @settings(max_examples=100)
    def test_https_detection_consistency(self, content):
        """Test that HTTPS detection is consistent with URL."""
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        # Property: HTTPS feature must match URL protocol
        if content.url.startswith('https://'):
            assert features.uses_https, "HTTPS feature must be True for https:// URLs"
        else:
            assert not features.uses_https, "HTTPS feature must be False for non-https URLs"
    
    @given(web_content_strategy())
    @settings(max_examples=100)
    def test_author_detection_consistency(self, content):
        """Test that author detection is consistent with content."""
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        # Property: has_author must match presence of author in content
        if content.author and len(content.author.strip()) > 0:
            assert features.has_author, "has_author must be True when author is present"
        else:
            assert not features.has_author, "has_author must be False when author is absent"
    
    @given(web_content_strategy())
    @settings(max_examples=100)
    def test_article_length_consistency(self, content):
        """Test that article length matches actual text length."""
        extractor = CredibilityFeatureExtractor()
        features = extractor.extract_features(content)
        
        # Property: article_length must match actual text length
        assert features.article_length == len(content.main_text), \
            f"Article length {features.article_length} must match text length {len(content.main_text)}"
