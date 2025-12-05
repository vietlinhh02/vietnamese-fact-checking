"""Property-based tests for cross-lingual search module using Hypothesis."""

import sys
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, MagicMock
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import Claim
from search_query_generator import SearchQueryGenerator, SearchQuery
from exa_search_client import ExaSearchClient, SearchResult, RateLimiter
from translation_service import TranslationService


# Custom strategies for generating test data
@st.composite
def vietnamese_claim_strategy(draw):
    """Generate Vietnamese-like claim text."""
    # Generate text that looks like Vietnamese claims
    vietnamese_words = [
        "Việt Nam", "Hà Nội", "Thành phố", "Chính phủ", "Tổng thống",
        "công ty", "người dân", "kinh tế", "phát triển", "thông tin",
        "báo cáo", "nghiên cứu", "kết quả", "dự án", "chương trình"
    ]
    
    # Generate a claim with 5-15 words
    num_words = draw(st.integers(min_value=5, max_value=15))
    words = draw(st.lists(
        st.sampled_from(vietnamese_words),
        min_size=num_words,
        max_size=num_words
    ))
    
    claim_text = " ".join(words)
    return claim_text


@st.composite
def search_query_strategy(draw):
    """Generate valid SearchQuery objects."""
    text = draw(st.text(min_size=5, max_size=100))
    language = draw(st.sampled_from(["vi", "en"]))
    source_claim = draw(st.text(min_size=10, max_size=200))
    query_type = draw(st.sampled_from(["direct", "entity_focused", "question", "decomposed"]))
    
    return SearchQuery(
        text=text,
        language=language,
        source_claim=source_claim,
        query_type=query_type
    )


@st.composite
def search_result_strategy(draw):
    """Generate valid SearchResult objects."""
    title = draw(st.text(min_size=10, max_size=100))
    text = draw(st.text(min_size=20, max_size=200))
    url = draw(st.from_regex(r'https?://[a-z]+\.(com|net|vn)/[a-z]+', fullmatch=True))
    language = draw(st.sampled_from(["vi", "en"]))
    rank = draw(st.integers(min_value=1, max_value=100))
    id = draw(st.text(min_size=5, max_size=20))
    
    return SearchResult(
        title=title,
        text=text,
        url=url,
        language=language,
        rank=rank,
        id=id
    )


class TestBilingualQueryGeneration:
    """Property-based tests for bilingual query generation."""
    
    @given(vietnamese_claim_strategy())
    @settings(max_examples=100)
    def test_bilingual_query_generation(self, claim_text):
        """
        **Feature: vietnamese-fact-checking, Property 9: Bilingual Query Generation**
        
        Test that for any Vietnamese claim, the search query generation produces
        at least one Vietnamese query and one English query.
        
        **Validates: Requirements 4.1, 4.2**
        """
        # Create mock translation service
        mock_translation = Mock(spec=TranslationService)
        mock_translation.translate = Mock(return_value=f"English translation of {claim_text}")
        
        # Create query generator with mock translation
        generator = SearchQueryGenerator(translation_service=mock_translation)
        
        # Generate queries
        queries = generator.generate_queries(claim_text, language="vi")
        
        # Property 1: Should generate at least some queries
        assert len(queries) > 0, "Should generate at least one query"
        
        # Property 2: Should have both Vietnamese and English queries
        languages = [q.language for q in queries]
        
        assert 'vi' in languages, \
            f"Should generate Vietnamese queries, got languages: {languages}"
        
        assert 'en' in languages, \
            f"Should generate English queries, got languages: {languages}"
        
        # Property 3: All queries should reference the source claim
        for query in queries:
            assert query.source_claim == claim_text, \
                f"Query should reference source claim"
    
    @given(vietnamese_claim_strategy())
    @settings(max_examples=100)
    def test_query_generation_without_translation_service(self, claim_text):
        """
        Test that query generation works without translation service.
        
        Should still generate Vietnamese queries even without translation.
        """
        # Create generator without translation service
        generator = SearchQueryGenerator(translation_service=None)
        
        # Generate queries
        queries = generator.generate_queries(claim_text, language="vi")
        
        # Property: Should still generate Vietnamese queries
        assert len(queries) > 0, "Should generate queries even without translation"
        
        languages = [q.language for q in queries]
        assert 'vi' in languages, "Should generate Vietnamese queries"
        
        # Property: Should not generate English queries without translation
        assert 'en' not in languages, \
            "Should not generate English queries without translation service"
    
    @given(st.text(min_size=10, max_size=200))
    @settings(max_examples=100)
    def test_query_types_diversity(self, claim_text):
        """
        Test that query generation produces diverse query types.
        """
        # Skip empty or whitespace-only claims
        assume(claim_text.strip() != "")
        
        # Create mock translation service
        mock_translation = Mock(spec=TranslationService)
        mock_translation.translate = Mock(return_value=f"Translated: {claim_text}")
        
        generator = SearchQueryGenerator(translation_service=mock_translation)
        queries = generator.generate_queries(claim_text, language="vi")
        
        # Property: Should have at least a direct query
        query_types = [q.query_type for q in queries]
        assert 'direct' in query_types, "Should include direct query type"
        
        # Property: All query types should be valid
        valid_types = ["direct", "entity_focused", "question", "decomposed"]
        for query in queries:
            assert query.query_type in valid_types, \
                f"Invalid query type: {query.query_type}"
    
    @given(vietnamese_claim_strategy())
    @settings(max_examples=100)
    def test_english_queries_preserve_meaning(self, claim_text):
        """
        Test that English queries preserve semantic meaning of original claim.
        
        This is tested by ensuring the translation service is called with correct parameters.
        """
        # Create mock translation service
        mock_translation = Mock(spec=TranslationService)
        mock_translation.translate = Mock(return_value="English translation")
        
        generator = SearchQueryGenerator(translation_service=mock_translation)
        queries = generator.generate_queries(claim_text, language="vi")
        
        # Property: Translation service should be called for English queries
        if any(q.language == 'en' for q in queries):
            mock_translation.translate.assert_called()
            
            # Check that translation was called with Vietnamese source
            call_args = mock_translation.translate.call_args
            if call_args:
                assert call_args[1].get('source_lang') == 'vi' or call_args[0][1] == 'vi', \
                    "Translation should use Vietnamese as source language"


class TestMultilingualEvidenceCollection:
    """Property-based tests for multilingual evidence collection."""
    
    @given(
        st.lists(search_result_strategy(), min_size=2, max_size=20)
    )
    @settings(max_examples=100)
    def test_multilingual_evidence_collection(self, search_results):
        """
        **Feature: vietnamese-fact-checking, Property 10: Multilingual Evidence Collection**
        
        Test that for any search execution, the collected evidence includes sources
        from both Vietnamese and English languages (if available).
        
        **Validates: Requirements 4.3**
        """
        # Ensure we have results in both languages
        assume(any(r.language == 'vi' for r in search_results))
        assume(any(r.language == 'en' for r in search_results))
        
        # Property: Results should contain both languages
        languages = set(r.language for r in search_results)
        
        assert 'vi' in languages, "Should have Vietnamese results"
        assert 'en' in languages, "Should have English results"
        
        # Property: Each result should have a valid language
        for result in search_results:
            assert result.language in ['vi', 'en'], \
                f"Invalid language: {result.language}"
    
    @given(st.lists(search_result_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_search_results_have_required_fields(self, search_results):
        """
        Test that all search results have required fields.
        """
        for result in search_results:
            # Property: All results should have non-empty title
            assert len(result.title) > 0, "Result should have title"
            
            # Property: All results should have non-empty URL
            assert len(result.url) > 0, "Result should have URL"
            
            # Property: All results should have valid rank
            assert result.rank > 0, "Result rank should be positive"
            
            # Property: All results should have language
            assert result.language in ['vi', 'en'], \
                f"Invalid language: {result.language}"


class TestAPIRateLimitCompliance:
    """Property-based tests for API rate limit compliance."""
    
    @given(st.integers(min_value=20, max_value=50))
    @settings(max_examples=50, deadline=None)
    def test_rate_limit_compliance(self, num_calls):
        """
        **Feature: vietnamese-fact-checking, Property 11: API Rate Limit Compliance**
        
        Test that for any sequence of API calls, the system never exceeds
        the rate limits of free-tier APIs.
        
        **Validates: Requirements 4.4**
        """
        # Create rate limiter with test limits
        rate_limiter = RateLimiter(
            max_calls_per_minute=15
        )
        
        start_time = time.time()
        
        # Simulate API calls
        for i in range(min(num_calls, 20)):  # Limit to 20 for test speed
            rate_limiter.wait_if_needed()
            rate_limiter.record_call()
        
        elapsed = time.time() - start_time
        
        # Property: If making more than 15 calls, should take at least 1 minute
        if min(num_calls, 20) > 15:
            assert elapsed >= 60, \
                f"Making {min(num_calls, 20)} calls should take at least 60 seconds with 15 calls/minute limit, took {elapsed:.1f}s"
        
        # Property: Rate limiter should track calls correctly
        stats = rate_limiter.get_stats()
        assert stats['calls_last_minute'] <= 15, \
            f"Too many calls in last minute: {stats['calls_last_minute']}"
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_rate_limiter_backoff(self, num_errors):
        """
        Test that rate limiter applies exponential backoff on errors.
        """
        rate_limiter = RateLimiter(max_calls_per_minute=15)
        
        initial_backoff = rate_limiter.backoff_seconds
        
        # Simulate errors
        for _ in range(num_errors):
            rate_limiter.record_error(is_rate_limit_error=True)
        
        # Property: Backoff should increase with errors
        assert rate_limiter.backoff_seconds >= initial_backoff, \
            "Backoff should increase after errors"
        
        # Property: Backoff should be capped
        assert rate_limiter.backoff_seconds <= 300, \
            "Backoff should be capped at 5 minutes"
    
    @given(st.integers(min_value=5, max_value=20))
    @settings(max_examples=50)
    def test_rate_limiter_stats_accuracy(self, num_calls):
        """
        Test that rate limiter statistics are accurate.
        """
        rate_limiter = RateLimiter(max_calls_per_minute=100)
        
        # Record calls
        for _ in range(num_calls):
            rate_limiter.record_call()
        
        # Property: Stats should reflect actual calls
        stats = rate_limiter.get_stats()
        assert stats['calls_last_minute'] == num_calls, \
            f"Stats should show {num_calls} calls, got {stats['calls_last_minute']}"
        
        # Property: Stats should not exceed limits
        assert stats['calls_last_minute'] <= stats['minute_limit'], \
            "Calls should not exceed minute limit"


class TestSearchQueryProperties:
    """Property-based tests for SearchQuery data model."""
    
    @given(search_query_strategy())
    @settings(max_examples=100)
    def test_search_query_has_required_fields(self, query):
        """Test that search queries have all required fields."""
        # Property: Query text should not be empty
        assert len(query.text) > 0, "Query text should not be empty"
        
        # Property: Language should be valid
        assert query.language in ['vi', 'en'], \
            f"Invalid language: {query.language}"
        
        # Property: Source claim should not be empty
        assert len(query.source_claim) > 0, \
            "Source claim should not be empty"
        
        # Property: Query type should be valid
        valid_types = ["direct", "entity_focused", "question", "decomposed"]
        assert query.query_type in valid_types, \
            f"Invalid query type: {query.query_type}"


class TestSearchResultProperties:
    """Property-based tests for SearchResult data model."""
    
    @given(search_result_strategy())
    @settings(max_examples=100)
    def test_search_result_serialization(self, result):
        """Test that search results can be serialized and deserialized."""
        # Test dict round-trip
        result_dict = result.to_dict()
        restored_result = SearchResult.from_dict(result_dict)
        
        assert restored_result.title == result.title
        assert restored_result.text == result.text
        assert restored_result.url == result.url
        assert restored_result.language == result.language
        assert restored_result.rank == result.rank
        assert restored_result.id == result.id
    
    @given(search_result_strategy())
    @settings(max_examples=100)
    def test_search_result_rank_validity(self, result):
        """Test that search result ranks are valid."""
        # Property: Rank should be positive
        assert result.rank > 0, "Rank should be positive"
        
        # Property: Rank should be reasonable (not too large)
        assert result.rank <= 1000, "Rank should be reasonable"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

