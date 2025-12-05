"""Unit tests for web crawler module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.web_crawler import StaticHTMLCrawler, WebCrawler, WebContent
from src.config import SearchConfig


class TestStaticHTMLCrawlerErrorHandling:
    """Unit tests for error handling in StaticHTMLCrawler."""
    
    def test_http_404_error(self):
        """Test handling of HTTP 404 errors."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            result = crawler.fetch_html("https://vnexpress.net/nonexistent")
            
            assert result is None
    
    def test_http_403_error(self):
        """Test handling of HTTP 403 Forbidden errors."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
            mock_get.return_value = mock_response
            
            result = crawler.fetch_html("https://vnexpress.net/forbidden")
            
            assert result is None
    
    def test_http_500_error(self):
        """Test handling of HTTP 500 Internal Server Error."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")
            mock_get.return_value = mock_response
            
            result = crawler.fetch_html("https://vnexpress.net/error")
            
            assert result is None
    
    def test_timeout_error(self):
        """Test handling of timeout errors."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
            
            result = crawler.fetch_html("https://vnexpress.net/slow")
            
            assert result is None
    
    def test_connection_error(self):
        """Test handling of connection errors."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            result = crawler.fetch_html("https://vnexpress.net/unreachable")
            
            assert result is None
    
    def test_encoding_utf8(self):
        """Test handling of UTF-8 encoded content."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.encoding = 'utf-8'
            mock_response.text = "Tiếng Việt content"
            mock_get.return_value = mock_response
            
            result = crawler.fetch_html("https://vnexpress.net/article")
            
            assert result == "Tiếng Việt content"
    
    def test_encoding_fallback(self):
        """Test fallback encoding when encoding is not specified."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.encoding = None
            mock_response.content = "Test content".encode('utf-8')
            mock_get.return_value = mock_response
            
            result = crawler.fetch_html("https://vnexpress.net/article")
            
            assert result == "Test content"
    
    def test_encoding_error_handling(self):
        """Test handling of encoding errors with replacement."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.encoding = None
            # Invalid UTF-8 sequence
            mock_response.content = b'\xff\xfe Invalid UTF-8'
            mock_get.return_value = mock_response
            
            result = crawler.fetch_html("https://vnexpress.net/article")
            
            # Should not raise exception, should use replacement characters
            assert result is not None
            assert isinstance(result, str)
    
    def test_non_approved_source_rejection(self):
        """Test that non-approved sources are rejected."""
        crawler = StaticHTMLCrawler()
        
        result = crawler.fetch_html("https://untrusted-source.com/article")
        
        assert result is None
    
    def test_extraction_failure_returns_none(self):
        """Test that extraction failures return None."""
        crawler = StaticHTMLCrawler()
        
        with patch('trafilatura.extract') as mock_extract:
            mock_extract.return_value = None
            
            result = crawler.extract_content("<html></html>", "https://vnexpress.net/test")
            
            assert result is None
    
    def test_extraction_exception_handling(self):
        """Test that extraction exceptions are handled gracefully."""
        crawler = StaticHTMLCrawler()
        
        with patch('trafilatura.extract') as mock_extract:
            mock_extract.side_effect = Exception("Extraction error")
            
            result = crawler.extract_content("<html></html>", "https://vnexpress.net/test")
            
            assert result is None
    
    def test_short_content_marked_as_failure(self):
        """Test that very short extracted content is marked as extraction failure."""
        crawler = StaticHTMLCrawler()
        
        with patch('trafilatura.extract') as mock_extract:
            with patch('trafilatura.extract_metadata') as mock_metadata:
                mock_extract.return_value = "Short"  # Less than 100 chars
                mock_metadata.return_value = Mock(
                    title="Test",
                    author=None,
                    date=None,
                    sitename=None,
                    description=None,
                    categories=None,
                    tags=None
                )
                
                result = crawler.extract_content("<html></html>", "https://vnexpress.net/test")
                
                assert result is not None
                assert not result.extraction_success
                assert len(result.main_text) < 100


class TestWebCrawlerIntegration:
    """Integration tests for WebCrawler."""
    
    def test_crawl_with_static_crawler(self):
        """Test crawling with static crawler."""
        crawler = WebCrawler(use_selenium=False)
        
        with patch.object(crawler.static_crawler, 'crawl') as mock_crawl:
            mock_crawl.return_value = WebContent(
                url="https://vnexpress.net/test",
                title="Test Article",
                main_text="Test content" * 50,
                extraction_success=True
            )
            
            result = crawler.crawl("https://vnexpress.net/test")
            
            assert result is not None
            assert result.title == "Test Article"
            assert result.extraction_success
    
    def test_is_approved_source(self):
        """Test approved source checking."""
        crawler = WebCrawler()
        
        assert crawler.is_approved_source("https://vnexpress.net/article")
        assert crawler.is_approved_source("https://vtv.vn/news")
        assert not crawler.is_approved_source("https://untrusted.com/fake")
    
    def test_crawl_error_propagation(self):
        """Test that crawl errors are properly propagated."""
        crawler = WebCrawler(use_selenium=False)
        
        with patch.object(crawler.static_crawler, 'crawl') as mock_crawl:
            mock_crawl.return_value = None
            
            result = crawler.crawl("https://vnexpress.net/error")
            
            assert result is None


class TestRateLimiting:
    """Tests for rate limiting functionality."""
    
    def test_rate_limiting_enforced(self):
        """Test that rate limiting delays requests."""
        crawler = StaticHTMLCrawler()
        crawler.min_request_interval = 0.1  # 100ms for testing
        
        import time
        
        # First request should not wait
        start = time.time()
        crawler._apply_rate_limit("test.com")
        first_duration = time.time() - start
        
        # Second request should wait
        start = time.time()
        crawler._apply_rate_limit("test.com")
        second_duration = time.time() - start
        
        # Second request should take at least min_request_interval
        assert second_duration >= crawler.min_request_interval * 0.9  # Allow 10% tolerance
    
    def test_rate_limiting_per_domain(self):
        """Test that rate limiting is applied per domain."""
        crawler = StaticHTMLCrawler()
        crawler.min_request_interval = 0.1
        
        import time
        
        # Request to domain1
        crawler._apply_rate_limit("domain1.com")
        
        # Immediate request to domain2 should not wait
        start = time.time()
        crawler._apply_rate_limit("domain2.com")
        duration = time.time() - start
        
        # Should be nearly instant (no rate limiting across domains)
        assert duration < 0.05


class TestRobotsTxt:
    """Tests for robots.txt compliance."""
    
    def test_robots_txt_caching(self):
        """Test that robots.txt is cached."""
        crawler = StaticHTMLCrawler()
        
        with patch('urllib.robotparser.RobotFileParser.read') as mock_read:
            # First call
            crawler._get_robots_parser("https://vnexpress.net/article1")
            # Second call to same domain
            crawler._get_robots_parser("https://vnexpress.net/article2")
            
            # Should only read once (cached)
            assert mock_read.call_count == 1
    
    def test_robots_txt_failure_handling(self):
        """Test handling of robots.txt fetch failures."""
        crawler = StaticHTMLCrawler()
        
        with patch('urllib.robotparser.RobotFileParser.read') as mock_read:
            mock_read.side_effect = Exception("Network error")
            
            # Should not raise exception
            parser = crawler._get_robots_parser("https://vnexpress.net/article")
            
            # Should cache None to avoid repeated failures
            assert parser is None
            assert "https://vnexpress.net" in crawler.robots_cache
    
    def test_can_fetch_without_robots_txt(self):
        """Test that fetching is allowed when robots.txt is unavailable."""
        crawler = StaticHTMLCrawler()
        
        with patch.object(crawler, '_get_robots_parser') as mock_parser:
            mock_parser.return_value = None
            
            result = crawler._can_fetch("https://vnexpress.net/article", "TestBot")
            
            # Should allow fetching when robots.txt unavailable
            assert result is True


class TestUserAgentRotation:
    """Tests for user agent rotation."""
    
    def test_user_agent_rotation(self):
        """Test that user agents are rotated."""
        crawler = StaticHTMLCrawler()
        
        user_agents = set()
        for _ in range(len(crawler.USER_AGENTS) * 2):
            ua = crawler._get_user_agent()
            user_agents.add(ua)
        
        # Should have seen all user agents
        assert len(user_agents) == len(crawler.USER_AGENTS)
    
    def test_user_agent_cycle(self):
        """Test that user agent rotation cycles through list."""
        crawler = StaticHTMLCrawler()
        
        first_ua = crawler._get_user_agent()
        
        # Get all user agents
        for _ in range(len(crawler.USER_AGENTS) - 1):
            crawler._get_user_agent()
        
        # Next one should be the first again
        cycled_ua = crawler._get_user_agent()
        
        assert cycled_ua == first_ua
