"""Tests for cache manager."""

import sys
from pathlib import Path
import pytest
import tempfile
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cache_manager import CacheManager


class TestCacheManager:
    """Test cache manager functionality."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create a temporary cache manager for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        manager = CacheManager(db_path=db_path, default_ttl_hours=24)
        yield manager
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test that cache manager initializes correctly."""
        assert cache_manager.db_path.exists()
        assert cache_manager.default_ttl_hours == 24
    
    def test_search_cache_set_and_get(self, cache_manager):
        """Test cache insertion and retrieval for search results."""
        query = "Việt Nam có 54 dân tộc"
        language = "vi"
        results = [
            {"title": "Result 1", "url": "http://test1.com"},
            {"title": "Result 2", "url": "http://test2.com"}
        ]
        
        # Set cache
        cache_manager.set_search_results(query, language, results)
        
        # Get cache
        cached_results = cache_manager.get_search_results(query, language)
        
        assert cached_results is not None
        assert len(cached_results) == 2
        assert cached_results[0]['title'] == "Result 1"
        assert cached_results[1]['url'] == "http://test2.com"
    
    def test_search_cache_miss(self, cache_manager):
        """Test cache miss for non-existent search query."""
        result = cache_manager.get_search_results("nonexistent query", "vi")
        assert result is None
    
    def test_search_cache_different_languages(self, cache_manager):
        """Test that same query in different languages are cached separately."""
        query = "test query"
        results_vi = [{"title": "Vietnamese result"}]
        results_en = [{"title": "English result"}]
        
        cache_manager.set_search_results(query, "vi", results_vi)
        cache_manager.set_search_results(query, "en", results_en)
        
        cached_vi = cache_manager.get_search_results(query, "vi")
        cached_en = cache_manager.get_search_results(query, "en")
        
        assert cached_vi[0]['title'] == "Vietnamese result"
        assert cached_en[0]['title'] == "English result"
    
    def test_content_cache_set_and_get(self, cache_manager):
        """Test cache insertion and retrieval for web content."""
        url = "https://vnexpress.net/article123"
        content = {
            "title": "Test Article",
            "text": "Article content here",
            "author": "Test Author"
        }
        
        # Set cache
        cache_manager.set_content(url, content)
        
        # Get cache
        cached_content = cache_manager.get_content(url)
        
        assert cached_content is not None
        assert cached_content['title'] == "Test Article"
        assert cached_content['text'] == "Article content here"
        assert cached_content['author'] == "Test Author"
    
    def test_content_cache_miss(self, cache_manager):
        """Test cache miss for non-existent URL."""
        result = cache_manager.get_content("https://nonexistent.com/article")
        assert result is None
    
    def test_credibility_cache_set_and_get(self, cache_manager):
        """Test cache insertion and retrieval for credibility scores."""
        domain = "vnexpress.net"
        score = 0.85
        features = {
            "is_state_managed": True,
            "uses_https": True,
            "has_author": True
        }
        
        # Set cache
        cache_manager.set_credibility(domain, score, features)
        
        # Get cache
        cached_data = cache_manager.get_credibility(domain)
        
        assert cached_data is not None
        assert cached_data['score'] == 0.85
        assert cached_data['features']['is_state_managed'] is True
        assert cached_data['features']['uses_https'] is True
    
    def test_credibility_cache_miss(self, cache_manager):
        """Test cache miss for non-existent domain."""
        result = cache_manager.get_credibility("nonexistent.com")
        assert result is None
    
    def test_cache_expiration(self, cache_manager):
        """Test that expired cache entries are not returned."""
        # Create cache manager with very short TTL
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        short_ttl_manager = CacheManager(db_path=db_path, default_ttl_hours=0)
        
        try:
            query = "test query"
            results = [{"title": "Test"}]
            
            # Set cache with 0 hour TTL (expires immediately)
            short_ttl_manager.set_search_results(query, "vi", results, ttl_hours=0)
            
            # Wait a moment to ensure expiration
            time.sleep(0.1)
            
            # Try to get - should return None due to expiration
            cached = short_ttl_manager.get_search_results(query, "vi")
            assert cached is None
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_cache_update(self, cache_manager):
        """Test that updating cache replaces old values."""
        query = "test query"
        language = "vi"
        
        # Set initial cache
        results1 = [{"title": "Result 1"}]
        cache_manager.set_search_results(query, language, results1)
        
        # Update cache
        results2 = [{"title": "Result 2"}, {"title": "Result 3"}]
        cache_manager.set_search_results(query, language, results2)
        
        # Get cache - should have updated values
        cached = cache_manager.get_search_results(query, language)
        assert len(cached) == 2
        assert cached[0]['title'] == "Result 2"
    
    def test_clear_expired(self, cache_manager):
        """Test clearing expired cache entries."""
        # Add some entries with short TTL
        cache_manager.set_search_results("query1", "vi", [{"test": 1}], ttl_hours=0)
        cache_manager.set_content("http://test.com", {"test": 1}, ttl_hours=0)
        cache_manager.set_credibility("test.com", 0.5, {}, ttl_hours=0)
        
        # Add some entries with long TTL
        cache_manager.set_search_results("query2", "vi", [{"test": 2}], ttl_hours=24)
        
        # Wait for expiration
        time.sleep(0.1)
        
        # Clear expired
        counts = cache_manager.clear_expired()
        
        # Should have cleared the expired entries
        assert counts['search'] >= 1
        assert counts['content'] >= 1
        assert counts['credibility'] >= 1
        
        # Long TTL entry should still be there
        cached = cache_manager.get_search_results("query2", "vi")
        assert cached is not None
    
    def test_clear_all(self, cache_manager):
        """Test clearing all cache entries."""
        # Add some entries
        cache_manager.set_search_results("query1", "vi", [{"test": 1}])
        cache_manager.set_content("http://test.com", {"test": 1})
        cache_manager.set_credibility("test.com", 0.5, {})
        
        # Clear all
        cache_manager.clear_all()
        
        # All should be gone
        assert cache_manager.get_search_results("query1", "vi") is None
        assert cache_manager.get_content("http://test.com") is None
        assert cache_manager.get_credibility("test.com") is None
    
    def test_get_stats(self, cache_manager):
        """Test getting cache statistics."""
        # Add some entries
        cache_manager.set_search_results("query1", "vi", [{"test": 1}])
        cache_manager.set_search_results("query2", "en", [{"test": 2}])
        cache_manager.set_content("http://test1.com", {"test": 1})
        cache_manager.set_credibility("test.com", 0.5, {})
        
        # Get stats
        stats = cache_manager.get_stats()
        
        assert 'search' in stats
        assert 'content' in stats
        assert 'credibility' in stats
        
        assert stats['search']['total'] >= 2
        assert stats['content']['total'] >= 1
        assert stats['credibility']['total'] >= 1
        
        # All should be valid (not expired)
        assert stats['search']['valid'] >= 2
        assert stats['content']['valid'] >= 1
        assert stats['credibility']['valid'] >= 1
    
    def test_concurrent_access(self, cache_manager):
        """Test that cache handles concurrent access correctly."""
        import threading
        
        query = "concurrent test"
        errors = []
        
        def write_operation(thread_id):
            """Write operation for concurrent testing."""
            try:
                for i in range(5):
                    cache_manager.set_search_results(
                        f"{query}_{thread_id}_{i}", 
                        "vi", 
                        [{"thread": thread_id, "id": i}]
                    )
            except Exception as e:
                errors.append(f"Write error in thread {thread_id}: {e}")
        
        def read_operation(thread_id):
            """Read operation for concurrent testing."""
            try:
                for i in range(5):
                    result = cache_manager.get_search_results(
                        f"{query}_{thread_id}_{i}", 
                        "vi"
                    )
                    if result is not None:
                        assert result[0]['thread'] == thread_id
                        assert result[0]['id'] == i
            except Exception as e:
                errors.append(f"Read error in thread {thread_id}: {e}")
        
        # Create and start write threads
        write_threads = []
        for i in range(5):
            thread = threading.Thread(target=write_operation, args=(i,))
            write_threads.append(thread)
            thread.start()
        
        # Wait for all writes to complete
        for thread in write_threads:
            thread.join()
        
        # Create and start read threads
        read_threads = []
        for i in range(5):
            thread = threading.Thread(target=read_operation, args=(i,))
            read_threads.append(thread)
            thread.start()
        
        # Wait for all reads to complete
        for thread in read_threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        # Verify all data was written correctly
        for thread_id in range(5):
            for i in range(5):
                result = cache_manager.get_search_results(
                    f"{query}_{thread_id}_{i}", 
                    "vi"
                )
                assert result is not None
                assert result[0]['thread'] == thread_id
                assert result[0]['id'] == i
    
    def test_concurrent_read_write_same_key(self, cache_manager):
        """Test concurrent reads and writes to the same cache key."""
        import threading
        
        query = "shared_query"
        language = "vi"
        errors = []
        read_results = []
        
        def write_operation(value):
            """Write operation that updates the same key."""
            try:
                cache_manager.set_search_results(
                    query, 
                    language, 
                    [{"value": value}]
                )
            except Exception as e:
                errors.append(f"Write error: {e}")
        
        def read_operation():
            """Read operation that reads the same key."""
            try:
                result = cache_manager.get_search_results(query, language)
                if result is not None:
                    read_results.append(result[0]['value'])
            except Exception as e:
                errors.append(f"Read error: {e}")
        
        # Initial write
        cache_manager.set_search_results(query, language, [{"value": 0}])
        
        # Create mixed read/write threads
        threads = []
        for i in range(10):
            if i % 2 == 0:
                thread = threading.Thread(target=write_operation, args=(i,))
            else:
                thread = threading.Thread(target=read_operation)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        # Final value should be one of the written values
        final_result = cache_manager.get_search_results(query, language)
        assert final_result is not None
        assert final_result[0]['value'] in [0, 2, 4, 6, 8]
    
    def test_concurrent_different_cache_types(self, cache_manager):
        """Test concurrent access to different cache types."""
        import threading
        
        errors = []
        
        def search_operations():
            """Operations on search cache."""
            try:
                for i in range(10):
                    cache_manager.set_search_results(f"query_{i}", "vi", [{"id": i}])
                    result = cache_manager.get_search_results(f"query_{i}", "vi")
                    assert result is not None
            except Exception as e:
                errors.append(f"Search cache error: {e}")
        
        def content_operations():
            """Operations on content cache."""
            try:
                for i in range(10):
                    cache_manager.set_content(f"http://test{i}.com", {"id": i})
                    result = cache_manager.get_content(f"http://test{i}.com")
                    assert result is not None
            except Exception as e:
                errors.append(f"Content cache error: {e}")
        
        def credibility_operations():
            """Operations on credibility cache."""
            try:
                for i in range(10):
                    cache_manager.set_credibility(f"domain{i}.com", 0.5 + i * 0.01, {"id": i})
                    result = cache_manager.get_credibility(f"domain{i}.com")
                    assert result is not None
            except Exception as e:
                errors.append(f"Credibility cache error: {e}")
        
        # Create threads for different cache types
        threads = [
            threading.Thread(target=search_operations),
            threading.Thread(target=content_operations),
            threading.Thread(target=credibility_operations)
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
    
    def test_unicode_content(self, cache_manager):
        """Test that cache handles Unicode content correctly."""
        query = "Việt Nam có 54 dân tộc"
        results = [
            {"title": "Kết quả 1", "snippet": "Thông tin về dân tộc"},
            {"title": "Kết quả 2", "snippet": "Dữ liệu chính thức"}
        ]
        
        cache_manager.set_search_results(query, "vi", results)
        cached = cache_manager.get_search_results(query, "vi")
        
        assert cached[0]['title'] == "Kết quả 1"
        assert cached[0]['snippet'] == "Thông tin về dân tộc"
    
    def test_large_content(self, cache_manager):
        """Test that cache handles large content correctly."""
        url = "http://test.com/large-article"
        large_text = "Lorem ipsum " * 10000  # Large text
        content = {
            "title": "Large Article",
            "text": large_text
        }
        
        cache_manager.set_content(url, content)
        cached = cache_manager.get_content(url)
        
        assert cached is not None
        assert len(cached['text']) == len(large_text)
        assert cached['text'] == large_text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
