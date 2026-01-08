"""Exa Search API client with rate limiting and caching."""

import logging
import time
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    id: str
    score: Optional[float] = None
    published_date: Optional[str] = None
    author: Optional[str] = None
    text: Optional[str] = None
    highlights: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    language: Optional[str] = None  # Added for compatibility/tracking
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from dictionary."""
        return cls(**data)


class RateLimiter:
    """
    Rate limiter for API calls.
    
    Implements sliding window rate limiting with exponential backoff.
    """
    
    def __init__(self, max_calls_per_minute: int = 15):
        """
        Initialize rate limiter.
        
        Args:
            max_calls_per_minute: Maximum API calls per minute
        """
        self.max_calls_per_minute = max_calls_per_minute
        
        # Track call timestamps
        self.minute_calls = deque()  # Timestamps of calls in last minute
        
        # Backoff state
        self.backoff_until = None
        self.backoff_seconds = 1
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = datetime.now()
        
        # Check if in backoff period
        if self.backoff_until and now < self.backoff_until:
            wait_seconds = (self.backoff_until - now).total_seconds()
            logger.info(f"Rate limiter: backing off for {wait_seconds:.1f} seconds")
            time.sleep(wait_seconds)
            self.backoff_until = None
            now = datetime.now()  # Update time after waiting
        
        # Clean old timestamps
        self._clean_old_timestamps()
        
        # Check minute limit
        if len(self.minute_calls) >= self.max_calls_per_minute:
            oldest_call = self.minute_calls[0]
            wait_until = oldest_call + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()
            
            if wait_seconds > 0:
                logger.info(f"Rate limiter: minute limit reached, waiting {wait_seconds:.1f} seconds")
                time.sleep(wait_seconds)
                now = datetime.now()  # Update time after waiting
                self._clean_old_timestamps()
    
    def record_call(self) -> None:
        """Record an API call."""
        now = datetime.now()
        self.minute_calls.append(now)
    
    def record_error(self, is_rate_limit_error: bool = False) -> None:
        """
        Record an API error and apply backoff.
        
        Args:
            is_rate_limit_error: Whether the error was a rate limit error
        """
        if is_rate_limit_error:
            # Exponential backoff
            self.backoff_seconds = min(self.backoff_seconds * 2, 300)  # Max 5 minutes
            self.backoff_until = datetime.now() + timedelta(seconds=self.backoff_seconds)
            logger.warning(f"Rate limit error: backing off for {self.backoff_seconds} seconds")
        else:
            # Linear backoff for other errors
            self.backoff_seconds = min(self.backoff_seconds + 1, 60)  # Max 1 minute
            self.backoff_until = datetime.now() + timedelta(seconds=self.backoff_seconds)
    
    def reset_backoff(self) -> None:
        """Reset backoff state after successful call."""
        self.backoff_seconds = 1
        self.backoff_until = None
    
    def _clean_old_timestamps(self) -> None:
        """Remove timestamps older than tracking window."""
        now = datetime.now()
        
        # Clean minute window
        while self.minute_calls and (now - self.minute_calls[0]) > timedelta(minutes=1):
            self.minute_calls.popleft()
    
    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics."""
        self._clean_old_timestamps()
        return {
            'calls_last_minute': len(self.minute_calls),
            'minute_limit': self.max_calls_per_minute
        }


class ExaSearchClient:
    """
    Exa Search API client.
    
    Features:
    - Semantic search (neural/auto/deep)
    - Content extraction (text, highlights, summary)
    - Result caching via CacheManager
    - Rate limiting and automatic retry
    """
    
    def __init__(
        self,
        api_key: str,
        cache_manager=None,
        max_results: int = 10,
        search_type: str = "auto",
        use_context: bool = True,
        rate_limit_rpm: int = 15
    ):
        """
        Initialize Exa Search client.
        
        Args:
            api_key: Exa API key
            cache_manager: CacheManager instance for result caching
            max_results: Maximum results per query
            search_type: Search type ('neural', 'auto', 'deep')
            use_context: Whether to request context for RAG
            rate_limit_rpm: Rate limit (requests per minute)
        """
        self.api_key = api_key
        self.cache_manager = cache_manager
        self.max_results = max_results
        self.search_type = search_type
        self.use_context = use_context
        
        # API endpoint
        self.base_url = "https://api.exa.ai/search"
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_calls_per_minute=rate_limit_rpm
        )
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'api_errors': 0,
            'rate_limit_errors': 0
        }
    
    def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_cache: bool = True,
        max_retries: int = 3,
        language: str = "vi"
    ) -> List[SearchResult]:
        """
        Execute a search query.
        
        Args:
            query: Search query text
            num_results: Number of results to return (overrides default)
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            start_published_date: ISO 8601 date string
            end_published_date: ISO 8601 date string
            use_cache: Whether to use cached results
            max_retries: Maximum number of retry attempts
            language: Language code for caching context (default: "vi")
        
        Returns:
            List of SearchResult objects
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []
        
        self.stats['total_queries'] += 1
        num_results = num_results or self.max_results
        
        # Generate cache key based on parameters
        cache_key = f"{query}_{num_results}_{self.search_type}_{self.use_context}"
        if include_domains:
            cache_key += f"_inc_{','.join(sorted(include_domains))}"
        
        # Check cache first
        if use_cache and self.cache_manager:
            cached = self.cache_manager.get_search_results(cache_key, language)
            if cached:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return [SearchResult.from_dict(r) for r in cached]
            self.stats['cache_misses'] += 1
        
        # Execute API call with retries
        for attempt in range(max_retries):
            try:
                results = self._execute_search_api(
                    query, 
                    num_results,
                    include_domains,
                    exclude_domains,
                    start_published_date,
                    end_published_date
                )
                
                # Cache results
                if use_cache and self.cache_manager:
                    results_dict = [r.to_dict() for r in results]
                    self.cache_manager.set_search_results(cache_key, language, results_dict)
                
                # Reset backoff on success
                self.rate_limiter.reset_backoff()
                
                return results
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limit error
                    self.stats['rate_limit_errors'] += 1
                    self.rate_limiter.record_error(is_rate_limit_error=True)
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        logger.error("Rate limit hit, max retries exceeded")
                        return []
                else:
                    # Other HTTP error
                    self.stats['api_errors'] += 1
                    self.rate_limiter.record_error(is_rate_limit_error=False)
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"API error {e.response.status_code}, retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        logger.error(f"API error {e.response.status_code}, max retries exceeded")
                        return []
            
            except Exception as e:
                self.stats['api_errors'] += 1
                logger.error(f"Search error: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    return []
        
        return []
    
    def _execute_search_api(
        self,
        query: str,
        num_results: int,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        start_published_date: Optional[str],
        end_published_date: Optional[str]
    ) -> List[SearchResult]:
        """
        Execute Exa Search API call.
        """
        # Wait if rate limit would be exceeded
        self.rate_limiter.wait_if_needed()
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "type": self.search_type,
            "numResults": min(num_results, 100),  # API max is 100
            "contents": {
                "text": {
                    "maxCharacters": 5000,  # Request more text content
                    "includeHtmlTags": False
                },
                "highlights": {
                    "numSentences": 5,  # More sentences for better context
                    "highlightsPerUrl": 3  # More highlights per result
                },
                "summary": {
                    "query": query  # Add summary based on query
                }
            }
        }
        
        if self.use_context:
            payload["contents"]["context"] = True
            
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if start_published_date:
            payload["startPublishedDate"] = start_published_date
        if end_published_date:
            payload["endPublishedDate"] = end_published_date
            
        # Make API request
        logger.debug(f"Executing Exa Search API call for query: {query[:50]}...")
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Record the call
        self.rate_limiter.record_call()
        self.stats['api_calls'] += 1
        
        # Check for errors
        response.raise_for_status()
        
        # Parse results
        data = response.json()
        results = []
        
        if 'results' in data:
            for idx, item in enumerate(data['results'], start=1):
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    id=item.get('id', ''),
                    score=item.get('score'), # Note: API spec shows highlightScores, but some endpoints might return score. Re-checking spec.
                    published_date=item.get('publishedDate'),
                    author=item.get('author'),
                    text=item.get('text'),
                    highlights=item.get('highlights', []),
                    summary=item.get('summary'),
                    rank=idx
                )
                results.append(result)
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def batch_search(
        self,
        queries: List[str],
        use_cache: bool = True
    ) -> Dict[str, List[SearchResult]]:
        """
        Execute multiple search queries.
        
        Args:
            queries: List of search queries
            use_cache: Whether to use cached results
        
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            query_results = self.search(query, use_cache=use_cache)
            results[query] = query_results
            
            # Small delay between queries
            time.sleep(0.5)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()
        stats['rate_limiter'] = self.rate_limiter.get_stats()
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'api_errors': 0,
            'rate_limit_errors': 0
        }

