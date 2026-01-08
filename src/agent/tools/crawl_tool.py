"""Crawl Tool for extracting content from URLs."""

import logging
from typing import Any, Dict, Optional

from src.agent.tools.base import Tool
from src.web_crawler import WebCrawler

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONTENT_TRUNCATION = 3000


class CrawlTool(Tool):
    """Tool for crawling and extracting content from URLs.
    
    This tool fetches web page content and extracts relevant information
    for fact verification. It maintains a cache of the last crawled content
    for evidence extraction.
    
    Attributes:
        web_crawler: The WebCrawler instance for content extraction.
        last_crawled_content: Cache of the most recently crawled content.
        content_truncation_length: Maximum characters to show in observation.
    """
    
    def __init__(
        self, 
        web_crawler: WebCrawler,
        content_truncation_length: int = DEFAULT_CONTENT_TRUNCATION
    ) -> None:
        """Initialize the crawl tool.
        
        Args:
            web_crawler: WebCrawler instance for content extraction.
            content_truncation_length: Max chars to include in observation.
        """
        super().__init__("crawl")
        self.web_crawler = web_crawler
        self.content_truncation_length = content_truncation_length
        # Store last crawled content for evidence extraction
        self.last_crawled_content: Dict[str, Any] = {}
    
    def execute(self, url: str, **kwargs: Any) -> str:
        """Crawl URL and return extracted content.
        
        Args:
            url: The URL to crawl and extract content from.
            **kwargs: Additional parameters (unused currently).
            
        Returns:
            Formatted string containing extracted content or error message.
        """
        try:
            # Extract content
            content = self.web_crawler.extract_content(url)
            
            if not content:
                self._clear_cache()
                return f"Failed to extract content from URL: {url}"
            
            # Store full content for evidence extraction
            self._cache_content(url, content)
            
            # Format content for observation
            return self._format_content(url, content)
            
        except Exception as e:
            logger.error(f"Crawl tool failed: {e}")
            self._clear_cache()
            return f"Crawling failed: {str(e)}"
    
    def _cache_content(self, url: str, content: Dict[str, Any]) -> None:
        """Cache the crawled content for later evidence extraction.
        
        Args:
            url: The URL that was crawled.
            content: The extracted content dictionary.
        """
        # Support both 'main_text' and 'text' fields for compatibility
        main_text = content.get('main_text', '') or content.get('text', '')
        
        self.last_crawled_content = {
            'url': url,
            'title': content.get('title', ''),
            'author': content.get('author'),
            'publish_date': content.get('publish_date'),
            'main_text': main_text,
            'metadata': content.get('metadata', {})
        }

    
    def _clear_cache(self) -> None:
        """Clear the content cache."""
        self.last_crawled_content = {}
    
    def _format_content(self, url: str, content: Dict[str, Any]) -> str:
        """Format extracted content into observation string.
        
        Args:
            url: The URL that was crawled.
            content: The extracted content dictionary.
            
        Returns:
            Formatted observation string.
        """
        # Support both 'main_text' and 'text' fields
        main_text = content.get('main_text', '') or content.get('text', '')
        truncation = self.content_truncation_length

        
        if len(main_text) > truncation:
            truncated_text = main_text[:truncation] + '...'
        else:
            truncated_text = main_text
        
        result = f"Content extracted from {url}:\n"
        result += f"Title: {content.get('title', 'N/A')}\n"
        result += f"Author: {content.get('author', 'N/A')}\n"
        result += f"Publish Date: {content.get('publish_date', 'N/A')}\n"
        result += f"Full Content Length: {len(main_text)} characters\n"
        result += f"Content: {truncated_text}"
        
        return result
    
    def get_cached_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content for a specific URL if available.
        
        Args:
            url: The URL to get cached content for.
            
        Returns:
            Cached content dictionary if available, None otherwise.
        """
        if self.last_crawled_content.get('url') == url:
            return self.last_crawled_content
        return None
