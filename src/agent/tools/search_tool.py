"""Search Tool for information retrieval using Exa API."""

import logging
from typing import Any, List, Optional

from src.agent.tools.base import Tool
from src.exa_search_client import ExaSearchClient

logger = logging.getLogger(__name__)

# Constants
DEFAULT_NUM_RESULTS = 5
MAX_SUMMARY_LENGTH = 300


class SearchTool(Tool):
    """Tool for searching information using Exa API.
    
    This tool performs web searches and returns formatted results
    that can be used for fact verification.
    
    Attributes:
        search_client: The ExaSearchClient instance for performing searches.
    """
    
    def __init__(self, search_client: ExaSearchClient) -> None:
        """Initialize the search tool.
        
        Args:
            search_client: ExaSearchClient instance for API calls.
        """
        super().__init__("search")
        self.search_client = search_client
    
    def execute(
        self, 
        query: str, 
        num_results: int = DEFAULT_NUM_RESULTS,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Search for information and return formatted results.
        
        Args:
            query: The search query string.
            num_results: Maximum number of results to return.
            include_domains: Optional list of domains to include.
            exclude_domains: Optional list of domains to exclude.
            **kwargs: Additional search parameters.
            
        Returns:
            Formatted string containing search results or error message.
        """
        try:
            # Perform search
            results = self.search_client.search(
                query=query,
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains
            )
            
            if not results:
                return f"No search results found for query: '{query}'"
            
            # Format results
            formatted_results = self._format_results(results)
            return f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Search tool failed: {e}")
            return f"Search failed: {str(e)}"
    
    def _format_results(self, results: List[Any]) -> List[str]:
        """Format search results into readable strings.
        
        Args:
            results: List of search result objects.
            
        Returns:
            List of formatted result strings.
        """
        formatted = []
        for i, result in enumerate(results, 1):
            summary = result.text[:MAX_SUMMARY_LENGTH] if result.text else ""
            formatted.append(
                f"{i}. {result.title}\n"
                f"   URL: {result.url}\n"
                f"   Summary: {summary}..."
            )
        return formatted
