"""Action parsing utilities for the ReAct agent."""

import re
from typing import Any, Dict, Optional, Tuple

# Action types
ACTION_SEARCH = "search"
ACTION_CRAWL = "crawl"
ACTION_ANALYZE_CREDIBILITY = "analyze_credibility"
ACTION_CONCLUDE = "conclude"

# Conclude keywords (Vietnamese and English)
CONCLUDE_KEYWORDS = [
    'conclude', 'kết luận', 'verdict', 
    'final verdict', 'finish verification'
]


class ActionParser:
    """Parser for extracting actions from LLM responses.
    
    Parses LLM output to extract thought/reasoning and action specifications.
    Supports multiple action formats including function-call style and
    keyword-based formats.
    """
    
    @staticmethod
    def parse_action(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse action from LLM response text.
        
        Expected formats:
        - search(query="Vietnam population")
        - crawl(url="https://example.com")
        - Action: search "query"
        - Action: conclude
        
        Args:
            text: The LLM response text to parse.
            
        Returns:
            Tuple of (action_name, parameters) or (None, None) if no action found.
        """
        # Try function call style patterns first
        result = ActionParser._parse_function_call(text)
        if result[0] is not None:
            return result
        
        # Try keyword-based patterns
        result = ActionParser._parse_keyword_action(text)
        if result[0] is not None:
            return result
        
        # Try URL extraction for crawl
        result = ActionParser._parse_url_action(text)
        if result[0] is not None:
            return result
        
        return None, None
    
    @staticmethod
    def _parse_function_call(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse function-call style actions.
        
        Examples:
        - search(query="Vietnam population")
        - crawl(url="https://example.com")
        
        Args:
            text: The text to parse.
            
        Returns:
            Tuple of (action_name, parameters) or (None, None).
        """
        # Search patterns
        search_patterns = [
            r'search\s*\(\s*query\s*=\s*["\']([^"\']+)["\']\s*\)',
            r'search\s*\(\s*["\']([^"\']+)["\']\s*\)',
        ]
        for pattern in search_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return ACTION_SEARCH, {"query": match.group(1)}
        
        # Crawl patterns
        crawl_patterns = [
            r'crawl\s*\(\s*url\s*=\s*["\']([^"\']+)["\']\s*\)',
            r'crawl\s*\(\s*["\']([^"\']+)["\']\s*\)',
        ]
        for pattern in crawl_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return ACTION_CRAWL, {"url": match.group(1)}
        
        # Analyze credibility patterns
        credibility_patterns = [
            r'analyze_credibility\s*\(\s*source_url\s*=\s*["\']([^"\']+)["\']\s*\)',
            r'analyze_credibility\s*\(\s*["\']([^"\']+)["\']\s*\)',
        ]
        for pattern in credibility_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return ACTION_ANALYZE_CREDIBILITY, {"source_url": match.group(1)}
        
        # Conclude pattern
        if re.search(r'conclude\s*\(\s*\)', text, re.IGNORECASE):
            return ACTION_CONCLUDE, {}
        
        return None, None
    
    @staticmethod
    def _parse_keyword_action(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse keyword-based action patterns.
        
        Examples:
        - Action: search "query"
        - Action: conclude
        
        Args:
            text: The text to parse.
            
        Returns:
            Tuple of (action_name, parameters) or (None, None).
        """
        text_lower = text.lower()
        
        # Check for conclude keywords
        for keyword in CONCLUDE_KEYWORDS:
            if keyword in text_lower:
                # Verify it's an action directive
                conclude_pattern = rf'action[:\s]*{keyword}|{keyword}\s*\('
                if re.search(conclude_pattern, text_lower):
                    return ACTION_CONCLUDE, {}
        
        # Check for search with Action: format
        action_search = re.search(
            r'action[:\s]+search[\s"\'\']+([^"\'\'\n]+)', 
            text, 
            re.IGNORECASE
        )
        if action_search:
            query = action_search.group(1).strip('"\'\'')
            return ACTION_SEARCH, {"query": query}
        
        # Check for crawl with Action: format
        action_crawl = re.search(
            r'action[:\s]+crawl[\s"\'\']*\s*(https?://[^\s"\'\'\n]+)', 
            text, 
            re.IGNORECASE
        )
        if action_crawl:
            return ACTION_CRAWL, {"url": action_crawl.group(1)}
        
        return None, None
    
    @staticmethod
    def _parse_url_action(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse URL-based crawl actions.
        
        Looks for URLs following crawl-like keywords.
        
        Args:
            text: The text to parse.
            
        Returns:
            Tuple of (action_name, parameters) or (None, None).
        """
        url_match = re.search(
            r'(?:crawl|access|visit|fetch)[^\n]*(https?://[^\s"\'\'\n]+)', 
            text, 
            re.IGNORECASE
        )
        if url_match:
            return ACTION_CRAWL, {"url": url_match.group(1)}
        
        return None, None
    
    @staticmethod
    def extract_thought(text: str) -> str:
        """Extract thought/reasoning from LLM response.
        
        Args:
            text: The LLM response text.
            
        Returns:
            Extracted thought string.
        """
        # Look for explicit thought patterns
        thought_patterns = [
            r'THOUGHT:\s*(.+?)(?=ACTION:|$)',
            r'Thought:\s*(.+?)(?=Action:|$)',
            r'I need to\s*(.+?)(?=ACTION:|Action:|$)',
            r'Let me\s*(.+?)(?=ACTION:|Action:|$)'
        ]
        
        for pattern in thought_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: return first sentence or paragraph
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip()
        
        # Final fallback: truncate text
        return text[:200] + "..." if len(text) > 200 else text
    
    @staticmethod
    def extract_urls_from_text(text: str) -> list:
        """Extract URLs from text.
        
        Args:
            text: Text containing URLs.
            
        Returns:
            List of extracted URLs.
        """
        # Primary pattern: URL: https://...
        pattern = r"URL:\s*(https?://\S+)"
        urls = re.findall(pattern, text)
        
        # Fallback: find any URLs
        if not urls:
            fallback_pattern = r"(https?://[^\s\)\]\"\\']+)"
            urls = re.findall(fallback_pattern, text)
        
        # Clean URLs
        cleaned = []
        for url in urls:
            url = url.rstrip('.,;:')
            if url and len(url) > 10:
                cleaned.append(url)
        
        return cleaned
