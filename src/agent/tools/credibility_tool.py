"""Credibility Tool for analyzing source credibility."""

import logging
from typing import Any, Dict, Optional

from src.agent.tools.base import Tool
from src.credibility_analyzer import CredibilityAnalyzer

logger = logging.getLogger(__name__)


class CredibilityTool(Tool):
    """Tool for analyzing source credibility.
    
    This tool evaluates the trustworthiness of information sources
    based on domain features, content analysis, and known source data.
    
    Attributes:
        credibility_analyzer: The CredibilityAnalyzer instance.
    """
    
    def __init__(self, credibility_analyzer: CredibilityAnalyzer) -> None:
        """Initialize the credibility tool.
        
        Args:
            credibility_analyzer: CredibilityAnalyzer instance for analysis.
        """
        super().__init__("analyze_credibility")
        self.credibility_analyzer = credibility_analyzer
    
    def execute(self, source_url: str, **kwargs: Any) -> str:
        """Analyze credibility of a source.
        
        Args:
            source_url: URL of the source to analyze.
            **kwargs: Additional parameters (unused currently).
            
        Returns:
            Formatted string containing credibility analysis or error message.
        """
        try:
            # Analyze credibility
            result = self.credibility_analyzer.analyze_source(source_url)
            
            if not result:
                return f"Failed to analyze credibility for: {source_url}"
            
            # Format credibility info
            return self._format_result(source_url, result)
            
        except Exception as e:
            logger.error(f"Credibility tool failed: {e}")
            return f"Credibility analysis failed: {str(e)}"
    
    def _format_result(self, source_url: str, result: Dict[str, Any]) -> str:
        """Format credibility analysis result into readable string.
        
        Args:
            source_url: The analyzed URL.
            result: Dictionary containing analysis results.
            
        Returns:
            Formatted analysis string.
        """
        output = f"Credibility analysis for {source_url}:\n"
        output += f"Overall Score: {result.get('overall_score', 0):.2f}/1.0\n"
        output += f"Domain Features: {result.get('domain_features', {})}\n"
        output += f"Content Features: {result.get('content_features', {})}\n"
        output += f"Explanation: {result.get('explanation', 'N/A')}"
        
        return output
