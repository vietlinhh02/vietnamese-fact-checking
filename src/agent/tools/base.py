"""Base Tool class for agent tools."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Tool:
    """Base class for agent tools.
    
    All tools extend this class and implement the execute method
    to perform specific actions during the ReAct reasoning loop.
    
    Attributes:
        name: The unique name identifier for this tool.
    """
    
    def __init__(self, name: str) -> None:
        """Initialize the tool with a name.
        
        Args:
            name: Unique identifier for this tool.
        """
        self.name = name
    
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters.
            
        Returns:
            String observation from tool execution.
            
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement execute()")
