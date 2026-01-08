"""Backward compatibility re-exports for react_agent module.

This file maintains backward compatibility with existing imports.
All functionality has been refactored into the src.agent package.

New code should import from src.agent directly:
    from src.agent import ReActAgent, create_react_agent

Legacy imports still work:
    from src.react_agent import ReActAgent, create_react_agent
"""

# Re-export all public symbols from the agent package
from src.agent.models import ActionType, ReasoningStep, AgentState
from src.agent.parsing import ActionParser
from src.agent.verdict_generator import VerdictGenerator
from src.agent.react_agent import ReActAgent, create_react_agent
from src.agent.tools import Tool, SearchTool, CrawlTool, CredibilityTool

# For backward compatibility with existing code
__all__ = [
    # Enums and Models
    "ActionType",
    "ReasoningStep",
    "AgentState",
    # Parser
    "ActionParser",
    # Verdict Generator
    "VerdictGenerator",
    # Main Agent
    "ReActAgent",
    "create_react_agent",
    # Tools
    "Tool",
    "SearchTool",
    "CrawlTool",
    "CredibilityTool",
]
