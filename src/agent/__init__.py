"""Agent package - ReAct Agent for autonomous fact-checking.

This package contains the modular implementation of the ReAct
(Reasoning + Acting) agent for Vietnamese fact-checking.

Modules:
    models: Data models (ActionType, ReasoningStep, AgentState)
    parsing: Action parsing utilities (ActionParser)
    verdict_generator: Verdict generation logic (VerdictGenerator)
    react_agent: Main ReActAgent class
    tools: Agent tools (Search, Crawl, Credibility)
"""

from src.agent.models import ActionType, ReasoningStep, AgentState
from src.agent.parsing import ActionParser
from src.agent.verdict_generator import VerdictGenerator
from src.agent.react_agent import ReActAgent, create_react_agent
from src.agent.tools import Tool, SearchTool, CrawlTool, CredibilityTool

__all__ = [
    # Models
    "ActionType",
    "ReasoningStep", 
    "AgentState",
    # Parsing
    "ActionParser",
    # Verdict
    "VerdictGenerator",
    # Agent
    "ReActAgent",
    "create_react_agent",
    # Tools
    "Tool",
    "SearchTool",
    "CrawlTool",
    "CredibilityTool",
]
