"""ReAct Agent Core for autonomous fact-checking.

This module contains the main ReActAgent class that implements
the Reasoning + Acting (ReAct) paradigm for fact verification.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.llm_controller import LLMController, PromptTemplates
from src.data_models import Claim, Evidence
from src.exa_search_client import ExaSearchClient
from src.web_crawler import WebCrawler
from src.credibility_analyzer import CredibilityAnalyzer

from src.agent.models import ActionType, ReasoningStep, AgentState
from src.agent.parsing import ActionParser
from src.agent.verdict_generator import VerdictGenerator
from src.agent.tools import Tool, SearchTool, CrawlTool, CredibilityTool

logger = logging.getLogger(__name__)

# Priority domains for Vietnamese fact-checking
PRIORITY_DOMAINS = [
    '.gov.vn',           # Government sources (highest priority)
    'gso.gov.vn',        # General Statistics Office
    'chinhphu.vn',       # Government portal
    'thuvienphapluat.vn', # Legal database
    'vnexpress.net',     # Major news
    'tuoitre.vn',
    'thanhnien.vn',
    'vtv.vn',
    'vov.vn',
    'nhandan.vn',
]

# Vietnamese character set for language detection
VIETNAMESE_CHARS = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"


class ReActAgent:
    """ReAct Agent for autonomous fact-checking.
    
    Implements the Reasoning + Acting paradigm where the agent:
    1. Thinks about what to do next (Reasoning)
    2. Takes an action (Acting)
    3. Observes the result
    4. Repeats until reaching a conclusion
    
    Attributes:
        llm_controller: LLM controller for reasoning.
        max_iterations: Maximum reasoning iterations.
        auto_crawl_search_results: Whether to auto-crawl search results.
        max_auto_crawl: Maximum URLs to auto-crawl per search.
        tools: Dictionary of available tools.
        action_parser: Parser for LLM action extraction.
        verdict_generator: Generator for final verdicts.
    """
    
    def __init__(
        self,
        llm_controller: LLMController,
        search_client: ExaSearchClient,
        web_crawler: WebCrawler,
        credibility_analyzer: CredibilityAnalyzer,
        max_iterations: int = 10,
        auto_crawl_search_results: bool = True,
        max_auto_crawl: int = 5
    ) -> None:
        """Initialize ReAct agent with tools.
        
        Args:
            llm_controller: LLM controller for reasoning.
            search_client: Search client for information retrieval.
            web_crawler: Web crawler for content extraction.
            credibility_analyzer: Credibility analyzer for source evaluation.
            max_iterations: Maximum reasoning iterations.
            auto_crawl_search_results: Auto-crawl top search URLs.
            max_auto_crawl: Maximum URLs to auto-crawl per search.
        """
        self.llm_controller = llm_controller
        self.max_iterations = max_iterations
        self.auto_crawl_search_results = auto_crawl_search_results
        self.max_auto_crawl = max_auto_crawl
        
        # Initialize tools
        self.tools: Dict[str, Tool] = {
            "search": SearchTool(search_client),
            "crawl": CrawlTool(web_crawler),
            "analyze_credibility": CredibilityTool(credibility_analyzer)
        }
        
        # Action parser
        self.action_parser = ActionParser()
        
        # Verdict generator
        self.verdict_generator = VerdictGenerator(llm_controller)
        
        logger.info("Initialized ReAct agent with tools: " + ", ".join(self.tools.keys()))
    
    def verify_claim(self, claim: Claim) -> AgentState:
        """Verify a claim using ReAct reasoning loop.
        
        Args:
            claim: Claim to verify.
            
        Returns:
            AgentState with complete reasoning trace and verdict.
        """
        logger.info(f"Starting claim verification: {claim.text}")
        
        # Initialize agent state
        state = AgentState(claim=claim, max_iterations=self.max_iterations)
        
        # Main ReAct loop
        consecutive_none_actions = 0
        MAX_CONSECUTIVE_NONE = 2
        
        while not state.is_complete and not state.should_terminate():
            step_number = len(state.reasoning_steps) + 1
            
            try:
                # Generate reasoning step
                reasoning_step = self._generate_reasoning_step(state, step_number)
                state.add_reasoning_step(reasoning_step)
                
                # Handle None actions
                if reasoning_step.action is None:
                    consecutive_none_actions += 1
                    logger.warning(
                        f"Step {step_number}: No action parsed "
                        f"(consecutive: {consecutive_none_actions})"
                    )
                    reasoning_step.observation = "no_observation"
                    
                    # Auto-conclude if enough evidence
                    if (consecutive_none_actions >= MAX_CONSECUTIVE_NONE 
                            and len(state.collected_evidence) > 0):
                        logger.info(
                            f"Auto-concluding: {consecutive_none_actions} consecutive "
                            f"None actions with {len(state.collected_evidence)} evidence"
                        )
                        conclude_step = ReasoningStep(
                            step_number=step_number + 1,
                            thought="Sufficient evidence collected. Proceeding to conclusion.",
                            action="conclude",
                            action_params={},
                            observation="Ready to conclude verification."
                        )
                        state.add_reasoning_step(conclude_step)
                        state.is_complete = True
                        break
                else:
                    consecutive_none_actions = 0
                    
                    # Execute action
                    observation = self._execute_action(
                        reasoning_step.action,
                        reasoning_step.action_params or {},
                        state
                    )
                    reasoning_step.observation = observation
                
                # Check if agent wants to conclude
                if reasoning_step.action == "conclude":
                    state.is_complete = True
                    break
                    
            except Exception as e:
                logger.error(f"Error in reasoning step {step_number}: {e}")
                error_step = ReasoningStep(
                    step_number=step_number,
                    thought=f"Error occurred: {str(e)}",
                    observation="Encountered an error, will try a different approach."
                )
                state.add_reasoning_step(error_step)
        
        # Generate final verdict
        if not state.final_verdict:
            state.final_verdict = self.verdict_generator.generate_final_verdict(state)
        
        logger.info(f"Completed claim verification with {len(state.reasoning_steps)} steps")
        return state
    
    def _generate_reasoning_step(
        self, 
        state: AgentState, 
        step_number: int
    ) -> ReasoningStep:
        """Generate a single reasoning step using LLM.
        
        Args:
            state: Current agent state.
            step_number: Current step number.
            
        Returns:
            ReasoningStep with thought and action.
        """
        # Prepare context
        evidence_summary = state.get_evidence_summary()
        history = self._format_history(state.reasoning_steps)
        
        # Create prompt
        messages = [
            {"role": "system", "content": PromptTemplates.get_system_prompt()},
            {"role": "user", "content": self._create_step_prompt(
                state.claim.text, history, step_number, evidence_summary
            )}
        ]
        
        # Generate response
        response = self.llm_controller.generate(
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        # Parse response
        thought = self.action_parser.extract_thought(response.content)
        action_name, action_params = self.action_parser.parse_action(response.content)
        
        return ReasoningStep(
            step_number=step_number,
            thought=thought,
            action=action_name,
            action_params=action_params
        )
    
    def _format_history(self, steps: List[ReasoningStep]) -> str:
        """Format previous reasoning steps for context.
        
        Args:
            steps: List of previous reasoning steps.
            
        Returns:
            Formatted history string.
        """
        if not steps:
            return ""
        
        history = "PREVIOUS STEPS:\n"
        for step in steps:
            history += f"Step {step.step_number}:\n"
            history += f"Thought: {step.thought}\n"
            if step.action:
                history += f"Action: {step.action}({step.action_params})\n"
            if step.observation:
                obs = step.observation[:500] + "..." if len(step.observation) > 500 else step.observation
                history += f"Observation: {obs}\n"
            history += "\n"
        
        return history
    
    def _create_step_prompt(
        self, 
        claim_text: str, 
        history: str, 
        step_number: int,
        evidence_summary: str
    ) -> str:
        """Create the prompt for a reasoning step.
        
        Args:
            claim_text: The claim text to verify.
            history: Formatted history of previous steps.
            step_number: Current step number.
            evidence_summary: Summary of collected evidence.
            
        Returns:
            Formatted prompt string.
        """
        return f"""
CLAIM TO VERIFY: {claim_text}

{history}
STEP {step_number}:

EVIDENCE COLLECTED SO FAR:
{evidence_summary}

Please provide your next reasoning step. Format your response as:

THOUGHT: [Your reasoning about what to do next]
ACTION: [The action to take: search("query"), crawl("url"), analyze_credibility("url"), or conclude()]

Available actions:
- search(query="your search query") - Search for information
- crawl(url="https://example.com") - Extract content from a specific URL
- analyze_credibility(source_url="https://example.com") - Analyze source credibility
- conclude() - Finish verification and provide verdict

Think step by step and be thorough in your analysis.
"""
    
    def _execute_action(
        self,
        action_name: str,
        action_params: Dict[str, Any],
        state: AgentState
    ) -> str:
        """Execute an action and return observation.
        
        Args:
            action_name: Name of action to execute.
            action_params: Parameters for the action.
            state: Current agent state.
            
        Returns:
            Observation string from action execution.
        """
        if action_name == "conclude":
            return "Ready to conclude verification."
        
        if action_name not in self.tools:
            return f"Unknown action: {action_name}"
        
        try:
            tool = self.tools[action_name]
            observation = tool.execute(**action_params)
            
            # Process observation
            if action_name == "crawl":
                url = action_params.get("url", "")
                self._extract_evidence_from_crawl(url, observation, state)
            elif action_name == "search":
                if self.auto_crawl_search_results:
                    self._auto_crawl_from_search(observation, state)
            
            return observation
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return f"Action failed: {str(e)}"
    
    def _prioritize_urls(self, urls: List[str]) -> List[str]:
        """Prioritize URLs with government and major news sources first.
        
        Args:
            urls: List of URLs to prioritize.
            
        Returns:
            Sorted list with priority URLs first.
        """
        priority_urls = []
        other_urls = []
        wikipedia_urls = []
        
        for url in urls:
            url_lower = url.lower()
            if 'wikipedia.org' in url_lower:
                wikipedia_urls.append(url)
            elif any(domain in url_lower for domain in PRIORITY_DOMAINS):
                priority_urls.append(url)
            else:
                other_urls.append(url)
        
        return priority_urls + other_urls + wikipedia_urls
    
    def _auto_crawl_from_search(self, observation: str, state: AgentState) -> None:
        """Auto-crawl URLs from search observation to collect evidence.
        
        Args:
            observation: Search result observation.
            state: Current agent state.
        """
        urls = ActionParser.extract_urls_from_text(observation)
        if not urls:
            return
        
        # Prioritize URLs
        urls = self._prioritize_urls(urls)
        
        crawled = state.get_crawled_urls()
        to_crawl = []
        
        for url in urls:
            if url in crawled:
                continue
            
            state.mark_url_crawled(url)
            to_crawl.append(url)
            
            if len(to_crawl) >= self.max_auto_crawl:
                break
        
        for url in to_crawl:
            try:
                crawl_obs = self.tools["crawl"].execute(url=url)
                self._extract_evidence_from_crawl(url, crawl_obs, state)
            except Exception as exc:
                logger.warning(f"Auto-crawl failed for {url}: {exc}")
    
    def _extract_evidence_from_crawl(
        self, 
        url: str, 
        observation: str, 
        state: AgentState
    ) -> None:
        """Extract evidence from crawl observation and add to state.
        
        Args:
            url: URL that was crawled.
            observation: Observation from crawl action.
            state: Current agent state.
        """
        try:
            # Get content from cache or parse observation
            full_content, title, author, publish_date = self._get_crawl_content(url, observation)
            
            # Skip if no meaningful content
            if not full_content or full_content == "N/A" or len(full_content) < 50:
                logger.warning(
                    f"Insufficient content from {url}: "
                    f"{len(full_content) if full_content else 0} chars"
                )
                return
            
            # Detect temporal context
            temporal_marker = self._detect_temporal_marker(full_content)
            
            # Create evidence object
            evidence = Evidence(
                text=full_content,
                source_url=url,
                source_title=title if title and title != "N/A" else "",
                source_author=author if author and author != "N/A" else "",
                publish_date=publish_date,
                language="vi" if self._is_vietnamese(full_content) else "en",
                temporal_marker=temporal_marker
            )
            
            state.add_evidence(evidence)
            logger.info(
                f"Added evidence from {url}: {len(full_content)} chars, "
                f"title='{title[:50] if title else 'N/A'}'"
            )
                
        except Exception as e:
            logger.warning(f"Failed to extract evidence from crawl: {e}")
    
    def _get_crawl_content(
        self, 
        url: str, 
        observation: str
    ) -> tuple:
        """Get content from crawl tool cache or parse observation.
        
        Args:
            url: URL that was crawled.
            observation: Crawl observation string.
            
        Returns:
            Tuple of (full_content, title, author, publish_date).
        """
        crawl_tool = self.tools.get("crawl")
        
        # Try cache first
        if crawl_tool and hasattr(crawl_tool, 'get_cached_content'):
            cached = crawl_tool.get_cached_content(url)
            if cached:
                logger.info(f"Using cached content for {url}")
                return (
                    cached.get('main_text', ''),
                    cached.get('title', ''),
                    cached.get('author', ''),
                    cached.get('publish_date')
                )
        
        # Fallback: parse observation
        return self._parse_crawl_observation(observation)
    
    def _parse_crawl_observation(self, observation: str) -> tuple:
        """Parse crawl observation to extract content.
        
        Args:
            observation: Crawl observation string.
            
        Returns:
            Tuple of (content, title, author, publish_date).
        """
        lines = observation.split('\n')
        content_lines = []
        title = ""
        author = ""
        in_content = False
        
        for line in lines:
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Author:"):
                author = line.replace("Author:", "").strip()
            elif line.startswith("Content:"):
                in_content = True
                content_lines.append(line.replace("Content:", "").strip())
            elif in_content:
                content_lines.append(line)
        
        content = '\n'.join(content_lines).strip()
        logger.warning(f"Using parsed observation content: {len(content)} chars")
        
        return content, title, author, None
    
    def _is_vietnamese(self, text: str) -> bool:
        """Simple heuristic to detect Vietnamese text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            True if text appears to be Vietnamese.
        """
        vietnamese_count = sum(1 for char in text.lower() if char in VIETNAMESE_CHARS)
        return vietnamese_count > len(text) * 0.02
    
    def _detect_temporal_marker(self, text: str) -> Optional[str]:
        """Detect if evidence refers to future, current, or past events.
        
        Args:
            text: Evidence text to analyze.
            
        Returns:
            Temporal marker string or None.
        """
        try:
            from src.temporal_context import is_date_in_past
        except ImportError:
            return "CURRENT"
        
        future_indicators = [
            'sẽ', 'dự kiến', 'kế hoạch', 
            'sắp tới', 'sẽ có hiệu lực', 'từ ngày'
        ]
        text_lower = text.lower()
        
        for indicator in future_indicators:
            if indicator in text_lower:
                date_patterns = [
                    r'(\d{1,2}/\d{1,2}/\d{4})',
                    r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
                    r'(\d{4})'
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        date_str = match.group(0)
                        is_past = is_date_in_past(date_str)
                        if is_past is False:
                            return "FUTURE_POLICY"
                        elif is_past is True:
                            return "CURRENT"
        
        return "CURRENT"


def create_react_agent(
    llm_controller: LLMController,
    search_client: ExaSearchClient,
    web_crawler: WebCrawler,
    credibility_analyzer: CredibilityAnalyzer,
    max_iterations: int = 10,
    auto_crawl_search_results: bool = True,
    max_auto_crawl: int = 2
) -> ReActAgent:
    """Create a ReAct agent with all necessary components.
    
    Args:
        llm_controller: LLM controller for reasoning.
        search_client: Search client for information retrieval.
        web_crawler: Web crawler for content extraction.
        credibility_analyzer: Credibility analyzer for source evaluation.
        max_iterations: Maximum reasoning iterations.
        auto_crawl_search_results: Auto-crawl top search URLs.
        max_auto_crawl: Maximum URLs to auto-crawl per search.
        
    Returns:
        Configured ReActAgent instance.
    """
    return ReActAgent(
        llm_controller=llm_controller,
        search_client=search_client,
        web_crawler=web_crawler,
        credibility_analyzer=credibility_analyzer,
        max_iterations=max_iterations,
        auto_crawl_search_results=auto_crawl_search_results,
        max_auto_crawl=max_auto_crawl
    )
