"""Agent data models - ActionType, ReasoningStep, AgentState."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from src.data_models import Claim, Evidence, Verdict

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the agent can take.
    
    Defines the available actions in the ReAct reasoning loop.
    """
    SEARCH = "search"
    CRAWL = "crawl"
    ANALYZE_CREDIBILITY = "analyze_credibility"
    CONCLUDE = "conclude"


@dataclass
class ReasoningStep:
    """A single step in the ReAct reasoning process.
    
    Each step records the agent's thought, action taken, and observation.
    
    Attributes:
        step_number: Sequential step number in the reasoning chain.
        thought: The agent's reasoning for this step.
        action: The action taken (search, crawl, etc.) or None.
        action_params: Parameters passed to the action.
        observation: Result/observation from executing the action.
        timestamp: When this step was created.
    """
    step_number: int
    thought: str
    action: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """State of the ReAct agent during fact-checking.
    
    Maintains all information about the current verification process
    including the claim, collected evidence, reasoning steps, and verdict.
    
    Attributes:
        claim: The claim being verified.
        reasoning_steps: List of reasoning steps taken.
        collected_evidence: Evidence collected during verification.
        working_memory: Temporary storage for agent state.
        is_complete: Whether verification is complete.
        final_verdict: The final verdict if complete.
        max_iterations: Maximum reasoning iterations allowed.
    """
    claim: Claim
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    collected_evidence: List[Evidence] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False
    final_verdict: Optional[Verdict] = None
    max_iterations: int = 10
    
    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trace.
        
        Args:
            step: The reasoning step to add.
        """
        self.reasoning_steps.append(step)
        thought_preview = step.thought[:100] if step.thought else ""
        logger.debug(f"Added reasoning step {step.step_number}: {thought_preview}...")
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the collection.
        
        Args:
            evidence: The evidence to add.
        """
        self.collected_evidence.append(evidence)
        logger.info(f"Added evidence from {evidence.source_url}")
    
    def get_evidence_summary(self) -> str:
        """Get a summary of collected evidence.
        
        Returns:
            Formatted string summarizing all collected evidence.
        """
        if not self.collected_evidence:
            return "No evidence collected yet."
        
        summary_parts = []
        for i, evidence in enumerate(self.collected_evidence, 1):
            stance_info = f" (Stance: {evidence.stance})" if evidence.stance else ""
            credibility_info = ""
            if evidence.credibility_score:
                credibility_info = f" (Credibility: {evidence.credibility_score:.2f})"
            
            text_preview = evidence.text[:200] if evidence.text else "No text"
            title = evidence.source_title or 'Untitled'
            
            summary_parts.append(
                f"{i}. {title}: {text_preview}...{stance_info}{credibility_info}"
            )
        
        return "\n".join(summary_parts)
    
    def should_terminate(self) -> bool:
        """Check if the agent should terminate reasoning.
        
        Returns:
            True if agent should stop, False otherwise.
        """
        # Terminate if max iterations reached
        if len(self.reasoning_steps) >= self.max_iterations:
            logger.info(f"Terminating: reached max iterations ({self.max_iterations})")
            return True
        
        # Terminate if we have sufficient evidence from multiple sources
        if len(self.collected_evidence) >= 3:
            return self._check_evidence_sufficiency()
        
        return False
    
    def _check_evidence_sufficiency(self) -> bool:
        """Check if collected evidence is sufficient to conclude.
        
        Returns:
            True if evidence is sufficient, False otherwise.
        """
        # Check source diversity - need at least 2 unique domains
        domains = self._get_unique_domains()
        
        # Require at least 2 different source domains
        if len(domains) < 2:
            logger.info("Not enough source diversity, continuing search...")
            return False
        
        # Check if Wikipedia is the ONLY non-empty source
        non_wiki_domains = [d for d in domains if 'wikipedia.org' not in d]
        if len(non_wiki_domains) == 0:
            logger.warning("Only Wikipedia sources found, continuing to find official sources...")
            return False
        
        # Check for contradictory evidence or high credibility
        stances = [e.stance for e in self.collected_evidence if e.stance]
        credibility_scores = [
            e.credibility_score 
            for e in self.collected_evidence 
            if e.credibility_score
        ]
        
        has_contradiction = len(set(stances)) > 1 if stances else False
        has_high_credibility = any(score > 0.8 for score in credibility_scores)
        
        if has_contradiction or has_high_credibility:
            logger.info(f"Terminating: sufficient evidence from {len(domains)} sources")
            return True
        
        return False
    
    def _get_unique_domains(self) -> Set[str]:
        """Get unique domains from collected evidence.
        
        Returns:
            Set of unique domain names.
        """
        domains = set()
        for evidence in self.collected_evidence:
            if evidence.source_url:
                try:
                    domain = urlparse(evidence.source_url).netloc.lower()
                    domains.add(domain)
                except Exception:
                    pass
        return domains
    
    def get_crawled_urls(self) -> Set[str]:
        """Get set of URLs that have been auto-crawled.
        
        Returns:
            Set of crawled URLs.
        """
        return self.working_memory.get("auto_crawled_urls", set())
    
    def get_crawled_domains(self) -> Set[str]:
        """Get set of domains that have been crawled.
        
        Returns:
            Set of crawled domain names.
        """
        return self.working_memory.get("domains_crawled", set())
    
    def mark_url_crawled(self, url: str) -> None:
        """Mark a URL as having been crawled.
        
        Args:
            url: The URL that was crawled.
        """
        if "auto_crawled_urls" not in self.working_memory:
            self.working_memory["auto_crawled_urls"] = set()
        if "domains_crawled" not in self.working_memory:
            self.working_memory["domains_crawled"] = set()
        
        self.working_memory["auto_crawled_urls"].add(url)
        
        try:
            domain = urlparse(url).netloc.lower()
            self.working_memory["domains_crawled"].add(domain)
        except Exception:
            pass
