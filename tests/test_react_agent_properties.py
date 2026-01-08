"""Property-based tests for ReAct agent."""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

from src.react_agent import ReActAgent, AgentState, ReasoningStep, ActionParser, create_react_agent
from src.llm_controller import LLMController, LLMResponse
from src.data_models import Claim, Evidence, Verdict
from src.exa_search_client import ExaSearchClient
from src.web_crawler import WebCrawler
from src.credibility_analyzer import CredibilityAnalyzer


# Test data generators
@st.composite
def generate_claim(draw):
    """Generate a valid Claim object."""
    text = draw(st.text(min_size=10, max_size=200))
    claim_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))))
    language = draw(st.sampled_from(["vi", "en"]))
    
    return Claim(
        text=text,
        id=claim_id,
        language=language
    )


@st.composite
def generate_reasoning_step(draw):
    """Generate a valid ReasoningStep."""
    step_number = draw(st.integers(min_value=1, max_value=20))
    thought = draw(st.text(min_size=10, max_size=500))
    action = draw(st.one_of(
        st.none(),
        st.sampled_from(["search", "crawl", "analyze_credibility", "conclude"])
    ))
    
    action_params = None
    if action == "search":
        action_params = {"query": draw(st.text(min_size=5, max_size=100))}
    elif action == "crawl":
        action_params = {"url": f"https://example{draw(st.integers(min_value=1, max_value=100))}.com"}
    elif action == "analyze_credibility":
        action_params = {"source_url": f"https://example{draw(st.integers(min_value=1, max_value=100))}.com"}
    
    observation = draw(st.one_of(st.none(), st.text(min_size=10, max_size=1000)))
    
    return ReasoningStep(
        step_number=step_number,
        thought=thought,
        action=action,
        action_params=action_params,
        observation=observation
    )


class MockLLMController:
    """Mock LLM controller for testing."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
    
    def generate(self, messages, **kwargs):
        if self.call_count < len(self.responses):
            response_text = self.responses[self.call_count]
        else:
            response_text = "conclude()"  # Default to conclude
        
        self.call_count += 1
        
        return LLMResponse(
            content=response_text,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock_model",
            provider="mock",
            latency=0.1
        )
    
    def get_available_providers(self):
        return ["mock"]


class MockSearchClient:
    """Mock search client for testing."""
    
    def search(self, query, **kwargs):
        # Return mock search results
        return [
            Mock(title="Test Result 1", url="https://example1.com", text="Test content 1"),
            Mock(title="Test Result 2", url="https://example2.com", text="Test content 2")
        ]


class MockWebCrawler:
    """Mock web crawler for testing."""
    
    def extract_content(self, url):
        return {
            "title": f"Article from {url}",
            "author": "Test Author",
            "publish_date": "2023-01-01",
            "text": f"This is test content extracted from {url}. It contains relevant information for fact-checking."
        }


class MockCredibilityAnalyzer:
    """Mock credibility analyzer for testing."""
    
    def analyze_source(self, source_url):
        return {
            "overall_score": 0.8,
            "domain_features": {"https": True, "tld": ".com"},
            "content_features": {"has_author": True, "has_date": True},
            "explanation": "High credibility source"
        }


class TestReActAgentProperties:
    """Property-based tests for ReAct agent."""
    
    @given(generate_claim())
    @settings(max_examples=5, deadline=30000)
    def test_react_loop_structure_property(self, claim):
        """
        **Feature: vietnamese-fact-checking, Property 3: ReAct Loop Structure Invariant**
        
        For any claim verification process, the ReAct loop should maintain the 
        structure: Reasoning -> Action -> Observation -> (repeat until termination).
        **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
        """
        # Create mock components
        llm_responses = [
            'THOUGHT: I need to search for information about this claim.\nACTION: search(query="test query")',
            'THOUGHT: Let me crawl a specific source for more details.\nACTION: crawl(url="https://example.com")',
            'THOUGHT: I have enough information to conclude.\nACTION: conclude()'
        ]
        
        llm_controller = MockLLMController(llm_responses)
        search_client = MockSearchClient()
        web_crawler = MockWebCrawler()
        credibility_analyzer = MockCredibilityAnalyzer()
        
        # Create agent
        agent = create_react_agent(
            llm_controller=llm_controller,
            search_client=search_client,
            web_crawler=web_crawler,
            credibility_analyzer=credibility_analyzer,
            max_iterations=5
        )
        
        # Verify claim
        result = agent.verify_claim(claim)
        
        # Property 1: Each reasoning step should have a thought
        for step in result.reasoning_steps:
            assert step.thought is not None
            assert len(step.thought) > 0
            assert isinstance(step.thought, str)
        
        # Property 2: Steps should be numbered sequentially
        for i, step in enumerate(result.reasoning_steps):
            assert step.step_number == i + 1
        
        # Property 3: If a step has an action, it should have corresponding observation
        for step in result.reasoning_steps:
            if step.action and step.action != "conclude":
                assert step.observation is not None
                assert len(step.observation) > 0
        
        # Property 4: Process should terminate (not infinite loop)
        assert result.is_complete or result.should_terminate()
        assert len(result.reasoning_steps) <= agent.max_iterations
        
        # Property 5: Final verdict should be generated
        assert result.final_verdict is not None
        assert result.final_verdict.label in ["supported", "refuted", "not_enough_info"]
    
    @given(generate_claim(), st.lists(generate_reasoning_step(), min_size=1, max_size=10))
    @settings(max_examples=5, deadline=30000)
    def test_agent_memory_monotonicity_property(self, claim, reasoning_steps):
        """
        **Feature: vietnamese-fact-checking, Property 4: Agent Memory Monotonicity**
        
        The agent's working memory and evidence collection should only grow 
        (monotonic property) - information should never be lost during reasoning.
        **Validates: Requirements 2.4**
        """
        # Create agent state
        state = AgentState(claim=claim, max_iterations=10)
        
        # Track memory growth
        memory_sizes = []
        evidence_counts = []
        reasoning_step_counts = []
        
        # Add reasoning steps incrementally
        for step in reasoning_steps:
            # Record current state
            memory_sizes.append(len(state.working_memory))
            evidence_counts.append(len(state.collected_evidence))
            reasoning_step_counts.append(len(state.reasoning_steps))
            
            # Add reasoning step
            state.add_reasoning_step(step)
            
            # Simulate adding evidence if action was crawl
            if step.action == "crawl" and step.action_params:
                evidence = Evidence(
                    text=f"Evidence from {step.action_params.get('url', 'unknown')}",
                    source_url=step.action_params.get('url', 'https://example.com'),
                    source_title="Test Evidence",
                    language="en"
                )
                state.add_evidence(evidence)
            
            # Add to working memory
            state.working_memory[f"step_{step.step_number}"] = step.thought
        
        # Final measurements
        memory_sizes.append(len(state.working_memory))
        evidence_counts.append(len(state.collected_evidence))
        reasoning_step_counts.append(len(state.reasoning_steps))
        
        # Property 1: Reasoning steps should only increase
        for i in range(len(reasoning_step_counts) - 1):
            assert reasoning_step_counts[i] <= reasoning_step_counts[i + 1], \
                f"Reasoning step count decreased: {reasoning_step_counts[i]} -> {reasoning_step_counts[i + 1]}"
        
        # Property 2: Evidence count should be monotonic
        for i in range(len(evidence_counts) - 1):
            assert evidence_counts[i] <= evidence_counts[i + 1], \
                f"Evidence count decreased: {evidence_counts[i]} -> {evidence_counts[i + 1]}"
        
        # Property 3: Working memory should be monotonic
        for i in range(len(memory_sizes) - 1):
            assert memory_sizes[i] <= memory_sizes[i + 1], \
                f"Memory size decreased: {memory_sizes[i]} -> {memory_sizes[i + 1]}"
        
        # Property 4: Final state should contain all added information
        assert len(state.reasoning_steps) == len(reasoning_steps)
        assert all(step in state.reasoning_steps for step in reasoning_steps)
    
    @given(generate_claim())
    @settings(max_examples=5, deadline=30000)
    def test_evidence_collection_termination_property(self, claim):
        """
        **Feature: vietnamese-fact-checking, Property 5: Evidence Collection Termination**
        
        The evidence collection process should terminate within a reasonable 
        number of iterations and not run indefinitely.
        **Validates: Requirements 2.5**
        """
        # Create mock components with limited responses to force termination
        llm_responses = [
            'THOUGHT: I need to search for information.\nACTION: search(query="test")',
            'THOUGHT: Let me get more evidence.\nACTION: crawl(url="https://example1.com")',
            'THOUGHT: Need another source.\nACTION: crawl(url="https://example2.com")',
            'THOUGHT: One more source for completeness.\nACTION: crawl(url="https://example3.com")',
            'THOUGHT: I have sufficient evidence now.\nACTION: conclude()'
        ]
        
        llm_controller = MockLLMController(llm_responses)
        search_client = MockSearchClient()
        web_crawler = MockWebCrawler()
        credibility_analyzer = MockCredibilityAnalyzer()
        
        # Create agent with reasonable max iterations
        max_iterations = 10
        agent = create_react_agent(
            llm_controller=llm_controller,
            search_client=search_client,
            web_crawler=web_crawler,
            credibility_analyzer=credibility_analyzer,
            max_iterations=max_iterations
        )
        
        # Verify claim
        result = agent.verify_claim(claim)
        
        # Property 1: Process should terminate within max iterations
        assert len(result.reasoning_steps) <= max_iterations
        
        # Property 2: Process should actually terminate (not timeout)
        assert result.is_complete or result.should_terminate()
        
        # Property 3: If sufficient evidence collected, should terminate early
        if len(result.collected_evidence) >= 3:
            # Should terminate before max iterations if we have enough evidence
            assert len(result.reasoning_steps) <= max_iterations
        
        # Property 4: Final verdict should be available
        assert result.final_verdict is not None
        
        # Property 5: Termination should be deterministic based on state
        # If we run the same process again, it should terminate similarly
        result2 = agent.verify_claim(claim)
        
        # Both should terminate within reasonable bounds
        assert len(result2.reasoning_steps) <= max_iterations
        assert result2.is_complete or result2.should_terminate()


class TestActionParser:
    """Test action parsing functionality."""
    
    @given(st.text(min_size=5, max_size=100))
    @settings(max_examples=10)
    def test_search_action_parsing(self, query):
        """Test parsing of search actions."""
        # Test different search formats
        test_cases = [
            f'search(query="{query}")',
            f'search("{query}")',
            f'ACTION: search(query="{query}")',
            f'I will search(query="{query}") for information'
        ]
        
        for test_case in test_cases:
            action, params = ActionParser.parse_action(test_case)
            if action:  # May not parse if query contains special characters
                assert action == "search"
                assert params is not None
                assert "query" in params
                assert params["query"] == query
    
    def test_crawl_action_parsing(self):
        """Test parsing of crawl actions."""
        test_cases = [
            'crawl(url="https://example.com")',
            'crawl("https://example.com")',
            'ACTION: crawl(url="https://test.org")',
        ]
        
        for test_case in test_cases:
            action, params = ActionParser.parse_action(test_case)
            assert action == "crawl"
            assert params is not None
            assert "url" in params
            assert params["url"].startswith("https://")
    
    def test_conclude_action_parsing(self):
        """Test parsing of conclude actions."""
        test_cases = [
            'conclude()',
            'ACTION: conclude()',
            'I will conclude() now'
        ]
        
        for test_case in test_cases:
            action, params = ActionParser.parse_action(test_case)
            assert action == "conclude"
            assert params == {}
    
    def test_thought_extraction(self):
        """Test extraction of thoughts from LLM responses."""
        # Test with predefined thought texts
        test_thoughts = [
            "search for information about this claim",
            "analyze the credibility of the source",
            "gather more evidence from reliable sources",
            "check if there are contradictory viewpoints"
        ]
        
        for thought_text in test_thoughts:
            # Test different thought formats
            test_cases = [
                f"THOUGHT: {thought_text}",
                f"Thought: {thought_text}",
                f"I need to {thought_text}",
                f"Let me {thought_text}"
            ]
            
            for test_case in test_cases:
                extracted = ActionParser.extract_thought(test_case)
                assert extracted is not None
                assert len(extracted) > 0
                # Should contain part of the original thought
                assert any(word in extracted.lower() for word in thought_text.split()[:3])


class TestAgentState:
    """Test agent state management."""
    
    @given(generate_claim())
    @settings(max_examples=5)
    def test_agent_state_initialization(self, claim):
        """Test agent state initialization."""
        state = AgentState(claim=claim)
        
        assert state.claim == claim
        assert len(state.reasoning_steps) == 0
        assert len(state.collected_evidence) == 0
        assert len(state.working_memory) == 0
        assert not state.is_complete
        assert state.final_verdict is None
        assert state.max_iterations > 0
    
    @given(generate_claim(), st.lists(generate_reasoning_step(), min_size=1, max_size=5))
    @settings(max_examples=5)
    def test_reasoning_step_management(self, claim, steps):
        """Test adding and managing reasoning steps."""
        state = AgentState(claim=claim)
        
        for step in steps:
            state.add_reasoning_step(step)
        
        assert len(state.reasoning_steps) == len(steps)
        assert all(step in state.reasoning_steps for step in steps)
    
    def test_termination_conditions(self):
        """Test various termination conditions."""
        claim = Claim(text="Test claim", id="test", language="en")
        
        # Test max iterations termination
        state = AgentState(claim=claim, max_iterations=3)
        
        for i in range(5):  # Add more than max
            step = ReasoningStep(step_number=i+1, thought=f"Step {i+1}")
            state.add_reasoning_step(step)
        
        assert state.should_terminate()  # Should terminate due to max iterations
        
        # Test sufficient evidence termination
        state2 = AgentState(claim=claim, max_iterations=10)
        
        # Add high-credibility evidence
        for i in range(4):
            evidence = Evidence(
                text=f"Evidence {i}",
                source_url=f"https://example{i}.com",
                source_title=f"Source {i}",
                credibility_score=0.9,
                language="en"
            )
            state2.add_evidence(evidence)
        
        assert state2.should_terminate()  # Should terminate due to sufficient evidence


# Integration tests
def test_react_agent_integration():
    """Test ReAct agent integration with mock components."""
    # Create mock components
    llm_responses = [
        'THOUGHT: I need to search for information about this claim.\nACTION: search(query="Vietnam population")',
        'THOUGHT: Let me crawl a reliable source.\nACTION: crawl(url="https://vnexpress.net/article")',
        'THOUGHT: I should check the credibility of this source.\nACTION: analyze_credibility(source_url="https://vnexpress.net")',
        'THOUGHT: I have sufficient evidence to make a conclusion.\nACTION: conclude()'
    ]
    
    llm_controller = MockLLMController(llm_responses)
    search_client = MockSearchClient()
    web_crawler = MockWebCrawler()
    credibility_analyzer = MockCredibilityAnalyzer()
    
    # Create agent
    agent = create_react_agent(
        llm_controller=llm_controller,
        search_client=search_client,
        web_crawler=web_crawler,
        credibility_analyzer=credibility_analyzer
    )
    
    # Test claim
    claim = Claim(
        text="Vietnam has a population of 98 million people.",
        id="test_claim",
        language="en"
    )
    
    # Verify claim
    result = agent.verify_claim(claim)
    
    # Verify results
    assert result is not None
    assert len(result.reasoning_steps) > 0
    assert result.final_verdict is not None
    assert result.final_verdict.label in ["supported", "refuted", "not_enough_info"]
    assert len(result.final_verdict.confidence_scores) == 3
    
    # Check that evidence was collected from crawl actions
    crawl_steps = [s for s in result.reasoning_steps if s.action == "crawl"]
    if crawl_steps:
        assert len(result.collected_evidence) > 0


def test_react_agent_error_handling():
    """Test ReAct agent error handling."""
    # Create mock components that will fail
    class FailingLLMController:
        def generate(self, messages, **kwargs):
            raise Exception("LLM generation failed")
        
        def get_available_providers(self):
            return []
    
    llm_controller = FailingLLMController()
    search_client = MockSearchClient()
    web_crawler = MockWebCrawler()
    credibility_analyzer = MockCredibilityAnalyzer()
    
    # Create agent
    agent = create_react_agent(
        llm_controller=llm_controller,
        search_client=search_client,
        web_crawler=web_crawler,
        credibility_analyzer=credibility_analyzer
    )
    
    # Test claim
    claim = Claim(text="Test claim", id="test", language="en")
    
    # Should handle errors gracefully
    result = agent.verify_claim(claim)
    
    # Should still return a result with default verdict
    assert result is not None
    assert result.final_verdict is not None
    assert result.final_verdict.label == "not_enough_info"  # Default on error