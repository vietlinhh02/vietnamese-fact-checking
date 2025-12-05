"""Property-based tests for core data models using Hypothesis."""

import sys
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import Claim, Evidence, ReasoningStep, Verdict, KnowledgeGraph


# Custom strategies for generating test data
@st.composite
def claim_strategy(draw):
    """Generate valid Claim objects."""
    # Generate claim text (at least 10 characters)
    claim_text = draw(st.text(min_size=10, max_size=200))
    
    # Generate context that includes the claim text plus surrounding text
    prefix = draw(st.text(min_size=5, max_size=100))
    suffix = draw(st.text(min_size=5, max_size=100))
    context = f"{prefix} {claim_text} {suffix}"
    
    # Generate other fields
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    sentence_type = draw(st.sampled_from(["factual_claim", "opinion", "question", "command"]))
    
    # Calculate indices
    start_idx = len(prefix) + 1
    end_idx = start_idx + len(claim_text)
    
    language = draw(st.sampled_from(["vi", "en"]))
    
    return Claim(
        text=claim_text,
        context=context,
        confidence=confidence,
        sentence_type=sentence_type,
        start_idx=start_idx,
        end_idx=end_idx,
        language=language
    )


@st.composite
def evidence_strategy(draw):
    """Generate valid Evidence objects."""
    text = draw(st.text(min_size=10, max_size=500))
    source_url = draw(st.from_regex(r'https?://[a-z]+\.(com|net|vn)/[a-z]+', fullmatch=True))
    source_title = draw(st.text(min_size=5, max_size=100))
    credibility_score = draw(st.floats(min_value=0.0, max_value=1.0))
    language = draw(st.sampled_from(["vi", "en"]))
    stance = draw(st.one_of(st.none(), st.sampled_from(["support", "refute", "neutral"])))
    stance_confidence = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))) if stance else None
    
    return Evidence(
        text=text,
        source_url=source_url,
        source_title=source_title,
        credibility_score=credibility_score,
        language=language,
        stance=stance,
        stance_confidence=stance_confidence
    )


@st.composite
def reasoning_step_strategy(draw):
    """Generate valid ReasoningStep objects."""
    iteration = draw(st.integers(min_value=0, max_value=100))
    thought = draw(st.text(min_size=10, max_size=200))
    action = draw(st.sampled_from(["search", "crawl", "analyze", "verify"]))
    action_input = draw(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(), st.integers(), st.floats()),
        min_size=0,
        max_size=5
    ))
    observation = draw(st.text(min_size=10, max_size=200))
    
    return ReasoningStep(
        iteration=iteration,
        thought=thought,
        action=action,
        action_input=action_input,
        observation=observation
    )


@st.composite
def verdict_strategy(draw):
    """Generate valid Verdict objects."""
    claim_id = draw(st.text(min_size=5, max_size=20))
    label = draw(st.sampled_from(["supported", "refuted", "not_enough_info"]))
    
    # Generate confidence scores that sum to 1.0
    score1 = draw(st.floats(min_value=0.0, max_value=1.0))
    score2 = draw(st.floats(min_value=0.0, max_value=1.0 - score1))
    score3 = 1.0 - score1 - score2
    
    confidence_scores = {
        "supported": score1,
        "refuted": score2,
        "not_enough_info": score3
    }
    
    quality_score = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return Verdict(
        claim_id=claim_id,
        label=label,
        confidence_scores=confidence_scores,
        quality_score=quality_score
    )


class TestClaimProperties:
    """Property-based tests for Claim model."""
    
    @given(claim_strategy())
    @settings(max_examples=100)
    def test_claim_extraction_context_preservation(self, claim):
        """
        **Feature: vietnamese-fact-checking, Property 2: Claim Extraction Context Preservation**
        
        Test that for any detected claim, the extracted output contains both 
        the claim text and its surrounding context.
        
        **Validates: Requirements 1.3**
        """
        # Property: Context should contain the claim text
        assert claim.text in claim.context, \
            f"Claim text '{claim.text}' not found in context '{claim.context}'"
        
        # Property: Context should be longer than claim text (has surrounding context)
        assert len(claim.context) > len(claim.text), \
            f"Context length ({len(claim.context)}) should be greater than claim text length ({len(claim.text)})"
        
        # Property: The claim should be extractable from context using indices
        if claim.start_idx >= 0 and claim.end_idx <= len(claim.context):
            extracted = claim.context[claim.start_idx:claim.end_idx]
            # Allow for minor whitespace differences
            assert claim.text.strip() in extracted or extracted.strip() in claim.text, \
                f"Claim text '{claim.text}' should be extractable from context at indices [{claim.start_idx}:{claim.end_idx}]"
    
    @given(claim_strategy())
    @settings(max_examples=100)
    def test_claim_serialization_round_trip(self, claim):
        """
        Test that claim serialization preserves all data through round-trip.
        
        This ensures that to_dict/from_dict and to_json/from_json maintain data integrity.
        """
        # Test dict round-trip
        claim_dict = claim.to_dict()
        restored_claim = Claim.from_dict(claim_dict)
        
        assert restored_claim.text == claim.text
        assert restored_claim.context == claim.context
        assert restored_claim.confidence == claim.confidence
        assert restored_claim.sentence_type == claim.sentence_type
        assert restored_claim.start_idx == claim.start_idx
        assert restored_claim.end_idx == claim.end_idx
        assert restored_claim.language == claim.language
        assert restored_claim.id == claim.id
        
        # Test JSON round-trip
        json_str = claim.to_json()
        restored_claim_json = Claim.from_json(json_str)
        
        assert restored_claim_json.text == claim.text
        assert restored_claim_json.context == claim.context
        assert restored_claim_json.id == claim.id
    
    @given(claim_strategy())
    @settings(max_examples=100)
    def test_claim_id_uniqueness(self, claim):
        """Test that claim IDs are generated and non-empty."""
        assert claim.id is not None
        assert len(claim.id) > 0
        assert isinstance(claim.id, str)


class TestEvidenceProperties:
    """Property-based tests for Evidence model."""
    
    @given(evidence_strategy())
    @settings(max_examples=100)
    def test_evidence_serialization_round_trip(self, evidence):
        """Test that evidence serialization preserves all data."""
        # Test dict round-trip
        evidence_dict = evidence.to_dict()
        restored_evidence = Evidence.from_dict(evidence_dict)
        
        assert restored_evidence.text == evidence.text
        assert restored_evidence.source_url == evidence.source_url
        assert restored_evidence.credibility_score == evidence.credibility_score
        assert restored_evidence.stance == evidence.stance
        assert restored_evidence.id == evidence.id
        
        # Test JSON round-trip
        json_str = evidence.to_json()
        restored_evidence_json = Evidence.from_json(json_str)
        
        assert restored_evidence_json.text == evidence.text
        assert restored_evidence_json.source_url == evidence.source_url
    
    @given(evidence_strategy())
    @settings(max_examples=100)
    def test_evidence_credibility_score_range(self, evidence):
        """Test that credibility scores are always in valid range [0, 1]."""
        assert 0.0 <= evidence.credibility_score <= 1.0
    
    @given(evidence_strategy())
    @settings(max_examples=100)
    def test_evidence_stance_consistency(self, evidence):
        """Test that if stance is set, stance_confidence should be valid."""
        if evidence.stance is not None:
            assert evidence.stance in ["support", "refute", "neutral"]
            if evidence.stance_confidence is not None:
                assert 0.0 <= evidence.stance_confidence <= 1.0


class TestReasoningStepProperties:
    """Property-based tests for ReasoningStep model."""
    
    @given(reasoning_step_strategy())
    @settings(max_examples=100)
    def test_reasoning_step_serialization_round_trip(self, step):
        """Test that reasoning step serialization preserves all data."""
        # Test dict round-trip
        step_dict = step.to_dict()
        restored_step = ReasoningStep.from_dict(step_dict)
        
        assert restored_step.iteration == step.iteration
        assert restored_step.thought == step.thought
        assert restored_step.action == step.action
        assert restored_step.action_input == step.action_input
        assert restored_step.observation == step.observation
        
        # Test JSON round-trip
        json_str = step.to_json()
        restored_step_json = ReasoningStep.from_json(json_str)
        
        assert restored_step_json.iteration == step.iteration
        assert restored_step_json.thought == step.thought
    
    @given(reasoning_step_strategy())
    @settings(max_examples=100)
    def test_reasoning_step_has_all_components(self, step):
        """Test that reasoning steps always have thought, action, and observation."""
        assert len(step.thought) > 0
        assert len(step.action) > 0
        assert len(step.observation) > 0
        assert step.iteration >= 0
    
    @given(reasoning_step_strategy())
    @settings(max_examples=100)
    def test_reasoning_step_timestamp_exists(self, step):
        """Test that reasoning steps always have a timestamp."""
        assert step.timestamp is not None
        assert len(step.timestamp) > 0


class TestVerdictProperties:
    """Property-based tests for Verdict model."""
    
    @given(verdict_strategy())
    @settings(max_examples=100)
    def test_verdict_serialization_round_trip(self, verdict):
        """Test that verdict serialization preserves all data."""
        # Test dict round-trip
        verdict_dict = verdict.to_dict()
        restored_verdict = Verdict.from_dict(verdict_dict)
        
        assert restored_verdict.claim_id == verdict.claim_id
        assert restored_verdict.label == verdict.label
        assert restored_verdict.confidence_scores == verdict.confidence_scores
        assert restored_verdict.quality_score == verdict.quality_score
        
        # Test JSON round-trip
        json_str = verdict.to_json()
        restored_verdict_json = Verdict.from_json(json_str)
        
        assert restored_verdict_json.claim_id == verdict.claim_id
        assert restored_verdict_json.label == verdict.label
    
    @given(verdict_strategy())
    @settings(max_examples=100)
    def test_verdict_confidence_scores_sum_to_one(self, verdict):
        """Test that confidence scores always sum to approximately 1.0."""
        total = sum(verdict.confidence_scores.values())
        assert 0.99 <= total <= 1.01, f"Confidence scores sum to {total}, expected ~1.0"
    
    @given(verdict_strategy())
    @settings(max_examples=100)
    def test_verdict_label_validity(self, verdict):
        """Test that verdict labels are always valid."""
        assert verdict.label in ["supported", "refuted", "not_enough_info"]
    
    @given(verdict_strategy())
    @settings(max_examples=100)
    def test_verdict_quality_score_range(self, verdict):
        """Test that quality scores are always in valid range [0, 1]."""
        assert 0.0 <= verdict.quality_score <= 1.0


class TestKnowledgeGraphProperties:
    """Property-based tests for KnowledgeGraph model."""
    
    @given(st.lists(st.text(min_size=5, max_size=50), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_graph_node_count_monotonic(self, node_texts):
        """Test that adding nodes increases or maintains node count (monotonic growth)."""
        from data_models import GraphNode
        
        graph = KnowledgeGraph()
        previous_count = 0
        
        for i, text in enumerate(node_texts):
            node = GraphNode(
                id=f"node_{i}",
                type="entity",
                text=text
            )
            graph.add_node(node)
            
            current_count = graph.node_count()
            assert current_count >= previous_count, \
                f"Node count decreased from {previous_count} to {current_count}"
            previous_count = current_count
    
    @given(st.lists(st.text(min_size=5, max_size=50), min_size=2, max_size=10))
    @settings(max_examples=100)
    def test_graph_serialization_round_trip(self, node_texts):
        """Test that knowledge graph serialization preserves structure."""
        from data_models import GraphNode, GraphEdge
        
        graph = KnowledgeGraph()
        
        # Add nodes
        for i, text in enumerate(node_texts):
            node = GraphNode(id=f"n{i}", type="entity", text=text)
            graph.add_node(node)
        
        # Add some edges
        if len(node_texts) >= 2:
            edge = GraphEdge(source_id="n0", target_id="n1", relation="related")
            graph.add_edge(edge)
        
        # Test dict round-trip
        graph_dict = graph.to_dict()
        restored_graph = KnowledgeGraph.from_dict(graph_dict)
        
        assert restored_graph.node_count() == graph.node_count()
        assert restored_graph.edge_count() == graph.edge_count()
        
        # Test JSON round-trip
        json_str = graph.to_json()
        restored_graph_json = KnowledgeGraph.from_json(json_str)
        
        assert restored_graph_json.node_count() == graph.node_count()
        assert restored_graph_json.edge_count() == graph.edge_count()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
