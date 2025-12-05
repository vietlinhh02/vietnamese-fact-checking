"""Tests for core data models."""

import sys
from pathlib import Path
import pytest
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import (
    Claim, Evidence, ReasoningStep, Verdict,
    GraphNode, GraphEdge, KnowledgeGraph, FactCheckResult
)


class TestClaim:
    """Test Claim data model."""
    
    def test_claim_creation(self):
        """Test creating a valid claim."""
        claim = Claim(
            text="Việt Nam có 54 dân tộc",
            context="Theo thống kê. Việt Nam có 54 dân tộc. Đây là dữ liệu chính thức.",
            confidence=0.95,
            sentence_type="factual_claim",
            start_idx=15,
            end_idx=38,
            language="vi"
        )
        
        assert claim.text == "Việt Nam có 54 dân tộc"
        assert claim.confidence == 0.95
        assert claim.id is not None
        assert len(claim.id) == 16
    
    def test_claim_validation_empty_text(self):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="Claim text cannot be empty"):
            Claim(text="", context="test")
    
    def test_claim_validation_confidence_range(self):
        """Test that confidence must be in [0, 1]."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            Claim(text="test", confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence must be in"):
            Claim(text="test", confidence=-0.1)
    
    def test_claim_validation_sentence_type(self):
        """Test that sentence_type must be valid."""
        with pytest.raises(ValueError, match="Invalid sentence_type"):
            Claim(text="test", sentence_type="invalid_type")
    
    def test_claim_validation_indices(self):
        """Test that indices must be valid."""
        with pytest.raises(ValueError, match="Indices cannot be negative"):
            Claim(text="test", start_idx=-1)
        
        with pytest.raises(ValueError, match="start_idx cannot be greater than end_idx"):
            Claim(text="test", start_idx=10, end_idx=5)
    
    def test_claim_serialization(self):
        """Test claim to_dict and from_dict."""
        claim = Claim(
            text="Test claim",
            context="Context for test claim",
            confidence=0.8
        )
        
        claim_dict = claim.to_dict()
        assert claim_dict['text'] == "Test claim"
        assert claim_dict['confidence'] == 0.8
        
        restored_claim = Claim.from_dict(claim_dict)
        assert restored_claim.text == claim.text
        assert restored_claim.confidence == claim.confidence
        assert restored_claim.id == claim.id
    
    def test_claim_json_serialization(self):
        """Test claim to_json and from_json."""
        claim = Claim(
            text="JSON test",
            context="Context",
            confidence=0.9
        )
        
        json_str = claim.to_json()
        assert isinstance(json_str, str)
        
        restored_claim = Claim.from_json(json_str)
        assert restored_claim.text == claim.text
        assert restored_claim.confidence == claim.confidence


class TestEvidence:
    """Test Evidence data model."""
    
    def test_evidence_creation(self):
        """Test creating valid evidence."""
        evidence = Evidence(
            text="Evidence text from source",
            source_url="https://vnexpress.net/article",
            source_title="Article Title",
            source_author="Author Name",
            publish_date="2024-01-15T10:30:00",
            credibility_score=0.85,
            language="vi",
            stance="support",
            stance_confidence=0.92
        )
        
        assert evidence.text == "Evidence text from source"
        assert evidence.source_url == "https://vnexpress.net/article"
        assert evidence.stance == "support"
        assert evidence.id is not None
    
    def test_evidence_validation_empty_text(self):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="Evidence text cannot be empty"):
            Evidence(text="", source_url="http://test.com")
    
    def test_evidence_validation_empty_url(self):
        """Test that empty URL raises error."""
        with pytest.raises(ValueError, match="Source URL cannot be empty"):
            Evidence(text="test", source_url="")
    
    def test_evidence_validation_credibility_range(self):
        """Test that credibility score must be in [0, 1]."""
        with pytest.raises(ValueError, match="Credibility score must be in"):
            Evidence(text="test", source_url="http://test.com", credibility_score=1.5)
    
    def test_evidence_validation_stance(self):
        """Test that stance must be valid."""
        with pytest.raises(ValueError, match="Invalid stance"):
            Evidence(
                text="test",
                source_url="http://test.com",
                stance="invalid_stance"
            )
    
    def test_evidence_validation_stance_confidence(self):
        """Test that stance confidence must be in [0, 1]."""
        with pytest.raises(ValueError, match="Stance confidence must be in"):
            Evidence(
                text="test",
                source_url="http://test.com",
                stance="support",
                stance_confidence=1.2
            )
    
    def test_evidence_serialization(self):
        """Test evidence serialization."""
        evidence = Evidence(
            text="Test evidence",
            source_url="http://test.com",
            credibility_score=0.7
        )
        
        evidence_dict = evidence.to_dict()
        restored_evidence = Evidence.from_dict(evidence_dict)
        
        assert restored_evidence.text == evidence.text
        assert restored_evidence.source_url == evidence.source_url
        assert restored_evidence.credibility_score == evidence.credibility_score


class TestReasoningStep:
    """Test ReasoningStep data model."""
    
    def test_reasoning_step_creation(self):
        """Test creating valid reasoning step."""
        step = ReasoningStep(
            iteration=1,
            thought="I need to search for information",
            action="search",
            action_input={"query": "test query"},
            observation="Found 5 results"
        )
        
        assert step.iteration == 1
        assert step.thought == "I need to search for information"
        assert step.action == "search"
        assert step.timestamp is not None
    
    def test_reasoning_step_validation_iteration(self):
        """Test that iteration must be non-negative."""
        with pytest.raises(ValueError, match="Iteration must be non-negative"):
            ReasoningStep(
                iteration=-1,
                thought="test",
                action="test",
                action_input={},
                observation="test"
            )
    
    def test_reasoning_step_validation_empty_fields(self):
        """Test that required fields cannot be empty."""
        with pytest.raises(ValueError, match="Thought cannot be empty"):
            ReasoningStep(
                iteration=0,
                thought="",
                action="test",
                action_input={},
                observation="test"
            )
        
        with pytest.raises(ValueError, match="Action cannot be empty"):
            ReasoningStep(
                iteration=0,
                thought="test",
                action="",
                action_input={},
                observation="test"
            )
        
        with pytest.raises(ValueError, match="Observation cannot be empty"):
            ReasoningStep(
                iteration=0,
                thought="test",
                action="test",
                action_input={},
                observation=""
            )
    
    def test_reasoning_step_serialization(self):
        """Test reasoning step serialization."""
        step = ReasoningStep(
            iteration=2,
            thought="Analyzing results",
            action="analyze",
            action_input={"data": [1, 2, 3]},
            observation="Analysis complete"
        )
        
        step_dict = step.to_dict()
        restored_step = ReasoningStep.from_dict(step_dict)
        
        assert restored_step.iteration == step.iteration
        assert restored_step.thought == step.thought
        assert restored_step.action_input == step.action_input


class TestVerdict:
    """Test Verdict data model."""
    
    def test_verdict_creation(self):
        """Test creating valid verdict."""
        verdict = Verdict(
            claim_id="claim123",
            label="supported",
            confidence_scores={
                "supported": 0.7,
                "refuted": 0.2,
                "not_enough_info": 0.1
            },
            supporting_evidence=["ev1", "ev2"],
            refuting_evidence=[],
            explanation="The claim is supported by evidence",
            quality_score=0.85
        )
        
        assert verdict.claim_id == "claim123"
        assert verdict.label == "supported"
        assert len(verdict.supporting_evidence) == 2
    
    def test_verdict_validation_empty_claim_id(self):
        """Test that claim ID cannot be empty."""
        with pytest.raises(ValueError, match="Claim ID cannot be empty"):
            Verdict(
                claim_id="",
                label="supported",
                confidence_scores={"supported": 1.0}
            )
    
    def test_verdict_validation_label(self):
        """Test that label must be valid."""
        with pytest.raises(ValueError, match="Invalid label"):
            Verdict(
                claim_id="test",
                label="invalid_label",
                confidence_scores={"supported": 1.0}
            )
    
    def test_verdict_validation_confidence_scores_empty(self):
        """Test that confidence scores cannot be empty."""
        with pytest.raises(ValueError, match="Confidence scores cannot be empty"):
            Verdict(
                claim_id="test",
                label="supported",
                confidence_scores={}
            )
    
    def test_verdict_validation_confidence_scores_range(self):
        """Test that confidence scores must be in [0, 1]."""
        with pytest.raises(ValueError, match="Confidence score .* must be in"):
            Verdict(
                claim_id="test",
                label="supported",
                confidence_scores={"supported": 1.5}
            )
    
    def test_verdict_validation_confidence_scores_sum(self):
        """Test that confidence scores must sum to 1.0."""
        with pytest.raises(ValueError, match="Confidence scores must sum to 1.0"):
            Verdict(
                claim_id="test",
                label="supported",
                confidence_scores={
                    "supported": 0.5,
                    "refuted": 0.3
                }
            )
    
    def test_verdict_validation_quality_score(self):
        """Test that quality score must be in [0, 1]."""
        with pytest.raises(ValueError, match="Quality score must be in"):
            Verdict(
                claim_id="test",
                label="supported",
                confidence_scores={"supported": 1.0},
                quality_score=1.5
            )
    
    def test_verdict_serialization_with_reasoning_trace(self):
        """Test verdict serialization with reasoning trace."""
        step1 = ReasoningStep(
            iteration=0,
            thought="Start",
            action="search",
            action_input={},
            observation="Results"
        )
        
        verdict = Verdict(
            claim_id="test",
            label="supported",
            confidence_scores={"supported": 0.6, "refuted": 0.3, "not_enough_info": 0.1},
            reasoning_trace=[step1]
        )
        
        verdict_dict = verdict.to_dict()
        restored_verdict = Verdict.from_dict(verdict_dict)
        
        assert len(restored_verdict.reasoning_trace) == 1
        assert restored_verdict.reasoning_trace[0].iteration == 0


class TestGraphNode:
    """Test GraphNode data model."""
    
    def test_graph_node_creation(self):
        """Test creating valid graph node."""
        node = GraphNode(
            id="node1",
            type="entity",
            text="Việt Nam",
            attributes={"entity_type": "LOCATION"}
        )
        
        assert node.id == "node1"
        assert node.type == "entity"
        assert node.text == "Việt Nam"
    
    def test_graph_node_validation_empty_id(self):
        """Test that node ID cannot be empty."""
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            GraphNode(id="", type="entity", text="test")
    
    def test_graph_node_validation_type(self):
        """Test that node type must be valid."""
        with pytest.raises(ValueError, match="Invalid node type"):
            GraphNode(id="test", type="invalid_type", text="test")
    
    def test_graph_node_validation_empty_text(self):
        """Test that node text cannot be empty."""
        with pytest.raises(ValueError, match="Node text cannot be empty"):
            GraphNode(id="test", type="entity", text="")
    
    def test_graph_node_with_embedding(self):
        """Test graph node with numpy embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        node = GraphNode(
            id="node1",
            type="entity",
            text="test",
            embedding=embedding
        )
        
        assert node.embedding is not None
        assert np.array_equal(node.embedding, embedding)
        
        # Test serialization
        node_dict = node.to_dict()
        assert isinstance(node_dict['embedding'], list)
        
        restored_node = GraphNode.from_dict(node_dict)
        assert np.array_equal(restored_node.embedding, embedding)


class TestGraphEdge:
    """Test GraphEdge data model."""
    
    def test_graph_edge_creation(self):
        """Test creating valid graph edge."""
        edge = GraphEdge(
            source_id="node1",
            target_id="node2",
            relation="supports",
            weight=0.9,
            evidence_source="http://test.com"
        )
        
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.relation == "supports"
        assert edge.weight == 0.9
    
    def test_graph_edge_validation_empty_ids(self):
        """Test that source and target IDs cannot be empty."""
        with pytest.raises(ValueError, match="Source ID cannot be empty"):
            GraphEdge(source_id="", target_id="node2", relation="test")
        
        with pytest.raises(ValueError, match="Target ID cannot be empty"):
            GraphEdge(source_id="node1", target_id="", relation="test")
    
    def test_graph_edge_validation_empty_relation(self):
        """Test that relation cannot be empty."""
        with pytest.raises(ValueError, match="Relation cannot be empty"):
            GraphEdge(source_id="node1", target_id="node2", relation="")
    
    def test_graph_edge_validation_weight_range(self):
        """Test that weight must be in [0, 1]."""
        with pytest.raises(ValueError, match="Weight must be in"):
            GraphEdge(source_id="node1", target_id="node2", relation="test", weight=1.5)


class TestKnowledgeGraph:
    """Test KnowledgeGraph data model."""
    
    def test_knowledge_graph_creation(self):
        """Test creating empty knowledge graph."""
        graph = KnowledgeGraph()
        
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
    
    def test_knowledge_graph_add_node(self):
        """Test adding nodes to graph."""
        graph = KnowledgeGraph()
        
        node1 = GraphNode(id="n1", type="entity", text="Entity 1")
        node2 = GraphNode(id="n2", type="entity", text="Entity 2")
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        assert graph.node_count() == 2
        assert graph.get_node("n1") == node1
        assert graph.get_node("n2") == node2
    
    def test_knowledge_graph_add_edge(self):
        """Test adding edges to graph."""
        graph = KnowledgeGraph()
        
        node1 = GraphNode(id="n1", type="entity", text="Entity 1")
        node2 = GraphNode(id="n2", type="entity", text="Entity 2")
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        edge = GraphEdge(source_id="n1", target_id="n2", relation="related_to")
        graph.add_edge(edge)
        
        assert graph.edge_count() == 1
    
    def test_knowledge_graph_add_edge_validation(self):
        """Test that edges require existing nodes."""
        graph = KnowledgeGraph()
        
        edge = GraphEdge(source_id="n1", target_id="n2", relation="test")
        
        with pytest.raises(ValueError, match="Source node .* not found"):
            graph.add_edge(edge)
    
    def test_knowledge_graph_get_edges(self):
        """Test getting edges from/to nodes."""
        graph = KnowledgeGraph()
        
        node1 = GraphNode(id="n1", type="entity", text="Entity 1")
        node2 = GraphNode(id="n2", type="entity", text="Entity 2")
        node3 = GraphNode(id="n3", type="entity", text="Entity 3")
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        edge1 = GraphEdge(source_id="n1", target_id="n2", relation="r1")
        edge2 = GraphEdge(source_id="n1", target_id="n3", relation="r2")
        edge3 = GraphEdge(source_id="n2", target_id="n3", relation="r3")
        
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        graph.add_edge(edge3)
        
        # Test get_edges_from
        edges_from_n1 = graph.get_edges_from("n1")
        assert len(edges_from_n1) == 2
        
        # Test get_edges_to
        edges_to_n3 = graph.get_edges_to("n3")
        assert len(edges_to_n3) == 2
    
    def test_knowledge_graph_serialization(self):
        """Test knowledge graph serialization."""
        graph = KnowledgeGraph()
        
        node1 = GraphNode(id="n1", type="entity", text="Entity 1")
        node2 = GraphNode(id="n2", type="entity", text="Entity 2")
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        edge = GraphEdge(source_id="n1", target_id="n2", relation="related")
        graph.add_edge(edge)
        
        # Test to_dict and from_dict
        graph_dict = graph.to_dict()
        restored_graph = KnowledgeGraph.from_dict(graph_dict)
        
        assert restored_graph.node_count() == 2
        assert restored_graph.edge_count() == 1
        
        # Test JSON serialization
        json_str = graph.to_json()
        restored_graph_json = KnowledgeGraph.from_json(json_str)
        
        assert restored_graph_json.node_count() == 2
        assert restored_graph_json.edge_count() == 1


class TestFactCheckResult:
    """Test FactCheckResult data model."""
    
    def test_fact_check_result_creation(self):
        """Test creating complete fact-check result."""
        claim = Claim(text="Test claim", context="Context")
        
        evidence = Evidence(
            text="Evidence text",
            source_url="http://test.com"
        )
        
        verdict = Verdict(
            claim_id=claim.id,
            label="supported",
            confidence_scores={"supported": 1.0}
        )
        
        graph = KnowledgeGraph()
        
        result = FactCheckResult(
            claim=claim,
            verdict=verdict,
            evidence=[evidence],
            reasoning_graph=graph,
            metadata={"version": "1.0"}
        )
        
        assert result.claim.text == "Test claim"
        assert result.verdict.label == "supported"
        assert len(result.evidence) == 1
        assert result.metadata["version"] == "1.0"
    
    def test_fact_check_result_serialization(self):
        """Test fact-check result serialization."""
        claim = Claim(text="Test", context="Context")
        evidence = Evidence(text="Evidence", source_url="http://test.com")
        verdict = Verdict(
            claim_id=claim.id,
            label="supported",
            confidence_scores={"supported": 1.0}
        )
        graph = KnowledgeGraph()
        
        result = FactCheckResult(
            claim=claim,
            verdict=verdict,
            evidence=[evidence],
            reasoning_graph=graph
        )
        
        # Test to_dict and from_dict
        result_dict = result.to_dict()
        restored_result = FactCheckResult.from_dict(result_dict)
        
        assert restored_result.claim.text == result.claim.text
        assert restored_result.verdict.label == result.verdict.label
        
        # Test JSON serialization
        json_str = result.to_json()
        restored_result_json = FactCheckResult.from_json(json_str)
        
        assert restored_result_json.claim.text == result.claim.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
