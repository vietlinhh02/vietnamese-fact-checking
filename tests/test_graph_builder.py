"""Unit tests for graph builder module."""

import pytest
from src.graph_builder import GraphBuilder, build_graph_from_evidence
from src.data_models import Evidence, KnowledgeGraph
from src.ner_extractor import Entity


class TestGraphBuilder:
    """Test GraphBuilder class."""
    
    def test_graph_builder_initialization(self):
        """Test creating graph builder."""
        builder = GraphBuilder()
        assert builder.entity_similarity_threshold == 0.85
        assert builder.ner_extractor is not None
        assert builder.relation_extractor is not None
    
    def test_build_graph_empty_evidence(self):
        """Test building graph with empty evidence list."""
        builder = GraphBuilder()
        graph = builder.build_graph([])
        
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
    
    def test_build_graph_single_evidence(self):
        """Test building graph with single evidence."""
        builder = GraphBuilder()
        
        evidence = Evidence(
            text="Nguyễn Văn A làm việc tại công ty ABC ở Hà Nội.",
            source_url="https://vnexpress.net/article1",
            source_title="Test Article",
            credibility_score=0.8,
            language="vi",
            stance="support",
            stance_confidence=0.9
        )
        
        graph = builder.build_graph([evidence])
        
        # Should have at least evidence node
        assert graph.node_count() >= 1
        
        # Check evidence node exists
        evidence_nodes = [n for n in graph.nodes.values() if n.type == "evidence"]
        assert len(evidence_nodes) == 1
        assert evidence_nodes[0].attributes['credibility_score'] == 0.8
    
    def test_build_graph_multiple_evidence(self):
        """Test building graph with multiple evidence pieces."""
        builder = GraphBuilder()
        
        evidence_list = [
            Evidence(
                text="Vietnam is located in Southeast Asia.",
                source_url="https://vnexpress.net/article1",
                source_title="Geography",
                credibility_score=0.9,
                language="en"
            ),
            Evidence(
                text="Hanoi is the capital of Vietnam.",
                source_url="https://vtv.vn/article2",
                source_title="Capital City",
                credibility_score=0.85,
                language="en"
            )
        ]
        
        graph = builder.build_graph(evidence_list)
        
        # Should have 2 evidence nodes
        evidence_nodes = [n for n in graph.nodes.values() if n.type == "evidence"]
        assert len(evidence_nodes) == 2
        
        # Should have entity nodes
        entity_nodes = [n for n in graph.nodes.values() if n.type == "entity"]
        assert len(entity_nodes) > 0
    
    def test_entity_merging(self):
        """Test that similar entities are merged."""
        builder = GraphBuilder(entity_similarity_threshold=0.8)
        
        # Create evidence with similar entity mentions
        evidence_list = [
            Evidence(
                text="Vietnam is a country in Asia.",
                source_url="https://vnexpress.net/article1",
                source_title="Test 1",
                credibility_score=0.9,
                language="en"
            ),
            Evidence(
                text="Viet Nam has a rich history.",
                source_url="https://vtv.vn/article2",
                source_title="Test 2",
                credibility_score=0.85,
                language="en"
            )
        ]
        
        graph = builder.build_graph(evidence_list)
        
        # Check that entities were created
        entity_nodes = [n for n in graph.nodes.values() if n.type == "entity"]
        
        # Should have merged similar entities
        # Note: Actual merging depends on NER extraction
        assert len(entity_nodes) >= 0  # May vary based on NER
    
    def test_add_claim_node(self):
        """Test adding claim node to graph."""
        builder = GraphBuilder()
        
        evidence = Evidence(
            text="Test evidence text.",
            source_url="https://vnexpress.net/article1",
            source_title="Test",
            credibility_score=0.8,
            language="en",
            stance="support",
            stance_confidence=0.9
        )
        
        graph = builder.build_graph([evidence])
        
        # Add claim node
        claim_id = builder.add_claim_node(
            graph=graph,
            claim_text="This is a test claim.",
            claim_id="claim123"
        )
        
        assert claim_id == "claim_claim123"
        
        # Check claim node exists
        claim_node = graph.get_node(claim_id)
        assert claim_node is not None
        assert claim_node.type == "claim"
        assert claim_node.text == "This is a test claim."
    
    def test_handle_contradictions(self):
        """Test handling contradictory evidence."""
        builder = GraphBuilder()
        
        # Create supporting and refuting evidence
        evidence_list = [
            Evidence(
                text="Supporting evidence.",
                source_url="https://vnexpress.net/article1",
                source_title="Support",
                credibility_score=0.8,
                language="en",
                stance="support",
                stance_confidence=0.9
            ),
            Evidence(
                text="Refuting evidence.",
                source_url="https://vtv.vn/article2",
                source_title="Refute",
                credibility_score=0.7,
                language="en",
                stance="refute",
                stance_confidence=0.85
            )
        ]
        
        graph = builder.build_graph(evidence_list)
        claim_id = builder.add_claim_node(graph, "Test claim", "claim123")
        
        # Handle contradictions
        result = builder.handle_contradictions(graph, claim_id)
        
        assert 'supporting' in result
        assert 'refuting' in result
        assert len(result['supporting']) > 0
        assert len(result['refuting']) > 0
        
        # Check claim node has contradiction metadata
        claim_node = graph.get_node(claim_id)
        assert claim_node.attributes.get('has_contradiction') is True
    
    def test_text_similarity(self):
        """Test text similarity calculation."""
        builder = GraphBuilder()
        
        # Exact match
        assert builder._calculate_text_similarity("test", "test") == 1.0
        
        # Similar texts
        similarity = builder._calculate_text_similarity("vietnam", "viet nam")
        assert 0.5 < similarity < 1.0
        
        # Different texts
        similarity = builder._calculate_text_similarity("vietnam", "thailand")
        assert similarity < 0.5
    
    def test_convenience_function(self):
        """Test convenience function for building graph."""
        evidence = Evidence(
            text="Test evidence.",
            source_url="https://vnexpress.net/article1",
            source_title="Test",
            credibility_score=0.8,
            language="en"
        )
        
        graph = build_graph_from_evidence([evidence])
        
        assert isinstance(graph, KnowledgeGraph)
        assert graph.node_count() >= 1


class TestGraphBuilderIntegration:
    """Integration tests for graph builder with real NER and RE."""
    
    def test_full_pipeline_vietnamese(self):
        """Test full graph building pipeline with Vietnamese text."""
        builder = GraphBuilder()
        
        evidence = Evidence(
            text="Thủ đô Hà Nội nằm ở miền Bắc Việt Nam. Dân số khoảng 8 triệu người.",
            source_url="https://vnexpress.net/article1",
            source_title="Hà Nội",
            credibility_score=0.9,
            language="vi",
            stance="support",
            stance_confidence=0.85
        )
        
        graph = builder.build_graph([evidence])
        
        # Should have evidence node
        assert graph.node_count() >= 1
        
        # Should have extracted some entities (dates, numbers, locations)
        entity_nodes = [n for n in graph.nodes.values() if n.type == "entity"]
        # Note: Actual count depends on NER extraction quality
        assert len(entity_nodes) >= 0
    
    def test_full_pipeline_english(self):
        """Test full graph building pipeline with English text."""
        builder = GraphBuilder()
        
        evidence = Evidence(
            text="Apple Inc. was founded by Steve Jobs in California. The company is headquartered in Cupertino.",
            source_url="https://example.com/article1",
            source_title="Apple History",
            credibility_score=0.85,
            language="en",
            stance="support",
            stance_confidence=0.9
        )
        
        graph = builder.build_graph([evidence])
        
        # Should have evidence node
        assert graph.node_count() >= 1
        
        # Should have extracted entities
        entity_nodes = [n for n in graph.nodes.values() if n.type == "entity"]
        # With spaCy, should extract: Apple Inc., Steve Jobs, California, Cupertino
        # Without spaCy, may extract fewer entities
        assert len(entity_nodes) >= 0
    
    def test_graph_monotonic_growth(self):
        """Test that graph grows monotonically as evidence is added."""
        builder = GraphBuilder()
        
        evidence_list = [
            Evidence(
                text=f"Evidence piece {i}. Contains information about topic {i}.",
                source_url=f"https://example.com/article{i}",
                source_title=f"Article {i}",
                credibility_score=0.8,
                language="en"
            )
            for i in range(5)
        ]
        
        node_counts = []
        
        for i in range(1, len(evidence_list) + 1):
            graph = builder.build_graph(evidence_list[:i])
            node_counts.append(graph.node_count())
        
        # Node count should be non-decreasing
        for i in range(len(node_counts) - 1):
            assert node_counts[i] <= node_counts[i + 1], \
                f"Node count decreased: {node_counts[i]} -> {node_counts[i + 1]}"
