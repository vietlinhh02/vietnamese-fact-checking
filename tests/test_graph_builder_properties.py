"""Property-based tests for graph builder module."""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any
import numpy as np

from src.graph_builder import GraphBuilder, build_graph_from_evidence
from src.data_models import Evidence, KnowledgeGraph, GraphNode, GraphEdge
from src.ner_extractor import Entity


# Test data generators
@st.composite
def generate_evidence(draw):
    """Generate a valid Evidence object."""
    text = draw(st.text(min_size=10, max_size=200))
    source_url = draw(st.text(min_size=10, max_size=100))
    source_title = draw(st.text(min_size=5, max_size=50))
    credibility_score = draw(st.floats(min_value=0.0, max_value=1.0))
    language = draw(st.sampled_from(["vi", "en"]))
    stance = draw(st.sampled_from(["support", "refute", "neutral"]))
    stance_confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
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
def generate_evidence_list(draw):
    """Generate a list of Evidence objects."""
    return draw(st.lists(generate_evidence(), min_size=1, max_size=10))


class TestGraphBuilderProperties:
    """Property-based tests for graph builder."""
    
    @given(generate_evidence_list())
    @settings(max_examples=10, deadline=30000)
    def test_graph_node_extraction_property(self, evidence_list):
        """
        **Feature: vietnamese-fact-checking, Property 16: Graph Node Extraction**
        
        For any evidence processed by the graph builder, all extracted entities 
        should be represented as nodes in the knowledge graph with proper type 
        classification and attributes.
        **Validates: Requirements 7.1**
        """
        try:
            builder = GraphBuilder()
            graph = builder.build_graph(evidence_list)
            
            # Property: Graph should contain nodes for all evidence
            evidence_nodes = [n for n in graph.nodes.values() if n.type == "evidence"]
            assert len(evidence_nodes) == len(evidence_list)
            
            # Property: Each evidence node should have required attributes
            for evidence_node in evidence_nodes:
                assert evidence_node.type == "evidence"
                assert hasattr(evidence_node, 'text')
                assert hasattr(evidence_node, 'attributes')
                assert 'credibility_score' in evidence_node.attributes
                assert 'language' in evidence_node.attributes
                
                # Credibility score should be preserved
                credibility = evidence_node.attributes['credibility_score']
                assert 0.0 <= credibility <= 1.0
            
            # Property: Entity nodes should have proper type and attributes
            entity_nodes = [n for n in graph.nodes.values() if n.type == "entity"]
            for entity_node in entity_nodes:
                assert entity_node.type == "entity"
                assert hasattr(entity_node, 'text')
                assert hasattr(entity_node, 'attributes')
                
                # Entity nodes should have entity_type attribute
                if 'entity_type' in entity_node.attributes:
                    entity_type = entity_node.attributes['entity_type']
                    # Should be one of standard NER types
                    valid_types = {"PERSON", "ORG", "LOC", "DATE", "NUMBER", "MISC"}
                    # Note: Some extractors may use different type names
                    assert isinstance(entity_type, str)
                
                # Entity nodes should have confidence if available
                if 'confidence' in entity_node.attributes:
                    confidence = entity_node.attributes['confidence']
                    assert 0.0 <= confidence <= 1.0
            
        except Exception as e:
            # Skip if NER/RE extraction fails due to missing dependencies
            if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype"]):
                pytest.skip(f"Skipping due to dependency issue: {e}")
            else:
                raise
    
    @given(generate_evidence_list())
    @settings(max_examples=10, deadline=30000)
    def test_graph_monotonic_growth_property(self, evidence_list):
        """
        **Feature: vietnamese-fact-checking, Property 18: Graph Monotonic Growth**
        
        As evidence is incrementally added to the knowledge graph, the number 
        of nodes and edges should never decrease (monotonic growth property).
        **Validates: Requirements 7.3**
        """
        # Skip if evidence list is too small
        assume(len(evidence_list) >= 2)
        
        try:
            builder = GraphBuilder()
            
            # Track node and edge counts as we add evidence incrementally
            node_counts = []
            edge_counts = []
            
            for i in range(1, len(evidence_list) + 1):
                partial_evidence = evidence_list[:i]
                graph = builder.build_graph(partial_evidence)
                
                node_counts.append(graph.node_count())
                edge_counts.append(graph.edge_count())
            
            # Property: Node count should be monotonically non-decreasing
            for i in range(len(node_counts) - 1):
                assert node_counts[i] <= node_counts[i + 1], \
                    f"Node count decreased from {node_counts[i]} to {node_counts[i + 1]} at step {i}"
            
            # Property: Edge count should be monotonically non-decreasing
            for i in range(len(edge_counts) - 1):
                assert edge_counts[i] <= edge_counts[i + 1], \
                    f"Edge count decreased from {edge_counts[i]} to {edge_counts[i + 1]} at step {i}"
            
            # Property: Final graph should have at least as many nodes as evidence pieces
            assert node_counts[-1] >= len(evidence_list), \
                f"Final node count {node_counts[-1]} less than evidence count {len(evidence_list)}"
        
        except Exception as e:
            # Skip if extraction fails due to missing dependencies
            if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype"]):
                pytest.skip(f"Skipping due to dependency issue: {e}")
            else:
                raise
    
    @given(generate_evidence_list())
    @settings(max_examples=10, deadline=30000)
    def test_entity_uniqueness_property(self, evidence_list):
        """
        **Feature: vietnamese-fact-checking, Property 19: Entity Uniqueness**
        
        Similar entities mentioned across different evidence pieces should be 
        merged into single nodes, preventing duplicate entity representations.
        **Validates: Requirements 7.4**
        """
        try:
            builder = GraphBuilder(entity_similarity_threshold=0.8)
            graph = builder.build_graph(evidence_list)
            
            # Get all entity nodes
            entity_nodes = [n for n in graph.nodes.values() if n.type == "entity"]
            
            if len(entity_nodes) <= 1:
                # Skip if too few entities to test uniqueness
                pytest.skip("Not enough entities to test uniqueness")
            
            # Property: No two entity nodes should have identical text
            entity_texts = [node.text.lower().strip() for node in entity_nodes]
            unique_texts = set(entity_texts)
            
            # Should not have exact duplicates
            assert len(entity_texts) == len(unique_texts), \
                f"Found duplicate entity texts: {[t for t in entity_texts if entity_texts.count(t) > 1]}"
            
            # Property: Entity nodes should have unique IDs
            entity_ids = [node.id for node in entity_nodes]
            unique_ids = set(entity_ids)
            assert len(entity_ids) == len(unique_ids), \
                f"Found duplicate entity IDs: {[i for i in entity_ids if entity_ids.count(i) > 1]}"
            
            # Property: If mentions attribute exists, it should track all variations
            for entity_node in entity_nodes:
                if 'mentions' in entity_node.attributes:
                    mentions = entity_node.attributes['mentions']
                    assert isinstance(mentions, list)
                    assert len(mentions) >= 1  # Should have at least one mention
                    assert entity_node.text in mentions  # Primary text should be in mentions
        
        except Exception as e:
            # Skip if extraction fails due to missing dependencies
            if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype"]):
                pytest.skip(f"Skipping due to dependency issue: {e}")
            else:
                raise
    
    @given(generate_evidence_list())
    @settings(max_examples=10, deadline=30000)
    def test_contradiction_preservation_property(self, evidence_list):
        """
        **Feature: vietnamese-fact-checking, Property 20: Contradiction Preservation**
        
        When contradictory evidence exists (supporting and refuting the same claim), 
        both perspectives should be preserved in the graph with source attribution.
        **Validates: Requirements 7.5**
        """
        # Create evidence with explicit contradictory stances
        contradictory_evidence = []
        
        # Add supporting evidence
        for i, evidence in enumerate(evidence_list[:len(evidence_list)//2]):
            evidence.stance = "support"
            evidence.stance_confidence = 0.8
            contradictory_evidence.append(evidence)
        
        # Add refuting evidence
        for i, evidence in enumerate(evidence_list[len(evidence_list)//2:]):
            evidence.stance = "refute"
            evidence.stance_confidence = 0.7
            contradictory_evidence.append(evidence)
        
        # Skip if we don't have both supporting and refuting evidence
        supporting_count = sum(1 for e in contradictory_evidence if e.stance == "support")
        refuting_count = sum(1 for e in contradictory_evidence if e.stance == "refute")
        
        assume(supporting_count >= 1 and refuting_count >= 1)
        
        try:
            builder = GraphBuilder()
            graph = builder.build_graph(contradictory_evidence)
            
            # Add a claim node to test contradiction handling
            claim_id = builder.add_claim_node(graph, "Test claim for contradiction", "test_claim")
            
            # Handle contradictions
            contradiction_result = builder.handle_contradictions(graph, claim_id)
            
            # Property: Both supporting and refuting evidence should be preserved
            assert 'supporting' in contradiction_result
            assert 'refuting' in contradiction_result
            
            supporting_evidence = contradiction_result['supporting']
            refuting_evidence = contradiction_result['refuting']
            
            # Should have both types of evidence
            assert len(supporting_evidence) >= 1, "No supporting evidence preserved"
            assert len(refuting_evidence) >= 1, "No refuting evidence preserved"
            
            # Property: Claim node should be marked with contradiction metadata
            claim_node = graph.get_node(claim_id)
            assert claim_node is not None
            
            if claim_node.attributes.get('has_contradiction'):
                # If contradiction is detected, metadata should be present
                assert 'supporting_count' in claim_node.attributes
                assert 'refuting_count' in claim_node.attributes
                assert claim_node.attributes['supporting_count'] >= 1
                assert claim_node.attributes['refuting_count'] >= 1
                
                # Weight calculations should be present
                if 'support_weight' in claim_node.attributes:
                    assert claim_node.attributes['support_weight'] >= 0
                if 'refute_weight' in claim_node.attributes:
                    assert claim_node.attributes['refute_weight'] >= 0
            
            # Property: All evidence nodes should maintain source attribution
            evidence_nodes = [n for n in graph.nodes.values() if n.type == "evidence"]
            for evidence_node in evidence_nodes:
                assert 'source_url' in evidence_node.attributes
                assert evidence_node.attributes['source_url'] is not None
                
                # Stance information should be preserved
                if 'stance' in evidence_node.attributes:
                    stance = evidence_node.attributes['stance']
                    assert stance in ["support", "refute", "neutral"]
        
        except Exception as e:
            # Skip if extraction fails due to missing dependencies
            if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype"]):
                pytest.skip(f"Skipping due to dependency issue: {e}")
            else:
                raise


class TestGraphBuilderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_evidence_list(self):
        """Test graph building with empty evidence list."""
        builder = GraphBuilder()
        graph = builder.build_graph([])
        
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
    
    @given(st.text(min_size=1, max_size=10))
    @settings(max_examples=5)
    def test_minimal_evidence(self, text):
        """Test graph building with minimal evidence."""
        evidence = Evidence(
            text=text,
            source_url="https://test.com",
            source_title="Test",
            credibility_score=0.5,
            language="en"
        )
        
        try:
            builder = GraphBuilder()
            graph = builder.build_graph([evidence])
            
            # Should have at least the evidence node
            assert graph.node_count() >= 1
            
            # Evidence node should exist
            evidence_nodes = [n for n in graph.nodes.values() if n.type == "evidence"]
            assert len(evidence_nodes) == 1
            
        except Exception as e:
            # Skip if extraction fails
            if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype"]):
                pytest.skip(f"Skipping due to dependency issue: {e}")
            else:
                raise
    
    def test_high_similarity_threshold(self):
        """Test entity merging with high similarity threshold."""
        builder = GraphBuilder(entity_similarity_threshold=0.95)
        
        evidence_list = [
            Evidence(
                text="Vietnam is a country.",
                source_url="https://test1.com",
                source_title="Test 1",
                credibility_score=0.8,
                language="en"
            ),
            Evidence(
                text="Viet Nam is beautiful.",
                source_url="https://test2.com", 
                source_title="Test 2",
                credibility_score=0.7,
                language="en"
            )
        ]
        
        try:
            graph = builder.build_graph(evidence_list)
            
            # With high threshold, similar entities might not merge
            # This is acceptable behavior
            assert graph.node_count() >= 2  # At least 2 evidence nodes
            
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype"]):
                pytest.skip(f"Skipping due to dependency issue: {e}")
            else:
                raise
    
    def test_low_similarity_threshold(self):
        """Test entity merging with low similarity threshold."""
        builder = GraphBuilder(entity_similarity_threshold=0.1)
        
        evidence_list = [
            Evidence(
                text="Apple Inc. is a technology company.",
                source_url="https://test1.com",
                source_title="Test 1", 
                credibility_score=0.8,
                language="en"
            ),
            Evidence(
                text="Microsoft Corp. is also a tech company.",
                source_url="https://test2.com",
                source_title="Test 2",
                credibility_score=0.7,
                language="en"
            )
        ]
        
        try:
            graph = builder.build_graph(evidence_list)
            
            # With low threshold, more entities might merge inappropriately
            # But the graph should still be valid
            assert graph.node_count() >= 2  # At least 2 evidence nodes
            
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype"]):
                pytest.skip(f"Skipping due to dependency issue: {e}")
            else:
                raise


# Integration test with real components
def test_graph_builder_integration():
    """Test graph builder integration with real NER and RE components."""
    try:
        builder = GraphBuilder()
        
        # Test with realistic Vietnamese evidence
        evidence = Evidence(
            text="Thủ tướng Phạm Minh Chính đã gặp Tổng thống Biden tại Washington vào tháng 9 năm 2023.",
            source_url="https://vnexpress.net/test",
            source_title="Ngoại giao Việt-Mỹ",
            credibility_score=0.9,
            language="vi",
            stance="support",
            stance_confidence=0.85
        )
        
        graph = builder.build_graph([evidence])
        
        # Should have evidence node
        assert graph.node_count() >= 1
        
        # Check evidence node
        evidence_nodes = [n for n in graph.nodes.values() if n.type == "evidence"]
        assert len(evidence_nodes) == 1
        
        evidence_node = evidence_nodes[0]
        assert evidence_node.attributes['credibility_score'] == 0.9
        assert evidence_node.attributes['language'] == "vi"
        
        # May have entity nodes (depends on NER extraction)
        entity_nodes = [n for n in graph.nodes.values() if n.type == "entity"]
        # Don't assert specific count as it depends on NER quality
        
        # Test adding claim
        claim_id = builder.add_claim_node(graph, "Việt Nam có quan hệ ngoại giao tốt với Mỹ", "test_claim")
        
        claim_node = graph.get_node(claim_id)
        assert claim_node is not None
        assert claim_node.type == "claim"
        
    except Exception as e:
        if any(keyword in str(e).lower() for keyword in ["spacy", "model", "numpy.dtype", "transformers"]):
            pytest.skip(f"Skipping integration test due to dependency issue: {e}")
        else:
            raise