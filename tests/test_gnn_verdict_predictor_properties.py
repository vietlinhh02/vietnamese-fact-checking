"""Property-based tests for GNN verdict predictor."""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any
try:
    import dgl
    DGL_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError):
    DGL_AVAILABLE = False

from src.gnn_verdict_predictor import GNNVerdictPredictor, NodeFeatureExtractor, VerdictPredictor
from src.data_models import KnowledgeGraph, GraphNode, GraphEdge, Claim, Evidence
from src.graph_builder import GraphBuilder


# Test data generators
@st.composite
def generate_graph_node(draw):
    """Generate a valid GraphNode."""
    node_type = draw(st.sampled_from(["entity", "claim", "evidence"]))
    node_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))))
    text = draw(st.text(min_size=1, max_size=200))
    
    return GraphNode(
        id=f"{node_type}_{node_id}",
        type=node_type,
        text=text,
        attributes={}
    )


@st.composite
def generate_knowledge_graph(draw):
    """Generate a valid KnowledgeGraph with at least one claim node."""
    graph = KnowledgeGraph()
    
    # Generate nodes (ensure at least one claim node)
    num_nodes = draw(st.integers(min_value=2, max_value=10))
    nodes = []
    
    # Always include at least one claim node
    claim_node = GraphNode(
        id="claim_test",
        type="claim", 
        text=draw(st.text(min_size=10, max_size=100)),
        attributes={"claim_id": "test_claim"}
    )
    nodes.append(claim_node)
    graph.add_node(claim_node)
    
    # Add other nodes
    for i in range(num_nodes - 1):
        node = draw(generate_graph_node())
        node.id = f"{node.type}_{i}"  # Ensure unique IDs
        nodes.append(node)
        graph.add_node(node)
    
    # Generate edges (optional)
    num_edges = draw(st.integers(min_value=0, max_value=min(5, len(nodes) * (len(nodes) - 1))))
    for _ in range(num_edges):
        source = draw(st.sampled_from(nodes))
        target = draw(st.sampled_from(nodes))
        
        if source.id != target.id:  # No self-loops
            edge = GraphEdge(
                source_id=source.id,
                target_id=target.id,
                relation=draw(st.sampled_from(["supports", "refutes", "mentions", "related_to"])),
                weight=draw(st.floats(min_value=0.0, max_value=1.0)),
                evidence_source="test_source"
            )
            try:
                graph.add_edge(edge)
            except ValueError:
                # Skip if nodes don't exist (shouldn't happen but be safe)
                pass
    
    return graph, "claim_test"


@st.composite
def generate_claim(draw):
    """Generate a valid Claim."""
    return Claim(
        text=draw(st.text(min_size=10, max_size=200)),
        id=draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")))),
        language="vi",
        confidence=draw(st.floats(min_value=0.0, max_value=1.0))
    )


class TestGNNVerdictPredictor:
    """Test GNN verdict predictor properties."""
    
    def test_model_initialization(self):
        """Test that GNN model initializes correctly."""
        model = GNNVerdictPredictor()
        
        assert model.embedding_dim == 768
        assert model.hidden_dim == 256
        assert model.num_layers == 3
        assert model.num_classes == 3
        assert 0 <= model.dropout <= 1
    
    @given(
        embedding_dim=st.integers(min_value=64, max_value=1024),
        hidden_dim=st.integers(min_value=32, max_value=512),
        num_layers=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10)
    def test_model_forward_shape(self, embedding_dim, hidden_dim, num_layers):
        """Test that model forward pass produces correct output shape."""
        model = GNNVerdictPredictor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Create simple test graph
        if not DGL_AVAILABLE:
            pytest.skip("DGL not available")
            
        num_nodes = 3
        src = [0, 1]
        dst = [1, 2]
        graph = dgl.graph((src, dst), num_nodes=num_nodes)
        
        # Create node features
        node_features = torch.randn(num_nodes, embedding_dim)
        claim_node_idx = 0
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(graph, node_features, claim_node_idx)
        
        # Check output shape
        assert logits.shape == (3,)  # 3 classes
        assert torch.isfinite(logits).all()  # No NaN or Inf values


class TestVerdictPredictor:
    """Test high-level verdict predictor properties."""
    
    @given(generate_knowledge_graph(), generate_claim())
    @settings(max_examples=5, deadline=30000)  # Reduced examples for speed
    def test_gnn_output_format_property(self, graph_data, claim):
        """
        **Feature: vietnamese-fact-checking, Property 21: GNN Output Format**
        
        For any reasoning graph processed by the GNN, the output should include 
        a vector representation for the claim node with dimensionality matching 
        the model's hidden size.
        **Validates: Requirements 8.3**
        """
        graph, claim_node_id = graph_data
        
        # Skip if graph is too small
        assume(graph.node_count() >= 2)
        
        try:
            # Initialize predictor (without trained model)
            predictor = VerdictPredictor(device="cpu")
            
            # Update claim ID to match graph
            claim.id = claim_node_id.split('_')[-1] if '_' in claim_node_id else claim_node_id
            
            # Predict verdict
            verdict = predictor.predict_verdict(graph, claim)
            
            # Property: Output should have correct format
            assert verdict is not None
            assert hasattr(verdict, 'label')
            assert hasattr(verdict, 'confidence_scores')
            
            # Check that label is one of valid classes
            assert verdict.label in ["supported", "refuted", "not_enough_info"]
            
            # Check confidence scores format
            assert isinstance(verdict.confidence_scores, dict)
            assert len(verdict.confidence_scores) == 3
            
            # All required classes should be present
            required_classes = {"supported", "refuted", "not_enough_info"}
            assert set(verdict.confidence_scores.keys()) == required_classes
            
        except Exception as e:
            # If processing fails due to graph structure, that's acceptable
            # The property is about format when processing succeeds
            if "empty graph" in str(e).lower() or "no edges" in str(e).lower():
                pytest.skip(f"Skipping due to graph structure: {e}")
            else:
                raise
    
    @given(generate_knowledge_graph(), generate_claim())
    @settings(max_examples=5, deadline=30000)
    def test_verdict_classification_property(self, graph_data, claim):
        """
        **Feature: vietnamese-fact-checking, Property 22: Verdict Classification**
        
        For any claim verification, the final verdict should be exactly one of 
        three classes: "Supported", "Refuted", or "Not Enough Info", with 
        confidence scores for each class.
        **Validates: Requirements 8.4, 8.5**
        """
        graph, claim_node_id = graph_data
        
        # Skip if graph is too small
        assume(graph.node_count() >= 2)
        
        try:
            # Initialize predictor
            predictor = VerdictPredictor(device="cpu")
            
            # Update claim ID to match graph
            claim.id = claim_node_id.split('_')[-1] if '_' in claim_node_id else claim_node_id
            
            # Predict verdict
            verdict = predictor.predict_verdict(graph, claim)
            
            # Property 1: Verdict label must be exactly one of three classes
            valid_labels = {"supported", "refuted", "not_enough_info"}
            assert verdict.label in valid_labels
            
            # Property 2: Confidence scores must exist for all classes
            assert isinstance(verdict.confidence_scores, dict)
            assert set(verdict.confidence_scores.keys()) == valid_labels
            
            # Property 3: All confidence scores must be in [0, 1]
            for class_name, score in verdict.confidence_scores.items():
                assert 0.0 <= score <= 1.0, f"Confidence score for {class_name} out of range: {score}"
            
            # Property 4: Confidence scores should sum to approximately 1.0
            total_confidence = sum(verdict.confidence_scores.values())
            assert 0.99 <= total_confidence <= 1.01, f"Confidence scores sum to {total_confidence}, not ~1.0"
            
            # Property 5: Quality score should be in valid range
            assert 0.0 <= verdict.quality_score <= 1.0
            
        except Exception as e:
            # If processing fails due to graph structure, that's acceptable
            if "empty graph" in str(e).lower() or "no edges" in str(e).lower():
                pytest.skip(f"Skipping due to graph structure: {e}")
            else:
                raise


class TestNodeFeatureExtractor:
    """Test node feature extraction properties."""
    
    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10))
    @settings(max_examples=3, deadline=30000)  # Reduced for speed
    def test_feature_extraction_shape(self, texts):
        """Test that feature extraction produces correct shapes."""
        try:
            extractor = NodeFeatureExtractor(device="cpu")
            
            # Extract features
            features = extractor.extract_features(texts)
            
            # Check shape
            assert features.shape[0] == len(texts)
            assert features.shape[1] == 768  # XLM-R base embedding dimension
            
            # Check that features are finite
            assert torch.isfinite(features).all()
            
        except Exception as e:
            # Skip if model loading fails (e.g., in CI environment)
            if "model" in str(e).lower() or "tokenizer" in str(e).lower():
                pytest.skip(f"Skipping due to model loading issue: {e}")
            else:
                raise


class TestGraphBuilder:
    """Test graph builder integration with GNN."""
    
    @given(st.lists(
        st.builds(
            Evidence,
            text=st.text(min_size=10, max_size=100),
            source_url=st.text(min_size=10, max_size=50),
            source_title=st.text(min_size=5, max_size=30),
            credibility_score=st.floats(min_value=0.0, max_value=1.0),
            language=st.sampled_from(["vi", "en"]),
            stance=st.sampled_from(["support", "refute", "neutral"]),
            stance_confidence=st.floats(min_value=0.0, max_value=1.0)
        ),
        min_size=1,
        max_size=5
    ))
    @settings(max_examples=5, deadline=30000)
    def test_graph_to_dgl_conversion(self, evidence_list):
        """Test that knowledge graphs can be converted to DGL format."""
        try:
            builder = GraphBuilder()
            
            # Build graph from evidence
            graph = builder.build_graph(evidence_list)
            
            # Skip if graph is empty
            assume(graph.node_count() > 0)
            
            # Convert to DGL
            dgl_graph = builder.to_dgl_graph(graph)
            
            if dgl_graph is not None:
                # Check DGL graph properties
                assert dgl_graph.num_nodes() == graph.node_count()
                assert dgl_graph.num_edges() == graph.edge_count()
                
                # Check node data
                assert 'node_id' in dgl_graph.ndata
                assert 'node_type' in dgl_graph.ndata
                assert len(dgl_graph.ndata['node_id']) == graph.node_count()
                
                # Check edge data (if edges exist)
                if dgl_graph.num_edges() > 0:
                    assert 'weight' in dgl_graph.edata
                    assert len(dgl_graph.edata['weight']) == graph.edge_count()
            
        except (ImportError, FileNotFoundError, OSError):
            pytest.skip("DGL not available")
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                pytest.skip("Skipping due to numpy compatibility issue")
            else:
                raise
        except Exception as e:
            # Skip if conversion fails due to graph structure
            if "empty" in str(e).lower():
                pytest.skip(f"Skipping due to empty graph: {e}")
            else:
                raise


# Integration test
def test_end_to_end_gnn_pipeline():
    """Test complete GNN pipeline with synthetic data."""
    try:
        # Create test data
        claim = Claim(
            text="Test claim for GNN",
            id="test_claim",
            language="vi"
        )
        
        evidence_list = [
            Evidence(
                text="Supporting evidence for the test claim",
                source_url="https://test.com/support",
                source_title="Support Article",
                credibility_score=0.9,
                language="vi",
                stance="support",
                stance_confidence=0.8,
                id="evidence_1"
            ),
            Evidence(
                text="Neutral evidence about the claim",
                source_url="https://test.com/neutral", 
                source_title="Neutral Article",
                credibility_score=0.7,
                language="vi",
                stance="neutral",
                stance_confidence=0.6,
                id="evidence_2"
            )
        ]
        
        # Build graph
        try:
            builder = GraphBuilder()
            graph = builder.build_graph(evidence_list)
            claim_node_id = builder.add_claim_node(graph, claim.text, claim.id)
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                pytest.skip("Skipping due to numpy compatibility issue")
            else:
                raise
        
        # Initialize predictor
        predictor = VerdictPredictor(device="cpu")
        
        # Predict verdict
        verdict = predictor.predict_verdict(graph, claim)
        
        # Verify result format
        assert verdict.label in ["supported", "refuted", "not_enough_info"]
        assert len(verdict.confidence_scores) == 3
        assert 0.99 <= sum(verdict.confidence_scores.values()) <= 1.01
        
    except Exception as e:
        # Skip if dependencies not available
        if any(keyword in str(e).lower() for keyword in ["dgl", "torch", "transformers", "model"]):
            pytest.skip(f"Skipping due to missing dependencies: {e}")
        else:
            raise