"""Graph Neural Network-based verdict predictor for fact-checking."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
try:
    import dgl
    from dgl.nn import GraphConv
    DGL_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError) as e:
    DGL_AVAILABLE = False
    # Don't log here as logger might not be set up yet
import json
import os
from pathlib import Path

from src.data_models import KnowledgeGraph, Verdict, Claim
from src.config import SystemConfig

# Setup logging
logger = logging.getLogger(__name__)


class GNNVerdictPredictor(nn.Module):
    """Graph Neural Network for verdict prediction on reasoning graphs."""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.1,
        aggregation: str = "mean"
    ):
        """Initialize GNN verdict predictor.
        
        Args:
            embedding_dim: Dimension of input node embeddings (XLM-R hidden size)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GCN layers
            num_classes: Number of verdict classes (3: supported, refuted, not_enough_info)
            dropout: Dropout rate
            aggregation: Aggregation method for readout ("mean", "max", "sum")
        """
        super(GNNVerdictPredictor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.aggregation = aggregation
        
        # Input projection layer
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # GCN layers
        self.gnn_layers = nn.ModuleList()
        if DGL_AVAILABLE:
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))
                else:
                    self.gnn_layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))
        else:
            # Fallback: use simple linear layers if DGL not available
            for i in range(num_layers):
                self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        if not DGL_AVAILABLE:
            logger.warning("DGL not available. Using fallback linear layers instead of GCN.")
        logger.info(f"Initialized GNN with {num_layers} layers, hidden_dim={hidden_dim}, classes={num_classes}")
    
    def forward(self, graph, node_features: torch.Tensor, claim_node_idx: int) -> torch.Tensor:
        """Forward pass through GNN.
        
        Args:
            graph: DGL graph
            node_features: Node feature matrix [num_nodes, embedding_dim]
            claim_node_idx: Index of the claim node to extract representation from
            
        Returns:
            Logits for verdict classification [num_classes]
        """
        # Project input features to hidden dimension
        h = self.input_projection(node_features)  # [num_nodes, hidden_dim]
        
        # Apply GNN layers with message passing
        if DGL_AVAILABLE and hasattr(graph, 'num_nodes'):
            # Use actual GNN layers with DGL graph
            for layer in self.gnn_layers:
                h = layer(graph, h)
                h = self.dropout_layer(h)
        else:
            # Fallback: simple feedforward without graph structure
            for layer in self.gnn_layers:
                h = F.relu(layer(h))
                h = self.dropout_layer(h)
        
        # Extract claim node representation
        claim_representation = h[claim_node_idx]  # [hidden_dim]
        
        # Apply classifier
        logits = self.classifier(claim_representation)  # [num_classes]
        
        return logits
    
    def predict(self, graph, node_features: torch.Tensor, claim_node_idx: int) -> Dict[str, float]:
        """Predict verdict with confidence scores.
        
        Args:
            graph: DGL graph
            node_features: Node feature matrix
            claim_node_idx: Index of claim node
            
        Returns:
            Dictionary with verdict label and confidence scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(graph, node_features, claim_node_idx)
            probabilities = F.softmax(logits, dim=0)
            
            # Map to verdict labels
            labels = ["supported", "refuted", "not_enough_info"]
            confidence_scores = {
                labels[i]: float(probabilities[i])
                for i in range(len(labels))
            }
            
            # Get predicted label
            predicted_idx = torch.argmax(probabilities).item()
            predicted_label = labels[predicted_idx]
            
            return {
                "label": predicted_label,
                "confidence_scores": confidence_scores
            }


class NodeFeatureExtractor:
    """Extract node features using XLM-RoBERTa embeddings."""
    
    def __init__(self, model_name: str = "xlm-roberta-base", device: str = "cuda"):
        """Initialize feature extractor.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run model on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized feature extractor with {model_name} on {self.device}")
    
    def extract_features(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """Extract embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Tensor of embeddings [num_texts, embedding_dim]
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, embedding_dim]
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)


class GNNTrainer:
    """Trainer for GNN verdict predictor."""
    
    def __init__(
        self,
        model: GNNVerdictPredictor,
        feature_extractor: NodeFeatureExtractor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda"
    ):
        """Initialize trainer.
        
        Args:
            model: GNN model to train
            feature_extractor: Node feature extractor
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.feature_extractor = feature_extractor
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized GNN trainer on {self.device}")
    
    def prepare_graph_data(self, graph: KnowledgeGraph, claim_node_id: str) -> Tuple[Any, torch.Tensor, int]:
        """Prepare graph data for training/inference.
        
        Args:
            graph: Knowledge graph
            claim_node_id: ID of claim node
            
        Returns:
            Tuple of (DGL graph, node features, claim node index)
        """
        # Convert to DGL graph if available
        if DGL_AVAILABLE:
            from src.graph_builder import GraphBuilder
            builder = GraphBuilder()
            dgl_graph = builder.to_dgl_graph(graph)
            
            if dgl_graph is None:
                raise ValueError("Failed to convert knowledge graph to DGL format")
        else:
            # Fallback: create dummy graph structure
            dgl_graph = None
        
        # Extract node texts for feature extraction
        if DGL_AVAILABLE and dgl_graph is not None:
            node_ids = dgl_graph.ndata['node_id']
            node_texts = []
            claim_node_idx = None
            
            for i, node_id in enumerate(node_ids):
                node = graph.get_node(node_id)
                if node:
                    node_texts.append(node.text)
                    if node_id == claim_node_id:
                        claim_node_idx = i
                else:
                    node_texts.append("")  # Fallback for missing nodes
            
            if claim_node_idx is None:
                raise ValueError(f"Claim node {claim_node_id} not found in graph")
        else:
            # Fallback: use all nodes from knowledge graph
            node_texts = []
            claim_node_idx = None
            
            for i, (node_id, node) in enumerate(graph.nodes.items()):
                node_texts.append(node.text)
                if node_id == claim_node_id:
                    claim_node_idx = i
            
            if claim_node_idx is None:
                raise ValueError(f"Claim node {claim_node_id} not found in graph")
        
        # Extract node features
        node_features = self.feature_extractor.extract_features(node_texts)
        
        if DGL_AVAILABLE and dgl_graph is not None:
            return dgl_graph.to(self.device), node_features.to(self.device), claim_node_idx
        else:
            return None, node_features.to(self.device), claim_node_idx
    
    def train_step(self, graph: KnowledgeGraph, claim_node_id: str, label: str) -> float:
        """Single training step.
        
        Args:
            graph: Knowledge graph
            claim_node_id: ID of claim node
            label: Ground truth label
            
        Returns:
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare data
        dgl_graph, node_features, claim_node_idx = self.prepare_graph_data(graph, claim_node_id)
        
        # Convert label to tensor
        label_map = {"supported": 0, "refuted": 1, "not_enough_info": 2}
        label_tensor = torch.tensor(label_map[label], dtype=torch.long).to(self.device)
        
        # Forward pass
        logits = self.model(dgl_graph, node_features, claim_node_idx)
        
        # Compute loss
        loss = self.criterion(logits.unsqueeze(0), label_tensor.unsqueeze(0))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, test_data: List[Tuple[KnowledgeGraph, str, str]]) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            test_data: List of (graph, claim_node_id, label) tuples
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for graph, claim_node_id, true_label in test_data:
                try:
                    dgl_graph, node_features, claim_node_idx = self.prepare_graph_data(graph, claim_node_id)
                    
                    # Get prediction
                    result = self.model.predict(dgl_graph, node_features, claim_node_idx)
                    predicted_label = result["label"]
                    
                    predictions.append(predicted_label)
                    true_labels.append(true_label)
                    
                    if predicted_label == true_label:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate sample: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate per-class metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        if len(set(true_labels)) > 1:  # Only if we have multiple classes
            report = classification_report(
                true_labels, predictions,
                target_names=["supported", "refuted", "not_enough_info"],
                output_dict=True,
                zero_division=0
            )
            
            return {
                "accuracy": accuracy,
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "f1": report["macro avg"]["f1-score"],
                "per_class": report
            }
        else:
            return {"accuracy": accuracy}
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'num_classes': self.model.num_classes,
                'dropout': self.model.dropout,
                'aggregation': self.model.aggregation
            }
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved model checkpoint to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded model checkpoint from {path}")


class VerdictPredictor:
    """High-level interface for GNN-based verdict prediction."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """Initialize verdict predictor.
        
        Args:
            model_path: Path to trained model (if None, uses untrained model)
            device: Device to run on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.feature_extractor = NodeFeatureExtractor(device=self.device)
        
        # Load model configuration from checkpoint if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint.get('model_config', {})
            
            # Create model with saved configuration
            self.model = GNNVerdictPredictor(**model_config)
            self.trainer = GNNTrainer(self.model, self.feature_extractor, device=self.device)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"Loaded trained model from {model_path}")
        else:
            # Use default configuration
            self.model = GNNVerdictPredictor()
            self.trainer = GNNTrainer(self.model, self.feature_extractor, device=self.device)
            logger.warning("No trained model loaded - using untrained model")
    
    def predict_verdict(self, graph: KnowledgeGraph, claim: Claim) -> Verdict:
        """Predict verdict for a claim using the reasoning graph.
        
        Args:
            graph: Reasoning graph containing evidence and entities
            claim: Claim to verify
            
        Returns:
            Verdict with prediction and confidence scores
        """
        try:
            # Find claim node in graph
            claim_node_id = None
            for node_id, node in graph.nodes.items():
                if node.type == "claim" and node.attributes.get('claim_id') == claim.id:
                    claim_node_id = node_id
                    break
            
            if claim_node_id is None:
                # Add claim node if not present
                from src.graph_builder import GraphBuilder
                builder = GraphBuilder()
                claim_node_id = builder.add_claim_node(graph, claim.text, claim.id)
            
            # Prepare graph data
            dgl_graph, node_features, claim_node_idx = self.trainer.prepare_graph_data(graph, claim_node_id)
            
            # Get prediction
            result = self.model.predict(dgl_graph, node_features, claim_node_idx)
            
            # Create verdict
            verdict = Verdict(
                claim_id=claim.id,
                label=result["label"],
                confidence_scores=result["confidence_scores"],
                supporting_evidence=[],  # Will be filled by caller
                refuting_evidence=[],    # Will be filled by caller
                explanation="",          # Will be filled by RAG module
                reasoning_trace=[],      # Will be filled by ReAct agent
                quality_score=max(result["confidence_scores"].values())  # Use max confidence as quality
            )
            
            logger.info(f"Predicted verdict: {result['label']} with confidence {verdict.quality_score:.3f}")
            return verdict
            
        except Exception as e:
            logger.error(f"Failed to predict verdict: {e}")
            # Return default verdict
            return Verdict(
                claim_id=claim.id,
                label="not_enough_info",
                confidence_scores={
                    "supported": 0.33,
                    "refuted": 0.33,
                    "not_enough_info": 0.34
                },
                quality_score=0.0
            )
    
    def train(self, training_data: List[Tuple[KnowledgeGraph, str, str]], epochs: int = 10) -> None:
        """Train the GNN model.
        
        Args:
            training_data: List of (graph, claim_node_id, label) tuples
            epochs: Number of training epochs
        """
        logger.info(f"Starting GNN training for {epochs} epochs on {len(training_data)} samples")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for graph, claim_node_id, label in training_data:
                try:
                    loss = self.trainer.train_step(graph, claim_node_id, label)
                    total_loss += loss
                    num_batches += 1
                except Exception as e:
                    logger.warning(f"Failed to train on sample: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("GNN training completed")
    
    def save(self, path: str) -> None:
        """Save trained model.
        
        Args:
            path: Path to save model
        """
        self.trainer.save_model(path)