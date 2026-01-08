#!/usr/bin/env python3
"""Simple training script for GNN verdict predictor without NER dependencies."""

import os
import sys
import logging
import argparse
import json
from typing import List, Tuple, Dict, Any
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gnn_verdict_predictor import VerdictPredictor, GNNVerdictPredictor, NodeFeatureExtractor, GNNTrainer
from src.data_models import KnowledgeGraph, GraphNode, GraphEdge, Claim, Evidence
from src.logging_config import setup_logging

# Setup logging
logger = logging.getLogger(__name__)


def create_simple_training_data(num_samples: int = 20) -> List[Tuple[KnowledgeGraph, str, str]]:
    """Create simple training data without NER dependencies.
    
    Args:
        num_samples: Number of training samples to generate
        
    Returns:
        List of (graph, claim_node_id, label) tuples
    """
    training_data = []
    
    # Sample data templates
    templates = [
        # Supported claims
        {
            "claim": "Vietnam joined ASEAN in 1995",
            "evidence_texts": [
                "Vietnam officially became the seventh member of ASEAN on July 28, 1995",
                "ASEAN welcomed Vietnam as a new member in 1995"
            ],
            "stances": ["support", "support"],
            "label": "supported"
        },
        # Refuted claims
        {
            "claim": "Hanoi is the capital of Thailand",
            "evidence_texts": [
                "Bangkok is the capital and largest city of Thailand",
                "Hanoi is the capital of Vietnam, not Thailand"
            ],
            "stances": ["refute", "refute"],
            "label": "refuted"
        },
        # Not enough info
        {
            "claim": "Vietnam's population will reach 100 million by 2030",
            "evidence_texts": [
                "Vietnam's current population is approximately 97 million people",
                "Population growth rate in Vietnam has been slowing down"
            ],
            "stances": ["neutral", "neutral"],
            "label": "not_enough_info"
        }
    ]
    
    for i in range(num_samples):
        template = templates[i % len(templates)]
        
        # Create knowledge graph
        graph = KnowledgeGraph()
        
        # Add evidence nodes
        evidence_node_ids = []
        for j, (evidence_text, stance) in enumerate(zip(template["evidence_texts"], template["stances"])):
            evidence_node_id = f"evidence_{i}_{j}"
            evidence_node = GraphNode(
                id=evidence_node_id,
                type="evidence",
                text=evidence_text,
                attributes={
                    'credibility_score': 0.8,
                    'stance': stance,
                    'stance_confidence': 0.9,
                    'source_url': f'https://example.com/article_{i}_{j}'
                }
            )
            graph.add_node(evidence_node)
            evidence_node_ids.append(evidence_node_id)
        
        # Add some entity nodes
        entity_node_ids = []
        entities = ["Vietnam", "ASEAN", "Thailand", "Bangkok", "Hanoi"]
        for j, entity in enumerate(entities[:2]):  # Just add 2 entities per graph
            entity_node_id = f"entity_{i}_{j}"
            entity_node = GraphNode(
                id=entity_node_id,
                type="entity",
                text=entity,
                attributes={
                    'entity_type': 'LOCATION',
                    'confidence': 0.9
                }
            )
            graph.add_node(entity_node)
            entity_node_ids.append(entity_node_id)
        
        # Add claim node
        claim_node_id = f"claim_{i}"
        claim_node = GraphNode(
            id=claim_node_id,
            type="claim",
            text=template["claim"],
            attributes={'claim_id': f'claim_{i}'}
        )
        graph.add_node(claim_node)
        
        # Add edges between evidence and claim
        for evidence_id in evidence_node_ids:
            evidence_node = graph.get_node(evidence_id)
            stance = evidence_node.attributes['stance']
            
            if stance == "support":
                relation = "supports"
            elif stance == "refute":
                relation = "refutes"
            else:
                relation = "mentions"
            
            edge = GraphEdge(
                source_id=evidence_id,
                target_id=claim_node_id,
                relation=relation,
                weight=0.8,
                evidence_source=evidence_node.attributes['source_url']
            )
            graph.add_edge(edge)
        
        # Add some edges between evidence and entities
        for evidence_id in evidence_node_ids:
            for entity_id in entity_node_ids:
                edge = GraphEdge(
                    source_id=evidence_id,
                    target_id=entity_id,
                    relation="mentions",
                    weight=0.6,
                    evidence_source=f'https://example.com/article_{i}'
                )
                graph.add_edge(edge)
        
        training_data.append((graph, claim_node_id, template["label"]))
    
    logger.info(f"Created {len(training_data)} simple training samples")
    return training_data


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN verdict predictor (simple version)")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="models/gnn_simple.pt", help="Output model path")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GNN hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting simple GNN training")
    
    # Create training data
    logger.info(f"Creating {args.num_samples} training samples")
    training_data = create_simple_training_data(args.num_samples)
    
    # Split data
    split_idx = int(len(training_data) * (1 - args.test_split))
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]
    
    if not test_data:
        test_data = train_data[:1]  # Use at least one sample for testing
    
    logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Initialize model
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create model with custom parameters
    model = GNNVerdictPredictor(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        embedding_dim=768  # XLM-R base size
    )
    
    feature_extractor = NodeFeatureExtractor(device=device)
    trainer = GNNTrainer(
        model=model,
        feature_extractor=feature_extractor,
        learning_rate=args.learning_rate,
        device=device
    )
    
    predictor = VerdictPredictor(device=device)
    predictor.trainer = trainer
    predictor.model = model
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs")
    predictor.train(train_data, epochs=args.epochs)
    
    # Evaluate model
    if test_data:
        logger.info("Evaluating model on test data")
        try:
            metrics = predictor.trainer.evaluate(test_data)
            logger.info(f"Test metrics: {metrics}")
            
            # Save metrics
            metrics_path = args.output.replace('.pt', '_metrics.json')
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
    
    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    predictor.save(args.output)
    logger.info(f"Saved trained model to {args.output}")
    
    # Test inference
    logger.info("Testing inference on first sample")
    test_graph, test_claim_id, true_label = test_data[0]
    
    # Create dummy claim for testing
    test_claim = Claim(text="Test claim", id=test_claim_id.split('_')[-1])
    
    try:
        verdict = predictor.predict_verdict(test_graph, test_claim)
        logger.info(f"Predicted: {verdict.label}, True: {true_label}")
        logger.info(f"Confidence scores: {verdict.confidence_scores}")
        
        # Check if prediction is reasonable
        if verdict.label in ["supported", "refuted", "not_enough_info"]:
            logger.info("✓ Prediction format is correct")
        else:
            logger.warning("✗ Prediction format is incorrect")
            
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()