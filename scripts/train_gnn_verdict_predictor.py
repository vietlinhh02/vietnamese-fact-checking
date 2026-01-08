#!/usr/bin/env python3
"""Training script for GNN-based verdict predictor."""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gnn_verdict_predictor import VerdictPredictor, GNNVerdictPredictor, NodeFeatureExtractor, GNNTrainer
from src.data_models import KnowledgeGraph, Claim, Evidence
from src.graph_builder import GraphBuilder
from src.logging_config import setup_logging
from src.config import SystemConfig

# Setup logging
logger = logging.getLogger(__name__)


def create_synthetic_training_data(num_samples: int = 100) -> List[Tuple[KnowledgeGraph, str, str]]:
    """Create synthetic training data for GNN training.
    
    Args:
        num_samples: Number of training samples to generate
        
    Returns:
        List of (graph, claim_node_id, label) tuples
    """
    training_data = []
    builder = GraphBuilder()
    
    # Sample claims and evidence for different verdict types
    sample_data = [
        # Supported claims
        {
            "claim": "Việt Nam là thành viên ASEAN từ năm 1995",
            "evidence": [
                {"text": "Việt Nam chính thức gia nhập ASEAN vào ngày 28 tháng 7 năm 1995", "stance": "support"},
                {"text": "ASEAN đã chào đón Việt Nam trở thành thành viên thứ 7 vào năm 1995", "stance": "support"}
            ],
            "label": "supported"
        },
        # Refuted claims  
        {
            "claim": "Hà Nội là thủ đô của Thái Lan",
            "evidence": [
                {"text": "Bangkok là thủ đô và thành phố lớn nhất của Thái Lan", "stance": "refute"},
                {"text": "Hà Nội là thủ đô của Việt Nam, không phải Thái Lan", "stance": "refute"}
            ],
            "label": "refuted"
        },
        # Not enough info
        {
            "claim": "Dân số Việt Nam sẽ đạt 100 triệu người vào năm 2030",
            "evidence": [
                {"text": "Dân số Việt Nam hiện tại khoảng 97 triệu người", "stance": "neutral"},
                {"text": "Tốc độ tăng dân số Việt Nam đang chậm lại", "stance": "neutral"}
            ],
            "label": "not_enough_info"
        }
    ]
    
    # Generate multiple variations of each sample
    for i in range(num_samples):
        sample = sample_data[i % len(sample_data)]
        
        # Create claim
        claim = Claim(
            text=sample["claim"],
            id=f"claim_{i}",
            language="vi"
        )
        
        # Create evidence
        evidence_list = []
        for j, ev_data in enumerate(sample["evidence"]):
            evidence = Evidence(
                text=ev_data["text"],
                source_url=f"https://example.com/article_{i}_{j}",
                source_title=f"Article {i}_{j}",
                credibility_score=0.8,
                language="vi",
                stance=ev_data["stance"],
                stance_confidence=0.9,
                id=f"evidence_{i}_{j}"
            )
            evidence_list.append(evidence)
        
        # Build graph
        graph = builder.build_graph(evidence_list)
        claim_node_id = builder.add_claim_node(graph, claim.text, claim.id)
        
        training_data.append((graph, claim_node_id, sample["label"]))
    
    logger.info(f"Created {len(training_data)} synthetic training samples")
    return training_data


def load_training_data_from_file(file_path: str) -> List[Tuple[KnowledgeGraph, str, str]]:
    """Load training data from JSONL file.
    
    Args:
        file_path: Path to JSONL file with training data
        
    Returns:
        List of (graph, claim_node_id, label) tuples
    """
    training_data = []
    builder = GraphBuilder()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Parse claim
                    claim_data = data['claim']
                    claim = Claim.from_dict(claim_data)
                    
                    # Parse evidence
                    evidence_list = []
                    for ev_data in data['evidence']:
                        evidence = Evidence.from_dict(ev_data)
                        evidence_list.append(evidence)
                    
                    # Build graph
                    graph = builder.build_graph(evidence_list)
                    claim_node_id = builder.add_claim_node(graph, claim.text, claim.id)
                    
                    # Get label
                    label = data['label']
                    
                    training_data.append((graph, claim_node_id, label))
                    
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(training_data)} training samples from {file_path}")
        
    except FileNotFoundError:
        logger.error(f"Training data file not found: {file_path}")
        return []
    
    return training_data


def evaluate_model(predictor: VerdictPredictor, test_data: List[Tuple[KnowledgeGraph, str, str]]) -> Dict[str, float]:
    """Evaluate trained model on test data.
    
    Args:
        predictor: Trained verdict predictor
        test_data: Test data
        
    Returns:
        Evaluation metrics
    """
    return predictor.trainer.evaluate(test_data)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN verdict predictor")
    parser.add_argument("--data", type=str, help="Path to training data JSONL file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic training data")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--output", type=str, default="models/gnn_verdict_predictor.pt", help="Output model path")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--hidden_dim", type=int, default=256, help="GNN hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting GNN verdict predictor training")
    
    # Load or create training data
    if args.data and os.path.exists(args.data):
        logger.info(f"Loading training data from {args.data}")
        training_data = load_training_data_from_file(args.data)
    elif args.synthetic:
        logger.info(f"Creating {args.num_samples} synthetic training samples")
        training_data = create_synthetic_training_data(args.num_samples)
    else:
        logger.error("No training data provided. Use --data or --synthetic flag")
        return
    
    if not training_data:
        logger.error("No training data available")
        return
    
    # Split data
    if len(training_data) > 1:
        train_data, test_data = train_test_split(
            training_data, 
            test_size=args.test_split, 
            random_state=42,
            stratify=[label for _, _, label in training_data]
        )
    else:
        train_data = training_data
        test_data = training_data[:1]  # Use same data for testing
    
    logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Initialize model
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create model with custom parameters
    model = GNNVerdictPredictor(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
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
        metrics = evaluate_model(predictor, test_data)
        logger.info(f"Test metrics: {metrics}")
        
        # Save metrics
        metrics_path = args.output.replace('.pt', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
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
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()