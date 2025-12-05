"""
Train XLM-RoBERTa stance detection model.
"""

import logging
import argparse
import torch
from transformers import AutoTokenizer

from src.stance_detector import (
    XLMRStanceClassifier,
    StanceDataset,
    StanceDetectorTrainer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train XLM-RoBERTa stance detection model"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/stance_detection/stance_train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/stance_detection/stance_val.jsonl",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/xlmr_stance",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-base",
        help="Base model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    
    args = parser.parse_args()
    
    logger.info("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = XLMRStanceClassifier(model_name=args.model_name)
    
    logger.info("Loading datasets")
    train_dataset = StanceDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = StanceDataset(
        args.val_file,
        tokenizer,
        max_length=args.max_length
    )
    
    logger.info("Initializing trainer")
    trainer = StanceDetectorTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_8bit=args.use_8bit
    )
    
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
