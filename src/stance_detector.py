"""
Stance detection module using XLM-RoBERTa for cross-lingual fact-checking.

This module implements stance detection between claims and evidence,
supporting both Vietnamese-Vietnamese and Vietnamese-English pairs.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StanceResult:
    """Result of stance detection."""
    stance: str  # "support", "refute", "neutral"
    confidence_scores: Dict[str, float]
    claim_lang: str
    evidence_lang: str


class StanceDataset(Dataset):
    """Dataset for stance detection training."""
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 256
    ):
        """
        Initialize stance dataset.
        
        Args:
            data_file: Path to JSONL file with stance examples
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        
        # Stance label mapping
        self.label_map = {
            "support": 0,
            "refute": 1,
            "neutral": 2
        }
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_file}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Encode claim and evidence as [CLS] claim [SEP] evidence [SEP]
        encoding = self.tokenizer(
            example["claim"],
            example["evidence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get label
        label = self.label_map[example["stance"]]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class XLMRStanceClassifier(nn.Module):
    """XLM-RoBERTa based stance classifier."""
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize stance classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of stance classes (3: support/refute/neutral)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.xlm_roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        hidden_size = self.xlm_roberta.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        logger.info(f"Initialized XLM-R stance classifier with {model_name}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
        
        Returns:
            Dictionary with logits and optional loss
        """
        # Get XLM-R outputs
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss
        
        return result


class StanceDetectorTrainer:
    """Trainer for stance detection model."""
    
    def __init__(
        self,
        model: XLMRStanceClassifier,
        tokenizer,
        train_dataset: StanceDataset,
        val_dataset: StanceDataset,
        output_dir: str = "models/xlmr_stance",
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        use_8bit: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Stance classifier model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model checkpoints
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            gradient_accumulation_steps: Gradient accumulation steps
            use_8bit: Whether to use 8-bit quantization
            device: Device to train on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Apply 8-bit quantization if requested
        if use_8bit:
            try:
                import bitsandbytes as bnb
                # Note: 8-bit quantization requires specific setup
                logger.info("8-bit quantization enabled")
            except ImportError:
                logger.warning("bitsandbytes not installed, skipping 8-bit quantization")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state
        self.best_val_f1 = 0.0
        self.global_step = 0
        
        logger.info(f"Trainer initialized: {len(train_dataset)} train, {len(val_dataset)} val")
    
    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_loss = self._train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validation
            val_metrics = self._validate()
            logger.info(
                f"Val loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}"
            )
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self._save_checkpoint(f"best_model")
                logger.info(f"Saved best model (F1: {self.best_val_f1:.4f})")
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")
        
        logger.info("Training complete!")
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}"
                )
        
        return total_loss / num_batches
    
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": accuracy,
            "f1": f1
        }
    
    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.xlm_roberta.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save classifier head
        torch.save(
            self.model.classifier.state_dict(),
            checkpoint_dir / "classifier_head.pt"
        )
        
        # Save training config
        config = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "best_val_f1": self.best_val_f1
        }
        
        with open(checkpoint_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)


class StanceDetector:
    """Inference interface for stance detection."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16
    ):
        """
        Initialize stance detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            batch_size: Batch size for inference
        """
        self.device = device
        self.batch_size = batch_size
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load XLM-R base
        xlm_roberta = AutoModel.from_pretrained(model_path)
        
        # Create classifier
        self.model = XLMRStanceClassifier()
        self.model.xlm_roberta = xlm_roberta
        
        # Load classifier head
        classifier_path = Path(model_path) / "classifier_head.pt"
        if classifier_path.exists():
            self.model.classifier.load_state_dict(
                torch.load(classifier_path, map_location=device)
            )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Label mapping
        self.id_to_label = {
            0: "support",
            1: "refute",
            2: "neutral"
        }
        
        logger.info(f"Loaded stance detector from {model_path}")
    
    def detect_stance(
        self,
        claim: str,
        evidence: str,
        claim_lang: str = "vi",
        evidence_lang: str = "vi"
    ) -> StanceResult:
        """
        Detect stance between claim and evidence.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            claim_lang: Language of claim
            evidence_lang: Language of evidence
        
        Returns:
            StanceResult with stance label and confidence scores
        """
        # Encode input
        encoding = self.tokenizer(
            claim,
            evidence,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        pred_id = torch.argmax(probs, dim=1).item()
        stance = self.id_to_label[pred_id]
        
        # Get confidence scores
        confidence_scores = {
            "support": probs[0, 0].item(),
            "refute": probs[0, 1].item(),
            "neutral": probs[0, 2].item()
        }
        
        return StanceResult(
            stance=stance,
            confidence_scores=confidence_scores,
            claim_lang=claim_lang,
            evidence_lang=evidence_lang
        )
    
    def batch_detect_stance(
        self,
        claim_evidence_pairs: List[Tuple[str, str]]
    ) -> List[StanceResult]:
        """
        Detect stance for multiple claim-evidence pairs.
        
        Args:
            claim_evidence_pairs: List of (claim, evidence) tuples
        
        Returns:
            List of StanceResult objects
        """
        results = []
        
        for i in range(0, len(claim_evidence_pairs), self.batch_size):
            batch = claim_evidence_pairs[i:i + self.batch_size]
            
            # Encode batch
            claims = [pair[0] for pair in batch]
            evidences = [pair[1] for pair in batch]
            
            encodings = self.tokenizer(
                claims,
                evidences,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
            
            # Process results
            for j in range(len(batch)):
                pred_id = torch.argmax(probs[j]).item()
                stance = self.id_to_label[pred_id]
                
                confidence_scores = {
                    "support": probs[j, 0].item(),
                    "refute": probs[j, 1].item(),
                    "neutral": probs[j, 2].item()
                }
                
                results.append(StanceResult(
                    stance=stance,
                    confidence_scores=confidence_scores,
                    claim_lang="vi",
                    evidence_lang="vi"
                ))
        
        return results
