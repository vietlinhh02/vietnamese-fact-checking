"""PhoBERT-based claim classifier for Vietnamese text."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import json

from src.claim_data_preparation import LabeledSentence, ClaimDatasetBuilder

logger = logging.getLogger(__name__)


class ClaimDataset(Dataset):
    """PyTorch dataset for claim detection."""
    
    def __init__(
        self,
        sentences: List[LabeledSentence],
        tokenizer,
        max_length: int = 256
    ):
        """Initialize dataset.
        
        Args:
            sentences: List of labeled sentences
            tokenizer: PhoBERT tokenizer
            max_length: Maximum sequence length
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sentence = self.sentences[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sentence.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Binary label: 1 for claim, 0 for non-claim
        label = 1 if sentence.is_claim else 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class PhoBERTClaimClassifier(nn.Module):
    """PhoBERT model with classification head for claim detection."""
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 2,
        dropout: float = 0.1
    ):
        """Initialize classifier.
        
        Args:
            model_name: Name of pretrained PhoBERT model
            num_labels: Number of output classes (2 for binary)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_labels)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary with logits and optional loss
        """
        # Get PhoBERT outputs
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
        
        return result


class ClaimClassifierTrainer:
    """Trainer for PhoBERT claim classifier."""
    
    def __init__(
        self,
        model: PhoBERTClaimClassifier,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = "models/claim_classifier"
    ):
        """Initialize trainer.
        
        Args:
            model: PhoBERT classifier model
            tokenizer: PhoBERT tokenizer
            device: Device to train on
            output_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_f1 = 0.0
    
    def train(
        self,
        train_data: List[LabeledSentence],
        val_data: List[LabeledSentence],
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        max_length: int = 256,
        use_fp16: bool = False
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            batch_size: Batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_length: Maximum sequence length
            use_fp16: Whether to use mixed precision (FP16)
            
        Returns:
            Dictionary with training history
        """
        # Create datasets
        train_dataset = ClaimDataset(train_data, self.tokenizer, max_length)
        val_dataset = ClaimDataset(val_data, self.tokenizer, max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup mixed precision if requested
        scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                scaler,
                use_fp16
            )
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self._validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Save best model based on F1 score
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_checkpoint('best_model')
                logger.info(f"Saved best model with F1: {self.best_f1:.4f}")
        
        return history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler,
        scaler,
        use_fp16: bool
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if use_fp16 and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        
        # Compute F1 score
        tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.phobert.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save classifier head separately
        torch.save(
            self.model.classifier.state_dict(),
            checkpoint_dir / 'classifier_head.pt'
        )
        
        # Save training info
        info = {
            'best_f1': self.best_f1,
            'model_name': 'vinai/phobert-base'
        }
        with open(checkpoint_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_name: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to load
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint {checkpoint_dir} does not exist")
        
        # Load PhoBERT
        self.model.phobert = AutoModel.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # Load classifier head
        classifier_path = checkpoint_dir / 'classifier_head.pt'
        if classifier_path.exists():
            self.model.classifier.load_state_dict(
                torch.load(classifier_path, map_location=self.device)
            )
        
        self.model = self.model.to(self.device)
        
        # Load training info
        info_path = checkpoint_dir / 'training_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.best_f1 = info.get('best_f1', 0.0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_dir}")


def train_claim_classifier(
    data_dir: str = "data/claim_detection",
    output_dir: str = "models/claim_classifier",
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    use_fp16: bool = False
) -> None:
    """Train PhoBERT claim classifier.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save model
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        use_fp16: Whether to use mixed precision
    """
    # Load data
    builder = ClaimDatasetBuilder(data_dir)
    train_data, val_data, test_data = builder.load_dataset()
    
    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val samples")
    
    # Initialize model and tokenizer
    logger.info("Loading PhoBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = PhoBERTClaimClassifier()
    
    # Initialize trainer
    trainer = ClaimClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_fp16=use_fp16
    )
    
    # Save final model
    trainer.save_checkpoint('final_model')
    
    # Save training history
    history_path = Path(output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training complete. Best F1: {trainer.best_f1:.4f}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Train the model
    train_claim_classifier(
        batch_size=8,  # Smaller batch size for memory efficiency
        num_epochs=2,  # Fewer epochs for quick testing
        use_fp16=torch.cuda.is_available()
    )
