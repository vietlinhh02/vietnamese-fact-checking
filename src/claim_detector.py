"""Claim detection and extraction pipeline for Vietnamese text."""

import torch
from typing import List, Optional, Tuple
import logging
from pathlib import Path
import re

from transformers import AutoTokenizer
from src.claim_classifier import PhoBERTClaimClassifier
from src.data_models import Claim

logger = logging.getLogger(__name__)


class VietnameseSentenceTokenizer:
    """Simple Vietnamese sentence tokenizer."""
    
    def __init__(self):
        """Initialize tokenizer."""
        # Vietnamese sentence ending patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
    
    def tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize text into sentences with positions.
        
        Args:
            text: Input text
            
        Returns:
            List of (sentence, start_idx, end_idx) tuples
        """
        sentences = []
        current_pos = 0
        
        # Split by sentence endings
        for match in self.sentence_endings.finditer(text):
            end_pos = match.end()
            sentence = text[current_pos:end_pos].strip()
            
            if sentence:
                sentences.append((sentence, current_pos, end_pos))
            
            current_pos = end_pos
        
        # Add remaining text as last sentence
        if current_pos < len(text):
            sentence = text[current_pos:].strip()
            if sentence:
                sentences.append((sentence, current_pos, len(text)))
        
        return sentences


class ClaimDetector:
    """Claim detection pipeline using PhoBERT."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.5,
        max_length: int = 256,
        batch_size: int = 8
    ):
        """Initialize claim detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            confidence_threshold: Minimum confidence for claim detection
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize sentence tokenizer
        self.sentence_tokenizer = VietnameseSentenceTokenizer()
        
        # Load model and tokenizer
        if model_path is None:
            # Use pretrained PhoBERT without fine-tuning (for testing)
            logger.warning("No model path provided, using base PhoBERT")
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.model = PhoBERTClaimClassifier()
        else:
            self._load_model(model_path)
        
        self.model = self.model.to(device)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> None:
        """Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        from transformers import AutoModel
        
        checkpoint_dir = Path(model_path)
        
        if not checkpoint_dir.exists():
            raise ValueError(f"Model path {checkpoint_dir} does not exist")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # Load model
        phobert = AutoModel.from_pretrained(checkpoint_dir)
        self.model = PhoBERTClaimClassifier()
        self.model.phobert = phobert
        
        # Load classifier head
        classifier_path = checkpoint_dir / 'classifier_head.pt'
        if classifier_path.exists():
            self.model.classifier.load_state_dict(
                torch.load(classifier_path, map_location=self.device)
            )
        
        logger.info(f"Loaded model from {checkpoint_dir}")
    
    def detect_claims(self, text: str) -> List[Claim]:
        """Detect claims in Vietnamese text.
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            List of detected claims with context
        """
        # Tokenize into sentences
        sentences = self.sentence_tokenizer.tokenize(text)
        
        if not sentences:
            return []
        
        # Process sentences in batches
        all_claims = []
        
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            batch_claims = self._process_batch(batch_sentences, text)
            all_claims.extend(batch_claims)
        
        return all_claims
    
    def _process_batch(
        self,
        sentences: List[Tuple[str, int, int]],
        full_text: str
    ) -> List[Claim]:
        """Process a batch of sentences.
        
        Args:
            sentences: List of (sentence, start_idx, end_idx) tuples
            full_text: Full input text for context extraction
            
        Returns:
            List of detected claims
        """
        # Tokenize batch
        sentence_texts = [s[0] for s in sentences]
        
        encodings = self.tokenizer(
            sentence_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
        
        # Extract claims
        claims = []
        
        for idx, (sentence_text, start_idx, end_idx) in enumerate(sentences):
            # Get probability of being a claim (class 1)
            claim_prob = probs[idx, 1].item()
            
            if claim_prob >= self.confidence_threshold:
                # Extract context (surrounding sentences)
                context = self._extract_context(full_text, start_idx, end_idx)
                
                # Determine sentence type based on confidence
                if claim_prob > 0.7:
                    sentence_type = "factual_claim"
                else:
                    sentence_type = "factual_claim"  # Still a claim, just lower confidence
                
                claim = Claim(
                    text=sentence_text,
                    context=context,
                    confidence=claim_prob,
                    sentence_type=sentence_type,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    language="vi"
                )
                
                claims.append(claim)
        
        return claims
    
    def _extract_context(
        self,
        text: str,
        start_idx: int,
        end_idx: int,
        context_sentences: int = 1
    ) -> str:
        """Extract surrounding context for a claim.
        
        Args:
            text: Full text
            start_idx: Start index of claim
            end_idx: End index of claim
            context_sentences: Number of sentences before/after to include
            
        Returns:
            Context string including the claim and surrounding sentences
        """
        # Get all sentences
        all_sentences = self.sentence_tokenizer.tokenize(text)
        
        # Find the claim sentence
        claim_idx = -1
        for idx, (sent, s_start, s_end) in enumerate(all_sentences):
            if s_start <= start_idx < s_end:
                claim_idx = idx
                break
        
        if claim_idx == -1:
            # Fallback: return just the claim
            return text[start_idx:end_idx]
        
        # Get context window
        context_start = max(0, claim_idx - context_sentences)
        context_end = min(len(all_sentences), claim_idx + context_sentences + 1)
        
        # Build context string
        context_parts = []
        for idx in range(context_start, context_end):
            context_parts.append(all_sentences[idx][0])
        
        return ' '.join(context_parts)
    
    def detect_claims_sliding_window(
        self,
        text: str,
        window_size: int = 512,
        stride: int = 256
    ) -> List[Claim]:
        """Detect claims in long documents using sliding window.
        
        This is useful for very long documents that exceed the model's
        maximum sequence length.
        
        Args:
            text: Input text
            window_size: Size of sliding window in characters
            stride: Stride between windows
            
        Returns:
            List of detected claims (deduplicated)
        """
        if len(text) <= window_size:
            # Text is short enough, process normally
            return self.detect_claims(text)
        
        # Process with sliding window
        all_claims = []
        seen_claims = set()  # For deduplication
        
        for start in range(0, len(text), stride):
            end = min(start + window_size, len(text))
            window_text = text[start:end]
            
            # Detect claims in window
            window_claims = self.detect_claims(window_text)
            
            # Adjust indices and deduplicate
            for claim in window_claims:
                # Adjust indices to full text
                claim.start_idx += start
                claim.end_idx += start
                
                # Deduplicate based on text and position
                claim_key = (claim.text, claim.start_idx)
                if claim_key not in seen_claims:
                    seen_claims.add(claim_key)
                    all_claims.append(claim)
            
            # Stop if we've reached the end
            if end >= len(text):
                break
        
        return all_claims


def detect_claims_in_text(
    text: str,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    use_sliding_window: bool = False
) -> List[Claim]:
    """Convenience function to detect claims in text.
    
    Args:
        text: Input Vietnamese text
        model_path: Path to trained model (optional)
        confidence_threshold: Minimum confidence for detection
        use_sliding_window: Whether to use sliding window for long texts
        
    Returns:
        List of detected claims
    """
    detector = ClaimDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    
    if use_sliding_window:
        return detector.detect_claims_sliding_window(text)
    else:
        return detector.detect_claims(text)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    sample_text = """
    Việt Nam có dân số khoảng 98 triệu người. Đây là một con số ấn tượng.
    Thủ đô Hà Nội là trung tâm chính trị của đất nước. Tôi nghĩ rằng Hà Nội rất đẹp.
    GDP của Việt Nam năm 2023 đạt 430 tỷ USD. Bạn có biết điều này không?
    """
    
    claims = detect_claims_in_text(sample_text)
    
    print(f"\nDetected {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"\n{i}. {claim.text}")
        print(f"   Confidence: {claim.confidence:.3f}")
        print(f"   Context: {claim.context}")
