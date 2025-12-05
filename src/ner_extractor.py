"""Named Entity Recognition module for Vietnamese and English text."""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a named entity extracted from text."""
    
    text: str
    type: str  # PERSON, ORG, LOC, DATE, NUMBER
    start_idx: int
    end_idx: int
    confidence: float = 1.0
    language: str = "vi"
    
    def __post_init__(self):
        """Validate entity data."""
        valid_types = ["PERSON", "ORG", "LOC", "DATE", "NUMBER", "MISC"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid entity type: {self.type}. Must be one of {valid_types}")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        
        if self.start_idx < 0 or self.end_idx < 0:
            raise ValueError("Indices cannot be negative")
        
        if self.start_idx > self.end_idx:
            raise ValueError("start_idx cannot be greater than end_idx")
    
    def to_dict(self) -> Dict:
        """Convert entity to dictionary."""
        return {
            'text': self.text,
            'type': self.type,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'confidence': self.confidence,
            'language': self.language
        }


class NERExtractor:
    """Named Entity Recognition extractor supporting Vietnamese and English."""
    
    def __init__(self):
        """Initialize NER models."""
        self.vi_model = None
        self.en_model = None
        self._load_models()
    
    def _load_models(self):
        """Load NER models for Vietnamese and English."""
        try:
            # Try to load spaCy for English NER
            import spacy
            try:
                self.en_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy English NER model")
            except OSError:
                logger.warning("spaCy en_core_web_sm not found. Install with: python -m spacy download en_core_web_sm")
                self.en_model = None
        except ImportError:
            logger.warning("spaCy not installed. English NER will use rule-based fallback")
            self.en_model = None
        
        # For Vietnamese, we'll use a rule-based approach initially
        # In production, this would be replaced with fine-tuned PhoBERT
        logger.info("Using rule-based Vietnamese NER (PhoBERT fine-tuning pending)")
        self.vi_model = "rule-based"
    
    def extract_entities(self, text: str, language: str = "vi") -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            language: Language code ("vi" or "en")
        
        Returns:
            List of Entity objects
        """
        if not text or not text.strip():
            return []
        
        if language == "en":
            return self._extract_english_entities(text)
        elif language == "vi":
            return self._extract_vietnamese_entities(text)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _extract_english_entities(self, text: str) -> List[Entity]:
        """Extract entities from English text using spaCy."""
        entities = []
        
        if self.en_model is not None:
            # Use spaCy NER
            doc = self.en_model(text)
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entities.append(Entity(
                        text=ent.text,
                        type=entity_type,
                        start_idx=ent.start_char,
                        end_idx=ent.end_char,
                        confidence=1.0,
                        language="en"
                    ))
        else:
            # Fallback to rule-based extraction
            entities = self._rule_based_extraction(text, "en")
        
        return entities
    
    def _extract_vietnamese_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from Vietnamese text.
        
        Note: This is a rule-based implementation. In production, this should be
        replaced with a fine-tuned PhoBERT model for Vietnamese NER.
        """
        # For now, use rule-based extraction
        # TODO: Replace with fine-tuned PhoBERT model
        entities = self._rule_based_extraction(text, "vi")
        
        return entities
    
    def _rule_based_extraction(self, text: str, language: str) -> List[Entity]:
        """
        Rule-based entity extraction as fallback.
        
        This is a simple implementation that uses patterns to detect entities.
        """
        entities = []
        
        # Extract dates (simple patterns)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or DD-MM-YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD or YYYY-MM-DD
            r'(?:ngày|tháng|năm)\s+\d{1,2}',   # Vietnamese date words
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    type="DATE",
                    start_idx=match.start(),
                    end_idx=match.end(),
                    confidence=0.8,
                    language=language
                ))
        
        # Extract numbers
        number_pattern = r'\b\d+(?:[.,]\d+)?\s*(?:triệu|tỷ|nghìn|million|billion|thousand)?\b'
        for match in re.finditer(number_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(),
                type="NUMBER",
                start_idx=match.start(),
                end_idx=match.end(),
                confidence=0.9,
                language=language
            ))
        
        # Extract capitalized phrases (potential names/organizations)
        # This is very basic and would need improvement
        if language == "en":
            cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
            for match in re.finditer(cap_pattern, text):
                # Heuristic: 2 words = likely PERSON, 3+ words = likely ORG
                word_count = len(match.group().split())
                entity_type = "PERSON" if word_count == 2 else "ORG"
                
                entities.append(Entity(
                    text=match.group(),
                    type=entity_type,
                    start_idx=match.start(),
                    end_idx=match.end(),
                    confidence=0.6,
                    language=language
                ))
        
        # Remove overlapping entities (keep higher confidence)
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[str]:
        """Map spaCy entity labels to our entity types."""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'LOC',  # Geopolitical entity
            'LOC': 'LOC',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'CARDINAL': 'NUMBER',
            'QUANTITY': 'NUMBER',
            'MONEY': 'NUMBER',
            'PERCENT': 'NUMBER',
        }
        return mapping.get(label)
    
    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping higher confidence ones."""
        if not entities:
            return []
        
        # Sort by start index
        sorted_entities = sorted(entities, key=lambda e: (e.start_idx, -e.confidence))
        
        result = []
        last_end = -1
        
        for entity in sorted_entities:
            if entity.start_idx >= last_end:
                result.append(entity)
                last_end = entity.end_idx
        
        return result
    
    def merge_duplicate_entities(self, entities: List[Entity], threshold: float = 0.85) -> List[Entity]:
        """
        Merge duplicate entities based on text similarity.
        
        Args:
            entities: List of entities to merge
            threshold: Similarity threshold for merging (0-1)
        
        Returns:
            List of merged entities
        """
        if not entities:
            return []
        
        # Group entities by type
        by_type: Dict[str, List[Entity]] = {}
        for entity in entities:
            if entity.type not in by_type:
                by_type[entity.type] = []
            by_type[entity.type].append(entity)
        
        merged = []
        
        for entity_type, entity_list in by_type.items():
            # For each type, merge similar entities
            seen = set()
            for entity in entity_list:
                if entity.text.lower() in seen:
                    continue
                
                # Check for similar entities
                similar_found = False
                for existing_text in seen:
                    if self._text_similarity(entity.text.lower(), existing_text) >= threshold:
                        similar_found = True
                        break
                
                if not similar_found:
                    merged.append(entity)
                    seen.add(entity.text.lower())
        
        return merged
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple character-based approach.
        
        For production, consider using Levenshtein distance or embedding similarity.
        """
        if text1 == text2:
            return 1.0
        
        # Simple Jaccard similarity on character bigrams
        def get_bigrams(s: str) -> set:
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(text1)
        bigrams2 = get_bigrams(text2)
        
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0


# Convenience function
def extract_entities(text: str, language: str = "vi") -> List[Entity]:
    """
    Extract named entities from text.
    
    Args:
        text: Input text
        language: Language code ("vi" or "en")
    
    Returns:
        List of Entity objects
    """
    extractor = NERExtractor()
    return extractor.extract_entities(text, language)
