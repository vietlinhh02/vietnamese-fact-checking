"""Relation extraction module for building knowledge graphs."""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

from src.ner_extractor import Entity

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class Relation:
    """Represents a relation between two entities."""
    
    subject: Entity
    relation_type: str
    object: Entity
    confidence: float = 1.0
    context: str = ""
    
    def __post_init__(self):
        """Validate relation data."""
        valid_relations = [
            "works_for", "located_in", "part_of", "member_of",
            "born_in", "died_in", "founded_by", "owns",
            "subsidiary_of", "parent_of", "married_to",
            "related_to", "mentions", "supports", "refutes"
        ]
        
        if self.relation_type not in valid_relations:
            # Allow custom relations but log a warning
            logger.debug(f"Using custom relation type: {self.relation_type}")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def to_dict(self) -> Dict:
        """Convert relation to dictionary."""
        return {
            'subject': self.subject.to_dict(),
            'relation_type': self.relation_type,
            'object': self.object.to_dict(),
            'confidence': self.confidence,
            'context': self.context
        }


class RelationExtractor:
    """Extract relations between entities from text."""
    
    def __init__(self):
        """Initialize relation extraction models."""
        self.xlmr_model = None
        self.dependency_parser = None
        self._load_models()
    
    def _load_models(self):
        """Load relation extraction models."""
        # For now, use rule-based and dependency parsing
        # TODO: Fine-tune XLM-R for relation classification
        logger.info("Using rule-based relation extraction (XLM-R fine-tuning pending)")
        
        # Try to load spaCy for dependency parsing
        try:
            import spacy
            try:
                self.dependency_parser = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy for dependency parsing")
            except OSError:
                logger.warning("spaCy model not found. Dependency parsing unavailable")
                self.dependency_parser = None
        except ImportError:
            logger.warning("spaCy not installed. Dependency parsing unavailable")
            self.dependency_parser = None
    
    def extract_relations(
        self,
        text: str,
        entities: List[Entity],
        language: str = "vi"
    ) -> List[Relation]:
        """
        Extract relations between entities in text.
        
        Args:
            text: Input text
            entities: List of entities found in the text
            language: Language code ("vi" or "en")
        
        Returns:
            List of Relation objects
        """
        if not text or not entities:
            return []
        
        if len(entities) < 2:
            # Need at least 2 entities for a relation
            return []
        
        relations = []
        
        # Try pattern-based extraction first
        pattern_relations = self._pattern_based_extraction(text, entities, language)
        relations.extend(pattern_relations)
        
        # Try dependency parsing for English
        if language == "en" and self.dependency_parser is not None:
            dep_relations = self._dependency_based_extraction(text, entities)
            relations.extend(dep_relations)
        
        # Remove duplicate relations
        relations = self._remove_duplicate_relations(relations)
        
        return relations
    
    def _pattern_based_extraction(
        self,
        text: str,
        entities: List[Entity],
        language: str
    ) -> List[Relation]:
        """Extract relations using pattern matching."""
        relations = []
        
        # Define relation patterns
        if language == "vi":
            patterns = self._get_vietnamese_patterns()
        else:
            patterns = self._get_english_patterns()
        
        # Check each pair of entities
        for i, subj in enumerate(entities):
            for obj in entities[i+1:]:
                # Get text between entities
                start = min(subj.end_idx, obj.end_idx)
                end = max(subj.start_idx, obj.start_idx)
                
                if start < end:
                    between_text = text[start:end].lower()
                    
                    # Check patterns
                    for pattern, relation_type in patterns:
                        if re.search(pattern, between_text, re.IGNORECASE):
                            # Determine subject and object based on entity order
                            if subj.start_idx < obj.start_idx:
                                relations.append(Relation(
                                    subject=subj,
                                    relation_type=relation_type,
                                    object=obj,
                                    confidence=0.7,
                                    context=between_text.strip()
                                ))
                            else:
                                relations.append(Relation(
                                    subject=obj,
                                    relation_type=relation_type,
                                    object=subj,
                                    confidence=0.7,
                                    context=between_text.strip()
                                ))
                            break
        
        return relations
    
    def _get_vietnamese_patterns(self) -> List[Tuple[str, str]]:
        """Get Vietnamese relation patterns."""
        return [
            (r'\blàm việc (?:tại|ở|cho)\b', 'works_for'),
            (r'\bở\b', 'located_in'),
            (r'\bthuộc\b', 'part_of'),
            (r'\blà thành viên của\b', 'member_of'),
            (r'\bsinh (?:tại|ở)\b', 'born_in'),
            (r'\bmất (?:tại|ở)\b', 'died_in'),
            (r'\bthành lập bởi\b', 'founded_by'),
            (r'\bsở hữu\b', 'owns'),
            (r'\blà công ty con của\b', 'subsidiary_of'),
            (r'\blà cha mẹ của\b', 'parent_of'),
            (r'\bkết hôn với\b', 'married_to'),
            (r'\bliên quan đến\b', 'related_to'),
        ]
    
    def _get_english_patterns(self) -> List[Tuple[str, str]]:
        """Get English relation patterns."""
        return [
            (r'\bworks? (?:for|at|in)\b', 'works_for'),
            (r'\b(?:in|at|located in)\b', 'located_in'),
            (r'\bpart of\b', 'part_of'),
            (r'\bmember of\b', 'member_of'),
            (r'\bborn in\b', 'born_in'),
            (r'\bdied in\b', 'died_in'),
            (r'\bfounded by\b', 'founded_by'),
            (r'\bowns?\b', 'owns'),
            (r'\bsubsidiary of\b', 'subsidiary_of'),
            (r'\bparent (?:company )?of\b', 'parent_of'),
            (r'\bmarried to\b', 'married_to'),
            (r'\brelated to\b', 'related_to'),
            (r'\bCEO of\b', 'works_for'),
            (r'\bpresident of\b', 'works_for'),
            (r'\bheadquartered in\b', 'located_in'),
        ]
    
    def _dependency_based_extraction(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relation]:
        """Extract relations using dependency parsing (English only)."""
        if self.dependency_parser is None:
            return []
        
        relations = []
        
        try:
            doc = self.dependency_parser(text)
            
            # Create entity span mapping
            entity_spans = {}
            for entity in entities:
                for token in doc:
                    if token.idx >= entity.start_idx and token.idx < entity.end_idx:
                        entity_spans[token.i] = entity
            
            # Look for dependency patterns
            for token in doc:
                if token.i in entity_spans:
                    subj_entity = entity_spans[token.i]
                    
                    # Check for common dependency patterns
                    # Pattern: subject -> verb -> object
                    if token.dep_ in ['nsubj', 'nsubjpass']:
                        head = token.head
                        for child in head.children:
                            if child.dep_ in ['dobj', 'pobj', 'attr'] and child.i in entity_spans:
                                obj_entity = entity_spans[child.i]
                                
                                # Infer relation type from verb
                                relation_type = self._infer_relation_from_verb(head.lemma_)
                                
                                relations.append(Relation(
                                    subject=subj_entity,
                                    relation_type=relation_type,
                                    object=obj_entity,
                                    confidence=0.6,
                                    context=head.text
                                ))
        
        except Exception as e:
            logger.warning(f"Dependency parsing failed: {e}")
        
        return relations
    
    def _infer_relation_from_verb(self, verb: str) -> str:
        """Infer relation type from verb lemma."""
        verb_mapping = {
            'work': 'works_for',
            'locate': 'located_in',
            'found': 'founded_by',
            'own': 'owns',
            'marry': 'married_to',
            'born': 'born_in',
            'die': 'died_in',
            'lead': 'works_for',
            'head': 'works_for',
            'manage': 'works_for',
        }
        
        return verb_mapping.get(verb, 'related_to')
    
    def _remove_duplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations."""
        if not relations:
            return []
        
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # Create a key based on subject, relation type, and object
            key = (
                relation.subject.text.lower(),
                relation.relation_type,
                relation.object.text.lower()
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations


# Convenience function
def extract_relations(
    text: str,
    entities: List[Entity],
    language: str = "vi"
) -> List[Relation]:
    """
    Extract relations between entities in text.
    
    Args:
        text: Input text
        entities: List of entities found in the text
        language: Language code ("vi" or "en")
    
    Returns:
        List of Relation objects
    """
    extractor = RelationExtractor()
    return extractor.extract_relations(text, entities, language)
