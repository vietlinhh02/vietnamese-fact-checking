"""Search query generator for cross-lingual fact-checking."""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Represents a search query."""
    text: str
    language: str  # "vi" or "en"
    source_claim: str
    query_type: str  # "direct", "entity_focused", "question", "decomposed"


class SearchQueryGenerator:
    """
    Generates search queries for fact-checking claims.
    
    Supports multiple query generation strategies:
    - Direct translation
    - Entity-focused queries
    - Question reformulation
    - Query decomposition for complex claims
    """
    
    def __init__(self, translation_service=None):
        """
        Initialize search query generator.
        
        Args:
            translation_service: TranslationService instance for cross-lingual queries
        """
        self.translation_service = translation_service
        
        # Vietnamese question words for reformulation
        self.vi_question_words = {
            'ai': 'who',
            'gì': 'what',
            'nào': 'which',
            'đâu': 'where',
            'khi nào': 'when',
            'tại sao': 'why',
            'như thế nào': 'how',
            'bao nhiêu': 'how many'
        }
    
    def generate_queries(self, claim: str, language: str = "vi") -> List[SearchQuery]:
        """
        Generate multiple search queries for a claim.
        
        Args:
            claim: Claim text to generate queries for
            language: Language of the claim (vi or en)
        
        Returns:
            List of SearchQuery objects
        """
        if not claim or not claim.strip():
            logger.warning("Empty claim provided for query generation")
            return []
        
        queries = []
        
        # Generate Vietnamese queries
        if language == "vi":
            # Direct query
            queries.append(SearchQuery(
                text=claim.strip(),
                language="vi",
                source_claim=claim,
                query_type="direct"
            ))
            
            # Entity-focused queries
            entity_queries = self._generate_entity_focused_queries(claim, "vi")
            queries.extend(entity_queries)
            
            # Question reformulation
            question_query = self._reformulate_as_question(claim, "vi")
            if question_query:
                queries.append(question_query)
            
            # Decomposed queries for complex claims
            decomposed = self._decompose_complex_claim(claim, "vi")
            queries.extend(decomposed)
        
        # Generate English queries via translation
        if self.translation_service:
            english_queries = self._generate_english_queries(claim, language)
            queries.extend(english_queries)
        else:
            logger.warning("Translation service not available, skipping English queries")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            query_key = (query.text.lower(), query.language)
            if query_key not in seen:
                seen.add(query_key)
                unique_queries.append(query)
        
        logger.info(f"Generated {len(unique_queries)} unique queries for claim: {claim[:50]}...")
        return unique_queries
    
    def _generate_entity_focused_queries(
        self,
        claim: str,
        language: str
    ) -> List[SearchQuery]:
        """
        Generate queries focused on key entities in the claim.
        
        Args:
            claim: Claim text
            language: Language of the claim
        
        Returns:
            List of entity-focused SearchQuery objects
        """
        queries = []
        
        # Extract potential entities (capitalized words/phrases)
        # This is a simple heuristic; in production, use NER
        entities = self._extract_simple_entities(claim)
        
        for entity in entities:
            # Create query with entity + key terms from claim
            key_terms = self._extract_key_terms(claim, exclude=entity)
            if key_terms:
                query_text = f"{entity} {' '.join(key_terms[:3])}"
                queries.append(SearchQuery(
                    text=query_text,
                    language=language,
                    source_claim=claim,
                    query_type="entity_focused"
                ))
        
        return queries
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """
        Extract potential entities using simple heuristics.
        
        This is a basic implementation. In production, use proper NER.
        
        Args:
            text: Text to extract entities from
        
        Returns:
            List of potential entity strings
        """
        entities = []
        
        # Find capitalized words (potential proper nouns)
        # For Vietnamese, this is less reliable but still useful
        words = text.split()
        
        current_entity = []
        for word in words:
            # Check if word starts with uppercase (after cleaning punctuation)
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word[0].isupper():
                current_entity.append(word)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        # Add last entity if exists
        if current_entity:
            entities.append(' '.join(current_entity))
        
        # Filter out single-letter entities and common words
        entities = [e for e in entities if len(e) > 2]
        
        return entities[:3]  # Return top 3 entities
    
    def _extract_key_terms(self, text: str, exclude: str = "") -> List[str]:
        """
        Extract key terms from text (excluding stopwords and specified terms).
        
        Args:
            text: Text to extract terms from
            exclude: Terms to exclude
        
        Returns:
            List of key terms
        """
        # Vietnamese stopwords (basic list)
        stopwords = {
            'là', 'của', 'và', 'có', 'được', 'trong', 'với', 'cho', 'từ', 'này',
            'đã', 'sẽ', 'đang', 'các', 'những', 'một', 'để', 'về', 'theo', 'như',
            'khi', 'nếu', 'vì', 'nhưng', 'mà', 'thì', 'đó', 'nó', 'họ', 'chúng'
        }
        
        # Tokenize and filter
        words = text.lower().split()
        exclude_words = set(exclude.lower().split())
        
        key_terms = []
        for word in words:
            # Clean punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Filter stopwords, excluded terms, and short words
            if (clean_word and 
                len(clean_word) > 2 and 
                clean_word not in stopwords and 
                clean_word not in exclude_words):
                key_terms.append(clean_word)
        
        return key_terms
    
    def _reformulate_as_question(
        self,
        claim: str,
        language: str
    ) -> Optional[SearchQuery]:
        """
        Reformulate claim as a question.
        
        Args:
            claim: Claim text
            language: Language of the claim
        
        Returns:
            SearchQuery object or None
        """
        if language == "vi":
            # Simple heuristic: add question words based on claim content
            claim_lower = claim.lower()
            
            # Check for different types of claims
            if any(word in claim_lower for word in ['ai', 'người', 'ông', 'bà']):
                question = f"Ai {claim.lower()}?"
            elif any(word in claim_lower for word in ['gì', 'cái gì', 'điều gì']):
                question = f"Cái gì {claim.lower()}?"
            elif any(word in claim_lower for word in ['đâu', 'nơi', 'chỗ']):
                question = f"Ở đâu {claim.lower()}?"
            elif any(word in claim_lower for word in ['khi', 'lúc', 'thời gian']):
                question = f"Khi nào {claim.lower()}?"
            else:
                # Default: add "có phải" (is it true that)
                question = f"Có phải {claim.lower()}?"
            
            return SearchQuery(
                text=question,
                language=language,
                source_claim=claim,
                query_type="question"
            )
        
        return None
    
    def _decompose_complex_claim(
        self,
        claim: str,
        language: str
    ) -> List[SearchQuery]:
        """
        Decompose complex claims into simpler sub-queries.
        
        Args:
            claim: Claim text
            language: Language of the claim
        
        Returns:
            List of decomposed SearchQuery objects
        """
        queries = []
        
        # Split on conjunctions
        if language == "vi":
            # Vietnamese conjunctions
            conjunctions = [' và ', ' hoặc ', ' nhưng ', ' mà ', ', ']
            
            for conj in conjunctions:
                if conj in claim:
                    parts = claim.split(conj)
                    for part in parts:
                        part = part.strip()
                        if len(part) > 10:  # Only keep substantial parts
                            queries.append(SearchQuery(
                                text=part,
                                language=language,
                                source_claim=claim,
                                query_type="decomposed"
                            ))
                    break  # Only split on first found conjunction
        
        return queries
    
    def _generate_english_queries(
        self,
        claim: str,
        source_language: str
    ) -> List[SearchQuery]:
        """
        Generate English queries via translation.
        
        Args:
            claim: Claim text
            source_language: Language of the claim
        
        Returns:
            List of English SearchQuery objects
        """
        queries = []
        
        if not self.translation_service:
            return queries
        
        # Translate claim to English
        if source_language == "vi":
            translated = self.translation_service.translate(
                claim,
                source_lang="vi",
                target_lang="en"
            )
            
            if translated:
                # Direct English query
                queries.append(SearchQuery(
                    text=translated,
                    language="en",
                    source_claim=claim,  # Keep original claim as source
                    query_type="direct"
                ))
                
                # Entity-focused English queries
                entity_queries = self._generate_entity_focused_queries(translated, "en")
                # Update source_claim for entity queries to reference original claim
                for eq in entity_queries:
                    eq.source_claim = claim
                queries.extend(entity_queries)
        
        return queries
    
    def generate_query_variations(self, query: SearchQuery) -> List[SearchQuery]:
        """
        Generate variations of a query for better coverage.
        
        Args:
            query: Original query
        
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add quoted version for exact match
        if '"' not in query.text:
            quoted = SearchQuery(
                text=f'"{query.text}"',
                language=query.language,
                source_claim=query.source_claim,
                query_type=f"{query.query_type}_exact"
            )
            variations.append(quoted)
        
        # Add key terms only (remove common words)
        key_terms = self._extract_key_terms(query.text)
        if len(key_terms) >= 2:
            simplified = SearchQuery(
                text=' '.join(key_terms[:5]),
                language=query.language,
                source_claim=query.source_claim,
                query_type=f"{query.query_type}_simplified"
            )
            variations.append(simplified)
        
        return variations
