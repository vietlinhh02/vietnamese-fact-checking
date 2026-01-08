"""Atomic claim decomposer for splitting compound claims into verifiable sub-claims."""

import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from .data_models import Claim
except ImportError:
    from data_models import Claim

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Result of decomposing a compound claim."""
    original_claim: Claim
    sub_claims: List[Claim]
    decomposition_method: str  # "rule_based", "llm", "none"
    

class ClaimDecomposer:
    """Decomposes compound claims into atomic sub-claims.
    
    A compound claim like:
        "Việt Nam có 63 tỉnh thành, giáp biên giới với 3 quốc gia và có diện tích 331,000 km²"
    
    Will be split into:
        1. "Việt Nam có 63 tỉnh thành"
        2. "Việt Nam giáp biên giới với 3 quốc gia"
        3. "Việt Nam có diện tích 331,000 km²"
    """
    
    # Vietnamese conjunctions and separators that indicate multiple claims
    CONJUNCTION_PATTERNS = [
        r',\s*(?:và|cùng với|đồng thời|ngoài ra)\s+',  # ", và", ", cùng với"
        r'\s+(?:và|cùng với|đồng thời)\s+',  # " và ", " cùng với "
        r';\s*',  # semicolon
    ]
    
    # Patterns that indicate a factual claim (contains numbers, dates, proper nouns)
    FACTUAL_INDICATORS = [
        r'\d+',  # numbers
        r'(?:năm|tháng|ngày)\s+\d+',  # dates
        r'(?:triệu|tỷ|nghìn|km²|km|m²|%)',  # units
        r'(?:có|là|đạt|chiếm|gồm|bao gồm)',  # factual verbs
    ]
    
    def __init__(
        self,
        llm_controller=None,
        use_llm: bool = False,
        min_sub_claim_length: int = 10
    ):
        """Initialize claim decomposer.
        
        Args:
            llm_controller: LLM controller for complex decomposition
            use_llm: Whether to use LLM for decomposition
            min_sub_claim_length: Minimum length for a valid sub-claim
        """
        self.llm_controller = llm_controller
        self.use_llm = use_llm and llm_controller is not None
        self.min_sub_claim_length = min_sub_claim_length
        
        # Compile patterns
        self.conjunction_regex = re.compile(
            '|'.join(self.CONJUNCTION_PATTERNS),
            re.IGNORECASE
        )
    
    def decompose(self, claim: Claim) -> DecompositionResult:
        """Decompose a claim into atomic sub-claims.
        
        Args:
            claim: The compound claim to decompose
            
        Returns:
            DecompositionResult with list of atomic sub-claims
        """
        text = claim.text.strip()
        
        # First try rule-based decomposition
        sub_claim_texts = self._rule_based_split(text)
        
        # If rule-based produced only 1 claim and LLM is available, try LLM
        if len(sub_claim_texts) <= 1 and self.use_llm:
            llm_sub_claims = self._llm_decompose(text)
            if len(llm_sub_claims) > 1:
                sub_claim_texts = llm_sub_claims
                method = "llm"
            else:
                method = "none"
        elif len(sub_claim_texts) > 1:
            method = "rule_based"
        else:
            method = "none"
        
        # Convert to Claim objects
        sub_claims = []
        for i, sub_text in enumerate(sub_claim_texts):
            sub_claim = Claim(
                text=sub_text,
                context=claim.context,
                confidence=claim.confidence,
                sentence_type=claim.sentence_type,
                start_idx=claim.start_idx,
                end_idx=claim.end_idx,
                language=claim.language,
                id=f"{claim.id}_sub_{i+1}" if claim.id else None,
                parent_claim_id=claim.id
            )
            sub_claims.append(sub_claim)
        
        # If no decomposition happened, return original claim
        if not sub_claims:
            sub_claims = [claim]
            method = "none"
        
        logger.info(f"Decomposed claim into {len(sub_claims)} sub-claims using {method}")
        
        return DecompositionResult(
            original_claim=claim,
            sub_claims=sub_claims,
            decomposition_method=method
        )
    
    def _rule_based_split(self, text: str) -> List[str]:
        """Split claim using rule-based patterns.
        
        Args:
            text: Claim text to split
            
        Returns:
            List of sub-claim texts
        """
        # Find the subject of the sentence (usually at the beginning)
        subject = self._extract_subject(text)
        
        # Try comma-based splitting first (most common pattern in Vietnamese)
        comma_claims = self._comma_based_split(text, subject)
        if len(comma_claims) >= 2:
            return comma_claims
        
        # Then try conjunction splitting
        parts = self.conjunction_regex.split(text)
        
        # If conjunction split worked, process those parts
        if len(parts) > 1:
            sub_claims = []
            for i, part in enumerate(parts):
                part = part.strip()
                
                # Try to split this part further by comma
                sub_parts = self._comma_based_split(part, subject)
                if len(sub_parts) >= 2:
                    sub_claims.extend(sub_parts)
                elif len(part) >= self.min_sub_claim_length and self._is_factual(part):
                    # Add subject if missing
                    if i > 0 and subject and not self._has_subject(part):
                        part = f"{subject} {part}"
                    sub_claims.append(part)
            
            if len(sub_claims) >= 2:
                return sub_claims
        
        return [text]  # No splitting possible
    
    def _comma_based_split(self, text: str, subject: str) -> List[str]:
        """Split by commas and conjunctions.
        
        Args:
            text: Claim text
            subject: Extracted subject
            
        Returns:
            List of sub-claim texts
        """
        # First split by comma followed by a factual verb
        comma_pattern = r',\s*(?=(?:có|là|đạt|chiếm|gồm|giáp|nằm|thuộc|bao gồm))'
        parts = re.split(comma_pattern, text)
        
        # Then split each part by conjunctions
        conjunction_pattern = r'\s+(?:và|cùng với|đồng thời)\s+(?=(?:có|là|đạt|chiếm|gồm|giáp|nằm|thuộc|bao gồm))'
        
        all_parts = []
        for part in parts:
            sub_parts = re.split(conjunction_pattern, part)
            all_parts.extend(sub_parts)
        
        if len(all_parts) <= 1:
            return []
        
        sub_claims = []
        for i, part in enumerate(all_parts):
            part = part.strip()
            
            if len(part) < self.min_sub_claim_length:
                continue
            
            # Check if contains factual content
            if not self._is_factual(part):
                continue
            
            # Add subject if missing
            if i > 0 and subject and not self._has_subject(part):
                part = f"{subject} {part}"
            
            sub_claims.append(part)
        
        return sub_claims
    
    def _extract_subject(self, text: str) -> str:
        """Extract the subject of the sentence.
        
        Args:
            text: Sentence text
            
        Returns:
            Subject string or empty string
        """
        # Common pattern: Subject + (có/là/đạt/...)
        match = re.match(
            r'^([^,]+?)\s+(?:có|là|đạt|chiếm|gồm|bao gồm|giáp|nằm|thuộc)',
            text,
            re.IGNORECASE
        )
        
        if match:
            return match.group(1).strip()
        
        # Fallback: first noun phrase (simple heuristic)
        words = text.split()
        if len(words) >= 2:
            # Return first 1-3 words as potential subject
            return ' '.join(words[:min(3, len(words))])
        
        return ""
    
    def _has_subject(self, text: str) -> bool:
        """Check if text already has a subject.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to have a subject
        """
        # Check if starts with a capitalized word or proper noun pattern
        if re.match(r'^[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', text):
            return True
        
        # Check if starts with a noun/pronoun
        subject_starters = ['việt nam', 'hà nội', 'thành phố', 'tỉnh', 'quốc gia', 'nước', 'đất nước']
        text_lower = text.lower()
        for starter in subject_starters:
            if text_lower.startswith(starter):
                return True
        
        return False
    
    def _is_factual(self, text: str) -> bool:
        """Check if text contains factual content.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains factual indicators
        """
        for pattern in self.FACTUAL_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _llm_decompose(self, text: str) -> List[str]:
        """Use LLM to decompose complex claims.
        
        Args:
            text: Claim text
            
        Returns:
            List of sub-claim texts
        """
        if not self.llm_controller:
            return [text]
        
        try:
            prompt = f"""Tách câu sau thành các tuyên bố sự thật riêng biệt (atomic claims).
Mỗi tuyên bố phải có thể kiểm chứng độc lập.
Giữ nguyên chủ ngữ cho mỗi tuyên bố.

Câu gốc: "{text}"

Liệt kê các tuyên bố (mỗi dòng một tuyên bố, không đánh số):"""

            messages = [
                {"role": "system", "content": "Bạn là chuyên gia phân tích ngôn ngữ. Tách câu thành các tuyên bố sự thật riêng biệt."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_controller.generate(
                messages=messages,
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse response - each line is a sub-claim
            lines = response.content.strip().split('\n')
            sub_claims = []
            for line in lines:
                line = line.strip()
                # Remove numbering if present
                line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
                if line and len(line) >= self.min_sub_claim_length:
                    sub_claims.append(line)
            
            return sub_claims if sub_claims else [text]
            
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            return [text]


def decompose_claims(
    claims: List[Claim],
    llm_controller=None,
    use_llm: bool = False
) -> Tuple[List[Claim], dict]:
    """Decompose a list of claims into atomic sub-claims.
    
    Args:
        claims: List of claims to decompose
        llm_controller: Optional LLM controller
        use_llm: Whether to use LLM for complex cases
        
    Returns:
        Tuple of (flattened list of atomic claims, mapping of original -> sub claims)
    """
    decomposer = ClaimDecomposer(
        llm_controller=llm_controller,
        use_llm=use_llm
    )
    
    all_sub_claims = []
    claim_mapping = {}  # original_claim_id -> [sub_claim_ids]
    
    for claim in claims:
        result = decomposer.decompose(claim)
        
        # Track mapping
        original_id = claim.id or str(id(claim))
        claim_mapping[original_id] = {
            "original_text": claim.text,
            "sub_claims": [sc.text for sc in result.sub_claims],
            "method": result.decomposition_method
        }
        
        all_sub_claims.extend(result.sub_claims)
    
    logger.info(f"Decomposed {len(claims)} claims into {len(all_sub_claims)} atomic sub-claims")
    
    return all_sub_claims, claim_mapping


if __name__ == "__main__":
    # Test the decomposer
    logging.basicConfig(level=logging.INFO)
    
    test_claims = [
        "Việt Nam có 63 tỉnh thành, giáp biên giới với 3 quốc gia và có diện tích khoảng 331,000 km²",
        "Hà Nội là thủ đô của Việt Nam, có dân số khoảng 8 triệu người",
        "GDP Việt Nam năm 2023 đạt 430 tỷ USD",  # Single claim, should not split
    ]
    
    decomposer = ClaimDecomposer()
    
    for text in test_claims:
        claim = Claim(text=text, language="vi")
        result = decomposer.decompose(claim)
        
        print(f"\n{'='*60}")
        print(f"Original: {text}")
        print(f"Method: {result.decomposition_method}")
        print(f"Sub-claims ({len(result.sub_claims)}):")
        for i, sc in enumerate(result.sub_claims, 1):
            print(f"  {i}. {sc.text}")
