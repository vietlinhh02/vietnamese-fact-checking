"""Self-verification module for Vietnamese fact-checking system."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from collections import defaultdict

try:
    from .data_models import Claim, Evidence, Verdict
    from .claim_detector import ClaimDetector, detect_claims_in_text
    from .search_query_generator import SearchQueryGenerator
    from .web_crawler import WebCrawler
    from .exa_search_client import ExaSearchClient
except ImportError:
    from data_models import Claim, Evidence, Verdict
    from claim_detector import ClaimDetector, detect_claims_in_text
    from search_query_generator import SearchQueryGenerator
    from web_crawler import WebCrawler
    from exa_search_client import ExaSearchClient

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: Claim
    is_verified: bool
    supporting_evidence: List[Evidence]
    confidence: float
    verification_method: str  # "evidence_match", "search_verification", "rule_based"
    explanation: str


@dataclass
class QualityScore:
    """Quality score for an explanation."""
    overall_score: float  # [0, 1]
    verified_claims: int
    total_claims: int
    verification_rate: float
    confidence_scores: Dict[str, float]
    flagged_claims: List[Claim]
    explanation: str


class ExplanationClaimExtractor:
    """Extracts factual claims from generated explanations."""
    
    def __init__(self, claim_detector: Optional[ClaimDetector] = None):
        """Initialize claim extractor.
        
        Args:
            claim_detector: Claim detector instance (optional)
        """
        self.claim_detector = claim_detector
        if self.claim_detector is None:
            # Use rule-based extraction as fallback
            logger.info("No claim detector provided, using rule-based extraction")
        
        # Patterns for factual statements in Vietnamese
        self.factual_patterns = [
            r'(?:theo|dựa trên|căn cứ vào).+?(?:\.|,)',  # According to...
            r'(?:số liệu|dữ liệu|thống kê).+?(?:\.|,)',  # Statistics/data...
            r'(?:nghiên cứu|báo cáo|khảo sát).+?(?:\.|,)',  # Research/report...
            r'(?:vào năm|trong năm|từ năm).+?(?:\.|,)',  # In year...
            r'(?:có|là|đạt|chiếm).+?(?:triệu|tỷ|phần trăm|%).+?(?:\.|,)',  # Numbers/percentages
            r'(?:tăng|giảm|tăng trưởng|suy giảm).+?(?:\.|,)',  # Growth/decline
        ]
        
        # Patterns to exclude (opinions, questions, citations, etc.)
        self.exclude_patterns = [
            r'(?:tôi nghĩ|theo tôi|có thể|có lẽ|dường như)',  # Opinions
            r'(?:\?|có phải|liệu)',  # Questions
            r'(?:nên|cần|phải|hãy)',  # Commands/suggestions
            r'^\s*\[?\d+\]?\s*(?:Sources?|Nguồn):?',  # Source citations
            r'^\s*\[?\d+\]?\s*\(',  # Citation references like "[1] (Source: ...)"
            r'^\s*Sources?:',  # Source headers
            r'^\s*Nguồn:',  # Vietnamese source headers
            r'^\s*\[?\d+\]?\s*[A-Z][^.]*\s*-\s*https?://',  # Source links
            r'^\s*---.*---\s*$',  # Section dividers
            r'^\s*\*\*[^*]+\*\*\s*$',  # Standalone markdown headers like "**VERDICT:**"
            r'^\s*#+\s+',  # Markdown headers like "## Title"
            r'^\s*\*\s*\*\*[^*]+\*\*:',  # Bullet points with bold headers
            r'(?:quá trình|reasoning|process|xác định|nhiệm vụ|task)',  # Process descriptions
            r'(?:kết luận|verdict|confidence|điểm tự tin)',  # Meta conclusions
            r'(?:bằng chứng mâu thuẫn|contradictory evidence)',  # Meta analysis
            # Additional meta-content filtering
            r'(?:Dựa trên|Căn cứ vào|Theo) (?:các|những)? (?:bằng chứng|nguồn tin|thông tin)',
            r'(?:Tuyên bố|Nhận định|Kết luận) (?:này|cần) (?:được|là|bao gồm)',
            r'(?:được|bị) (?:hỗ trợ|bác bỏ|xác nhận|đánh giá)',
            r'(?:Kiểm tra|Xác minh) (?:tính|sự) (?:xác thực|thật)',
            r'(?:Giải thích|Chi tiết|Lý giải|Do đó|Vì vậy)',
        ]
    
    def extract_claims(self, explanation: str) -> List[Claim]:
        """Extract factual claims from explanation text.
        
        Args:
            explanation: Generated explanation text
            
        Returns:
            List of extracted claims
        """
        if self.claim_detector:
            return self._extract_with_model(explanation)
        else:
            return self._extract_with_rules(explanation)
    
    def _extract_with_model(self, explanation: str) -> List[Claim]:
        """Extract claims using trained claim detection model.
        
        Args:
            explanation: Explanation text
            
        Returns:
            List of detected claims
        """
        try:
            claims = self.claim_detector.detect_claims(explanation)
            logger.info(f"Extracted {len(claims)} claims using model")
            return claims
        except Exception as e:
            logger.error(f"Model-based extraction failed: {e}")
            # Fallback to rule-based
            return self._extract_with_rules(explanation)
    
    def _extract_with_rules(self, explanation: str) -> List[Claim]:
        """Extract claims using rule-based patterns.
        
        Args:
            explanation: Explanation text
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Split into sentences
        sentences = self._split_sentences(explanation)
        
        for i, sentence in enumerate(sentences):
            # Skip if sentence matches exclude patterns
            if any(re.search(pattern, sentence, re.IGNORECASE) 
                   for pattern in self.exclude_patterns):
                continue
            
            # Check if sentence matches factual patterns
            is_factual = any(re.search(pattern, sentence, re.IGNORECASE) 
                           for pattern in self.factual_patterns)
            
            # Only consider sentences with meaningful factual content
            has_meaningful_numbers = bool(re.search(r'(?:có|đạt|là|chiếm|tăng|giảm).*?\d+(?:[.,]\d+)*(?:\s*(?:triệu|tỷ|%|phần trăm))', sentence))
            has_meaningful_dates = bool(re.search(r'(?:năm|vào|từ|trong)\s+\d{4}', sentence))
            has_factual_entities = bool(re.search(r'(?:Việt Nam|GDP|dân số|tỉnh thành)', sentence, re.IGNORECASE))
            
            # Skip meta-analysis and process descriptions
            is_meta_content = bool(re.search(r'(?:quá trình|reasoning|nhiệm vụ|kết luận|verdict|bằng chứng chỉ ra|evidence shows)', sentence, re.IGNORECASE))
            
            # Require at least one strong factual indicator and not meta content
            if not is_meta_content and (is_factual or (has_meaningful_numbers and len(sentence) > 20) or (has_meaningful_dates and len(sentence) > 15) or (has_factual_entities and len(sentence) > 10)):
                # Calculate confidence based on patterns matched
                confidence = 0.6  # Start higher for filtered claims
                if is_factual:
                    confidence += 0.3
                if has_meaningful_numbers:
                    confidence += 0.2
                if has_meaningful_dates:
                    confidence += 0.1
                if has_factual_entities:
                    confidence += 0.1
                
                confidence = min(confidence, 1.0)
                
                # Get context (surrounding sentences)
                context_start = max(0, i - 1)
                context_end = min(len(sentences), i + 2)
                context = ' '.join(sentences[context_start:context_end])
                
                claim = Claim(
                    text=sentence.strip(),
                    context=context,
                    confidence=confidence,
                    sentence_type="factual_claim",
                    start_idx=0,  # Would need full text processing for exact positions
                    end_idx=len(sentence),
                    language="vi"
                )
                
                claims.append(claim)
        
        logger.info(f"Extracted {len(claims)} claims using rules")
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting for Vietnamese
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class ClaimVerifier:
    """Verifies extracted claims against collected evidence."""
    
    def __init__(
        self,
        search_client: Optional[ExaSearchClient] = None,
        web_crawler: Optional[WebCrawler] = None,
        query_generator: Optional[SearchQueryGenerator] = None,
        verification_threshold: float = 0.4,
        max_verification_attempts: int = 3
    ):
        """Initialize claim verifier.
        
        Args:
            search_client: Search client for verification
            web_crawler: Web crawler for evidence collection
            query_generator: Query generator for search
            verification_threshold: Minimum similarity for verification
            max_verification_attempts: Maximum attempts per claim
        """
        self.search_client = search_client
        self.web_crawler = web_crawler
        self.query_generator = query_generator
        self.verification_threshold = verification_threshold
        self.max_verification_attempts = max_verification_attempts
        
        logger.info("Initialized claim verifier")
    
    def run_verification_loop(
        self,
        claims: List[Claim],
        collected_evidence: List[Evidence]
    ) -> List[VerificationResult]:
        """Run verification loop for multiple claims.
        
        Args:
            claims: List of claims to verify
            collected_evidence: Previously collected evidence
            
        Returns:
            List of verification results
        """
        results = []
        
        logger.info(f"Starting verification loop for {len(claims)} claims")
        
        for i, claim in enumerate(claims, 1):
            logger.info(f"Verifying claim {i}/{len(claims)}: {claim.text[:50]}...")
            
            # Verify claim with multiple attempts if needed
            result = self._verify_claim_with_retries(claim, collected_evidence)
            results.append(result)
            
            # Log result
            status = "✓ VERIFIED" if result.is_verified else "✗ UNVERIFIED"
            logger.info(f"  {status} (confidence: {result.confidence:.2f}, method: {result.verification_method})")
            
            # Flag potential hallucinations
            if not result.is_verified and result.claim.confidence > 0.7:
                logger.warning(f"  ⚠ POTENTIAL HALLUCINATION: High-confidence claim not verified")
        
        verified_count = sum(1 for r in results if r.is_verified)
        logger.info(f"Verification loop complete: {verified_count}/{len(claims)} claims verified")
        
        return results
    
    def _verify_claim_with_retries(
        self,
        claim: Claim,
        collected_evidence: List[Evidence]
    ) -> VerificationResult:
        """Verify claim with multiple attempts and strategies.
        
        Args:
            claim: Claim to verify
            collected_evidence: Previously collected evidence
            
        Returns:
            Best verification result from all attempts
        """
        attempts = []
        
        # Attempt 1: Check against collected evidence
        evidence_result = self._verify_against_evidence(claim, collected_evidence)
        attempts.append(evidence_result)
        
        if evidence_result.is_verified:
            return evidence_result
        
        # Attempt 2: Quick search verification (if available)
        if self.search_client and self.query_generator:
            search_result = self._verify_with_search(claim, max_results=2)
            attempts.append(search_result)
            
            if search_result.is_verified:
                return search_result
        
        # Attempt 3: Relaxed threshold verification
        relaxed_result = self._verify_against_evidence(
            claim, collected_evidence, threshold=0.4
        )
        attempts.append(relaxed_result)
        
        if relaxed_result.is_verified:
            # Mark as low confidence verification
            relaxed_result.confidence *= 0.7
            relaxed_result.verification_method = "relaxed_threshold"
            return relaxed_result
        
        # Return best attempt (highest confidence)
        best_attempt = max(attempts, key=lambda x: x.confidence)
        return best_attempt

    def verify_claim(
        self,
        claim: Claim,
        collected_evidence: List[Evidence],
        max_search_results: int = 3
    ) -> VerificationResult:
        """Verify a single claim against evidence and additional searches.
        
        Args:
            claim: Claim to verify
            collected_evidence: Previously collected evidence
            max_search_results: Maximum number of search results to check
            
        Returns:
            Verification result
        """
        # First, check against collected evidence
        evidence_result = self._verify_against_evidence(claim, collected_evidence)
        
        if evidence_result.is_verified:
            return evidence_result
        
        # If not verified, perform additional search
        if self.search_client and self.query_generator:
            search_result = self._verify_with_search(claim, max_search_results)
            if search_result.is_verified:
                return search_result
        
        # If still not verified, return negative result
        return VerificationResult(
            claim=claim,
            is_verified=False,
            supporting_evidence=[],
            confidence=0.0,
            verification_method="no_verification",
            explanation=f"Could not find supporting evidence for: {claim.text}"
        )
    
    def _verify_against_evidence(
        self,
        claim: Claim,
        evidence_list: List[Evidence],
        threshold: Optional[float] = None
    ) -> VerificationResult:
        """Verify claim against collected evidence.
        
        Args:
            claim: Claim to verify
            evidence_list: List of evidence to check against
            threshold: Custom similarity threshold (optional)
            
        Returns:
            Verification result
        """
        supporting_evidence = []
        max_similarity = 0.0
        verification_threshold = threshold or self.verification_threshold
        
        for evidence in evidence_list:
            similarity = self._compute_similarity(claim.text, evidence.text)
            
            if similarity > verification_threshold:  # Threshold for considering evidence as supporting
                supporting_evidence.append(evidence)
                max_similarity = max(max_similarity, similarity)
        
        is_verified = len(supporting_evidence) > 0
        confidence = max_similarity if is_verified else 0.0
        
        explanation = (
            f"Found {len(supporting_evidence)} supporting evidence pieces"
            if is_verified else "No supporting evidence found in collected data"
        )
        
        return VerificationResult(
            claim=claim,
            is_verified=is_verified,
            supporting_evidence=supporting_evidence,
            confidence=confidence,
            verification_method="evidence_match",
            explanation=explanation
        )
    
    def _verify_with_search(
        self,
        claim: Claim,
        max_results: int
    ) -> VerificationResult:
        """Verify claim with additional web search.
        
        Args:
            claim: Claim to verify
            max_results: Maximum search results to check
            
        Returns:
            Verification result
        """
        try:
            # Generate search query
            queries = self.query_generator.generate_queries(claim.text)
            
            if not queries:
                return VerificationResult(
                    claim=claim,
                    is_verified=False,
                    supporting_evidence=[],
                    confidence=0.0,
                    verification_method="search_failed",
                    explanation="Could not generate search queries"
                )
            
            # Search for evidence
            all_results = []
            for query in queries[:2]:  # Limit to 2 queries
                results = self.search_client.search(query.text, num_results=max_results)
                all_results.extend(results)
            
            if not all_results:
                return VerificationResult(
                    claim=claim,
                    is_verified=False,
                    supporting_evidence=[],
                    confidence=0.0,
                    verification_method="no_search_results",
                    explanation="No search results found"
                )
            
            # Crawl and check content
            supporting_evidence = []
            
            for result in all_results[:max_results]:
                if self.web_crawler:
                    content = self.web_crawler.crawl(result.url)
                    if content and content.main_text:
                        similarity = self._compute_similarity(claim.text, content.main_text)
                        
                        if similarity > 0.4:  # Lower threshold for search results
                            evidence = Evidence(
                                text=content.main_text[:500],  # Truncate for storage
                                source_url=result.url,
                                source_title=content.title or result.title,
                                source_author=content.author,
                                publish_date=content.publish_date,
                                credibility_score=0.7,  # Default for search results
                                language="vi"
                            )
                            supporting_evidence.append(evidence)
            
            is_verified = len(supporting_evidence) > 0
            confidence = 0.7 if is_verified else 0.0
            
            explanation = (
                f"Found {len(supporting_evidence)} supporting sources through search"
                if is_verified else "No supporting evidence found through search"
            )
            
            return VerificationResult(
                claim=claim,
                is_verified=is_verified,
                supporting_evidence=supporting_evidence,
                confidence=confidence,
                verification_method="search_verification",
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Search verification failed: {e}")
            return VerificationResult(
                claim=claim,
                is_verified=False,
                supporting_evidence=[],
                confidence=0.0,
                verification_method="search_error",
                explanation=f"Search verification error: {str(e)}"
            )
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between claim (text1) and evidence (text2).
        
        Uses a simple containment score: what fraction of claim keywords/numbers 
        appear in the evidence text? This handles long evidence texts better than Jaccard.
        """
        # Normalize
        t1 = text1.lower()
        t2 = text2.lower()
        
        # 1. Keyword containment
        # Extract keywords (simple split, filter small words)
        words1 = [w for w in re.findall(r'\w+', t1) if len(w) > 2]
        words2 = set(re.findall(r'\w+', t2)) # Set for fast lookup
        
        if not words1:
            return 0.0
            
        # Count how many claim words appear in evidence
        match_count = sum(1 for w in words1 if w in words2)
        keyword_score = match_count / len(words1)
        
        # 2. Number containment (CRITICAL for facts)
        # Extract numbers (simple digit sequences)
        nums1 = re.findall(r'\d+', t1)
        
        if nums1:
            nums2 = set(re.findall(r'\d+', t2))
            num_matches = sum(1 for n in nums1 if n in nums2)
            num_score = num_matches / len(nums1)
            
            # If claim has numbers, they are very important for fact verification.
            # Weight: 50% keywords, 50% numbers
            return 0.5 * keyword_score + 0.5 * num_score
            
        return keyword_score
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out common stop words
        stop_words = {
            'và', 'của', 'trong', 'với', 'là', 'có', 'được', 'này', 'đó', 'các', 'một',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'
        }
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords


class QualityScorer:
    """Computes quality scores for explanations based on verification results."""
    
    def __init__(
        self, 
        min_verification_rate: float = 0.7,
        confidence_weight: float = 0.3,
        method_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize quality scorer.
        
        Args:
            min_verification_rate: Minimum verification rate for good quality
            confidence_weight: Weight for confidence scores in overall score
            method_weights: Weights for different verification methods
        """
        self.min_verification_rate = min_verification_rate
        self.confidence_weight = confidence_weight
        
        # Default weights for verification methods
        self.method_weights = method_weights or {
            "evidence_match": 1.0,
            "search_verification": 0.8,
            "relaxed_threshold": 0.6,
            "no_verification": 0.0
        }
    
    def compute_quality_score(
        self,
        verification_results: List[VerificationResult]
    ) -> QualityScore:
        """Compute quality score based on verification results.
        
        Args:
            verification_results: List of verification results
            
        Returns:
            Quality score with details
        """
        if not verification_results:
            return QualityScore(
                overall_score=0.0,
                verified_claims=0,
                total_claims=0,
                verification_rate=0.0,
                confidence_scores={},
                flagged_claims=[],
                explanation="No claims were identified for verification."
            )
        
        # Count verified claims
        verified_claims = sum(1 for result in verification_results if result.is_verified)
        total_claims = len(verification_results)
        verification_rate = verified_claims / total_claims
        
        # Compute confidence scores by method
        confidence_scores = defaultdict(list)
        for result in verification_results:
            confidence_scores[result.verification_method].append(result.confidence)
        
        # Average confidence by method
        avg_confidence_scores = {
            method: sum(scores) / len(scores) if scores else 0.0
            for method, scores in confidence_scores.items()
        }
        
        # Identify flagged claims (unverified)
        flagged_claims = [
            result.claim for result in verification_results 
            if not result.is_verified
        ]
        
        # Overall score follows the spec: verified_claims / total_claims
        overall_score = verification_rate
        
        # Generate explanation
        explanation = self._generate_quality_explanation(
            verification_rate, verified_claims, total_claims, flagged_claims
        )
        
        # Add warning if quality is low (for confidence capping)
        if verification_rate < 0.7:
             explanation = "⚠️ LOW VERIFICATION RATE: " + explanation
        
        return QualityScore(
            overall_score=overall_score,
            verified_claims=verified_claims,
            total_claims=total_claims,
            verification_rate=verification_rate,
            confidence_scores=avg_confidence_scores,
            flagged_claims=flagged_claims,
            explanation=explanation
        )
    
    def _compute_weighted_score(self, verification_results: List[VerificationResult]) -> float:
        """Compute weighted quality score.
        
        Args:
            verification_results: List of verification results
            
        Returns:
            Weighted quality score [0, 1]
        """
        if not verification_results:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for result in verification_results:
            # Get method weight
            method_weight = self.method_weights.get(result.verification_method, 0.5)
            
            # Compute claim score
            if result.is_verified:
                # Verified claims get full method weight * confidence
                claim_score = method_weight * result.confidence
            else:
                # Unverified claims get penalty
                claim_score = 0.0
            
            weighted_sum += claim_score
            total_weight += method_weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize by total possible weight
        base_score = weighted_sum / total_weight
        
        # Apply confidence adjustment
        avg_confidence = sum(r.confidence for r in verification_results) / len(verification_results)
        confidence_adjustment = avg_confidence * self.confidence_weight
        
        # Final score
        final_score = base_score * (1 - self.confidence_weight) + confidence_adjustment
        
        return max(0.0, min(1.0, final_score))
    
    def _generate_quality_explanation(
        self,
        verification_rate: float,
        verified_claims: int,
        total_claims: int,
        flagged_claims: List[Claim]
    ) -> str:
        """Generate explanation for quality score.
        
        Args:
            verification_rate: Rate of verified claims
            verified_claims: Number of verified claims
            total_claims: Total number of claims
            flagged_claims: List of unverified claims
            
        Returns:
            Quality explanation text
        """
        explanation = f"Verification Summary: {verified_claims}/{total_claims} claims verified "
        explanation += f"({verification_rate:.1%} verification rate).\n\n"
        
        if verification_rate >= self.min_verification_rate:
            explanation += "✓ High quality: Most claims are supported by evidence.\n"
        elif verification_rate >= 0.5:
            explanation += "⚠ Medium quality: Some claims lack supporting evidence.\n"
        else:
            explanation += "✗ Low quality: Many claims are unsupported.\n"
        
        if flagged_claims:
            explanation += f"\nFlagged claims ({len(flagged_claims)}):\n"
            for i, claim in enumerate(flagged_claims[:3], 1):  # Show first 3
                explanation += f"{i}. {claim.text[:100]}...\n"
            
            if len(flagged_claims) > 3:
                explanation += f"... and {len(flagged_claims) - 3} more.\n"
        
        return explanation


class SelfVerificationModule:
    """Main self-verification module that orchestrates the verification process."""
    
    def __init__(
        self,
        claim_extractor: Optional[ExplanationClaimExtractor] = None,
        claim_verifier: Optional[ClaimVerifier] = None,
        quality_scorer: Optional[QualityScorer] = None
    ):
        """Initialize self-verification module.
        
        Args:
            claim_extractor: Claim extractor for explanations
            claim_verifier: Claim verifier
            quality_scorer: Quality scorer
        """
        self.claim_extractor = claim_extractor or ExplanationClaimExtractor()
        self.claim_verifier = claim_verifier or ClaimVerifier()
        self.quality_scorer = quality_scorer or QualityScorer()
        
        logger.info("Initialized self-verification module")
    
    def verify_explanation(
        self,
        explanation: str,
        collected_evidence: List[Evidence]
    ) -> Tuple[QualityScore, List[VerificationResult]]:
        """Verify an explanation for hallucinations and quality.
        
        Args:
            explanation: Generated explanation text
            collected_evidence: Evidence used to generate the explanation
            
        Returns:
            Tuple of (quality_score, verification_results)
        """
        logger.info("Starting explanation verification")
        
        # Step 1: Extract claims from explanation
        claims = self.claim_extractor.extract_claims(explanation)
        logger.info(f"Extracted {len(claims)} claims from explanation")
        
        if not claims:
            # No claims to verify
            quality_score = QualityScore(
                overall_score=1.0,  # No claims means no hallucinations
                verified_claims=0,
                total_claims=0,
                verification_rate=1.0,
                confidence_scores={},
                flagged_claims=[],
                explanation="No factual claims found in explanation"
            )
            return quality_score, []
        
        # Step 2: Run verification loop for all claims
        verification_results = self.claim_verifier.run_verification_loop(claims, collected_evidence)
        
        # Step 3: Compute quality score
        quality_score = self.quality_scorer.compute_quality_score(verification_results)
        
        logger.info(f"Verification complete: {quality_score.verification_rate:.1%} verified, "
                   f"quality score: {quality_score.overall_score:.2f}")
        
        return quality_score, verification_results
    
    def correct_hallucinations(
        self,
        explanation: str,
        verification_results: List[VerificationResult],
        quality_score: QualityScore,
        correction_strategy: str = "adaptive"
    ) -> str:
        """Correct hallucinations in the explanation.
        
        Args:
            explanation: Original explanation
            verification_results: Results from verification
            quality_score: Quality score for the explanation
            correction_strategy: Strategy for correction ("remove", "flag", "revise", "adaptive")
            
        Returns:
            Corrected explanation
        """
        # Adaptive strategy based on quality score
        if correction_strategy == "adaptive":
            if quality_score.overall_score >= 0.8:
                correction_strategy = "flag"  # High quality, just flag uncertain claims
            elif quality_score.overall_score >= 0.5:
                correction_strategy = "revise"  # Medium quality, revise claims
            else:
                correction_strategy = "remove"  # Low quality, remove unverified claims
        
        logger.info(f"Applying correction strategy: {correction_strategy}")
        
        if correction_strategy == "remove":
            return self._remove_unverified_claims(explanation, verification_results)
        elif correction_strategy == "flag":
            return self._flag_unverified_claims(explanation, verification_results)
        elif correction_strategy == "revise":
            return self._revise_unverified_claims(explanation, verification_results)
        else:
            logger.warning(f"Unknown correction strategy: {correction_strategy}")
            return explanation
    
    def _remove_unverified_claims(
        self,
        explanation: str,
        verification_results: List[VerificationResult]
    ) -> str:
        """Remove unverified claims from explanation.
        
        Args:
            explanation: Original explanation
            verification_results: Verification results
            
        Returns:
            Explanation with unverified claims removed
        """
        # Get unverified claim texts
        unverified_texts = [
            result.claim.text for result in verification_results
            if not result.is_verified
        ]
        
        corrected = explanation
        
        # Remove unverified sentences
        for claim_text in unverified_texts:
            # Simple removal - could be improved with better text processing
            corrected = corrected.replace(claim_text, "")
        
        # Clean up extra whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        return corrected
    
    def _flag_unverified_claims(
        self,
        explanation: str,
        verification_results: List[VerificationResult]
    ) -> str:
        """Flag unverified claims in explanation.
        
        Args:
            explanation: Original explanation
            verification_results: Verification results
            
        Returns:
            Explanation with unverified claims flagged
        """
        corrected = explanation
        
        # Add flags to unverified claims
        for result in verification_results:
            if not result.is_verified:
                claim_text = result.claim.text
                flagged_text = f"{claim_text} [CHƯA XÁC MINH]"
                corrected = corrected.replace(claim_text, flagged_text)
        
        return corrected
    
    def _revise_unverified_claims(
        self,
        explanation: str,
        verification_results: List[VerificationResult]
    ) -> str:
        """Revise unverified claims with caveats.
        
        Args:
            explanation: Original explanation
            verification_results: Verification results
            
        Returns:
            Explanation with unverified claims revised
        """
        corrected = explanation
        
        # Add caveats to unverified claims
        for result in verification_results:
            if not result.is_verified:
                claim_text = result.claim.text
                revised_text = f"Theo một số nguồn, {claim_text.lower()}"
                corrected = corrected.replace(claim_text, revised_text)
        
        return corrected
    
    def should_regenerate_explanation(
        self,
        quality_score: QualityScore,
        regeneration_threshold: float = 0.3
    ) -> bool:
        """Determine if explanation should be regenerated.
        
        Args:
            quality_score: Quality score for the explanation
            regeneration_threshold: Threshold below which to regenerate
            
        Returns:
            True if explanation should be regenerated
        """
        return quality_score.overall_score < regeneration_threshold
    
    def get_correction_summary(
        self,
        verification_results: List[VerificationResult],
        quality_score: QualityScore
    ) -> str:
        """Generate summary of corrections made.
        
        Args:
            verification_results: Verification results
            quality_score: Quality score
            
        Returns:
            Summary text
        """
        summary = f"Self-Verification Summary:\n"
        summary += f"- Quality Score: {quality_score.overall_score:.2f}/1.00\n"
        summary += f"- Verification Rate: {quality_score.verification_rate:.1%}\n"
        summary += f"- Claims Verified: {quality_score.verified_claims}/{quality_score.total_claims}\n"
        
        if quality_score.flagged_claims:
            summary += f"- Flagged Claims: {len(quality_score.flagged_claims)}\n"
        
        # Method breakdown
        method_counts = defaultdict(int)
        for result in verification_results:
            method_counts[result.verification_method] += 1
        
        summary += "\nVerification Methods Used:\n"
        for method, count in method_counts.items():
            summary += f"- {method}: {count} claims\n"
        
        return summary


class SelfVerificationOutputFormatter:
    """Formats self-verification results for different output types."""
    
    @staticmethod
    def to_structured_output(
        quality_score: QualityScore,
        verification_results: List[VerificationResult],
        correction_applied: bool = False,
        correction_strategy: str = "none",
        original_length: int = 0,
        corrected_length: int = 0
    ) -> Dict[str, Any]:
        """Convert verification results to structured output format for Gemini.
        
        Args:
            quality_score: Quality score object
            verification_results: List of verification results
            correction_applied: Whether correction was applied
            correction_strategy: Strategy used for correction
            original_length: Original explanation length
            corrected_length: Corrected explanation length
            
        Returns:
            Structured output dictionary matching Gemini schema
        """
        # Determine quality level
        if quality_score.overall_score >= 0.8:
            quality_level = "HIGH"
        elif quality_score.overall_score >= 0.5:
            quality_level = "MEDIUM"
        else:
            quality_level = "LOW"
        
        # Generate recommendations
        recommendations = []
        if quality_score.overall_score < 0.8:
            if quality_score.flagged_claims:
                recommendations.append("Review and verify flagged claims with additional sources")
            if quality_score.verification_rate < 0.7:
                recommendations.append("Improve evidence collection for better claim verification")
            if quality_score.overall_score < 0.3:
                recommendations.append("Consider regenerating explanation due to low quality")
        
        return {
            "quality_assessment": {
                "overall_score": quality_score.overall_score,
                "verification_rate": quality_score.verification_rate,
                "verified_claims": quality_score.verified_claims,
                "total_claims": quality_score.total_claims,
                "flagged_claims": len(quality_score.flagged_claims),
                "quality_level": quality_level,
                "confidence_scores": quality_score.confidence_scores,
                "explanation": quality_score.explanation
            },
            "verification_results": [
                {
                    "claim_text": result.claim.text,
                    "is_verified": result.is_verified,
                    "confidence": result.confidence,
                    "verification_method": result.verification_method,
                    "explanation": result.explanation,
                    "supporting_evidence_count": len(result.supporting_evidence)
                }
                for result in verification_results
            ],
            "correction_applied": correction_applied,
            "correction_strategy": correction_strategy,
            "original_length": original_length,
            "corrected_length": corrected_length,
            "length_change": corrected_length - original_length,
            "recommendations": recommendations
        }
    
    @staticmethod
    def format_quality_summary(quality_score: QualityScore) -> str:
        """Format quality score summary.
        
        Args:
            quality_score: Quality score object
            
        Returns:
            Formatted summary string
        """
        summary = f"--- VERIFICATION SUMMARY ---\n"
        summary += f"Quality Score: {quality_score.overall_score:.2f}/1.00\n"
        summary += f"Verification Rate: {quality_score.verification_rate:.1%}\n"
        summary += f"Claims Verified: {quality_score.verified_claims}/{quality_score.total_claims}\n"
        
        if quality_score.flagged_claims:
            summary += f"Flagged Claims: {len(quality_score.flagged_claims)}\n"
        
        # Quality assessment
        if quality_score.overall_score >= 0.8:
            summary += "Status: ✓ HIGH QUALITY - Most claims verified\n"
        elif quality_score.overall_score >= 0.5:
            summary += "Status: ⚠ MEDIUM QUALITY - Some claims need verification\n"
        else:
            summary += "Status: ✗ LOW QUALITY - Many claims unverified\n"
        
        return summary
    
    @staticmethod
    def format_detailed_results(
        verification_results: List[VerificationResult],
        quality_score: QualityScore
    ) -> str:
        """Format detailed verification results.
        
        Args:
            verification_results: List of verification results
            quality_score: Quality score object
            
        Returns:
            Formatted detailed results string
        """
        output = f"--- DETAILED VERIFICATION RESULTS ---\n"
        output += f"Overall Quality: {quality_score.overall_score:.2f}/1.00\n"
        output += f"Verification Rate: {quality_score.verification_rate:.1%}\n\n"
        
        # Group by verification status
        verified_claims = [r for r in verification_results if r.is_verified]
        unverified_claims = [r for r in verification_results if not r.is_verified]
        
        if verified_claims:
            output += f"✓ VERIFIED CLAIMS ({len(verified_claims)}):\n"
            output += "-" * 50 + "\n"
            for i, result in enumerate(verified_claims, 1):
                output += f"{i}. {result.claim.text[:80]}...\n"
                output += f"   Confidence: {result.confidence:.2f}\n"
                output += f"   Method: {result.verification_method}\n"
                output += f"   Evidence: {len(result.supporting_evidence)} pieces\n\n"
        
        if unverified_claims:
            output += f"✗ UNVERIFIED CLAIMS ({len(unverified_claims)}):\n"
            output += "-" * 50 + "\n"
            for i, result in enumerate(unverified_claims, 1):
                output += f"{i}. {result.claim.text[:80]}...\n"
                output += f"   Confidence: {result.confidence:.2f}\n"
                output += f"   Method: {result.verification_method}\n"
                output += f"   Issue: {result.explanation}\n\n"
        
        return output
    
    @staticmethod
    def format_json_output(
        quality_score: QualityScore,
        verification_results: List[VerificationResult]
    ) -> Dict[str, Any]:
        """Format results as JSON-serializable dictionary.
        
        Args:
            quality_score: Quality score object
            verification_results: List of verification results
            
        Returns:
            JSON-serializable dictionary
        """
        return {
            "quality_score": quality_score.overall_score,
            "verification_rate": quality_score.verification_rate,
            "verified_claims": quality_score.verified_claims,
            "total_claims": quality_score.total_claims,
            "flagged_claims": len(quality_score.flagged_claims),
            "confidence_scores": quality_score.confidence_scores,
            "explanation": quality_score.explanation,
            "verification_results": [
                {
                    "claim_text": result.claim.text,
                    "is_verified": result.is_verified,
                    "confidence": result.confidence,
                    "verification_method": result.verification_method,
                    "explanation": result.explanation,
                    "supporting_evidence_count": len(result.supporting_evidence)
                }
                for result in verification_results
            ]
        }
    
    @staticmethod
    def format_correction_report(
        original_explanation: str,
        corrected_explanation: str,
        quality_score: QualityScore,
        correction_strategy: str
    ) -> str:
        """Format correction report.
        
        Args:
            original_explanation: Original explanation text
            corrected_explanation: Corrected explanation text
            quality_score: Quality score object
            correction_strategy: Strategy used for correction
            
        Returns:
            Formatted correction report
        """
        report = f"--- HALLUCINATION CORRECTION REPORT ---\n"
        report += f"Strategy Applied: {correction_strategy.upper()}\n"
        report += f"Quality Score: {quality_score.overall_score:.2f}/1.00\n"
        report += f"Verification Rate: {quality_score.verification_rate:.1%}\n\n"
        
        # Length comparison
        original_length = len(original_explanation)
        corrected_length = len(corrected_explanation)
        length_change = corrected_length - original_length
        
        report += f"Text Length Changes:\n"
        report += f"  Original: {original_length} characters\n"
        report += f"  Corrected: {corrected_length} characters\n"
        report += f"  Change: {length_change:+d} characters\n\n"
        
        # Flagged claims
        if quality_score.flagged_claims:
            report += f"Flagged Claims ({len(quality_score.flagged_claims)}):\n"
            for i, claim in enumerate(quality_score.flagged_claims[:3], 1):
                report += f"  {i}. {claim.text[:60]}...\n"
            if len(quality_score.flagged_claims) > 3:
                report += f"  ... and {len(quality_score.flagged_claims) - 3} more\n"
            report += "\n"
        
        # Recommendation
        if quality_score.overall_score < 0.3:
            report += "RECOMMENDATION: Consider regenerating explanation\n"
        elif quality_score.overall_score < 0.6:
            report += "RECOMMENDATION: Review and manually verify flagged claims\n"
        else:
            report += "RECOMMENDATION: Explanation quality is acceptable\n"
        
        return report
    
    @staticmethod
    def format_console_output(
        quality_score: QualityScore,
        verification_results: List[VerificationResult],
        show_details: bool = True
    ) -> str:
        """Format results for console output.
        
        Args:
            quality_score: Quality score object
            verification_results: List of verification results
            show_details: Whether to show detailed results
            
        Returns:
            Formatted console output string
        """
        output = "\n" + "=" * 60 + "\n"
        output += "SELF-VERIFICATION RESULTS\n"
        output += "=" * 60 + "\n\n"
        
        # Quality summary
        output += f"Quality Score: {quality_score.overall_score:.2f}/1.00\n"
        output += f"Verification Rate: {quality_score.verification_rate:.1%}\n"
        output += f"Claims Verified: {quality_score.verified_claims}/{quality_score.total_claims}\n"
        
        if quality_score.flagged_claims:
            output += f"Flagged Claims: {len(quality_score.flagged_claims)}\n"
        
        # Status indicator
        if quality_score.overall_score >= 0.8:
            output += "\n✓ HIGH QUALITY: Most claims are supported by evidence\n"
        elif quality_score.overall_score >= 0.5:
            output += "\n⚠ MEDIUM QUALITY: Some claims lack supporting evidence\n"
        else:
            output += "\n✗ LOW QUALITY: Many claims are unsupported\n"
        
        if show_details and verification_results:
            output += "\n" + "-" * 60 + "\n"
            output += "CLAIM-BY-CLAIM ANALYSIS\n"
            output += "-" * 60 + "\n"
            
            for i, result in enumerate(verification_results, 1):
                status = "✓ VERIFIED" if result.is_verified else "✗ UNVERIFIED"
                output += f"\n{i}. {result.claim.text[:70]}...\n"
                output += f"   Status: {status}\n"
                output += f"   Confidence: {result.confidence:.2f}\n"
                output += f"   Method: {result.verification_method}\n"
                
                if result.supporting_evidence:
                    output += f"   Evidence: {len(result.supporting_evidence)} pieces\n"
        
        output += "\n" + "=" * 60 + "\n"
        return output


def print_verification_results(
    quality_score: QualityScore,
    verification_results: List[VerificationResult],
    output_format: str = "console"
) -> str:
    """Print verification results in specified format.
    
    Args:
        quality_score: Quality score object
        verification_results: List of verification results
        output_format: Output format ("console", "summary", "detailed", "json")
        
    Returns:
        Formatted output string
    """
    formatter = SelfVerificationOutputFormatter()
    
    if output_format == "console":
        return formatter.format_console_output(quality_score, verification_results)
    elif output_format == "summary":
        return formatter.format_quality_summary(quality_score)
    elif output_format == "detailed":
        return formatter.format_detailed_results(verification_results, quality_score)
    elif output_format == "json":
        import json
        return json.dumps(
            formatter.format_json_output(quality_score, verification_results),
            indent=2,
            ensure_ascii=False
        )
    else:
        raise ValueError(f"Unknown output format: {output_format}")


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample explanation with potential hallucinations
    sample_explanation = """
    Tuyên bố về dân số Việt Nam được hỗ trợ bởi bằng chứng. 
    Theo thống kê chính thức, Việt Nam có 98 triệu dân vào năm 2023.
    Đây là con số chính xác được Tổng cục Thống kê công bố.
    Ngoài ra, GDP của Việt Nam đạt 500 tỷ USD năm 2023.
    Việt Nam cũng có 65 tỉnh thành phố trực thuộc trung ương.
    """
    
    # Sample evidence
    sample_evidence = [
        Evidence(
            text="Việt Nam có dân số khoảng 98 triệu người theo thống kê năm 2023",
            source_url="https://gso.gov.vn/population-2023",
            source_title="Thống kê dân số 2023",
            credibility_score=0.9,
            language="vi"
        )
    ]
    
    # Test self-verification
    verifier = SelfVerificationModule()
    quality_score, results = verifier.verify_explanation(sample_explanation, sample_evidence)
    
    # Test different output formats
    print("=== CONSOLE OUTPUT ===")
    print(print_verification_results(quality_score, results, "console"))
    
    print("\n=== SUMMARY OUTPUT ===")
    print(print_verification_results(quality_score, results, "summary"))
    
    print("\n=== JSON OUTPUT ===")
    print(print_verification_results(quality_score, results, "json"))
