"""RAG-based explanation generator for Vietnamese fact-checking system."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import defaultdict

try:
    from .data_models import Claim, Evidence, Verdict, ReasoningStep
    from .llm_controller import LLMController, create_llm_controller
    from .exa_search_client import ExaSearchClient
    from .web_crawler import WebCrawler
    from .search_query_generator import SearchQueryGenerator
except ImportError:
    from data_models import Claim, Evidence, Verdict, ReasoningStep
    from llm_controller import LLMController, create_llm_controller
    from exa_search_client import ExaSearchClient
    from web_crawler import WebCrawler
    from search_query_generator import SearchQueryGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvidenceScore:
    """Scored evidence piece for RAG retrieval."""
    evidence: Evidence
    relevance_score: float
    stance_weight: float
    credibility_weight: float
    final_score: float


@dataclass
class RAGContext:
    """Context for RAG generation."""
    claim: Claim
    verdict_label: str
    confidence_scores: Dict[str, float]
    top_evidence: List[EvidenceScore]
    reasoning_trace: List[ReasoningStep]
    contradictory_evidence: Optional[Dict[str, List[EvidenceScore]]] = None


class EvidenceRetriever:
    """Retrieves and scores evidence for RAG generation."""
    
    # Minimum thresholds for evidence quality
    MIN_RELEVANCE_THRESHOLD = 0.15  # Filter out very low relevance evidence
    MIN_EVIDENCE_LENGTH = 20  # Minimum meaningful text length
    
    def __init__(
        self,
        stance_weight: float = 0.4,
        credibility_weight: float = 0.3,
        relevance_weight: float = 0.3,
        top_k: int = 5
    ):
        """Initialize evidence retriever.
        
        Args:
            stance_weight: Weight for stance alignment in scoring
            credibility_weight: Weight for source credibility in scoring
            relevance_weight: Weight for text relevance in scoring
            top_k: Number of top evidence pieces to retrieve
        """
        self.stance_weight = stance_weight
        self.credibility_weight = credibility_weight
        self.relevance_weight = relevance_weight
        self.top_k = top_k
        
        logger.info(f"Initialized evidence retriever with weights: stance={stance_weight}, "
                   f"credibility={credibility_weight}, relevance={relevance_weight}")
    
    def retrieve_evidence(
        self,
        claim: Claim,
        evidence_list: List[Evidence],
        verdict_label: str
    ) -> Tuple[List[EvidenceScore], Optional[Dict[str, List[EvidenceScore]]]]:
        """Retrieve and score evidence for RAG generation.
        
        Args:
            claim: The claim being verified
            evidence_list: List of collected evidence
            verdict_label: Predicted verdict label
            
        Returns:
            Tuple of (top_evidence, contradictory_evidence)
        """
        if not evidence_list:
            logger.warning("No evidence provided for retrieval")
            return [], None
        
        # Score all evidence pieces, filtering out low-quality ones
        scored_evidence = []
        filtered_count = 0
        for evidence in evidence_list:
            # Pre-filter: check minimum text quality
            meaningful_text = re.sub(r'[^\w\s]', '', evidence.text)
            if len(meaningful_text.strip()) < self.MIN_EVIDENCE_LENGTH:
                logger.debug(f"Filtering short evidence: '{evidence.text[:30]}...'")
                filtered_count += 1
                continue
            
            score = self._score_evidence(claim, evidence, verdict_label)
            
            # Post-filter: check minimum relevance
            if score.relevance_score < self.MIN_RELEVANCE_THRESHOLD:
                logger.debug(f"Filtering low-relevance evidence: '{evidence.text[:50]}...' (relevance={score.relevance_score:.2f})")
                filtered_count += 1
                continue
            
            scored_evidence.append(score)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} low-quality evidence pieces")
        
        # Sort by final score
        scored_evidence.sort(key=lambda x: x.final_score, reverse=True)
        
        # Get top-k evidence
        top_evidence = scored_evidence[:self.top_k]
        
        # Detect contradictory evidence
        contradictory = self._detect_contradictory_evidence(scored_evidence, verdict_label)
        
        logger.info(f"Retrieved {len(top_evidence)} top evidence pieces, "
                   f"found {len(contradictory) if contradictory else 0} contradictory groups")
        
        return top_evidence, contradictory
    
    def _score_evidence(self, claim: Claim, evidence: Evidence, verdict_label: str) -> EvidenceScore:
        """Score a single evidence piece for relevance to the claim and verdict.
        
        Args:
            claim: The claim being verified
            evidence: Evidence piece to score
            verdict_label: Predicted verdict label
            
        Returns:
            EvidenceScore with computed scores
        """
        # Stance alignment score
        stance_score = self._compute_stance_score(evidence, verdict_label)
        
        # Credibility score (already computed)
        credibility_score = evidence.credibility_score
        
        # Text relevance score (simple keyword overlap for now)
        relevance_score = self._compute_relevance_score(claim.text, evidence.text)
        
        # Additional check for Vietnamese text encoding issues
        if relevance_score < 0.1 and len(evidence.text) > 100:
             # Try simpler containment check for robust matching
             claim_keywords = self._extract_keywords(claim.text.lower())
             if any(kw in evidence.text.lower() for kw in claim_keywords if len(kw) > 4):
                 relevance_score = max(relevance_score, 0.3)

        # Combine scores
        final_score = (
            self.stance_weight * stance_score +
            self.credibility_weight * credibility_score +
            self.relevance_weight * relevance_score
        )
        
        return EvidenceScore(
            evidence=evidence,
            relevance_score=relevance_score,
            stance_weight=stance_score,
            credibility_weight=credibility_score,
            final_score=final_score
        )
    
    def _compute_stance_score(self, evidence: Evidence, verdict_label: str) -> float:
        """Compute stance alignment score.
        
        Args:
            evidence: Evidence piece
            verdict_label: Predicted verdict label
            
        Returns:
            Stance alignment score (0.0 to 1.0)
        """
        if not evidence.stance or not evidence.stance_confidence:
            return 0.5  # Neutral if no stance detected
        
        # Normalize stance string
        evidence_stance = evidence.stance.lower().strip()
        
        # Map verdict to expected stance
        expected_stance_map = {
            "supported": "support",
            "refuted": "refute",
            "not_enough_info": "neutral"
        }
        
        expected_stance = expected_stance_map.get(verdict_label, "neutral")
        
        if evidence_stance == expected_stance:
            return evidence.stance_confidence
        elif evidence_stance == "neutral":
            return 0.5
        else:
            # Contradictory stance - lower score but not zero
            return 0.2 * (1 - evidence.stance_confidence)
    
    def _compute_relevance_score(self, claim_text: str, evidence_text: str) -> float:
        """Compute text relevance score using containment.
        
        Args:
            claim_text: Claim text
            evidence_text: Evidence text
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Simple keyword containment: what % of claim keywords are in evidence?
        claim_words = self._extract_keywords(claim_text.lower())
        evidence_words = set(self._extract_keywords(evidence_text.lower()))
        
        if not claim_words:
            return 0.0
            
        # Count matches
        matches = sum(1 for w in claim_words if w in evidence_words)
        containment_score = matches / len(claim_words)
        
        return containment_score
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple tokenization).
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out common stop words (basic list)
        stop_words = {
            'và', 'của', 'trong', 'với', 'là', 'có', 'được', 'này', 'đó', 'các', 'một',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'
        }
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    def _detect_contradictory_evidence(
        self,
        scored_evidence: List[EvidenceScore],
        verdict_label: str
    ) -> Optional[Dict[str, List[EvidenceScore]]]:
        """Detect contradictory evidence groups.
        
        Args:
            scored_evidence: List of scored evidence
            verdict_label: Predicted verdict label
            
        Returns:
            Dictionary mapping stance to evidence lists, or None if no contradictions
        """
        # Group evidence by stance
        stance_groups = defaultdict(list)
        
        for score in scored_evidence:
            if score.evidence.stance and score.evidence.stance_confidence and score.evidence.stance_confidence > 0.6:
                stance_groups[score.evidence.stance].append(score)
        
        # Check if we have contradictory stances
        has_support = len(stance_groups.get("support", [])) > 0
        has_refute = len(stance_groups.get("refute", [])) > 0
        
        if has_support and has_refute:
            logger.info("Detected contradictory evidence: both supporting and refuting evidence found")
            return dict(stance_groups)
        
        return None


class RAGGenerator:
    """Generates explanations using Retrieval-Augmented Generation."""
    
    def __init__(self, llm_controller: Optional[LLMController] = None):
        """Initialize RAG generator.
        
        Args:
            llm_controller: LLM controller for generation
        """
        self.llm_controller = llm_controller or create_llm_controller()
        logger.info("Initialized RAG generator")
    
    def generate_explanation(self, context: RAGContext) -> str:
        """Generate explanation using RAG.
        
        Args:
            context: RAG context with claim, verdict, and evidence
            
        Returns:
            Generated explanation with citations
        """
        try:
            # Create prompt with evidence
            prompt = self._create_rag_prompt(context)
            
            # Generate explanation
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_controller.generate(
                messages=messages,
                max_tokens=800,
                temperature=0.3
            )
            
            explanation = response.content.strip()
            
            # Post-process to ensure citations are properly formatted
            explanation = self._format_citations(explanation, context.top_evidence)
            
            logger.info(f"Generated explanation with {len(context.top_evidence)} evidence pieces")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate RAG explanation: {e}")
            return self._generate_fallback_explanation(context)
    
    def _create_rag_prompt(self, context: RAGContext) -> str:
        """Create RAG prompt with claim, verdict, and evidence.
        
        Args:
            context: RAG context
            
        Returns:
            Formatted prompt string
        """
        # Format evidence with citations
        evidence_text = ""
        for i, scored_ev in enumerate(context.top_evidence, 1):
            ev = scored_ev.evidence
            evidence_text += f"\n[{i}] {ev.text}\n"
            evidence_text += f"    Source: {ev.source_title} ({ev.source_url})\n"
            evidence_text += f"    Credibility: {ev.credibility_score:.2f}\n"
            if ev.stance:
                evidence_text += f"    Stance: {ev.stance} (confidence: {ev.stance_confidence:.2f})\n"
        
        # Format reasoning trace
        reasoning_text = ""
        if context.reasoning_trace:
            reasoning_text = "\n\nREASONING TRACE:\n"
            for i, step in enumerate(context.reasoning_trace[-3:], 1):  # Last 3 steps
                reasoning_text += f"{i}. {step.thought}\n"
                reasoning_text += f"   Action: {step.action}\n"
                reasoning_text += f"   Result: {step.observation[:200]}...\n\n"
        
        # Handle contradictory evidence
        contradiction_text = ""
        if context.contradictory_evidence:
            contradiction_text = "\n\nCONTRADICTORY EVIDENCE DETECTED:\n"
            for stance, evidence_list in context.contradictory_evidence.items():
                contradiction_text += f"\n{stance.upper()} evidence:\n"
                for scored_ev in evidence_list[:2]:  # Top 2 per stance
                    ev = scored_ev.evidence
                    contradiction_text += f"- {ev.text[:150]}... (Source: {ev.source_title})\n"
        
        prompt = f"""
CLAIM TO VERIFY: {context.claim.text}

VERDICT: {context.verdict_label.upper()}
CONFIDENCE SCORES:
- Supported: {context.confidence_scores.get('supported', 0):.2f}
- Refuted: {context.confidence_scores.get('refuted', 0):.2f}
- Not Enough Info: {context.confidence_scores.get('not_enough_info', 0):.2f}

EVIDENCE:{evidence_text}

{reasoning_text}

{contradiction_text}

Please generate a comprehensive explanation for this fact-checking result. Include:
1. Clear statement of the verdict and confidence
2. Summary of key evidence with inline citations [1], [2], etc.
3. Explanation of the reasoning process
4. If contradictory evidence exists, present both sides fairly
5. Source URLs for verification

Ensure all facts are grounded in the provided evidence and include proper citations.
"""
        
        return prompt.strip()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for RAG generation with temporal context."""
        try:
            from src.temporal_context import get_current_datetime
            current_dt = get_current_datetime()
        except ImportError:
            from datetime import datetime
            current_dt = datetime.now().strftime("%B %d, %Y at %H:%M")
        
        return f"""You are an expert fact-checker generating explanations for Vietnamese claims. 

CURRENT DATE AND TIME: {current_dt}

Your task is to create clear, accurate explanations that:
- Are grounded entirely in the provided evidence
- Account for temporal context (compare dates in evidence vs current date)
- Include inline citations [1], [2], etc. for all factual claims
- Present information objectively and fairly
- Explain the reasoning process transparently
- Handle contradictory evidence by presenting both sides
- Include source URLs for verification
- If evidence refers to events with specific dates, clarify whether those events are past, present, or future

Write in a clear, professional tone suitable for researchers and the general public.
Use Vietnamese when appropriate for Vietnamese sources, but ensure explanations are accessible."""
    
    def _format_citations(self, explanation: str, evidence_list: List[EvidenceScore]) -> str:
        """Format and validate citations in the explanation.
        
        Args:
            explanation: Generated explanation text
            evidence_list: List of evidence with scores
            
        Returns:
            Explanation with properly formatted citations
        """
        # Add source list at the end if not present
        if "Sources:" not in explanation and "Nguồn:" not in explanation:
            explanation += "\n\nSources:\n"
            for i, scored_ev in enumerate(evidence_list, 1):
                ev = scored_ev.evidence
                explanation += f"[{i}] {ev.source_title} - {ev.source_url}\n"
        
        return explanation
    
    def _generate_fallback_explanation(self, context: RAGContext) -> str:
        """Generate fallback explanation when RAG fails.
        
        Args:
            context: RAG context
            
        Returns:
            Basic explanation
        """
        verdict_map = {
            "supported": "được hỗ trợ bởi bằng chứng",
            "refuted": "bị bác bỏ bởi bằng chứng", 
            "not_enough_info": "không có đủ thông tin để xác minh"
        }
        
        verdict_text = verdict_map.get(context.verdict_label, "không xác định")
        confidence = max(context.confidence_scores.values()) if context.confidence_scores else 0.5
        
        explanation = f"Tuyên bố '{context.claim.text}' {verdict_text} với độ tin cậy {confidence:.2f}.\n\n"
        
        if context.top_evidence:
            explanation += "Bằng chứng chính:\n"
            for i, scored_ev in enumerate(context.top_evidence[:3], 1):
                ev = scored_ev.evidence
                explanation += f"[{i}] {ev.text[:200]}... (Nguồn: {ev.source_title})\n"
            
            explanation += "\nSources:\n"
            for i, scored_ev in enumerate(context.top_evidence[:3], 1):
                ev = scored_ev.evidence
                explanation += f"[{i}] {ev.source_title} - {ev.source_url}\n"
        
        return explanation


class ReasoningTraceFormatter:
    """Formats reasoning traces for human readability."""
    
    @staticmethod
    def format_trace(reasoning_steps: List[ReasoningStep]) -> str:
        """Format reasoning trace for inclusion in explanation.
        
        Args:
            reasoning_steps: List of reasoning steps from ReAct loop
            
        Returns:
            Formatted reasoning trace
        """
        if not reasoning_steps:
            return "No reasoning trace available."
        
        formatted = "REASONING PROCESS:\n\n"
        
        for i, step in enumerate(reasoning_steps, 1):
            formatted += f"Step {i}:\n"
            formatted += f"Thought: {step.thought}\n"
            formatted += f"Action: {step.action}\n"
            
            # Format action input
            if step.action_input:
                if isinstance(step.action_input, dict):
                    for key, value in step.action_input.items():
                        formatted += f"  {key}: {value}\n"
                else:
                    formatted += f"  Input: {step.action_input}\n"
            
            # Truncate long observations
            observation = step.observation
            if len(observation) > 300:
                observation = observation[:300] + "..."
            
            formatted += f"Observation: {observation}\n\n"
        
        return formatted


class RAGExplanationGenerator:
    """Main RAG-based explanation generator with self-verification."""
    
    def __init__(
        self,
        llm_controller: Optional[LLMController] = None,
        evidence_retriever: Optional[EvidenceRetriever] = None,
        enable_self_verification: bool = True,
        search_client: Optional[ExaSearchClient] = None,
        web_crawler: Optional[WebCrawler] = None,
        query_generator: Optional[SearchQueryGenerator] = None
    ):
        """Initialize RAG explanation generator.
        
        Args:
            llm_controller: LLM controller for generation
            evidence_retriever: Evidence retriever for scoring and selection
            enable_self_verification: Whether to enable self-verification
            search_client: Search client for verification
            web_crawler: Web crawler for verification
            query_generator: Query generator for verification
        """
        self.llm_controller = llm_controller or create_llm_controller()
        self.evidence_retriever = evidence_retriever or EvidenceRetriever()
        self.rag_generator = RAGGenerator(self.llm_controller)
        self.enable_self_verification = enable_self_verification
        
        # Initialize self-verification module if enabled
        if self.enable_self_verification:
            try:
                from .self_verification import SelfVerificationModule, ClaimVerifier
                
                # Setup claim verifier with search capabilities if provided
                claim_verifier = None
                if search_client:
                    # If query_generator not provided, create a default one
                    if not query_generator:
                        from .search_query_generator import SearchQueryGenerator
                        query_generator = SearchQueryGenerator()
                        
                    claim_verifier = ClaimVerifier(
                        search_client=search_client,
                        web_crawler=web_crawler,
                        query_generator=query_generator
                    )
                
                self.self_verifier = SelfVerificationModule(claim_verifier=claim_verifier)
                logger.info(f"Self-verification enabled (with search: {bool(search_client)})")
            except ImportError as e:
                logger.warning(f"Self-verification module not available, disabling: {e}")
                self.enable_self_verification = False
                self.self_verifier = None
        else:
            self.self_verifier = None
        
        logger.info("Initialized RAG explanation generator")
    
    def generate_explanation(
        self,
        claim: Claim,
        verdict: Verdict,
        evidence_list: List[Evidence],
        reasoning_steps: List[ReasoningStep],
        apply_self_verification: bool = True
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate comprehensive explanation using RAG with optional self-verification.
        
        Args:
            claim: The verified claim
            verdict: Verdict with label and confidence scores
            evidence_list: List of collected evidence
            reasoning_steps: ReAct reasoning steps
            apply_self_verification: Whether to apply self-verification
            
        Returns:
            Tuple of (explanation, verification_metadata)
            - explanation: Generated explanation with citations and reasoning trace
            - verification_metadata: Self-verification results (if enabled)
        """
        try:
            # Retrieve and score evidence
            top_evidence, contradictory = self.evidence_retriever.retrieve_evidence(
                claim, evidence_list, verdict.label
            )
            
            # Create RAG context
            context = RAGContext(
                claim=claim,
                verdict_label=verdict.label,
                confidence_scores=verdict.confidence_scores,
                top_evidence=top_evidence,
                reasoning_trace=reasoning_steps,
                contradictory_evidence=contradictory
            )
            
            # Generate explanation
            explanation = self.rag_generator.generate_explanation(context)
            
            # Apply self-verification if enabled (BEFORE appending reasoning trace)
            verification_metadata = None
            if self.enable_self_verification and apply_self_verification and self.self_verifier:
                explanation, verification_metadata = self._apply_self_verification(
                    explanation, evidence_list
                )

            # Add reasoning trace if requested (AFTER verification)
            if reasoning_steps:
                trace = ReasoningTraceFormatter.format_trace(reasoning_steps)
                explanation += f"\n\n{trace}"
            
            logger.info("Successfully generated RAG explanation")
            return explanation, verification_metadata
            
        except Exception as e:
            logger.error(f"Failed to generate RAG explanation: {e}")
            # Return basic explanation as fallback
            basic_explanation = self._generate_basic_explanation(claim, verdict, evidence_list)
            return basic_explanation, None
    
    def _generate_basic_explanation(
        self,
        claim: Claim,
        verdict: Verdict,
        evidence_list: List[Evidence]
    ) -> str:
        """Generate basic explanation as fallback.
        
        Args:
            claim: The verified claim
            verdict: Verdict with label and confidence scores
            evidence_list: List of collected evidence
            
        Returns:
            Basic explanation
        """
        verdict_map = {
            "supported": "được hỗ trợ bởi bằng chứng",
            "refuted": "bị bác bỏ bởi bằng chứng",
            "not_enough_info": "không có đủ thông tin để xác minh"
        }
        
        verdict_text = verdict_map.get(verdict.label, "không xác định")
        confidence = max(verdict.confidence_scores.values()) if verdict.confidence_scores else 0.5
        
        explanation = f"Tuyên bố '{claim.text}' {verdict_text} với độ tin cậy {confidence:.2f}.\n\n"
        
        if evidence_list:
            explanation += "Bằng chứng được thu thập:\n"
            for i, evidence in enumerate(evidence_list[:3], 1):
                explanation += f"[{i}] {evidence.text[:200]}...\n"
                explanation += f"    Nguồn: {evidence.source_title} ({evidence.source_url})\n\n"
        
        return explanation
    
    def _apply_self_verification(
        self,
        explanation: str,
        evidence_list: List[Evidence],
        use_structured_output: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply self-verification to the generated explanation.
        
        Args:
            explanation: Generated explanation
            evidence_list: Evidence used for generation
            
        Returns:
            Tuple of (corrected_explanation, verification_metadata)
        """
        try:
            logger.info("Applying self-verification to explanation")
            
            # Run self-verification
            quality_score, verification_results = self.self_verifier.verify_explanation(
                explanation, evidence_list
            )
            
            # Apply corrections based on quality score
            corrected_explanation = self.self_verifier.correct_hallucinations(
                explanation, verification_results, quality_score, "adaptive"
            )
            
            # Import output formatter and schemas
            from .self_verification import SelfVerificationOutputFormatter
            from .verification_schemas import VerificationSchemas
            
            formatter = SelfVerificationOutputFormatter()
            
            # Create structured output if requested
            if use_structured_output:
                verification_metadata = formatter.to_structured_output(
                    quality_score=quality_score,
                    verification_results=verification_results,
                    correction_applied=corrected_explanation != explanation,
                    correction_strategy="adaptive",
                    original_length=len(explanation),
                    corrected_length=len(corrected_explanation)
                )
            else:
                # Legacy format for backward compatibility
                verification_metadata = formatter.format_json_output(quality_score, verification_results)
                verification_metadata.update({
                    "correction_applied": corrected_explanation != explanation,
                    "original_length": len(explanation),
                    "corrected_length": len(corrected_explanation),
                    "length_change": len(corrected_explanation) - len(explanation)
                })
            
            # Add verification summary to explanation using proper formatting
            verification_summary = formatter.format_quality_summary(quality_score)
            
            # Add method breakdown
            method_counts = {}
            for result in verification_results:
                method = result.verification_method
                method_counts[method] = method_counts.get(method, 0) + 1
            
            verification_summary += "\nVerification Methods Used:\n"
            for method, count in method_counts.items():
                verification_summary += f"- {method}: {count} claims\n"
            
            corrected_explanation += f"\n\n{verification_summary}"
            
            logger.info(f"Self-verification complete: quality={quality_score.overall_score:.2f}, "
                       f"rate={quality_score.verification_rate:.1%}")
            
            return corrected_explanation, verification_metadata
            
        except Exception as e:
            logger.error(f"Self-verification failed: {e}")
            # Return original explanation if verification fails
            return explanation, {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    claim = Claim(text="Việt Nam có 63 tỉnh thành", confidence=0.9)
    
    evidence = Evidence(
        text="Việt Nam hiện có 63 tỉnh thành phố trực thuộc trung ương",
        source_url="https://example.com/vietnam-provinces",
        source_title="Danh sách tỉnh thành Việt Nam",
        credibility_score=0.8,
        stance="support",
        stance_confidence=0.9
    )
    
    verdict = Verdict(
        claim_id=claim.id,
        label="supported",
        confidence_scores={"supported": 0.85, "refuted": 0.10, "not_enough_info": 0.05}
    )
    
    reasoning_step = ReasoningStep(
        iteration=1,
        thought="I need to verify the number of provinces in Vietnam",
        action="search",
        action_input={"query": "Vietnam provinces number 63"},
        observation="Found information confirming 63 provinces and cities"
    )
    
    # Test RAG generator
    generator = RAGExplanationGenerator()
    explanation, verification_metadata = generator.generate_explanation(
        claim=claim,
        verdict=verdict,
        evidence_list=[evidence],
        reasoning_steps=[reasoning_step]
    )
    
    print("Generated explanation:")
    print(explanation)
    
    if verification_metadata:
        print(f"\nVerification metadata:")
        print(f"Quality score: {verification_metadata.get('quality_score', 'N/A')}")
        print(f"Verification rate: {verification_metadata.get('verification_rate', 'N/A')}")