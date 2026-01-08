"""Verdict generation logic for the ReAct agent."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.data_models import Evidence, Verdict
from src.agent.models import AgentState, ReasoningStep

logger = logging.getLogger(__name__)


class VerdictGenerator:
    """Generates verdicts based on collected evidence.
    
    This class handles all verdict-related logic including:
    - Final verdict generation using LLM
    - Evidence-based fallback verdict generation
    - Confidence capping and adjustment
    - Sub-verdict aggregation
    
    Attributes:
        llm_controller: LLM controller for generating verdicts.
    """
    
    def __init__(self, llm_controller: Any) -> None:
        """Initialize the verdict generator.
        
        Args:
            llm_controller: LLMController instance for verdict generation.
        """
        self.llm_controller = llm_controller
    
    def get_verdict_system_prompt(self) -> str:
        """Get system prompt for verdict generation with temporal context.
        
        Returns:
            System prompt string for the LLM.
        """
        try:
            from src.temporal_context import get_temporal_context_prompt
            temporal = get_temporal_context_prompt()
        except ImportError:
            temporal = f"CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}"
        
        return f"""You are a fact-checking expert. Based on the evidence collected, provide a final verdict for the claim.

{temporal}

CRITICAL TEMPORAL REASONING RULES:
1. Compare ALL dates in evidence with the CURRENT DATE above
2. If evidence mentions a FUTURE date (after current date), mark it as "FUTURE POLICY" - do NOT use it to refute CURRENT facts
3. If evidence mentions a PAST date that has already occurred, treat that information as CURRENT/HISTORICAL
4. For claims about current state (e.g., "Vietnam has X provinces"), only use evidence about the CURRENT state

SOURCE PRIORITY (use higher priority sources when they conflict):
1. Official government sources (.gov.vn, chinhphu.vn) - HIGHEST
2. Official statistics (gso.gov.vn) - HIGH  
3. Major news agencies (VNExpress, Tuoi Tre, VTV) - MEDIUM
4. Wikipedia, others - LOWER

DECISION GUIDELINES:
- Be decisive! If evidence leans heavily one way, choose SUPPORTED or REFUTED.
- Use "Partially Supported" ONLY if:
  a) The claim has multiple parts and some are true while others are false.
  b) The claim is true only under specific conditions not fully met.
  c) There is a genuine, unresolved conflict between high-credibility sources.
- Do NOT use "Partially Supported" just because you are slightly uncertain. If evidence is good, go for it.
- Trust official government data over general news/wiki if they conflict.
- For numerical claims with "about/approx/khoảng", use SUPPORTED if the difference is small (< 5%).

Your verdict must be exactly one of:
- Supported
- Refuted
- Partially Supported
- Not Enough Info

Provide confidence scores for each possibility and a brief explanation."""
    
    def generate_final_verdict(self, state: AgentState) -> Verdict:
        """Generate final verdict based on collected evidence.
        
        Args:
            state: Agent state with collected evidence.
            
        Returns:
            Final verdict for the claim.
        """
        try:
            # Prepare evidence summary
            evidence_summary = state.get_evidence_summary()
            reasoning_trace = self._format_reasoning_trace(state.reasoning_steps)
            
            # Apply legal/temporal analysis to evidence
            temporal_warnings = self._analyze_temporal_context(state.collected_evidence)
            
            # Create prompt for final verdict
            messages = [
                {"role": "system", "content": self.get_verdict_system_prompt()},
                {"role": "user", "content": f"""
CLAIM: {state.claim.text}

EVIDENCE COLLECTED:
{evidence_summary}

REASONING TRACE:
{reasoning_trace}

Please provide your final verdict in this format:

VERDICT: [Supported/Refuted/Partially Supported/Mixed/Not Enough Info]
CONFIDENCE_SUPPORTED: [0.0-1.0]
CONFIDENCE_REFUTED: [0.0-1.0]
CONFIDENCE_NOT_ENOUGH_INFO: [0.0-1.0]
EXPLANATION: [Brief explanation of your reasoning]
"""}
            ]
            
            response = self.llm_controller.generate(
                messages=messages,
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse verdict
            verdict_text = response.content
            verdict_label = self._extract_verdict_label(verdict_text)
            confidence_scores = self._extract_confidence_scores(verdict_text)
            explanation = self._extract_explanation(verdict_text)
            
            # Apply confidence capping based on evidence quality
            capped_scores, cap_reason, temporal_notes, adjusted_label = self._apply_confidence_cap(
                confidence_scores,
                state.collected_evidence,
                temporal_warnings,
                current_label=verdict_label
            )
            
            # Create verdict object
            verdict = Verdict(
                claim_id=state.claim.id,
                label=adjusted_label or verdict_label,
                confidence_scores=capped_scores,
                supporting_evidence=[e for e in state.collected_evidence if e.stance == "support"],
                refuting_evidence=[e for e in state.collected_evidence if e.stance == "refute"],
                explanation=explanation,
                reasoning_trace=[step.thought for step in state.reasoning_steps],
                quality_score=max(capped_scores.values()) if capped_scores else 0.5,
                confidence_cap_reason=cap_reason,
                temporal_context=temporal_notes if temporal_notes else None
            )
            
            return verdict
            
        except Exception as e:
            logger.error(f"Failed to generate final verdict: {e}")
            return self.generate_evidence_based_verdict(state, error_reason=str(e))
    
    def _analyze_temporal_context(self, evidence_list: List[Evidence]) -> List[str]:
        """Analyze temporal context of evidence.
        
        Args:
            evidence_list: List of collected evidence.
            
        Returns:
            List of temporal warning messages.
        """
        temporal_warnings = []
        
        try:
            from src.legal_temporal_reasoning import LegalTemporalReasoner
            temporal_reasoner = LegalTemporalReasoner()
            
            for evidence in evidence_list:
                analysis = temporal_reasoner.analyze(evidence)
                evidence.legal_status = analysis.legal_status.value
                if not analysis.can_refute_current:
                    temporal_warnings.append(
                        f"⚠️ {evidence.source_title[:30] if evidence.source_title else 'Source'}: {analysis.reasoning}"
                    )
        except ImportError:
            pass
        
        return temporal_warnings
    
    def _format_reasoning_trace(self, steps: List[ReasoningStep]) -> str:
        """Format reasoning steps into readable trace.
        
        Args:
            steps: List of reasoning steps.
            
        Returns:
            Formatted trace string.
        """
        trace_parts = []
        for step in steps:
            part = f"Step {step.step_number}: {step.thought}"
            if step.action:
                part += f"\nAction: {step.action}({step.action_params})"
            if step.observation:
                obs_preview = step.observation[:200] if step.observation else ""
                part += f"\nObservation: {obs_preview}..."
            trace_parts.append(part)
        
        return "\n\n".join(trace_parts)
    
    def _extract_verdict_label(self, text: str) -> str:
        """Extract verdict label from LLM response.
        
        Args:
            text: LLM response text.
            
        Returns:
            Normalized verdict label string.
        """
        # Check specific labels first
        if re.search(r'Partially Supported', text, re.IGNORECASE):
            return "partially_supported"
        if re.search(r'Mixed', text, re.IGNORECASE):
            return "mixed"
        if re.search(r'Context Dependent', text, re.IGNORECASE):
            return "context_dependent"
        
        # Check standard labels
        patterns = [
            r'VERDICT:\s*(Supported|Refuted|Not Enough Info)',
            r'verdict:\s*(supported|refuted|not enough info)',
            r'(Supported|Refuted|Not Enough Info)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                label = match.group(1).lower()
                if "support" in label:
                    return "supported"
                elif "refut" in label:
                    return "refuted"
                elif "not enough" in label or "insufficient" in label:
                    return "not_enough_info"
        
        return "not_enough_info"
    
    def _extract_confidence_scores(self, text: str) -> Dict[str, float]:
        """Extract confidence scores from LLM response.
        
        Args:
            text: LLM response text.
            
        Returns:
            Dictionary of normalized confidence scores.
        """
        scores = {
            "supported": 0.33,
            "refuted": 0.33,
            "not_enough_info": 0.34
        }
        
        patterns = [
            r'CONFIDENCE_SUPPORTED:\s*([0-9.]+)',
            r'CONFIDENCE_REFUTED:\s*([0-9.]+)',
            r'CONFIDENCE_NOT_ENOUGH_INFO:\s*([0-9.]+)'
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    key = list(scores.keys())[i]
                    scores[key] = score
        
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from LLM response.
        
        Args:
            text: LLM response text.
            
        Returns:
            Cleaned explanation string.
        """
        content = "No explanation provided."
        
        patterns = [
            r'EXPLANATION:\s*(.+?)(?=\n\n|$)',
            r'Explanation:\s*(.+?)(?=\n\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                break
        
        # Fallback: last paragraph
        if content == "No explanation provided.":
            paragraphs = text.split('\n\n')
            if paragraphs:
                content = paragraphs[-1].strip()
        
        # Cleanup internal artifacts
        content = re.sub(
            r'(Step \d+:|Thought:|Action:|Observation:).*?(\n|$)', 
            '', 
            content, 
            flags=re.IGNORECASE
        )
        content = re.sub(r'\w+\(.*?\)', '', content)
        
        return content.strip()
    
    def generate_evidence_based_verdict(
        self, 
        state: AgentState, 
        error_reason: str = ""
    ) -> Verdict:
        """Generate verdict based on evidence when LLM is unavailable.
        
        This is a fallback method using rule-based logic.
        
        Args:
            state: Agent state with collected evidence.
            error_reason: Reason for the fallback.
            
        Returns:
            Verdict based on evidence analysis.
        """
        evidence_list = state.collected_evidence
        claim_text = state.claim.text.lower()
        
        if not evidence_list:
            return Verdict(
                claim_id=state.claim.id,
                label="not_enough_info",
                confidence_scores={
                    "supported": 0.1,
                    "refuted": 0.1,
                    "not_enough_info": 0.8
                },
                explanation=f"Không thu thập được bằng chứng. Lỗi: {error_reason}",
                quality_score=0.0,
                confidence_cap_reason="No evidence collected; verdict generated without LLM"
            )
        
        # Analyze evidence
        signals = self._analyze_evidence_signals(evidence_list, claim_text)
        
        # Determine verdict
        label, confidence_scores = self._determine_verdict_from_signals(signals)
        
        # Build explanation
        explanation = self._build_fallback_explanation(signals, error_reason)
        
        return Verdict(
            claim_id=state.claim.id,
            label=label,
            confidence_scores=confidence_scores,
            supporting_evidence=[e for e, t, _ in signals['key_evidence'] if t == "supporting"],
            refuting_evidence=[e for e, t, _ in signals['key_evidence'] if t == "refuting"],
            explanation=explanation,
            reasoning_trace=[step.thought for step in state.reasoning_steps],
            quality_score=signals['avg_credibility'] * 0.7,
            confidence_cap_reason="Evidence-based fallback due to API error; LLM unavailable"
        )
    
    def _analyze_evidence_signals(
        self, 
        evidence_list: List[Evidence], 
        claim_text: str
    ) -> Dict[str, Any]:
        """Analyze evidence for supporting/refuting signals.
        
        Args:
            evidence_list: List of evidence to analyze.
            claim_text: Lowercase claim text for comparison.
            
        Returns:
            Dictionary containing analysis results.
        """
        supporting_signals = 0
        refuting_signals = 0
        gov_source_count = 0
        total_credibility = 0.0
        key_evidence = []
        
        for evidence in evidence_list:
            evidence_text = evidence.text.lower() if evidence.text else ""
            source_url = evidence.source_url or ""
            domain = urlparse(source_url).netloc.lower() if source_url else ""
            
            # Count government sources
            is_gov_source = '.gov.vn' in domain or 'chinhphu.vn' in domain
            if is_gov_source:
                gov_source_count += 1
            
            # Track credibility
            if evidence.credibility_score and evidence.credibility_score > 0:
                total_credibility += evidence.credibility_score
            
            # Check temporal context
            if evidence.temporal_marker == "FUTURE_POLICY":
                continue
            
            # Extract and compare numbers
            signals = self._compare_numbers(
                claim_text, evidence_text, is_gov_source
            )
            supporting_signals += signals['supporting']
            refuting_signals += signals['refuting']
            key_evidence.extend(signals['evidence'])
            
            # Legal status check
            if evidence.legal_status in ["effective", "implemented"]:
                if any(word in evidence_text for word in claim_text.split() if len(word) > 3):
                    weight = 2 if is_gov_source else 1
                    key_evidence.append((evidence, "relevant", evidence.source_title or ""))
        
        avg_credibility = total_credibility / len(evidence_list) if evidence_list else 0.0
        
        return {
            'supporting': supporting_signals,
            'refuting': refuting_signals,
            'gov_sources': gov_source_count,
            'avg_credibility': avg_credibility,
            'key_evidence': key_evidence
        }
    
    def _compare_numbers(
        self, 
        claim_text: str, 
        evidence_text: str, 
        is_gov_source: bool
    ) -> Dict[str, Any]:
        """Compare numbers between claim and evidence.
        
        Args:
            claim_text: Claim text for comparison.
            evidence_text: Evidence text for comparison.
            is_gov_source: Whether evidence is from government source.
            
        Returns:
            Dictionary with supporting/refuting signal counts.
        """
        result = {'supporting': 0, 'refuting': 0, 'evidence': []}
        
        claim_numbers = re.findall(r'\d+', claim_text)
        evidence_numbers = re.findall(r'\d+', evidence_text)
        weight = 2 if is_gov_source else 1
        
        for claim_num in claim_numbers:
            if claim_num not in claim_text:
                continue
                
            for ev_num in evidence_numbers:
                if ev_num == claim_num:
                    continue
                    
                try:
                    diff = abs(int(ev_num) - int(claim_num))
                    if diff <= 5:
                        continue
                    
                    # Check context matching
                    if 'tỉnh' in claim_text and 'tỉnh' in evidence_text:
                        result['refuting'] += weight
                    elif 'km' in claim_text and 'km' in evidence_text:
                        ratio = diff / int(claim_num) if int(claim_num) > 0 else 1
                        if ratio < 0.05:
                            result['supporting'] += weight
                        else:
                            result['refuting'] += weight
                except ValueError:
                    pass
        
        return result
    
    def _determine_verdict_from_signals(
        self, 
        signals: Dict[str, Any]
    ) -> Tuple[str, Dict[str, float]]:
        """Determine verdict label and scores from signals.
        
        Args:
            signals: Analysis signals dictionary.
            
        Returns:
            Tuple of (label, confidence_scores).
        """
        supporting = signals['supporting']
        refuting = signals['refuting']
        
        if refuting > supporting and refuting >= 2:
            label = "refuted"
            confidence_scores = {
                "supported": 0.1,
                "refuted": min(0.75, 0.5 + (refuting - supporting) * 0.1),
                "not_enough_info": 0.15
            }
        elif supporting > refuting and supporting >= 2:
            label = "supported"
            confidence_scores = {
                "supported": min(0.75, 0.5 + (supporting - refuting) * 0.1),
                "refuted": 0.1,
                "not_enough_info": 0.15
            }
        elif refuting > 0 or supporting > 0:
            label = "mixed" if refuting > 0 and supporting > 0 else "not_enough_info"
            confidence_scores = {
                "supported": 0.3 if supporting > 0 else 0.2,
                "refuted": 0.3 if refuting > 0 else 0.2,
                "not_enough_info": 0.4
            }
        else:
            label = "not_enough_info"
            confidence_scores = {
                "supported": 0.25,
                "refuted": 0.25,
                "not_enough_info": 0.5
            }
        
        # Normalize
        total = sum(confidence_scores.values())
        confidence_scores = {k: v / total for k, v in confidence_scores.items()}
        
        return label, confidence_scores
    
    def _build_fallback_explanation(
        self, 
        signals: Dict[str, Any], 
        error_reason: str
    ) -> str:
        """Build explanation for fallback verdict.
        
        Args:
            signals: Analysis signals.
            error_reason: Reason for fallback.
            
        Returns:
            Explanation string.
        """
        parts = [f"Verdict generated using evidence-based fallback (LLM unavailable: {error_reason[:50]}...)"]
        
        if signals['gov_sources'] > 0:
            parts.append(f"Sử dụng {signals['gov_sources']} nguồn chính phủ.")
        
        for ev, ev_type, reason in signals['key_evidence'][:3]:
            source_name = ev.source_title[:50] if ev.source_title else (
                ev.source_url[:50] if ev.source_url else "Unknown"
            )
            parts.append(f"[{ev_type.upper()}] {source_name}: {reason}")
        
        return "\n".join(parts)
    
    def _apply_confidence_cap(
        self, 
        confidence_scores: Dict[str, float], 
        evidence_list: List[Evidence],
        temporal_warnings: List[str],
        current_label: str = None
    ) -> Tuple[Dict[str, float], Optional[str], Optional[str], Optional[str]]:
        """Apply confidence capping and verdict adjustment.
        
        Args:
            confidence_scores: Original confidence scores.
            evidence_list: List of evidence.
            temporal_warnings: List of temporal warnings.
            current_label: Current verdict label.
            
        Returns:
            Tuple of (capped_scores, cap_reason, temporal_notes, adjusted_label).
        """
        capped_scores = confidence_scores.copy()
        cap_reason = None
        temporal_notes = "; ".join(temporal_warnings) if temporal_warnings else None
        adjusted_label = current_label
        
        caps = []
        
        # Evidence text quality check
        caps.extend(self._check_text_quality(evidence_list))
        if caps and caps[0][0] == 0.4:
            adjusted_label = "not_enough_info"
        
        # Credibility cap
        caps.extend(self._check_credibility(evidence_list))
        
        # Temporal cap
        caps.extend(self._check_temporal(evidence_list, temporal_warnings))
        
        # Source diversity cap
        caps.extend(self._check_source_diversity(evidence_list))
        
        # Apply lowest cap
        if caps:
            min_cap, reason = min(caps, key=lambda x: x[0])
            for label, score in capped_scores.items():
                if score > min_cap:
                    capped_scores[label] = min_cap
                    cap_reason = reason
        
        # Logic consistency check
        adjusted_label = self._check_logic_consistency(
            current_label, capped_scores, cap_reason, adjusted_label
        )
        
        # Renormalize
        capped_scores = self._renormalize_scores(capped_scores)
        
        return capped_scores, cap_reason, temporal_notes, adjusted_label
    
    def _check_text_quality(self, evidence_list: List[Evidence]) -> List[Tuple[float, str]]:
        """Check evidence text quality.
        
        Args:
            evidence_list: List of evidence.
            
        Returns:
            List of (cap_value, reason) tuples.
        """
        caps = []
        valid_text_evidence = []
        empty_count = 0
        short_count = 0
        
        for e in evidence_list:
            text_length = len(e.text.strip()) if e.text else 0
            if text_length == 0 or e.text in ["...", "N/A", ""]:
                empty_count += 1
            elif text_length < 100:
                short_count += 1
            else:
                valid_text_evidence.append(e)
        
        total = len(evidence_list)
        valid_ratio = len(valid_text_evidence) / total if total > 0 else 0
        
        if total > 0:
            if valid_ratio == 0:
                logger.warning(f"All {total} evidence pieces have empty/invalid text!")
                caps.append((0.4, f"All evidence has empty/invalid text ({total} pieces)"))
            elif valid_ratio < 0.5:
                logger.warning(f"Low evidence text quality: {len(valid_text_evidence)}/{total} valid")
                caps.append((0.6, f"Low evidence text quality ({len(valid_text_evidence)}/{total} valid)"))
            elif empty_count > 0:
                logger.info(f"Some evidence has empty text: {empty_count}/{total}")
                caps.append((0.8, f"{empty_count} evidence pieces with empty text"))
        
        return caps
    
    def _check_credibility(self, evidence_list: List[Evidence]) -> List[Tuple[float, str]]:
        """Check evidence credibility.
        
        Args:
            evidence_list: List of evidence.
            
        Returns:
            List of (cap_value, reason) tuples.
        """
        caps = []
        valid_cred = [e for e in evidence_list if e.credibility_score and e.credibility_score > 0]
        
        if valid_cred:
            avg = sum(e.credibility_score for e in valid_cred) / len(valid_cred)
            if avg < 0.5:
                caps.append((0.7, f"Low average credibility ({avg:.2f})"))
            elif avg < 0.7:
                caps.append((0.85, f"Medium average credibility ({avg:.2f})"))
        elif evidence_list:
            caps.append((0.75, "Evidence credibility not scored"))
        else:
            caps.append((0.5, "No evidence collected"))
        
        return caps
    
    def _check_temporal(
        self, 
        evidence_list: List[Evidence], 
        temporal_warnings: List[str]
    ) -> List[Tuple[float, str]]:
        """Check temporal context.
        
        Args:
            evidence_list: List of evidence.
            temporal_warnings: List of warnings.
            
        Returns:
            List of (cap_value, reason) tuples.
        """
        caps = []
        
        if not temporal_warnings:
            return caps
        
        # Check for past-effective evidence
        has_past_effective = False
        try:
            from src.legal_temporal_reasoning import LegalTemporalReasoner, LegalStatus
            reasoner = LegalTemporalReasoner()
            for e in evidence_list:
                analysis = reasoner.analyze(e)
                if analysis.is_past_effective or analysis.legal_status in [
                    LegalStatus.EFFECTIVE, LegalStatus.IMPLEMENTED
                ]:
                    has_past_effective = True
                    break
        except ImportError:
            pass
        
        if not has_past_effective:
            caps.append((0.6, "Evidence contains future/pending policies"))
        
        return caps
    
    def _check_source_diversity(self, evidence_list: List[Evidence]) -> List[Tuple[float, str]]:
        """Check source diversity.
        
        Args:
            evidence_list: List of evidence.
            
        Returns:
            List of (cap_value, reason) tuples.
        """
        caps = []
        domains = set()
        
        for e in evidence_list:
            if e.source_url:
                try:
                    domain = urlparse(e.source_url).netloc.lower()
                    domains.add(domain)
                except Exception:
                    pass
        
        if len(domains) < 2:
            caps.append((0.7, "Single source domain"))
        
        return caps
    
    def _check_logic_consistency(
        self, 
        current_label: str, 
        capped_scores: Dict[str, float],
        cap_reason: Optional[str],
        adjusted_label: Optional[str]
    ) -> Optional[str]:
        """Check and adjust for logic consistency.
        
        Args:
            current_label: Current verdict label.
            capped_scores: Capped confidence scores.
            cap_reason: Current cap reason.
            adjusted_label: Already adjusted label.
            
        Returns:
            Final adjusted label.
        """
        max_score = max(capped_scores.values()) if capped_scores else 0.0
        
        if current_label in ["supported", "refuted"] and max_score < 0.7:
            if current_label == "refuted":
                return "mixed"
            elif current_label == "supported":
                return "partially_supported"
        
        return adjusted_label
    
    def _renormalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Renormalize scores to sum to 1.
        
        Args:
            scores: Scores to normalize.
            
        Returns:
            Normalized scores.
        """
        total = sum(scores.values())
        if total < 1.0:
            remainder = 1.0 - total
            if "not_enough_info" in scores:
                scores["not_enough_info"] += remainder
            elif "mixed" in scores:
                scores["mixed"] += remainder
            else:
                for label in scores:
                    scores[label] /= total
        
        return scores
    
    def aggregate_sub_verdicts(self, sub_verdicts: List[Verdict]) -> Verdict:
        """Aggregate multiple sub-claim verdicts into final verdict.
        
        Args:
            sub_verdicts: List of sub-claim verdicts.
            
        Returns:
            Aggregated verdict.
        """
        if not sub_verdicts:
            return Verdict(
                claim_id="aggregated",
                label="not_enough_info",
                confidence_scores={"not_enough_info": 1.0},
                explanation="No sub-verdicts to aggregate"
            )
        
        labels = [v.label for v in sub_verdicts]
        avg_scores = self._average_scores(sub_verdicts)
        final_label = self._determine_aggregate_label(labels)
        
        return Verdict(
            claim_id=sub_verdicts[0].claim_id,
            label=final_label,
            confidence_scores=avg_scores,
            explanation=f"Aggregated from {len(sub_verdicts)} sub-claims: {', '.join(labels)}"
        )
    
    def _average_scores(self, verdicts: List[Verdict]) -> Dict[str, float]:
        """Calculate average scores across verdicts.
        
        Args:
            verdicts: List of verdicts.
            
        Returns:
            Averaged score dictionary.
        """
        all_keys = set()
        for v in verdicts:
            all_keys.update(v.confidence_scores.keys())
        
        avg_scores = {}
        for key in all_keys:
            scores = [v.confidence_scores.get(key, 0.0) for v in verdicts]
            if scores:
                avg_scores[key] = sum(scores) / len(scores)
        
        return avg_scores
    
    def _determine_aggregate_label(self, labels: List[str]) -> str:
        """Determine aggregate label from sub-labels.
        
        Args:
            labels: List of verdict labels.
            
        Returns:
            Aggregate label string.
        """
        if all(l == "supported" for l in labels):
            return "supported"
        elif all(l == "refuted" for l in labels):
            return "refuted"
        elif all(l == "not_enough_info" for l in labels):
            return "not_enough_info"
        elif "refuted" in labels and "supported" in labels:
            return "mixed"
        elif "supported" in labels:
            return "partially_supported"
        elif "refuted" in labels:
            return "mixed"
        else:
            return "context_dependent"
