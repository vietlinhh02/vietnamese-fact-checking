"""Verdict logic validation module for Vietnamese fact-checking system.

This module validates that verdict labels are logically consistent with
the collected evidence and temporal analysis. It acts as a safety guardrail
rather than a rigid rule enforcement system.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from src.data_models import Evidence, Verdict, LegalStatus
    from src.legal_temporal_reasoning import LegalTemporalReasoner, LegalTemporalAnalysis
except ImportError:
    from data_models import Evidence, Verdict, LegalStatus
    from legal_temporal_reasoning import LegalTemporalReasoner, LegalTemporalAnalysis

logger = logging.getLogger(__name__)


@dataclass
class VerdictValidationResult:
    """Result of verdict validation."""
    is_consistent: bool
    explanation: str
    suggested_label: Optional[str] = None
    confidence_adjustment: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class VerdictLogicValidator:
    """Validates verdicts, trusting the AI's reasoning unless clearly contradictory."""
    
    # Relaxed thresholds
    STRONG_SUPPORT_THRESHOLD = 0.6  # Lowered from 0.7
    HIGH_CREDIBILITY_THRESHOLD = 0.6 # Lowered from 0.7
    
    def __init__(self):
        self.temporal_reasoner = LegalTemporalReasoner()
        logger.info("Initialized VerdictLogicValidator (Relaxed Mode)")
    
    def validate_verdict(
        self,
        verdict_label: str,
        confidence_scores: Dict[str, float],
        evidence_list: List[Evidence],
        claim_text: str = ""
    ) -> VerdictValidationResult:
        """Validate verdict consistency."""
        warnings = []
        
        # 1. Sanity check: Evidence existence
        if not evidence_list and verdict_label not in ["not_enough_info"]:
            return VerdictValidationResult(
                is_consistent=False,
                explanation="No evidence collected but verdict is decisive",
                suggested_label="not_enough_info",
                warnings=["Auto-corrected to NEI due to zero evidence"]
            )

        # 2. Check for egregious mismatches (e.g. Refuting evidence but Supported verdict)
        # We trust the AI more now, so we look for 'impossible' states rather than 'weak' ones.
        
        refuted_score = confidence_scores.get("refuted", 0.0)
        supported_score = confidence_scores.get("supported", 0.0)
        
        # If AI says Supported (high confidence) but Refuted score is somehow massive (unlikely if Softmax is used correctly, but good to check)
        if verdict_label == "supported" and refuted_score > 0.8:
             warnings.append("Verdict is Supported but Refuted score is very high (0.8+). Review reasoning.")
             # We warn, but maybe don't force overturn unless it's 100% contradictory
        
        # 3. Government Refutation Override (Safety mechanism)
        # If a government source explicitly REFUTES it (e.g. passed law), we should probably respect that.
        # But we'll make it a strong suggestion rather than a force if the AI has a reason.
        
        has_gov_refutation = self._has_government_refutation_evidence(evidence_list)
        if has_gov_refutation and verdict_label == "supported":
            return VerdictValidationResult(
                is_consistent=False,
                explanation="Government source directly contradicts this claim (e.g. passed law/resolution).",
                suggested_label="refuted",
                warnings=["Auto-corrected: Supported -> Refuted based on Government Evidence"]
            )
            
        # 4. Temporal Consistency
        # Handled inside the AI prompt mostly, but basic check:
        # If all evidence is future-dated policy, we shouldn't refute a current claim purely on that.
        # (This logic is complex to validate via rules, trusting AI reasoning here).

        # If we got here, we mostly trust the AI.
        return VerdictValidationResult(
            is_consistent=True, 
            explanation="Verdict consistent with relaxed validation rules", 
            warnings=warnings
        )
    
    def _has_government_refutation_evidence(self, evidence_list: List[Evidence]) -> bool:
        """Check if any evidence is from government source with passed/effective resolution."""
        # Simplified check
        gov_patterns = ['.gov.vn', 'chinhphu.vn', 'quochoi.vn']
        passed_patterns = ['đã thông qua', 'đã biểu quyết', 'có hiệu lực', 'chính thức']
        
        for evidence in evidence_list:
            if evidence.source_url and any(p in evidence.source_url.lower() for p in gov_patterns):
                if evidence.text and any(p in evidence.text.lower() for p in passed_patterns):
                    return True
        return False
    
    def _get_avg_credibility(self, evidence_list: List[Evidence]) -> float:
        valid_scores = [e.credibility_score for e in evidence_list if e.credibility_score > 0]
        if not valid_scores:
            return 0.5
        return sum(valid_scores) / len(valid_scores)
    
    def auto_correct_verdict(
        self,
        verdict: Verdict,
        evidence_list: List[Evidence],
        claim_text: str = ""
    ) -> Tuple[Verdict, bool, str]:
        """Auto-correct verdict if validation fails."""
        validation = self.validate_verdict(
            verdict_label=verdict.label,
            confidence_scores=verdict.confidence_scores,
            evidence_list=evidence_list,
            claim_text=claim_text
        )
        
        if validation.is_consistent:
            return verdict, False, ""
        
        old_label = verdict.label
        verdict.label = validation.suggested_label
        verdict.confidence_cap_reason = f"[Auto-corrected: {old_label} -> {verdict.label}] {validation.explanation}"
        
        logger.info(f"Verdict auto-corrected: {old_label} -> {verdict.label}")
        return verdict, True, validation.explanation
