"""Legal and temporal reasoning module for Vietnamese fact-checking.

This module handles the distinction between:
- PASSED: Document approved but not yet effective
- EFFECTIVE: Document currently in effect
- IMPLEMENTED: Policy fully implemented
- PENDING: Future policy/plan

Critical for avoiding incorrect refutation of current facts based on future policies.
"""

import re
import logging
from typing import Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass

try:
    from src.data_models import Evidence, LegalStatus
except ImportError:
    from data_models import Evidence, LegalStatus

logger = logging.getLogger(__name__)


@dataclass
class LegalTemporalAnalysis:
    """Result of legal/temporal analysis."""
    legal_status: LegalStatus
    effective_date: Optional[datetime] = None
    confidence_weight: float = 1.0  # Weight for verdict confidence
    reasoning: str = ""
    can_refute_current: bool = True  # Whether this evidence can refute current state
    is_past_effective: bool = False  # True if effective_date is in the past (policy already in effect)


class LegalTemporalReasoner:
    """Handles legal status and temporal reasoning for Vietnamese fact-checking.
    
    Key principle: A passed law/resolution is NOT the same as an effective one.
    Future policies should NOT be used to refute claims about current state.
    """
    
    # Vietnamese patterns for legal status detection
    # Expanded to cover diverse journalistic and legal phrasing
    
    PASSED_PATTERNS = [
        # Direct approval verbs
        r'(?:đã\s+)?(?:được\s+)?(?:thông qua|phê duyệt|phê chuẩn|ban hành|tán thành|biểu quyết)',
        r'(?:đã\s+)?(?:nhất trí|chấp thuận|đồng ý)\s+(?:thông qua|phương án)',
        # Subject + Action
        r'(?:quốc hội|chính phủ|ủy ban|hđnd).*?(?:bấm nút|chốt|quyết định)',
        # Passive voice
        r'được\s+(?:đa số|tuyệt đại đa số).*?(?:tán thành|ủng hộ)',
    ]
    
    EFFECTIVE_PATTERNS = [
        # Explicit effective date
        r'(?:có|mang|bắt đầu|chính thức).*?(?:hiệu lực|giá trị)(?:\s+(?:thi hành|áp dụng))?(?:\s+(?:từ|kể từ|vào))?\s+(?:ngày\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(?:áp dụng|thực hiện)(?:\s+(?:từ|kể từ))?\s+(?:ngày\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        # Current status markers
        r'(?:đang|hiện)\s+(?:có|là).*?(?:hiệu lực|giá trị)',
        r'đã\s+(?:có|mang).*?(?:hiệu lực|tác dụng)',
        r'(?:vừa|mới).*?(?:đi vào|bắt đầu).*?(?:thực tiễn|cuộc sống)',
        r'đang\s+(?:được|bị).*?(?:chi phối|điều chỉnh)\s+bởi',
    ]
    
    PENDING_PATTERNS = [
        # Future markers
        r'(?:sẽ|dự kiến|kế hoạch|lộ trình|sắp).*?(?:có|mang).*?(?:hiệu lực|giá trị)',
        r'(?:chờ|đợi).*?(?:ban hành|hướng dẫn|nghị định|thông tư)',
        r'(?:chưa).*?(?:có|mang).*?(?:hiệu lực|giá trị)',
        r'(?:thời gian|giai đoạn|điều khoản).*?(?:chuyển tiếp|đệm)',
        r'hiệu lực(?:\s+(?:từ|kể từ))?\s+(?:ngày\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4}).*?(?:trở đi|về sau)',
        r'(?:từ|sau)\s+năm\s+(\d{4})',
    ]
    
    IMPLEMENTED_PATTERNS = [
        # Completion markers
        r'(?:đã|vừa).*?(?:hoàn thành|hoàn tất|xong).*?(?:việc|quá trình)?\s*(?:triển khai|sáp nhập|thực hiện)',
        r'(?:đã|chính thức).*?(?:đi vào|đưa vào).*?(?:hoạt động|vận hành|sử dụng)',
        r'(?:kết thúc|khép lại).*?(?:giai đoạn|quá trình).*?(?:thực hiện|triển khai)',
        r'(?:hiện nay|bây giờ).*?(?:đã|là)',
    ]
    
    # Date patterns for extraction
    DATE_PATTERNS = [
        # DD/MM/YYYY or DD-MM-YYYY
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        # ngày DD tháng MM năm YYYY
        r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
        # tháng MM/YYYY or tháng MM năm YYYY
        r'tháng\s+(\d{1,2})[/-](\d{4})',
        r'tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
        # năm YYYY
        r'năm\s+(\d{4})',
    ]
    
    def __init__(self, current_date: Optional[datetime] = None):
        """Initialize reasoner.
        
        Args:
            current_date: Override current date for testing
        """
        self.current_date = current_date or datetime.now()
    
    def analyze(self, evidence: Evidence) -> LegalTemporalAnalysis:
        """Analyze evidence for legal status and temporal context.
        
        Args:
            evidence: Evidence to analyze
            
        Returns:
            LegalTemporalAnalysis with status, dates, and confidence weight
        """
        text = evidence.text.lower()
        
        # Try to detect legal status from patterns
        status, reasoning = self._classify_legal_status(text)
        
        # Try to extract effective date
        effective_date = self._detect_effective_date(evidence.text)
        
        # Determine if this is currently effective
        is_past_effective = False
        if effective_date:
            if effective_date > self.current_date:
                # Future effective date - this is PENDING
                status = LegalStatus.PENDING
                reasoning = f"Có hiệu lực từ {effective_date.strftime('%d/%m/%Y')} (tương lai)"
            else:
                # Effective date is today or past - this is EFFECTIVE
                status = LegalStatus.EFFECTIVE
                reasoning = f"Có hiệu lực từ {effective_date.strftime('%d/%m/%Y')} (đã có hiệu lực)"
                is_past_effective = True  # Critical: this policy is NOW in effect
        
        # Calculate confidence weight based on status
        confidence_weight, can_refute = self._get_confidence_weight(status)
        
        return LegalTemporalAnalysis(
            legal_status=status,
            effective_date=effective_date,
            confidence_weight=confidence_weight,
            reasoning=reasoning,
            can_refute_current=can_refute,
            is_past_effective=is_past_effective
        )
    
    def _classify_legal_status(self, text: str) -> Tuple[LegalStatus, str]:
        """Classify legal document status from text.
        
        Args:
            text: Lowercase evidence text
            
        Returns:
            Tuple of (LegalStatus, reasoning)
        """
        # Check IMPLEMENTED first (strongest signal)
        for pattern in self.IMPLEMENTED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return LegalStatus.IMPLEMENTED, "Phát hiện dấu hiệu đã triển khai hoàn toàn"
        
        # Check PENDING (future policy)
        for pattern in self.PENDING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return LegalStatus.PENDING, "Phát hiện dấu hiệu chính sách tương lai/dự kiến"
        
        # Check EFFECTIVE
        for pattern in self.EFFECTIVE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return LegalStatus.EFFECTIVE, "Phát hiện dấu hiệu đang có hiệu lực"
        
        # Check PASSED
        for pattern in self.PASSED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return LegalStatus.PASSED, "Phát hiện dấu hiệu đã thông qua (nhưng cần xác nhận hiệu lực)"
        
        return LegalStatus.UNKNOWN, "Không xác định được trạng thái pháp lý"
    
    def _detect_effective_date(self, text: str) -> Optional[datetime]:
        """Extract effective date from legal document.
        
        Args:
            text: Evidence text
            
        Returns:
            Datetime if found, None otherwise
        """
        # Look for "có hiệu lực từ" patterns first (most reliable)
        effective_patterns = [
            r'có hiệu lực(?:\s+(?:từ|kể từ))?\s+(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'có hiệu lực(?:\s+(?:từ|kể từ))?\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
            r'áp dụng(?:\s+(?:từ|kể từ))?\s+(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'(?:kể|bắt đầu)\s+từ(?:\s+ngày)?\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        ]
        
        for pattern in effective_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    day = int(match.group(1))
                    month = int(match.group(2))
                    year = int(match.group(3))
                    return datetime(year, month, day)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _get_confidence_weight(self, status: LegalStatus) -> Tuple[float, bool]:
        """Get confidence weight based on legal status.
        
        Args:
            status: Legal status
            
        Returns:
            Tuple of (confidence_weight, can_refute_current)
            
        Weights:
        - EFFECTIVE/IMPLEMENTED: Full weight, can refute current claims
        - PASSED: Moderate weight, cannot refute current claims alone
        - PENDING: Low weight, cannot refute current claims
        - UNKNOWN: Default weight, requires corroboration
        """
        weights = {
            LegalStatus.IMPLEMENTED: (1.0, True),
            LegalStatus.EFFECTIVE: (1.0, True),
            LegalStatus.PASSED: (0.6, False),  # CRITICAL: Cannot refute current state alone
            LegalStatus.PENDING: (0.3, False),  # CRITICAL: Future policy, very limited use
            LegalStatus.UNKNOWN: (0.7, True),
        }
        return weights.get(status, (0.7, True))
    
    def is_currently_effective(self, evidence: Evidence) -> bool:
        """Determine if evidence refers to currently effective state.
        
        Args:
            evidence: Evidence to check
            
        Returns:
            True if evidence is about current/effective state
        """
        analysis = self.analyze(evidence)
        return analysis.legal_status in [LegalStatus.EFFECTIVE, LegalStatus.IMPLEMENTED]
    
    def can_use_to_refute(self, evidence: Evidence) -> Tuple[bool, str]:
        """Determine if evidence can be used to refute current claims.
        
        Args:
            evidence: Evidence to check
            
        Returns:
            Tuple of (can_refute, reason)
        """
        analysis = self.analyze(evidence)
        
        if not analysis.can_refute_current:
            reasons = {
                LegalStatus.PASSED: "Văn bản đã thông qua nhưng chưa xác nhận có hiệu lực",
                LegalStatus.PENDING: "Chính sách tương lai, không thể bác bỏ thực tại hiện tại",
            }
            reason = reasons.get(
                analysis.legal_status, 
                f"Trạng thái pháp lý: {analysis.legal_status.value}"
            )
            return False, reason
        
        return True, "Có thể sử dụng làm bằng chứng bác bỏ"


def apply_temporal_confidence_weight(
    evidence_list: List[Evidence],
    reasoner: Optional[LegalTemporalReasoner] = None
) -> Tuple[float, List[str]]:
    """Apply temporal confidence weighting to evidence list.
    
    Args:
        evidence_list: List of evidence pieces
        reasoner: LegalTemporalReasoner instance (created if not provided)
        
    Returns:
        Tuple of (weighted_confidence, list of warnings)
    """
    if not evidence_list:
        return 0.0, []
    
    if reasoner is None:
        reasoner = LegalTemporalReasoner()
    
    warnings = []
    total_weight = 0.0
    weighted_credibility = 0.0
    
    for evidence in evidence_list:
        analysis = reasoner.analyze(evidence)
        
        # Apply temporal weight
        weight = analysis.confidence_weight
        total_weight += weight
        weighted_credibility += evidence.credibility_score * weight
        
        # Log warnings for non-effective evidence
        if analysis.legal_status == LegalStatus.PENDING:
            warnings.append(
                f"⚠️ FUTURE POLICY: '{evidence.source_title[:50]}...' - {analysis.reasoning}"
            )
        elif analysis.legal_status == LegalStatus.PASSED:
            warnings.append(
                f"⚠️ PASSED BUT NOT EFFECTIVE: '{evidence.source_title[:50]}...' - "
                f"Cần xác nhận ngày có hiệu lực"
            )
    
    if total_weight > 0:
        avg_weighted_credibility = weighted_credibility / total_weight
    else:
        avg_weighted_credibility = 0.0
    
    return avg_weighted_credibility, warnings
