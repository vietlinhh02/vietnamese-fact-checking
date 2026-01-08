"""Source credibility analysis module with dynamic scoring."""

import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse
from src.web_crawler import WebContent

logger = logging.getLogger(__name__)

@dataclass
class CredibilityFeatures:
    """Features used for credibility scoring."""
    domain: str
    tld: str
    uses_https: bool = False
    has_author: bool = False
    has_publish_date: bool = False
    article_length: int = 0
    mbfc_rating: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CredibilityScore:
    """Credibility score for a source."""
    overall_score: float  # [0, 1]
    feature_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CredibilityFeatureExtractor:
    """Extract credibility features from web content."""
    
    def extract_features(self, content: WebContent) -> CredibilityFeatures:
        parsed_url = urlparse(content.url)
        domain = parsed_url.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
            
        tld = domain.split('.')[-1] if '.' in domain else ""
        
        return CredibilityFeatures(
            domain=domain,
            tld=tld,
            uses_https=parsed_url.scheme == 'https',
            has_author=bool(content.author),
            has_publish_date=bool(content.publish_date),
            article_length=len(content.main_text) if content.main_text else 0
        )

class RuleBasedCredibilityScorer:
    """Dynamic rule-based credibility scoring system."""
    
    def compute_score(self, features: CredibilityFeatures) -> CredibilityScore:
        score = 0.5  # Base score for unknown sources
        scores = {}
        explanations = []

        # 1. Domain Reputation (TLD & Government check)
        # Strong boost for government/education
        if features.domain.endswith('.gov.vn') or 'chinhphu.vn' in features.domain:
            score = 0.95
            scores['domain_auth'] = 0.95
            explanations.append("Official Government Source")
        elif features.domain.endswith('.edu.vn'):
            score = 0.90
            scores['domain_auth'] = 0.90
            explanations.append("Educational Institution")
        elif features.domain.endswith('.org.vn') or features.domain.endswith('.org'):
            score = 0.75
            scores['domain_auth'] = 0.75
            explanations.append("Organization Domain")
        elif features.domain.endswith('.vn'):
            # National TLD usually implies some registration oversight
            score = max(score, 0.65)
            scores['domain_auth'] = 0.65
            explanations.append("National Domain (.vn)")
        else:
            scores['domain_auth'] = 0.5

        # 2. Content Quality Signals (Bonus only)
        bonus = 0.0
        if features.has_author:
            bonus += 0.05
            explanations.append("Author listed")
        if features.has_publish_date:
            bonus += 0.05
            explanations.append("Date listed")
        if features.article_length > 500:
            bonus += 0.05
            explanations.append("Substantial content")

        # Apply bonus but cap at 1.0
        final_score = min(1.0, score + bonus)
        
        # 3. HTTPS Check (Penalty if missing)
        if not features.uses_https:
            final_score *= 0.8
            explanations.append("No HTTPS (penalty)")

        return CredibilityScore(
            overall_score=final_score,
            feature_scores=scores,
            explanation="; ".join(explanations)
        )

class CredibilityAnalyzer:
    """Main credibility analyzer."""
    
    def __init__(self, mbfc_api_key: Optional[str] = None, web_crawler: Optional[Any] = None):
        self.feature_extractor = CredibilityFeatureExtractor()
        self.scorer = RuleBasedCredibilityScorer()
        self.web_crawler = web_crawler

    def analyze_credibility(self, content: WebContent) -> CredibilityScore:
        features = self.feature_extractor.extract_features(content)
        return self.scorer.compute_score(features)

# Convenience function
def analyze_credibility(content: WebContent, mbfc_api_key: Optional[str] = None) -> CredibilityScore:
    analyzer = CredibilityAnalyzer(mbfc_api_key)
    return analyzer.analyze_credibility(content)
