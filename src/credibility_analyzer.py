"""Source credibility analysis module for Vietnamese fact-checking system."""

import logging
import re
import time
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse
from datetime import datetime

from src.web_crawler import WebContent

logger = logging.getLogger(__name__)


@dataclass
class CredibilityFeatures:
    """Features used for credibility scoring."""
    
    # Domain features
    domain: str
    tld: str  # .vn, .com, .org, etc.
    domain_age_days: Optional[int] = None
    uses_https: bool = False
    
    # Content features
    has_author: bool = False
    has_publish_date: bool = False
    article_length: int = 0
    writing_quality_score: float = 0.0
    
    # External signals
    is_state_managed: bool = False
    mbfc_rating: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CredibilityFeatures":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CredibilityScore:
    """Credibility score for a source."""
    
    overall_score: float  # [0, 1]
    feature_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    explanation: str = ""
    
    def __post_init__(self):
        """Validate credibility score."""
        if not 0 <= self.overall_score <= 1:
            raise ValueError(f"Overall score must be in [0, 1], got {self.overall_score}")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CredibilityScore":
        """Create from dictionary."""
        return cls(**data)


class CredibilityFeatureExtractor:
    """Extract credibility features from web content."""
    
    # Vietnamese state-managed news sources
    STATE_MANAGED_SOURCES = [
        "vnexpress.net",
        "vtv.vn",
        "vov.vn",
        "tuoitre.vn",
        "thanhnien.vn",
        "baochinhphu.vn",
        "vietnamplus.vn",
        "nhandan.vn",
        "vietnamnet.vn",
        "dantri.com.vn"
    ]
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def _extract_domain_features(self, content: WebContent) -> Dict[str, Any]:
        """
        Extract domain-related features.
        
        Args:
            content: WebContent object
            
        Returns:
            Dictionary of domain features
        """
        parsed_url = urlparse(content.url)
        
        # Extract domain
        domain = parsed_url.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Extract TLD
        tld = domain.split('.')[-1] if '.' in domain else ""
        
        # Check HTTPS
        uses_https = parsed_url.scheme == 'https'
        
        # Check if state-managed
        is_state_managed = any(source in domain for source in self.STATE_MANAGED_SOURCES)
        
        return {
            'domain': domain,
            'tld': tld,
            'uses_https': uses_https,
            'is_state_managed': is_state_managed
        }
    
    def _extract_content_features(self, content: WebContent) -> Dict[str, Any]:
        """
        Extract content-related features.
        
        Args:
            content: WebContent object
            
        Returns:
            Dictionary of content features
        """
        # Check for author
        has_author = content.author is not None and len(content.author.strip()) > 0
        
        # Check for publish date
        has_publish_date = content.publish_date is not None and len(content.publish_date.strip()) > 0
        
        # Article length
        article_length = len(content.main_text)
        
        # Writing quality score (basic heuristic)
        writing_quality_score = self._compute_writing_quality(content.main_text)
        
        return {
            'has_author': has_author,
            'has_publish_date': has_publish_date,
            'article_length': article_length,
            'writing_quality_score': writing_quality_score
        }
    
    def _compute_writing_quality(self, text: str) -> float:
        """
        Compute basic writing quality score.
        
        Args:
            text: Article text
            
        Returns:
            Quality score [0, 1]
        """
        if not text or len(text) < 50:
            return 0.0
        
        score = 0.0
        
        # Check for proper capitalization (sentences start with capital letters)
        sentences = re.split(r'[.!?]+', text)
        capitalized_sentences = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if len(sentences) > 0:
            capitalization_ratio = capitalized_sentences / len(sentences)
            score += 0.3 * capitalization_ratio
        
        # Check for proper punctuation (sentences end with punctuation)
        punctuation_count = len(re.findall(r'[.!?]', text))
        words = text.split()
        if len(words) > 0:
            punctuation_ratio = min(punctuation_count / (len(words) / 15), 1.0)  # ~15 words per sentence
            score += 0.3 * punctuation_ratio
        
        # Check for reasonable sentence length (not too short, not too long)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Optimal range: 10-30 words per sentence
            if 10 <= avg_sentence_length <= 30:
                score += 0.2
            elif 5 <= avg_sentence_length < 10 or 30 < avg_sentence_length <= 50:
                score += 0.1
        
        # Check for paragraph structure (presence of newlines)
        paragraphs = text.split('\n')
        if len(paragraphs) > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def extract_features(self, content: WebContent) -> CredibilityFeatures:
        """
        Extract all credibility features from web content.
        
        Args:
            content: WebContent object
            
        Returns:
            CredibilityFeatures object
        """
        # Extract domain features
        domain_features = self._extract_domain_features(content)
        
        # Extract content features
        content_features = self._extract_content_features(content)
        
        # Combine features
        features = CredibilityFeatures(
            domain=domain_features['domain'],
            tld=domain_features['tld'],
            uses_https=domain_features['uses_https'],
            is_state_managed=domain_features['is_state_managed'],
            has_author=content_features['has_author'],
            has_publish_date=content_features['has_publish_date'],
            article_length=content_features['article_length'],
            writing_quality_score=content_features['writing_quality_score']
        )
        
        logger.debug(f"Extracted features for {content.url}: {features}")
        
        return features


class MediaBiasFactCheckAPI:
    """Client for Media Bias Fact Check API (optional integration)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize MBFC API client.
        
        Args:
            api_key: API key for MBFC (if available)
        """
        self.api_key = api_key
        self.base_url = "https://api.mediabiasfactcheck.com/v1"
        self.session = requests.Session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Cache for API responses
        self.cache: Dict[str, Optional[str]] = {}
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def get_rating(self, domain: str) -> Optional[str]:
        """
        Get MBFC rating for a domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            Rating string (e.g., "High", "Medium", "Low") or None if unavailable
        """
        # Check cache
        if domain in self.cache:
            return self.cache[domain]
        
        # If no API key, return None
        if not self.api_key:
            logger.debug("MBFC API key not configured, skipping external rating")
            self.cache[domain] = None
            return None
        
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Make API request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json'
            }
            
            response = self.session.get(
                f"{self.base_url}/sources/{domain}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                rating = data.get('factual_reporting', None)
                self.cache[domain] = rating
                logger.info(f"MBFC rating for {domain}: {rating}")
                return rating
            elif response.status_code == 404:
                # Domain not found in MBFC database
                logger.debug(f"Domain {domain} not found in MBFC database")
                self.cache[domain] = None
                return None
            else:
                logger.warning(f"MBFC API returned status {response.status_code} for {domain}")
                self.cache[domain] = None
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MBFC rating for {domain}: {e}")
            self.cache[domain] = None
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching MBFC rating for {domain}: {e}")
            self.cache[domain] = None
            return None


class RuleBasedCredibilityScorer:
    """Rule-based credibility scoring system."""
    
    # Scoring weights (without MBFC)
    WEIGHTS = {
        'state_managed': 0.4,
        'https': 0.1,
        'author_date': 0.1,
        'writing_quality': 0.2,
        'article_length': 0.1,
        'vn_tld': 0.1
    }
    
    # Scoring weights (with MBFC)
    WEIGHTS_WITH_MBFC = {
        'state_managed': 0.3,
        'https': 0.05,
        'author_date': 0.05,
        'writing_quality': 0.1,
        'article_length': 0.05,
        'vn_tld': 0.05,
        'mbfc': 0.4
    }
    
    # MBFC rating to score mapping
    MBFC_SCORES = {
        'Very High': 1.0,
        'High': 0.8,
        'Mostly Factual': 0.6,
        'Mixed': 0.4,
        'Low': 0.2,
        'Very Low': 0.0
    }
    
    def __init__(self, mbfc_api: Optional[MediaBiasFactCheckAPI] = None):
        """
        Initialize the scorer.
        
        Args:
            mbfc_api: Optional MBFC API client
        """
        self.mbfc_api = mbfc_api
        self.weights = self.WEIGHTS_WITH_MBFC if mbfc_api else self.WEIGHTS
    
    def _score_state_managed(self, features: CredibilityFeatures) -> float:
        """Score based on state-managed status."""
        return self.weights['state_managed'] if features.is_state_managed else 0.0
    
    def _score_https(self, features: CredibilityFeatures) -> float:
        """Score based on HTTPS usage."""
        return self.weights['https'] if features.uses_https else 0.0
    
    def _score_author_date(self, features: CredibilityFeatures) -> float:
        """Score based on presence of author and date."""
        if features.has_author and features.has_publish_date:
            return self.weights['author_date']
        elif features.has_author or features.has_publish_date:
            return self.weights['author_date'] * 0.5
        else:
            return 0.0
    
    def _score_writing_quality(self, features: CredibilityFeatures) -> float:
        """Score based on writing quality."""
        return self.weights['writing_quality'] * features.writing_quality_score
    
    def _score_article_length(self, features: CredibilityFeatures) -> float:
        """Score based on article length."""
        # Optimal length: 500-5000 characters
        if 500 <= features.article_length <= 5000:
            return self.weights['article_length']
        elif 200 <= features.article_length < 500 or 5000 < features.article_length <= 10000:
            return self.weights['article_length'] * 0.5
        else:
            return 0.0
    
    def _score_vn_tld(self, features: CredibilityFeatures) -> float:
        """Score based on .vn TLD."""
        return self.weights['vn_tld'] if features.tld == 'vn' else 0.0
    
    def _score_mbfc(self, features: CredibilityFeatures) -> float:
        """Score based on MBFC rating."""
        if not self.mbfc_api or not features.mbfc_rating:
            return 0.0
        
        # Get score from rating
        rating_score = self.MBFC_SCORES.get(features.mbfc_rating, 0.5)
        return self.weights['mbfc'] * rating_score
    
    def _generate_explanation(self, features: CredibilityFeatures, feature_scores: Dict[str, float]) -> str:
        """
        Generate explanation for credibility score.
        
        Args:
            features: CredibilityFeatures object
            feature_scores: Dictionary of individual feature scores
            
        Returns:
            Explanation string
        """
        explanations = []
        
        if features.is_state_managed:
            explanations.append(f"State-managed Vietnamese source (+{feature_scores['state_managed']:.2f})")
        
        if features.uses_https:
            explanations.append(f"Uses HTTPS (+{feature_scores['https']:.2f})")
        
        if features.has_author and features.has_publish_date:
            explanations.append(f"Has author and publish date (+{feature_scores['author_date']:.2f})")
        elif features.has_author:
            explanations.append(f"Has author (+{feature_scores['author_date']:.2f})")
        elif features.has_publish_date:
            explanations.append(f"Has publish date (+{feature_scores['author_date']:.2f})")
        
        if features.writing_quality_score > 0:
            explanations.append(f"Writing quality: {features.writing_quality_score:.2f} (+{feature_scores['writing_quality']:.2f})")
        
        if feature_scores['article_length'] > 0:
            explanations.append(f"Appropriate article length ({features.article_length} chars) (+{feature_scores['article_length']:.2f})")
        
        if features.tld == 'vn':
            explanations.append(f"Vietnamese TLD (.vn) (+{feature_scores['vn_tld']:.2f})")
        
        if features.mbfc_rating and 'mbfc' in feature_scores:
            explanations.append(f"MBFC rating: {features.mbfc_rating} (+{feature_scores['mbfc']:.2f})")
        
        return "; ".join(explanations)
    
    def compute_score(self, features: CredibilityFeatures) -> CredibilityScore:
        """
        Compute credibility score from features.
        
        Args:
            features: CredibilityFeatures object
            
        Returns:
            CredibilityScore object
        """
        # Compute individual feature scores
        feature_scores = {
            'state_managed': self._score_state_managed(features),
            'https': self._score_https(features),
            'author_date': self._score_author_date(features),
            'writing_quality': self._score_writing_quality(features),
            'article_length': self._score_article_length(features),
            'vn_tld': self._score_vn_tld(features)
        }
        
        # Add MBFC score if available
        if self.mbfc_api:
            feature_scores['mbfc'] = self._score_mbfc(features)
        
        # Compute overall score
        overall_score = sum(feature_scores.values())
        
        # Ensure score is in [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Generate explanation
        explanation = self._generate_explanation(features, feature_scores)
        
        score = CredibilityScore(
            overall_score=overall_score,
            feature_scores=feature_scores,
            confidence=1.0,
            explanation=explanation
        )
        
        logger.info(f"Computed credibility score: {overall_score:.2f} for domain {features.domain}")
        
        return score


class CredibilityAnalyzer:
    """Main credibility analyzer combining feature extraction and scoring."""
    
    def __init__(self, mbfc_api_key: Optional[str] = None):
        """
        Initialize the credibility analyzer.
        
        Args:
            mbfc_api_key: Optional API key for Media Bias Fact Check
        """
        self.feature_extractor = CredibilityFeatureExtractor()
        
        # Initialize MBFC API if key provided
        self.mbfc_api = None
        if mbfc_api_key:
            try:
                self.mbfc_api = MediaBiasFactCheckAPI(mbfc_api_key)
                logger.info("MBFC API integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize MBFC API: {e}")
        
        self.scorer = RuleBasedCredibilityScorer(self.mbfc_api)
    
    def analyze_credibility(self, content: WebContent) -> CredibilityScore:
        """
        Analyze credibility of web content.
        
        Args:
            content: WebContent object
            
        Returns:
            CredibilityScore object
        """
        # Extract features
        features = self.feature_extractor.extract_features(content)
        
        # Get MBFC rating if API available
        if self.mbfc_api:
            try:
                mbfc_rating = self.mbfc_api.get_rating(features.domain)
                features.mbfc_rating = mbfc_rating
            except Exception as e:
                logger.warning(f"Failed to get MBFC rating for {features.domain}: {e}")
        
        # Compute score
        score = self.scorer.compute_score(features)
        
        return score


# Convenience function
def analyze_credibility(content: WebContent, mbfc_api_key: Optional[str] = None) -> CredibilityScore:
    """
    Analyze credibility of web content.
    
    Args:
        content: WebContent object
        mbfc_api_key: Optional API key for Media Bias Fact Check
        
    Returns:
        CredibilityScore object
    """
    analyzer = CredibilityAnalyzer(mbfc_api_key=mbfc_api_key)
    return analyzer.analyze_credibility(content)
