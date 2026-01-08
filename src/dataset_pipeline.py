"""Dataset construction pipeline for Vietnamese fact-checking."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.claim_detector import ClaimDetector
from src.credibility_analyzer import CredibilityAnalyzer
from src.data_models import Claim, Evidence, ReasoningStep, Verdict
from src.exa_search_client import ExaSearchClient
from src.react_agent import ReActAgent
from src.web_crawler import WebCrawler, WebContent

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class CollectedClaim:
    """Claim extracted from a source with metadata."""

    claim: Claim
    source_url: str
    source_title: str = ""
    source_author: Optional[str] = None
    publish_date: Optional[str] = None
    collected_at: str = field(default_factory=_utc_now_iso)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert claim source metadata to dict."""
        return {
            "claim_source": {
                "url": self.source_url,
                "title": self.source_title,
                "author": self.source_author,
                "publish_date": self.publish_date
            },
            "collected_at": self.collected_at
        }


@dataclass
class DatasetRecord:
    """Single dataset record with claim, evidence, label, and metadata."""

    claim: Claim
    evidence: List[Evidence]
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)

    def has_credible_evidence(self, min_score: float = 0.5) -> bool:
        """Check if at least one evidence item meets credibility threshold."""
        return any(ev.credibility_score >= min_score for ev in self.evidence)

    def _ensure_metadata(self) -> None:
        """Ensure required metadata fields exist."""
        if "source_urls" not in self.metadata:
            self.metadata["source_urls"] = [ev.source_url for ev in self.evidence]
        if "collected_at" not in self.metadata:
            self.metadata["collected_at"] = _utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to serializable dict."""
        self._ensure_metadata()
        return {
            "claim": self.claim.to_dict(),
            "evidence": [ev.to_dict() for ev in self.evidence],
            "label": self.label,
            "metadata": self.metadata,
            "reasoning_trace": [step.to_dict() for step in self.reasoning_trace]
        }


class ClaimCollector:
    """Collect claims from Vietnamese news sources."""

    def __init__(
        self,
        web_crawler: WebCrawler,
        claim_detector: ClaimDetector,
        search_client: Optional[ExaSearchClient] = None,
        approved_domains: Optional[List[str]] = None
    ):
        self.web_crawler = web_crawler
        self.claim_detector = claim_detector
        self.search_client = search_client
        self.approved_domains = approved_domains or []

    def collect_from_urls(
        self,
        urls: Iterable[str],
        use_sliding_window: bool = False
    ) -> List[CollectedClaim]:
        """Crawl URLs and extract claims with metadata."""
        collected: List[CollectedClaim] = []
        seen = set()

        for url in urls:
            content = self.web_crawler.crawl(url)
            if not content or not content.main_text:
                continue

            claims = (
                self.claim_detector.detect_claims_sliding_window(content.main_text)
                if use_sliding_window
                else self.claim_detector.detect_claims(content.main_text)
            )

            for claim in claims:
                key = (claim.text, content.url)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(
                    CollectedClaim(
                        claim=claim,
                        source_url=content.url,
                        source_title=content.title,
                        source_author=content.author,
                        publish_date=content.publish_date
                    )
                )

        logger.info(f"Collected {len(collected)} claims from URLs")
        return collected

    def collect_from_search(
        self,
        queries: Iterable[str],
        max_results: int = 5,
        use_sliding_window: bool = False
    ) -> List[CollectedClaim]:
        """Search for articles and collect claims from results."""
        if self.search_client is None:
            raise ValueError("search_client is required to collect from search")

        urls: List[str] = []
        for query in queries:
            results = self.search_client.search(
                query=query,
                num_results=max_results,
                include_domains=self.approved_domains or None
            )
            urls.extend([result.url for result in results if result.url])

        return self.collect_from_urls(urls, use_sliding_window=use_sliding_window)


class EvidenceGatherer:
    """Gather evidence for claims using a ReAct agent."""

    def __init__(
        self,
        react_agent: ReActAgent,
        credibility_analyzer: Optional[CredibilityAnalyzer] = None
    ):
        self.react_agent = react_agent
        self.credibility_analyzer = credibility_analyzer

    def gather_for_claim(
        self,
        claim: Claim
    ) -> Tuple[List[Evidence], Optional[Verdict], List[ReasoningStep]]:
        """Run the ReAct agent and return evidence, verdict, and reasoning steps."""
        try:
            state = self.react_agent.verify_claim(claim)
        except Exception as exc:
            logger.error(f"Evidence gathering failed for claim: {exc}")
            return [], None, []

        evidence = self._score_evidence(state.collected_evidence)
        return evidence, state.final_verdict, state.reasoning_steps

    def _score_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Ensure evidence items have credibility scores."""
        if not self.credibility_analyzer:
            return evidence_list

        for evidence in evidence_list:
            if evidence.credibility_score > 0:
                continue

            content = WebContent(
                url=evidence.source_url,
                title=evidence.source_title or "",
                main_text=evidence.text,
                author=evidence.source_author,
                publish_date=evidence.publish_date,
                extraction_success=True,
                extraction_method="evidence_excerpt"
            )
            try:
                score = self.credibility_analyzer.analyze_credibility(content)
                evidence.credibility_score = score.overall_score
            except Exception as exc:
                logger.warning(f"Credibility scoring failed: {exc}")

        return evidence_list


class AutoLabeler:
    """Generate initial labels and confidence scores."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def label(self, verdict: Optional[Verdict]) -> Tuple[str, float, bool]:
        """Return label, confidence, and review flag."""
        if verdict is None:
            return "not_enough_info", 0.0, True

        confidence = max(verdict.confidence_scores.values()) if verdict.confidence_scores else 0.0
        needs_review = confidence < self.min_confidence
        return verdict.label, confidence, needs_review


class DatasetBuilder:
    """Construct dataset records from collected claims."""

    def __init__(
        self,
        claim_collector: ClaimCollector,
        evidence_gatherer: EvidenceGatherer,
        auto_labeler: AutoLabeler,
        min_credibility_score: float = 0.5
    ):
        self.claim_collector = claim_collector
        self.evidence_gatherer = evidence_gatherer
        self.auto_labeler = auto_labeler
        self.min_credibility_score = min_credibility_score

    def build_records(self, collected_claims: Iterable[CollectedClaim]) -> List[DatasetRecord]:
        """Build dataset records from collected claims."""
        records: List[DatasetRecord] = []

        for collected in collected_claims:
            evidence, verdict, reasoning_steps = self.evidence_gatherer.gather_for_claim(collected.claim)
            if not evidence:
                continue

            if not any(ev.credibility_score >= self.min_credibility_score for ev in evidence):
                continue

            label, confidence, needs_review = self.auto_labeler.label(verdict)
            metadata = collected.to_metadata()
            metadata.update({
                "confidence": confidence,
                "needs_review": needs_review,
                "evidence_count": len(evidence),
                "source_urls": [ev.source_url for ev in evidence],
                "verdict_confidence_scores": verdict.confidence_scores if verdict else {}
            })

            records.append(
                DatasetRecord(
                    claim=collected.claim,
                    evidence=evidence,
                    label=label,
                    metadata=metadata,
                    reasoning_trace=reasoning_steps
                )
            )

        logger.info(f"Built {len(records)} dataset records")
        return records


class DatasetExporter:
    """Export dataset records to JSONL with train/val/test splits."""

    def __init__(
        self,
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        dataset_name: str = "factcheck"
    ):
        self.output_dir = Path(output_dir)
        self.split_ratios = split_ratios
        self.seed = seed
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, records: List[DatasetRecord]) -> Dict[str, Path]:
        """Export records to JSONL files and return paths."""
        if not records:
            raise ValueError("No records to export")

        train, val, test = self._split_records(records)
        paths = {
            "train": self.output_dir / f"{self.dataset_name}_train.jsonl",
            "val": self.output_dir / f"{self.dataset_name}_val.jsonl",
            "test": self.output_dir / f"{self.dataset_name}_test.jsonl"
        }

        self._write_jsonl(paths["train"], train)
        self._write_jsonl(paths["val"], val)
        self._write_jsonl(paths["test"], test)

        logger.info(
            "Exported dataset splits - "
            f"train: {len(train)}, val: {len(val)}, test: {len(test)}"
        )
        return paths

    def _split_records(self, records: List[DatasetRecord]) -> Tuple[List[DatasetRecord], List[DatasetRecord], List[DatasetRecord]]:
        """Split records into train/val/test sets."""
        rng = random.Random(self.seed)
        shuffled = records[:]
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_size = int(total * self.split_ratios[0])
        val_size = int(total * self.split_ratios[1])

        train = shuffled[:train_size]
        val = shuffled[train_size:train_size + val_size]
        test = shuffled[train_size + val_size:]

        if not test and val:
            test.append(val.pop())

        return train, val, test

    def _write_jsonl(self, path: Path, records: List[DatasetRecord]) -> None:
        """Write records to a JSONL file."""
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
