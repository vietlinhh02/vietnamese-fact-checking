"""Property-based tests for dataset construction pipeline."""

import json
from tempfile import TemporaryDirectory

from hypothesis import given, settings, strategies as st

from src.data_models import Claim, Evidence, Verdict
from src.dataset_pipeline import AutoLabeler, CollectedClaim, DatasetBuilder, DatasetExporter, DatasetRecord


@st.composite
def evidence_strategy(draw):
    """Generate a valid Evidence item."""
    text = draw(st.text(min_size=5, max_size=50))
    url_id = draw(st.integers(min_value=1, max_value=1000))
    score = draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    return Evidence(
        text=text,
        source_url=f"https://example{url_id}.com",
        credibility_score=score,
        language="vi"
    )


@st.composite
def dataset_record_strategy(draw):
    """Generate a dataset record for export validation."""
    claim_text = draw(st.text(min_size=5, max_size=50))
    claim = Claim(text=claim_text, language="vi")
    evidence_list = draw(st.lists(evidence_strategy(), min_size=1, max_size=3))
    label = draw(st.sampled_from(["supported", "refuted", "not_enough_info"]))
    return DatasetRecord(
        claim=claim,
        evidence=evidence_list,
        label=label,
        metadata={}
    )


class DummyGatherer:
    """Simple gatherer for deterministic tests."""

    def __init__(self, evidence_list, verdict):
        self.evidence_list = evidence_list
        self.verdict = verdict

    def gather_for_claim(self, claim):
        return self.evidence_list, self.verdict, []


class DummyCollector:
    """Placeholder collector for DatasetBuilder wiring."""

    pass


def _make_verdict(claim_id: str) -> Verdict:
    return Verdict(
        claim_id=claim_id,
        label="supported",
        confidence_scores={
            "supported": 0.8,
            "refuted": 0.1,
            "not_enough_info": 0.1
        }
    )


@given(st.lists(evidence_strategy(), min_size=0, max_size=5))
@settings(max_examples=10, deadline=30000)
def test_dataset_evidence_association_property(evidence_list):
    """
    **Feature: vietnamese-fact-checking, Property 31: Dataset Evidence Association**

    For any claim in the constructed dataset, there should exist at least one
    associated evidence item collected from credible sources.
    **Validates: Requirements 11.2**
    """
    claim = Claim(text="Test claim", language="vi")
    collected = CollectedClaim(
        claim=claim,
        source_url="https://example.com"
    )
    verdict = _make_verdict(claim.id)
    gatherer = DummyGatherer(evidence_list, verdict)
    builder = DatasetBuilder(
        claim_collector=DummyCollector(),
        evidence_gatherer=gatherer,
        auto_labeler=AutoLabeler(min_confidence=0.7),
        min_credibility_score=0.5
    )

    records = builder.build_records([collected])
    for record in records:
        assert record.has_credible_evidence(min_score=0.5)


@given(st.lists(dataset_record_strategy(), min_size=1, max_size=5))
@settings(max_examples=5, deadline=30000)
def test_dataset_schema_compliance_property(records):
    """
    **Feature: vietnamese-fact-checking, Property 32: Dataset Schema Compliance**

    For any exported dataset, each record should contain all required fields:
    claim text, evidence list, label, and metadata (source URLs, timestamps).
    **Validates: Requirements 11.5**
    """
    with TemporaryDirectory() as tmp_dir:
        exporter = DatasetExporter(output_dir=str(tmp_dir), dataset_name="testset")
        paths = exporter.export(records)

        for path in paths.values():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    assert "claim" in record
                    assert "evidence" in record
                    assert "label" in record
                    assert "metadata" in record

                    assert record["claim"].get("text")
                    assert isinstance(record["evidence"], list)
                    assert record["label"] in ["supported", "refuted", "not_enough_info"]

                    metadata = record["metadata"]
                    assert metadata.get("source_urls")
                    assert metadata.get("collected_at")

                    for evidence in record["evidence"]:
                        assert evidence.get("source_url")
