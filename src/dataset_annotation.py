"""Optional annotation interface for dataset verification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GRADIO_AVAILABLE = False


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class AnnotationInterface:
    """Simple Gradio UI for label verification."""

    input_path: str
    output_path: Optional[str] = None
    label_options: List[str] = field(default_factory=lambda: ["supported", "refuted", "not_enough_info"])

    def __post_init__(self) -> None:
        if self.output_path is None:
            stem = Path(self.input_path).stem
            self.output_path = str(Path(self.input_path).with_name(f"{stem}.annotated.jsonl"))

    def _load_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with Path(self.input_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _save_records(self, records: List[Dict[str, Any]]) -> None:
        with Path(self.output_path).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _format_evidence(self, evidence_list: List[Dict[str, Any]]) -> str:
        if not evidence_list:
            return "No evidence available."

        lines = []
        for idx, evidence in enumerate(evidence_list, start=1):
            title = evidence.get("source_title") or "Untitled"
            url = evidence.get("source_url") or ""
            snippet = (evidence.get("text") or "")[:300].strip()
            lines.append(f"{idx}. {title}\n   {url}\n   {snippet}")
        return "\n\n".join(lines)

    def launch(self) -> None:
        if not GRADIO_AVAILABLE:
            raise ImportError("gradio is not installed. Install it to use the annotation UI.")

        records = self._load_records()
        total = len(records)

        def load_record(index: int):
            if index >= total:
                return "", "", "", "", f"Completed {total} records."

            record = records[index]
            claim_text = record.get("claim", {}).get("text", "")
            evidence_text = self._format_evidence(record.get("evidence", []))
            predicted_label = record.get("label", "not_enough_info")
            notes = record.get("metadata", {}).get("annotator_notes", "")
            status = f"Record {index + 1} of {total}"
            return claim_text, evidence_text, predicted_label, predicted_label, notes, status

        def save_and_next(index: int, label: str, notes: str):
            if index < total:
                record = records[index]
                record["label"] = label
                record.setdefault("metadata", {})
                record["metadata"]["annotator_notes"] = notes
                record["metadata"]["annotated_at"] = _utc_now_iso()

            next_index = index + 1
            if next_index >= total:
                self._save_records(records)
                return next_index, "", "", "", "", f"Completed {total} records. Saved to {self.output_path}"

            return (next_index,) + load_record(next_index)

        with gr.Blocks(title="Dataset Annotation") as demo:
            index_state = gr.State(0)
            gr.Markdown("## Dataset Annotation Interface")

            claim_box = gr.Textbox(label="Claim", lines=3, interactive=False)
            evidence_box = gr.Textbox(label="Evidence", lines=10, interactive=False)
            predicted_box = gr.Textbox(label="Predicted Label", interactive=False)
            label_dropdown = gr.Dropdown(self.label_options, label="Correct Label")
            notes_box = gr.Textbox(label="Annotator Notes", lines=3)
            status_box = gr.Textbox(label="Status", interactive=False)
            save_btn = gr.Button("Save and Next")

            demo.load(
                fn=lambda idx: load_record(idx),
                inputs=index_state,
                outputs=[claim_box, evidence_box, predicted_box, label_dropdown, notes_box, status_box]
            )

            save_btn.click(
                fn=save_and_next,
                inputs=[index_state, label_dropdown, notes_box],
                outputs=[index_state, claim_box, evidence_box, predicted_box, label_dropdown, notes_box, status_box]
            )

        demo.launch()
