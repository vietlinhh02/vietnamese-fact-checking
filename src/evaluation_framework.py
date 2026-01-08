"""Evaluation and comparison framework for the fact-checking system."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data_models import Claim
from src.experiment_tracking import ExperimentTracker

logger = logging.getLogger(__name__)

LABELS = ("supported", "refuted", "not_enough_info")


@dataclass
class ComponentConfig:
    """Configuration flags for pipeline components."""

    use_search: bool = True
    use_crawl: bool = True
    use_credibility: bool = True
    use_self_verification: bool = True
    name: str = "full"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_search": self.use_search,
            "use_crawl": self.use_crawl,
            "use_credibility": self.use_credibility,
            "use_self_verification": self.use_self_verification,
            "name": self.name
        }

    def key(self) -> Tuple[bool, bool, bool, bool]:
        return (
            self.use_search,
            self.use_crawl,
            self.use_credibility,
            self.use_self_verification
        )


@dataclass
class EvaluationConfig:
    """Runtime configuration for evaluation runs."""

    dataset_path: str
    output_dir: str = "experiments/evaluation"
    run_name: str = "evaluation"
    random_seed: int = 42
    max_claims: Optional[int] = None
    labels: Tuple[str, ...] = LABELS
    component_config: ComponentConfig = field(default_factory=ComponentConfig)
    save_predictions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "random_seed": self.random_seed,
            "max_claims": self.max_claims,
            "labels": list(self.labels),
            "component_config": self.component_config.to_dict(),
            "save_predictions": self.save_predictions
        }


@dataclass
class EvaluationSample:
    """Single evaluation sample."""

    claim: Claim
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationPrediction:
    """Prediction for a sample."""

    label: str
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    raw_output: Optional[Any] = None


@dataclass
class EvaluationResult:
    """Prediction result with correctness."""

    sample: EvaluationSample
    prediction: EvaluationPrediction
    is_correct: bool


class NoOpSearchClient:
    """Search client that returns no results for ablation."""

    def search(self, *args, **kwargs):
        return []


class NoOpWebCrawler:
    """Crawler that returns no content for ablation."""

    def extract_content(self, *args, **kwargs):
        return None


class NoOpCredibilityAnalyzer:
    """Credibility analyzer that returns a neutral score for ablation."""

    def analyze_source(self, *args, **kwargs):
        return {"overall_score": 0.0, "explanation": "credibility disabled"}

    def analyze_credibility(self, *args, **kwargs):
        class _Score:
            overall_score = 0.0
        return _Score()


class ReActPredictor:
    """Adapter that uses ReActAgent for prediction."""

    def __init__(
        self,
        llm_controller,
        search_client,
        web_crawler,
        credibility_analyzer,
        max_iterations: int = 10,
        self_verification_module: Optional[Any] = None,
        self_verification_threshold: float = 0.3
    ):
        self.llm_controller = llm_controller
        self.search_client = search_client
        self.web_crawler = web_crawler
        self.credibility_analyzer = credibility_analyzer
        self.max_iterations = max_iterations
        self.self_verification_module = self_verification_module
        self.self_verification_threshold = self_verification_threshold
        self._agent_cache: Dict[Tuple[bool, bool, bool, bool], Any] = {}

    def _get_agent(self, component_config: ComponentConfig):
        key = component_config.key()
        if key in self._agent_cache:
            return self._agent_cache[key]

        from src.react_agent import create_react_agent

        search_client = self.search_client if component_config.use_search else NoOpSearchClient()
        web_crawler = self.web_crawler if component_config.use_crawl else NoOpWebCrawler()
        credibility_analyzer = (
            self.credibility_analyzer
            if component_config.use_credibility
            else NoOpCredibilityAnalyzer()
        )
        agent = create_react_agent(
            llm_controller=self.llm_controller,
            search_client=search_client,
            web_crawler=web_crawler,
            credibility_analyzer=credibility_analyzer,
            max_iterations=self.max_iterations
        )
        self._agent_cache[key] = agent
        return agent

    def predict(self, claim: Claim, component_config: ComponentConfig) -> EvaluationPrediction:
        agent = self._get_agent(component_config)
        state = agent.verify_claim(claim)
        verdict = state.final_verdict
        if verdict is None:
            return EvaluationPrediction(label="not_enough_info", confidence_scores={})
        if component_config.use_self_verification and self.self_verification_module:
            try:
                quality_score, _ = self.self_verification_module.verify_explanation(
                    verdict.explanation,
                    state.collected_evidence
                )
                if quality_score.overall_score < self.self_verification_threshold:
                    verdict.label = "not_enough_info"
            except Exception as exc:
                logger.warning("Self-verification failed: %s", exc)
        return EvaluationPrediction(
            label=verdict.label,
            confidence_scores=verdict.confidence_scores,
            raw_output=verdict
        )


def load_dataset(path: str) -> List[EvaluationSample]:
    """Load dataset from JSON or JSONL files."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: List[Dict[str, Any]] = []
    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        with dataset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError("Unsupported dataset format; expected list")
        records = data

    samples = []
    for record in records:
        claim = _parse_claim(record)
        label = _normalize_label(_extract_label(record))
        metadata = record.get("metadata", {})
        samples.append(EvaluationSample(claim=claim, label=label, metadata=metadata))

    return samples


def _parse_claim(record: Dict[str, Any]) -> Claim:
    claim_data = record.get("claim")
    if isinstance(claim_data, dict):
        return Claim.from_dict(claim_data)
    if isinstance(claim_data, str):
        return Claim(text=claim_data)
    if "text" in record:
        return Claim(text=record["text"])
    if "claim_text" in record:
        return Claim(text=record["claim_text"])
    raise ValueError("Record missing claim field")


def _extract_label(record: Dict[str, Any]) -> str:
    for key in ("label", "verdict", "verdict_label", "gold_label"):
        if key in record:
            return record[key]
    raise ValueError("Record missing label field")


def _normalize_label(label: str) -> str:
    label = str(label).strip().lower().replace(" ", "_")
    mapping = {
        "support": "supported",
        "supported": "supported",
        "refute": "refuted",
        "refuted": "refuted",
        "not_enough_info": "not_enough_info",
        "not_enoughinfo": "not_enough_info",
        "not_enough": "not_enough_info",
        "nei": "not_enough_info"
    }
    if label not in mapping:
        raise ValueError(f"Unsupported label: {label}")
    return mapping[label]


def compute_confusion_matrix(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    labels: Sequence[str]
) -> List[List[int]]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for true, pred in zip(true_labels, pred_labels):
        i = label_index[true]
        j = label_index[pred]
        matrix[i][j] += 1
    return matrix


def compute_metrics(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    labels: Sequence[str]
) -> Dict[str, Any]:
    if not true_labels:
        raise ValueError("No labels provided for metrics")

    confusion = compute_confusion_matrix(true_labels, pred_labels, labels)
    total = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / total if total else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    supports: Dict[str, int] = {}

    for idx, label in enumerate(labels):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(len(labels)) if row != idx)
        fn = sum(confusion[idx][col] for col in range(len(labels)) if col != idx)
        support = sum(confusion[idx])
        supports[label] = support

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    macro_precision = precision_sum / len(labels)
    macro_recall = recall_sum / len(labels)
    macro_f1 = f1_sum / len(labels)

    micro_tp = correct
    micro_fp = total - correct
    micro_fn = total - correct
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class,
        "supports": supports,
        "confusion_matrix": confusion
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def paired_t_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict[str, float]:
    """Paired t-test with approximate p-value when scipy is unavailable."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have the same length")
    n = len(scores_a)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "n": n}

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(variance) if variance > 0 else 0.0
    if std_diff == 0:
        return {"t_stat": 0.0, "p_value": 1.0, "n": n}

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_value = _approx_t_p_value(t_stat, n - 1)
    return {"t_stat": t_stat, "p_value": p_value, "n": n}


def _approx_t_p_value(t_stat: float, df: int) -> float:
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t_stat), df=df))
        return float(max(min(p, 1.0), 0.0))
    except Exception:
        # Normal approximation for large df or fallback when scipy is missing.
        if df > 30:
            p = 2 * (1 - _normal_cdf(abs(t_stat)))
        else:
            p = 2 * (1 - _normal_cdf(abs(t_stat)))
        return float(max(min(p, 1.0), 0.0))


def mcnemar_test(
    true_labels: Sequence[str],
    preds_a: Sequence[str],
    preds_b: Sequence[str]
) -> Dict[str, float]:
    """McNemar's test with continuity correction."""
    if not (len(true_labels) == len(preds_a) == len(preds_b)):
        raise ValueError("Label lists must have the same length")

    b = 0
    c = 0
    for true, a, b_pred in zip(true_labels, preds_a, preds_b):
        a_correct = a == true
        b_correct = b_pred == true
        if a_correct and not b_correct:
            b += 1
        elif b_correct and not a_correct:
            c += 1

    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p_value": p_value}


def build_precision_recall_curve(
    scores: Sequence[float],
    true_labels: Sequence[str],
    positive_label: str
) -> List[Dict[str, float]]:
    if not scores:
        return []

    thresholds = sorted(set(scores), reverse=True)
    curve = []
    total_positive = sum(1 for label in true_labels if label == positive_label)
    if total_positive == 0:
        return []

    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for score, label in zip(scores, true_labels):
            pred_positive = score >= threshold
            actual_positive = label == positive_label
            if pred_positive and actual_positive:
                tp += 1
            elif pred_positive and not actual_positive:
                fp += 1
            elif not pred_positive and actual_positive:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        curve.append({"threshold": threshold, "precision": precision, "recall": recall})

    return curve


def build_pr_curves(
    true_labels: Sequence[str],
    predictions: Sequence[EvaluationPrediction],
    labels: Sequence[str]
) -> Dict[str, List[Dict[str, float]]]:
    curves: Dict[str, List[Dict[str, float]]] = {}
    for label in labels:
        scores = [
            prediction.confidence_scores.get(label, 0.0) if prediction.confidence_scores else 0.0
            for prediction in predictions
        ]
        if any(score > 0 for score in scores):
            curves[label] = build_precision_recall_curve(scores, true_labels, label)
    return curves


def generate_latex_table(metrics: Dict[str, Any], labels: Sequence[str]) -> str:
    header = "Label & Precision & Recall & F1 \\\\"
    rows = []
    for label in labels:
        stats = metrics["per_class"][label]
        rows.append(
            f"{label} & {stats['precision']:.3f} & {stats['recall']:.3f} & {stats['f1']:.3f} \\\\"
        )
    rows.append(
        f"Macro Avg & {metrics['precision']:.3f} & {metrics['recall']:.3f} & {metrics['f1']:.3f} \\\\"
    )
    table = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        header,
        "\\hline",
        *rows,
        "\\hline",
        "\\end{tabular}"
    ]
    return "\n".join(table)


def plot_confusion_matrix(confusion: List[List[int]], labels: Sequence[str], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(confusion[i][j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curves(
    curves: Dict[str, List[Dict[str, float]]],
    output_path: str
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, curve in curves.items():
        if not curve:
            continue
        recalls = [point["recall"] for point in curve]
        precisions = [point["precision"] for point in curve]
        ax.plot(recalls, precisions, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


class EvaluationPipeline:
    """Run evaluation and generate reports."""

    def __init__(
        self,
        predictor: Any,
        experiment_tracker: Optional[ExperimentTracker] = None
    ):
        self.predictor = predictor
        self.experiment_tracker = experiment_tracker

    def run(self, config: EvaluationConfig) -> Dict[str, Any]:
        samples = load_dataset(config.dataset_path)
        if config.max_claims is not None:
            random.seed(config.random_seed)
            samples = random.sample(samples, min(config.max_claims, len(samples)))
        if not samples:
            raise ValueError("No samples available for evaluation")

        results: List[EvaluationResult] = []
        predictions: List[EvaluationPrediction] = []
        errors: List[str] = []

        for sample in samples:
            try:
                prediction = self._predict(sample.claim, config.component_config)
            except Exception as exc:
                prediction = EvaluationPrediction(label="not_enough_info")
                errors.append(f"{sample.claim.id}: {exc}")
            if prediction.label not in config.labels:
                logger.warning("Unknown prediction label '%s'; mapping to not_enough_info", prediction.label)
                prediction.label = "not_enough_info"
            is_correct = prediction.label == sample.label
            results.append(EvaluationResult(sample=sample, prediction=prediction, is_correct=is_correct))
            predictions.append(prediction)

        true_labels = [result.sample.label for result in results]
        pred_labels = [result.prediction.label for result in results]
        metrics = compute_metrics(true_labels, pred_labels, config.labels)
        pr_curves = build_pr_curves(true_labels, predictions, config.labels)

        report = {
            "run_name": config.run_name,
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "dataset_path": config.dataset_path,
            "sample_count": len(samples),
            "component_config": config.component_config.to_dict(),
            "metrics": metrics,
            "precision_recall_curves": pr_curves,
            "errors": errors
        }

        if config.save_predictions:
            report["results"] = [
                {
                    "claim_id": result.sample.claim.id,
                    "claim_text": result.sample.claim.text,
                    "true_label": result.sample.label,
                    "predicted_label": result.prediction.label,
                    "confidence_scores": result.prediction.confidence_scores,
                    "is_correct": result.is_correct
                }
                for result in results
            ]

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{config.run_name}_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        latex_table = generate_latex_table(metrics, config.labels)
        (output_dir / f"{config.run_name}_metrics_table.tex").write_text(latex_table, encoding="utf-8")
        (output_dir / f"{config.run_name}_config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        try:
            plot_confusion_matrix(metrics["confusion_matrix"], config.labels, str(output_dir / f"{config.run_name}_confusion.png"))
        except Exception as exc:
            logger.warning("Confusion matrix plot skipped: %s", exc)

        if pr_curves:
            try:
                plot_precision_recall_curves(pr_curves, str(output_dir / f"{config.run_name}_pr_curve.png"))
            except Exception as exc:
                logger.warning("Precision-recall plot skipped: %s", exc)

        if self.experiment_tracker:
            self.experiment_tracker.log_experiment(
                name=config.run_name,
                config=config.to_dict(),
                metrics=metrics,
                seed=config.random_seed,
                artifacts={
                    "report": str(report_path),
                    "latex_table": str(output_dir / f"{config.run_name}_metrics_table.tex")
                }
            )

        return report

    def _predict(self, claim: Claim, component_config: ComponentConfig) -> EvaluationPrediction:
        if hasattr(self.predictor, "predict"):
            return self.predictor.predict(claim, component_config)
        if callable(self.predictor):
            return self.predictor(claim, component_config)
        raise ValueError("Predictor must be callable or expose predict()")


class AblationStudyRunner:
    """Run ablation studies across multiple component configurations."""

    def __init__(self, pipeline: EvaluationPipeline):
        self.pipeline = pipeline

    def run(
        self,
        base_config: EvaluationConfig,
        component_configs: Dict[str, ComponentConfig]
    ) -> Dict[str, Any]:
        results: Dict[str, Dict[str, Any]] = {}
        baseline_config = EvaluationConfig(
            dataset_path=base_config.dataset_path,
            output_dir=base_config.output_dir,
            run_name=f"{base_config.run_name}_baseline",
            random_seed=base_config.random_seed,
            max_claims=base_config.max_claims,
            labels=base_config.labels,
            component_config=base_config.component_config,
            save_predictions=True
        )
        baseline_report = self.pipeline.run(baseline_config)
        results["baseline"] = baseline_report

        for name, component_config in component_configs.items():
            run_config = EvaluationConfig(
                dataset_path=base_config.dataset_path,
                output_dir=base_config.output_dir,
                run_name=f"{base_config.run_name}_{name}",
                random_seed=base_config.random_seed,
                max_claims=base_config.max_claims,
                labels=base_config.labels,
                component_config=component_config,
                save_predictions=True
            )
            results[name] = self.pipeline.run(run_config)

        comparison = self._compare_runs(results, base_key="baseline")
        summary = {
            "baseline": baseline_report["metrics"],
            "variants": {
                name: report["metrics"] for name, report in results.items() if name != "baseline"
            },
            "comparisons": comparison
        }
        summary_path = Path(base_config.output_dir) / f"{base_config.run_name}_ablation_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _compare_runs(self, reports: Dict[str, Dict[str, Any]], base_key: str) -> List[Dict[str, Any]]:
        if base_key not in reports:
            raise ValueError("Baseline run not found for comparison")

        base_report = reports[base_key]
        base_results = base_report.get("results", [])
        base_true = [item["true_label"] for item in base_results]
        base_pred = [item["predicted_label"] for item in base_results]
        base_accuracy = base_report["metrics"]["accuracy"]

        comparisons = []
        for name, report in reports.items():
            if name == base_key:
                continue
            variant_results = report.get("results", [])
            if len(variant_results) != len(base_results):
                raise ValueError("Variant results length mismatch with baseline")
            variant_pred = [item["predicted_label"] for item in variant_results]
            variant_accuracy = report["metrics"]["accuracy"]

            per_sample_base = [1.0 if pred == true else 0.0 for pred, true in zip(base_pred, base_true)]
            per_sample_variant = [1.0 if pred == true else 0.0 for pred, true in zip(variant_pred, base_true)]

            t_test = paired_t_test(per_sample_base, per_sample_variant)
            mcnemar = mcnemar_test(base_true, base_pred, variant_pred)

            comparisons.append(
                {
                    "variant": name,
                    "baseline_accuracy": base_accuracy,
                    "variant_accuracy": variant_accuracy,
                    "accuracy_delta": variant_accuracy - base_accuracy,
                    "paired_t_test": t_test,
                    "mcnemar_test": mcnemar
                }
            )

        return comparisons
