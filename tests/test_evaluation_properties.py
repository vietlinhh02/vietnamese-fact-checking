"""Property-based tests for evaluation framework."""

from tempfile import TemporaryDirectory

from hypothesis import given, settings, strategies as st

from src.evaluation_framework import LABELS, compute_metrics
from src.experiment_tracking import ExperimentTracker


@st.composite
def labels_and_predictions(draw):
    size = draw(st.integers(min_value=1, max_value=50))
    true_labels = draw(st.lists(st.sampled_from(LABELS), min_size=size, max_size=size))
    pred_labels = draw(st.lists(st.sampled_from(LABELS), min_size=size, max_size=size))
    return true_labels, pred_labels


@given(labels_and_predictions())
@settings(max_examples=20, deadline=30000)
def test_evaluation_metrics_completeness_property(data):
    """
    **Feature: vietnamese-fact-checking, Property 35: Evaluation Metrics Completeness**

    Metrics should be computed for any non-empty set of labels and predictions.
    **Validates: Requirements 13.1**
    """
    true_labels, pred_labels = data
    metrics = compute_metrics(true_labels, pred_labels, LABELS)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["micro_f1"] <= 1.0

    assert set(metrics["per_class"].keys()) == set(LABELS)
    assert len(metrics["confusion_matrix"]) == len(LABELS)
    assert sum(metrics["supports"].values()) == len(true_labels)


@st.composite
def experiment_payloads(draw):
    seed = draw(st.integers(min_value=0, max_value=10_000))
    config = {
        "learning_rate": draw(st.floats(min_value=1e-5, max_value=1e-1, allow_nan=False, allow_infinity=False)),
        "batch_size": draw(st.integers(min_value=1, max_value=64)),
        "use_search": draw(st.booleans()),
        "use_crawl": draw(st.booleans())
    }
    metrics = {
        "accuracy": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "f1": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    }
    return seed, config, metrics


@given(experiment_payloads())
@settings(max_examples=10, deadline=30000)
def test_experiment_reproducibility_property(payload):
    """
    **Feature: vietnamese-fact-checking, Property 38: Experiment Reproducibility**

    Logged experiments must preserve configuration and seed for reproducibility.
    **Validates: Requirements 13.5**
    """
    seed, config, metrics = payload
    with TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(db_path=f"{tmp_dir}/experiments.db")
        run_hash = tracker.compute_run_hash(config, seed)
        experiment_id = tracker.log_experiment(
            name="test_run",
            config=config,
            metrics=metrics,
            seed=seed
        )
        record = tracker.get_experiment(experiment_id)

        assert record is not None
        assert record.seed == seed
        assert record.config == config
        assert record.metrics == metrics
        assert record.run_hash == run_hash
