import pytest

from rageval.gate import evaluate_gate, regression_checks, threshold_checks

METRICS = {"accuracy": 0.9, "context_recall": 0.8, "latency_p95_ms": 120.0, "faithfulness": None}


def test_min_and_max_thresholds():
    checks = {c.name: c.passed for c in threshold_checks(METRICS, {"min_accuracy": 0.8, "max_latency_p95_ms": 100})}
    assert checks == {"min_accuracy": True, "max_latency_p95_ms": False}


def test_threshold_is_inclusive():
    assert threshold_checks(METRICS, {"min_context_recall": 0.8})[0].passed


def test_missing_metric_value_fails_closed():
    assert not threshold_checks(METRICS, {"min_faithfulness": 0.5})[0].passed


@pytest.mark.parametrize("key", ["accuracy", "avg_accuracy", "min_unknown_metric"])
def test_invalid_threshold_keys_are_rejected(key):
    with pytest.raises(ValueError):
        threshold_checks(METRICS, {key: 1})


def test_regression_against_baseline():
    baseline = {"accuracy": 0.97, "context_recall": 0.82}
    checks = {c.metric: c.passed for c in regression_checks(METRICS, baseline, max_drop=0.05)}
    assert checks == {"accuracy": False, "context_recall": True}


def test_gate_passes_only_if_every_check_passes():
    passed, _ = evaluate_gate(METRICS, {"min_accuracy": 0.8})
    assert passed
    passed, _ = evaluate_gate(METRICS, {"min_accuracy": 0.8}, baseline={"accuracy": 1.0}, max_drop=0.05)
    assert not passed
