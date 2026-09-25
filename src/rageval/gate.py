"""Quality gate: turns metrics into a pass/fail decision a CI pipeline can act on.

Two kinds of checks:

* **Absolute thresholds** (``min_<metric>`` / ``max_<metric>``): the floor the product must
  never go below, whatever the baseline says.
* **Regression checks** against a baseline report (usually the one produced on ``main``):
  catch a slow drift that is still above the floor — e.g. accuracy 0.95 -> 0.86.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Metrics where a higher value is better; used for regression checks.
HIGHER_IS_BETTER = ("accuracy", "answer_similarity", "context_recall", "context_precision", "faithfulness")


@dataclass(frozen=True)
class Check:
    name: str
    metric: str
    comparator: str  # ">=" or "<="
    threshold: float
    actual: float | None
    passed: bool
    kind: str = "threshold"  # or "regression"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def threshold_checks(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[Check]:
    checks = []
    for key, limit in thresholds.items():
        direction, _, metric = key.partition("_")
        if direction not in ("min", "max") or not metric:
            raise ValueError(f"invalid threshold key {key!r}: expected min_<metric> or max_<metric>")
        if metric not in metrics:
            raise ValueError(f"threshold {key!r} refers to unknown metric {metric!r}")
        actual = metrics[metric]
        if actual is None:
            passed = False  # a metric we cannot compute must not pass silently
        else:
            passed = actual >= limit if direction == "min" else actual <= limit
        checks.append(Check(key, metric, ">=" if direction == "min" else "<=", float(limit), actual, passed))
    return checks


def regression_checks(metrics: dict[str, Any], baseline: dict[str, Any], max_drop: float) -> list[Check]:
    checks = []
    for metric in HIGHER_IS_BETTER:
        before, now = baseline.get(metric), metrics.get(metric)
        if before is None or now is None:
            continue
        floor = round(before - max_drop, 4)
        checks.append(Check(f"no_regression_{metric}", metric, ">=", floor, now, now >= floor, kind="regression"))
    return checks


def evaluate_gate(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    baseline: dict[str, Any] | None = None,
    max_drop: float | None = None,
) -> tuple[bool, list[Check]]:
    checks = threshold_checks(metrics, thresholds)
    if baseline is not None and max_drop is not None:
        checks += regression_checks(metrics, baseline, max_drop)
    return all(c.passed for c in checks), checks
