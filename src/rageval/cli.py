"""Command-line entry point.

Exit codes (the CI contract):
    0  quality gate passed
    1  quality gate failed (metrics below threshold or regression vs baseline)
    2  configuration / usage error (bad dataset, unknown system, invalid config)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rageval import __version__
from rageval.dataset import DatasetError, load_dataset
from rageval.evaluator import CaseResult, EvalSettings, aggregate, aggregate_by_tag, run_evaluation
from rageval.gate import evaluate_gate
from rageval.metrics.answer import DEFAULT_REFUSAL_MARKERS
from rageval.metrics.similarity import get_similarity
from rageval.report import build_report, write_reports
from rageval.systems import load_system

EXIT_OK, EXIT_GATE_FAILED, EXIT_CONFIG_ERROR = 0, 1, 2
PROVENANCE_KEYS = ("rageval_version", "timestamp", "git_sha", "dataset", "system")


def _parse_options(pairs: list[str]) -> dict[str, str]:
    options = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"invalid --system-opt {pair!r}: expected key=value")
        options[key] = value
    return options


def _git_sha() -> str | None:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - resolved from PATH on purpose (any CI image)
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_baseline(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"::notice::baseline {p} not found — skipping regression checks", file=sys.stderr)
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rageval", description="Evaluate a RAG system or agent and gate on quality.")
    parser.add_argument("--version", action="version", version=f"rageval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the evaluation suite and apply the quality gate")
    run.add_argument("--dataset", default="evals/dataset.json")
    run.add_argument("--config", default="evals/config.toml", help="thresholds and scoring settings")
    run.add_argument("--system", default="local", help="'local', 'http' or 'module:Class'")
    run.add_argument("--system-opt", action="append", default=[], metavar="KEY=VALUE", help="passed to the system")
    run.add_argument("--baseline", help="previous report.json to detect regressions against")
    run.add_argument("--out", default="reports", help="output directory for report.json / report.md")
    run.add_argument("--concurrency", type=int, default=None)
    run.add_argument("--tag", action="append", default=[], help="only run cases with this tag (repeatable)")
    run.add_argument("--no-fail", action="store_true", help="always exit 0 (report-only mode)")
    run.add_argument("-q", "--quiet", action="store_true")

    base = sub.add_parser("baseline", help="promote a report to the committed baseline")
    base.add_argument("report", nargs="?", default="reports/report.json")
    base.add_argument("--out", default="evals/baseline.json")
    return parser


def cmd_baseline(args: argparse.Namespace) -> int:
    """Keep only what regression checks need: aggregate metrics and their provenance."""
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        if not report["gate"]["passed"]:
            print("refusing to promote a report whose quality gate failed", file=sys.stderr)
            return EXIT_GATE_FAILED
        baseline = {
            "schema_version": report["schema_version"],
            "metadata": {k: report["metadata"].get(k) for k in PROVENANCE_KEYS},
            "metrics": report["metrics"],
        }
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"cannot read report {args.report}: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    Path(args.out).write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"baseline written to {args.out} (accuracy={report['metrics']['accuracy']})")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    try:
        with open(args.config, "rb") as fh:
            config = tomllib.load(fh)
        scoring = config.get("scoring", {})
        dataset = load_dataset(args.dataset)
        cases = [c for c in dataset.cases if not args.tag or set(args.tag) & set(c.tags)]
        if not cases:
            raise DatasetError(f"no case matches tags {args.tag}")
        system = load_system(args.system, _parse_options(args.system_opt))
        settings = EvalSettings(
            similarity=get_similarity(scoring.get("similarity_backend", "lexical")),
            min_case_similarity=float(scoring.get("min_case_similarity", 0.5)),
            refusal_markers=tuple(scoring.get("refusal_markers", DEFAULT_REFUSAL_MARKERS)),
            fact_coverage=float(scoring.get("fact_coverage", 0.8)),
        )
    except (OSError, ValueError, TypeError, ImportError, AttributeError, tomllib.TOMLDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    def progress(r: CaseResult) -> None:
        if not args.quiet:
            mark = "PASS" if r.correct else "FAIL"
            print(f"  [{mark}] {r.id:<28} {r.latency_ms:>8.1f} ms  {'; '.join(r.failures)[:70]}")

    if not args.quiet:
        print(f"rageval {__version__} — {len(cases)} cases against {system.describe()}")
    concurrency = args.concurrency or int(config.get("run", {}).get("concurrency", 4))
    results = run_evaluation(system, cases, settings, concurrency=concurrency, on_result=progress)

    metrics = aggregate(results)
    baseline = _load_baseline(args.baseline)
    regression = config.get("regression", {})
    try:
        passed, checks = evaluate_gate(
            metrics,
            {k: float(v) for k, v in config.get("thresholds", {}).items()},
            baseline,
            regression.get("max_drop") if regression.get("enabled", True) else None,
        )
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    metadata = {
        "rageval_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "dataset": {"name": dataset.name, "version": dataset.version, "sha256": dataset.sha256, "path": args.dataset},
        "system": system.describe(),
        "scoring": {k: v for k, v in scoring.items() if k != "refusal_markers"},
        "tags": args.tag,
    }
    report = build_report(
        metadata=metadata,
        metrics=metrics,
        by_tag=aggregate_by_tag(results),
        results=results,
        passed=passed,
        checks=checks,
        baseline=baseline,
    )
    json_path, md_path = write_reports(report, args.out)

    # Surface the report natively in GitHub Actions.
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md_path.read_text(encoding="utf-8"))

    if not args.quiet:
        print()
        for c in checks:
            print(f"  {'✔' if c.passed else '✘'} {c.name:<34} {c.actual!s:>8} {c.comparator} {c.threshold}")
        print(f"\nQuality gate: {'PASSED' if passed else 'FAILED'}  ·  reports: {json_path}, {md_path}")
    if not passed and os.environ.get("GITHUB_ACTIONS"):
        failing = ", ".join(c.name for c in checks if not c.passed)
        print(f"::error title=RAG quality gate failed::{failing}")
    return EXIT_OK if passed or args.no_fail else EXIT_GATE_FAILED


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        return cmd_baseline(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
