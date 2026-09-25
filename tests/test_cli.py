"""End-to-end: the CLI exit code is the contract with the CI pipeline."""

import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from app.rag_app import RAGApp
from app.server import make_handler
from rageval.cli import EXIT_CONFIG_ERROR, EXIT_GATE_FAILED, EXIT_OK, main


def run(tmp_path, *extra):
    code = main(["run", "-q", "--out", str(tmp_path), *extra])
    report = json.loads((tmp_path / "report.json").read_text()) if (tmp_path / "report.json").exists() else None
    return code, report


def test_healthy_system_passes_the_gate(tmp_path):
    code, report = run(tmp_path)
    assert code == EXIT_OK
    assert report["gate"]["passed"]
    assert (tmp_path / "report.md").read_text().startswith("## RAG evaluation — quality gate ✅ PASSED")


@pytest.mark.parametrize(
    "option, failing_check",
    [
        ("top_k=1", "min_accuracy"),  # retriever starved: multi-fact questions lose context
        ("min_coverage=0.2", "min_refusal_accuracy"),  # guardrail loosened: hallucinations on out-of-scope
        ("min_coverage=0.8", "min_answer_similarity"),  # over-cautious: refuses answerable questions
    ],
)
def test_degraded_configurations_fail_the_gate(tmp_path, option, failing_check):
    code, report = run(tmp_path, "--system-opt", option)
    assert code == EXIT_GATE_FAILED
    assert failing_check in {c["name"] for c in report["gate"]["checks"] if not c["passed"]}


def test_no_fail_flag_reports_without_blocking(tmp_path):
    code, report = run(tmp_path, "--system-opt", "top_k=1", "--no-fail")
    assert code == EXIT_OK and not report["gate"]["passed"]


def test_regression_against_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics": {"accuracy": 1.0}}))
    code, report = run(tmp_path / "out", "--baseline", str(baseline))
    assert code == EXIT_GATE_FAILED
    assert [c["name"] for c in report["gate"]["checks"] if not c["passed"]] == ["no_regression_accuracy"]


def test_committed_baseline_matches_current_system(tmp_path):
    code, _ = run(tmp_path, "--baseline", "evals/baseline.json")
    assert code == EXIT_OK


@pytest.mark.parametrize(
    "args", [["--system", "does-not-exist"], ["--dataset", "missing.json"], ["--tag", "no-such-tag"]]
)
def test_configuration_errors_exit_2(tmp_path, args):
    assert run(tmp_path, *args)[0] == EXIT_CONFIG_ERROR


def test_http_system_end_to_end(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(RAGApp()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/answer"
        code, report = run(tmp_path, "--system", "http", "--system-opt", f"url={url}")
    finally:
        server.shutdown()
    assert code == EXIT_OK
    assert report["metadata"]["system"] == {"type": "http", "url": url}
    assert report["metrics"]["latency_p95_ms"] > 0


def test_baseline_promotion_refuses_failing_reports(tmp_path):
    run(tmp_path, "--system-opt", "top_k=1", "--no-fail")
    assert main(["baseline", str(tmp_path / "report.json"), "--out", str(tmp_path / "b.json")]) == EXIT_GATE_FAILED
    run(tmp_path)
    assert main(["baseline", str(tmp_path / "report.json"), "--out", str(tmp_path / "b.json")]) == EXIT_OK
    assert "cases" not in json.loads((tmp_path / "b.json").read_text())


def test_http_system_rejects_non_http_schemes(tmp_path):
    assert run(tmp_path, "--system", "http", "--system-opt", "url=file:///etc/passwd")[0] == EXIT_CONFIG_ERROR
