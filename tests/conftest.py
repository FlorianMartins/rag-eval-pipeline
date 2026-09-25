from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    # The CLI resolves evals/ and app/ relative to the repository root, like it does in CI.
    monkeypatch.chdir(ROOT)
