.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
IMAGE ?= rageval:local

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/rageval:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q -e '.[dev]'

install: $(BIN)/rageval ## Create the virtualenv and install the project

test: install ## Lint + unit tests
	$(BIN)/ruff check . && $(BIN)/ruff format --check .
	$(BIN)/pytest --cov=rageval --cov=app

eval: install ## Run the evaluation suite and the quality gate (exit 1 if it fails)
	$(BIN)/rageval run --baseline evals/baseline.json

eval-degraded: install ## Demo: starve the retriever (top_k=1) and watch the gate fail
	$(BIN)/rageval run --system-opt top_k=1 --baseline evals/baseline.json --out reports/degraded

eval-hallucination: install ## Demo: loosen the refusal guardrail and watch the gate fail
	$(BIN)/rageval run --system-opt min_coverage=0.2 --baseline evals/baseline.json --out reports/hallucination

baseline: eval ## Promote the latest (passing) report to evals/baseline.json
	$(BIN)/rageval baseline reports/report.json

serve: install ## Run the RAG app as an HTTP service on :8000
	$(BIN)/python -m app.server --port 8000

eval-http: install ## Evaluate the running HTTP service (run `make serve` first)
	$(BIN)/rageval run --system http --system-opt url=http://127.0.0.1:8000/answer --out reports/http

docker-build: ## Build the evaluation image
	docker build -t $(IMAGE) .

docker-eval: docker-build ## Run the evaluation in Docker, reports written to ./reports
	mkdir -p reports && docker run --rm --user "$$(id -u):$$(id -g)" -v "$(CURDIR)/reports:/workspace/reports" $(IMAGE)

clean: ## Remove caches and generated reports
	rm -rf reports .pytest_cache .ruff_cache .coverage **/__pycache__

.PHONY: help install test eval eval-degraded eval-hallucination baseline serve eval-http docker-build docker-eval clean
