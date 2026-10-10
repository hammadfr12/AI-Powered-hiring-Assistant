# AI-Powered Hiring Assistant — common tasks.
#
# `make help` lists everything.  All targets use the project virtualenv at
# .venv so they behave identically on a fresh clone.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
STREAMLIT := .venv/bin/streamlit
PYTEST  := .venv/bin/python -m pytest
RUFF    := .venv/bin/python -m ruff
PORT    ?= 8501

.DEFAULT_GOAL := help
.PHONY: help venv install run test test-cov lint format check corpus index eval clean clean-all zip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	python3.12 -m venv .venv || python3 -m venv .venv
	$(PIP) install --upgrade pip wheel setuptools

install: venv ## Install dependencies into .venv
	$(PIP) install -r requirements.txt

run: ## Launch the Streamlit app (PORT=8501 by default)
	$(STREAMLIT) run app.py --server.port=$(PORT) --browser.gatherUsageStats=false

test: ## Run the test suite
	$(PYTEST) tests/ -q

test-cov: ## Run tests with a coverage report
	$(PYTEST) tests/ -q --cov=src/hiring_assistant --cov-report=term-missing

lint: ## Check code style and common mistakes
	$(RUFF) check src/ tests/ app.py scripts/

format: ## Auto-fix lint issues and format
	$(RUFF) check --fix src/ tests/ app.py scripts/
	$(RUFF) format src/ tests/ app.py scripts/

check: lint test ## Lint and test — the pre-commit gate

corpus: ## Regenerate the synthetic corpus on disk
	$(PY) scripts/build_corpus.py --force

index: ## Build (or rebuild) the vector index from stored documents
	$(PY) scripts/build_index.py --rebuild

eval: ## Run the RAG benchmark and write reports to docs/
	$(PY) scripts/run_evaluation.py

clean: ## Remove caches and build artefacts (keeps the index and database)
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Also remove the index, database and generated corpus
	rm -rf data/index data/synthetic data/*.sqlite3 data/*.sqlite3-wal data/*.sqlite3-shm

zip: ## Package the project for distribution
	$(PY) scripts/make_zip.py
