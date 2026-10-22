# =============================================================================
# Store Staffing Analyzer
#
# Developer convenience targets.  Run `make help` for the full list.
# Recipe lines below MUST start with a real tab character.
# =============================================================================

.DEFAULT_GOAL := help

PYTHON  ?= python
PIP     ?= $(PYTHON) -m pip
COMPOSE ?= docker compose
DATA    ?= data/RevI-Test.csv
IMAGE   ?= store_staffing:latest

.PHONY: help install install-prod run serve test lint format docker-build docker-run clean

help: ## Show this help message
	@echo "Store Staffing Analyzer"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

install: ## Install the development dependencies
	$(PIP) install -r requirements-dev.txt

install-prod: ## Install the runtime dependencies only
	$(PIP) install -r requirements.txt

run: ## Run the analysis pipeline on the sample dataset
	$(PYTHON) run.py $(DATA)

serve: ## Start the FastAPI app and web dashboard with auto-reload
	$(PYTHON) -m uvicorn store_staffing.api.main:app --reload --port 8000

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Lint with ruff and verify formatting with black
	$(PYTHON) -m ruff check store_staffing/
	$(PYTHON) -m black --check store_staffing/ --line-length 100

format: ## Auto-format the code with black and fix lint issues with ruff
	$(PYTHON) -m black store_staffing/ --line-length 100
	$(PYTHON) -m ruff check --fix store_staffing/

docker-build: ## Build the production Docker image
	docker build -t $(IMAGE) .

docker-run: ## Run the production Docker image on http://localhost:8000
	docker run --rm -it -p 8000:8000 -v ./data:/app/data:ro $(IMAGE)

clean: ## Remove caches, build artefacts and generated output
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov
	rm -rf build dist *.egg-info
	rm -rf output reports runs
