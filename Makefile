.DEFAULT_GOAL := help
BACKEND := backend
UV := uv

.PHONY: help install dev lint format typecheck test check clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install backend deps (incl. dev extras)
	cd $(BACKEND) && $(UV) sync --extra dev

dev: ## Run the API with reload on :8000
	cd $(BACKEND) && $(UV) run uvicorn api.main:create_app --factory --reload --port 8000

lint: ## Ruff check
	cd $(BACKEND) && $(UV) run ruff check .

format: ## Ruff format + autofix
	cd $(BACKEND) && $(UV) run ruff format . && $(UV) run ruff check . --fix

typecheck: ## mypy --strict
	cd $(BACKEND) && $(UV) run mypy api core tests

test: ## pytest with coverage
	cd $(BACKEND) && $(UV) run pytest -q --cov=api --cov=core --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \
		-o -name .ruff_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} +
