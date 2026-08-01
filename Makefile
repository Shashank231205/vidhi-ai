.DEFAULT_GOAL := help
BACKEND := backend
FRONTEND := frontend
UV := uv

.PHONY: help install hooks dev dev-api dev-web lint format typecheck test check clean web-build

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: hooks ## Install backend and frontend deps, and git hooks
	cd $(BACKEND) && $(UV) sync --extra dev
	cd $(FRONTEND) && npm install

hooks: ## Run CI's gates before every push instead of after
	git config core.hooksPath .githooks

# The browser only ever uses :3000 — Next proxies /api/* to the API process,
# so there is one URL to open and no CORS. API_PORT must match frontend/.env.local.
API_PORT ?= 8010
WEB_PORT ?= 3000

dev: ## Run API + frontend; open http://localhost:3000
	@echo "API  → http://127.0.0.1:$(API_PORT)  (proxied at /api)"
	@echo "Open → http://localhost:$(WEB_PORT)"
	@trap 'kill 0' EXIT INT TERM; \
	(cd $(BACKEND) && $(UV) run uvicorn api.main:create_app --factory --reload --port $(API_PORT)) & \
	(cd $(FRONTEND) && npm run dev -- --port $(WEB_PORT)) & \
	wait

dev-api: ## Run only the API
	cd $(BACKEND) && $(UV) run uvicorn api.main:create_app --factory --reload --port $(API_PORT)

dev-web: ## Run only the frontend
	cd $(FRONTEND) && npm run dev -- --port $(WEB_PORT)

lint: ## Ruff check
	cd $(BACKEND) && $(UV) run ruff check .

format: ## Ruff format + autofix
	cd $(BACKEND) && $(UV) run ruff format . && $(UV) run ruff check . --fix

typecheck: ## mypy --strict
	cd $(BACKEND) && $(UV) run mypy api core tests

test: ## pytest with coverage
	cd $(BACKEND) && $(UV) run pytest -q --cov=api --cov=core --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

web-build: ## Type-check, lint, and build the frontend
	cd $(FRONTEND) && npm run lint && npm run build

clean: ## Remove caches
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \
		-o -name .ruff_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} +
