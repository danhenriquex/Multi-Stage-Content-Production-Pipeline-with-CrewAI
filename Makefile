.PHONY: help build up down restart rebuild logs health campaign test test-cov lint lint-fix init install

PROJECT = content-pipeline
COMPOSE  = docker compose -p $(PROJECT) -f docker-compose.yml --env-file .env

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {sub("\\\\n",sprintf("\n%22c"," "), $$2); printf "  \033[36m%-20s\033[0m  %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Setup ────────────────────────────────────────────────────────────────────

init: ## First-time setup: run scripts/init.sh
	@bash scripts/init.sh

install: ## Install Python dependencies via uv
	uv venv --python 3.11
	uv pip install -e ".[dev]"

# ── Docker ──────────────────────────────────────────────────────────────────

build: ## Build all images
	$(COMPOSE) build

up: ## Start all services
	$(COMPOSE) up -d
	@echo "\n✓ Services started"
	@echo "  API:     http://localhost:8080/docs"
	@echo "  Dagster: http://localhost:3001"
	@echo "  MLflow:  http://localhost:5050"

down: ## Stop all services
	$(COMPOSE) down

down-volumes: ## Stop all services and remove volumes (DESTRUCTIVE)
	$(COMPOSE) down --volumes

restart: ## Restart one service: make restart s=api
	$(COMPOSE) restart $(s)

rebuild: ## Rebuild + restart one service: make rebuild s=api
	$(COMPOSE) build $(s)
	$(COMPOSE) up -d $(s)

logs: ## Tail all logs (or specific: make logs s=api)
ifdef s
	$(COMPOSE) logs -f $(s)
else
	$(COMPOSE) logs -f
endif

logs-api: ## Tail API logs
	$(COMPOSE) logs -f api

logs-dagster: ## Tail Dagster logs
	$(COMPOSE) logs -f dagster

# ── Health ──────────────────────────────────────────────────────────────────

health: ## Check all service endpoints
	@echo "\n=== Health ==="
	@curl -sf http://localhost:8080/health | python3 -m json.tool && echo "✓ api" || echo "✗ api"
	@curl -sf http://localhost:5050/api/2.0/mlflow/experiments/search > /dev/null && echo "✓ mlflow" || echo "✗ mlflow"

# ── Campaigns ────────────────────────────────────────────────────────────────

campaign: ## Trigger a test campaign
	@curl -s -X POST http://localhost:8080/campaigns \
		-H "Content-Type: application/json" \
		-d '{"title":"AI CRM Launch","brief":"Launch campaign for new AI-powered CRM targeting SMBs","brand_voice":"Professional yet approachable","target_audience":"SMB founders and sales managers","keywords":["AI CRM","sales automation","SMB"]}' \
		| python3 -m json.tool

status: ## Check campaign status: make status id=<campaign_id>
	@curl -s http://localhost:8080/campaigns/$(id) | python3 -m json.tool

campaigns: ## List all campaigns
	@curl -s http://localhost:8080/campaigns | python3 -m json.tool

db: ## Connect to PostgreSQL
	$(COMPOSE) exec postgres psql -U content -d content_pipeline

# ── Tests ────────────────────────────────────────────────────────────────────

test: ## Run unit tests
	uv run pytest tests/unit/ -v --tb=short

test-cov: ## Run tests with coverage report
	uv run pytest tests/unit/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov -v

# ── Lint ─────────────────────────────────────────────────────────────────────

lint: ## Check code style
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix: ## Auto-fix lint issues
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/
