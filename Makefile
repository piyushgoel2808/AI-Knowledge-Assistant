# =============================================================================
# AI Knowledge Assistant — Makefile (NewsLens-AI Style Workflow)
# =============================================================================
# Dedicated Ports (no conflict with NewsLens-AI on 8000/5173):
#   Frontend UI:  http://localhost:5174
#   Backend API:  http://localhost:8001
# =============================================================================

.PHONY: help up down restart logs status build

.DEFAULT_GOAL := help

COMPOSE := docker compose

help: ## Show available commands
	@echo ""
	@echo "AI Knowledge Assistant Commands"
	@echo "==============================="
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z0-9_-]+:.*?##/ \
		{ printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

up: ## Start stack in background on ports 5174 (UI) and 8001 (API) without rebuilding
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example"; fi
	$(COMPOSE) up -d
	@echo ""
	@echo "Frontend UI:  http://localhost:5174"
	@echo "Backend API:  http://localhost:8001/health"
	@echo "Swagger Docs: http://localhost:8001/docs"

down: ## Stop running containers
	$(COMPOSE) down

restart: ## Restart containers quickly without rebuilding
	$(COMPOSE) restart

logs: ## Follow live logs from backend and frontend
	$(COMPOSE) logs -f

status: ## Check container status and backend health
	$(COMPOSE) ps
	@echo ""
	@curl -s http://localhost:8001/health || echo "Backend is still starting..."
	@echo ""

build: ## Rebuild Docker images (only needed when requirements.txt or package.json changes)
	$(COMPOSE) up -d --build
