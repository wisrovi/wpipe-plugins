# =============================================================================
# MAKEFILE
# =============================================================================

# Author: William Steve Rodriguez Villamizar
# LinkedIn: https://www.linkedin.com/in/william-steve-rodriguez/

.PHONY: help install lint format security test clean docs diagrams diagrams-render docker-test

# Variables
PYTHON := python
PYTEST := pytest
PROJECT_DIR := src/wpipe_plugins
SKILL_DIR := src/wpipe_plugins/skills/excalidraw-diagram/references

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "========================================"
	@echo "  Project Makefile - Available Commands"
	@echo "========================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# INSTALLATION
# =============================================================================
install: ## Install dependencies using pyproject.toml
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linters (ruff, black, isort)
	ruff check $(PROJECT_DIR)/ tests/
	black --check $(PROJECT_DIR)/ tests/

format: ## Format code (ruff, black)
	ruff check --fix $(PROJECT_DIR)/ tests/
	black $(PROJECT_DIR)/ tests/

security: ## Run security checks (bandit)
	bandit -r $(PROJECT_DIR)/ -ll

# =============================================================================
# TESTING
# =============================================================================
test: ## Run tests with coverage locally
	$(PYTEST) tests/ --cov=$(PROJECT_DIR) --cov-report=term-missing -v

docker-test: ## Run tests with coverage in Docker (User Preference)
	./run_tests_docker.sh

# =============================================================================
# DIAGRAMS
# =============================================================================
diagrams: ## Create diagrams directory
	mkdir -p docs/diagrams

diagrams-render: diagrams ## Render all Excalidraw diagrams to PNG
	@echo "Rendering Excalidraw diagrams..."
	@for f in docs/diagrams/*.excalidraw; do \
		if [ -f "$$f" ]; then \
			name=$$(basename $$f .excalidraw); \
			echo "  Rendering $$name..."; \
			cd $(SKILL_DIR) && uv run python render_excalidraw.py ../../../../../$$f --output ../../../../../docs/diagrams/$$name.png --scale 2 && cd ../../../../..; \
		fi \
	done
	@echo "Diagrams rendered successfully!"

# =============================================================================
# CLEANING
# =============================================================================
clean: ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f .coverage
	rm -rf htmlcov/
	rm -rf build/ dist/ *.egg-info/ .eggs/

.DEFAULT_GOAL := help
