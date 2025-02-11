# echoscribe developer entry points. Every target runs against ./.venv so the
# host interpreter is never involved.

PY_DIRS := src tests examples scripts

.DEFAULT_GOAL := help

.PHONY: help build test test-all format format-check lint typecheck release clean demo

help: ## Show the available targets.
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build a wheel from the current source tree.
	.venv/bin/python -m build --wheel --no-isolation

test: ## Run the fast suite (everything except @pytest.mark.slow).
	.venv/bin/python -m pytest -q -m "not slow"

test-all: ## Run the whole suite including slow and model-marked tests.
	.venv/bin/python -m pytest -q

format: ## Rewrite sources with ruff format and sort imports.
	.venv/bin/ruff format src tests examples scripts
	.venv/bin/ruff check --fix --select I src tests examples scripts

format-check: ## Verify formatting and lint without writing anything.
	.venv/bin/ruff format --check src tests examples scripts && .venv/bin/ruff check src tests examples scripts

lint: ## Report lint findings only.
	.venv/bin/ruff check src tests examples scripts

typecheck: ## Run mypy over the installed package sources.
	.venv/bin/python -m mypy src

release: ## Build sdist plus wheel and write dist/SHA256SUMS.
	rm -rf dist
	.venv/bin/python -m build
	cd dist && sha256sum * > SHA256SUMS

demo: ## Run the full offline synthesize-train-decode-score loop.
	.venv/bin/python -m echoscribe.cli demo --workdir /tmp/echoscribe-demo

clean: ## Remove local build and cache artifacts.
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
