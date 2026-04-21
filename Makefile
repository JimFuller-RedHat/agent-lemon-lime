.PHONY: help install test test-integration lint format typecheck check fix clean discover assert init \
        hello-world gateway-start gateway-stop

UV := uv run
OPENSHELL := .venv/bin/openshell
HELLO_WORLD_DIR := examples/hello_world
HELLO_WORLD_POLICY := $(HELLO_WORLD_DIR)/.agent-lemon/hello-world-assert.yaml
HELLO_WORLD_PROMPT ?= Say hello.
HELLO_WORLD_PROVIDERS ?=

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install          Install package in editable mode"
	@echo "  test             Run unit tests (excludes integration)"
	@echo "  test-integration Run integration tests (requires OpenShell cluster)"
	@echo "  lint             Check code with ruff"
	@echo "  format           Format code with ruff"
	@echo "  typecheck        Run ty type checker"
	@echo "  check            lint + typecheck + test"
	@echo "  fix              Auto-fix ruff issues and format"
	@echo "  clean            Remove build artifacts and caches"
	@echo ""
	@echo "  discover         Run agent-lemon discover (reads agent-lemon.yaml)"
	@echo "  assert           Run agent-lemon assert (reads agent-lemon.yaml)"
	@echo "  init             Generate agent-lemon.yaml template in current dir"
	@echo ""
	@echo "  gateway-start    Start local OpenShell gateway (k3s in Docker)"
	@echo "  gateway-stop     Stop local OpenShell gateway"
	@echo "  hello-world      Run hello_world agent in OpenShell sandbox with SCP"
	@echo "                   HELLO_WORLD_PROMPT='...'           override the agent prompt"
	@echo "                   HELLO_WORLD_PROVIDERS='a b'       space-separated provider names"

install:
	uv sync
sync: install

test:
	$(UV) python -m pytest tests/ -v -k "not integration"

test-integration:
	$(UV) python -m pytest tests/ -v -m integration

lint:
	$(UV) ruff check src/ tests/ examples/

format:
	$(UV) ruff format src/ tests/ examples/

typecheck:
	$(UV) ty check src/

check: lint typecheck test

fix:
	$(UV) ruff check --fix src/ tests/ examples/
	$(UV) ruff format src/ tests/ examples/

clean:
	rm -rf .agent-lemon/ dist/ .ruff_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

discover:
	$(UV) agent-lemon discover

assert:
	$(UV) agent-lemon assert

init:
	$(UV) agent-lemon init

gateway-start:
	$(OPENSHELL) gateway start

gateway-stop:
	$(OPENSHELL) gateway stop

hello-world:
	$(OPENSHELL) sandbox create \
		--policy $(HELLO_WORLD_POLICY) \
		--upload $(HELLO_WORLD_DIR) \
		$(foreach p,$(HELLO_WORLD_PROVIDERS),--provider $(p)) \
		--no-keep \
		-- uv run agent.py --prompt "$(HELLO_WORLD_PROMPT)"
