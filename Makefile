.PHONY: test smoke greedy-rl lint typecheck

test:
	uv run pytest

smoke:
	uv run aczero smoke-test

greedy-rl:
	sh scripts/test_greedy_rl_agent.sh

lint:
	uv run ruff check .

typecheck:
	uv run mypy src
