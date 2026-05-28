.PHONY: run test lint-imports typecheck

run:
	uv run --extra vision uvicorn inference.entrypoints.app:app --reload

test:
	uv run --extra dev pytest

lint-imports:
	uv run --extra dev lint-imports

typecheck:
	uv run --extra dev mypy src
