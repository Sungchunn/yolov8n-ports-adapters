.PHONY: backend frontend test lint-imports typecheck frontend-build frontend-typecheck

backend:
	cd backend && MODEL_PATH=../models/yolov8n.pt uv run --extra vision uvicorn inference.entrypoints.app:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && bun run dev

test:
	cd backend && uv run --extra dev pytest

lint-imports:
	cd backend && uv run --extra dev lint-imports

typecheck:
	cd backend && uv run --extra dev mypy src

frontend-build:
	cd frontend && bun run build

frontend-typecheck:
	cd frontend && bun run typecheck
