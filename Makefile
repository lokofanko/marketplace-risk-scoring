.PHONY: help sync-backend sync-ml test run docker-up docker-down docker-logs

help:
	@echo "Available commands:"
	@echo "  make sync-backend    Synchronize the backend uv environment"
	@echo "  make sync-ml         Synchronize the ML service uv environment"
	@echo "  make test            Run ML service tests"
	@echo "  make run             Run ML service locally on port 8000"
	@echo "  make docker-up       Build and start ML service and PostgreSQL"
	@echo "  make docker-down     Stop Docker Compose services"
	@echo "  make docker-logs     Follow Docker Compose logs"

sync-backend:
	cd backend && uv sync

sync-ml:
	cd ml_service && uv sync

test:
	cd ml_service && uv run pytest -q

run:
	cd ml_service && uv run uvicorn app.main:app --reload --port 8000

docker-up:
	docker compose up --build --detach

docker-down:
	docker compose down

docker-logs:
	docker compose logs --follow
