.PHONY: help sync-backend sync-ml

help:
	@echo "Available commands:"
	@echo "  make sync-backend    Synchronize the backend uv environment"
	@echo "  make sync-ml         Synchronize the ML service uv environment"

sync-backend:
	cd backend && uv sync

sync-ml:
	cd ml_service && uv sync
