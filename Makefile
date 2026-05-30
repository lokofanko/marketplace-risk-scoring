.PHONY: help sync

help:
	@echo "Available commands:"
	@echo "  make sync    Synchronize the uv environment"

sync:
	uv sync
