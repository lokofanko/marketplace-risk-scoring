# Architecture

The repository is a monorepo with an imported Django marketplace scaffold and a
standalone FastAPI ML service.

```text
backend -> POST /score -> FastAPI -> sklearn Pipeline -> threshold policy
                              |                              |
                              +------ PostgreSQL logs <------+
```

The backend owns users, listings, and moderation workflow. The ML service owns
feature validation, inference, decision policy, model metadata, and prediction audit
logs. Docker Compose runs only the ML service and PostgreSQL in the current prototype.
