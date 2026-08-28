# API Contract

The backend sends listing and account features to `POST /score`. The ML service
returns a probability, a threshold-based decision, model version, and readable risk
factors. Example payloads live in `shared/contracts/`.

Decision policy for `logreg_v1`:

- score below `0.30`: `low` / `approve`;
- score from `0.30` up to `0.75`: `medium` / `manual_review`;
- score at or above `0.75`: `high` / `block`.

FastAPI returns `422` for invalid request fields, `503` when the model artifact is
unavailable, and `503` from `/logs` when PostgreSQL is not configured or reachable.
