-- The ML service also creates this table on startup if it does not exist.
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    listing_id VARCHAR(100) NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    risk_level VARCHAR(20) NOT NULL,
    recommended_action VARCHAR(30) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
