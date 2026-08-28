-- Recent model decisions for a simple audit view.
SELECT
    listing_id,
    risk_score,
    risk_level,
    recommended_action,
    model_version,
    created_at
FROM prediction_logs
ORDER BY created_at DESC
LIMIT 20;
