import pytest

from app.policy import decide_risk


@pytest.mark.parametrize(
    ("risk_score", "expected"),
    [
        (0.0, ("low", "approve")),
        (0.2999, ("low", "approve")),
        (0.3, ("medium", "manual_review")),
        (0.7499, ("medium", "manual_review")),
        (0.75, ("high", "block")),
        (1.0, ("high", "block")),
    ],
)
def test_policy_thresholds(risk_score: float, expected: tuple[str, str]):
    assert decide_risk(risk_score) == expected


def test_policy_rejects_invalid_probability():
    with pytest.raises(ValueError):
        decide_risk(1.1)
