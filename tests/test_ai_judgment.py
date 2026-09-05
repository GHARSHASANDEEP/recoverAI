import pandas as pd

from src.model.ai_judgment import judge_recovery_case
from src.engine.recovery_agent import select_next_action


def _scores():
    return pd.DataFrame([
        {"action": "retry", "recovery_probability": 0.82, "expected_recovery": 8200.0, "action_cost": 2.0, "erv": 8198.0},
        {"action": "reminder", "recovery_probability": 0.45, "expected_recovery": 4500.0, "action_cost": 1.0, "erv": 4499.0},
        {"action": "escalate", "recovery_probability": 0.2, "expected_recovery": 2000.0, "action_cost": 20.0, "erv": 1980.0},
    ])


def test_judgment_explains_allowed_recommendation():
    result = judge_recovery_case(
        {
            "failure_category": "temporary_bank_failure",
            "communication_opt_in": True,
            "attempt_number": 1,
            "recovery_amount": 10000,
        },
        _scores(),
        "retry",
    )

    assert result["verdict"] == "automate"
    assert result["recommended_action"] == "retry"
    assert result["confidence_level"] in {"medium", "high"}
    assert result["evidence"]
    assert "reassess" in result["counterfactual"]


def test_judgment_explains_risk_decline_block():
    result = judge_recovery_case(
        {
            "failure_category": "risk_decline",
            "communication_opt_in": True,
            "attempt_number": 1,
            "recovery_amount": 10000,
        },
        _scores(),
        "escalate",
    )

    assert result["recommended_action"] == "escalate"
    assert "retry" in result["blocked_actions"]
    assert any("unsafe for automatic retry" in item for item in result["evidence"])


def test_ai_selects_permitted_action_over_policy_default():
    result = select_next_action(
        {
            "case_id": "CASE_AI_1",
            "failure_category": "insufficient_funds",
            "communication_opt_in": True,
            "attempt_number": 1,
            "recovery_amount": 10000,
        },
        _scores(),
        set(),
    )

    assert result["final_action"] == "retry"
    assert result["final_action"] != "reminder"