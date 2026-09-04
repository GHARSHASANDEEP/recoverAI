from src.engine.decision_engine import evaluate_guardrails


def test_judge_opted_out_customer():
    case = {
        "failure_category": "insufficient_funds",
        "communication_opt_in": False,
        "attempt_number": 1,
        "recovery_amount": 20000,
        "erv": 10000,
    }

    result = evaluate_guardrails(
        case,
        "reminder",
    )

    assert result["allowed"] is False
    assert "opt" in result["reason"].lower()


def test_judge_risk_decline():
    case = {
        "failure_category": "risk_decline",
        "communication_opt_in": True,
        "attempt_number": 1,
        "recovery_amount": 20000,
        "erv": 10000,
    }

    result = evaluate_guardrails(
        case,
        "retry",
    )

    assert result["allowed"] is False
    assert "risk_decline" in result["reason"]


def test_judge_max_attempts():
    case = {
        "failure_category": "authentication_failed",
        "communication_opt_in": True,
        "attempt_number": 3,
        "recovery_amount": 20000,
        "erv": 10000,
    }

    result = evaluate_guardrails(
        case,
        "retry",
    )

    assert result["allowed"] is False
    assert "limit" in result["reason"].lower()


def test_judge_normal_failure():
    case = {
        "failure_category": "temporary_bank_failure",
        "communication_opt_in": True,
        "attempt_number": 1,
        "recovery_amount": 20000,
        "erv": 10000,
    }

    result = evaluate_guardrails(
        case,
        "retry",
    )

    assert result["allowed"] is True


def test_judge_escalation():
    case = {
        "failure_category": "risk_decline",
        "communication_opt_in": False,
        "attempt_number": 10,
        "recovery_amount": 20000,
        "erv": 10000,
    }

    result = evaluate_guardrails(
        case,
        "escalate",
    )

    assert result["allowed"] is True