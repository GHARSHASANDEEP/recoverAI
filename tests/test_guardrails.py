from src.engine.decision_engine import (
    evaluate_guardrails,
)


def make_case(
    failure_category="temporary_bank_failure",
    communication_opt_in=True,
    attempt_number=1,
    erv=1000.0,
    recovery_amount=10000.0,
):
    return {
        "case_id": "TEST_CASE",
        "customer_id": "TEST_CUSTOMER",
        "failure_category": failure_category,
        "customer_segment": "regular",
        "communication_opt_in": communication_opt_in,
        "attempt_number": attempt_number,
        "recovery_amount": recovery_amount,
        "erv": erv,
    }


def assert_blocked(case, action):
    result = evaluate_guardrails(
        case,
        action,
    )

    assert result["allowed"] is False

    return result


def assert_allowed(case, action):
    result = evaluate_guardrails(
        case,
        action,
    )

    assert result["allowed"] is True

    return result


# --------------------------------------------------
# RETRY GUARDRAILS
# --------------------------------------------------


def test_risk_decline_blocks_retry():

    case = make_case(
        failure_category="risk_decline"
    )

    result = assert_blocked(
        case,
        "retry",
    )

    assert "risk_decline" in result[
        "reason"
    ]


def test_blocked_instrument_blocks_retry():

    case = make_case(
        failure_category="blocked_instrument"
    )

    result = assert_blocked(
        case,
        "retry",
    )

    assert "blocked_instrument" in result[
        "reason"
    ]


def test_expired_instrument_blocks_retry():

    case = make_case(
        failure_category="expired_instrument"
    )

    result = assert_blocked(
        case,
        "retry",
    )

    assert "expired_instrument" in result[
        "reason"
    ]


def test_retry_limit_blocks_retry():

    case = make_case(
        attempt_number=3
    )

    result = assert_blocked(
        case,
        "retry",
    )

    assert "retry limit" in result[
        "reason"
    ].lower()


def test_normal_failure_allows_retry():

    case = make_case(
        failure_category="temporary_bank_failure",
        attempt_number=1,
    )

    assert_allowed(
        case,
        "retry",
    )


# --------------------------------------------------
# REMINDER GUARDRAILS
# --------------------------------------------------


def test_opted_out_customer_blocks_reminder():

    case = make_case(
        communication_opt_in=False
    )

    result = assert_blocked(
        case,
        "reminder",
    )

    assert "opt" in result[
        "reason"
    ].lower()


def test_opted_in_customer_allows_reminder():

    case = make_case(
        communication_opt_in=True
    )

    assert_allowed(
        case,
        "reminder",
    )


# --------------------------------------------------
# ESCALATION
# --------------------------------------------------


def test_escalation_is_allowed():

    case = make_case()

    result = assert_allowed(
        case,
        "escalate",
    )

    assert "manual" in result[
        "reason"
    ].lower()


# --------------------------------------------------
# ECONOMIC STOP CONDITION
# --------------------------------------------------


def test_negative_erv_blocks_action():

    case = make_case(
        erv=-1.0
    )

    result = assert_blocked(
        case,
        "retry",
    )

    assert "economically" in result[
        "reason"
    ].lower()


def test_zero_erv_blocks_action():

    case = make_case(
        erv=0.0
    )

    result = assert_blocked(
        case,
        "reminder",
    )

    assert "economically" in result[
        "reason"
    ].lower()