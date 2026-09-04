from src.engine.recovery_policy import (
    get_initial_action,
    get_permitted_actions,
    is_action_permitted,
)


def test_insufficient_funds_starts_with_reminder():
    assert (
        get_initial_action("insufficient_funds")
        == "reminder"
    )


def test_authentication_failure_starts_with_reminder():
    assert (
        get_initial_action("authentication_failed")
        == "reminder"
    )


def test_risk_decline_allows_only_escalation():
    assert (
        get_initial_action("risk_decline")
        == "escalate"
    )

    assert get_permitted_actions(
        "risk_decline"
    ) == ["escalate"]


def test_blocked_instrument_allows_only_escalation():
    assert (
        get_initial_action("blocked_instrument")
        == "escalate"
    )

    assert is_action_permitted(
        "blocked_instrument",
        "retry",
    ) is False

    assert is_action_permitted(
        "blocked_instrument",
        "escalate",
    ) is True


def test_temporary_failure_starts_with_retry():
    assert (
        get_initial_action("temporary_failure")
        == "retry"
    )


def test_insufficient_funds_permits_recovery_path():
    permitted = get_permitted_actions(
        "insufficient_funds"
    )

    assert "reminder" in permitted
    assert "retry" in permitted
    assert "escalate" in permitted


def test_unknown_failure_uses_conservative_path():
    assert (
        get_initial_action("something_unknown")
        == "reminder"
    )


from src.engine.recovery_policy import (
    get_recovery_sequence,
    get_next_policy_action,
)


def test_insufficient_funds_sequence():
    sequence = get_recovery_sequence(
        "insufficient_funds"
    )

    assert sequence == [
        "reminder",
        "retry",
        "escalate",
    ]


def test_insufficient_funds_does_not_skip_to_escalation():
    action = get_next_policy_action(
        "insufficient_funds",
        attempted_actions=[],
    )

    assert action == "reminder"


def test_insufficient_funds_moves_to_retry_after_reminder():
    action = get_next_policy_action(
        "insufficient_funds",
        attempted_actions=[
            "reminder",
        ],
    )

    assert action == "retry"


def test_insufficient_funds_escalates_after_recovery_path():
    action = get_next_policy_action(
        "insufficient_funds",
        attempted_actions=[
            "reminder",
            "retry",
        ],
    )

    assert action == "escalate"


def test_risk_decline_can_only_escalate():
    sequence = get_recovery_sequence(
        "risk_decline"
    )

    assert sequence == [
        "escalate",
    ]


def test_risk_decline_cannot_retry():
    action = get_next_policy_action(
        "risk_decline",
        attempted_actions=[],
    )

    assert action == "escalate"


def test_blocked_instrument_can_only_escalate():
    sequence = get_recovery_sequence(
        "blocked_instrument"
    )

    assert sequence == [
        "escalate",
    ]


def test_temporary_bank_failure_retries_first():
    action = get_next_policy_action(
        "temporary_bank_failure",
        attempted_actions=[],
    )

    assert action == "retry"


def test_timeout_retries_first():
    action = get_next_policy_action(
        "timeout",
        attempted_actions=[],
    )

    assert action == "retry"


def test_authentication_failure_starts_with_reminder():
    action = get_next_policy_action(
        "authentication_failed",
        attempted_actions=[],
    )

    assert action == "reminder"


def test_expired_instrument_starts_with_reminder():
    action = get_next_policy_action(
        "expired_instrument",
        attempted_actions=[],
    )

    assert action == "reminder"


def test_exhausted_recovery_path_stops():
    action = get_next_policy_action(
        "insufficient_funds",
        attempted_actions=[
            "reminder",
            "retry",
            "escalate",
        ],
    )

    assert action == "stop"