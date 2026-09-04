import pytest

from src.engine.recovery_state_machine import (
    RecoveryStateMachine,
    NEW,
    ANALYZING,
    RECOVERY_READY,
    ACTION_SENT,
    WAITING,
    REASSESS,
    RECOVERED,
    ESCALATED,
    STOPPED,
    determine_post_action_state,
)


def test_normal_recovery_lifecycle():

    machine = RecoveryStateMachine(
        "TEST_001"
    )

    assert machine.get_state() == NEW

    machine.transition(ANALYZING)
    machine.transition(RECOVERY_READY)
    machine.transition(ACTION_SENT)
    machine.transition(WAITING)
    machine.transition(RECOVERED)

    assert machine.get_state() == RECOVERED
    assert machine.is_terminal() is True


def test_failed_action_moves_to_reassessment():

    next_state = determine_post_action_state(
        execution_status="executed",
        verification_status="not_recovered",
    )

    assert next_state == REASSESS


def test_verified_action_moves_to_recovered():

    next_state = determine_post_action_state(
        execution_status="executed",
        verification_status="verified",
    )

    assert next_state == RECOVERED


def test_escalation_moves_to_escalated():

    next_state = determine_post_action_state(
        execution_status="escalated",
        verification_status="not_applicable",
    )

    assert next_state == ESCALATED


def test_stopped_action_moves_to_stopped():

    next_state = determine_post_action_state(
        execution_status="stopped",
        verification_status="not_applicable",
    )

    assert next_state == STOPPED


def test_reassessment_can_return_to_recovery():

    machine = RecoveryStateMachine(
        "TEST_002"
    )

    machine.transition(ANALYZING)
    machine.transition(RECOVERY_READY)
    machine.transition(ACTION_SENT)
    machine.transition(WAITING)
    machine.transition(REASSESS)
    machine.transition(RECOVERY_READY)

    assert (
        machine.get_state()
        == RECOVERY_READY
    )


def test_recovered_case_cannot_resume():

    machine = RecoveryStateMachine(
        "TEST_003"
    )

    machine.transition(ANALYZING)
    machine.transition(RECOVERY_READY)
    machine.transition(ACTION_SENT)
    machine.transition(WAITING)
    machine.transition(RECOVERED)

    with pytest.raises(ValueError):
        machine.transition(
            REASSESS
        )


def test_escalated_case_cannot_resume():

    machine = RecoveryStateMachine(
        "TEST_004"
    )

    machine.transition(ANALYZING)
    machine.transition(ESCALATED)

    with pytest.raises(ValueError):
        machine.transition(
            RECOVERY_READY
        )