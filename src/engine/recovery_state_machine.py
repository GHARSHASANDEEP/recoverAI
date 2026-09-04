"""
RecoverAI recovery state machine.

The state machine manages the lifecycle of a recovery case.
It does not calculate ERV or make ML predictions.

Responsibilities:
    - Track the current recovery state
    - Validate state transitions
    - Determine the next state after execution
    - Prevent invalid lifecycle transitions
"""


# ---------------------------------------------------------
# Recovery states
# ---------------------------------------------------------

NEW = "new"
ANALYZING = "analyzing"
RECOVERY_READY = "recovery_ready"
ACTION_SENT = "action_sent"
WAITING = "waiting"
REASSESS = "reassess"
RECOVERED = "recovered"
ESCALATED = "escalated"
STOPPED = "stopped"


# ---------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------

VALID_TRANSITIONS = {

    NEW: {
        ANALYZING,
        STOPPED,
    },

    ANALYZING: {
        RECOVERY_READY,
        ESCALATED,
        STOPPED,
    },

    RECOVERY_READY: {
        ACTION_SENT,
        ESCALATED,
        STOPPED,
    },

    ACTION_SENT: {
        WAITING,
        RECOVERED,
        REASSESS,
        ESCALATED,
        STOPPED,
    },

    WAITING: {
        RECOVERED,
        REASSESS,
        ESCALATED,
        STOPPED,
    },

    REASSESS: {
        RECOVERY_READY,
        ESCALATED,
        STOPPED,
    },

    RECOVERED: set(),

    ESCALATED: set(),

    STOPPED: set(),
}


class RecoveryStateMachine:
    """
    Manage the lifecycle of one recovery case.
    """

    def __init__(
        self,
        case_id: str,
    ):
        self.case_id = case_id
        self.state = NEW
        self.history = [NEW]

    def transition(
        self,
        next_state: str,
    ) -> str:
        """
        Move the case to a valid next state.

        Raises ValueError when an invalid transition
        is attempted.
        """

        allowed_states = VALID_TRANSITIONS.get(
            self.state,
            set(),
        )

        if next_state not in allowed_states:
            raise ValueError(
                f"Invalid state transition for "
                f"{self.case_id}: "
                f"{self.state} -> {next_state}"
            )

        self.state = next_state
        self.history.append(
            next_state
        )

        return self.state

    def get_state(self) -> str:
        """Return the current state."""

        return self.state

    def get_history(self) -> list:
        """Return the complete state history."""

        return list(
            self.history
        )

    def is_terminal(self) -> bool:
        """Return True when no further action is expected."""

        return self.state in {
            RECOVERED,
            ESCALATED,
            STOPPED,
        }


def determine_post_action_state(
    execution_status: str,
    verification_status: str,
) -> str:
    """
    Determine the next state after an action.

    This function deliberately does not decide which
    action should be taken. It only interprets the
    result of execution and verification.
    """

    if verification_status == "verified":
        return RECOVERED

    if execution_status == "escalated":
        return ESCALATED

    if execution_status == "stopped":
        return STOPPED

    if verification_status == "not_recovered":
        return REASSESS

    return WAITING


def create_recovery_state_machine(
    case_id: str,
) -> RecoveryStateMachine:
    """Create a new state machine for a recovery case."""

    return RecoveryStateMachine(
        case_id=case_id
    )