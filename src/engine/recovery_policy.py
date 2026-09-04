"""
Failure-aware recovery policy engine.

The policy determines:
    - Which recovery actions are permitted.
    - Which action should be attempted first.
    - The preferred recovery sequence.

The policy does NOT predict success.
The policy does NOT calculate ERV.
The policy does NOT make ML predictions.

ML and economic signals may be used later to rank
actions within the policy-approved recovery path.

Policy has authority over what actions are permitted
and the safe progression of recovery.
"""

from typing import Dict, List


# ---------------------------------------------------------
# Recovery states
# ---------------------------------------------------------

STATE_NEW = "new"
STATE_ANALYZING = "analyzing"
STATE_RECOVERY_READY = "recovery_ready"
STATE_ACTION_SENT = "action_sent"
STATE_WAITING = "waiting"
STATE_REASSESS = "reassess"
STATE_RECOVERED = "recovered"
STATE_ESCALATED = "escalated"
STATE_STOPPED = "stopped"


# ---------------------------------------------------------
# Failure categories
# ---------------------------------------------------------

INSUFFICIENT_FUNDS = "insufficient_funds"
AUTHENTICATION_FAILED = "authentication_failed"
RISK_DECLINE = "risk_decline"
BLOCKED_INSTRUMENT = "blocked_instrument"
EXPIRED_INSTRUMENT = "expired_instrument"
TEMPORARY_FAILURE = "temporary_failure"
TEMPORARY_BANK_FAILURE = "temporary_bank_failure"
TIMEOUT = "timeout"
LIMIT_EXCEEDED = "limit_exceeded"
UNKNOWN_FAILURE = "unknown_failure"


# ---------------------------------------------------------
# Recovery actions
# ---------------------------------------------------------

REMINDER = "reminder"
RETRY = "retry"
ESCALATE = "escalate"
STOP = "stop"


# ---------------------------------------------------------
# Policy definition
# ---------------------------------------------------------

RECOVERY_POLICIES = {

    # -----------------------------------------------------
    # Insufficient funds
    #
    # Customer-resolvable condition.
    # Do not immediately escalate after the first failure.
    # -----------------------------------------------------

    INSUFFICIENT_FUNDS: {
        "initial_action": REMINDER,

        "permitted_actions": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "recovery_sequence": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "reason": (
            "Insufficient funds is treated as a "
            "customer-resolvable condition. RecoverAI "
            "starts with a reminder, then permits a "
            "controlled retry before escalation."
        ),
    },

    # -----------------------------------------------------
    # Authentication failure
    # -----------------------------------------------------

    AUTHENTICATION_FAILED: {
        "initial_action": REMINDER,

        "permitted_actions": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "recovery_sequence": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "reason": (
            "Authentication failure may require customer "
            "intervention. RecoverAI starts with a reminder "
            "before attempting another payment."
        ),
    },

    # -----------------------------------------------------
    # Temporary failure
    # -----------------------------------------------------

    TEMPORARY_FAILURE: {
        "initial_action": RETRY,

        "permitted_actions": [
            RETRY,
            REMINDER,
            ESCALATE,
        ],

        "recovery_sequence": [
            RETRY,
            REMINDER,
            ESCALATE,
        ],

        "reason": (
            "Temporary failures may succeed on a controlled "
            "retry. If recovery does not occur, RecoverAI "
            "moves toward customer assistance and escalation."
        ),
    },

    # -----------------------------------------------------
    # Temporary bank failure
    # -----------------------------------------------------

    TEMPORARY_BANK_FAILURE: {
        "initial_action": RETRY,

        "permitted_actions": [
            RETRY,
            REMINDER,
            ESCALATE,
        ],

        "recovery_sequence": [
            RETRY,
            REMINDER,
            ESCALATE,
        ],

        "reason": (
            "Temporary bank failures are suitable for "
            "controlled automated retry before escalation."
        ),
    },

    # -----------------------------------------------------
    # Timeout
    # -----------------------------------------------------

    TIMEOUT: {
        "initial_action": RETRY,

        "permitted_actions": [
            RETRY,
            REMINDER,
            ESCALATE,
        ],

        "recovery_sequence": [
            RETRY,
            REMINDER,
            ESCALATE,
        ],

        "reason": (
            "Timeouts may result from transient payment "
            "conditions, so RecoverAI attempts a controlled "
            "retry before escalation."
        ),
    },

    # -----------------------------------------------------
    # Limit exceeded
    # -----------------------------------------------------

    LIMIT_EXCEEDED: {
        "initial_action": REMINDER,

        "permitted_actions": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "recovery_sequence": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "reason": (
            "A limit-related failure may require customer "
            "action before another payment attempt."
        ),
    },

    # -----------------------------------------------------
    # Risk decline
    #
    # No automated retry.
    # -----------------------------------------------------

    RISK_DECLINE: {
        "initial_action": ESCALATE,

        "permitted_actions": [
            ESCALATE,
        ],

        "recovery_sequence": [
            ESCALATE,
        ],

        "reason": (
            "Risk-related declines should not be "
            "automatically retried."
        ),
    },

    # -----------------------------------------------------
    # Blocked instrument
    #
    # No automated retry.
    # -----------------------------------------------------

    BLOCKED_INSTRUMENT: {
        "initial_action": ESCALATE,

        "permitted_actions": [
            ESCALATE,
        ],

        "recovery_sequence": [
            ESCALATE,
        ],

        "reason": (
            "The payment instrument is blocked, so "
            "automated retry is prohibited."
        ),
    },

    # -----------------------------------------------------
    # Expired instrument
    # -----------------------------------------------------

    EXPIRED_INSTRUMENT: {
        "initial_action": REMINDER,

        "permitted_actions": [
            REMINDER,
            ESCALATE,
        ],

        "recovery_sequence": [
            REMINDER,
            ESCALATE,
        ],

        "reason": (
            "The customer may need to update the payment "
            "instrument before recovery can continue."
        ),
    },

    # -----------------------------------------------------
    # Unknown failure
    #
    # Conservative adaptive path.
    # -----------------------------------------------------

    UNKNOWN_FAILURE: {
        "initial_action": REMINDER,

        "permitted_actions": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "recovery_sequence": [
            REMINDER,
            RETRY,
            ESCALATE,
        ],

        "reason": (
            "Unknown failures use a conservative "
            "customer-assisted recovery path."
        ),
    },
}


# ---------------------------------------------------------
# Policy functions
# ---------------------------------------------------------

def normalize_failure_category(
    failure_category,
) -> str:
    """Normalize failure category into a known policy."""

    if failure_category is None:
        return UNKNOWN_FAILURE

    value = str(
        failure_category
    ).strip().lower()

    if not value:
        return UNKNOWN_FAILURE

    # Handle combined categories.
    if "|" in value:
        value = value.split("|")[0]

    return value


def get_recovery_policy(
    failure_category,
) -> Dict:
    """Return the policy for a failure category."""

    category = normalize_failure_category(
        failure_category
    )

    return RECOVERY_POLICIES.get(
        category,
        RECOVERY_POLICIES[
            UNKNOWN_FAILURE
        ],
    )


def get_initial_action(
    failure_category,
) -> str:
    """Return the policy-defined initial action."""

    policy = get_recovery_policy(
        failure_category
    )

    return policy[
        "initial_action"
    ]


def get_permitted_actions(
    failure_category,
) -> List[str]:
    """Return actions permitted by failure policy."""

    policy = get_recovery_policy(
        failure_category
    )

    return list(
        policy[
            "permitted_actions"
        ]
    )


def get_recovery_sequence(
    failure_category,
) -> List[str]:
    """
    Return the preferred recovery sequence.

    This sequence describes the intended progression
    through recovery. It does not execute actions and
    does not evaluate economic value.
    """

    policy = get_recovery_policy(
        failure_category
    )

    return list(
        policy[
            "recovery_sequence"
        ]
    )


def get_next_policy_action(
    failure_category,
    attempted_actions=None,
) -> str:
    """
    Return the next action in the policy-defined
    recovery sequence that has not already been attempted.

    This prevents an action such as insufficient_funds
    from jumping directly to escalation merely because
    an economic score is slightly higher.
    """

    if attempted_actions is None:
        attempted_actions = set()

    else:
        attempted_actions = set(
            attempted_actions
        )

    sequence = get_recovery_sequence(
        failure_category
    )

    for action in sequence:

        if action not in attempted_actions:
            return action

    return STOP


def is_action_permitted(
    failure_category,
    action: str,
) -> bool:
    """Check whether an action is allowed by policy."""

    permitted = get_permitted_actions(
        failure_category
    )

    return action in permitted


def build_recovery_state(
    case: Dict,
) -> Dict:
    """
    Determine the current recovery state.

    This is intentionally deterministic.

    The state machine does not make an economic
    decision. It establishes where the case is
    in the recovery lifecycle.
    """

    if case.get(
        "verified_recovered",
        False,
    ):
        return {
            "state": STATE_RECOVERED,
            "reason": (
                "Recovery has been verified."
            ),
        }

    if case.get(
        "final_status"
    ) == "escalated":
        return {
            "state": STATE_ESCALATED,
            "reason": (
                "Case has been routed to manual review."
            ),
        }

    if case.get(
        "final_status"
    ) == "stopped":
        return {
            "state": STATE_STOPPED,
            "reason": (
                "Recovery has been stopped by policy."
            ),
        }

    attempts = int(
        case.get(
            "attempts",
            0,
        )
    )

    if attempts == 0:
        return {
            "state": STATE_NEW,
            "reason": (
                "Recovery case has not been attempted."
            ),
        }

    if case.get(
        "waiting_for_outcome",
        False,
    ):
        return {
            "state": STATE_WAITING,
            "reason": (
                "An action has been executed and "
                "the outcome is pending."
            ),
        }

    if case.get(
        "needs_reassessment",
        False,
    ):
        return {
            "state": STATE_REASSESS,
            "reason": (
                "Previous recovery action did not "
                "produce verified recovery."
            ),
        }

    return {
        "state": STATE_RECOVERY_READY,
        "reason": (
            "Case is ready for the next permitted "
            "recovery action."
        ),
    }


def build_recovery_path(
    failure_category,
) -> Dict:
    """
    Build the complete initial recovery pathway.

    This exposes both the permitted actions and
    the preferred recovery sequence.
    """

    category = normalize_failure_category(
        failure_category
    )

    policy = get_recovery_policy(
        category
    )

    return {
        "failure_category": category,

        "initial_action": policy[
            "initial_action"
        ],

        "permitted_actions": list(
            policy[
                "permitted_actions"
            ]
        ),

        "recovery_sequence": list(
            policy[
                "recovery_sequence"
            ]
        ),

        "policy_reason": policy[
            "reason"
        ],
    }