# RecoverAI - Synthetic Outcome Rules
#
# IMPORTANT:
# These probabilities define the hidden synthetic benchmark environment.
# They are NOT real Razorpay statistics and are NOT exposed to the
# RecoverAI decision engine. The agent must estimate these through ML.
#
# There is ONE authoritative probability table: HIDDEN_RECOVERY_PROBABILITIES.
# get_hidden_recovery_probability() applies segment, amount, and attempt
# adjustments on top of it. The legacy BASE_RECOVERY_PROBABILITIES table
# below is kept only for backward compatibility with older batch scripts
# and is NOT used by the live agent or unseen evaluation.

import hashlib


# Single authoritative hidden probability table used by the simulator.
HIDDEN_RECOVERY_PROBABILITIES = {
    "temporary_bank_failure": {
        "retry": 0.85,
        "reminder": 0.60,
        "escalate": 0.70,
        "stop": 0.00,
    },
    "timeout": {
        "retry": 0.78,
        "reminder": 0.55,
        "escalate": 0.65,
        "stop": 0.00,
    },
    "insufficient_funds": {
        "retry": 0.25,
        "reminder": 0.62,
        "escalate": 0.50,
        "stop": 0.00,
    },
    "authentication_failed": {
        "retry": 0.20,
        "reminder": 0.58,
        "escalate": 0.45,
        "stop": 0.00,
    },
    "expired_instrument": {
        "retry": 0.08,
        "reminder": 0.52,
        "escalate": 0.48,
        "stop": 0.00,
    },
    "blocked_instrument": {
        "retry": 0.05,
        "reminder": 0.35,
        "escalate": 0.40,
        "stop": 0.00,
    },
    "risk_decline": {
        "retry": 0.02,
        "reminder": 0.15,
        "escalate": 0.10,
        "stop": 0.00,
    },
    "limit_exceeded": {
        "retry": 0.12,
        "reminder": 0.40,
        "escalate": 0.35,
        "stop": 0.00,
    },
    "unknown_failure": {
        "retry": 0.35,
        "reminder": 0.45,
        "escalate": 0.40,
        "stop": 0.00,
    },
}

# Kept for backward compatibility only. Not used by the live agent.
BASE_RECOVERY_PROBABILITIES = HIDDEN_RECOVERY_PROBABILITIES


def get_hidden_recovery_probability(
    failure_category: str | None,
    action: str,
    customer_segment: str,
    amount: float,
    attempt_number: int = 1,
) -> float:
    """
    Return the hidden synthetic ground-truth probability.

    This value represents the simulated world, NOT a model
    prediction. It must never be passed directly to the
    decision engine.
    """

    category = (
        failure_category
        if failure_category in HIDDEN_RECOVERY_PROBABILITIES
        else "unknown_failure"
    )

    probability = HIDDEN_RECOVERY_PROBABILITIES[category].get(action, 0.0)

    segment_adjustment = {
        "high_value": 0.05,
        "regular": 0.00,
        "new": -0.05,
    }
    probability += segment_adjustment.get(customer_segment, 0.0)

    if amount >= 100000:
        probability -= 0.05
    elif amount >= 25000:
        probability -= 0.02

    if attempt_number >= 2:
        probability -= 0.10
    if attempt_number >= 3:
        probability -= 0.15

    return round(max(0.0, min(1.0, probability)), 4)


ACTION_COSTS = {
    "retry": 2.00,
    "reminder": 5.00,
    "escalate": 25.00,
    "stop": 0.00,
}


AMOUNT_BUCKETS = [
    (0, 5000, "0_5k"),
    (5000, 25000, "5k_25k"),
    (25000, 100000, "25k_100k"),
    (100000, float("inf"), "100k_plus"),
]


def get_amount_bucket(
    amount: float,
) -> str:
    """Map an amount to a stable benchmark bucket."""

    amount = float(amount)

    for lower, upper, bucket in AMOUNT_BUCKETS:

        if lower <= amount < upper:
            return bucket

    return "100k_plus"


def get_action_cost(
    action: str,
) -> float:
    """Return the benchmark cost of an action."""

    return ACTION_COSTS.get(
        action,
        0.0,
    )


def stable_random_value(
    case_id: str,
) -> float:
    """
    Produce a deterministic pseudo-random value
    between 0 and 1 for a case.

    This makes benchmark outcomes reproducible.
    """

    digest = hashlib.sha256(
        case_id.encode("utf-8")
    ).hexdigest()

    integer_value = int(
        digest[:16],
        16,
    )

    return integer_value / float(16**16)

