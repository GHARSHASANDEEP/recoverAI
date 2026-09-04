"""Durable feedback and customer recovery memory.

This is the learning boundary for production outcomes. The benchmark does not
silently treat simulated outcomes as training feedback; callers must append
verified provider outcomes explicitly.
"""

import json
from pathlib import Path


DEFAULT_FEEDBACK_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "recovery_feedback.jsonl"
)


def append_verified_outcome(
    outcome: dict,
    path: str | Path = DEFAULT_FEEDBACK_PATH,
) -> None:
    """Append one verified provider outcome for later retraining."""

    required = {
        "case_id",
        "customer_id",
        "action",
        "recovered",
        "verified_at",
    }
    missing = sorted(required.difference(outcome))
    if missing:
        raise ValueError("Verified outcome missing: " + ", ".join(missing))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(outcome, sort_keys=True) + "\n")


def load_verified_outcomes(
    path: str | Path = DEFAULT_FEEDBACK_PATH,
) -> list[dict]:
    """Load explicitly recorded provider outcomes."""

    source = Path(path)
    if not source.exists():
        return []

    outcomes = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            outcomes.append(json.loads(line))
    return outcomes


def customer_recovery_memory(
    customer_id: str,
    path: str | Path = DEFAULT_FEEDBACK_PATH,
) -> dict:
    """Summarize verified action history for one customer."""

    outcomes = [
        outcome
        for outcome in load_verified_outcomes(path)
        if str(outcome.get("customer_id")) == str(customer_id)
    ]
    recovered = [outcome for outcome in outcomes if outcome.get("recovered")]

    return {
        "customer_id": str(customer_id),
        "verified_interventions": len(outcomes),
        "successful_interventions": len(recovered),
        "recovery_rate": len(recovered) / len(outcomes) if outcomes else None,
        "last_action": outcomes[-1].get("action") if outcomes else None,
    }