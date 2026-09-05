"""Explainable AI judgment over scored recovery actions.

This layer synthesizes model outputs and case context into a human-readable
judgment. It is advisory: deterministic policy and guardrails still authorize
all financial actions.
"""

import pandas as pd

from src.engine.confidence import assess_confidence
from src.engine.decision_engine import evaluate_guardrails
from src.engine.recovery_policy import get_permitted_actions


def judge_recovery_case(
    case: dict,
    action_scores: pd.DataFrame,
    policy_action: str,
) -> dict:
    """Return an evidence-backed judgment without authorizing an action."""

    if action_scores.empty:
        return {
            "verdict": "escalate",
            "recommended_action": "escalate",
            "confidence_score": 0.0,
            "confidence_level": "low",
            "evidence": ["No model action scores were available."],
            "blocked_actions": [],
            "counterfactual": "Collect a valid model score before automating recovery.",
        }

    evaluated = []
    for row in action_scores.to_dict(orient="records"):
        action = str(row["action"])
        guardrail = evaluate_guardrails({**case, "erv": row.get("erv", 0.0)}, action)
        evaluated.append({
            **row,
            "guardrail_allowed": bool(guardrail["allowed"]),
            "guardrail_reason": guardrail["reason"],
        })

    scored = pd.DataFrame(evaluated)
    permitted_actions = set(get_permitted_actions(case.get("failure_category")))
    scored["policy_allowed"] = scored["action"].isin(permitted_actions)
    allowed = scored[
        scored["guardrail_allowed"] & scored["policy_allowed"]
    ].sort_values(
        ["erv", "recovery_probability"], ascending=[False, False]
    )
    policy_rows = scored[scored["action"] == policy_action]
    policy_row = policy_rows.iloc[0] if not policy_rows.empty else None

    if allowed.empty:
        return {
            "verdict": "stop",
            "recommended_action": "stop",
            "confidence_score": 1.0,
            "confidence_level": "high",
            "evidence": ["Every candidate action failed deterministic guardrails."],
            "blocked_actions": scored.loc[~scored["guardrail_allowed"], "action"].tolist(),
            "counterfactual": "No automated action is safe; escalate or stop under policy.",
        }

    best = allowed.iloc[0]
    second_probability = float(allowed.iloc[1]["recovery_probability"]) if len(allowed) > 1 else 0.0
    confidence = assess_confidence(
        probability=float(best["recovery_probability"]),
        best_score=float(best["recovery_probability"]),
        second_score=second_probability,
        available_actions=len(allowed),
    )

    evidence = [
        f"Model ranks {best['action']} highest among guardrail-allowed actions by expected recovery value.",
        f"Estimated recovery probability is {float(best['recovery_probability']):.1%}.",
        f"The policy sequence starts with {policy_action} for {case.get('failure_category', 'unknown_failure')}.",
    ]
    if policy_row is not None and not bool(policy_row["guardrail_allowed"]):
        evidence.append(f"Policy action {policy_action} is blocked: {policy_row['guardrail_reason']}")
    if case.get("communication_opt_in") is False:
        evidence.append("Customer communication consent is unavailable, so reminders are unsafe.")
    if case.get("failure_category") in {"risk_decline", "blocked_instrument"}:
        evidence.append("This failure category is unsafe for automatic retry.")

    blocked = scored.loc[
        ~(scored["guardrail_allowed"] & scored["policy_allowed"]),
        "action",
    ].tolist()
    counterfactual = (
        f"If {best['action']} becomes unavailable, reassess the next permitted action; "
        "do not bypass the policy sequence."
    )

    return {
        "verdict": "automate" if best["action"] != "escalate" else "escalate",
        "recommended_action": str(best["action"]),
        "confidence_score": float(confidence["confidence_score"]),
        "confidence_level": confidence["confidence_level"],
        "evidence": evidence,
        "blocked_actions": blocked,
        "counterfactual": counterfactual,
    }
