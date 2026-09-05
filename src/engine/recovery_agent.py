from datetime import datetime
from src.engine.confidence import (
    assess_confidence,
)
import pandas as pd

from src.engine.decision_engine import (
    evaluate_guardrails,
)

from src.engine.executor import (
    execute_and_verify,
)

from src.engine.recovery_policy import (
    get_initial_action,
    get_permitted_actions,
    get_next_policy_action,
    get_recovery_sequence,
)

from src.engine.recovery_state_machine import (
    RecoveryStateMachine,
    ANALYZING,
    RECOVERY_READY,
    ACTION_SENT,
    WAITING,
    REASSESS,
    RECOVERED,
    ESCALATED,
    STOPPED,
)


MAX_AUTOMATED_ATTEMPTS = 3


def select_next_action(
    case: dict,
    action_scores: pd.DataFrame,
    attempted_actions: set,
    policy_action: str = None,
):
    """
    Select an action from the policy-approved candidate set.

    Without ``policy_action``, this preserves the original
    highest-ERV selection behavior for compatibility and
    isolated evaluation.

    During the actual recovery lifecycle, ``policy_action``
    is supplied by the failure-aware recovery sequence.

    ERV can evaluate the policy-approved action, but it
    cannot skip an earlier recovery stage and jump directly
    to a later action such as escalation.

    Confidence is returned as an explanatory signal and
    does not override policy.
    """

    candidates = action_scores[
        ~action_scores["action"].isin(
            attempted_actions
        )
    ].copy()

    permitted_actions = get_permitted_actions(
        case.get("failure_category")
    )

    candidates = candidates[
        candidates["action"].isin(
            permitted_actions
        )
    ].copy()

    # During the real recovery lifecycle, the policy owns
    # the recovery sequence. ERV is evaluated only for the
    # current policy-approved stage and cannot skip directly
    # to a later action such as escalation.
    if policy_action is not None:
        candidates = candidates[
            candidates["action"] == policy_action
        ].copy()

    if candidates.empty:
        return {
            "final_action": "stop",
            "decision_reason": (
                "No untried permitted recovery "
                "action remains."
            ),
            "guardrail_status": "blocked",
            "erv": 0.0,
            "recovery_probability": 0.0,
            "expected_recovery": 0.0,
            "action_cost": 0.0,
            "confidence_score": 0.0,
            "confidence_level": "low",
            "probability_confidence": 0.0,
            "action_margin": 0.0,
            "margin_confidence": 0.0,
        }

    evaluated = []

    for row in candidates.to_dict(
        orient="records"
    ):

        action = row["action"]

        case_with_erv = {
            **case,
            "erv": row["erv"],
        }

        guardrail = evaluate_guardrails(
            case_with_erv,
            action,
        )

        evaluated.append(
            {
                **row,
                "guardrail_allowed": (
                    guardrail["allowed"]
                ),
                "guardrail_reason": (
                    guardrail["reason"]
                ),
            }
        )

    evaluated_df = pd.DataFrame(
        evaluated
    )

    allowed = evaluated_df[
        evaluated_df[
            "guardrail_allowed"
        ]
    ].copy()

    if allowed.empty:
        return {
            "final_action": "stop",
            "decision_reason": (
                "No remaining permitted action "
                "passes guardrails."
            ),
            "guardrail_status": "blocked",
            "erv": 0.0,
            "recovery_probability": 0.0,
            "expected_recovery": 0.0,
            "action_cost": 0.0,
            "confidence_score": 0.0,
            "confidence_level": "low",
            "probability_confidence": 0.0,
            "action_margin": 0.0,
            "margin_confidence": 0.0,
        }

    allowed = allowed.sort_values(
        [
            "erv",
            "recovery_probability",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    best = allowed.iloc[0]

    best_probability = float(
        best["recovery_probability"]
    )

    if len(allowed) > 1:
        second_probability = float(
            allowed.iloc[1][
                "recovery_probability"
            ]
        )
    else:
        second_probability = 0.0

    confidence = assess_confidence(
        probability=best_probability,
        best_score=best_probability,
        second_score=second_probability,
        available_actions=len(allowed),
    )

    return {
        "final_action": best[
            "action"
        ],
        "decision_reason": best[
            "guardrail_reason"
        ],
        "guardrail_status": "passed",
        "erv": float(
            best["erv"]
        ),
        "recovery_probability": (
            best_probability
        ),
        "expected_recovery": float(
            best["expected_recovery"]
        ),
        "action_cost": float(
            best["action_cost"]
        ),
        "confidence_score": confidence[
            "confidence_score"
        ],
        "confidence_level": confidence[
            "confidence_level"
        ],
        "probability_confidence": confidence[
            "probability_confidence"
        ],
        "action_margin": confidence[
            "action_margin"
        ],
        "margin_confidence": confidence[
            "margin_confidence"
        ],
    }



def assess_action_confidence(
    action: str,
    action_scores: pd.DataFrame,
    permitted_actions,
) -> dict:
    """
    Measure confidence for a specific policy-selected action.

    This is observational only. It does not change the selected
    action, policy, guardrails, ERV ranking, or execution behavior.
    """

    permitted = action_scores[
        action_scores["action"].isin(
            permitted_actions
        )
    ].copy()

    selected = permitted[
        permitted["action"] == action
    ].copy()

    if selected.empty:
        return {
            "confidence_score": 0.0,
            "confidence_level": "not_evaluated",
            "probability_confidence": 0.0,
            "action_margin": 0.0,
            "margin_confidence": 0.0,
            "recovery_probability": None,
        }

    selected_probability = float(
        selected.iloc[0]["recovery_probability"]
    )

    competitors = permitted[
        permitted["action"] != action
    ].copy()

    if competitors.empty:
        second_probability = 0.0
    else:
        second_probability = float(
            competitors[
                "recovery_probability"
            ].max()
        )

    confidence = assess_confidence(
        probability=selected_probability,
        best_score=selected_probability,
        second_score=second_probability,
        available_actions=len(permitted),
    )

    return {
        **confidence,
        "recovery_probability": selected_probability,
    }

def _state_event(
    from_state: str,
    to_state: str,
    reason: str,
    **extra,
):
    """
    Build a consistent state-transition audit event.
    """

    event = {
        "timestamp": datetime.now().isoformat(),
        "event": "state_transition",
        "from_state": from_state,
        "to_state": to_state,
        "reason": reason,
    }

    event.update(extra)

    return event


def run_recovery_case(
    case: dict,
    action_scores: pd.DataFrame,
) -> dict:
    """
    Run the adaptive recovery lifecycle.

    Flow:

        NEW
          ↓
        ANALYZING
          ↓
        RECOVERY_READY
          ↓
        ACTION_SENT
          ↓
        WAITING
          ↓
        ┌───────────────┐
        │               │
     VERIFIED       NOT VERIFIED
        │               │
        ↓               ↓
     RECOVERED       REASSESS
                        ↓
                 RECOVERY_READY
                        ↓
                   NEXT ACTION

    Policy determines the recovery sequence.
    Guardrails determine whether an action is safe.
    ERV evaluates the current policy-approved action
    but cannot skip recovery stages.
    Confidence is recorded as an explanatory signal.
    """

    audit_events = []

    attempted_actions = set()

    attempt_count = 0

    total_recovered = 0.0

    current_case = dict(case)

    # --------------------------------------------------
    # State machine
    # --------------------------------------------------

    state_machine = RecoveryStateMachine(
        case_id=case["case_id"]
    )

    # NEW -> ANALYZING

    state_machine.transition(
        ANALYZING
    )

    audit_events.append(
        _state_event(
            "new",
            ANALYZING,
            "Recovery case analysis started.",
        )
    )

    # ANALYZING -> RECOVERY_READY

    state_machine.transition(
        RECOVERY_READY
    )

    audit_events.append(
        _state_event(
            ANALYZING,
            RECOVERY_READY,
            (
                "Case passed initial analysis "
                "and is ready for recovery."
            ),
        )
    )

    # --------------------------------------------------
    # Initial action
    # --------------------------------------------------

    failure_category = current_case.get(
        "failure_category"
    )

    permitted_actions = get_permitted_actions(
        failure_category
    )

    if permitted_actions:

        policy_initial_action = (
            get_initial_action(
                failure_category
            )
        )

    else:

        policy_initial_action = None

    initial_action = policy_initial_action

    if initial_action is None:

        initial_action = current_case.get(
            "final_action",
            "stop",
        )

    # Make policy decision visible to the audit trail.

    current_case[
        "final_action"
    ] = initial_action

    audit_events.append(
        {
            "timestamp": datetime.now().isoformat(),
            "event": "policy_decision",
            "failure_category": failure_category,
            "initial_action": initial_action,
            "permitted_actions": (
                permitted_actions
            ),
            "recovery_sequence": get_recovery_sequence(
                failure_category
            ),
            "reason": (
                "Initial recovery action selected "
                "from failure-aware recovery policy."
            ),
        }
    )

    # --------------------------------------------------
    # Initial-action confidence
    # --------------------------------------------------

    initial_confidence = assess_action_confidence(
        action=initial_action,
        action_scores=action_scores,
        permitted_actions=permitted_actions,
    )

    current_case.update(
        {
            "confidence_score": initial_confidence[
                "confidence_score"
            ],
            "confidence_level": initial_confidence[
                "confidence_level"
            ],
            "probability_confidence": initial_confidence[
                "probability_confidence"
            ],
            "action_margin": initial_confidence[
                "action_margin"
            ],
            "margin_confidence": initial_confidence[
                "margin_confidence"
            ],
        }
    )

    audit_events.append(
        {
            "timestamp": datetime.now().isoformat(),
            "event": "initial_confidence_evaluation",
            "action": initial_action,
            "recovery_probability": initial_confidence[
                "recovery_probability"
            ],
            "confidence_score": initial_confidence[
                "confidence_score"
            ],
            "confidence_level": initial_confidence[
                "confidence_level"
            ],
            "probability_confidence": initial_confidence[
                "probability_confidence"
            ],
            "action_margin": initial_confidence[
                "action_margin"
            ],
            "margin_confidence": initial_confidence[
                "margin_confidence"
            ],
            "reason": (
                "Confidence measured for the policy-selected "
                "initial action. It does not override the action."
            ),
        }
    )

    # --------------------------------------------------
    # Confidence-gated escalation for high-value cases.
    #
    # Low confidence on a high-value case means the model
    # is uncertain. Automated action on uncertain high-value
    # cases is riskier than escalating to a human.
    # This is the one place confidence changes behavior.
    # --------------------------------------------------

    HIGH_VALUE_THRESHOLD = 50000.0
    LOW_CONFIDENCE_THRESHOLD = 0.40

    is_high_value = float(
        current_case.get("recovery_amount", 0.0)
    ) >= HIGH_VALUE_THRESHOLD

    is_low_confidence = (
        initial_confidence["confidence_score"]
        < LOW_CONFIDENCE_THRESHOLD
    )

    if is_high_value and is_low_confidence and initial_action not in (
        "escalate", "stop"
    ):
        audit_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "confidence_gated_escalation",
                "original_action": initial_action,
                "confidence_score": initial_confidence["confidence_score"],
                "recovery_amount": current_case.get("recovery_amount"),
                "reason": (
                    "Low model confidence on a high-value case. "
                    "Automated action replaced with escalation to "
                    "reduce risk of incorrect recovery attempt."
                ),
            }
        )
        current_case["final_action"] = "escalate"
        current_case["decision_reason"] = (
            f"Confidence-gated escalation: score "
            f"{initial_confidence['confidence_score']:.3f} is below "
            f"{LOW_CONFIDENCE_THRESHOLD} threshold on a "
            f"₹{current_case.get('recovery_amount', 0):,.0f} case."
        )
    # --------------------------------------------------

    while True:

        action = current_case.get(
            "final_action",
            "stop",
        )

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if action == "stop":

            previous_state = (
                state_machine.get_state()
            )

            if previous_state != STOPPED:

                state_machine.transition(
                    STOPPED
                )

                audit_events.append(
                    _state_event(
                        previous_state,
                        STOPPED,
                        (
                            current_case.get(
                                "decision_reason",
                                "Recovery stopped.",
                            )
                        ),
                    )
                )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "stopped",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        # --------------------------------------------------
        # ESCALATE
        # --------------------------------------------------

        if action == "escalate":

            previous_state = (
                state_machine.get_state()
            )

            if previous_state != ESCALATED:

                state_machine.transition(
                    ESCALATED
                )

                audit_events.append(
                    _state_event(
                        previous_state,
                        ESCALATED,
                        (
                            current_case.get(
                                "decision_reason",
                                "Case routed to manual review.",
                            )
                        ),
                        action="escalate",
                    )
                )

            audit_events.append(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "event": "escalation",
                    "action": "escalate",
                    "reason": (
                        current_case.get(
                            "decision_reason",
                            "Case routed to manual review.",
                        )
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "escalated",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        # --------------------------------------------------
        # AUTOMATED ATTEMPT LIMIT
        # --------------------------------------------------

        if attempt_count >= (
            MAX_AUTOMATED_ATTEMPTS
        ):

            previous_state = (
                state_machine.get_state()
            )

            if previous_state != ESCALATED:

                state_machine.transition(
                    ESCALATED
                )

                audit_events.append(
                    _state_event(
                        previous_state,
                        ESCALATED,
                        (
                            "Maximum automated "
                            "attempt limit reached."
                        ),
                    )
                )

            audit_events.append(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "event": "escalation",
                    "reason": (
                        "Maximum automated "
                        "attempts reached."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "escalated",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        # --------------------------------------------------
        # EXECUTE
        # --------------------------------------------------

        attempt_count += 1

        current_case[
            "attempt_number"
        ] = attempt_count

        attempted_actions.add(
            action
        )

        audit_events.append(
            {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "event": "action_selected",
                "attempt_number": attempt_count,
                "action": action,
                "reason": current_case.get(
                    "decision_reason",
                    "",
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }
        )

        # RECOVERY_READY -> ACTION_SENT

        previous_state = (
            state_machine.get_state()
        )

        state_machine.transition(
            ACTION_SENT
        )

        audit_events.append(
            _state_event(
                previous_state,
                ACTION_SENT,
                (
                    "Approved recovery action "
                    "is being executed."
                ),
                attempt_number=attempt_count,
                action=action,
            )
        )

        # --------------------------------------------------
        # Execute + verify
        # --------------------------------------------------

        result = execute_and_verify(
            current_case
        )

        # ACTION_SENT -> WAITING

        state_machine.transition(
            WAITING
        )

        audit_events.append(
            _state_event(
                ACTION_SENT,
                WAITING,
                (
                    "Action executed; waiting for "
                    "recovery verification."
                ),
                attempt_number=attempt_count,
                action=action,
            )
        )

        recovered = bool(
            result.get(
                "verified_recovered",
                False,
            )
        )

        recovered_amount = float(
            result.get(
                "verified_amount",
                0.0,
            )
        )

        total_recovered += (
            recovered_amount
        )

        execution_status = result.get(
            "execution_status"
        )

        verification_status = result.get(
            "verification_status"
        )

        audit_events.append(
            {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "event": "action_executed",
                "attempt_number": attempt_count,
                "action": action,
                "execution_status": (
                    execution_status
                ),
                "verification_status": (
                    verification_status
                ),
                "verified_recovered": recovered,
                "verified_amount": (
                    recovered_amount
                ),
                "verification_reason": result.get(
                    "verification_reason"
                ),
            }
        )

        # --------------------------------------------------
        # SUCCESS
        # --------------------------------------------------

        if recovered:

            # WAITING -> RECOVERED

            state_machine.transition(
                RECOVERED
            )

            audit_events.append(
                _state_event(
                    WAITING,
                    RECOVERED,
                    (
                        "Payment recovery was "
                        "successfully verified."
                    ),
                    attempt_number=attempt_count,
                    action=action,
                )
            )

            audit_events.append(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "event": "recovery_verified",
                    "attempt_number": (
                        attempt_count
                    ),
                    "action": action,
                    "amount": recovered_amount,
                    "reason": (
                        "Recovery successfully "
                        "verified."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "recovered",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        # --------------------------------------------------
        # EXECUTION TERMINAL STATES
        # --------------------------------------------------

        if execution_status == "escalated":

            state_machine.transition(
                ESCALATED
            )

            audit_events.append(
                _state_event(
                    WAITING,
                    ESCALATED,
                    (
                        "Execution routed the "
                        "case to manual review."
                    ),
                    attempt_number=attempt_count,
                    action=action,
                )
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "escalated",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        if execution_status == "stopped":

            state_machine.transition(
                STOPPED
            )

            audit_events.append(
                _state_event(
                    WAITING,
                    STOPPED,
                    (
                        "Execution was stopped "
                        "by policy."
                    ),
                    attempt_number=attempt_count,
                    action=action,
                )
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "stopped",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        # --------------------------------------------------
        # FAILED ACTION
        # --------------------------------------------------

        audit_events.append(
            {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "event": "recovery_failed",
                "attempt_number": attempt_count,
                "action": action,
                "reason": (
                    "Action executed but "
                    "recovery was not verified."
                ),
            }
        )

        # WAITING -> REASSESS

        state_machine.transition(
            REASSESS
        )

        audit_events.append(
            _state_event(
                WAITING,
                REASSESS,
                (
                    "Recovery was not verified; "
                    "case requires reassessment."
                ),
                attempt_number=attempt_count,
                action=action,
            )
        )

        # --------------------------------------------------
        # Customer opt-out
        # --------------------------------------------------

        if not bool(
            current_case.get(
                "communication_opt_in",
                False,
            )
        ):

            state_machine.transition(
                STOPPED
            )

            audit_events.append(
                _state_event(
                    REASSESS,
                    STOPPED,
                    (
                        "Customer communication "
                        "opt-out."
                    ),
                )
            )

            audit_events.append(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "event": "stopping_decision",
                    "reason": (
                        "Customer communication "
                        "opt-out."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "stopped",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        # --------------------------------------------------
        # Attempt limit after failed action
        # --------------------------------------------------

        if attempt_count >= (
            MAX_AUTOMATED_ATTEMPTS
        ):

            state_machine.transition(
                ESCALATED
            )

            audit_events.append(
                _state_event(
                    REASSESS,
                    ESCALATED,
                    (
                        "Maximum automated "
                        "attempts reached."
                    ),
                )
            )

            audit_events.append(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "event": "escalation",
                    "reason": (
                        "Maximum automated "
                        "attempts reached."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "escalated",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
                "state_history": (
                    state_machine.get_history()
                ),
                "final_state": (
                    state_machine.get_state()
                ),
                "confidence_score": current_case.get(
                    "confidence_score",
                    0.0,
                ),
                "confidence_level": current_case.get(
                    "confidence_level",
                    "unknown",
                ),
            }

        # --------------------------------------------------
        # Select next policy-defined action
        # --------------------------------------------------

        state_machine.transition(
            RECOVERY_READY
        )

        audit_events.append(
            _state_event(
                REASSESS,
                RECOVERY_READY,
                (
                    "Case reassessed and ready "
                    "for the next policy-defined "
                    "recovery action."
                ),
            )
        )

        # The policy sequence determines the next stage.
        # This is deliberately separated from ERV so that
        # ERV cannot jump over a customer-resolvable step
        # and immediately escalate the case.
        policy_next_action = get_next_policy_action(
            failure_category,
            attempted_actions,
        )

        audit_events.append(
            {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "event": "policy_next_action",
                "previous_action": action,
                "next_action": policy_next_action,
                "attempted_actions": sorted(
                    attempted_actions
                ),
                "recovery_sequence": get_recovery_sequence(
                    failure_category
                ),
                "reason": (
                    "Next action selected from the "
                    "failure-aware recovery sequence. "
                    "ERV cannot skip this policy stage."
                ),
            }
        )

        # ERV/ML now evaluates only the action that the
        # recovery policy has selected for this stage.
        next_decision = select_next_action(
            current_case,
            action_scores,
            attempted_actions,
            policy_action=policy_next_action,
        )

        audit_events.append(
            {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "event": "next_action_evaluation",
                "previous_action": action,
                "policy_action": policy_next_action,
                "next_action": next_decision[
                    "final_action"
                ],
                "erv": next_decision[
                    "erv"
                ],
                "recovery_probability": (
                    next_decision[
                        "recovery_probability"
                    ]
                ),
                "expected_recovery": (
                    next_decision[
                        "expected_recovery"
                    ]
                ),
                "confidence_score": (
                    next_decision[
                        "confidence_score"
                    ]
                ),
                "confidence_level": (
                    next_decision[
                        "confidence_level"
                    ]
                ),
                "probability_confidence": (
                    next_decision[
                        "probability_confidence"
                    ]
                ),
                "action_margin": (
                    next_decision[
                        "action_margin"
                    ]
                ),
                "margin_confidence": (
                    next_decision[
                        "margin_confidence"
                    ]
                ),
                "reason": next_decision[
                    "decision_reason"
                ],
            }
        )

        current_case.update(
            next_decision
        )


def main():
    """
    This module is intentionally not a CLI runner.

    Use agent_batch.py for batch execution.
    """

    print(
        "Recovery agent module loaded."
    )


if __name__ == "__main__":
    main()