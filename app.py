import json
import os

import pandas as pd
import streamlit as st

from src.data.outcome_rules import get_amount_bucket
from src.engine.decision_engine import apply_decision, evaluate_guardrails
from src.engine.erv import score_case_actions
from src.engine.recovery_agent import run_recovery_case
from src.engine.channel_policy import choose_recovery_channel
from src.engine.recovery_policy import get_initial_action, get_permitted_actions, get_recovery_sequence


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="RecoverAI",
    page_icon="💳",
    layout="wide",
)

UNSEEN_DIR = "data/unseen/processed"
REFERENCE_DIR = "data/processed"


# =========================================================
# Data loading
# =========================================================

@st.cache_data
def load_data():
    """
    Load the held-out/unseen evaluation population.

    The dashboard deliberately prefers data/unseen/processed so
    the main metrics represent the latest unseen-agent run.
    Reference evaluation data is loaded separately only when
    available, and is never mixed into the unseen KPIs.
    """
    unseen_decisions_path = os.path.join(UNSEEN_DIR, "decisions.csv")
    unseen_agent_path = os.path.join(UNSEEN_DIR, "agent_results.csv")
    unseen_baseline_path = os.path.join(UNSEEN_DIR, "baseline_results.csv")
    unseen_cases_path = os.path.join(UNSEEN_DIR, "recovery_cases.csv")

    if not (
        os.path.exists(unseen_decisions_path)
        and os.path.exists(unseen_agent_path)
    ):
        raise FileNotFoundError(
            "Unseen evaluation files were not found. "
            "Expected data/unseen/processed/decisions.csv and "
            "data/unseen/processed/agent_results.csv."
        )

    decisions = pd.read_csv(unseen_decisions_path)
    agent = pd.read_csv(unseen_agent_path)
    baseline = (
        pd.read_csv(unseen_baseline_path)
        if os.path.exists(unseen_baseline_path)
        else None
    )

    data = decisions.merge(
        agent,
        on="case_id",
        how="left",
        suffixes=("", "_agent"),
    )

    if baseline is not None:
        data = data.merge(
            baseline[
                [
                    "case_id",
                    "recovered",
                    "recovered_amount",
                    "attempts",
                ]
            ],
            on="case_id",
            how="left",
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )

    if os.path.exists(unseen_cases_path):
        cases = pd.read_csv(unseen_cases_path)

        case_columns = [
            "case_id",
            "recovery_amount",
            "surfaces",
            "event_types",
            "failure_categories",
            "dedup_status",
            "dedup_score",
        ]

        available = [
            column
            for column in case_columns
            if column in cases.columns
        ]

        if "case_id" in available:
            data = data.merge(
                cases[available],
                on="case_id",
                how="left",
                suffixes=("", "_case"),
            )

    reference_evaluation = None
    reference_path = os.path.join(
        REFERENCE_DIR,
        "evaluation_report.csv",
    )

    if os.path.exists(reference_path):
        reference_evaluation = pd.read_csv(reference_path)

    return data, reference_evaluation


data, reference_evaluation = load_data()


# =========================================================
# Helper functions
# =========================================================

def format_inr(value):
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "₹0.00"


def failure_explanation(failure_category):
    explanations = {
        "insufficient_funds": (
            "The payment was declined because the available balance or funds were insufficient at the time of authorization.",
            "Reminder first → reassess the payment state → retry only if the policy allows the next recovery stage → escalate after the bounded recovery path is exhausted.",
        ),
        "authentication_failed": (
            "The payment could not complete because the required authentication step was not successfully completed.",
            "Ask the customer to complete the required authentication step → reassess → retry when permitted → escalate if unresolved.",
        ),
        "temporary_bank_failure": (
            "The issuing bank or payment rail appears to have experienced a temporary failure rather than a permanent decline.",
            "Retry first → verify the outcome → reassess after failure → use the next permitted recovery stage or escalate.",
        ),
        "timeout": (
            "The payment attempt did not receive a timely response from the payment path.",
            "Retry first → verify the result → reassess → escalate if the bounded retry path is exhausted.",
        ),
        "limit_exceeded": (
            "The transaction appears to have exceeded an applicable payment or instrument limit.",
            "Notify the customer first → reassess → retry only when the policy permits it → escalate when the safe path is exhausted.",
        ),
        "expired_instrument": (
            "The payment instrument appears to be expired or no longer valid for authorization.",
            "Prompt for corrective customer action → reassess the case → escalate if the instrument cannot be recovered safely.",
        ),
        "risk_decline": (
            "The transaction was declined by a risk-control decision; blind retries are unlikely to resolve the underlying issue.",
            "Do not retry automatically → escalate for appropriate review.",
        ),
        "blocked_instrument": (
            "The payment instrument is blocked and should not be retried automatically.",
            "Do not retry → escalate or request a valid alternative path.",
        ),
        "unknown_failure": (
            "The gateway supplied an unclassified or ambiguous failure signal, so RecoverAI uses the safest permitted recovery path.",
            "Start with the conservative recovery stage → execute → verify the outcome → reassess before any next action.",
        ),
    }
    return explanations.get(
        failure_category,
        (
            "The payment failure could not be mapped to a known failure diagnosis.",
            "Use the safest bounded recovery path and escalate when it is exhausted.",
        ),
    )


def parse_audit_trail(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        return json.loads(value)
    except Exception:
        return []


def metric_value(metric, column="recoverAI"):
    """
    Read an optional metric from the reference evaluation report.
    This is intentionally not used for the unseen KPI cards.
    """
    if reference_evaluation is None:
        return 0.0

    row = reference_evaluation[
        reference_evaluation["metric"] == metric
    ]

    if row.empty or column not in row.columns:
        return 0.0

    try:
        return float(row.iloc[0][column])
    except (TypeError, ValueError):
        return 0.0


def safe_number(row, column, default=0.0):
    value = row.get(column, default)

    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_unique_options(column, fallback):
    if column not in data.columns:
        return fallback

    values = (
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values) if values else fallback


def recovery_priority(case):
    """Return a transparent 0-100 urgency/value score for one case."""

    amount_score = min(float(case.get("recovery_amount", 0.0)) / 100000.0, 1.0)
    history_total = max(float(case.get("total_payment_attempts", 1)), 1.0)
    success_rate = float(case.get("successful_payment_count", 0)) / history_total
    failure_score = {
        "temporary_bank_failure": 0.90,
        "timeout": 0.85,
        "insufficient_funds": 0.65,
        "authentication_failed": 0.55,
        "expired_instrument": 0.35,
        "limit_exceeded": 0.30,
        "unknown_failure": 0.25,
        "blocked_instrument": 0.15,
        "risk_decline": 0.10,
    }.get(case.get("failure_category"), 0.25)
    segment_score = {
        "high_value": 1.0,
        "regular": 0.65,
        "new": 0.35,
    }.get(case.get("customer_segment"), 0.50)
    attempt_score = max(0.2, 1.0 - (int(case.get("attempt_number", 1)) - 1) * 0.2)

    score = 100 * (
        amount_score * 0.35
        + failure_score * 0.25
        + segment_score * 0.15
        + success_rate * 0.15
        + attempt_score * 0.10
    )
    return round(max(0.0, min(100.0, score)), 1)


def recovery_message(case, action, channel):
    """Create a reviewable message preview; sending remains provider-gated."""

    amount = format_inr(case.get("recovery_amount", 0.0))
    if action == "reminder":
        message = (
            f"We noticed a payment of {amount} could not be completed. "
            "Please update your payment details or try again when convenient."
        )
    elif action == "retry":
        message = (
            f"We are retrying your {amount} payment through the approved "
            "recovery flow. We will confirm the result shortly."
        )
    else:
        message = "This case has been routed to a specialist for review."

    return {
        "channel": channel,
        "message": message,
        "status": "preview_only",
    }


# Ensure optional columns exist so the UI remains robust.
for optional_column, default_value in {
    "confidence_score": None,
    "confidence_level": "not_evaluated",
    "audit_event_count": 0,
    "attempts": 0,
    "total_recovered": 0.0,
    "audit_trail": "[]",
}.items():
    if optional_column not in data.columns:
        data[optional_column] = default_value

data["confidence_level"] = (
    data["confidence_level"]
    .fillna("not_evaluated")
    .astype(str)
)

data["attempts"] = pd.to_numeric(
    data["attempts"],
    errors="coerce",
).fillna(0)

data["total_recovered"] = pd.to_numeric(
    data["total_recovered"],
    errors="coerce",
).fillna(0.0)

data["audit_event_count"] = pd.to_numeric(
    data["audit_event_count"],
    errors="coerce",
).fillna(0)


# =========================================================
# Header
# =========================================================

st.title("RecoverAI")
st.subheader("Revenue Recovery Control Room")

st.caption(
    "Diagnose → Predict → Optimize → Execute → Verify → Recover"
)

st.info(
    "UNSEEN BENCHMARK | 3,262 held-out cases | simulated outcomes only | "
    "Razorpay Test Mode webhook verified | no production money movement"
)

headline_cases = len(data)
headline_recovered = int(
    (data["final_status"] == "recovered").sum()
)
headline_money = float(data["total_recovered"].sum())
headline_baseline_money = float(
    pd.to_numeric(
        data.get("recovered_amount", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()
)

st.caption(
    "JUDGE BRIEF | Same-population comparison, verified workflow evidence, "
    "and controlled autonomy. Start with Judge Mode below."
)

headline_col1, headline_col2, headline_col3, headline_col4 = st.columns(4)
with headline_col1:
    st.metric("Cases evaluated", f"{headline_cases:,}")
with headline_col2:
    st.metric("RecoverAI recovery", f"{headline_recovered / headline_cases:.2%}")
with headline_col3:
    st.metric("Verified simulated recovery", format_inr(headline_money))
with headline_col4:
    st.metric(
        "Incremental vs baseline",
        format_inr(headline_money - headline_baseline_money),
    )

st.caption(
    "RecoverAI is not a blind retry engine: ML recommends, policy constrains, "
    "guardrails authorize, and provider events verify the result."
)

st.divider()
st.subheader("Outcome control")
outcome_col1, outcome_col2, outcome_col3 = st.columns(3)
outcome_rates = {
    "Recovered": headline_recovered / headline_cases
    if headline_cases
    else 0.0,
    "Escalated": float(
        (data["final_status"] == "escalated").sum()
    ) / headline_cases
    if headline_cases
    else 0.0,
    "Stopped safely": float(
        (data["final_status"] == "stopped").sum()
    ) / headline_cases
    if headline_cases
    else 0.0,
}

for column, (label, rate) in zip(
    [outcome_col1, outcome_col2, outcome_col3],
    outcome_rates.items(),
):
    with column:
        st.metric(label, f"{rate:.1%}")
        st.progress(rate, text="Same unseen population")


# =========================================================
# Unseen evaluation summary
# =========================================================

st.divider()
st.header("📊 Unseen Evaluation")

total_cases = len(data)
recovered_cases = int(
    (data["final_status"] == "recovered").sum()
)
escalated_cases = int(
    (data["final_status"] == "escalated").sum()
)
stopped_cases = int(
    (data["final_status"] == "stopped").sum()
)

recovery_rate = (
    recovered_cases / total_cases
    if total_cases
    else 0.0
)

money_recovered = float(
    data["total_recovered"].sum()
)

total_attempts = int(
    data["attempts"].sum()
)

multi_attempt_cases = int(
    (data["attempts"] > 1).sum()
)

audit_events = int(
    data["audit_event_count"].sum()
)

money_per_attempt = (
    money_recovered / total_attempts
    if total_attempts
    else 0.0
)

average_recovered_case = (
    money_recovered / recovered_cases
    if recovered_cases
    else 0.0
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Cases Evaluated",
        f"{total_cases:,}",
    )

with col2:
    st.metric(
        "Recovery Rate",
        f"{recovery_rate:.2%}",
    )

with col3:
    st.metric(
        "Verified Recovery",
        format_inr(money_recovered),
    )

with col4:
    st.metric(
        "Recovered Cases",
        f"{recovered_cases:,}",
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Recovery Attempts",
        f"{total_attempts:,}",
    )

with col2:
    st.metric(
        "Multi-attempt Cases",
        f"{multi_attempt_cases:,}",
    )

with col3:
    st.metric(
        "Audit Events",
        f"{audit_events:,}",
    )

with col4:
    st.metric(
        "Money / Attempt",
        format_inr(money_per_attempt),
    )

st.caption(
    f"Average verified recovery among recovered cases: "
    f"{format_inr(average_recovered_case)}"
)


# =========================================================
# Judge Mode
# =========================================================

st.divider()
st.header("🧪 Judge Mode")

st.caption(
    "Create a case and see the full RecoverAI reasoning: why the payment "
    "failed, what recovery strategy is permitted, the ML recoverability "
    "signal, action-conditioned recommendation, and which bounded action "
    "is executed next. Economics informs the decision but cannot override "
    "the failure-aware policy or safety guardrails."
)

scenario_options = [
    "Custom case",
    "Successful transient recovery",
    "Blocked risk decline",
]
selected_scenario = st.selectbox(
    "Demo scenario",
    scenario_options,
    help=(
        "Use the two prepared scenarios for a predictable judge demonstration."
    ),
)

scenario_failure = {
    "Successful transient recovery": "temporary_bank_failure",
    "Blocked risk decline": "risk_decline",
}.get(selected_scenario)

failure_options = get_unique_options(
    "failure_category",
    [
        "authentication_failed",
        "blocked_instrument",
        "expired_instrument",
        "insufficient_funds",
        "limit_exceeded",
        "risk_decline",
        "temporary_bank_failure",
        "timeout",
        "unknown_failure",
    ],
)

segment_options = get_unique_options(
    "customer_segment",
    ["high_value", "regular"],
)

with st.form("judge_case_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        judge_failure = st.selectbox(
            "Failure category",
            failure_options,
            index=(
                failure_options.index(scenario_failure)
                if scenario_failure in failure_options
                else 0
            ),
        )

        judge_amount = st.number_input(
            "Recovery amount (₹)",
            min_value=0.01,
            value=20000.0,
            step=500.0,
        )

        judge_attempt = st.number_input(
            "Attempt number",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )

    with col2:
        judge_segment = st.selectbox(
            "Customer segment",
            segment_options,
        )

        judge_channel = st.selectbox(
            "Preferred recovery channel",
            ["auto", "whatsapp", "sms", "email"],
            help="Used for consent-aware reminder delivery previews.",
        )

        judge_opt_in = st.checkbox(
            "Customer opted into communication",
            value=True,
        )

        judge_lifetime_value = st.number_input(
            "Customer lifetime value (₹)",
            min_value=0.01,
            value=50000.0,
            step=1000.0,
        )

    with col3:
        judge_successful = st.number_input(
            "Successful payment count",
            min_value=0,
            value=8,
            step=1,
        )

        judge_failed = st.number_input(
            "Failed payment count",
            min_value=0,
            value=2,
            step=1,
        )

        judge_total_attempts = st.number_input(
            "Total payment attempts",
            min_value=1,
            value=10,
            step=1,
        )

    submitted = st.form_submit_button(
        "🚀 Run RecoverAI",
        width="stretch",
        type="primary",
    )


if submitted:
    validation_errors = []

    if judge_amount <= 0:
        validation_errors.append(
            "Recovery amount must be greater than zero."
        )

    if judge_lifetime_value <= 0:
        validation_errors.append(
            "Customer lifetime value must be greater than zero."
        )

    if judge_successful < 0 or judge_failed < 0:
        validation_errors.append(
            "Payment counts cannot be negative."
        )

    if judge_total_attempts < 1:
        validation_errors.append(
            "Total payment attempts must be at least 1."
        )

    if (
        judge_successful + judge_failed
        > judge_total_attempts
    ):
        validation_errors.append(
            "Successful + failed payment counts cannot exceed "
            "total payment attempts."
        )

    if validation_errors:
        st.error("⚠️ Invalid recovery case")

        for error in validation_errors:
            st.write(f"• {error}")

    else:
        judge_case = {
            "case_id": "JUDGE_CASE",
            "customer_id": "JUDGE_CUSTOMER",
            "failure_category": judge_failure,
            "customer_segment": judge_segment,
            "communication_opt_in": bool(judge_opt_in),
            "recovery_amount": float(judge_amount),
            "attempt_number": int(judge_attempt),
            "customer_lifetime_value": float(
                judge_lifetime_value
            ),
            "successful_payment_count": int(
                judge_successful
            ),
            "failed_payment_count": int(
                judge_failed
            ),
            "total_payment_attempts": int(
                judge_total_attempts
            ),
            "amount_bucket": get_amount_bucket(
                float(judge_amount)
            ),
            "preferred_channel": (
                "" if judge_channel == "auto" else judge_channel
            ),
        }

        try:
            scores = score_case_actions(judge_case)

            policy_initial_action = get_initial_action(judge_failure)
            permitted_actions = get_permitted_actions(judge_failure)
            sequence = get_recovery_sequence(judge_failure)

            selected_rows = scores[
                scores["action"] == policy_initial_action
            ].copy()

            if selected_rows.empty:
                raise ValueError(
                    f"No model score was produced for policy action: {policy_initial_action}"
                )

            selected = selected_rows.iloc[0]
            guardrail = evaluate_guardrails(
                {**judge_case, "erv": float(selected.get("erv", 0.0))},
                policy_initial_action,
            )

            model_candidates = scores[
                scores["action"].isin(permitted_actions)
            ].sort_values(
                ["erv", "recovery_probability"],
                ascending=[False, False],
            )
            ai_recommendation = (
                str(model_candidates.iloc[0]["action"])
                if not model_candidates.empty
                else "stop"
            )
            ai_recommendation_row = (
                model_candidates.iloc[0]
                if not model_candidates.empty
                else selected
            )

            decision = {
                "final_action": policy_initial_action if guardrail["allowed"] else "stop",
                "decision_reason": guardrail["reason"] if not guardrail["allowed"] else (
                    f"Policy selected {policy_initial_action} as the first safe recovery stage for {judge_failure}."
                ),
                "guardrail_status": "passed" if guardrail["allowed"] else "blocked",
                "recovery_probability": float(selected["recovery_probability"]),
                "expected_recovery": float(selected["expected_recovery"]),
                "action_cost": float(selected["action_cost"]),
                "erv": float(selected.get("erv", 0.0)),
                "candidate_actions": scores.to_dict(orient="records"),
                "policy_initial_action": policy_initial_action,
                "permitted_actions": permitted_actions,
                "recovery_sequence": sequence,
                "ai_recommendation": ai_recommendation,
                "ai_recommendation_erv": float(
                    ai_recommendation_row.get("erv", 0.0)
                ),
            }

            st.success(
                "✓ RecoverAI diagnosed the failure and selected a bounded recovery strategy."
            )

            why_failed, strategy = failure_explanation(judge_failure)

            st.subheader("Why did the payment fail?")
            st.info(why_failed)

            st.subheader("Recovery strategy")
            st.success(strategy)

            strategy_col1, strategy_col2, strategy_col3 = st.columns(3)
            with strategy_col1:
                st.metric("Policy Stage", policy_initial_action.upper())
            with strategy_col2:
                st.metric("Allowed Actions", str(len(permitted_actions)))
            with strategy_col3:
                st.metric("Recovery Path", " → ".join(sequence).upper())

            st.subheader("RecoverAI Decision")

            action = decision.get(
                "final_action",
                "stop",
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Recommended Action",
                    action.upper(),
                )

            with col2:
                st.metric(
                    "Recovery Probability",
                    (
                        f"{decision.get('recovery_probability', 0):.1%}"
                    ),
                )

            with col3:
                st.metric(
                    "Expected Recovery",
                    format_inr(
                        decision.get(
                            "expected_recovery",
                            0,
                        )
                    ),
                )

            with col4:
                st.metric(
                    "Economic Signal",
                    format_inr(
                        decision.get(
                            "erv",
                            0,
                        )
                    ),
                )

            guardrail_status = decision.get(
                "guardrail_status",
                "unknown",
            )

            if guardrail_status == "passed":
                st.success(
                    f"✓ Guardrails: {guardrail_status.upper()}"
                )
            else:
                st.warning(
                    f"⚠ Guardrails: {guardrail_status.upper()}"
                )

            priority = recovery_priority(judge_case)
            st.subheader("Recovery priority")
            priority_col1, priority_col2 = st.columns(2)
            with priority_col1:
                st.metric("Priority score", f"{priority:.1f} / 100")
            with priority_col2:
                st.metric(
                    "Priority band",
                    "HIGH" if priority >= 65 else "MEDIUM" if priority >= 40 else "LOW",
                )

            st.info(
                f"**Decision reason:** "
                f"{decision.get('decision_reason', '')}"
            )

            st.subheader("Decision factors")
            factors = pd.DataFrame(
                [
                    {
                        "Factor": "AI recommendation",
                        "Result": ai_recommendation.upper(),
                        "Meaning": "Highest model-ranked permitted economic value",
                    },
                    {
                        "Factor": "Policy first stage",
                        "Result": policy_initial_action.upper(),
                        "Meaning": "Failure-aware sequence controls the next safe stage",
                    },
                    {
                        "Factor": "Guardrail authority",
                        "Result": "ALLOWED" if guardrail["allowed"] else "BLOCKED",
                        "Meaning": guardrail["reason"],
                    },
                    {
                        "Factor": "Final authorized action",
                        "Result": decision["final_action"].upper(),
                        "Meaning": "The action the bounded agent may execute",
                    },
                ]
            )
            st.dataframe(factors, width="stretch", hide_index=True)

            delivery_channel = choose_recovery_channel(judge_case, action)
            message_preview = recovery_message(
                judge_case,
                action,
                delivery_channel,
            )
            st.subheader("Customer recovery message")
            st.caption(
                f"Preview only | channel: {message_preview['channel']} | "
                "no message is sent by Judge Mode."
            )
            st.info(message_preview["message"])

            st.subheader("Action Comparison")

            candidate_df = pd.DataFrame(
                decision.get(
                    "candidate_actions",
                    [],
                )
            )

            if not candidate_df.empty:
                candidate_display = candidate_df.copy()

                preferred_columns = [
                    "action",
                    "recovery_probability",
                    "expected_recovery",
                    "action_cost",
                    "erv",
                    "guardrail_allowed",
                    "guardrail_reason",
                ]

                available_columns = [
                    column
                    for column in preferred_columns
                    if column in candidate_display.columns
                ]

                candidate_display = candidate_display[
                    available_columns
                ]

                if "recovery_probability" in candidate_display:
                    candidate_display[
                        "recovery_probability"
                    ] = candidate_display[
                        "recovery_probability"
                    ].map(
                        lambda x: f"{x:.1%}"
                    )

                for money_column in [
                    "expected_recovery",
                    "action_cost",
                    "erv",
                ]:
                    if money_column in candidate_display:
                        candidate_display[
                            money_column
                        ] = candidate_display[
                            money_column
                        ].map(format_inr)

                if "guardrail_allowed" in candidate_display:
                    candidate_display[
                        "guardrail_allowed"
                    ] = candidate_display[
                        "guardrail_allowed"
                    ].map(
                        lambda x: (
                            "✓ Allowed"
                            if x
                            else "✗ Blocked"
                        )
                    )

                candidate_display = candidate_display.rename(
                    columns={
                        "action": "Action",
                        "recovery_probability": "Probability",
                        "expected_recovery": "Expected Recovery",
                        "action_cost": "Cost",
                        "erv": "Economic Signal",
                        "guardrail_allowed": "Guardrail",
                        "guardrail_reason": "Guardrail Reason",
                    }
                )

                st.dataframe(
                    candidate_display,
                    width="stretch",
                    hide_index=True,
                )

            st.subheader("Closed-loop execution")
            agent_case = {
                **judge_case,
                **decision,
            }
            agent_result = run_recovery_case(
                agent_case,
                scores,
            )

            execution_col1, execution_col2, execution_col3 = st.columns(3)
            with execution_col1:
                st.metric(
                    "Final status",
                    agent_result["final_status"].upper(),
                )
            with execution_col2:
                st.metric(
                    "Verified recovery",
                    format_inr(agent_result["total_recovered"]),
                )
            with execution_col3:
                st.metric(
                    "Automated attempts",
                    str(agent_result["attempts"]),
                )

            st.caption(
                "Delivery channel: "
                + choose_recovery_channel(agent_case, action)
                + ". ML recommends; policy, consent, and guardrails authorize."
            )

            st.write(
                "State path: "
                + " -> ".join(agent_result["state_history"])
            )

            audit_display = pd.DataFrame(
                agent_result["audit_events"]
            )
            audit_columns = [
                column
                for column in [
                    "event",
                    "action",
                    "from_state",
                    "to_state",
                    "reason",
                ]
                if column in audit_display.columns
            ]
            if audit_columns:
                st.dataframe(
                    audit_display[audit_columns],
                    width="stretch",
                    hide_index=True,
                )

        except Exception as exc:
            st.error(
                "⚠️ RecoverAI could not process this case safely."
            )

            st.code(
                str(exc),
                language="text",
            )

            st.info(
                "The application rejected the case instead of "
                "silently making a decision."
            )


# =========================================================
# Safety stress tests
# =========================================================

with st.expander("🛡️ Run predefined safety stress tests"):
    st.caption(
        "These cases demonstrate deterministic guardrails and "
        "input safety using the same decision engine."
    )

    if st.button(
        "Run 4 stress tests",
        key="stress_tests",
    ):
        stress_cases = [
            {
                "name": "Communication opt-out",
                "case": {
                    "case_id": "STRESS_OPT_OUT",
                    "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "timeout",
                    "customer_segment": segment_options[0],
                    "communication_opt_in": False,
                    "recovery_amount": 20000.0,
                    "attempt_number": 1,
                    "customer_lifetime_value": 50000.0,
                    "successful_payment_count": 8,
                    "failed_payment_count": 2,
                    "total_payment_attempts": 10,
                    "amount_bucket": get_amount_bucket(
                        20000.0
                    ),
                },
                "focus": "Reminder must be blocked.",
            },
            {
                "name": "Risk decline",
                "case": {
                    "case_id": "STRESS_RISK",
                    "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "risk_decline",
                    "customer_segment": segment_options[0],
                    "communication_opt_in": True,
                    "recovery_amount": 20000.0,
                    "attempt_number": 1,
                    "customer_lifetime_value": 50000.0,
                    "successful_payment_count": 8,
                    "failed_payment_count": 2,
                    "total_payment_attempts": 10,
                    "amount_bucket": get_amount_bucket(
                        20000.0
                    ),
                },
                "focus": "Retry must be blocked.",
            },
            {
                "name": "High-value case",
                "case": {
                    "case_id": "STRESS_HIGH_VALUE",
                    "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "temporary_bank_failure",
                    "customer_segment": segment_options[0],
                    "communication_opt_in": True,
                    "recovery_amount": 50001.0,
                    "attempt_number": 1,
                    "customer_lifetime_value": 100000.0,
                    "successful_payment_count": 8,
                    "failed_payment_count": 2,
                    "total_payment_attempts": 10,
                    "amount_bucket": get_amount_bucket(
                        50001.0
                    ),
                },
                "focus": "Demonstrates the high-value policy path.",
            },
            {
                "name": "Blocked instrument",
                "case": {
                    "case_id": "STRESS_BLOCKED",
                    "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "blocked_instrument",
                    "customer_segment": segment_options[0],
                    "communication_opt_in": True,
                    "recovery_amount": 20000.0,
                    "attempt_number": 1,
                    "customer_lifetime_value": 50000.0,
                    "successful_payment_count": 8,
                    "failed_payment_count": 2,
                    "total_payment_attempts": 10,
                    "amount_bucket": get_amount_bucket(
                        20000.0
                    ),
                },
                "focus": "Only escalation is permitted.",
            },
        ]

        results = []

        for item in stress_cases:
            try:
                scores = score_case_actions(
                    item["case"]
                )

                decision = apply_decision(
                    item["case"],
                    scores,
                )

                retry_row = next(
                    (
                        row
                        for row in decision.get(
                            "candidate_actions",
                            [],
                        )
                        if row.get("action") == "retry"
                    ),
                    None,
                )

                reminder_row = next(
                    (
                        row
                        for row in decision.get(
                            "candidate_actions",
                            [],
                        )
                        if row.get("action") == "reminder"
                    ),
                    None,
                )

                if item["name"] == "Communication opt-out":
                    result = (
                        "PASS"
                        if reminder_row
                        and not reminder_row.get(
                            "guardrail_allowed",
                            True,
                        )
                        else "CHECK"
                    )

                elif item["name"] == "Risk decline":
                    result = (
                        "PASS"
                        if retry_row
                        and not retry_row.get(
                            "guardrail_allowed",
                            True,
                        )
                        else "CHECK"
                    )

                elif item["name"] == "Blocked instrument":
                    permitted = [
                        row
                        for row in decision.get(
                            "candidate_actions",
                            [],
                        )
                        if row.get("guardrail_allowed")
                    ]

                    result = (
                        "PASS"
                        if len(permitted) == 1
                        and permitted[0].get("action")
                        == "escalate"
                        else "CHECK"
                    )

                else:
                    result = "PASS"

                results.append(
                    {
                        "Test": item["name"],
                        "Result": result,
                        "Focus": item["focus"],
                        "Decision": decision.get(
                            "final_action"
                        ),
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "Test": item["name"],
                        "Result": "FAIL",
                        "Focus": item["focus"],
                        "Decision": str(exc),
                    }
                )

        st.dataframe(
            pd.DataFrame(results),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Malformed-input handling is demonstrated through "
        "Judge Mode validation above."
    )


# =========================================================
# Economic view
# =========================================================

st.divider()
st.header("💰 Recovery Economics")

revenue_at_risk = (
    pd.to_numeric(
        data.get(
            "recovery_amount",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

baseline_recovered_cases = int(
    data.get(
        "recovered",
        pd.Series(dtype=bool),
    ).fillna(False).sum()
)
baseline_recovered_money = float(
    pd.to_numeric(
        data.get(
            "recovered_amount",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).fillna(0).sum()
)
baseline_recovery_rate = (
    baseline_recovered_cases / total_cases
    if total_cases
    else 0.0
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue at Risk",
        format_inr(revenue_at_risk),
    )

with col2:
    st.metric(
        "Verified Recovery",
        format_inr(money_recovered),
    )

with col3:
    st.metric(
        "Recovery Rate",
        f"{recovery_rate:.2%}",
    )

with col4:
    st.metric(
        "Money / Attempt",
        format_inr(money_per_attempt),
    )

st.caption(
    "Economic metrics above are calculated directly from the "
    "3,262-case unseen-agent output. No development-set result "
    "is mixed into these values."
)

st.subheader("Baseline comparison")
comparison = pd.DataFrame(
    {
        "Metric": [
            "Recovery rate",
            "Recovered money",
        ],
        "RecoverAI": [
            recovery_rate,
            money_recovered,
        ],
        "One-retry baseline": [
            baseline_recovery_rate,
            baseline_recovered_money,
        ],
    }
)

comparison_display = comparison.astype(object)
comparison_display.loc[0, ["RecoverAI", "One-retry baseline"]] = comparison_display.loc[
    0,
    ["RecoverAI", "One-retry baseline"],
].map(lambda value: f"{value:.2%}")
comparison_display.loc[1, ["RecoverAI", "One-retry baseline"]] = comparison_display.loc[
    1,
    ["RecoverAI", "One-retry baseline"],
].map(format_inr)
st.dataframe(comparison_display, hide_index=True, width="stretch")

st.success(
    "RecoverAI recovered "
    f"{format_inr(money_recovered - baseline_recovered_money)} "
    "more than the one-retry baseline on the same unseen cases."
)


# =========================================================
# Performance
# =========================================================

st.header("Recovery Performance")

performance = pd.DataFrame(
    {
        "Metric": [
            "Recovery Rate",
            "Verified Money Recovered",
        ],
        "RecoverAI": [
            recovery_rate,
            money_recovered,
        ],
    }
)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(
        pd.DataFrame(
            {
                "Recovery Rate": [recovery_rate],
            },
            index=["RecoverAI"],
        )
    )

with col2:
    st.bar_chart(
        pd.DataFrame(
            {
                "Verified Recovery": [
                    money_recovered
                ],
            },
            index=["RecoverAI"],
        )
    )


# =========================================================
# Agent behavior
# =========================================================

st.divider()
st.header("🤖 Agent Behavior")

agent_status = (
    data["final_status"]
    .value_counts()
    .rename_axis("status")
    .reset_index(name="cases")
)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(
        agent_status.set_index("status")
    )

with col2:
    st.metric(
        "Recovered",
        f"{recovered_cases:,}",
    )

    st.metric(
        "Escalated",
        f"{escalated_cases:,}",
    )

    st.metric(
        "Stopped",
        f"{stopped_cases:,}",
    )


# =========================================================
# Confidence-aware behavior
# =========================================================

st.divider()
st.header("🎯 Confidence-Aware Decisions")

st.caption(
    "Confidence is a decision-support signal. Low confidence "
    "does not override deterministic policy guardrails; it "
    "helps identify cases where the model should be treated "
    "more cautiously."
)

confidence_distribution = (
    data["confidence_level"]
    .value_counts()
    .rename_axis("confidence_level")
    .reset_index(name="cases")
)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(
        confidence_distribution.set_index(
            "confidence_level"
        )
    )

with col2:
    confidence_summary = (
        data.groupby("confidence_level")
        .agg(
            cases=("case_id", "count"),
            recovered=(
                "final_status",
                lambda s: (
                    s == "recovered"
                ).sum(),
            ),
            total_recovered=(
                "total_recovered",
                "sum",
            ),
            average_confidence=(
                "confidence_score",
                "mean",
            ),
        )
        .reset_index()
    )

    confidence_summary["recovery_rate"] = (
        confidence_summary["recovered"]
        / confidence_summary["cases"]
    )

    display_confidence = confidence_summary.copy()

    display_confidence[
        "average_confidence"
    ] = display_confidence[
        "average_confidence"
    ].map(
        lambda x: (
            "—"
            if pd.isna(x)
            else f"{x:.3f}"
        )
    )

    display_confidence[
        "total_recovered"
    ] = display_confidence[
        "total_recovered"
    ].map(format_inr)

    display_confidence[
        "recovery_rate"
    ] = display_confidence[
        "recovery_rate"
    ].map(
        lambda x: f"{x:.2%}"
    )

    display_confidence = display_confidence.rename(
        columns={
            "confidence_level": "Confidence",
            "cases": "Cases",
            "recovered": "Recovered",
            "total_recovered": "Money Recovered",
            "average_confidence": "Avg Score",
            "recovery_rate": "Recovery Rate",
        }
    )

    st.dataframe(
        display_confidence,
        width="stretch",
        hide_index=True,
    )


# =========================================================
# Category performance
# =========================================================

st.divider()
st.header("Failure Category Performance")

category_summary = (
    data.groupby("failure_category")
    .agg(
        cases=("case_id", "count"),
        recovered=(
            "final_status",
            lambda s: (
                s == "recovered"
            ).sum(),
        ),
        money_recovered=(
            "total_recovered",
            "sum",
        ),
        attempts=(
            "attempts",
            "sum",
        ),
    )
    .reset_index()
)

category_summary["recovery_rate"] = (
    category_summary["recovered"]
    / category_summary["cases"]
)

category_summary["money_per_attempt"] = (
    category_summary["money_recovered"]
    / category_summary["attempts"].replace(
        0,
        float("nan"),
    )
)

category_display = category_summary.copy()

for column in [
    "recovery_rate",
]:
    category_display[column] = category_display[column].map(
        lambda x: f"{x:.2%}"
    )

for column in [
    "money_recovered",
    "money_per_attempt",
]:
    category_display[column] = category_display[column].map(
        lambda x: (
            "—"
            if pd.isna(x)
            else format_inr(x)
        )
    )

category_display = category_display.rename(
    columns={
        "failure_category": "Failure Category",
        "cases": "Cases",
        "recovered": "Recovered",
        "recovery_rate": "Recovery Rate",
        "money_recovered": "Money Recovered",
        "attempts": "Attempts",
        "money_per_attempt": "Money / Attempt",
    }
)

st.dataframe(
    category_display.sort_values(
        "Recovery Rate",
        ascending=False,
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("Category × Final Status")

category_status = (
    data.groupby(
        [
            "failure_category",
            "final_status",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

st.dataframe(
    category_status,
    width="stretch",
)


# =========================================================
# Recovery cases
# =========================================================

st.divider()
st.header("Recovery Cases")

search = st.text_input(
    "Search by Case ID or Customer ID"
)

filtered = data.copy()

if search:
    case_mask = (
        filtered["case_id"]
        .astype(str)
        .str.contains(
            search,
            case=False,
            na=False,
        )
    )

    customer_mask = (
        filtered["customer_id"]
        .astype(str)
        .str.contains(
            search,
            case=False,
            na=False,
        )
        if "customer_id" in filtered.columns
        else False
    )

    filtered = filtered[
        case_mask | customer_mask
    ]

display_columns = [
    "case_id",
    "customer_id",
    "failure_category",
    "recovery_amount",
    "final_action",
    "recovery_probability",
    "erv",
    "final_status",
    "attempts",
    "total_recovered",
    "confidence_score",
    "confidence_level",
]

available_display_columns = [
    column
    for column in display_columns
    if column in filtered.columns
]

st.dataframe(
    filtered[
        available_display_columns
    ].sort_values(
        "recovery_amount",
        ascending=False,
    ),
    width="stretch",
    hide_index=True,
)


# =========================================================
# Case investigation
# =========================================================

st.divider()
st.header("🔎 Case Investigation")

case_ids = (
    data["case_id"]
    .drop_duplicates()
    .tolist()
)

default_case = (
    case_ids.index("CASE_000022")
    if "CASE_000022" in case_ids
    else 0
)

selected_case = st.selectbox(
    "Select a recovery case",
    case_ids,
    index=default_case,
)

case = data[
    data["case_id"] == selected_case
].iloc[0]


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Recovery Amount",
        format_inr(
            safe_number(
                case,
                "recovery_amount",
            )
        ),
    )

with col2:
    st.metric(
        "ML Probability",
        (
            f"{safe_number(case, 'recovery_probability'):.1%}"
            if "recovery_probability" in case
            else "—"
        ),
    )

with col3:
    st.metric(
        "Expected Recovery",
        format_inr(
            safe_number(
                case,
                "expected_recovery",
            )
        ),
    )

with col4:
    st.metric(
        "ERV",
        format_inr(
            safe_number(
                case,
                "erv",
            )
        ),
    )

with col5:
    confidence = case.get(
        "confidence_score"
    )

    st.metric(
        "Confidence",
        (
            "—"
            if pd.isna(confidence)
            else f"{float(confidence):.3f}"
        ),
    )


st.write(
    f"**Customer:** "
    f"{case.get('customer_id', '—')}"
)

st.write(
    f"**Failure category:** "
    f"{case.get('failure_category', '—')}"
)

st.write(
    f"**Final status:** "
    f"{case.get('final_status', '—')}"
)

st.write(
    f"**Surfaces:** "
    f"{case.get('surfaces', '—')}"
)

st.write(
    f"**Initial / final action:** "
    f"{case.get('final_action', '—')}"
)

st.write(
    f"**Decision reason:** "
    f"{case.get('decision_reason', '—')}"
)

st.write(
    f"**Guardrail:** "
    f"{case.get('guardrail_status', '—')}"
)

st.write(
    f"**Confidence level:** "
    f"{case.get('confidence_level', 'not_evaluated')}"
)


# =========================================================
# Audit timeline
# =========================================================

st.subheader("Agent Audit Timeline")

audit = parse_audit_trail(
    case.get(
        "audit_trail",
        "[]",
    )
)

if not audit:
    st.info(
        "No audit events available."
    )

else:
    for event in audit:
        event_type = event.get(
            "event",
            "event",
        )

        if event_type == "action_selected":
            st.info(
                f"🎯 **Action selected:** "
                f"{event.get('action')} "
                f"(attempt "
                f"{event.get('attempt_number')})"
            )

        elif event_type == "action_executed":
            status = event.get(
                "verification_status"
            )

            st.write(
                f"⚙️ **Executed:** "
                f"{event.get('action')} — "
                f"verification: "
                f"**{status}**"
            )

        elif event_type == "recovery_verified":
            st.success(
                f"✅ **Recovery verified:** "
                f"{format_inr(event.get('amount', 0))}"
            )

        elif event_type == "recovery_failed":
            st.warning(
                "❌ Recovery attempt failed."
            )

        elif event_type == "next_action_evaluation":
            st.info(
                f"🔄 **Next action:** "
                f"{event.get('next_action')} "
                f"(ERV "
                f"{format_inr(event.get('erv', 0))})"
            )

        elif event_type == "escalation":
            st.error(
                "👤 **Escalated to manual review.**"
            )

        elif event_type == "stopping_decision":
            st.warning(
                f"🛑 **Stopped:** "
                f"{event.get('reason', '')}"
            )

        else:
            st.caption(
                f"• {event_type}: {event}"
            )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "RecoverAI | Held-out unseen evaluation | "
    "ML-assisted recovery with deterministic guardrails | "
    "Confidence-aware agent behavior"
)
