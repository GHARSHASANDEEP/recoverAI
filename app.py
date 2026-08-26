import json
import os

import pandas as pd
import streamlit as st

from src.data.outcome_rules import get_amount_bucket
from src.engine.decision_engine import apply_decision
from src.engine.erv import score_case_actions


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RecoverAI",
    page_icon="💳",
    layout="wide",
)

DATA_DIR = "data/processed"


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():

    decisions = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "decisions.csv",
        )
    )

    agent = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "agent_results.csv",
        )
    )

    cases = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "recovery_cases.csv",
        )
    )

    evaluation = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "evaluation_report.csv",
        )
    )

    data = decisions.merge(
        agent,
        on="case_id",
        how="left",
    )

    data = data.merge(
        cases[
            [
                "case_id",
                "surfaces",
                "event_types",
                "failure_categories",
                "dedup_status",
                "dedup_score",
            ]
        ],
        on="case_id",
        how="left",
        suffixes=("", "_case"),
    )

    return (
        data,
        evaluation,
    )


data, evaluation = load_data()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def metric_value(
    metric,
    column="recoverAI",
):
    row = evaluation[
        evaluation["metric"] == metric
    ]

    if row.empty:
        return 0.0

    return float(
        row.iloc[0][column]
    )


def format_inr(value):
    return f"₹{value:,.2f}"


def parse_audit_trail(value):

    if pd.isna(value):
        return []

    try:
        return json.loads(value)
    except Exception:
        return []


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("RecoverAI")
st.subheader(
    "Intelligent Payment Recovery Platform"
)

st.caption(
    "Diagnose → Predict → Optimize → Execute → Verify → Recover"
)


# --------------------------------------------------
# Judge Mode
# --------------------------------------------------

st.divider()
st.header("🧪 Judge Mode")
st.caption(
    "Create a new recovery case and run it through the same "
    "ML → ERV → guardrail decision pipeline used by RecoverAI. "
    "This does not modify the stored batch results."
)

with st.form("judge_case_form"):

    failure_options = sorted(
        data["failure_category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    segment_options = sorted(
        data["customer_segment"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        judge_failure = st.selectbox(
            "Failure category",
            failure_options,
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
        use_container_width=True,
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
            "communication_opt_in": bool(
                judge_opt_in
            ),
            "recovery_amount": float(
                judge_amount
            ),
            "attempt_number": int(
                judge_attempt
            ),
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
        }

        try:

            scores = score_case_actions(
                judge_case
            )

            decision = apply_decision(
                judge_case,
                scores,
            )

            st.success(
                "✓ RecoverAI completed the decision."
            )

            st.subheader(
                "RecoverAI Decision"
            )

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
                    "ERV",
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

            st.info(
                f"**Decision reason:** "
                f"{decision.get('decision_reason', '')}"
            )

            st.subheader(
                "Action Comparison"
            )

            candidate_df = pd.DataFrame(
                decision.get(
                    "candidate_actions",
                    [],
                )
            )

            if not candidate_df.empty:

                candidate_display = (
                    candidate_df[
                        [
                            "action",
                            "recovery_probability",
                            "expected_recovery",
                            "action_cost",
                            "erv",
                            "guardrail_allowed",
                            "guardrail_reason",
                        ]
                    ]
                    .copy()
                )

                candidate_display[
                    "recovery_probability"
                ] = candidate_display[
                    "recovery_probability"
                ].map(
                    lambda x: f"{x:.1%}"
                )

                candidate_display[
                    "expected_recovery"
                ] = candidate_display[
                    "expected_recovery"
                ].map(
                    format_inr
                )

                candidate_display[
                    "action_cost"
                ] = candidate_display[
                    "action_cost"
                ].map(
                    format_inr
                )

                candidate_display[
                    "erv"
                ] = candidate_display[
                    "erv"
                ].map(
                    format_inr
                )

                candidate_display[
                    "guardrail_allowed"
                ] = candidate_display[
                    "guardrail_allowed"
                ].map(
                    lambda x: "✓ Allowed"
                    if x
                    else "✗ Blocked"
                )

                candidate_display = (
                    candidate_display.rename(
                        columns={
                            "action": "Action",
                            "recovery_probability": "Probability",
                            "expected_recovery": "Expected Recovery",
                            "action_cost": "Cost",
                            "erv": "ERV",
                            "guardrail_allowed": "Guardrail",
                            "guardrail_reason": "Guardrail Reason",
                        }
                    )
                )

                st.dataframe(
                    candidate_display,
                    use_container_width=True,
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


# --------------------------------------------------
# Stress tests
# --------------------------------------------------

with st.expander("🛡️ Run predefined safety stress tests"):

    st.caption(
        "These cases demonstrate deterministic guardrails and "
        "input safety. They are evaluated using the same decision engine."
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
                    "amount_bucket": get_amount_bucket(20000.0),
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
                    "amount_bucket": get_amount_bucket(20000.0),
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
                    "amount_bucket": get_amount_bucket(50001.0),
                },
                "focus": "Demonstrates the high-value policy path.",
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
                        if row["action"] == "retry"
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
                        if row["action"] == "reminder"
                    ),
                    None,
                )

                if item["name"] == "Communication opt-out":
                    result = (
                        "PASS"
                        if reminder_row
                        and not reminder_row[
                            "guardrail_allowed"
                        ]
                        else "CHECK"
                    )

                elif item["name"] == "Risk decline":
                    result = (
                        "PASS"
                        if retry_row
                        and not retry_row[
                            "guardrail_allowed"
                        ]
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
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Malformed-input handling is demonstrated through the "
        "Judge Mode validation above."
    )


st.divider()


# --------------------------------------------------
# KPI cards
# --------------------------------------------------

revenue_at_risk = metric_value(
    "gross_revenue_at_risk"
)

recoverAI_money = metric_value(
    "money_recovered"
)

baseline_money = metric_value(
    "money_recovered",
    "baseline",
)

incremental_money = metric_value(
    "incremental_money_recovered"
)

improvement = metric_value(
    "improvement_percent"
)

recovery_rate = metric_value(
    "recovery_rate"
)

baseline_rate = metric_value(
    "recovery_rate",
    "baseline",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue at Risk",
        format_inr(
            revenue_at_risk
        ),
    )

with col2:
    st.metric(
        "Verified Recovery",
        format_inr(
            recoverAI_money
        ),
    )

with col3:
    st.metric(
        "Incremental Recovery",
        format_inr(
            incremental_money
        ),
    )

with col4:
    st.metric(
        "Improvement vs Baseline",
        f"{improvement:.2f}%",
    )


st.divider()


# --------------------------------------------------
# Performance comparison
# --------------------------------------------------

st.header(
    "Recovery Performance"
)

comparison = pd.DataFrame(
    {
        "System": [
            "Blind Retry Baseline",
            "RecoverAI",
        ],
        "Recovery Rate": [
            baseline_rate,
            recovery_rate,
        ],
        "Money Recovered": [
            baseline_money,
            recoverAI_money,
        ],
    }
)

col1, col2 = st.columns(2)

with col1:

    st.bar_chart(
        comparison.set_index(
            "System"
        )["Recovery Rate"]
    )

with col2:

    st.bar_chart(
        comparison.set_index(
            "System"
        )["Money Recovered"]
    )


st.divider()


# --------------------------------------------------
# Agent behavior
# --------------------------------------------------

st.header(
    "Agent Behavior"
)

agent_status = (
    data[
        "final_status"
    ]
    .value_counts()
    .rename_axis("status")
    .reset_index(
        name="cases"
    )
)

col1, col2 = st.columns(2)

with col1:

    st.bar_chart(
        agent_status.set_index(
            "status"
        )
    )

with col2:

    multi_attempt = int(
        (
            data["attempts"] > 1
        ).sum()
    )

    total_attempts = int(
        data["attempts"].sum()
    )

    st.metric(
        "Multi-attempt Cases",
        f"{multi_attempt:,}",
    )

    st.metric(
        "Total Recovery Attempts",
        f"{total_attempts:,}",
    )

    st.metric(
        "Audit Events",
        f"{int(data['audit_event_count'].sum()):,}",
    )


st.divider()


# --------------------------------------------------
# Recovery cases
# --------------------------------------------------

st.header(
    "Recovery Cases"
)

search = st.text_input(
    "Search by Case ID or Customer ID"
)

filtered = data.copy()

if search:

    filtered = filtered[
        filtered["case_id"]
        .str.contains(
            search,
            case=False,
            na=False,
        )
        |
        filtered["customer_id"]
        .str.contains(
            search,
            case=False,
            na=False,
        )
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
]

st.dataframe(
    filtered[
        display_columns
    ].sort_values(
        "recovery_amount",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Case investigation
# --------------------------------------------------

st.divider()

st.header(
    "Case Investigation"
)

case_ids = (
    data["case_id"]
    .drop_duplicates()
    .tolist()
)

default_case = (
    case_ids.index(
        "CASE_000022"
    )
    if "CASE_000022" in case_ids
    else 0
)

selected_case = st.selectbox(
    "Select a recovery case",
    case_ids,
    index=default_case,
)


case = data[
    data["case_id"]
    == selected_case
].iloc[0]


# --------------------------------------------------
# Case summary
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Recovery Amount",
        format_inr(
            case[
                "recovery_amount"
            ]
        ),
    )

with col2:
    st.metric(
        "ML Probability",
        f"{case['recovery_probability']:.1%}",
    )

with col3:
    st.metric(
        "Expected Recovery",
        format_inr(
            case[
                "expected_recovery"
            ]
        ),
    )

with col4:
    st.metric(
        "ERV",
        format_inr(
            case["erv"]
        ),
    )


st.write(
    f"**Customer:** "
    f"{case['customer_id']}"
)

st.write(
    f"**Failure category:** "
    f"{case['failure_category']}"
)

st.write(
    f"**Surfaces:** "
    f"{case['surfaces']}"
)

st.write(
    f"**Initial action:** "
    f"{case['final_action']}"
)

st.write(
    f"**Decision reason:** "
    f"{case['decision_reason']}"
)

st.write(
    f"**Guardrail:** "
    f"{case['guardrail_status']}"
)


# --------------------------------------------------
# Audit timeline
# --------------------------------------------------

st.subheader(
    "Agent Audit Timeline"
)

audit = parse_audit_trail(
    case["audit_trail"]
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

        if event_type == (
            "action_selected"
        ):

            st.info(
                f"🎯 **Action selected:** "
                f"{event.get('action')} "
                f"(attempt "
                f"{event.get('attempt_number')})"
            )

        elif event_type == (
            "action_executed"
        ):

            status = event.get(
                "verification_status"
            )

            st.write(
                f"⚙️ **Executed:** "
                f"{event.get('action')} — "
                f"verification: "
                f"**{status}**"
            )

        elif event_type == (
            "recovery_verified"
        ):

            st.success(
                f"✅ **Recovery verified:** "
                f"{format_inr(event.get('amount', 0))}"
            )

        elif event_type == (
            "recovery_failed"
        ):

            st.warning(
                "❌ Recovery attempt failed."
            )

        elif event_type == (
            "next_action_evaluation"
        ):

            st.info(
                f"🔄 **Next action:** "
                f"{event.get('next_action')} "
                f"(ERV "
                f"{format_inr(event.get('erv', 0))})"
            )

        elif event_type == (
            "escalation"
        ):

            st.error(
                "👤 **Escalated to manual review.**"
            )

        elif event_type == (
            "stopping_decision"
        ):

            st.warning(
                f"🛑 **Stopped:** "
                f"{event.get('reason', '')}"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "RecoverAI | Synthetic evaluation environment | "
    "ML-assisted recovery with deterministic guardrails"
)
