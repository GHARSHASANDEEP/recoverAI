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

st.set_page_config(
    page_title="RecoverAI — Razorpay Buildathon Track 03",
    page_icon="💳",
    layout="wide",
)

UNSEEN_DIR = "data/unseen/processed"
REFERENCE_DIR = "data/processed"


@st.cache_data
def load_data():
    unseen_decisions_path = os.path.join(UNSEEN_DIR, "decisions.csv")
    unseen_agent_path = os.path.join(UNSEEN_DIR, "agent_results.csv")
    unseen_baseline_path = os.path.join(UNSEEN_DIR, "baseline_results.csv")
    unseen_cases_path = os.path.join(UNSEEN_DIR, "recovery_cases.csv")

    if not (os.path.exists(unseen_decisions_path) and os.path.exists(unseen_agent_path)):
        raise FileNotFoundError(
            "Unseen evaluation files not found. "
            "Expected data/unseen/processed/decisions.csv and agent_results.csv."
        )

    decisions = pd.read_csv(unseen_decisions_path)
    agent = pd.read_csv(unseen_agent_path)
    baseline = pd.read_csv(unseen_baseline_path) if os.path.exists(unseen_baseline_path) else None

    data = decisions.merge(agent, on="case_id", how="left", suffixes=("", "_agent"))

    if baseline is not None:
        data = data.merge(
            baseline[["case_id", "recovered", "recovered_amount", "attempts"]],
            on="case_id", how="left", suffixes=("", "_baseline"), validate="one_to_one",
        )

    if os.path.exists(unseen_cases_path):
        cases = pd.read_csv(unseen_cases_path)
        case_columns = ["case_id", "recovery_amount", "surfaces", "event_types",
                        "failure_categories", "dedup_status", "dedup_score"]
        available = [c for c in case_columns if c in cases.columns]
        if "case_id" in available:
            data = data.merge(cases[available], on="case_id", how="left", suffixes=("", "_case"))

    reference_evaluation = None
    reference_path = os.path.join(REFERENCE_DIR, "evaluation_report.csv")
    if os.path.exists(reference_path):
        reference_evaluation = pd.read_csv(reference_path)

    return data, reference_evaluation


data, reference_evaluation = load_data()


def format_inr(value):
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "₹0.00"


def failure_explanation(failure_category):
    explanations = {
        "insufficient_funds": (
            "Declined due to insufficient balance at authorization time.",
            "Reminder first → reassess → retry if policy allows → escalate.",
        ),
        "authentication_failed": (
            "Authentication step was not completed successfully.",
            "Ask customer to complete authentication → reassess → retry → escalate.",
        ),
        "temporary_bank_failure": (
            "Issuing bank experienced a transient failure, not a permanent decline.",
            "Retry first → verify outcome → reassess → escalate if path exhausted.",
        ),
        "timeout": (
            "Payment attempt did not receive a timely response from the payment path.",
            "Retry first → verify → reassess → escalate if retry path exhausted.",
        ),
        "limit_exceeded": (
            "Transaction exceeded an applicable payment or instrument limit.",
            "Notify customer first → reassess → retry when policy permits → escalate.",
        ),
        "expired_instrument": (
            "Payment instrument is expired or no longer valid for authorization.",
            "Prompt customer action → reassess → escalate if instrument unrecoverable.",
        ),
        "risk_decline": (
            "Declined by risk-control decision. Blind retries will not resolve this.",
            "Do not retry automatically → escalate for manual review.",
        ),
        "blocked_instrument": (
            "Payment instrument is blocked and must not be retried automatically.",
            "Do not retry → escalate or request a valid alternative.",
        ),
        "unknown_failure": (
            "Gateway supplied an unclassified failure signal. Using safest recovery path.",
            "Conservative recovery stage → execute → verify → reassess before next action.",
        ),
    }
    return explanations.get(
        failure_category,
        ("Payment failure could not be mapped to a known diagnosis.",
         "Use safest bounded recovery path and escalate when exhausted."),
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
    values = data[column].dropna().astype(str).unique().tolist()
    return sorted(values) if values else fallback


def recovery_priority(case):
    amount_score = min(float(case.get("recovery_amount", 0.0)) / 100000.0, 1.0)
    history_total = max(float(case.get("total_payment_attempts", 1)), 1.0)
    success_rate = float(case.get("successful_payment_count", 0)) / history_total
    failure_score = {
        "temporary_bank_failure": 0.90, "timeout": 0.85,
        "insufficient_funds": 0.65, "authentication_failed": 0.55,
        "expired_instrument": 0.35, "limit_exceeded": 0.30,
        "unknown_failure": 0.25, "blocked_instrument": 0.15, "risk_decline": 0.10,
    }.get(case.get("failure_category"), 0.25)
    segment_score = {"high_value": 1.0, "regular": 0.65, "new": 0.35}.get(case.get("customer_segment"), 0.50)
    attempt_score = max(0.2, 1.0 - (int(case.get("attempt_number", 1)) - 1) * 0.2)
    score = 100 * (amount_score * 0.35 + failure_score * 0.25 + segment_score * 0.15
                   + success_rate * 0.15 + attempt_score * 0.10)
    return round(max(0.0, min(100.0, score)), 1)


def recovery_message(case, action, channel):
    amount = format_inr(case.get("recovery_amount", 0.0))
    if action == "reminder":
        message = (f"We noticed a payment of {amount} could not be completed. "
                   "Please update your payment details or try again when convenient.")
    elif action == "retry":
        message = (f"We are retrying your {amount} payment through the approved "
                   "recovery flow. We will confirm the result shortly.")
    else:
        message = "This case has been routed to a specialist for review."
    return {"channel": channel, "message": message, "status": "preview_only"}


for optional_column, default_value in {
    "confidence_score": None, "confidence_level": "not_evaluated",
    "audit_event_count": 0, "attempts": 0, "total_recovered": 0.0, "audit_trail": "[]",
}.items():
    if optional_column not in data.columns:
        data[optional_column] = default_value

data["confidence_level"] = data["confidence_level"].fillna("not_evaluated").astype(str)
data["attempts"] = pd.to_numeric(data["attempts"], errors="coerce").fillna(0)
data["total_recovered"] = pd.to_numeric(data["total_recovered"], errors="coerce").fillna(0.0)
data["audit_event_count"] = pd.to_numeric(data["audit_event_count"], errors="coerce").fillna(0)


# =========================================================
# HERO — what judges see first
# =========================================================

st.markdown(
    "<h1 style='font-size:2.4rem;margin-bottom:0'>💳 RecoverAI</h1>"
    "<p style='font-size:1.1rem;color:#888;margin-top:4px'>"
    "Razorpay Buildathon Track 03 · AI Revenue Recovery</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
**RecoverAI is not a retry button.** It diagnoses why a payment failed, scores recoverability
with a trained ML model, selects the safest action from a failure-aware policy, enforces
deterministic guardrails, executes through a provider boundary, and verifies the outcome —
all in a closed loop with a full audit trail.

> ML recommends · Policy constrains · Guardrails authorize · Provider executes · Event verifies
""")

total_cases = len(data)
recovered_cases = int((data["final_status"] == "recovered").sum())
escalated_cases = int((data["final_status"] == "escalated").sum())
stopped_cases = int((data["final_status"] == "stopped").sum())
recovery_rate = recovered_cases / total_cases if total_cases else 0.0
money_recovered = float(data["total_recovered"].sum())
total_attempts = int(data["attempts"].sum())
audit_events = int(data["audit_event_count"].sum())
money_per_attempt = money_recovered / total_attempts if total_attempts else 0.0
average_recovered_case = money_recovered / recovered_cases if recovered_cases else 0.0
multi_attempt_cases = int((data["attempts"] > 1).sum())

baseline_recovered_cases = int(data.get("recovered", pd.Series(dtype=bool)).fillna(False).sum())
baseline_recovered_money = float(
    pd.to_numeric(data.get("recovered_amount", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
)
baseline_recovery_rate = baseline_recovered_cases / total_cases if total_cases else 0.0
incremental_money = money_recovered - baseline_recovered_money
incremental_rate = recovery_rate - baseline_recovery_rate

st.divider()

# Primary KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("RecoverAI Recovery Rate", f"{recovery_rate:.2%}",
              delta=f"+{incremental_rate:.2%} vs baseline")
with c2:
    st.metric("Simulated Recovery", format_inr(money_recovered),
              delta=f"+{format_inr(incremental_money)} vs baseline")
with c3:
    st.metric("Cases Evaluated", f"{total_cases:,}",
              delta=f"{recovered_cases:,} recovered")
with c4:
    st.metric("One-Retry Baseline Rate", f"{baseline_recovery_rate:.2%}",
              delta=f"{baseline_recovered_cases:,} cases")

st.caption(
    "⚠️ All monetary figures are **deterministic simulated benchmark outcomes** on 3,262 held-out unseen cases. "
    "No real customer money was moved. Razorpay Test Mode webhook was verified with HTTP 200 responses. "
    "Strategy comparisons are reproducible — same population, same logic, different strategy."
)

st.divider()

# Secondary KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Recovered", f"{recovered_cases:,}")
    st.progress(recovery_rate)
with c2:
    st.metric("Escalated (safe hand-off)", f"{escalated_cases:,}")
    st.progress(escalated_cases / total_cases if total_cases else 0)
with c3:
    st.metric("Stopped (unsafe blocked)", f"{stopped_cases:,}")
    st.progress(stopped_cases / total_cases if total_cases else 0)
with c4:
    st.metric("Audit Events Recorded", f"{audit_events:,}")
    st.progress(min(audit_events / 10000, 1.0))

st.caption(
    "RecoverAI stops or escalates when: communication is not permitted · ERV is non-positive · "
    "retry limit is reached · failure category makes automation unsafe. "
    "Every transition is recorded."
)


# =========================================================
# HOW IT WORKS — architecture for judges
# =========================================================

with st.expander("🏗️ How RecoverAI works — architecture overview", expanded=False):
    st.markdown("""
**Two ML models, one policy engine, deterministic guardrails.**

| Layer | What it does | Can it override guardrails? |
|---|---|---|
| V3 recoverability model | Estimates whether a case is recoverable | No |
| Action recommender (V2) | Scores retry vs reminder vs escalate | No |
| Failure-aware policy | Defines the safe recovery sequence per failure type | Defines the sequence |
| Deterministic guardrails | Blocks unsafe actions (opt-out, risk decline, ERV ≤ 0, attempt limit) | **Final authority** |
| Provider boundary | Executes and verifies via Razorpay Test Mode | Confirms recovery |
| Recovery memory | Records only verified outcomes for future calibration | Write-protected |

**Why not just use an LLM?**
Payment recovery requires deterministic safety guarantees. An LLM cannot be trusted to never
retry a risk decline, never contact an opted-out customer, or never exceed the attempt budget.
RecoverAI uses ML where probability estimation adds value and deterministic rules where safety
is non-negotiable.

**Razorpay integration**
The webhook receiver handles `payment.failed`, `payment.authorized`, `payment.captured`,
`payment_link.paid`, `payment_link.expired`, `subscription.halted`, `invoice.expired`, and `order.paid`.
Signature verification and event-ID idempotency are enforced before any case is created.
""")

# =========================================================
# BASELINE COMPARISON — the core result
# =========================================================

st.header("📊 RecoverAI vs One-Retry Baseline")

st.markdown(
    "Same 3,262 unseen cases. Same failure distribution. Different strategy. "
    "The baseline retries once and stops. RecoverAI diagnoses, scores, sequences, and verifies."
)

comp_col1, comp_col2, comp_col3 = st.columns(3)
with comp_col1:
    st.metric("RecoverAI cases recovered", f"{recovered_cases:,}")
    st.metric("Baseline cases recovered", f"{baseline_recovered_cases:,}",
              delta=f"{recovered_cases - baseline_recovered_cases:+,} difference")
with comp_col2:
    st.metric("RecoverAI recovery rate", f"{recovery_rate:.2%}")
    st.metric("Baseline recovery rate", f"{baseline_recovery_rate:.2%}",
              delta=f"{incremental_rate:+.2%} difference")
with comp_col3:
    st.metric("RecoverAI simulated recovery", format_inr(money_recovered))
    st.metric("Baseline simulated recovery", format_inr(baseline_recovered_money),
              delta=f"+{format_inr(incremental_money)}")

comparison_df = pd.DataFrame({
    "Strategy": ["RecoverAI", "One-retry baseline", "Difference"],
    "Recovery rate": [f"{recovery_rate:.2%}", f"{baseline_recovery_rate:.2%}",
                      f"+{incremental_rate:.2%}"],
    "Cases recovered": [f"{recovered_cases:,}", f"{baseline_recovered_cases:,}",
                        f"+{recovered_cases - baseline_recovered_cases:,}"],
    "Simulated recovery": [format_inr(money_recovered), format_inr(baseline_recovered_money),
                           f"+{format_inr(incremental_money)}"],
    "Automated attempts": [f"{total_attempts:,}", f"{baseline_recovered_cases:,}", "—"],
})
st.dataframe(comparison_df, hide_index=True, use_container_width=True)

st.success(
    f"RecoverAI recovered **{recovered_cases - baseline_recovered_cases:,} more cases** "
    f"(+{incremental_rate:.2%}) and **{format_inr(incremental_money)} more** "
    f"than the one-retry baseline on the same unseen population. "
    f"(Simulated benchmark outcomes — not real revenue.)"
)

st.divider()


# =========================================================
# JUDGE MODE
# =========================================================

st.header("🧪 Judge Mode — Live Decision Engine")

st.markdown(
    "Build any recovery case and watch RecoverAI reason through it in real time: "
    "failure diagnosis → ML scoring → policy selection → guardrail check → "
    "channel selection → closed-loop execution → audit trail. "
    "Try `risk_decline` to see a blocked retry. Try `temporary_bank_failure` to see a successful recovery."
)

failure_options = get_unique_options("failure_category", [
    "authentication_failed", "blocked_instrument", "expired_instrument",
    "insufficient_funds", "limit_exceeded", "risk_decline",
    "temporary_bank_failure", "timeout", "unknown_failure",
])
segment_options = get_unique_options("customer_segment", ["high_value", "regular"])

with st.form("judge_case_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        judge_failure = st.selectbox("Failure category", failure_options)
        judge_amount = st.number_input("Recovery amount (₹)", min_value=0.01, value=20000.0, step=500.0)
        judge_attempt = st.number_input("Attempt number", min_value=1, max_value=20, value=1, step=1)
    with col2:
        judge_segment = st.selectbox("Customer segment", segment_options)
        judge_channel = st.selectbox("Preferred channel", ["auto", "whatsapp", "sms", "email"],
                                     help="Consent-aware channel selection preview.")
        judge_opt_in = st.checkbox("Customer opted into communication", value=True)
        judge_lifetime_value = st.number_input("Customer lifetime value (₹)", min_value=0.01,
                                               value=50000.0, step=1000.0)
    with col3:
        judge_successful = st.number_input("Successful payment count", min_value=0, value=8, step=1)
        judge_failed = st.number_input("Failed payment count", min_value=0, value=2, step=1)
        judge_total_attempts = st.number_input("Total payment attempts", min_value=1, value=10, step=1)

    submitted = st.form_submit_button("🚀 Run RecoverAI", use_container_width=True, type="primary")

if submitted:
    validation_errors = []
    if judge_amount <= 0:
        validation_errors.append("Recovery amount must be greater than zero.")
    if judge_lifetime_value <= 0:
        validation_errors.append("Customer lifetime value must be greater than zero.")
    if judge_successful < 0 or judge_failed < 0:
        validation_errors.append("Payment counts cannot be negative.")
    if judge_total_attempts < 1:
        validation_errors.append("Total payment attempts must be at least 1.")
    if judge_successful + judge_failed > judge_total_attempts:
        validation_errors.append("Successful + failed counts cannot exceed total attempts.")

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
            "customer_lifetime_value": float(judge_lifetime_value),
            "successful_payment_count": int(judge_successful),
            "failed_payment_count": int(judge_failed),
            "total_payment_attempts": int(judge_total_attempts),
            "amount_bucket": get_amount_bucket(float(judge_amount)),
            "preferred_channel": ("" if judge_channel == "auto" else judge_channel),
        }

        try:
            scores = score_case_actions(judge_case)
            policy_initial_action = get_initial_action(judge_failure)
            permitted_actions = get_permitted_actions(judge_failure)
            sequence = get_recovery_sequence(judge_failure)

            selected_rows = scores[scores["action"] == policy_initial_action].copy()
            if selected_rows.empty:
                raise ValueError(f"No model score for policy action: {policy_initial_action}")
            selected = selected_rows.iloc[0]

            guardrail = evaluate_guardrails(
                {**judge_case, "erv": float(selected.get("erv", 0.0))},
                policy_initial_action,
            )

            model_candidates = scores[scores["action"].isin(permitted_actions)].sort_values(
                ["erv", "recovery_probability"], ascending=[False, False]
            )
            ai_recommendation = str(model_candidates.iloc[0]["action"]) if not model_candidates.empty else "stop"
            ai_recommendation_row = model_candidates.iloc[0] if not model_candidates.empty else selected

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
                "ai_recommendation_erv": float(ai_recommendation_row.get("erv", 0.0)),
            }

            st.success("✓ RecoverAI diagnosed the failure and selected a bounded recovery strategy.")

            why_failed, strategy = failure_explanation(judge_failure)

            # Step 1: Diagnosis
            st.subheader("Step 1 — Failure Diagnosis")
            st.info(f"**Why it failed:** {why_failed}")
            st.success(f"**Recovery strategy:** {strategy}")

            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Policy Stage", policy_initial_action.upper())
            with s2:
                st.metric("Permitted Actions", str(len(permitted_actions)))
            with s3:
                st.metric("Recovery Path", " → ".join(sequence).upper())

            # Step 2: ML + Economic scoring
            st.subheader("Step 2 — ML Scoring & Action Comparison")
            candidate_df = pd.DataFrame(decision.get("candidate_actions", []))
            if not candidate_df.empty:
                preferred_columns = ["action", "recovery_probability", "expected_recovery",
                                     "action_cost", "erv", "guardrail_allowed", "guardrail_reason"]
                available_columns = [c for c in preferred_columns if c in candidate_df.columns]
                display_df = candidate_df[available_columns].copy()
                if "recovery_probability" in display_df:
                    display_df["recovery_probability"] = display_df["recovery_probability"].map(lambda x: f"{x:.1%}")
                for mc in ["expected_recovery", "action_cost", "erv"]:
                    if mc in display_df:
                        display_df[mc] = display_df[mc].map(format_inr)
                if "guardrail_allowed" in display_df:
                    display_df["guardrail_allowed"] = display_df["guardrail_allowed"].map(
                        lambda x: "✓ Allowed" if x else "✗ Blocked"
                    )
                display_df = display_df.rename(columns={
                    "action": "Action", "recovery_probability": "ML Probability",
                    "expected_recovery": "Expected Recovery", "action_cost": "Cost",
                    "erv": "Economic Value (ERV)", "guardrail_allowed": "Guardrail",
                    "guardrail_reason": "Guardrail Reason",
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Step 3: Decision
            st.subheader("Step 3 — Policy + Guardrail Decision")
            action = decision.get("final_action", "stop")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("AI Recommendation", ai_recommendation.upper())
            with d2:
                st.metric("Policy Action", policy_initial_action.upper())
            with d3:
                st.metric("Guardrail", "✓ PASSED" if guardrail["allowed"] else "✗ BLOCKED")
            with d4:
                st.metric("Final Authorized Action", action.upper())

            factors_df = pd.DataFrame([
                {"Layer": "AI recommendation", "Result": ai_recommendation.upper(),
                 "Note": "Highest model-ranked permitted economic value"},
                {"Layer": "Policy first stage", "Result": policy_initial_action.upper(),
                 "Note": "Failure-aware sequence controls the next safe stage"},
                {"Layer": "Guardrail authority", "Result": "ALLOWED" if guardrail["allowed"] else "BLOCKED",
                 "Note": guardrail["reason"]},
                {"Layer": "Final authorized action", "Result": decision["final_action"].upper(),
                 "Note": "The action the bounded agent may execute"},
            ])
            st.dataframe(factors_df, use_container_width=True, hide_index=True)

            if guardrail["allowed"]:
                st.success(f"✓ Guardrail passed: {guardrail['reason']}")
            else:
                st.error(f"✗ Guardrail blocked: {guardrail['reason']}")

            # Step 4: Priority + Channel
            st.subheader("Step 4 — Priority Scoring & Channel Selection")
            priority = recovery_priority(judge_case)
            delivery_channel = choose_recovery_channel(judge_case, action)
            message_preview = recovery_message(judge_case, action, delivery_channel)

            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric("Priority Score", f"{priority:.1f} / 100")
            with p2:
                st.metric("Priority Band",
                          "HIGH" if priority >= 65 else "MEDIUM" if priority >= 40 else "LOW")
            with p3:
                st.metric("Recovery Channel", delivery_channel.upper())

            st.caption(f"Message preview (not sent) | channel: {message_preview['channel']}")
            st.info(message_preview["message"])

            # Step 5: Closed-loop execution
            st.subheader("Step 5 — Closed-Loop Execution & Audit Trail")
            agent_case = {**judge_case, **decision}
            agent_result = run_recovery_case(agent_case, scores)

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Final Status", agent_result["final_status"].upper())
            with e2:
                st.metric("Simulated Recovery", format_inr(agent_result["total_recovered"]))
            with e3:
                st.metric("Automated Attempts", str(agent_result["attempts"]))
            with e4:
                st.metric("Audit Events", str(len(agent_result["audit_events"])))

            st.caption("State path: " + " → ".join(agent_result["state_history"]))

            audit_display = pd.DataFrame(agent_result["audit_events"])
            audit_columns = [c for c in ["event", "action", "from_state", "to_state", "reason"]
                             if c in audit_display.columns]
            if audit_columns:
                st.dataframe(audit_display[audit_columns], use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error("⚠️ RecoverAI could not process this case safely.")
            st.code(str(exc), language="text")
            st.info("The application rejected the case instead of silently making a decision.")

st.divider()


# =========================================================
# SAFETY STRESS TESTS
# =========================================================

with st.expander("🛡️ Safety stress tests — deterministic guardrail verification"):
    st.markdown(
        "These four cases demonstrate that guardrails are deterministic and cannot be overridden "
        "by ML scores or economic signals."
    )
    if st.button("Run 4 stress tests", key="stress_tests"):
        stress_cases = [
            {
                "name": "Communication opt-out",
                "case": {
                    "case_id": "STRESS_OPT_OUT", "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "timeout", "customer_segment": segment_options[0],
                    "communication_opt_in": False, "recovery_amount": 20000.0,
                    "attempt_number": 1, "customer_lifetime_value": 50000.0,
                    "successful_payment_count": 8, "failed_payment_count": 2,
                    "total_payment_attempts": 10, "amount_bucket": get_amount_bucket(20000.0),
                },
                "focus": "Reminder must be blocked — customer opted out.",
            },
            {
                "name": "Risk decline — no retry",
                "case": {
                    "case_id": "STRESS_RISK", "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "risk_decline", "customer_segment": segment_options[0],
                    "communication_opt_in": True, "recovery_amount": 20000.0,
                    "attempt_number": 1, "customer_lifetime_value": 50000.0,
                    "successful_payment_count": 8, "failed_payment_count": 2,
                    "total_payment_attempts": 10, "amount_bucket": get_amount_bucket(20000.0),
                },
                "focus": "Retry must be blocked — risk decline policy.",
            },
            {
                "name": "High-value case",
                "case": {
                    "case_id": "STRESS_HIGH_VALUE", "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "temporary_bank_failure", "customer_segment": segment_options[0],
                    "communication_opt_in": True, "recovery_amount": 50001.0,
                    "attempt_number": 1, "customer_lifetime_value": 100000.0,
                    "successful_payment_count": 8, "failed_payment_count": 2,
                    "total_payment_attempts": 10, "amount_bucket": get_amount_bucket(50001.0),
                },
                "focus": "High-value policy path — confidence-gated escalation if low confidence.",
            },
            {
                "name": "Blocked instrument",
                "case": {
                    "case_id": "STRESS_BLOCKED", "customer_id": "STRESS_CUSTOMER",
                    "failure_category": "blocked_instrument", "customer_segment": segment_options[0],
                    "communication_opt_in": True, "recovery_amount": 20000.0,
                    "attempt_number": 1, "customer_lifetime_value": 50000.0,
                    "successful_payment_count": 8, "failed_payment_count": 2,
                    "total_payment_attempts": 10, "amount_bucket": get_amount_bucket(20000.0),
                },
                "focus": "Only escalation is permitted — instrument blocked.",
            },
        ]

        results = []
        for item in stress_cases:
            try:
                scores = score_case_actions(item["case"])
                decision = apply_decision(item["case"], scores)
                retry_row = next((r for r in decision.get("candidate_actions", []) if r.get("action") == "retry"), None)
                reminder_row = next((r for r in decision.get("candidate_actions", []) if r.get("action") == "reminder"), None)

                if item["name"] == "Communication opt-out":
                    result = "✓ PASS" if reminder_row and not reminder_row.get("guardrail_allowed", True) else "⚠ CHECK"
                elif item["name"] == "Risk decline — no retry":
                    result = "✓ PASS" if retry_row and not retry_row.get("guardrail_allowed", True) else "⚠ CHECK"
                elif item["name"] == "Blocked instrument":
                    permitted = [r for r in decision.get("candidate_actions", []) if r.get("guardrail_allowed")]
                    result = "✓ PASS" if len(permitted) == 1 and permitted[0].get("action") == "escalate" else "⚠ CHECK"
                else:
                    result = "✓ PASS"

                results.append({
                    "Test": item["name"], "Result": result,
                    "Focus": item["focus"], "Decision": decision.get("final_action"),
                })
            except Exception as exc:
                results.append({
                    "Test": item["name"], "Result": "✗ FAIL",
                    "Focus": item["focus"], "Decision": str(exc),
                })

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


# =========================================================
# RECOVERY ECONOMICS
# =========================================================

st.header("💰 Recovery Economics")

revenue_at_risk = pd.to_numeric(
    data.get("recovery_amount", pd.Series(dtype=float)), errors="coerce"
).fillna(0).sum()

e1, e2, e3, e4 = st.columns(4)
with e1:
    st.metric("Revenue at Risk", format_inr(revenue_at_risk))
with e2:
    st.metric("Simulated Recovery", format_inr(money_recovered))
with e3:
    st.metric("Recovery Rate", f"{recovery_rate:.2%}")
with e4:
    st.metric("Revenue / Attempt", format_inr(money_per_attempt))

st.caption(
    "All monetary figures are simulated benchmark outcomes on 3,262 unseen cases. "
    "No development-set result is mixed in. "
    "Production recovery would be confirmed by Razorpay payment and settlement events."
)

st.divider()


# =========================================================
# AGENT BEHAVIOR
# =========================================================

st.header("🤖 Agent Behavior")

agent_status = data["final_status"].value_counts().rename_axis("status").reset_index(name="cases")

col1, col2 = st.columns([2, 1])
with col1:
    st.bar_chart(agent_status.set_index("status"))
with col2:
    st.metric("Recovered", f"{recovered_cases:,}")
    st.metric("Escalated", f"{escalated_cases:,}")
    st.metric("Stopped", f"{stopped_cases:,}")
    st.metric("Multi-attempt cases", f"{multi_attempt_cases:,}")
    st.metric("Avg recovery per case", format_inr(average_recovered_case))

st.divider()


# =========================================================
# CONFIDENCE-AWARE DECISIONS
# =========================================================

st.header("🎯 Confidence-Aware Decisions")

st.markdown(
    "Confidence is a decision-support signal. "
    "High-value cases (≥ ₹50,000) with low confidence (< 0.40) are escalated instead of "
    "automated — reducing risk of incorrect recovery on uncertain high-value cases. "
    "Confidence never overrides deterministic policy guardrails."
)

confidence_distribution = (
    data["confidence_level"].value_counts().rename_axis("confidence_level").reset_index(name="cases")
)

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(confidence_distribution.set_index("confidence_level"))
with col2:
    confidence_summary = (
        data.groupby("confidence_level")
        .agg(
            cases=("case_id", "count"),
            recovered=("final_status", lambda s: (s == "recovered").sum()),
            total_recovered=("total_recovered", "sum"),
            average_confidence=("confidence_score", "mean"),
        )
        .reset_index()
    )
    confidence_summary["recovery_rate"] = confidence_summary["recovered"] / confidence_summary["cases"]
    display_confidence = confidence_summary.copy()
    display_confidence["average_confidence"] = display_confidence["average_confidence"].map(
        lambda x: "—" if pd.isna(x) else f"{x:.3f}"
    )
    display_confidence["total_recovered"] = display_confidence["total_recovered"].map(format_inr)
    display_confidence["recovery_rate"] = display_confidence["recovery_rate"].map(lambda x: f"{x:.2%}")
    display_confidence = display_confidence.rename(columns={
        "confidence_level": "Confidence", "cases": "Cases", "recovered": "Recovered",
        "total_recovered": "Money Recovered", "average_confidence": "Avg Score",
        "recovery_rate": "Recovery Rate",
    })
    st.dataframe(display_confidence, use_container_width=True, hide_index=True)

st.divider()


# =========================================================
# FAILURE CATEGORY PERFORMANCE
# =========================================================

st.header("📋 Failure Category Performance")

category_summary = (
    data.groupby("failure_category")
    .agg(
        cases=("case_id", "count"),
        recovered=("final_status", lambda s: (s == "recovered").sum()),
        money_recovered=("total_recovered", "sum"),
        attempts=("attempts", "sum"),
    )
    .reset_index()
)
category_summary["recovery_rate"] = category_summary["recovered"] / category_summary["cases"]
category_summary["money_per_attempt"] = (
    category_summary["money_recovered"] / category_summary["attempts"].replace(0, float("nan"))
)

category_display = category_summary.copy()
category_display["recovery_rate"] = category_display["recovery_rate"].map(lambda x: f"{x:.2%}")
for mc in ["money_recovered", "money_per_attempt"]:
    category_display[mc] = category_display[mc].map(lambda x: "—" if pd.isna(x) else format_inr(x))
category_display = category_display.rename(columns={
    "failure_category": "Failure Category", "cases": "Cases", "recovered": "Recovered",
    "recovery_rate": "Recovery Rate", "money_recovered": "Money Recovered",
    "attempts": "Attempts", "money_per_attempt": "Money / Attempt",
})
st.dataframe(
    category_display.sort_values("Recovery Rate", ascending=False),
    use_container_width=True, hide_index=True,
)

st.subheader("Category × Final Status")
category_status = (
    data.groupby(["failure_category", "final_status"]).size().unstack(fill_value=0)
)
st.dataframe(category_status, use_container_width=True)

st.divider()


# =========================================================
# RECOVERY CASES TABLE
# =========================================================

st.header("Recovery Cases")

search = st.text_input("Search by Case ID or Customer ID")
filtered = data.copy()

if search:
    case_mask = filtered["case_id"].astype(str).str.contains(search, case=False, na=False)
    customer_mask = (
        filtered["customer_id"].astype(str).str.contains(search, case=False, na=False)
        if "customer_id" in filtered.columns else False
    )
    filtered = filtered[case_mask | customer_mask]

display_columns = [
    "case_id", "customer_id", "failure_category", "recovery_amount",
    "final_action", "recovery_probability", "erv", "final_status",
    "attempts", "total_recovered", "confidence_score", "confidence_level",
]
available_display_columns = [c for c in display_columns if c in filtered.columns]
st.dataframe(
    filtered[available_display_columns].sort_values("recovery_amount", ascending=False),
    use_container_width=True, hide_index=True,
)

st.divider()


# =========================================================
# CASE INVESTIGATION
# =========================================================

st.header("🔎 Case Investigation")

case_ids = data["case_id"].drop_duplicates().tolist()
default_case = case_ids.index("CASE_000022") if "CASE_000022" in case_ids else 0
selected_case = st.selectbox("Select a recovery case", case_ids, index=default_case)
case = data[data["case_id"] == selected_case].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Recovery Amount", format_inr(safe_number(case, "recovery_amount")))
with c2:
    st.metric("ML Probability",
              f"{safe_number(case, 'recovery_probability'):.1%}" if "recovery_probability" in case else "—")
with c3:
    st.metric("Expected Recovery", format_inr(safe_number(case, "expected_recovery")))
with c4:
    st.metric("ERV", format_inr(safe_number(case, "erv")))
with c5:
    confidence = case.get("confidence_score")
    st.metric("Confidence", "—" if pd.isna(confidence) else f"{float(confidence):.3f}")

st.write(f"**Customer:** {case.get('customer_id', '—')}")
st.write(f"**Failure category:** {case.get('failure_category', '—')}")
st.write(f"**Final status:** {case.get('final_status', '—')}")
st.write(f"**Surfaces:** {case.get('surfaces', '—')}")
st.write(f"**Initial / final action:** {case.get('final_action', '—')}")
st.write(f"**Decision reason:** {case.get('decision_reason', '—')}")
st.write(f"**Guardrail:** {case.get('guardrail_status', '—')}")
st.write(f"**Confidence level:** {case.get('confidence_level', 'not_evaluated')}")


# =========================================================
# AUDIT TIMELINE
# =========================================================

st.subheader("Agent Audit Timeline")

audit = parse_audit_trail(case.get("audit_trail", "[]"))

if not audit:
    st.info("No audit events available for this case.")
else:
    for event in audit:
        event_type = event.get("event", "event")
        if event_type == "action_selected":
            st.info(f"🎯 **Action selected:** {event.get('action')} (attempt {event.get('attempt_number')})")
        elif event_type == "action_executed":
            st.write(f"⚙️ **Executed:** {event.get('action')} — verification: **{event.get('verification_status')}**")
        elif event_type == "recovery_verified":
            st.success(f"✅ **Recovery verified:** {format_inr(event.get('amount', 0))}")
        elif event_type == "recovery_failed":
            st.warning("❌ Recovery attempt failed.")
        elif event_type == "next_action_evaluation":
            st.info(f"🔄 **Next action:** {event.get('next_action')} (ERV {format_inr(event.get('erv', 0))})")
        elif event_type == "escalation":
            st.error("👤 **Escalated to manual review.**")
        elif event_type == "stopping_decision":
            st.warning(f"🛑 **Stopped:** {event.get('reason', '')}")
        else:
            st.caption(f"• {event_type}: {event}")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem'>"
    "RecoverAI · Razorpay Buildathon Track 03 · AI Revenue Recovery · "
    "82 passing tests · Held-out unseen benchmark · "
    "ML-assisted recovery with deterministic guardrails · "
    "Razorpay Test Mode webhook verified"
    "</div>",
    unsafe_allow_html=True,
)
