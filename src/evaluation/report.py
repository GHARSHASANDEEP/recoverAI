import os

import pandas as pd


BASELINE_PATH = (
    "data/processed/baseline_results.csv"
)

AGENT_PATH = (
    "data/processed/agent_results.csv"
)

ERV_PATH = (
    "data/processed/erv_scores.csv"
)

CASES_PATH = (
    "data/processed/recovery_cases.csv"
)

OUTPUT_PATH = (
    "data/processed/evaluation_report.csv"
)

DECISIONS_PATH = (
    "data/processed/decisions.csv"
)


def main():

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    agent = pd.read_csv(
        AGENT_PATH
    )

    erv = pd.read_csv(
        ERV_PATH
    )

    cases = pd.read_csv(
        CASES_PATH
    )

    decisions = pd.read_csv(
        DECISIONS_PATH
    )

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------

    baseline_cases = len(
        baseline
    )

    baseline_recovered = int(
        baseline["recovered"].sum()
    )

    baseline_money = float(
        baseline[
            "recovered_amount"
        ].sum()
    )

    agent_cases = len(
        agent
    )

    agent_recovered = int(
        (
            agent["final_status"]
            == "recovered"
        ).sum()
    )

    agent_money = float(
        agent[
            "total_recovered"
        ].sum()
    )

    baseline_rate = (
        baseline_recovered
        / baseline_cases
        if baseline_cases
        else 0.0
    )

    agent_rate = (
        agent_recovered
        / agent_cases
        if agent_cases
        else 0.0
    )

    incremental_money = (
        agent_money
        - baseline_money
    )

    improvement_pct = (
        incremental_money
        / baseline_money
        * 100
        if baseline_money
        else 0.0
    )

    # --------------------------------------------------
    # Agent metrics
    # --------------------------------------------------

    escalated = int(
        (
            agent["final_status"]
            == "escalated"
        ).sum()
    )

    stopped = int(
        (
            agent["final_status"]
            == "stopped"
        ).sum()
    )

    multi_attempt = int(
        (
            agent["attempts"]
            > 1
        ).sum()
    )

    total_attempts = int(
        agent["attempts"].sum()
    )

    # --------------------------------------------------
    # ERV metrics
    # --------------------------------------------------

    gross_risk = float(
        cases[
            "recovery_amount"
        ].sum()
    )

    total_erv = float(
        decisions["erv"].sum()
    )

    total_expected_recovery = float(
        decisions[
            "expected_recovery"
        ].sum()
    )

    total_action_cost = float(
        decisions[
            "action_cost"
        ].sum()
    )

    # --------------------------------------------------
    # Summary table
    # --------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "metric": "cases",
                "baseline": baseline_cases,
                "recoverAI": agent_cases,
            },
            {
                "metric": "recovered_cases",
                "baseline": baseline_recovered,
                "recoverAI": agent_recovered,
            },
            {
                "metric": "recovery_rate",
                "baseline": baseline_rate,
                "recoverAI": agent_rate,
            },
            {
                "metric": "money_recovered",
                "baseline": baseline_money,
                "recoverAI": agent_money,
            },
            {
                "metric": "incremental_money_recovered",
                "baseline": 0.0,
                "recoverAI": incremental_money,
            },
            {
                "metric": "improvement_percent",
                "baseline": 0.0,
                "recoverAI": improvement_pct,
            },
            {
                "metric": "gross_revenue_at_risk",
                "baseline": gross_risk,
                "recoverAI": gross_risk,
            },
            {
                "metric": "total_expected_recovery",
                "baseline": 0.0,
                "recoverAI": total_expected_recovery,
            },
            {
                "metric": "total_erv",
                "baseline": 0.0,
                "recoverAI": total_erv,
            },
            {
                "metric": "action_cost",
                "baseline": 0.0,
                "recoverAI": total_action_cost,
            },
            {
                "metric": "escalated_cases",
                "baseline": 0,
                "recoverAI": escalated,
            },
            {
                "metric": "stopped_cases",
                "baseline": 0,
                "recoverAI": stopped,
            },
            {
                "metric": "multi_attempt_cases",
                "baseline": 0,
                "recoverAI": multi_attempt,
            },
            {
                "metric": "total_recovery_attempts",
                "baseline": baseline_cases,
                "recoverAI": total_attempts,
            },
        ]
    )

    # --------------------------------------------------
    # Recovery by action
    # --------------------------------------------------

    action_summary = (
        decisions.groupby(
            "final_action"
        )
        .agg(
            cases=(
                "case_id",
                "count",
            ),
            expected_recovery=(
                "expected_recovery",
                "sum",
            ),
            action_cost=(
                "action_cost",
                "sum",
            ),
            erv=(
                "erv",
                "sum",
            ),
        )
        .reset_index()
    )

    action_summary = (
        action_summary.rename(
            columns={
                "final_action": "action"
            }
        )
    )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    action_summary.to_csv(
        "data/processed/evaluation_by_action.csv",
        index=False,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        "✓ Evaluation report generated."
    )

    print()

    print(
        "=============================="
    )

    print(
        "RECOVERAI EVALUATION"
    )

    print(
        "=============================="
    )

    print()

    print(
        f"Cases evaluated: "
        f"{agent_cases:,}"
    )

    print()

    print(
        "BASELINE"
    )

    print(
        f"Recovered cases: "
        f"{baseline_recovered:,}"
    )

    print(
        f"Recovery rate: "
        f"{baseline_rate:.2%}"
    )

    print(
        f"Money recovered: "
        f"₹{baseline_money:,.2f}"
    )

    print()

    print(
        "RECOVERAI"
    )

    print(
        f"Recovered cases: "
        f"{agent_recovered:,}"
    )

    print(
        f"Recovery rate: "
        f"{agent_rate:.2%}"
    )

    print(
        f"Money recovered: "
        f"₹{agent_money:,.2f}"
    )

    print()

    print(
        "BUSINESS IMPACT"
    )

    print(
        f"Incremental recovery: "
        f"₹{incremental_money:,.2f}"
    )

    print(
        f"Improvement over baseline: "
        f"{improvement_pct:.2f}%"
    )

    print()

    print(
        "AGENT BEHAVIOR"
    )

    print(
        f"Escalated: "
        f"{escalated:,}"
    )

    print(
        f"Stopped: "
        f"{stopped:,}"
    )

    print(
        f"Multi-attempt cases: "
        f"{multi_attempt:,}"
    )

    print(
        f"Total attempts: "
        f"{total_attempts:,}"
    )

    print()

    print(
        "ECONOMIC METRICS"
    )

    print(
        f"Revenue at risk: "
        f"₹{gross_risk:,.2f}"
    )

    print(
        f"Expected recovery: "
        f"₹{total_expected_recovery:,.2f}"
    )

    print(
        f"ERV: "
        f"₹{total_erv:,.2f}"
    )

    print(
        f"Action cost: "
        f"₹{total_action_cost:,.2f}"
    )

    print()

    print(
        "Recovery by action:"
    )

    print(
        action_summary.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()