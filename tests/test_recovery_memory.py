from src.model.recovery_memory import (
    append_verified_outcome,
    customer_recovery_memory,
    load_verified_outcomes,
)


def test_verified_feedback_creates_customer_memory(tmp_path):
    path = tmp_path / "feedback.jsonl"
    append_verified_outcome(
        {
            "case_id": "CASE_1",
            "customer_id": "CUSTOMER_1",
            "action": "reminder",
            "recovered": True,
            "verified_at": "2026-09-04T00:00:00+00:00",
        },
        path,
    )

    assert len(load_verified_outcomes(path)) == 1
    assert customer_recovery_memory("CUSTOMER_1", path)[
        "recovery_rate"
    ] == 1.0


def test_unverified_feedback_schema_is_rejected(tmp_path):
    try:
        append_verified_outcome(
            {"case_id": "CASE_1"},
            tmp_path / "feedback.jsonl",
        )
    except ValueError as error:
        assert "customer_id" in str(error)
    else:
        raise AssertionError("Incomplete feedback must be rejected")