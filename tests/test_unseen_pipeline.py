from pathlib import Path

import pandas as pd

from src.engine import unseen_agent_batch
from src.engine import unseen_baseline_batch
from src.engine import unseen_decision_batch


ROOT = Path(__file__).resolve().parents[1]


def test_unseen_batches_use_unseen_customer_source():
    expected = ROOT / "data" / "unseen" / "raw" / "customers.csv"

    assert Path(unseen_decision_batch.CUSTOMERS_PATH) == expected
    assert Path(unseen_agent_batch.CUSTOMERS_PATH) == expected
    assert Path(unseen_baseline_batch.CUSTOMERS_PATH) == expected


def test_all_unseen_cases_have_customer_context():
    cases = pd.read_csv(
        ROOT / "data" / "unseen" / "processed" / "recovery_cases.csv",
        usecols=["customer_id"],
    )
    customers = pd.read_csv(
        ROOT / "data" / "unseen" / "raw" / "customers.csv",
        usecols=["customer_id"],
    )

    assert cases["customer_id"].isin(customers["customer_id"]).all()


def test_razorpay_payment_failure_normalization():
    from src.integrations.razorpay_events import normalize_webhook

    event = normalize_webhook(
        {
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_test",
                        "customer_id": "cust_test",
                        "amount": 125000,
                        "created_at": 1725000000,
                        "error": {"code": "BANK_TEMPORARY_ERROR"},
                    }
                }
            },
        }
    )

    assert event["event_id"] == "pay_test"
    assert event["customer_id"] == "cust_test"
    assert event["event_amount"] == 1250.0
    assert event["failure_category"] == "temporary_bank_failure"


def test_action_recommender_is_action_conditioned():
    from src.model.action_recommender import predict_action_probability

    case = {
        "failure_category": "insufficient_funds",
        "customer_segment": "regular",
        "amount_bucket": "5k_25k",
        "attempt_number": 1,
        "recovery_amount": 10000,
        "customer_lifetime_value": 50000,
        "successful_payment_count": 8,
        "failed_payment_count": 2,
        "total_payment_attempts": 10,
        "communication_opt_in": True,
    }

    retry = predict_action_probability(case, "retry")
    reminder = predict_action_probability(case, "reminder")

    assert 0.0 <= retry <= 1.0
    assert 0.0 <= reminder <= 1.0
    assert retry != reminder


def test_live_provider_builds_explicit_payment_link_payload():
    from src.integrations.provider_boundary import RazorpayProviderConfig
    from src.integrations.razorpay_provider import RazorpayLiveProvider

    provider = RazorpayLiveProvider(
        RazorpayProviderConfig("key", "secret", "webhook")
    )
    payload = provider.build_payment_link_payload(
        {
            "case_id": "CASE_1",
            "recovery_amount": 1250.50,
            "customer_email": "customer@example.com",
        },
        "https://merchant.example/recovery/callback",
    )

    assert payload["amount"] == 125050
    assert payload["reference_id"] == "CASE_1"
    assert payload["customer"]["email"] == "customer@example.com"


def test_webhook_signature_and_idempotency():
    import hashlib
    import hmac

    from src.integrations.razorpay_events import (
        WebhookEventStore,
        parse_and_normalize_webhook,
    )

    body = b'{"id":"evt_1","event":"payment.failed","payload":{"payment":{"entity":{"id":"pay_1","customer_id":"cust_1","amount":1000}}}}'
    secret = "buildathon-secret"
    signature = hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    store = WebhookEventStore()

    event = parse_and_normalize_webhook(body, signature, secret, store)
    duplicate = parse_and_normalize_webhook(body, signature, secret, store)

    assert event["event_id"] == "pay_1"
    assert duplicate is None


def test_payment_link_paid_event_is_supported():
    from src.integrations.razorpay_events import normalize_webhook

    event = normalize_webhook(
        {
            "event": "payment_link.paid",
            "payload": {
                "payment_link": {
                    "entity": {
                        "id": "plink_1",
                        "amount": 500000,
                        "customer_details": {
                            "contact": "+919492556744",
                        },
                    }
                }
            },
        }
    )

    assert event["event_type"] == "payment_link_paid"
    assert event["event_status"] == "paid"
    assert event["event_amount"] == 5000.0
    assert event["customer_id"] == "plink_1"


def test_payment_link_notes_list_does_not_crash():
    from src.integrations.razorpay_events import normalize_webhook

    event = normalize_webhook(
        {
            "event": "payment_link.paid",
            "payload": {
                "payment_link": {
                    "entity": {
                        "id": "plink_2",
                        "amount": 1000,
                        "notes": [],
                    }
                }
            },
        }
    )

    assert event["event_id"] == "plink_2"


def test_webhook_creates_recovery_workflow(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import json

    from src.integrations import webhook_server

    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "workflow-secret")
    monkeypatch.setattr(
        webhook_server,
        "EVENT_LOG",
        tmp_path / "events.jsonl",
    )
    monkeypatch.setattr(
        webhook_server,
        "RECOVERY_LOG",
        tmp_path / "workflow.jsonl",
    )
    monkeypatch.setattr(
        webhook_server,
        "_EVENT_STORE",
        webhook_server.WebhookEventStore(),
    )
    body = json.dumps(
        {
            "id": "evt_workflow",
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_workflow",
                        "customer_id": "cust_workflow",
                        "amount": 100000,
                        "error": {"code": "BANK_TEMPORARY_ERROR"},
                    }
                }
            },
        }
    ).encode()
    signature = hmac.new(
        b"workflow-secret", body, hashlib.sha256
    ).hexdigest()

    result = webhook_server.process_webhook(body, signature, "evt_workflow")

    assert result["recovery_workflow"]["status"] == "recovery_ready"
    assert result["recovery_workflow"]["next_action"] == "retry"
    assert (tmp_path / "workflow.jsonl").exists()