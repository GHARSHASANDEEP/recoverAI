"""End-to-end webhook integration tests.

Covers: signature verification -> idempotency -> normalization ->
recovery workflow decision -> memory boundary trigger.
"""

import hashlib
import hmac
import json

import pytest

from src.integrations.razorpay_events import (
    WebhookEventStore,
    parse_and_normalize_webhook,
    verify_webhook_signature,
)
from src.integrations.webhook_server import process_webhook


SECRET = "test_webhook_secret_abc123"


def _sign(body: bytes) -> str:
    return hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _failed_payload(payment_id="pay_t001", customer_id="CUST_00001",
                    amount=2000000, error_code="BANK_TEMPORARY_ERROR"):
    return {
        "event": "payment.failed",
        "id": f"evt_{payment_id}",
        "payload": {"payment": {"entity": {
            "id": payment_id,
            "customer_id": customer_id,
            "amount": amount,
            "currency": "INR",
            "created_at": 1700000000,
            "error": {"code": error_code, "reason": error_code},
        }}},
    }


def _captured_payload(payment_id="pay_t002", customer_id="CUST_00001", amount=2000000):
    return {
        "event": "payment.captured",
        "id": f"evt_{payment_id}",
        "payload": {"payment": {"entity": {
            "id": payment_id,
            "customer_id": customer_id,
            "amount": amount,
            "currency": "INR",
            "created_at": 1700000100,
        }}},
    }


def _authorized_payload(payment_id="pay_auth001", customer_id="CUST_00001", amount=2000000):
    payload = _captured_payload(payment_id, customer_id, amount)
    payload["event"] = "payment.authorized"
    return payload


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

def test_valid_signature_accepted():
    body = b'{"event":"payment.failed"}'
    assert verify_webhook_signature(body, _sign(body), SECRET) is True


def test_invalid_signature_rejected():
    body = b'{"event":"payment.failed"}'
    assert verify_webhook_signature(body, "bad_sig", SECRET) is False


def test_empty_signature_rejected():
    body = b'{"event":"payment.failed"}'
    assert verify_webhook_signature(body, "", SECRET) is False


def test_wrong_secret_rejected():
    body = b'{"event":"payment.failed"}'
    assert verify_webhook_signature(body, _sign(body), "wrong_secret") is False


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_duplicate_event_rejected(tmp_path):
    store = WebhookEventStore(tmp_path / "ids.txt")
    assert store.claim("evt_001") is True
    assert store.claim("evt_001") is False


def test_different_events_both_accepted(tmp_path):
    store = WebhookEventStore(tmp_path / "ids.txt")
    assert store.claim("evt_001") is True
    assert store.claim("evt_002") is True


def test_idempotency_persists_across_instances(tmp_path):
    path = tmp_path / "ids.txt"
    WebhookEventStore(path).claim("evt_persist")
    assert WebhookEventStore(path).claim("evt_persist") is False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_payment_failed_normalizes_correctly(tmp_path):
    payload = _failed_payload()
    body = json.dumps(payload).encode()
    store = WebhookEventStore(tmp_path / "ids.txt")
    event = parse_and_normalize_webhook(body, _sign(body), SECRET, store, "evt_n001")
    assert event["event_type"] == "payment_failed"
    assert event["event_status"] == "failed"
    assert event["failure_category"] == "temporary_bank_failure"
    assert event["customer_id"] == "CUST_00001"
    assert event["event_amount"] == 20000.0


def test_payment_captured_normalizes_correctly(tmp_path):
    payload = _captured_payload()
    body = json.dumps(payload).encode()
    store = WebhookEventStore(tmp_path / "ids.txt")
    event = parse_and_normalize_webhook(body, _sign(body), SECRET, store, "evt_n002")
    assert event["event_type"] == "payment_captured"
    assert event["event_status"] == "captured"


def test_unsupported_event_raises(tmp_path):
    payload = {"event": "refund.created", "id": "evt_r001", "payload": {}}
    body = json.dumps(payload).encode()
    store = WebhookEventStore(tmp_path / "ids.txt")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_and_normalize_webhook(body, _sign(body), SECRET, store, "evt_r001")


def test_completely_empty_entity_raises(tmp_path):
    payload = {
        "event": "payment.failed",
        "id": "evt_nc001",
        "payload": {"payment": {"entity": {}}},
    }
    body = json.dumps(payload).encode()
    store = WebhookEventStore(tmp_path / "ids.txt")
    with pytest.raises(ValueError, match="customer_id"):
        parse_and_normalize_webhook(body, _sign(body), SECRET, store, "evt_nc001")


# ---------------------------------------------------------------------------
# Full process_webhook path
# ---------------------------------------------------------------------------

def _patch_server(monkeypatch, tmp_path):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr("src.integrations.webhook_server.EVENT_LOG",
                        tmp_path / "events.jsonl")
    monkeypatch.setattr("src.integrations.webhook_server.EVENT_ID_STORE",
                        tmp_path / "ids.txt")
    monkeypatch.setattr("src.integrations.webhook_server.RECOVERY_LOG",
                        tmp_path / "recovery.jsonl")
    monkeypatch.setattr("src.integrations.webhook_server._EVENT_STORE",
                        WebhookEventStore(tmp_path / "ids.txt"))


def test_payment_failed_produces_recovery_ready(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    body = json.dumps(_failed_payload()).encode()
    result = process_webhook(body, _sign(body), "evt_f001")
    assert result["status"] == "accepted"
    assert result["recovery_workflow"]["status"] == "recovery_ready"
    assert result["recovery_workflow"]["next_action"] == "retry"


def test_payment_captured_produces_recovered(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    body = json.dumps(_captured_payload()).encode()
    result = process_webhook(body, _sign(body), "evt_c001")
    assert result["status"] == "accepted"
    assert result["recovery_workflow"]["status"] == "recovered"
    assert result["recovery_workflow"]["verified"] is True


def test_payment_authorized_is_not_recovery_confirmation(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    body = json.dumps(_authorized_payload()).encode()
    result = process_webhook(body, _sign(body), "evt_auth001")
    assert result["recovery_workflow"]["status"] == "observed"
    assert result["recovery_workflow"]["verified"] is False


def test_memory_failure_is_not_acknowledged_and_can_retry(tmp_path, monkeypatch):
    from src.integrations import webhook_server

    _patch_server(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "src.integrations.webhook_server.append_verified_outcome",
        lambda outcome, **_: (_ for _ in ()).throw(OSError("disk full")),
    )
    body = json.dumps(_captured_payload()).encode()
    with pytest.raises(OSError, match="disk full"):
        process_webhook(body, _sign(body), "evt_retry001")
    assert webhook_server._EVENT_STORE.claim("evt_retry001") is True


def test_payment_link_case_id_is_preserved(tmp_path):
    from src.integrations.razorpay_events import normalize_webhook

    event = normalize_webhook({
        "event": "payment_link.paid",
        "payload": {"payment_link": {"entity": {
            "id": "plink_case001",
            "amount": 1000,
            "currency": "INR",
            "notes": {"case_id": "CASE_original"},
            "customer_id": "cust_1",
        }}},
    })
    assert event["case_id"] == "CASE_original"


def test_invalid_event_is_not_claimed(tmp_path):
    payload = {"event": "refund.created", "id": "evt_invalid"}
    body = json.dumps(payload).encode()
    store = WebhookEventStore(tmp_path / "ids.txt")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_and_normalize_webhook(body, _sign(body), SECRET, store)
    assert store.claim("evt_invalid") is True


def test_duplicate_webhook_returns_duplicate(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    body = json.dumps(_failed_payload()).encode()
    process_webhook(body, _sign(body), "evt_dup001")
    result = process_webhook(body, _sign(body), "evt_dup001")
    assert result["status"] == "duplicate"


def test_captured_webhook_writes_memory_boundary(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    feedback_path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(
        "src.integrations.webhook_server.append_verified_outcome",
        lambda outcome, **_: feedback_path.open("a").write(
            json.dumps(outcome) + "\n"
        ),
    )
    body = json.dumps(_captured_payload()).encode()
    result = process_webhook(body, _sign(body), "evt_mem001")
    assert result["recovery_workflow"]["status"] == "recovered"
    assert feedback_path.exists()
    record = json.loads(feedback_path.read_text().strip())
    assert record["recovered"] is True
    assert record["customer_id"] == "CUST_00001"


def test_invalid_signature_raises(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    body = json.dumps(_failed_payload()).encode()
    with pytest.raises(ValueError, match="signature"):
        process_webhook(body, "bad_signature", "evt_sig001")


def test_risk_decline_produces_escalate_action(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    payload = _failed_payload(
        payment_id="pay_risk001",
        error_code="RISK_DECLINED",
    )
    body = json.dumps(payload).encode()
    result = process_webhook(body, _sign(body), "evt_risk001")
    assert result["recovery_workflow"]["next_action"] == "escalate"
