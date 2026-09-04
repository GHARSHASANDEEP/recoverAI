"""Translate Razorpay-style webhook payloads into RecoverAI events.

The adapter keeps provider-specific payload details at the boundary. It does
not execute payments or send customer communications.
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.taxonomy import normalize_failure_code


EVENT_TYPES = {
    "payment.failed": ("payment", "payment_failed", "failed"),
    "payment.authorized": ("payment", "payment_authorized", "authorized"),
    "payment.captured": ("payment", "payment_captured", "captured"),
    "subscription.halted": (
        "subscription",
        "subscription_failed",
        "failed",
    ),
    "invoice.expired": ("invoice", "invoice_overdue", "overdue"),
    "checkout.abandoned": ("checkout", "checkout_abandoned", "abandoned"),
    "order.paid": ("order", "order_paid", "paid"),
    "payment_link.paid": ("payment_link", "payment_link_paid", "paid"),
    "payment_link.partially_paid": (
        "payment_link",
        "payment_link_partially_paid",
        "partially_paid",
    ),
    "payment_link.expired": (
        "payment_link",
        "payment_link_expired",
        "expired",
    ),
}


def verify_webhook_signature(
    raw_body: bytes,
    signature: str,
    webhook_secret: str,
) -> bool:
    """Verify a Razorpay webhook signature before processing an event."""

    if not raw_body or not signature or not webhook_secret:
        return False

    expected = hmac.new(
        webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


class WebhookEventStore:
    """Small in-memory idempotency store for webhook processing.

    Replace this with a durable database or Redis store in production.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._processed_ids = set()
        if self.path and self.path.exists():
            self._processed_ids = {
                line.strip()
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

    def claim(self, event_id: str) -> bool:
        """Claim an event ID; return False when it was already processed."""

        if not event_id or event_id in self._processed_ids:
            return False

        self._processed_ids.add(event_id)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(event_id + "\n")
        return True


def parse_and_normalize_webhook(
    raw_body: bytes,
    signature: str,
    webhook_secret: str,
    event_store: WebhookEventStore,
    event_id: str = "",
) -> dict | None:
    """Verify, deduplicate, parse, and normalize one webhook body."""

    if not verify_webhook_signature(raw_body, signature, webhook_secret):
        raise ValueError("Invalid Razorpay webhook signature")

    payload = json.loads(raw_body.decode("utf-8"))
    provider_event_id = event_id or str(payload.get("id", ""))
    if not event_store.claim(provider_event_id):
        return None

    return normalize_webhook(payload)


def _entity(payload: dict) -> dict:
    payload_data = payload.get("payload", {})

    for key in (
        "payment",
        "subscription",
        "invoice",
        "checkout",
        "payment_link",
    ):
        candidate = payload_data.get(key, {})
        if isinstance(candidate, dict):
            entity = candidate.get("entity", candidate)
            if isinstance(entity, dict) and entity:
                return entity

    return {}


def _amount_inr(entity: dict) -> float:
    amount = entity.get("amount", 0.0)
    return float(amount) / 100.0 if amount else 0.0


def _mapping(value) -> dict:
    """Return mapping-shaped provider data or an empty mapping."""

    return value if isinstance(value, dict) else {}


def normalize_webhook(payload: dict) -> dict:
    """Normalize one supported Razorpay-style webhook payload.

    Amounts are expected in paise, as in Razorpay payment payloads. A custom
    checkout abandonment event may provide an amount already in rupees by
    setting ``amount_unit`` to ``INR``.
    """

    event_name = str(payload.get("event", "")).strip().lower()
    if event_name not in EVENT_TYPES:
        raise ValueError(f"Unsupported Razorpay event: {event_name or 'unknown'}")

    surface, event_type, event_status = EVENT_TYPES[event_name]
    entity = _entity(payload)
    customer_details = _mapping(entity.get("customer_details", {}))
    notes = _mapping(entity.get("notes", {}))
    customer_id = (
        entity.get("customer_id")
        or notes.get("customer_id")
        or customer_details.get("customer_id")
        or entity.get("id")
    )

    if not customer_id:
        raise ValueError("Razorpay event is missing customer_id")

    amount = _amount_inr(entity)
    if entity.get("amount_unit") == "INR":
        amount = float(entity.get("amount", 0.0))

    created_at = entity.get("created_at", payload.get("created_at"))
    if created_at is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    else:
        timestamp = datetime.fromtimestamp(
            float(created_at), tz=timezone.utc
        ).isoformat()

    error = _mapping(entity.get("error", {}))
    raw_failure_code = (
        error.get("code")
        or error.get("reason")
        or entity.get("reason")
    )

    return {
        "customer_id": str(customer_id),
        "event_id": str(entity.get("id") or payload.get("id") or "unknown"),
        "surface": surface,
        "event_type": event_type,
        "event_amount": amount,
        "event_status": event_status,
        "failure_category": normalize_failure_code(raw_failure_code),
        "event_timestamp": timestamp,
    }