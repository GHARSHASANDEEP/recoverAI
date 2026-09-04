"""Minimal local HTTP receiver for Razorpay Test Mode webhooks.

Run from the repository root:

    python -m src.integrations.webhook_server

Expose port 8000 through an HTTPS tunnel before registering the URL with
Razorpay. This receiver normalizes accepted events and persists the resulting
recovery workflow decision; it does not execute live payment actions itself.
"""

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.integrations.razorpay_events import (
    WebhookEventStore,
    parse_and_normalize_webhook,
)
from src.engine.recovery_policy import get_initial_action
from src.model.recovery_memory import append_verified_outcome


HOST = os.environ.get("RECOVERAI_WEBHOOK_HOST", "127.0.0.1")
PORT = int(os.environ.get("RECOVERAI_WEBHOOK_PORT", "8000"))
EVENT_LOG = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "razorpay_webhook_events.jsonl"
)
EVENT_ID_STORE = EVENT_LOG.with_name("razorpay_webhook_event_ids.txt")
RECOVERY_LOG = EVENT_LOG.with_name("razorpay_recovery_workflow.jsonl")

_EVENT_STORE = WebhookEventStore(EVENT_ID_STORE)


def process_webhook(
    raw_body: bytes,
    signature: str,
    event_id: str = "",
) -> dict:
    """Verify, deduplicate, normalize, and persist one webhook."""

    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
    if not secret:
        raise RuntimeError("RAZORPAY_WEBHOOK_SECRET is not configured")

    event = parse_and_normalize_webhook(
        raw_body,
        signature,
        secret,
        _EVENT_STORE,
        event_id,
    )
    if event is None:
        return {"status": "duplicate"}

    try:
        EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_LOG.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "received_at": datetime.now(timezone.utc).isoformat(),
                        "event": event,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    except Exception:
        _EVENT_STORE.release(event["delivery_id"])
        raise

    recovery_status = "observed"
    next_action = None
    if event["event_type"] in {
        "payment_failed",
        "subscription_failed",
    }:
        recovery_status = "recovery_ready"
        next_action = get_initial_action(event["failure_category"])
    elif event["event_status"] in {"captured", "paid"}:
        try:
            append_verified_outcome({
                "case_id": event["case_id"],
                "event_id": event["delivery_id"],
                "customer_id": event["customer_id"],
                "action": "payment_link",
                "recovered": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "amount": event["event_amount"],
                "source": "razorpay_webhook",
            })
        except Exception:
            _EVENT_STORE.release(event["delivery_id"])
            raise
        recovery_status = "recovered"

    workflow = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "case_id": event["case_id"],
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "customer_id": event["customer_id"],
        "status": recovery_status,
        "next_action": next_action,
        "verified": recovery_status == "recovered",
        "execution_mode": "provider_webhook_observation",
    }
    try:
        with RECOVERY_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(workflow, sort_keys=True) + "\n")
    except Exception:
        _EVENT_STORE.release(event["delivery_id"])
        raise

    return {
        "status": "accepted",
        "event": event,
        "recovery_workflow": workflow,
    }


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP endpoint compatible with Razorpay webhook POST delivery."""

    def _send_json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/webhooks/razorpay":
            self._send_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Invalid webhook body length")
            raw_body = self.rfile.read(length)
            result = process_webhook(
                raw_body,
                self.headers.get("X-Razorpay-Signature", ""),
                self.headers.get("x-razorpay-event-id", ""),
            )
            self._send_json(200, result)
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception:
            self._send_json(500, {"error": "Webhook processing failed"})

    def log_message(self, format_string: str, *args) -> None:
        print(f"[webhook] {format_string % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), WebhookHandler)
    print(f"RecoverAI webhook receiver listening on http://{HOST}:{PORT}")
    print("Webhook endpoint: /webhooks/razorpay")
    print("Health endpoint: /health")
    server.serve_forever()


if __name__ == "__main__":
    main()