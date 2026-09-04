"""Production boundary for payment recovery providers.

The buildathon uses the simulator for batch evaluation, but this module
also provides a working Razorpay Test Mode payment-link action that can
be called for real webhook-triggered recovery cases.

Live execution requires RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET to be set.
It is deliberately isolated: the batch benchmark never calls execute_live().
"""

import hashlib
import time
from dataclasses import dataclass
from os import environ
from typing import Optional


@dataclass(frozen=True)
class RazorpayProviderConfig:
    """Configuration metadata without exposing secrets in logs or UI."""

    key_id: str
    key_secret: str
    webhook_secret: str

    @classmethod
    def from_environment(cls) -> "RazorpayProviderConfig":
        values = {
            "key_id": environ.get("RAZORPAY_KEY_ID", ""),
            "key_secret": environ.get("RAZORPAY_KEY_SECRET", ""),
            "webhook_secret": environ.get("RAZORPAY_WEBHOOK_SECRET", ""),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "Razorpay provider is not configured; missing "
                + ", ".join(missing)
            )
        return cls(**values)


def _idempotency_key(case_id: str, action: str) -> str:
    """Stable idempotency key — same case+action always produces the same key."""
    return hashlib.sha256(f"{case_id}:{action}".encode()).hexdigest()[:32]


def create_payment_link(
    config: RazorpayProviderConfig,
    case: dict,
    timeout_seconds: int = 10,
) -> dict:
    """Create a Razorpay Test Mode payment link for a recovery reminder.

    This is the one real Razorpay API call in RecoverAI. It is only
    invoked for webhook-triggered cases where:
      - communication_opt_in is True
      - action is 'reminder'
      - credentials are configured

    The batch benchmark never calls this function.
    """
    import requests  # local import — not needed by the simulator path

    amount_paise = int(float(case.get("recovery_amount", 0)) * 100)
    if amount_paise <= 0:
        raise ValueError("recovery_amount must be positive to create a payment link")

    if not bool(case.get("communication_opt_in", False)):
        raise ValueError("Cannot create payment link: customer has not opted in")

    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "description": "Payment recovery — please complete your pending payment",
        "reminder_enable": True,
        "notify": {"sms": False, "email": False},  # merchant controls delivery
        "notes": {
            "case_id": str(case.get("case_id", "")),
            "customer_id": str(case.get("customer_id", "")),
            "failure_category": str(case.get("failure_category", "")),
            "source": "recoverai",
        },
    }

    response = requests.post(
        "https://api.razorpay.com/v1/payment_links",
        json=payload,
        auth=(config.key_id, config.key_secret),
        headers={
            "Idempotency-Key": _idempotency_key(
                str(case.get("case_id", "")), "reminder"
            ),
            "Content-Type": "application/json",
        },
        timeout=timeout_seconds,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Razorpay payment link creation failed: "
            f"HTTP {response.status_code} — {response.text[:200]}"
        )

    data = response.json()
    return {
        "provider": "razorpay",
        "action": "payment_link_created",
        "payment_link_id": data.get("id"),
        "short_url": data.get("short_url"),
        "status": data.get("status"),
        "amount_inr": amount_paise / 100,
        "case_id": case.get("case_id"),
        "execution_mode": "razorpay_test_mode_live",
    }


class RecoveryProviderBoundary:
    """Explicit seam between RecoverAI decisions and real side effects."""

    def __init__(self, config: Optional[RazorpayProviderConfig] = None):
        self._config = config

    def execute(self, action: str, case: dict) -> dict:
        """Execute a recovery action against the real Razorpay Test Mode API.

        Only 'reminder' is supported via payment link creation.
        All other actions raise to prevent unintended side effects.
        """
        if self._config is None:
            raise RuntimeError(
                "Live provider execution requires a RazorpayProviderConfig. "
                "Call RecoveryProviderBoundary(RazorpayProviderConfig.from_environment())."
            )

        if action == "reminder":
            return create_payment_link(self._config, case)

        raise RuntimeError(
            f"Live provider execution for action '{action}' is not implemented. "
            "Only 'reminder' (payment link creation) is supported in Test Mode."
        )
