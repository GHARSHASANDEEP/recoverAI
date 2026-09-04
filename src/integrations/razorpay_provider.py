"""Optional live Razorpay Payment Link provider.

This module is never used by the synthetic benchmark. A production caller must
explicitly construct it with credentials and choose when to create a payment
link for an approved retry or reminder action.
"""

from dataclasses import dataclass

import requests

from src.integrations.provider_boundary import RazorpayProviderConfig


@dataclass
class RazorpayLiveProvider:
    config: RazorpayProviderConfig
    base_url: str = "https://api.razorpay.com/v1"
    timeout_seconds: float = 10.0

    def build_payment_link_payload(
        self,
        case: dict,
        callback_url: str,
    ) -> dict:
        """Build a reviewable Payment Link request in paise."""

        if not callback_url:
            raise ValueError("callback_url is required for live recovery")

        amount = float(case.get("recovery_amount", 0.0))
        if amount <= 0:
            raise ValueError("recovery_amount must be positive")

        customer = {}
        if case.get("customer_name"):
            customer["name"] = str(case["customer_name"])
        if case.get("customer_email"):
            customer["email"] = str(case["customer_email"])
        if case.get("customer_contact"):
            customer["contact"] = str(case["customer_contact"])

        payload = {
            "amount": int(round(amount * 100)),
            "currency": "INR",
            "accept_partial": False,
            "description": "RecoverAI payment recovery",
            "reference_id": str(case.get("case_id", "recovery")),
            "callback_url": callback_url,
            "callback_method": "get",
            "reminder_enable": True,
            "notify": {
                "sms": bool(customer.get("contact")),
                "email": bool(customer.get("email")),
            },
        }
        if customer:
            payload["customer"] = customer
        return payload

    def create_payment_link(self, case: dict, callback_url: str) -> dict:
        """Create a live Payment Link after the caller's approval checks."""

        response = requests.post(
            f"{self.base_url}/payment_links",
            auth=(self.config.key_id, self.config.key_secret),
            json=self.build_payment_link_payload(case, callback_url),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()