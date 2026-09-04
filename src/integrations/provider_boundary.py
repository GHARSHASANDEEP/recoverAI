"""Production boundary for payment recovery providers.

The buildathon uses the simulator, but production actions must be delegated to
an approved provider implementation after credentials, idempotency, consent,
and settlement verification are configured.
"""

from dataclasses import dataclass
from os import environ


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


class RecoveryProviderBoundary:
    """Explicit seam between RecoverAI decisions and real side effects."""

    def execute(self, action: str, case: dict) -> dict:
        """Reject unconfigured live execution instead of pretending success."""

        raise RuntimeError(
            "Live provider execution is disabled in the benchmark. "
            "Inject an approved Razorpay provider implementation before use."
        )