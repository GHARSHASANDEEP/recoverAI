"""Consent-aware channel selection for recovery interventions."""


CHANNELS = {"whatsapp", "sms", "email", "manual_review"}


def choose_recovery_channel(case: dict, action: str) -> str:
    """Choose a delivery channel without bypassing consent or escalation."""

    if action == "escalate":
        return "manual_review"

    if not bool(case.get("communication_opt_in", False)):
        return "manual_review" if action == "reminder" else "payment_rail"

    preferred = str(case.get("preferred_channel", "")).strip().lower()
    if preferred in CHANNELS - {"manual_review"}:
        return preferred

    if case.get("customer_contact"):
        return "whatsapp"
    if case.get("customer_email"):
        return "email"
    return "manual_review" if action == "reminder" else "payment_rail"