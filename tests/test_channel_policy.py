from src.engine.channel_policy import choose_recovery_channel


def test_opted_in_customer_uses_preferred_channel():
    assert choose_recovery_channel(
        {"communication_opt_in": True, "preferred_channel": "whatsapp"},
        "reminder",
    ) == "whatsapp"


def test_opted_out_reminder_goes_to_manual_review():
    assert choose_recovery_channel(
        {"communication_opt_in": False},
        "reminder",
    ) == "manual_review"


def test_escalation_always_uses_manual_review():
    assert choose_recovery_channel(
        {"communication_opt_in": True, "preferred_channel": "sms"},
        "escalate",
    ) == "manual_review"