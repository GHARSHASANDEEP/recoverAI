# RecoverAI - Payment Failure Taxonomy

FAILURE_TAXONOMY = {
    "temporary_bank_failure": {
        "description": "Temporary bank or payment-network issue.",
        "default_recovery_actions": [
            "retry",
            "reminder"
        ],
    },

    "timeout": {
        "description": "Payment processing timed out before completion.",
        "default_recovery_actions": [
            "retry"
        ],
    },

    "insufficient_funds": {
        "description": "Customer account does not have sufficient available funds.",
        "default_recovery_actions": [
            "reminder"
        ],
    },

    "authentication_failed": {
        "description": "Required customer authentication was unsuccessful.",
        "default_recovery_actions": [
            "reminder"
        ],
    },

    "expired_instrument": {
        "description": "Payment instrument such as a card has expired.",
        "default_recovery_actions": [
            "reminder"
        ],
    },

    "blocked_instrument": {
        "description": "Payment instrument is blocked or unusable.",
        "default_recovery_actions": [
            "escalate"
        ],
    },

    "risk_decline": {
        "description": "Transaction was declined because of risk or policy controls.",
        "default_recovery_actions": [
            "stop"
        ],
    },

    "limit_exceeded": {
        "description": "Transaction exceeded an applicable payment or account limit.",
        "default_recovery_actions": [
            "reminder",
            "escalate"
        ],
    },

    "unknown_failure": {
        "description": "Failure could not be confidently classified.",
        "default_recovery_actions": [
            "escalate"
        ],
    },
}


FAILURE_CATEGORIES = list(FAILURE_TAXONOMY.keys())


# Actions supported by RecoverAI.
ACTIONS = [
    "retry",
    "reminder",
    "escalate",
    "stop",
]

# Representative raw failure codes used by our synthetic benchmark.
#
# These are simulation labels, not claims that every code is an
# official NPCI/card-network response code.

RAW_FAILURE_CODE_MAP = {
    "BANK_TEMPORARY_ERROR": "temporary_bank_failure",
    "BANK_SERVICE_UNAVAILABLE": "temporary_bank_failure",

    "PROCESSING_TIMEOUT": "timeout",
    "NETWORK_TIMEOUT": "timeout",

    "INSUFFICIENT_BALANCE": "insufficient_funds",
    "FUNDS_NOT_AVAILABLE": "insufficient_funds",

    "AUTHENTICATION_FAILED": "authentication_failed",
    "CUSTOMER_AUTH_FAILED": "authentication_failed",

    "CARD_EXPIRED": "expired_instrument",
    "INSTRUMENT_EXPIRED": "expired_instrument",

    "CARD_BLOCKED": "blocked_instrument",
    "INSTRUMENT_BLOCKED": "blocked_instrument",

    "RISK_DECLINED": "risk_decline",
    "POLICY_DECLINED": "risk_decline",

    "LIMIT_EXCEEDED": "limit_exceeded",
    "TRANSACTION_LIMIT_EXCEEDED": "limit_exceeded",

    "UNKNOWN_ERROR": "unknown_failure",
    "UNSPECIFIED_DECLINE": "unknown_failure",
}

def normalize_failure_code(raw_failure_code: str) -> str:
    """
    Convert a raw synthetic failure code into a normalized
    RecoverAI failure category.
    """

    if not raw_failure_code:
        return "unknown_failure"

    return RAW_FAILURE_CODE_MAP.get(
        raw_failure_code,
        "unknown_failure"
    )