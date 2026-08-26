# RecoverAI - Synthetic Data Configuration

RANDOM_SEED = 20260825

# -----------------------------
# Dataset sizes
# -----------------------------

NUM_CUSTOMERS = 2000

NUM_PAYMENTS = 7000
NUM_SUBSCRIPTIONS = 2000
NUM_CHECKOUTS = 3500
NUM_INVOICES = 2000


# -----------------------------
# Transaction amount ranges (INR)
# -----------------------------

MIN_PAYMENT_AMOUNT = 500
MAX_PAYMENT_AMOUNT = 100000

MIN_SUBSCRIPTION_AMOUNT = 500
MAX_SUBSCRIPTION_AMOUNT = 25000

MIN_CHECKOUT_AMOUNT = 500
MAX_CHECKOUT_AMOUNT = 100000

MIN_INVOICE_AMOUNT = 1000
MAX_INVOICE_AMOUNT = 150000


# -----------------------------
# Recovery policy
# -----------------------------

MAX_AUTOMATIC_RETRIES = 2

RETRY_COOLDOWN_MINUTES = 30

MAX_AUTOMATED_CONTACT_ATTEMPTS = 2

RECOVERY_WINDOW_HOURS = 72

MANUAL_REVIEW_THRESHOLD = 50000


# -----------------------------
# Customer behavior
# -----------------------------

CUSTOMER_OPT_OUT_RATE = 0.08

NEW_CUSTOMER_RATE = 0.15

HIGH_VALUE_CUSTOMER_RATE = 0.15


# -----------------------------
# Data quality / noise
# -----------------------------

MISSING_VALUE_RATE = 0.03

AMBIGUOUS_FAILURE_RATE = 0.05

# -----------------------------
# Subscription configuration
# -----------------------------

SUBSCRIPTION_ACTIVE_RATE = 0.70
SUBSCRIPTION_FAILED_RATE = 0.20
SUBSCRIPTION_CANCELLED_RATE = 0.10

MIN_SUBSCRIPTION_AMOUNT = 500
MAX_SUBSCRIPTION_AMOUNT = 25000

# -----------------------------
# Checkout configuration
# -----------------------------

CHECKOUT_COMPLETED_RATE = 0.55
CHECKOUT_ABANDONED_RATE = 0.35
CHECKOUT_EXPIRED_RATE = 0.10

CHECKOUT_STAGES = [
    "payment_page",
    "address",
    "review",
    "payment_attempt",
]

# -----------------------------
# Invoice configuration
# -----------------------------

INVOICE_PAID_RATE = 0.70
INVOICE_OVERDUE_RATE = 0.22
INVOICE_CANCELLED_RATE = 0.08

MIN_INVOICE_AMOUNT = 1000
MAX_INVOICE_AMOUNT = 150000