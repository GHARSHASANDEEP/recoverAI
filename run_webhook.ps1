param(
    [Parameter(Mandatory = $true)]
    [string]$WebhookSecret
)

$env:RAZORPAY_WEBHOOK_SECRET = $WebhookSecret
& ".\.venv\Scripts\python.exe" -m src.integrations.webhook_server