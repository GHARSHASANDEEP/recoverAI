# RecoverAI Submission Checklist

## Public proof to show

- Public GitHub repository containing the latest code and README.
- Five-minute unlisted pitch video.
- Streamlit dashboard at `http://localhost:8501`.
- Razorpay Test Mode Payment Link marked paid.
- Signed webhook delivery returning HTTP 200.
- Judge Mode successful recovery.
- Judge Mode blocked `risk_decline` retry.
- Same-unseen-population baseline comparison.

## Five-minute flow

1. **Problem, 0:00-0:40:** Revenue loss spans failed payments, abandoned
   checkouts, halted subscriptions, and overdue invoices.
2. **Architecture, 0:40-1:20:** Event normalization, case construction,
   case-level recoverability, action-conditioned scoring, policy, guardrails,
   execution, verification, and audit trail.
3. **Successful case, 1:20-2:20:** Use `temporary_bank_failure`; show retry,
   verification, recovered status, state path, and audit events.
4. **Safe failure, 2:20-3:00:** Use `risk_decline`; show retry blocked and
   manual escalation.
5. **Business proof, 3:00-4:00:** Show the unseen benchmark comparison:
   RecoverAI 51.93% and INR 44.32M versus baseline 35.84% and INR 31.00M.
6. **Provider proof, 4:00-4:35:** Show Razorpay Test Mode Payment Link,
   signed webhook, duplicate protection, and HTTP 200 delivery.
7. **Judgment, 4:35-5:00:** State that benchmark outcomes are simulated,
   the model needs merchant calibration, and production actions require an
   approved Razorpay provider and verified settlement events.

## What broke

The first unseen run joined `UNSEEN_` cases to development customers. The
coverage check exposed zero matching customers. The pipeline was corrected to
use unseen customers, fail-fast on incomplete joins, and regenerate results.

## Honest claims

The INR amounts are deterministic simulated benchmark outcomes, not real
customer recovery. Razorpay Test Mode webhook delivery is real integration
validation, but the demo does not claim live customer charging. The action
model is a prototype ranking signal and must be calibrated with merchant
outcomes before production use.

## Local commands

```powershell
# Start webhook backend; enter the secret only in your terminal.
.\run_webhook.ps1 -WebhookSecret "YOUR_RAZORPAY_WEBHOOK_SECRET"

# In another terminal, expose port 8000 with an authenticated HTTPS tunnel.
ngrok http 8000

# Register the public URL plus this path in Razorpay Test Mode:
# /webhooks/razorpay

# Start the dashboard in another terminal.
.\.venv\Scripts\python.exe -m streamlit run app.py
```