 # RecoverAI

RecoverAI is an explainable revenue-recovery agent for Razorpay-style payment
and receivables events. It detects revenue at risk, diagnoses the failure,
selects a safe intervention, executes a bounded recovery workflow, verifies
the result, and escalates unresolved cases with an audit trail.

This project was built for Razorpay Buildathon Track 03: **AI Revenue
Recovery**.

## What It Demonstrates

- Payment failures, failed subscriptions, checkout abandonment, and overdue
	invoices are normalized into one recovery-event schema.
- Related payment and checkout events are deduplicated into one case.
- A V3 case-level model estimates recoverability without using action or
	outcome leakage as a feature.
- A separately identified action-conditioned recommender compares retry,
	reminder, and escalation probabilities for the next-best-action view.
- A verified-outcome memory boundary records provider-confirmed outcomes for
	future merchant-specific calibration without treating simulator results as
	production feedback.
- A consent-aware channel policy routes approved reminders to a preferred
	WhatsApp, SMS, or email channel and sends opt-outs or escalations to manual
	review.
- A failure-aware policy determines the safe recovery sequence.
- Deterministic guardrails control opt-in, retry eligibility, positive expected
	recovery value, attempt limits, and escalation.
- The agent executes and verifies actions through a state machine.
- Every decision and state transition is recorded in an audit trail.
- Results are compared with a one-retry baseline on an isolated unseen dataset.

## Architecture

```text
Razorpay-style events or synthetic data
				|
				v
Normalization and failure taxonomy
				|
				v
Deduplication and recovery-case construction
				|
				+--> V3 recoverability model
				|
				v
Failure-aware policy --> guardrails --> bounded agent
																			|
																			v
															execute and verify
																			|
																			v
												 recovered, stopped, or escalated
```

The V3 model supplies a case-level recoverability signal. The action
recommender is trained on the synthetic benchmark action rows and is a
prototype ranking signal, not a production-trained policy. Neither model
overrides policy or guardrails. This keeps high-risk actions explainable and
auditable.

### Decision Ownership

The system uses controlled autonomy rather than unconstrained automation:

```text
ML models recommend recoverability and next-best action
	-> policy defines the permitted recovery sequence
	-> guardrails enforce consent, safety, value, and attempt limits
	-> provider adapter performs an approved side effect
	-> Razorpay webhook verifies the outcome
	-> recovery memory stores only verified provider feedback
```

This separation is intentional. The ML model cannot authorize a risky retry,
contact an opted-out customer, bypass an attempt limit, or mark a payment
recovered without a provider-confirmed event.

## Run The Demo

### Prerequisites

- Windows, macOS, or Linux
- Python 3.11 or newer
- Git
- Test Mode Razorpay account only for webhook testing

Create a virtual environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use `source .venv/bin/activate` and `python3` instead.

### Generate And Evaluate

Run these commands from the repository root. The unseen commands overwrite
files under `data/unseen/processed`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.data.unseen_generator
.\.venv\Scripts\python.exe -m src.engine.unseen_case_builder
.\.venv\Scripts\python.exe -m src.engine.unseen_erv_batch
.\.venv\Scripts\python.exe -m src.engine.unseen_decision_batch
.\.venv\Scripts\python.exe -m src.engine.unseen_agent_batch
.\.venv\Scripts\python.exe -m src.engine.unseen_baseline_batch
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard reads only `data/unseen/processed` for its primary metrics.
Development evaluation files remain available as a reference, but are not
mixed into the unseen KPIs.

### Local Webhook Demo

Set the Razorpay Test Mode webhook secret in the same terminal used to start
the receiver. Never commit the secret:

```powershell
.\run_webhook.ps1 -WebhookSecret "YOUR_RAZORPAY_WEBHOOK_SECRET"
```

In a second terminal, expose port 8000 with an authenticated HTTPS tunnel:

```powershell
ngrok http 8000
```

Register the public URL plus `/webhooks/razorpay` in Razorpay Test Mode. Check
the receiver with `http://127.0.0.1:8000/health`. Accepted events and workflow
decisions are written to ignored local files under `data/processed`.

### Test The Project

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py src
```

Do not commit `.env`, webhook logs, API keys, webhook secrets, or ngrok
tokens. The repository includes `.env.example` only as a names-and-placeholders
reference.

### Production Deployment

Test Mode is the correct mode for this demo. It cannot accept real customer
payments and must not be described as production. To deploy the same design
for real merchants, deploy `src.integrations.webhook_server` or an equivalent
HTTPS service on a stable host, store Live Mode Razorpay credentials in the
host's secret manager, configure a durable event store, and replace the
simulator executor with an approved provider implementation. Enable live
actions only after consent checks, idempotency, callback handling, and
settlement verification are tested. Never put live keys in source, `.env`
files committed to Git, screenshots, or videos.

## Recovery Policy

Examples of policy behavior:

| Failure | First action | Safe sequence |
| --- | --- | --- |
| Temporary bank failure | Retry | Retry, reminder, escalate |
| Timeout | Retry | Retry, reminder, escalate |
| Insufficient funds | Reminder | Reminder, retry, escalate |
| Authentication failure | Reminder | Reminder, retry, escalate |
| Risk decline | Escalate | Escalate |
| Blocked instrument | Escalate | Escalate |
| Expired instrument | Reminder | Reminder, retry, escalate |

The agent stops when communication is not permitted, an automated action is
not economically positive, or the bounded recovery path is exhausted.

## Razorpay Event Boundary

`src/integrations/razorpay_events.py` translates supported webhook-style
events into RecoverAI's internal event contract:

- `payment.failed`
- `payment.authorized`
- `payment.captured`
- `payment_link.paid`
- `payment_link.expired`
- `subscription.halted`
- `invoice.expired`
- `checkout.abandoned` (merchant-side event)

The adapter only normalizes events. Payment execution and messaging remain
behind the executor and policy boundaries, which makes idempotency, provider
credentials, and production approval explicit integration concerns.

Webhook processing verifies the signature and rejects duplicate event IDs
before normalization. The provider boundary deliberately refuses live side
effects until a configured, approved Razorpay implementation is injected.

`src/integrations/razorpay_provider.py` includes an optional live Payment Link
client with INR-to-paise conversion, request timeouts, customer notification
flags, and case reference IDs. It is intentionally not called by the demo;
the merchant must provide credentials, callback handling, consent checks, and
settlement verification before enabling it.

For a local Test Mode webhook demo, configure
`RAZORPAY_WEBHOOK_SECRET`, run
`python -m src.integrations.webhook_server`, and expose port 8000 through an
HTTPS tunnel. Register the resulting URL ending in `/webhooks/razorpay`.
The receiver verifies `X-Razorpay-Signature`, rejects duplicates across
restarts, and stores normalized events in
`data/processed/razorpay_webhook_events.jsonl`. Use `run_webhook.ps1` to start
it without putting secrets in source files.

Accepted failure events now create a persisted `recovery_ready` workflow with
the policy-defined next action. Authorized, captured, and paid events create a
verified `recovered` workflow update. This is provider-event orchestration;
the benchmark simulator remains separate, and live side effects still require
an explicitly injected provider.

`src/model/recovery_memory.py` is the learning boundary. Only explicitly
verified outcomes can be recorded, and the customer memory summary can be
used to calibrate future intervention ranking. No simulated benchmark result
is silently written into this feedback store.

## Evaluation

The evaluation uses a separate unseen population with `UNSEEN_` identifiers.
The current checked-in run contains 3,262 cases and compares RecoverAI with a
one-retry baseline. Re-run the pipeline before presenting metrics so the
numbers match the current model and generated data.

The important business metrics are:

- verified recovery rate
- verified recovered amount
- incremental recovered amount versus baseline
- automated attempts and action cost
- stopped and escalated cases
- audit-event coverage

The synthetic outcome simulator is a benchmark mechanism, not a claim about
real-world payment success rates. A production deployment would replace it
with verified Razorpay payment and settlement events. The action recommender
also requires retraining or calibration on merchant outcomes before it can
be used for production intervention ranking.

## Submission Readiness

The repository is ready for a Track 03 prototype submission. Demonstrate the
following in the five-minute video:

1. A `temporary_bank_failure` case that recovers through retry and verification.
2. A `risk_decline` case where retry is blocked and manual escalation wins.
3. A Razorpay Test Mode Payment Link marked paid and its signed webhook
	returning HTTP 200.
4. The unseen benchmark comparison: RecoverAI 51.93% recovery and INR 44.32M
	versus the one-retry baseline at 35.84% and INR 31.00M.
5. The explicit disclaimer that benchmark outcomes are simulated and Test
	Mode is not production payment processing.

The submission still requires a public GitHub repository, a five-minute
unlisted video, eligibility confirmation, and rotation of any credentials
that were exposed during local testing. The temporary ngrok URL is for the
recording only; a production deployment needs a stable HTTPS host.

## Repository Map

- `src/data`: synthetic generators, taxonomy, hidden benchmark outcomes
- `src/engine`: normalization, deduplication, policy, guardrails, agent, and
	batch workflows
- `src/model`: model training, prediction, and comparison utilities
- `src/evaluation`: business metric reports
- `src/integrations`: provider event adapters
- `tests`: policy, safety, state-machine, confidence, and pipeline tests
- `app.py`: Streamlit dashboard and Judge Mode
