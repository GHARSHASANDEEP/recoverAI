# RecoverAI

**Razorpay Buildathon Track 03 — AI Revenue Recovery**

RecoverAI recovered **51.93% of failed payments** versus **35.84% for a one-retry baseline** — a
**+16.09 percentage-point improvement** — on 3,262 held-out unseen cases, recovering
**₹44,315,137 simulated** versus ₹30,996,494 for the baseline, a difference of **₹13,318,643**.

> These are deterministic simulated benchmark outcomes on an isolated unseen population.
> All customer identities, contact data, and payment records are synthetic. No real customer
> money was moved. Strategy comparisons use the checked-in unseen files and are reproducible.
> Production recovery is confirmed only by captured or paid Razorpay events, never authorization alone.

---

## The Core Idea

A payment failure is not a single problem. Insufficient funds needs a reminder. A timeout needs
a retry. A risk decline must never be retried automatically. An expired instrument needs the
customer to act first. A blocked instrument goes straight to manual review.

RecoverAI diagnoses the failure, selects the safest action from a failure-aware policy, enforces
deterministic guardrails, executes through a provider boundary, and verifies the outcome — all in
a closed loop with a full audit trail.

```
revenue-risk event
  → normalize + diagnose failure
  → build deduplicated recovery case
  → V3 ML recoverability score
  → action-conditioned recommender (retry vs reminder vs escalate)
  → failure-aware policy sequence
  → deterministic guardrails
  → provider execution
  → outcome verification
  → recovered · escalated · stopped
```

**ML recommends. Policy constrains. Guardrails authorize. Provider executes. Event verifies.**

---

## Measured Result

| Strategy | Recovery rate | Cases recovered | Simulated recovery |
|---|---:|---:|---:|
| RecoverAI | 51.93% | 1,694 | ₹44,315,137 |
| One-retry baseline | 35.84% | 1,169 | ₹30,996,494 |
| Difference | **+16.09 pp** | **+525** | **+₹13,318,643** |

RecoverAI also recorded 4,092 automated attempts, 1,480 safe escalations, and 88 stopped cases.
Every transition has an audit event.

---

## Why Not Just Use an LLM?

Payment recovery requires deterministic safety guarantees. An LLM cannot be trusted to:

- Never retry a risk decline
- Never contact an opted-out customer
- Never exceed the attempt budget
- Never mark a payment recovered without provider confirmation

RecoverAI uses ML where probability estimation adds value and deterministic rules where safety
is non-negotiable. The two layers are explicitly separated and tested independently.

---

## The AI Layer

Two complementary models:

- **V3 recoverability model** (`HistGradientBoosting`, accuracy 0.810, ROC-AUC 0.732, PR-AUC 0.902)
  estimates whether a revenue-risk case is recoverable.
- **Action recommender V2** scores retry, reminder, and escalation for the current case context
  using action-conditioned probability × amount − cost (Expected Recovery Value).

Neither model can override consent, risk policy, retry limits, or merchant guardrails.
The action recommender is a prototype ranking signal trained on the synthetic benchmark.
A production deployment would retrain on verified merchant outcomes.

---

## The Safety Layer

| Failure | First action | Recovery path |
|---|---|---|
| Temporary bank failure | Retry | Retry → reminder → escalate |
| Timeout | Retry | Retry → reminder → escalate |
| Insufficient funds | Reminder | Reminder → retry → escalate |
| Authentication failure | Reminder | Reminder → retry → escalate |
| Risk decline | Escalate | Escalate only — no retry |
| Blocked instrument | Escalate | Escalate only — no retry |
| Expired instrument | Reminder | Reminder → escalate |

Guardrails that cannot be overridden by any model score:

- Customer opted out → reminder blocked
- `risk_decline` or `blocked_instrument` → retry blocked
- Attempt limit reached → automated actions blocked, escalation allowed
- ERV ≤ 0 → automated actions blocked, escalation allowed
- High-value case (≥ ₹50,000) + low confidence (< 0.40) → confidence-gated escalation

---

## Razorpay Integration

Working Test Mode provider boundary with HMAC-SHA256 signature verification, INR amount
validation, consent checks, and event-ID idempotency before any case is created.

Supported events: `payment.failed`, `payment.authorized`, `payment.captured`,
`payment_link.paid`, `payment_link.expired`, `subscription.halted`, `invoice.expired`, `order.paid`.

A `payment.failed` event becomes a `recovery_ready` case with a policy-selected next action.
A `payment.captured` or `payment_link.paid` event becomes a verified `recovered` update, preserving
the original recovery case ID from Payment Link notes. `payment.authorized` remains observed until capture.

---

## System Architecture

```mermaid
flowchart LR
    A[Razorpay events<br/>payment.failed<br/>checkout abandoned<br/>subscription halted<br/>invoice overdue] --> B[Normalize + diagnose]
    B --> C[Build recovery case]
    C --> D[V3 ML<br/>Recoverability score]
    C --> E[Action ML<br/>Retry vs reminder vs escalate]
    D --> F[Failure-aware policy]
    E --> F
    F --> G{Deterministic guardrails}
    G -->|Blocked| H[Stop or escalate]
    G -->|Approved| I[Choose channel]
    I --> J[Execute via provider]
    J --> K[Verify outcome]
    K -->|Recovered| L[Recovered revenue]
    K -->|Not recovered| M[Reassess]
    M --> F
    L --> N[Audit trail + verified memory]
    H --> N
```

```mermaid
sequenceDiagram
    participant R as Razorpay Test Mode
    participant W as Webhook receiver
    participant A as RecoverAI agent
    participant V as Verification event

    R->>W: Signed payment event
    W->>W: Verify HMAC signature + event ID
    W->>A: Normalize payment.failed
    A->>A: Score → policy → guardrails
    A->>A: Create recovery_ready workflow
    V-->>A: payment.captured confirmation
    A->>A: Persist audit + verified outcome
```

---

## Run It Locally

Python 3.11+ recommended. All data, model artifacts, and evaluation outputs are checked in —
the dashboard runs without any external credentials.

```powershell
git clone https://github.com/GHARSHASANDEEP/recoverAI.git
cd recoverAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py src
```

87 tests covering policy, guardrails, state transitions, unseen-data joins, action scoring,
channels, memory, webhook signatures, idempotency, and Razorpay event payloads. 0 failures.

### Run the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`. Start with **Judge Mode** — build any case and watch the full
reasoning chain: diagnosis → ML scoring → policy → guardrail → channel → execution → audit trail.

Try `risk_decline` to see a blocked retry. Try `temporary_bank_failure` to see a successful recovery.

### Regenerate the unseen evaluation

```powershell
.\.venv\Scripts\python.exe -m src.data.unseen_generator
.\.venv\Scripts\python.exe -m src.engine.unseen_case_builder
.\.venv\Scripts\python.exe -m src.engine.unseen_erv_batch
.\.venv\Scripts\python.exe -m src.engine.unseen_decision_batch
.\.venv\Scripts\python.exe -m src.engine.unseen_agent_batch
.\.venv\Scripts\python.exe -m src.engine.unseen_baseline_batch
```

### Run the webhook receiver

```powershell
.\run_webhook.ps1 -WebhookSecret "YOUR_RAZORPAY_WEBHOOK_SECRET"
```

Listens on `http://127.0.0.1:8000`. Routes: `GET /health`, `POST /webhooks/razorpay`.
Expose via an authenticated HTTPS tunnel and register the URL in Razorpay Test Mode. The
receiver is intentionally observation-only: it verifies and records provider events; a caller
must separately invoke the Test Mode payment-link provider after policy approval.

---

## Repository Map

| Path | Contents |
|---|---|
| `app.py` | Streamlit dashboard — hero KPIs, Judge Mode, baseline comparison, audit timeline |
| `src/engine/` | Normalization, case building, policy, guardrails, ERV, agent, batch evaluation |
| `src/model/` | V3 training, V2 action recommender, confidence scoring, recovery memory |
| `src/integrations/` | Razorpay webhook receiver, event normalization, provider boundary |
| `src/data/` | Generators, taxonomy, outcome simulator |
| `tests/` | 87 regression tests |
| `models/` | Trained artifacts + V3 evaluation report |
| `data/unseen/processed/` | Held-out evaluation outputs |

---

## Learning Boundary

`src/model/recovery_memory.py` accepts only explicitly verified provider outcomes.
Simulator results are never silently written as merchant feedback.
A production deployment would retrain the action recommender on verified merchant outcomes.

---

## Project Status

Working Track 03 prototype. Reproducible unseen benchmark. Test Mode Razorpay webhook receiver.
Policy-controlled agent. 87 passing tests. 0 compile errors.

Does not claim simulated benchmark money is real revenue or that Test Mode is production processing.
