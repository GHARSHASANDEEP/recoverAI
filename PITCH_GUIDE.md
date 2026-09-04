# RecoverAI: Five-Minute Pitch Guide

## One-sentence positioning

RecoverAI is a safety-first AI revenue-recovery agent that diagnoses failed payments, chooses a bounded intervention, verifies the result, and escalates when recovery is unsafe.

## Recording setup

- Show the Streamlit dashboard at the start, with the browser URL hidden if it contains local machine details.
- Use the checked-in unseen benchmark numbers shown on the dashboard.
- Do not call simulated money real revenue.
- Keep the video between 4:30 and 5:00.
- Keep the GitHub repository URL visible in the final frame.

## Timed script and screen actions

### 0:00-0:30: The problem

Say:

> Failed payments are not one problem. A timeout may deserve a retry, insufficient funds may need a reminder, and a risk decline must not be retried automatically. A blind retry system either leaves revenue behind or creates avoidable customer and compliance risk. RecoverAI closes that loop safely.

Show the dashboard title and the recovery pipeline.

### 0:30-1:05: The product

Say:

> RecoverAI takes a revenue-risk event, normalizes and diagnoses it, scores recoverability, applies a failure-aware policy, checks deterministic guardrails, selects a channel, executes through a provider boundary, and verifies recovery from a provider event. The key design rule is simple: ML recommends, policy constrains, guardrails authorize, and provider events verify.

Point to the architecture section and the safety guardrails.

### 1:05-1:45: The measured result

Say:

> On 3,262 held-out unseen synthetic cases, RecoverAI recovered 1,694 cases, or 51.93 percent. A one-retry baseline recovered 1,169 cases, or 35.84 percent. That is a 16.09 percentage-point improvement, 525 additional cases, and INR 13.32 million more simulated recovery. RecoverAI recorded 4,092 attempts, 1,480 safe escalations, and 88 stopped cases.

Then say:

> These are deterministic simulator outcomes, not real customer money. The comparison is still useful because both strategies run on the same unseen population and the same revenue-at-risk distribution.

Show the benchmark comparison table and the simulated-results disclaimer.

### 1:45-2:35: A successful case

In Judge Mode, choose:

- Failure: `temporary_bank_failure`
- Amount: INR 20,000
- Customer opted in: enabled
- Attempt: 1

Click **Run RecoverAI**.

Say:

> This is a transient bank failure. The policy starts with retry, the model supplies a probability and economic value, and the guardrail authorizes the action because it is permitted, consent-safe, within the attempt budget, and economically positive. The simulator then verifies the outcome and records every state transition in the audit trail.

Show diagnosis, model scores, policy action, guardrail result, final status, and audit events.

### 2:35-3:20: A safety case

Run a second case:

- Failure: `risk_decline`
- Amount: INR 20,000
- Customer opted in: enabled

Say:

> The important behavior is what the system refuses to do. A risk decline is not retried automatically, even if a model score were favorable. Policy permits escalation only, and the guardrail blocks retry. This is a bounded financial workflow, not an LLM improvising a money action.

Then run or show the opt-out stress test and say:

> Consent is another hard boundary. An opted-out customer cannot receive a reminder. These controls are deterministic and tested independently from the model.

### 3:20-4:05: Razorpay integration and verification

Say:

> At the provider boundary, RecoverAI verifies the raw Razorpay webhook body with HMAC-SHA256, validates INR amounts, deduplicates event delivery, preserves the recovery case ID, and records verified outcomes only from captured or paid events. Authorization alone remains observed and is not counted as recovered. The webhook receiver is intentionally observation-only; the Test Mode payment-link provider is a separate approved execution boundary.

Show the Razorpay integration section and, if available, the webhook test output. Do not imply production processing or real money movement.

### 4:05-4:35: What broke and how it was fixed

Say:

> During development, I found two trust failures. The checked-in evaluation report used an older 3,312-case population while the dashboard used 3,262 unseen cases. I made the unseen population canonical and regenerated the report. I also found that an authorization event could be treated as recovery and that a failed feedback write could still be acknowledged. I separated authorization from capture, made verified feedback idempotent and retry-safe, and added regression tests.

Show the test count and GitHub repository.

### 4:35-5:00: Close

Say:

> RecoverAI is not a retry button. It is a measurable recovery workflow with economic reasoning, failure-aware policies, consent and risk guardrails, provider verification, and an audit trail. On the same unseen population, it recovered 525 more cases than a one-retry baseline while explicitly stopping or escalating unsafe cases. The repository is public, reproducible, and covered by 87 tests. Thank you.

End on the dashboard benchmark table or repository page.

## Judge questions and short answers

### Is the benchmark real?

No. The identities, payments, and outcomes are synthetic. The benchmark is a controlled comparison on a held-out unseen population. Production outcomes would require verified Razorpay events and merchant data.

### What is the AI doing?

The V3 model estimates case recoverability. The action-conditioned model ranks expected action value. Neither model can override policy, consent, risk restrictions, retry limits, or economic guardrails.

### Why not use an LLM to decide?

Payment actions need deterministic safety guarantees. An LLM can help explain or summarize, but it should not authorize retries, customer contact, or recovery claims.

### Does the webhook execute payments?

No. It verifies and records Razorpay Test Mode events. The payment-link provider is a separate execution boundary that requires explicit credentials and an approved action.

### What would change for production?

Use a transactional database or Redis for cross-process idempotency, durable workflow state, provider-side idempotency, merchant-specific verified outcomes, monitoring, and human approval policies for higher-risk actions.

## Submission checklist

- [ ] Eligibility and in-person Bangalore availability confirmed.
- [ ] Public repository URL works in an incognito browser.
- [ ] Fresh clone can install dependencies, run tests, and launch the dashboard.
- [ ] Five-minute video is unlisted or public and under five minutes.
- [ ] Video shows one successful case and one blocked or escalated case.
- [ ] Video says `simulated` for every benchmark money claim.
- [ ] Video explains ML versus policy and guardrails.
- [ ] Video explains the failure story and the fixes.
- [ ] Application uses the exact repository URL and project name.
