# Synergex Med AI Call QA — Deep Dive

This document explains the repository’s architecture, end-to-end data pipeline, and why each stage exists.

## 1) System Goal

The system automates QA for healthcare call transcripts by converting unstructured dialog into a structured, auditable quality decision. It targets:

- HIPAA/compliance risk detection
- Service-quality scoring (professionalism/accuracy/resolution)
- Escalation routing (`pass` / `needs_review` / `escalate`)

## 2) End-to-End Workflow (Single Call)

1. **Request ingress** (`POST /analyze-call`) with typed input fields.
2. **Input validation** via Pydantic (`CallTranscript`).
3. **Orchestration layer** (`QAAnalyzer`) checks trivial edge cases and builds prompts.
4. **LLM provider call** through abstraction (`LLMProvider`): Gemini or OpenAI.
5. **Structured output enforcement** using `QAAnalysisResult` schema.
6. **Post-LLM consistency checks** (escalation sanity rules).
7. **Response return** as `QAAnalysisResponse`.
8. **Observability** logs to console + JSONL.

## 3) Data Pipeline by Stage (What + Why + Benefit)

### Stage A — Input Contract and Validation

- **What happens:** Incoming JSON is validated against `CallTranscript` and `Department` enum.
- **Why chosen:** Prevents malformed payloads from ever reaching expensive LLM steps.
- **Benefits:**
  - Predictable downstream behavior
  - Earlier and cheaper failure modes
  - Removes ambiguity around department-specific rules

### Stage B — API Routing and Dependency Injection

- **What happens:** FastAPI route fetches `QAAnalyzer` from app state.
- **Why chosen:** Keeps endpoint thin and business logic testable.
- **Benefits:**
  - Clean separation of concerns
  - Provider can be swapped without route changes

### Stage C — Orchestration (`QAAnalyzer`)

- **What happens:**
  - Rejects too-short transcripts to avoid fake confidence
  - Builds prompts
  - Calls provider
  - Applies post-LLM consistency guardrails
- **Why chosen:** Centralized policy enforcement independent of model behavior.
- **Benefits:**
  - Model-agnostic control plane
  - Better reliability than prompt-only control

### Stage D — Prompt Construction

- **What happens:**
  - Global system policy (strict anti-hallucination + escalation thresholds)
  - Department-specific rule injection (Scheduling/Records/etc.)
  - Duration-aware note for short calls
- **Why chosen:** QA criteria are domain-specific and context-sensitive.
- **Benefits:**
  - Lower false positives
  - Better precision per workflow
  - Balanced analysis (`positive_interaction` required)

### Stage E — LLM Execution via Provider Abstraction

- **What happens:** Provider implementation calls model API and parses schema result.
- **Why chosen:** Different providers offer different strengths/costs/latency.
- **Benefits:**
  - Easy provider switching (`gemini`/`openai`)
  - Retry and failure handling isolated from business logic

### Stage F — Structured Output / Pydantic Enforcement

- **What happens:** Response must match `QAAnalysisResult` exactly.
- **Why chosen:** Free-form output is brittle and hard to integrate safely.
- **Benefits:**
  - Strongly typed contract for downstream systems
  - Deterministic integration and easier testing
  - Lower parser breakage risk

### Stage G — Post-LLM Sanity Rules

- **What happens:**
  - Prevent escalate without critical flag
  - Align `overall_assessment` and `escalation_required`
- **Why chosen:** LLMs can still produce logically inconsistent but schema-valid outputs.
- **Benefits:**
  - Safety against silent policy drift
  - Trustworthy automation in high-stakes domains

### Stage H — Batch Path

- **What happens:** `/batch-analyze` runs calls concurrently with per-item failure isolation.
- **Why chosen:** Real operations process call sets, not just one call.
- **Benefits:**
  - Throughput scaling
  - One bad call does not fail the full batch

### Stage I — Observability

- **What happens:** Structured logs include latency, model/provider metadata, token usage.
- **Why chosen:** LLM systems need production diagnostics and audit traces.
- **Benefits:**
  - Easier incident triage
  - Better cost/performance analysis

## 4) How This Repo Addresses Your Capability Checklist

### 4.1 “Can you follow a detailed spec and deliver exactly what was asked?”

**Evidence in design:**
- Explicit input/output contracts and enums
- Department-specific deterministic rule injection
- Post-LLM rule validation to enforce policy

**Interpretation:** The architecture is intentionally spec-first, not prompt-only.

### 4.2 “Can you control LLM output reliably using structured outputs / Pydantic?”

**Evidence in design:**
- Providers request schema-constrained output into `QAAnalysisResult`
- API routes also enforce typed response models

**Interpretation:** Control is done at both generation and application boundaries.

### 4.3 “Is prompting strategy thoughtful (false positives + edge cases)?”

**Evidence in design:**
- “Do not invent issues”, “do not assume worst” constraints
- Explicit escalation threshold definitions
- Edge-case behavior for short or trivial transcripts
- Department rules as contextual priors

**Interpretation:** Prompting is tuned for precision and non-punitive QA.

### 4.4 “Does the system distinguish real issues vs minor imperfections?”

**Evidence in design:**
- Assessment classes (`pass`, `needs_review`, `escalate`)
- Severity levels (`critical`, `moderate`, `minor`, `positive`)
- Escalation hard-gated by critical evidence

**Interpretation:** Triage is explicitly multi-level, not binary.

### 4.5 “Clean backend code with separation of concerns?”

**Evidence in design:**
- API layer (routes), orchestration (service), prompts (builder), providers, models, utils

**Interpretation:** This is cleanly modular and extensible.

### 4.6 “Can decisions be explained clearly in README?”

**Evidence in design/docs:**
- README explains architecture, prompting rationale, edge handling, provider swap, and evaluation approach.

## 5) Why These Choices Make Sense for Medical Call QA

- **Healthcare risk profile:** false escalations hurt staff; false negatives hurt compliance.
- **Typed outputs:** mandatory for auditability and downstream automation.
- **Layered safeguards:** prompt constraints + schema constraints + post-check rules improve robustness.
- **Department-aware logic:** mirrors real QA operations where success criteria differ by workflow.

## 6) Practical Extension Possibilities

1. **Human-in-the-loop review queue** for `needs_review` with adjudication feedback loop.
2. **Calibration set + confusion metrics** per flag type and department.
3. **Prompt/version registry** to track policy revisions.
4. **Evidence quality scoring** (e.g., excerpt specificity, quote fidelity).
5. **Policy engine extraction** (move escalation rules into declarative config).
6. **Security hardening**:
   - PHI redaction at log boundary
   - encrypted transcript storage
   - role-based access around outputs
7. **Cost controls**:
   - cache for duplicate transcripts
   - adaptive model tiering based on call complexity
8. **Analytics dashboard** from JSONL logs:
   - escalation rate by department/agent
   - drift detection in `needs_review` share

## 7) Known Trade-offs

- LLM judgment still depends on transcript quality.
- No native confidence score yet (only indirect signals via reasoning and flags).
- Strong schema does not guarantee truth; it guarantees structure and consistency.
- Prompt-based policy can drift without regression tests against labeled data.

## 8) Suggested “Production-Ready” Next Steps

1. Build a labeled validation dataset and run routine regression evals.
2. Add deterministic policy tests for escalation edge cases.
3. Add PII-safe logging middleware.
4. Version prompts and models with explicit rollout/rollback.
5. Add queue-based async processing for high call volumes.

