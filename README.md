# Synergex Med — AI Call Quality Analysis System

A FastAPI-based AI system that analyzes phone call transcripts and produces structured quality assessments. Built to replace a manual human QA process across 39+ clinic locations handling inbound calls from patients, law offices, and insurance companies.

---

## Quick Start

### 1. Clone / unzip and navigate to the project

```bash
cd synergex-qa
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 5. Run the server

```bash
uvicorn main:app --reload
```

### 6. Analyze a call

```bash
curl -X POST http://localhost:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d @samples/clean_call.json | python -m json.tool
```

### 7. Interactive API docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

---

## How It Works

### Architecture Overview

```
POST /analyze-call
        │
        ▼
  Input Validation (Pydantic)
        │
        ▼
  QAAnalyzer.analyze()
        │
        ├── build_user_prompt() ← Department-specific rules injected
        │
        ├── LLMProvider.analyze() ← Gemini 2.5 Flash (or OpenAI)
        │       └── response_schema=QAAnalysisResult  ← constrained decoding
        │
        ├── Post-LLM sanity checks (escalation consistency)
        │
        └── QAAnalysisResult (validated Pydantic object)
```

### Prompting Strategy

The prompt is built in two phases, inspired by **Chain-of-Thought Prompting** (Wei et al., 2022):

**Phase 1 — Evidence Anchoring:**
Before filling the schema, the model is instructed to *mentally note* specific evidence quotes for each potential issue category (HIPAA, rudeness, misinformation). This grounds the model's assessment in real transcript quotes rather than inferences, which is the single most effective guard against hallucination and false positives.

**Phase 2 — Structured Fill:**
The model fills the `QAAnalysisResult` schema using Gemini's `response_schema` constrained decoding. This means the Gemini API enforces the schema at the **token generation level** — it is physically impossible for the model to output invalid JSON or missing required fields.

**Key prompt design decisions:**

| Decision | Rationale |
|----------|-----------|
| Explicit PHI category list | Model knows exactly what counts as a HIPAA concern — no guessing |
| `"escalate"` threshold definition hardcoded | Prevents over-flagging and false escalations |
| `"if ambiguous, note it — do NOT assume the worst"` | Non-punitive, protects agents from false accusations |
| `"positive_interaction"` flag type mentioned prominently | Encourages balanced, non-punitive assessments |
| `transcript_excerpt` required for every flag | Forces citation of evidence — not paraphrase |
| `temperature=0` | Deterministic classification outputs |
| Department rules injected per-call | Context-aware analysis (Scheduling checks ≠ Records checks) |

### Department-Specific Rules

Each call's `department` field triggers specific rule injection into the prompt:

| Department | Key QA Rules |
|------------|-------------|
| **Scheduling** | Appointment date + time must be confirmed; callback/confirmation offered |
| **Onboarding** | Lien agreement must be mentioned; clear next steps given |
| **Helpdesk** | Issue must be resolved or escalated with reference/ticket; not left hanging |
| **Follow-Ups** | Next contact date/time must be established |
| **Records** | **Authorization must be verified BEFORE any PHI discussion** (critical HIPAA gate) |

### Edge Case Handling

| Scenario | Handling |
|----------|----------|
| **Very short call** (< 30s) | Warning injected into prompt; prefer `needs_review`; note confidence limitation |
| **Disconnected call** | Treated as ambiguous; agent graded on what was said; resolution_score reflects incompleteness |
| **Empty/trivial transcript** (< 20 chars) | LLM not called; returns `needs_review` with `protocol_violation` flag explaining missing data |
| **Ambiguous PHI disclosure** | Described with "appears to" language; escalation only if clearly confirmed |
| **No issues at all** | Returns `pass` with positive_interaction flags; no invented issues |

### Why Gemini 2.5 Flash?

- **`response_schema` constrained decoding**: Schema enforcement happens at the token generation level — zero JSON parsing failures, zero structure mismatches. This is architecturally superior to asking the model to "output JSON."
- **Free tier available**: Generous free quota suitable for development and demonstration.
- **Speed**: Flash tier is optimized for low latency — critical for real-time call monitoring.
- **Quality**: 2.5-generation model handles nuanced tone analysis and ambiguity well.

---

## Provider Abstraction — Swapping to OpenAI

The `LLMProvider` abstract base class makes provider swapping trivial:

```python
# app/providers/base.py — the contract
class LLMProvider(ABC):
    async def analyze(self, system_prompt, user_prompt, call_id) -> QAAnalysisResult: ...
```

To switch to OpenAI (gpt-4o-mini):

```bash
# In .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

No code changes required. Both providers implement the identical interface. Adding Anthropic Claude would require only a new `ClaudeProvider` class implementing `LLMProvider`.

---

## Running the Evaluation Script

```bash
# Terminal 1 — start server
uvicorn main:app --reload

# Terminal 2 — run evaluation
python eval/evaluate.py
```

Expected output:
```
✅ TEST PASSED — Clean scheduling call (pass)
✅ TEST PASSED — Records call with HIPAA violation (escalate)
✅ TEST PASSED — Short disconnected call (needs_review)
✅ BATCH TEST PASSED
🎉 ALL TESTS PASSED (4/4)
```

---

## API Reference

### `POST /analyze-call`

**Request:**
```json
{
  "call_id": "CALL-001",
  "agent_name": "Maria Santos",
  "call_date": "2026-04-01",
  "call_duration_seconds": 187,
  "department": "Scheduling",
  "transcript": "Agent: Thank you for calling...\nCaller: Hi, I need..."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "call_id": "CALL-001",
    "overall_assessment": "pass",
    "assessment_reasoning": "The agent handled the call professionally...",
    "compliance_flags": [
      {
        "type": "positive_interaction",
        "severity": "positive",
        "description": "Agent confirmed appointment details clearly.",
        "transcript_excerpt": "I've got you scheduled for Thursday, April 10th at 2:00 PM..."
      }
    ],
    "agent_performance": {
      "professionalism_score": 0.95,
      "accuracy_score": 0.90,
      "resolution_score": 1.0,
      "strengths": ["Warm, professional tone", "Proactively offered reminder info"],
      "improvements": ["Could have verified contact info for confirmation"]
    },
    "escalation_required": false,
    "escalation_reason": null
  }
}
```

### `POST /batch-analyze`

Accepts `{"calls": [...]}` with 1–50 call objects. Processes in parallel.

### `GET /health`

Returns `{"status": "ok", "service": "synergex-qa"}`.

---

## Logging & Observability

Every call produces structured log entries in two places:

- **Console**: Human-readable timestamped logs
- **`logs/qa_analysis.jsonl`**: Machine-readable JSONL for analytics/monitoring

Each analysis logs: `call_id`, `provider`, `model`, `latency_ms`, `overall_assessment`, `flag_count`.

Example JSONL entry:
```json
{
  "timestamp": "2026-04-01T12:00:01+00:00",
  "level": "INFO",
  "message": "analysis_complete",
  "call_id": "CALL-001-CLEAN",
  "overall_assessment": "pass",
  "escalation_required": false,
  "flag_count": 1,
  "total_latency_ms": 1842
}
```

---

## Project Structure

```
synergex-qa/
├── main.py                         # FastAPI app entry point
├── requirements.txt
├── .env.example
│
├── app/
│   ├── models/
│   │   ├── input_models.py         # CallTranscript, BatchCallTranscripts
│   │   └── output_models.py        # QAAnalysisResult, ComplianceFlag, etc.
│   ├── providers/
│   │   ├── base.py                 # Abstract LLMProvider interface
│   │   ├── gemini.py               # Gemini 2.5 Flash (primary)
│   │   └── openai_provider.py      # OpenAI gpt-4o-mini (swap demo)
│   ├── services/
│   │   ├── prompt_builder.py       # Department-aware two-phase prompts
│   │   └── qa_analyzer.py          # Orchestration + post-LLM validation
│   ├── api/
│   │   └── routes.py               # FastAPI route handlers
│   └── utils/
│       └── logger.py               # Structured JSONL logger
│
├── samples/
│   ├── clean_call.json             # Expected: pass
│   ├── problematic_call.json       # Expected: escalate (HIPAA + rudeness)
│   └── edge_case_call.json         # Expected: needs_review (disconnected)
│
├── eval/
│   └── evaluate.py                 # Schema + correctness evaluation script
│
└── logs/
    └── qa_analysis.jsonl           # Auto-generated structured logs
```

---

## Tradeoffs & Design Decisions

### Why `response_schema` over `instructor` / JSON mode?

`instructor` is excellent for multi-model support but adds a dependency and a parsing layer. Since this system targets Gemini as primary with OpenAI as a demonstrated swap, native SDK APIs (`response_schema` for Gemini, `beta.chat.completions.parse` for OpenAI) are simpler, more reliable, and have zero extra dependencies. The provider abstraction already handles the swap cleanly.

### Why `temperature=0`?

Quality analysis is a classification task, not a creative task. Determinism ensures that the same transcript always produces the same assessment, which is critical for:
- Agent trust (no random score variance)
- Debugging and reproducibility
- Fair comparison across calls

### Why async with thread pool for SDK calls?

The `google-generativeai` and `openai` SDKs are synchronous. Rather than blocking the FastAPI event loop, each LLM call is wrapped in `asyncio.get_event_loop().run_in_executor(None, ...)`, preserving async concurrency for the batch endpoint.

### Why post-LLM consistency checks?

LLMs can occasionally produce logically inconsistent outputs (e.g., `overall_assessment="escalate"` but `escalation_required=False`). The `_validate_and_fix()` method catches and corrects these internal inconsistencies with an audit trail in the logs. This is a defense-in-depth measure on top of schema enforcement.

### Tradeoff: no real-time HIPAA scrubbing

In a production system, transcripts would be scrubbed of PHI before sending to any cloud LLM. This proof-of-concept skips that step as it's out of scope. The system correctly *detects* HIPAA violations in the transcript content but does not sanitize the payload itself.

### Tradeoff: synchronous SDK calls vs. native async

Using thread pool executors is less efficient than a native async SDK. Google's `google-cloud-aiplatform` via Vertex AI offers async clients but requires GCP setup. For local development simplicity, the executor pattern is appropriate.

---

## Research References

1. **Wei et al. (2022)** — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" — basis for two-phase evidence-then-assessment prompting
2. **Google AI SDK docs** — `response_schema` constrained decoding for guaranteed JSON structure
3. **OpenAI Structured Outputs** — `beta.chat.completions.parse()` for native Pydantic integration
4. **HIPAA Safe Harbor § 164.514(b)** — 18 PHI identifier categories informing our detection taxonomy
5. **tenacity** — retry library with exponential backoff for resilient API calls
