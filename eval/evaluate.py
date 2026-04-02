"""
Evaluation script for the Synergex Med Call QA System.

Runs all 3 sample transcripts against the running API, validates:
1. Response schema conforms to QAAnalysisResult
2. Expected overall_assessment values match
3. escalation_required correctness
4. Schema validation via Pydantic model_validate

Usage:
    # First start the server: uvicorn main:app --reload
    python eval/evaluate.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.output_models import OverallAssessment, QAAnalysisResult

# ──────────────────────────────────────────────────────────────────────────────
# Test Case Definitions
# ──────────────────────────────────────────────────────────────────────────────

SAMPLES_DIR = Path(__file__).parent.parent / "samples"

TEST_CASES = [
    {
        "file": "clean_call.json",
        "expected_assessment": OverallAssessment.PASS,
        "expected_escalation": False,
        "description": "Clean scheduling call — professional, no issues",
        "notes": "Should produce 'pass' with high professionalism/resolution scores and at least one positive_interaction flag.",
    },
    {
        "file": "problematic_call.json",
        "expected_assessment": OverallAssessment.ESCALATE,
        "expected_escalation": True,
        "description": "Records call with HIPAA violation + explicit rudeness",
        "notes": "Must flag hipaa_concern (critical) and rudeness. Must escalate.",
    },
    {
        "file": "edge_case_call.json",
        "expected_assessment": OverallAssessment.NEEDS_REVIEW,
        "expected_escalation": False,
        "description": "Short disconnected follow-up call — ambiguous outcome",
        "notes": "Should note call disconnection, limited analysis confidence, no critical flags.",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_sample(filename: str) -> dict[str, Any]:
    path = SAMPLES_DIR / filename
    with open(path) as f:
        return json.load(f)


def print_header(text: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}")


def print_result_summary(result: QAAnalysisResult) -> None:
    print(f"  overall_assessment  : {result.overall_assessment.value.upper()}")
    print(f"  escalation_required : {result.escalation_required}")
    print(f"  escalation_reason   : {result.escalation_reason}")
    print(f"  professionalism     : {result.agent_performance.professionalism_score:.2f}")
    print(f"  accuracy            : {result.agent_performance.accuracy_score:.2f}")
    print(f"  resolution          : {result.agent_performance.resolution_score:.2f}")
    print(f"  flags ({len(result.compliance_flags)}):")
    for flag in result.compliance_flags:
        emoji = "🔴" if flag.severity.value == "critical" else "🟡" if flag.severity.value == "moderate" else "🟢" if flag.severity.value == "positive" else "⚪"
        print(f"    {emoji} [{flag.severity.value}] {flag.type.value}: {flag.description[:80]}...")
    print(f"  reasoning: {result.assessment_reasoning[:120]}...")


def run_single_test(
    client: httpx.Client,
    base_url: str,
    test_case: dict[str, Any],
    idx: int,
) -> bool:
    """Run a single test case. Returns True if passed, False if failed."""
    print_header(f"TEST {idx}: {test_case['description']}")
    print(f"  File: {test_case['file']}")
    print(f"  Notes: {test_case['notes']}")

    # Load sample
    payload = load_sample(test_case["file"])
    call_id = payload.get("call_id", "UNKNOWN")

    # Call API
    print(f"\n  📡 POST {base_url}/analyze-call (call_id={call_id})")
    try:
        response = client.post(f"{base_url}/analyze-call", json=payload, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"  ❌ HTTP Error {e.response.status_code}: {e.response.text[:200]}")
        return False
    except httpx.RequestError as e:
        print(f"  ❌ Connection Error: {e}")
        print("     Is the server running? Try: uvicorn main:app --reload")
        return False

    raw = response.json()

    # Schema validation
    print("\n  🔍 Validating response schema...")
    try:
        result = QAAnalysisResult.model_validate(raw["data"])
        print("  ✅ Schema validation PASSED")
    except Exception as e:
        print(f"  ❌ Schema validation FAILED: {e}")
        return False

    # Print summary
    print_result_summary(result)

    # Assertion checks
    passed = True

    # Check overall_assessment
    if result.overall_assessment == test_case["expected_assessment"]:
        print(f"\n  ✅ overall_assessment = '{result.overall_assessment.value}' (expected: '{test_case['expected_assessment'].value}')")
    else:
        print(f"\n  ❌ overall_assessment = '{result.overall_assessment.value}' (expected: '{test_case['expected_assessment'].value}')")
        passed = False

    # Check escalation_required
    if result.escalation_required == test_case["expected_escalation"]:
        print(f"  ✅ escalation_required = {result.escalation_required} (expected: {test_case['expected_escalation']})")
    else:
        print(f"  ❌ escalation_required = {result.escalation_required} (expected: {test_case['expected_escalation']})")
        passed = False

    # Check escalation consistency
    if result.escalation_required and result.escalation_reason is None:
        print("  ❌ escalation_reason is None but escalation_required=True — inconsistency!")
        passed = False
    elif not result.escalation_required and result.escalation_reason is not None:
        print("  ⚠️  escalation_reason is set but escalation_required=False (minor inconsistency)")
    else:
        print("  ✅ escalation_reason consistency OK")

    # Check score ranges
    perf = result.agent_performance
    scores_valid = all(
        0.0 <= s <= 1.0
        for s in [perf.professionalism_score, perf.accuracy_score, perf.resolution_score]
    )
    if scores_valid:
        print("  ✅ All scores in valid range [0.0, 1.0]")
    else:
        print("  ❌ Score(s) out of valid range!")
        passed = False

    print(f"\n  {'✅ TEST PASSED' if passed else '❌ TEST FAILED'}")
    return passed


def run_batch_test(client: httpx.Client, base_url: str) -> bool:
    """Test the batch endpoint with all 3 samples."""
    print_header("BATCH TEST: /batch-analyze with all 3 samples")
    payloads = [load_sample(tc["file"]) for tc in TEST_CASES]

    try:
        response = client.post(
            f"{base_url}/batch-analyze",
            json={"calls": payloads},
            timeout=180.0,
        )
        response.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"  ❌ Batch request failed: {e}")
        return False

    raw = response.json()
    total = raw.get("total", 0)
    results = raw.get("results", [])

    print(f"  📦 Received {total} results")
    if total != 3:
        print(f"  ❌ Expected 3 results, got {total}")
        return False

    # Validate each schema
    passed = True
    for i, r in enumerate(results):
        try:
            QAAnalysisResult.model_validate(r)
            print(f"  ✅ Result {i+1} schema OK ({r.get('call_id', '?')}): {r.get('overall_assessment', '?')}")
        except Exception as e:
            print(f"  ❌ Result {i+1} schema FAILED: {e}")
            passed = False

    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Synergex QA API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip the batch endpoint test",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    print(f"\n🏥 Synergex Med Call QA System — Evaluation Suite")
    print(f"   Target: {base_url}")
    print(f"   Tests: {len(TEST_CASES)} individual + {'0' if args.skip_batch else '1'} batch")

    # Health check
    with httpx.Client() as client:
        try:
            health = client.get(f"{base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f"   Server: ✅ ONLINE ({health.json()})")
        except Exception as e:
            print(f"   Server: ❌ OFFLINE — {e}")
            print("   Please start the server: uvicorn main:app --reload")
            sys.exit(1)

        # Individual tests
        results = []
        for i, test_case in enumerate(TEST_CASES, start=1):
            passed = run_single_test(client, base_url, test_case, i)
            results.append(passed)

        # Batch test
        if not args.skip_batch:
            batch_passed = run_batch_test(client, base_url)
            results.append(batch_passed)

    # Final summary
    print_header("EVALUATION SUMMARY")
    total = len(results)
    passed_count = sum(results)
    for i, (passed, tc) in enumerate(zip(results[: len(TEST_CASES)], TEST_CASES), start=1):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Test {i}: {status} — {tc['description']}")
    if not args.skip_batch:
        batch_status = "✅ PASS" if results[-1] else "❌ FAIL"
        print(f"  Batch : {batch_status} — /batch-analyze with all 3 samples")

    print(f"\n  Result: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("  🎉 ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("  ⚠️  SOME TESTS FAILED — review output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
