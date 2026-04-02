"""
Core QA analyzer service.

This is the orchestration layer — it:
1. Builds the prompt (via prompt_builder)
2. Calls the LLM provider (Gemini or OpenAI)
3. Validates the result (post-LLM sanity checks)
4. Returns the structured QAAnalysisResult

Edge case handling:
- Very short calls (< 30s): notes limited confidence in the prompt
- Empty/trivial transcripts: returns a graceful needs_review result without LLM call
- Inconsistent escalation: if LLM says escalate=True but no critical flag exists,
  downgrade to needs_review and log a warning (guard against hallucination)
"""

from __future__ import annotations

import time

from app.models.input_models import CallTranscript
from app.models.output_models import (
    AgentPerformance,
    ComplianceFlag,
    FlagType,
    OverallAssessment,
    QAAnalysisResult,
    Severity,
)
from app.providers.base import LLMProvider, LLMProviderError
from app.services.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum transcript length (characters) to attempt full LLM analysis
_MIN_TRANSCRIPT_LENGTH = 20


class QAAnalyzer:
    """
    Orchestrates the full QA analysis pipeline for a single call transcript.
    Injected with an LLMProvider at construction time — provider-agnostic.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def analyze(self, call: CallTranscript) -> QAAnalysisResult:
        """
        Perform full quality analysis on a call transcript.

        Returns:
            QAAnalysisResult with all fields populated.

        Raises:
            LLMProviderError: If the LLM provider fails after retries.
        """
        start = time.monotonic()

        # ── Edge case: trivially short or empty transcript ──
        stripped = call.transcript.strip()
        if len(stripped) < _MIN_TRANSCRIPT_LENGTH:
            logger.warning(
                "transcript_too_short_for_analysis",
                call_id=call.call_id,
                transcript_length=len(stripped),
            )
            return self._trivial_result(call, reason="Transcript is too short for meaningful analysis.")

        logger.info(
            "analysis_started",
            call_id=call.call_id,
            agent=call.agent_name,
            department=call.department.value,
            duration_seconds=call.call_duration_seconds,
            provider=self._provider.provider_name,
            model=self._provider.model_name,
        )

        # ── Build prompts ──
        system_prompt = SYSTEM_PROMPT
        user_prompt = build_user_prompt(call)

        # ── Call LLM ──
        try:
            result = await self._provider.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                call_id=call.call_id,
            )
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(
                provider=self._provider.provider_name,
                message=f"Unexpected error during analysis of call_id={call.call_id}: {e}",
                original_error=e,
            ) from e

        # ── Post-LLM sanity checks ──
        result = self._validate_and_fix(result, call)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "analysis_complete",
            call_id=call.call_id,
            overall_assessment=result.overall_assessment.value,
            escalation_required=result.escalation_required,
            flag_count=len(result.compliance_flags),
            total_latency_ms=elapsed_ms,
        )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_and_fix(self, result: QAAnalysisResult, call: CallTranscript) -> QAAnalysisResult:
        """
        Apply post-LLM consistency checks to guard against hallucination.

        Rule 1: If escalation_required=True but no flag with severity=critical
                exists → downgrade to needs_review. The LLM cannot escalate
                without evidence.

        Rule 2: If overall_assessment=escalate but escalation_required=False
                → force escalation_required=True (internal consistency).

        Rule 3: If overall_assessment=pass but escalation_required=True
                → downgrade assessment to escalate (consistency).
        """
        has_critical = any(f.severity == Severity.CRITICAL for f in result.compliance_flags)

        # Rule 1: Escalation requires a critical flag
        if result.escalation_required and not has_critical:
            logger.warning(
                "escalation_downgraded_no_critical_flag",
                call_id=call.call_id,
                original_assessment=result.overall_assessment.value,
            )
            result = result.model_copy(update={
                "escalation_required": False,
                "overall_assessment": OverallAssessment.NEEDS_REVIEW,
                "escalation_reason": None,
                "assessment_reasoning": (
                    result.assessment_reasoning
                    + " [QA System Note: Escalation was downgraded to 'needs_review' "
                    "because no critical-severity flag was identified in the transcript.]"
                ),
            })

        # Rule 2: escalate assessment → ensure escalation_required=True
        if result.overall_assessment == OverallAssessment.ESCALATE and not result.escalation_required:
            result = result.model_copy(update={
                "escalation_required": True,
                "escalation_reason": result.escalation_reason or "Critical issue detected — see compliance flags.",
            })

        # Rule 3: pass assessment → cannot have escalation_required=True
        if result.overall_assessment == OverallAssessment.PASS and result.escalation_required:
            result = result.model_copy(update={
                "overall_assessment": OverallAssessment.ESCALATE,
            })

        return result

    @staticmethod
    def _trivial_result(call: CallTranscript, reason: str) -> QAAnalysisResult:
        """Return a safe default result for transcripts too short to analyze."""
        return QAAnalysisResult(
            call_id=call.call_id,
            overall_assessment=OverallAssessment.NEEDS_REVIEW,
            assessment_reasoning=(
                f"{reason} The call duration was {call.call_duration_seconds} seconds. "
                "Manual review is recommended to determine call quality."
            ),
            compliance_flags=[
                ComplianceFlag(
                    type=FlagType.PROTOCOL_VIOLATION,
                    severity=Severity.MINOR,
                    description=(
                        "Transcript is insufficient for automated quality analysis. "
                        "This may indicate the call was cut short, disconnected, or improperly recorded."
                    ),
                    transcript_excerpt=call.transcript.strip()[:200] or "[empty transcript]",
                )
            ],
            agent_performance=AgentPerformance(
                professionalism_score=0.5,
                accuracy_score=0.5,
                resolution_score=0.0,
                strengths=["Insufficient data to assess strengths"],
                improvements=["Ensure full call transcript is captured for quality review"],
            ),
            escalation_required=False,
            escalation_reason=None,
        )
