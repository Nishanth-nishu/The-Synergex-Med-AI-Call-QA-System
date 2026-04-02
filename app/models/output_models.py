"""
Output Pydantic models for the Synergex Med Call QA API.
These models define the exact schema enforced by the LLM response_schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class OverallAssessment(str, Enum):
    PASS = "pass"
    NEEDS_REVIEW = "needs_review"
    ESCALATE = "escalate"


class FlagType(str, Enum):
    HIPAA_CONCERN = "hipaa_concern"
    MISINFORMATION = "misinformation"
    RUDENESS = "rudeness"
    PROTOCOL_VIOLATION = "protocol_violation"
    POSITIVE_INTERACTION = "positive_interaction"


class Severity(str, Enum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"
    POSITIVE = "positive"


class ComplianceFlag(BaseModel):
    """A single quality flag raised during analysis."""

    type: Annotated[
        FlagType,
        Field(description="Category of the flag: hipaa_concern | misinformation | rudeness | protocol_violation | positive_interaction"),
    ]
    severity: Annotated[
        Severity,
        Field(description="Impact level: critical | moderate | minor | positive"),
    ]
    description: Annotated[
        str,
        Field(
            description=(
                "1–2 sentence description of the specific issue or positive behavior observed. "
                "Be factual. If ambiguous, note the ambiguity explicitly."
            )
        ),
    ]
    transcript_excerpt: Annotated[
        str,
        Field(
            description=(
                "The exact portion of the transcript that supports this flag. "
                "Must be a direct quote from the transcript, not paraphrased."
            )
        ),
    ]


class AgentPerformance(BaseModel):
    """Quantitative and qualitative performance assessment of the agent."""

    professionalism_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Score from 0.0 to 1.0 rating the agent's tone, language, empathy, and courtesy. "
                "1.0 = exemplary professionalism."
            ),
        ),
    ]
    accuracy_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Score from 0.0 to 1.0 rating correctness of information provided. "
                "If no verifiable information was given, score based on what was said. "
                "1.0 = fully accurate."
            ),
        ),
    ]
    resolution_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Score from 0.0 to 1.0 rating whether the caller's issue was addressed or resolved. "
                "1.0 = fully resolved."
            ),
        ),
    ]
    strengths: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=3,
            description="1–3 specific things the agent did well, grounded in the transcript.",
        ),
    ]
    improvements: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=3,
            description="1–3 specific, actionable areas for improvement. Do not invent problems not in the transcript.",
        ),
    ]


class QAAnalysisResult(BaseModel):
    """
    Full quality analysis result for a single call transcript.
    This is the schema enforced as the LLM response_schema.
    """

    call_id: Annotated[str, Field(description="Echo of the input call_id for traceability")]
    overall_assessment: Annotated[
        OverallAssessment,
        Field(
            description=(
                "Overall assessment: 'pass' (no significant issues), "
                "'needs_review' (ambiguous or moderate issues requiring human check), "
                "'escalate' (critical: HIPAA violation, explicit rudeness, or dangerous misinformation CONFIRMED in transcript)."
            )
        ),
    ]
    assessment_reasoning: Annotated[
        str,
        Field(
            description=(
                "2–4 sentences explaining the overall assessment. "
                "Must reference specific evidence from the transcript. "
                "Separate factual observations from AI inferences."
            )
        ),
    ]
    compliance_flags: Annotated[
        list[ComplianceFlag],
        Field(
            description=(
                "List of compliance flags detected. May be empty if no issues found. "
                "Include positive_interaction flags for exemplary behavior. "
                "Only flag what is actually present in the transcript."
            )
        ),
    ]
    agent_performance: Annotated[
        AgentPerformance,
        Field(description="Quantitative and qualitative performance assessment"),
    ]
    escalation_required: Annotated[
        bool,
        Field(
            description=(
                "True ONLY if a critical-severity flag was detected (confirmed HIPAA violation, "
                "explicit rudeness, or dangerous misinformation). False otherwise."
            )
        ),
    ]
    escalation_reason: Annotated[
        str | None,
        Field(
            description=(
                "If escalation_required is true, explain specifically why. "
                "If false, this must be null."
            )
        ),
    ]


class QAAnalysisResponse(BaseModel):
    """API response wrapper for single call analysis."""

    success: bool = True
    data: QAAnalysisResult


class BatchQAAnalysisResponse(BaseModel):
    """API response wrapper for batch call analysis."""

    success: bool = True
    total: int
    results: list[QAAnalysisResult]


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: str
    detail: str | None = None
