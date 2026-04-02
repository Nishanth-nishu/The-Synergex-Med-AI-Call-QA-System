"""
FastAPI routes for the Synergex Med QA API.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.models.input_models import BatchCallTranscripts, CallTranscript
from app.models.output_models import (
    BatchQAAnalysisResponse,
    ErrorResponse,
    QAAnalysisResponse,
    QAAnalysisResult,
)
from app.providers.base import LLMProviderError
from app.services.qa_analyzer import QAAnalyzer
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def get_analyzer(request: Request) -> QAAnalyzer:
    """Dependency injection: retrieve the QAAnalyzer from app state."""
    return request.app.state.analyzer


@router.post(
    "/analyze-call",
    response_model=QAAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a single call transcript",
    description=(
        "Receives a phone call transcript and returns a structured quality analysis "
        "including compliance flags, agent performance scores, and escalation determination."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error in request body"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable after retries"},
    },
    tags=["Quality Analysis"],
)
async def analyze_call(
    call: CallTranscript,
    analyzer: QAAnalyzer = Depends(get_analyzer),
) -> QAAnalysisResponse:
    """
    POST /analyze-call

    Analyze a single call transcript for quality, compliance, and agent performance.
    """
    logger.info("analyze_call_request", call_id=call.call_id, agent=call.agent_name)

    try:
        result = await analyzer.analyze(call)
    except LLMProviderError as e:
        logger.error("llm_provider_error", call_id=call.call_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM provider failed: {e}",
        ) from e
    except Exception as e:
        logger.error("unexpected_analysis_error", call_id=call.call_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during analysis: {e}",
        ) from e

    return QAAnalysisResponse(success=True, data=result)


@router.post(
    "/batch-analyze",
    response_model=BatchQAAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze multiple call transcripts in parallel",
    description=(
        "Accepts a list of call transcripts (1–50) and returns quality analysis for all. "
        "Calls are processed in parallel. Errors on individual calls are included in results "
        "as 'needs_review' with an error note rather than failing the entire batch."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error in request body"},
    },
    tags=["Quality Analysis"],
)
async def batch_analyze(
    batch: BatchCallTranscripts,
    analyzer: QAAnalyzer = Depends(get_analyzer),
) -> BatchQAAnalysisResponse:
    """
    POST /batch-analyze

    Analyze multiple call transcripts in parallel.
    Individual call failures are handled gracefully — the batch will not fail
    due to a single call error.
    """
    logger.info("batch_analyze_request", call_count=len(batch.calls))

    async def safe_analyze(call: CallTranscript) -> QAAnalysisResult:
        """Analyze a call, returning an error result on failure instead of raising."""
        try:
            return await analyzer.analyze(call)
        except LLMProviderError as e:
            logger.error("batch_call_failed", call_id=call.call_id, error=str(e))
            return QAAnalyzer._trivial_result(
                call,
                reason=f"Analysis failed due to LLM provider error: {e}",
            )
        except Exception as e:
            logger.error("batch_call_unexpected_error", call_id=call.call_id, error=str(e))
            return QAAnalyzer._trivial_result(
                call,
                reason=f"Analysis failed due to unexpected error: {e}",
            )

    results = await asyncio.gather(*[safe_analyze(call) for call in batch.calls])

    return BatchQAAnalysisResponse(
        success=True,
        total=len(results),
        results=list(results),
    )
