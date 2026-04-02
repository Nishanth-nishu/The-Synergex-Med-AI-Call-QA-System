"""
OpenAI provider for the Synergex Med QA system.

Demonstrates the clean provider swap: this implements the exact same LLMProvider
interface as GeminiProvider. Switch by setting LLM_PROVIDER=openai in your .env.

Uses the OpenAI beta parse API which natively understands Pydantic BaseModel —
schema is derived automatically and the response is parsed into a typed object.

Retry strategy: exponential backoff via tenacity (3 attempts, 2–16s wait).
"""

from __future__ import annotations

import asyncio
import os
import time
from functools import partial

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.models.output_models import QAAnalysisResult
from app.providers.base import LLMProvider, LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)

_RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError)


class OpenAIProvider(LLMProvider):
    """
    OpenAI gpt-4o provider using beta.chat.completions.parse for structured output.

    The parse() method accepts a Pydantic class directly as response_format,
    handles schema generation internally, and returns a Pydantic model instance —
    the same interface contract as GeminiProvider.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self._client = OpenAI(api_key=api_key)
        self._model_name = model or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        call_id: str,
    ) -> QAAnalysisResult:
        """Call OpenAI with structured output enforcement and retry logic."""
        start_time = time.monotonic()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(self._call_openai_with_retry, system_prompt, user_prompt, call_id),
            )
        except Exception as e:
            raise LLMProviderError(
                provider=self.provider_name,
                message=f"All retry attempts exhausted for call_id={call_id}: {e}",
                original_error=e,
            ) from e

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "openai_analysis_complete",
            call_id=call_id,
            provider=self.provider_name,
            model=self._model_name,
            latency_ms=elapsed_ms,
            overall_assessment=result.overall_assessment.value,
        )
        return result

    def _call_openai_with_retry(
        self, system_prompt: str, user_prompt: str, call_id: str
    ) -> QAAnalysisResult:
        return self._call_openai_inner(system_prompt, user_prompt, call_id)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    def _call_openai_inner(
        self, system_prompt: str, user_prompt: str, call_id: str
    ) -> QAAnalysisResult:
        completion = self._client.beta.chat.completions.parse(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=QAAnalysisResult,
            temperature=0.0,
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned a null parsed response — possible refusal")

        # Log token usage if available
        if completion.usage:
            logger.info(
                "openai_token_usage",
                call_id=call_id,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )

        return parsed
