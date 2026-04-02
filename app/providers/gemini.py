"""
Google Gemini provider for the Synergex Med QA system with Multi-Key Rotation
and Automatic Model Fallback.

This provider handles both:
1.  429 RESOURCE_EXHAUSTED: Rotates to the next API key.
2.  404 NOT_FOUND: Tries alternative model aliases (e.g. gemini-2.5-flash, gemini-flash).

Retry strategy:
- Per-key: exponential backoff via tenacity (2 attempts).
- Across keys: rotates through all provided keys on quota exhaustion.
"""

from __future__ import annotations

import asyncio
import os
import time
from functools import partial

from google import genai
from google.genai import types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.models.output_models import QAAnalysisResult
from app.providers.base import LLMProvider, LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Exceptions that warrant a retry (transient failures)
try:
    from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, ServiceUnavailable
    _RETRYABLE_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable, GoogleAPIError)
except ImportError:
    _RETRYABLE_EXCEPTIONS = (Exception,)  # type: ignore[assignment]


class GeminiProvider(LLMProvider):
    """
    Gemini provider with key rotation and model fallback logic.
    """

    # List of models to try in order if the primary fails with 404
    MODEL_FALLBACKS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash",
        "gemini-1.5-flash",
    ]

    def __init__(self, api_keys: list[str] | None = None, model: str | None = None):
        if not api_keys:
            keys_str = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
            if not keys_str:
                raise ValueError("Neither GEMINI_API_KEYS nor GEMINI_API_KEY is set")
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

        self._api_keys = api_keys
        self._current_key_index = 0
        
        # User specified model or the first fallback
        self._requested_model = model or os.environ.get("GEMINI_MODEL")
        self._current_model = self._requested_model or self.MODEL_FALLBACKS[0]
        
        self._setup_client()

    def _setup_client(self) -> None:
        """Initialize the Gemini client using the current key."""
        current_key = self._api_keys[self._current_key_index]
        self._client = genai.Client(api_key=current_key)
        logger.info(
            "gemini_client_setup",
            key_index=self._current_key_index,
            key_prefix=current_key[:6] + "...",
            model=self._current_model,
        )

    def _rotate_key(self) -> bool:
        """Rotate to the next API key."""
        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
        self._setup_client()
        return self._current_key_index != 0

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._current_model

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        call_id: str,
    ) -> QAAnalysisResult:
        """
        Call Gemini with rotation and model fallback.
        """
        start_time = time.monotonic()
        
        max_rotation_attempts = len(self._api_keys)
        last_error = None

        for attempt in range(max_rotation_attempts):
            try:
                result = await self._try_all_models(system_prompt, user_prompt, call_id)
                
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                logger.info(
                    "gemini_analysis_complete",
                    call_id=call_id,
                    key_index=self._current_key_index,
                    model=self._current_model,
                    latency_ms=elapsed_ms,
                    overall_assessment=result.overall_assessment.value,
                )
                return result

            except Exception as e:
                last_error = e
                # Check for 429 quota error
                is_quota_error = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                
                if is_quota_error and len(self._api_keys) > 1:
                    logger.warning(
                        "gemini_quota_exceeded_rotating_key",
                        call_id=call_id,
                        key_index=self._current_key_index,
                        error=str(e),
                    )
                    self._rotate_key()
                    continue
                else:
                    # Final failure after rotation or non-quota error
                    raise LLMProviderError(
                        provider=self.provider_name,
                        message=f"Gemini failed (key {self._current_key_index+1}/{len(self._api_keys)}): {e}",
                        original_error=e,
                    ) from e

        raise LLMProviderError(
            provider=self.provider_name,
            message=f"All {len(self._api_keys)} keys exhausted or failed.",
            original_error=last_error,
        )

    async def _try_all_models(self, system_prompt: str, user_prompt: str, call_id: str) -> QAAnalysisResult:
        """Try the current model, and fallback to others if 404 occurs."""
        models_to_try = [self._current_model]
        for m in self.MODEL_FALLBACKS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_404_error = None
        for model in models_to_try:
            self._current_model = model
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    partial(self._call_gemini_with_retry, system_prompt, user_prompt, call_id),
                )
            except Exception as e:
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    last_404_error = e
                    logger.warning("gemini_model_not_found_trying_next", model=model, call_id=call_id)
                    continue
                # If it's not a 404 (e.g. it's a 429), re-raise to the rotation level
                raise e
        
        if last_404_error:
            raise last_404_error
        raise Exception("No models worked")

    def _call_gemini_with_retry(
        self, system_prompt: str, user_prompt: str, call_id: str
    ) -> QAAnalysisResult:
        return self._call_gemini_inner(system_prompt, user_prompt, call_id)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        stop=stop_after_attempt(2),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    def _call_gemini_inner(
        self, system_prompt: str, user_prompt: str, call_id: str
    ) -> QAAnalysisResult:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=QAAnalysisResult,
            temperature=0.0,
        )

        response = self._client.models.generate_content(
            model=self._current_model,
            contents=user_prompt,
            config=config,
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            logger.info("gemini_token_usage", call_id=call_id, total_tokens=getattr(usage, "total_token_count", None))

        if hasattr(response, "parsed") and response.parsed is not None:
            return response.parsed if isinstance(response.parsed, QAAnalysisResult) else QAAnalysisResult.model_validate(response.parsed)

        import json
        return QAAnalysisResult.model_validate(json.loads(response.text))
