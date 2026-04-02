"""
Abstract base class for LLM providers.
All providers must implement this interface, enabling seamless swapping
between Gemini, OpenAI, Anthropic, etc. without changing application logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.input_models import CallTranscript
    from app.models.output_models import QAAnalysisResult


class LLMProvider(ABC):
    """
    Abstract LLM provider interface.

    Implementations must handle:
    - Building the API request with the correct schema enforcement
    - Retry logic on transient failures
    - Logging of tokens, latency, etc.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of this provider (e.g., 'gemini', 'openai')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier being used (e.g., 'gemini-2.5-flash')."""
        ...

    @abstractmethod
    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        call_id: str,
    ) -> "QAAnalysisResult":
        """
        Call the LLM with the given prompts and return a parsed QAAnalysisResult.

        Args:
            system_prompt: The system-level instructions for the LLM.
            user_prompt: The user-turn content (transcript + call metadata).
            call_id: Used for logging/tracing.

        Returns:
            A fully-validated QAAnalysisResult instance.

        Raises:
            LLMProviderError: On non-retryable failures after all retries exhausted.
        """
        ...


class LLMProviderError(Exception):
    """Raised when the LLM provider fails after all retry attempts."""

    def __init__(self, provider: str, message: str, original_error: Exception | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.original_error = original_error
