"""
Input Pydantic models for the Synergex Med Call QA API.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class Department(str, Enum):
    SCHEDULING = "Scheduling"
    ONBOARDING = "Onboarding"
    HELPDESK = "Helpdesk"
    FOLLOW_UPS = "Follow-Ups"
    RECORDS = "Records"


class CallTranscript(BaseModel):
    """Inbound payload representing a single phone call transcript."""

    call_id: Annotated[str, Field(description="Unique identifier for the call")]
    agent_name: Annotated[str, Field(description="Name of the virtual assistant agent")]
    call_date: Annotated[
        str, Field(description="Date of the call in YYYY-MM-DD format", pattern=r"^\d{4}-\d{2}-\d{2}$")
    ]
    call_duration_seconds: Annotated[
        int, Field(ge=0, description="Duration of the call in seconds")
    ]
    department: Annotated[
        Department,
        Field(description="Department handling the call (Scheduling, Onboarding, Helpdesk, Follow-Ups, Records)"),
    ]
    transcript: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Multi-turn conversation between agent and caller. "
                "Each line is prefixed with 'Agent:' or 'Caller:'."
            ),
        ),
    ]

    @field_validator("transcript")
    @classmethod
    def transcript_not_whitespace_only(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("Transcript must contain actual content, not just whitespace")
        return v

    @field_validator("agent_name")
    @classmethod
    def agent_name_not_empty(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("agent_name must not be empty")
        return v.strip()


class BatchCallTranscripts(BaseModel):
    """Container for batch analysis requests."""

    calls: Annotated[
        list[CallTranscript],
        Field(min_length=1, max_length=50, description="List of call transcripts to analyze (1–50)"),
    ]
