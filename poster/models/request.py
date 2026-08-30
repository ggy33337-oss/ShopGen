# -*- coding: utf-8 -*-

from typing import Any

from pydantic import BaseModel, Field, field_validator

from poster.models.common import MAX_SHORT_TEXT_LENGTH, MAX_TEXT_LENGTH, normalize_text


class PosterGenerationRequest(BaseModel):
    user_text: str = Field(..., min_length=1, max_length=1000)
    conversation_id: str = Field(default="default", max_length=64)
    file_name: str | None = None
    file_content_type: str = ""
    file_content: bytes | None = None
    file_names: list[str] = Field(default_factory=list)
    file_content_types: list[str] = Field(default_factory=list)
    file_contents: list[bytes] = Field(default_factory=list)
    poster_type: str = Field(default="商业海报", max_length=MAX_SHORT_TEXT_LENGTH)
    campaign: str = Field(default="", max_length=MAX_TEXT_LENGTH)
    target_audience: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)

    @field_validator("user_text", "conversation_id", "file_name", "file_content_type", "poster_type", "campaign", "target_audience", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_text(value)
