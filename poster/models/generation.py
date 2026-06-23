from typing import Any

from pydantic import BaseModel, Field, field_validator

from poster.models.common import MAX_PROMPT_LENGTH, normalize_text


class FinalPromptResult(BaseModel):
    final_prompt: str = Field(default="", max_length=MAX_PROMPT_LENGTH)

    @field_validator("final_prompt", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_text(value)


class ImageGenerationResult(BaseModel):
    image_url: str = ""

    @field_validator("image_url", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_text(value)
