from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_REPLY_TEXT_LENGTH = 4000
MAX_IMAGE_PROMPT_LENGTH = 2000


class VisualModelResponse(BaseModel):
    reply_text: str = Field(default="", max_length=MAX_REPLY_TEXT_LENGTH)
    image_prompt: str = Field(default="", max_length=MAX_IMAGE_PROMPT_LENGTH)

    model_config = ConfigDict(extra="ignore")

    @field_validator("reply_text", "image_prompt", mode="before")
    @classmethod
    def normalize_string(cls, value: Any):
        if value is None:
            return ""
        return str(value).strip()
