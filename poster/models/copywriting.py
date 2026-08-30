# -*- coding: utf-8 -*-

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from poster.models.common import MAX_SHORT_TEXT_LENGTH, normalize_text


class Copywriting(BaseModel):
    headline: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    subheadline: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    cta: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)

    @field_validator("headline", "subheadline", "cta", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_text(value)


class ImagePrompt(BaseModel):
    subject: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    visual_style: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    lighting: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    composition: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    quality: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)

    @field_validator("subject", "visual_style", "lighting", "composition", "quality", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_text(value)


class CopywritingResult(BaseModel):
    copywriting: Copywriting = Field(default_factory=Copywriting)
    image_prompt: ImagePrompt = Field(default_factory=ImagePrompt)

    model_config = ConfigDict(extra="ignore")
