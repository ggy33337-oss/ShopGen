from typing import Any

from pydantic import BaseModel, Field, field_validator

from poster.models.common import MAX_SHORT_TEXT_LENGTH, MAX_TEXT_LENGTH, normalize_text, normalize_text_list


class PosterSchema(BaseModel):
    product: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    poster_type: str = Field(default="商业海报", max_length=MAX_SHORT_TEXT_LENGTH)
    campaign: str = Field(default="", max_length=MAX_TEXT_LENGTH)
    style: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    layout: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    color_scheme: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    selling_points: list[str] = Field(default_factory=list)
    target_audience: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)

    @field_validator(
        "product",
        "poster_type",
        "campaign",
        "style",
        "layout",
        "color_scheme",
        "target_audience",
        mode="before",
    )
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return normalize_text(value)

    @field_validator("selling_points", mode="before")
    @classmethod
    def clean_points(cls, value: Any) -> list[str]:
        return normalize_text_list(value)
