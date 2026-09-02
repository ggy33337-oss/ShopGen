# -*- coding: utf-8 -*-

from typing import Any

from pydantic import BaseModel, Field


class PosterCopywritingDTO(BaseModel):
    headline: str = ""
    subheadline: str = ""
    cta: str = ""


class PosterImageDTO(BaseModel):
    image_url: str = ""


class PosterResponseDTO(BaseModel):
    copywriting: PosterCopywritingDTO
    poster: PosterImageDTO
    metadata: dict[str, str]
    conversation_id: str = "default"


class PosterTaskCreatedDTO(BaseModel):
    task_id: str
    status: str = "queued"
    conversation_id: str = "default"


class PosterTaskStatusDTO(BaseModel):
    task_id: str
    status: str
    conversation_id: str = "default"
    result: PosterResponseDTO | None = None
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
