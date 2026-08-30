# -*- coding: utf-8 -*-

from pydantic import BaseModel


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
