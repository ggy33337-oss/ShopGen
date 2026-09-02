# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field, field_validator


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    category: str = Field(default="", max_length=100)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query", "category")
    @classmethod
    def normalize_text(cls, value):
        return " ".join(str(value or "").strip().split())


class KnowledgeMatchDTO(BaseModel):
    source_id: str
    chunk_id: str
    title: str
    content: str
    chunk_type: str
    category: str = ""
    source_filename: str = ""
    source_title: str = ""
    page_number: int = 0
    section: str = ""
    score: float
    vector_score: float = 0.0
    rerank_score: float = 0.0


class KnowledgeUploadResponse(BaseModel):
    source_id: str
    title: str
    category: str = ""
    text_vector_count: int
    image_vector_count: int


class KnowledgeSourceDTO(BaseModel):
    source_id: str
    title: str
    category: str = ""
    source_filename: str = ""
    vector_count: int
