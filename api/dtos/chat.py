# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field, field_validator


MAX_MESSAGE_LENGTH = 1000


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_id: str = Field(default="default", max_length=64)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("请输入有效内容")
        return value

    @field_validator("conversation_id")
    @classmethod
    def normalize_conversation_id(cls, value):
        value = str(value or "default").strip()
        if not value:
            return "default"
        return value


class ChatResponse(BaseModel):
    text: str = ""
    image_url: str = ""
    latency_ms: int = Field(default=0, ge=0)
    conversation_id: str = "default"
    pipeline: str = ""
    status: str = ""
    knowledge_status: str = "not_used"
    task_id: str = ""

    @field_validator(
        "text",
        "image_url",
        "conversation_id",
        "pipeline",
        "status",
        "knowledge_status",
        "task_id",
        mode="before",
    )
    @classmethod
    def normalize_string(cls, value):
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, value):
        if value == "":
            return value
        if value.startswith(("http://", "https://", "/api/images/")):
            return value
        return ""
