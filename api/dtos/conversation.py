# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field, field_validator


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="新会话", max_length=40)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value):
        value = str(value or "新会话").strip()
        return value or "新会话"


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str = "新会话"
    updated_at: str = ""
    message_count: int = Field(default=0, ge=0)


class ConversationMessage(BaseModel):
    role: str
    content: str = ""
    image_url: str = ""

    @field_validator("role", "content", "image_url", mode="before")
    @classmethod
    def normalize_string(cls, value):
        if value is None:
            return ""
        return str(value).strip()


class ConversationDetail(BaseModel):
    conversation_id: str
    title: str = "新会话"
    messages: list[ConversationMessage] = Field(default_factory=list)
