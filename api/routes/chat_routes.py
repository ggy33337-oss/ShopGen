# -*- coding: utf-8 -*-

from fastapi import APIRouter

from api.dtos.chat import ChatRequest, ChatResponse
from services.chat_service import ask


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response, _model_name, latency_ms, conversation_id = await ask(
        request.message,
        request.conversation_id,
    )
    if isinstance(response, dict):
        text = response.get("text", "")
        image_url = extract_chat_image_url(response)
    else:
        text = response
        image_url = ""

    return ChatResponse(
        text=text,
        image_url=image_url,
        latency_ms=latency_ms,
        conversation_id=conversation_id,
        pipeline=response.get("pipeline", "") if isinstance(response, dict) else "",
        status=response.get("status", "") if isinstance(response, dict) else "",
        knowledge_status=(
            response.get("knowledge_status", "not_used")
            if isinstance(response, dict)
            else "not_used"
        ),
        task_id=response.get("task_id", "") if isinstance(response, dict) else "",
    )


def extract_chat_image_url(response):
    if not isinstance(response, dict):
        return ""

    direct_url = str(response.get("image_url") or "").strip()
    if direct_url:
        return direct_url
    choices = response.get("image", {}).get("output", {}).get("choices", [])
    for choice in choices:
        content = choice.get("message", {}).get("content", [])
        for item in content:
            image_url = item.get("image")
            if image_url:
                return image_url
    return ""
