# -*- coding: utf-8 -*-

import asyncio
import time

from core.config import read_config
from infra.conversation_store import (
    append_messages,
    append_visual_history,
    get_history_messages,
    get_recent_visual_history,
    normalize_conversation_id,
)
from infra.logger import write_log
from llm.qwen_client import DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL, DEFAULT_VL_MODEL
from runtime.models import ImageTaskState
from runtime.orchestrator import ImageGenerationOrchestrator


async def ask(user_input, conversation_id=None):
    start_time = time.perf_counter()
    conversation_id = normalize_conversation_id(conversation_id)
    user_input = str(user_input or "").strip()
    if not user_input:
        return "请输入有效内容。", "", 0, conversation_id

    values = read_config(".env")
    result = None
    try:
        history_messages, visual_history = await asyncio.gather(
            get_history_messages(conversation_id),
            get_recent_visual_history(conversation_id),
        )
        state = ImageTaskState(
            user_input=user_input,
            conversation_id=conversation_id,
            history_messages=history_messages,
            visual_history=visual_history,
        )
        result = await asyncio.to_thread(ImageGenerationOrchestrator(values).run, state)
        response = build_chat_response(result)
    except Exception as exc:
        response = {"text": f"抱歉，发生了一个错误：{exc}", "status": "failed"}

    assistant_text = str(response.get("text") or "").strip()
    image_url = extract_image_url(response)
    assistant_message = {"role": "assistant", "content": assistant_text}
    if image_url:
        assistant_message["image_url"] = image_url
        await append_visual_history(
            conversation_id,
            {
                "image_url": image_url,
                "user_input": user_input,
                "assistant_text": assistant_text,
                "image_prompt": result.image_prompt if result else "",
            },
        )
    await append_messages(
        conversation_id,
        [{"role": "user", "content": user_input}, assistant_message],
    )

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    await asyncio.to_thread(
        write_log,
        {
            "pipeline": result.route if result else "failed",
            "status": result.status if result else "failed",
            "task_id": result.task_id if result else "",
            "conversation_id": conversation_id,
            "user_input": user_input,
            "response_text": assistant_text,
            "image_url": image_url,
            "image_prompt": result.image_prompt if result else "",
            "knowledge_status": result.knowledge_status if result else "not_used",
            "actions": list(result.actions) if result else [],
            "latency_ms": latency_ms,
            "text_model": values.get("QWEN_TEXT_MODEL") or DEFAULT_TEXT_MODEL,
            "vl_model": values.get("QWEN_VL_MODEL") or DEFAULT_VL_MODEL,
            "image_model": values.get("QWEN_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
        },
    )
    model_name = values.get("QWEN_TEXT_MODEL") or DEFAULT_TEXT_MODEL
    return response, model_name, latency_ms, conversation_id


def build_chat_response(result):
    response = {
        "text": result.reply_text,
        "pipeline": result.route,
        "status": result.status,
        "knowledge_status": result.knowledge_status,
        "task_id": result.task_id,
    }
    if result.image_url:
        response["image_url"] = result.image_url
        response["image"] = {
            "output": {
                "choices": [
                    {"message": {"content": [{"image": result.image_url}]}}
                ]
            }
        }
    return response


def extract_assistant_text(response):
    if isinstance(response, dict):
        return str(response.get("text") or "").strip()
    return str(response or "").strip()


def extract_image_url(response):
    if not isinstance(response, dict):
        return ""
    direct_url = str(response.get("image_url") or "").strip()
    if direct_url:
        return direct_url
    choices = response.get("image", {}).get("output", {}).get("choices", [])
    for choice in choices:
        for item in choice.get("message", {}).get("content", []):
            image_url = item.get("image") if isinstance(item, dict) else ""
            if image_url:
                return str(image_url).strip()
    return ""
