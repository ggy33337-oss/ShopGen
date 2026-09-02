# -*- coding: utf-8 -*-

import asyncio

from fastapi import APIRouter, HTTPException

from api.dtos.conversation import (
    ConversationCreateRequest,
    ConversationDetail,
    ConversationSummary,
)
from infra.conversation_store import (
    create_conversation,
    delete_conversation,
    list_conversations,
    load_conversation,
)
from infra.redis_image_cache import RedisImageCache, extract_image_id


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
async def conversations():
    return await list_conversations()


@router.post("", response_model=ConversationDetail)
async def new_conversation(request: ConversationCreateRequest):
    conversation = await create_conversation(request.title)
    return build_conversation_detail(conversation)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def conversation_detail(conversation_id: str):
    conversation = await load_conversation(conversation_id)
    return build_conversation_detail(conversation)


@router.delete("/{conversation_id}")
async def remove_conversation(conversation_id: str):
    conversation = await load_conversation(conversation_id)
    deleted = await delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    image_ids = {
        extract_image_id(item.get("image_url"))
        for item in conversation.get("visual_history", [])
    }
    edit_session = conversation.get("visual_edit_session", {})
    image_ids.update(
        {
            extract_image_id(edit_session.get("root_image_url")),
            extract_image_id(edit_session.get("latest_result_url")),
        }
    )
    image_ids.discard("")
    if image_ids:
        try:
            await asyncio.to_thread(RedisImageCache().delete_images, image_ids)
        except RuntimeError:
            # 数据库删除已经成功；Redis 的 TTL 会兜底清理短暂不可达的缓存。
            pass
    return {"deleted": True, "conversation_id": conversation_id}


def build_conversation_detail(conversation):
    return ConversationDetail(
        conversation_id=conversation["conversation_id"],
        title=conversation.get("title", "新会话"),
        messages=conversation.get("messages", []),
    )
