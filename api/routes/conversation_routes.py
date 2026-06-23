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


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def conversations():
    return list_conversations()


@router.post("", response_model=ConversationDetail)
def new_conversation(request: ConversationCreateRequest):
    conversation = create_conversation(request.title)
    return build_conversation_detail(conversation)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def conversation_detail(conversation_id: str):
    conversation = load_conversation(conversation_id)
    return build_conversation_detail(conversation)


@router.delete("/{conversation_id}")
def remove_conversation(conversation_id: str):
    deleted = delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": True, "conversation_id": conversation_id}


def build_conversation_detail(conversation):
    return ConversationDetail(
        conversation_id=conversation["conversation_id"],
        title=conversation.get("title", "新会话"),
        messages=conversation.get("messages", []),
    )
