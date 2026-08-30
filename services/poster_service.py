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
from poster.file_parser import UploadedContent, parse_uploaded_file
from poster.models.copywriting import Copywriting
from poster.models.generation import ImageGenerationResult
from poster.models.request import PosterGenerationRequest
from poster.models.response import PosterPayload
from runtime.models import ImageTaskState
from runtime.orchestrator import ImageGenerationOrchestrator


class PosterService:
    """海报接口适配层；流程控制由 ImageGenerationOrchestrator 统一负责。"""

    def __init__(self, values, orchestrator=None):
        self.values = values
        self.orchestrator = orchestrator or ImageGenerationOrchestrator(values)

    async def generate(self, request: PosterGenerationRequest):
        started_at = time.perf_counter()
        conversation_id = normalize_conversation_id(request.conversation_id)
        uploaded_content = self._parse_uploaded_content(request)
        history_messages, visual_history = await asyncio.gather(
            get_history_messages(conversation_id),
            get_recent_visual_history(conversation_id),
        )
        state = ImageTaskState(
            user_input=request.user_text,
            conversation_id=conversation_id,
            history_messages=history_messages,
            visual_history=visual_history,
            uploaded_content=uploaded_content,
            force_image=True,
            poster_context={
                "poster_type": request.poster_type,
                "campaign": request.campaign,
                "target_audience": request.target_audience,
            },
        )
        result = await asyncio.to_thread(self.orchestrator.run, state)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        response = self._build_response(result, conversation_id, latency_ms)
        await self._write_conversation(request, result, conversation_id)
        await asyncio.to_thread(self._write_log, request, result, conversation_id, latency_ms)
        return response

    @staticmethod
    def _parse_uploaded_content(request):
        payloads = list(
            zip(
                request.file_names,
                request.file_content_types,
                request.file_contents,
            )
        )
        if not payloads and request.file_content:
            payloads = [(request.file_name, request.file_content_type, request.file_content)]
        if not payloads:
            return None
        parsed_files = [parse_uploaded_file(name, content_type, content) for name, content_type, content in payloads]
        parsed_files = [item for item in parsed_files if item]
        if len(parsed_files) == 1:
            return parsed_files[0]
        return UploadedContent(
            filename="、".join(item.filename for item in parsed_files),
            content_type="multipart/mixed",
            extension=".multi",
            text="\n\n".join(item.text for item in parsed_files if item.text),
            data_urls=tuple(url for item in parsed_files for url in item.data_urls),
        )

    @staticmethod
    def _build_response(result, conversation_id, latency_ms):
        return PosterPayload(
            copywriting=Copywriting(
                headline=result.copywriting.headline,
                subheadline=result.copywriting.subheadline,
                cta=result.copywriting.cta,
            ),
            poster=ImageGenerationResult(image_url=result.image_url),
            metadata={
                "pipeline": result.route,
                "status": result.status,
                "knowledge_status": result.knowledge_status,
                "message": result.reply_text,
                "task_id": result.task_id,
                "generation_time": f"{latency_ms} ms",
            },
            conversation_id=conversation_id,
        )

    @staticmethod
    async def _write_conversation(request, result, conversation_id):
        user_text = request.user_text
        file_names = request.file_names or ([request.file_name] if request.file_name else [])
        if file_names:
            user_text = f"{user_text}\n附件：{'、'.join(file_names)}"
        if result.image_url:
            await append_visual_history(
                conversation_id,
                {
                    "image_url": result.image_url,
                    "user_input": request.user_text,
                    "assistant_text": result.reply_text,
                    "image_prompt": result.image_prompt,
                },
            )
        await append_messages(
            conversation_id,
            [
                {"role": "user", "content": user_text},
                {
                    "role": "assistant",
                    "content": result.reply_text,
                    "image_url": result.image_url,
                },
            ],
        )

    def _write_log(self, request, result, conversation_id, latency_ms):
        write_log(
            {
                "pipeline": result.route,
                "status": result.status,
                "task_id": result.task_id,
                "conversation_id": conversation_id,
                "user_input": request.user_text,
                "image_url": result.image_url,
                "image_prompt": result.image_prompt,
                "knowledge_status": result.knowledge_status,
                "actions": list(result.actions),
                "latency_ms": latency_ms,
                "text_model": self.values.get("QWEN_TEXT_MODEL") or DEFAULT_TEXT_MODEL,
                "vl_model": self.values.get("QWEN_VL_MODEL") or DEFAULT_VL_MODEL,
                "image_model": self.values.get("QWEN_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
            }
        )


async def generate_poster(request: PosterGenerationRequest):
    return await PosterService(read_config(".env")).generate(request)


async def generate_schema_driven_poster(
    user_text,
    conversation_id="default",
    file_name=None,
    file_content_type="",
    file_content=None,
    poster_type="商业海报",
    campaign="",
    target_audience="",
):
    request = PosterGenerationRequest(
        user_text=user_text,
        conversation_id=conversation_id,
        file_name=file_name,
        file_content_type=file_content_type,
        file_content=file_content,
        poster_type=poster_type,
        campaign=campaign,
        target_audience=target_audience,
    )
    return await generate_poster(request)
