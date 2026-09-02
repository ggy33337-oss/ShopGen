# -*- coding: utf-8 -*-

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone

from core.config import read_config
from infra.conversation_store import (
    append_messages,
    append_visual_history,
    get_history_messages,
    get_recent_visual_history,
    normalize_conversation_id,
    save_visual_edit_session,
)
from infra.logger import generation_log_context, generation_stage, write_generation_event, write_log
from llm.qwen_client import DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL, DEFAULT_VL_MODEL
from llm.qwen_reranker import DEFAULT_RERANK_MODEL
from poster.file_parser import UploadedContent, parse_uploaded_file
from poster.models.copywriting import Copywriting
from poster.models.generation import ImageGenerationResult
from poster.models.request import PosterGenerationRequest
from poster.models.response import PosterPayload
from runtime.models import ImageTaskState
from runtime.orchestrator import ImageGenerationOrchestrator
from services.visual_reference_loader import load_reference_image


DEFAULT_POSTER_TOTAL_TIMEOUT_SECONDS = 480


class PosterService:
    """海报接口适配层；流程控制由 ImageGenerationOrchestrator 统一负责。"""

    def __init__(self, values, orchestrator=None):
        self.values = values
        self.orchestrator = orchestrator or ImageGenerationOrchestrator(values)

    async def generate(self, request: PosterGenerationRequest, task_id=None, deadline_monotonic=None, deadline_at=None):
        started_at = time.perf_counter()
        conversation_id = normalize_conversation_id(request.conversation_id)
        task_id = str(task_id or uuid.uuid4().hex)
        state = None
        file_names = request.file_names or ([request.file_name] if request.file_name else [])
        log_path = self.values.get("GENERATION_LOG_PATH") or "logs/generation.jsonl"
        deadline_seconds = int(
            self.values.get("POSTER_TOTAL_TIMEOUT_SECONDS")
            or DEFAULT_POSTER_TOTAL_TIMEOUT_SECONDS
        )
        deadline_at = deadline_at or (
            datetime.now(timezone.utc) + timedelta(seconds=deadline_seconds)
        ).isoformat()
        with generation_log_context(
            log_path=log_path,
            task_id=task_id,
            conversation_id=conversation_id,
            request_kind="poster",
            deadline_seconds=deadline_seconds,
            deadline_at=deadline_at,
            deadline_monotonic=deadline_monotonic,
        ):
            write_generation_event(
                "request",
                "started",
                user_input=request.user_text,
                file_count=len(file_names),
                file_names=file_names,
                file_content_types=request.file_content_types,
            )
            try:
                with generation_stage("upload.parse", file_count=len(file_names)):
                    uploaded_content = self._parse_uploaded_content(request)
                with generation_stage("root_reference.cache"):
                    root_reference_url = await asyncio.to_thread(
                        self._cache_root_reference,
                        uploaded_content,
                    )
                with generation_stage("history.load"):
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
                    image_model=request.image_model,
                    task_id=task_id,
                )
                result = await asyncio.to_thread(self.orchestrator.run, state)
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                response = self._build_response(result, conversation_id, latency_ms)
                with generation_stage("conversation.save"):
                    await self._write_conversation(request, result, conversation_id)
                    if result.image_url and root_reference_url:
                        await save_visual_edit_session(
                            conversation_id,
                            root_image_url=root_reference_url,
                            latest_result_url=result.image_url,
                            revision=1,
                        )
                await asyncio.to_thread(self._write_log, request, result, conversation_id, latency_ms)
                write_generation_event(
                    "request",
                    "completed",
                    route=result.route,
                    latency_ms=latency_ms,
                    image_model=request.image_model or self.values.get("QWEN_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
                )
                return response
            except asyncio.CancelledError:
                write_generation_event(
                    "request",
                    "cancelled",
                    route=state.route if state else "unresolved",
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    error_category="client_disconnect_or_server_shutdown",
                    image_model=request.image_model or self.values.get("QWEN_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
                )
                raise
            except Exception as exc:
                write_generation_event(
                    "request",
                    "failed",
                    route=state.route if state else "unresolved",
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                    image_model=request.image_model or self.values.get("QWEN_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
                )
                raise

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

    def _cache_root_reference(self, uploaded_content):
        data_urls = tuple(getattr(uploaded_content, "data_urls", ()) or ())
        if not data_urls:
            return ""
        reference = load_reference_image(self.values, data_urls[0])
        if not reference:
            raise RuntimeError("无法读取最初上传的参考图片。")
        cached = self.orchestrator.image_cache.put_image(
            reference.content,
            reference.content_type,
        )
        write_generation_event(
            "root_reference.cached",
            "completed",
            root_image_url=cached.public_url,
            image_bytes=len(reference.content),
            content_type=reference.content_type,
        )
        return cached.public_url

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
        assistant_text = PosterService._format_copywriting(result)
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
                    "content": assistant_text,
                    "image_url": result.image_url,
                },
            ],
        )

    @staticmethod
    def _format_copywriting(result):
        copywriting = getattr(result, "copywriting", None)
        lines = [
            f"主标题：{str(getattr(copywriting, 'headline', '') or '').strip()}",
            f"副标题：{str(getattr(copywriting, 'subheadline', '') or '').strip()}",
            f"行动语：{str(getattr(copywriting, 'cta', '') or '').strip()}",
        ]
        lines = [line for line in lines if line.split("：", 1)[1].strip()]
        return "\n".join(lines) or str(result.reply_text or "").strip()

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
                "image_model": request.image_model or self.values.get("QWEN_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
                "rerank_model": self.values.get("QWEN_RERANK_MODEL") or DEFAULT_RERANK_MODEL,
                "rerank_min_score": float(
                    self.values.get("KNOWLEDGE_RERANK_MIN_SCORE") or 0.55
                ),
            }
        )


async def generate_poster(
    request: PosterGenerationRequest,
    task_id=None,
    deadline_monotonic=None,
    deadline_at=None,
):
    return await PosterService(read_config(".env")).generate(
        request,
        task_id=task_id,
        deadline_monotonic=deadline_monotonic,
        deadline_at=deadline_at,
    )


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
