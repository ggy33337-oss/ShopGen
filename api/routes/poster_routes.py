# -*- coding: utf-8 -*-

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.dtos.poster import PosterTaskCreatedDTO, PosterTaskStatusDTO
from core.config import read_config
from infra.logger import generation_log_context, write_generation_event
from poster.models.request import PosterGenerationRequest
from services.poster_service import DEFAULT_POSTER_TOTAL_TIMEOUT_SECONDS
from services.poster_tasks import get_poster_task_manager


router = APIRouter(prefix="/api/poster", tags=["poster"])


@router.post("/generate", response_model=PosterTaskCreatedDTO, status_code=202)
async def generate_poster_route(
    message: str = Form(..., min_length=1, max_length=1000),
    conversation_id: str = Form(default="default", max_length=64),
    poster_type: str = Form(default="商业海报", max_length=80),
    campaign: str = Form(default="", max_length=1000),
    target_audience: str = Form(default="", max_length=120),
    image_model: str = Form(default="", max_length=80),
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
):
    task_id = uuid.uuid4().hex
    uploaded_files = list(files or [])
    if file and file.filename:
        uploaded_files.insert(0, file)
    values = read_config(".env")
    log_path = values.get("GENERATION_LOG_PATH") or "logs/generation.jsonl"
    deadline_seconds = int(
        values.get("POSTER_TOTAL_TIMEOUT_SECONDS")
        or DEFAULT_POSTER_TOTAL_TIMEOUT_SECONDS
    )
    request_deadline = time.perf_counter() + deadline_seconds
    deadline_at = (
        datetime.now(timezone.utc) + timedelta(seconds=deadline_seconds)
    ).isoformat()
    receive_started_at = time.perf_counter()
    with generation_log_context(
        log_path=log_path,
        task_id=task_id,
        conversation_id=conversation_id,
        request_kind="poster",
        deadline_seconds=deadline_seconds,
        deadline_monotonic=request_deadline,
        deadline_at=deadline_at,
    ):
        write_generation_event(
            "upload.receive",
            "started",
            file_count=len(uploaded_files),
            file_names=[item.filename for item in uploaded_files if item and item.filename],
        )
        try:
            file_payload = await read_upload_files(uploaded_files)
        except Exception as exc:
            write_generation_event(
                "upload.receive",
                "failed",
                latency_ms=int((time.perf_counter() - receive_started_at) * 1000),
                error_type=type(exc).__name__,
                error_message=str(exc)[:2000],
            )
            raise
        total_bytes = sum(len(content or b"") for content in file_payload["file_contents"])
        write_generation_event(
            "upload.receive",
            "completed",
            latency_ms=int((time.perf_counter() - receive_started_at) * 1000),
            file_count=len(file_payload["file_contents"]),
            received_bytes=total_bytes,
        )
    request = PosterGenerationRequest(
        user_text=message,
        conversation_id=conversation_id,
        poster_type=poster_type,
        campaign=campaign,
        target_audience=target_audience,
        image_model=image_model,
        **file_payload,
    )
    task = await get_poster_task_manager().submit(
        request,
        values,
        task_id,
        deadline_monotonic=request_deadline,
        deadline_at=deadline_at,
    )
    return PosterTaskCreatedDTO(
        task_id=task.task_id,
        status=task.status,
        conversation_id=request.conversation_id,
    )


@router.get("/tasks/{task_id}", response_model=PosterTaskStatusDTO)
async def get_poster_task(task_id: str):
    task = await get_poster_task_manager().get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期。")
    result = task.result.model_dump() if task.result is not None else None
    return PosterTaskStatusDTO(
        task_id=task.task_id,
        status=task.status,
        conversation_id=task.conversation_id,
        result=result,
        error=task.error,
        created_at=str(task.created_at),
        updated_at=str(task.updated_at),
        metadata={"polling": True},
    )


async def read_upload_files(files: list[UploadFile]):
    valid_files = [item for item in files if item and item.filename]
    if not valid_files:
        return {
            "file_name": None,
            "file_content_type": "",
            "file_content": None,
            "file_names": [],
            "file_content_types": [],
            "file_contents": [],
        }

    contents = await asyncio.gather(*(item.read() for item in valid_files))
    return {
        "file_name": valid_files[0].filename,
        "file_content_type": valid_files[0].content_type or "",
        "file_content": contents[0],
        "file_names": [item.filename for item in valid_files],
        "file_content_types": [item.content_type or "" for item in valid_files],
        "file_contents": contents,
    }
