# -*- coding: utf-8 -*-

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.dtos.poster import PosterResponseDTO
from poster.models.request import PosterGenerationRequest
from services.poster_service import generate_poster


router = APIRouter(prefix="/api/poster", tags=["poster"])


@router.post("/generate", response_model=PosterResponseDTO)
async def generate_poster_route(
    message: str = Form(..., min_length=1, max_length=1000),
    conversation_id: str = Form(default="default", max_length=64),
    poster_type: str = Form(default="商业海报", max_length=80),
    campaign: str = Form(default="", max_length=1000),
    target_audience: str = Form(default="", max_length=120),
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
):
    uploaded_files = list(files or [])
    if file and file.filename:
        uploaded_files.insert(0, file)
    file_payload = await read_upload_files(uploaded_files)
    request = PosterGenerationRequest(
        user_text=message,
        conversation_id=conversation_id,
        poster_type=poster_type,
        campaign=campaign,
        target_audience=target_audience,
        **file_payload,
    )

    try:
        return await generate_poster(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
