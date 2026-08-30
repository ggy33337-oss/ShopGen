# -*- coding: utf-8 -*-

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.dtos.knowledge import (
    KnowledgeMatchDTO,
    KnowledgeSearchRequest,
    KnowledgeSourceDTO,
    KnowledgeUploadResponse,
)
from services.knowledge_service import KnowledgeService


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge(
    file: UploadFile = File(...),
    title: str = Form(default="", max_length=255),
    category: str = Form(default="", max_length=100),
):
    try:
        result = await asyncio.to_thread(
            KnowledgeService().upload,
            file.filename or "",
            file.content_type or "",
            await file.read(),
            title,
            category,
        )
        return KnowledgeUploadResponse(**result.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/search", response_model=list[KnowledgeMatchDTO])
async def search_knowledge(request: KnowledgeSearchRequest):
    try:
        return await asyncio.to_thread(
            KnowledgeService().search,
            request.query,
            request.limit,
            request.category,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/sources", response_model=list[KnowledgeSourceDTO])
async def knowledge_sources(limit: int = 100):
    try:
        return await asyncio.to_thread(KnowledgeService().list_sources, min(max(limit, 1), 500))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
