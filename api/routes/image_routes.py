# -*- coding: utf-8 -*-

import asyncio

from fastapi import APIRouter, HTTPException, Response

from infra.redis_image_cache import RedisImageCache


router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("/{image_id}")
async def get_cached_image(image_id: str):
    cached = await asyncio.to_thread(RedisImageCache().get_image, image_id)
    if not cached:
        raise HTTPException(status_code=404, detail="图片不存在或缓存已过期")
    return Response(
        content=cached.content,
        media_type=cached.content_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )
