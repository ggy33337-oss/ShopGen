# -*- coding: utf-8 -*-

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import redis

from core.config import read_config


VALID_IMAGE_ID = re.compile(r"^[a-f0-9]{32}$")
PUBLIC_IMAGE_PREFIX = "/api/images/"


@dataclass(frozen=True)
class CachedImage:
    image_id: str
    content: bytes
    content_type: str

    @property
    def public_url(self):
        return f"{PUBLIC_IMAGE_PREFIX}{self.image_id}"


class RedisImageCache:
    def __init__(self, values=None, client=None):
        self.values = values or read_config(".env")
        self.ttl_seconds = int(self.values.get("REDIS_IMAGE_TTL_SECONDS") or 604800)
        self.max_image_bytes = int(self.values.get("REDIS_MAX_IMAGE_BYTES") or 20 * 1024 * 1024)
        self.key_prefix = str(self.values.get("REDIS_IMAGE_KEY_PREFIX") or "ecommerce:image").strip()
        self.client = client or self._create_client()

    def _create_client(self):
        redis_url = str(self.values.get("REDIS_URL") or "").strip()
        common = {
            "decode_responses": False,
            "socket_connect_timeout": int(self.values.get("REDIS_CONNECT_TIMEOUT") or 5),
            "socket_timeout": int(self.values.get("REDIS_SOCKET_TIMEOUT") or 5),
            "health_check_interval": 30,
        }
        if redis_url:
            return redis.Redis.from_url(redis_url, **common)
        return redis.Redis(
            host=str(self.values.get("REDIS_HOST") or "127.0.0.1").strip(),
            port=int(self.values.get("REDIS_PORT") or 6379),
            db=int(self.values.get("REDIS_DB") or 0),
            password=str(self.values.get("REDIS_PASSWORD") or "") or None,
            **common,
        )

    def put_image(self, content, content_type):
        content = bytes(content or b"")
        if not content:
            raise RuntimeError("不能向 Redis 缓存空图片。")
        if len(content) > self.max_image_bytes:
            raise RuntimeError(
                f"图片大小超过 Redis 缓存上限：{len(content)} > {self.max_image_bytes} 字节。"
            )
        image_id = uuid.uuid4().hex
        key = self._key(image_id)
        try:
            with self.client.pipeline(transaction=True) as pipeline:
                pipeline.hset(
                    key,
                    mapping={
                        "data": content,
                        "content_type": str(content_type or "image/png"),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if self.ttl_seconds > 0:
                    pipeline.expire(key, self.ttl_seconds)
                pipeline.execute()
        except redis.RedisError as exc:
            raise RuntimeError(f"Redis 图片缓存写入失败：{exc}") from exc
        return CachedImage(image_id=image_id, content=content, content_type=content_type or "image/png")

    def get_image(self, image_id):
        image_id = normalize_image_id(image_id)
        if not image_id:
            return None
        try:
            data, content_type = self.client.hmget(self._key(image_id), "data", "content_type")
        except redis.RedisError as exc:
            raise RuntimeError(f"Redis 图片缓存读取失败：{exc}") from exc
        if not data:
            return None
        return CachedImage(
            image_id=image_id,
            content=bytes(data),
            content_type=(content_type or b"image/png").decode("utf-8"),
        )

    def delete_image(self, image_id):
        image_id = normalize_image_id(image_id)
        if not image_id:
            return False
        try:
            return bool(self.client.delete(self._key(image_id)))
        except redis.RedisError as exc:
            raise RuntimeError(f"Redis 图片缓存删除失败：{exc}") from exc

    def delete_images(self, image_ids):
        keys = [self._key(image_id) for image_id in map(normalize_image_id, image_ids) if image_id]
        if not keys:
            return 0
        try:
            return int(self.client.delete(*keys))
        except redis.RedisError as exc:
            raise RuntimeError(f"Redis 图片缓存批量删除失败：{exc}") from exc

    def ping(self):
        try:
            return bool(self.client.ping())
        except redis.RedisError as exc:
            raise RuntimeError(f"Redis 连接失败：{exc}") from exc

    def _key(self, image_id):
        return f"{self.key_prefix}:{image_id}"


def normalize_image_id(image_id):
    value = str(image_id or "").strip().lower()
    return value if VALID_IMAGE_ID.fullmatch(value) else ""


def extract_image_id(image_url):
    value = str(image_url or "").strip()
    if not value.startswith(PUBLIC_IMAGE_PREFIX):
        return ""
    return normalize_image_id(value[len(PUBLIC_IMAGE_PREFIX):].split("?", 1)[0])
