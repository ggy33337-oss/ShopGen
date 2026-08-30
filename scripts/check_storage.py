# -*- coding: utf-8 -*-

import asyncio
import json
import uuid

from sqlalchemy import text

from infra.conversation_store import (
    append_messages,
    append_visual_history,
    create_conversation,
    delete_conversation,
    load_conversation,
)
from infra.database import dispose_async_engine, get_async_engine
from infra.redis_image_cache import RedisImageCache


async def check_mysql():
    conversation_id = ""
    try:
        engine = await get_async_engine()
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT DATABASE() AS database_name, @@character_set_connection AS charset_name")
            )
            server = dict(result.mappings().one())

        conversation = await create_conversation("中文连接验证")
        conversation_id = conversation["conversation_id"]
        await append_messages(
            conversation_id,
            [
                {"role": "user", "content": "把之前的图片换成蓝色"},
                {"role": "assistant", "content": "已进入历史图片编辑链路。"},
            ],
        )
        await append_visual_history(
            conversation_id,
            {
                "turn_id": uuid.uuid4().hex,
                "image_url": "/api/images/" + uuid.uuid4().hex,
                "user_input": "把之前的图片换成蓝色",
                "assistant_text": "测试完成",
                "image_prompt": "保留构图，将主色改为蓝色",
            },
        )
        loaded = await load_conversation(conversation_id)
        async with engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT TABLE_NAME, TABLE_COLLATION FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN "
                    "('conversations', 'messages', 'visual_history') ORDER BY TABLE_NAME"
                )
            )
            collations = [dict(row) for row in result.mappings().all()]

        expected_messages = ["把之前的图片换成蓝色", "已进入历史图片编辑链路。"]
        actual_messages = [item["content"] for item in loaded["messages"]]
        if loaded["title"] != "中文连接验证" or actual_messages != expected_messages:
            raise RuntimeError("MySQL 中文内容往返校验失败。")
        if any(item["TABLE_COLLATION"] != "utf8mb4_unicode_ci" for item in collations):
            raise RuntimeError("MySQL 表排序规则校验失败。")
        return {
            "ok": True,
            "database": server["database_name"],
            "connection_charset": server["charset_name"],
            "title": loaded["title"],
            "messages": actual_messages,
            "visual_count": len(loaded["visual_history"]),
            "table_collations": collations,
        }
    finally:
        if conversation_id:
            await delete_conversation(conversation_id)
        await dispose_async_engine()


def check_redis():
    cache = RedisImageCache()
    stored = cache.put_image(b"storage-check", "image/png")
    try:
        loaded = cache.get_image(stored.image_id)
        if loaded is None or loaded.content != b"storage-check":
            raise RuntimeError("Redis 图片内容往返校验失败。")
        return {"ok": True, "image_url": stored.public_url}
    finally:
        cache.delete_image(stored.image_id)


async def main():
    result = {"mysql": await check_mysql()}
    try:
        result["redis"] = check_redis()
    except RuntimeError as exc:
        result["redis"] = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
