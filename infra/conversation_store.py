# -*- coding: utf-8 -*-

import asyncio
import re
import uuid
from datetime import datetime

from sqlalchemy import text

from infra.database import get_async_engine


DEFAULT_CONVERSATION_ID = "default"
VALID_CONVERSATION_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_VISUAL_HISTORY_ITEMS = 20
_SCHEMA_LOCK = asyncio.Lock()
_SCHEMA_ENGINE_ID = None


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id VARCHAR(64) NOT NULL,
        title VARCHAR(255) NOT NULL DEFAULT '新会话',
        created_at DATETIME(6) NOT NULL,
        updated_at DATETIME(6) NOT NULL,
        PRIMARY KEY (conversation_id),
        KEY idx_conversations_updated_at (updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        conversation_id VARCHAR(64) NOT NULL,
        role VARCHAR(16) NOT NULL,
        content LONGTEXT NOT NULL,
        image_url TEXT NOT NULL,
        created_at DATETIME(6) NOT NULL,
        PRIMARY KEY (id),
        KEY idx_messages_conversation_id_id (conversation_id, id),
        CONSTRAINT fk_messages_conversation
            FOREIGN KEY (conversation_id)
            REFERENCES conversations(conversation_id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS visual_history (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        conversation_id VARCHAR(64) NOT NULL,
        turn_id VARCHAR(64) NOT NULL,
        image_url TEXT NOT NULL,
        user_input TEXT NOT NULL,
        assistant_text TEXT NOT NULL,
        image_prompt MEDIUMTEXT NOT NULL,
        created_at DATETIME(6) NOT NULL,
        PRIMARY KEY (id),
        KEY idx_visual_history_conversation_id_id (conversation_id, id),
        CONSTRAINT fk_visual_history_conversation
            FOREIGN KEY (conversation_id)
            REFERENCES conversations(conversation_id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
)


def normalize_conversation_id(conversation_id):
    if not conversation_id:
        return DEFAULT_CONVERSATION_ID
    conversation_id = str(conversation_id).strip()
    if VALID_CONVERSATION_ID.fullmatch(conversation_id):
        return conversation_id
    return uuid.uuid4().hex


async def get_ready_engine():
    engine = await get_async_engine()
    await initialize_database(engine)
    return engine


async def initialize_database(engine):
    global _SCHEMA_ENGINE_ID

    engine_id = id(engine)
    if _SCHEMA_ENGINE_ID == engine_id:
        return
    async with _SCHEMA_LOCK:
        if _SCHEMA_ENGINE_ID == engine_id:
            return
        try:
            async with engine.begin() as connection:
                for statement in SCHEMA_STATEMENTS:
                    await connection.execute(text(statement))
        except Exception as exc:
            raise RuntimeError(f"MySQL 表结构初始化失败：{exc}") from exc
        _SCHEMA_ENGINE_ID = engine_id


def create_empty_conversation(conversation_id):
    now = datetime.now().isoformat()
    return {
        "conversation_id": conversation_id,
        "title": "新会话",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "visual_history": [],
    }


async def ensure_conversation(connection, conversation_id, title="新会话"):
    safe_id = normalize_conversation_id(conversation_id)
    now = datetime.now()
    await connection.execute(
        text(
            """
            INSERT IGNORE INTO conversations (
                conversation_id, title, created_at, updated_at
            ) VALUES (
                :conversation_id, :title, :created_at, :updated_at
            )
            """
        ),
        {
            "conversation_id": safe_id,
            "title": title or "新会话",
            "created_at": now,
            "updated_at": now,
        },
    )
    return safe_id


async def create_conversation(title="新会话"):
    conversation_id = uuid.uuid4().hex
    now = datetime.now()
    engine = await get_ready_engine()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                INSERT INTO conversations (
                    conversation_id, title, created_at, updated_at
                ) VALUES (
                    :conversation_id, :title, :created_at, :updated_at
                )
                """
            ),
            {
                "conversation_id": conversation_id,
                "title": title or "新会话",
                "created_at": now,
                "updated_at": now,
            },
        )
    return await load_conversation(conversation_id)


async def load_conversation(conversation_id):
    safe_id = normalize_conversation_id(conversation_id)
    engine = await get_ready_engine()
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                WHERE conversation_id = :conversation_id
                """
            ),
            {"conversation_id": safe_id},
        )
        row = result.mappings().first()
        if row is None:
            return create_empty_conversation(safe_id)
        return {
            "conversation_id": row["conversation_id"],
            "title": row["title"] or "新会话",
            "created_at": serialize_datetime(row["created_at"]),
            "updated_at": serialize_datetime(row["updated_at"]),
            "messages": await load_messages(connection, safe_id),
            "visual_history": await load_visual_history(connection, safe_id),
        }


async def load_messages(connection, conversation_id):
    result = await connection.execute(
        text(
            """
            SELECT role, content, image_url
            FROM messages
            WHERE conversation_id = :conversation_id
            ORDER BY id ASC
            """
        ),
        {"conversation_id": conversation_id},
    )
    messages = []
    for row in result.mappings().all():
        message = {"role": row["role"], "content": row["content"] or ""}
        if row["image_url"]:
            message["image_url"] = row["image_url"]
        messages.append(message)
    return messages


async def load_visual_history(connection, conversation_id):
    result = await connection.execute(
        text(
            """
            SELECT turn_id, image_url, user_input, assistant_text, image_prompt, created_at
            FROM visual_history
            WHERE conversation_id = :conversation_id
            ORDER BY id ASC
            """
        ),
        {"conversation_id": conversation_id},
    )
    return [normalize_visual_record(dict(row)) for row in result.mappings().all()]


async def save_conversation(conversation):
    conversation_id = normalize_conversation_id(conversation.get("conversation_id"))
    title = conversation.get("title") or build_conversation_title(conversation)
    engine = await get_ready_engine()
    async with engine.begin() as connection:
        await ensure_conversation(connection, conversation_id, title)
        await connection.execute(
            text(
                """
                UPDATE conversations
                SET title = :title, updated_at = :updated_at
                WHERE conversation_id = :conversation_id
                """
            ),
            {
                "title": title or "新会话",
                "updated_at": datetime.now(),
                "conversation_id": conversation_id,
            },
        )
        await connection.execute(
            text("DELETE FROM messages WHERE conversation_id = :conversation_id"),
            {"conversation_id": conversation_id},
        )
        await connection.execute(
            text("DELETE FROM visual_history WHERE conversation_id = :conversation_id"),
            {"conversation_id": conversation_id},
        )
        await insert_messages(connection, conversation_id, conversation.get("messages", []))
        await insert_visual_history(
            connection,
            conversation_id,
            conversation.get("visual_history", []),
        )
    return await load_conversation(conversation_id)


async def append_messages(conversation_id, messages):
    safe_id = normalize_conversation_id(conversation_id)
    engine = await get_ready_engine()
    async with engine.begin() as connection:
        await ensure_conversation(connection, safe_id)
        await insert_messages(connection, safe_id, messages)
        await maybe_update_title(connection, safe_id)
        await connection.execute(
            text(
                """
                UPDATE conversations
                SET updated_at = :updated_at
                WHERE conversation_id = :conversation_id
                """
            ),
            {"updated_at": datetime.now(), "conversation_id": safe_id},
        )
    return await load_conversation(safe_id)


async def insert_messages(connection, conversation_id, messages):
    now = datetime.now()
    rows = []
    for item in messages or []:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        image_url = str(item.get("image_url") or "").strip()
        if role in {"user", "assistant"} and content:
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "role": role,
                    "content": content,
                    "image_url": image_url,
                    "created_at": now,
                }
            )
    if rows:
        await connection.execute(
            text(
                """
                INSERT INTO messages (
                    conversation_id, role, content, image_url, created_at
                ) VALUES (
                    :conversation_id, :role, :content, :image_url, :created_at
                )
                """
            ),
            rows,
        )


async def append_visual_history(conversation_id, visual_record):
    safe_id = normalize_conversation_id(conversation_id)
    record = normalize_visual_record(visual_record)
    if not record:
        return await load_conversation(safe_id)
    engine = await get_ready_engine()
    async with engine.begin() as connection:
        await ensure_conversation(connection, safe_id)
        await insert_visual_history(connection, safe_id, [record])
        await trim_visual_history(connection, safe_id)
        await connection.execute(
            text(
                """
                UPDATE conversations
                SET updated_at = :updated_at
                WHERE conversation_id = :conversation_id
                """
            ),
            {"updated_at": datetime.now(), "conversation_id": safe_id},
        )
    return await load_conversation(safe_id)


async def insert_visual_history(connection, conversation_id, visual_history):
    rows = []
    for item in visual_history or []:
        record = normalize_visual_record(item)
        if record:
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "turn_id": record["turn_id"],
                    "image_url": record["image_url"],
                    "user_input": record["user_input"],
                    "assistant_text": record["assistant_text"],
                    "image_prompt": record["image_prompt"],
                    "created_at": parse_datetime(record["created_at"]),
                }
            )
    if rows:
        await connection.execute(
            text(
                """
                INSERT INTO visual_history (
                    conversation_id, turn_id, image_url, user_input,
                    assistant_text, image_prompt, created_at
                ) VALUES (
                    :conversation_id, :turn_id, :image_url, :user_input,
                    :assistant_text, :image_prompt, :created_at
                )
                """
            ),
            rows,
        )


async def trim_visual_history(connection, conversation_id):
    await connection.execute(
        text(
            """
            DELETE FROM visual_history
            WHERE conversation_id = :conversation_id
              AND id NOT IN (
                  SELECT id FROM (
                      SELECT id
                      FROM visual_history
                      WHERE conversation_id = :retained_conversation_id
                      ORDER BY id DESC
                      LIMIT :history_limit
                  ) AS retained_visual_history
              )
            """
        ),
        {
            "conversation_id": conversation_id,
            "retained_conversation_id": conversation_id,
            "history_limit": MAX_VISUAL_HISTORY_ITEMS,
        },
    )


async def maybe_update_title(connection, conversation_id):
    result = await connection.execute(
        text("SELECT title FROM conversations WHERE conversation_id = :conversation_id"),
        {"conversation_id": conversation_id},
    )
    row = result.mappings().first()
    if row and row["title"] != "新会话":
        return
    result = await connection.execute(
        text(
            """
            SELECT content
            FROM messages
            WHERE conversation_id = :conversation_id AND role = 'user'
            ORDER BY id ASC
            LIMIT 1
            """
        ),
        {"conversation_id": conversation_id},
    )
    first_user_message = result.mappings().first()
    if not first_user_message:
        return
    title = str(first_user_message["content"] or "").strip()[:24] or "新会话"
    await connection.execute(
        text(
            """
            UPDATE conversations
            SET title = :title
            WHERE conversation_id = :conversation_id
            """
        ),
        {"title": title, "conversation_id": conversation_id},
    )


async def delete_conversation(conversation_id):
    safe_id = normalize_conversation_id(conversation_id)
    engine = await get_ready_engine()
    async with engine.begin() as connection:
        result = await connection.execute(
            text("DELETE FROM conversations WHERE conversation_id = :conversation_id"),
            {"conversation_id": safe_id},
        )
        return result.rowcount > 0


async def get_history_messages(conversation_id):
    conversation = await load_conversation(conversation_id)
    history = []
    for item in conversation.get("messages", []):
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content.strip()})
    return history


async def get_recent_visual_history(conversation_id, limit=3):
    safe_id = normalize_conversation_id(conversation_id)
    engine = await get_ready_engine()
    async with engine.begin() as connection:
        await ensure_conversation(connection, safe_id)
        result = await connection.execute(
            text(
                """
                SELECT turn_id, image_url, user_input, assistant_text, image_prompt, created_at
                FROM visual_history
                WHERE conversation_id = :conversation_id
                ORDER BY id DESC
                LIMIT :history_limit
                """
            ),
            {"conversation_id": safe_id, "history_limit": int(limit)},
        )
        rows = result.mappings().all()
    return [normalize_visual_record(dict(row)) for row in reversed(rows)]


def normalize_visual_record(record):
    if not isinstance(record, dict):
        return {}
    image_url = str(record.get("image_url") or "").strip()
    image_prompt = str(record.get("image_prompt") or "").strip()
    user_input = str(record.get("user_input") or "").strip()
    assistant_text = str(record.get("assistant_text") or "").strip()
    if not image_url and not image_prompt:
        return {}
    return {
        "turn_id": str(record.get("turn_id") or uuid.uuid4().hex).strip(),
        "image_url": image_url,
        "user_input": user_input[:1000],
        "assistant_text": assistant_text[:1000],
        "image_prompt": image_prompt[:3000],
        "created_at": serialize_datetime(record.get("created_at") or datetime.now()),
    }


def build_conversation_title(conversation):
    for item in conversation.get("messages", []):
        if item.get("role") == "user":
            content = str(item.get("content") or "").strip()
            if content:
                return content[:24]
    return "新会话"


async def list_conversations():
    engine = await get_ready_engine()
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.updated_at,
                    COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                GROUP BY c.conversation_id, c.title, c.updated_at
                ORDER BY c.updated_at DESC
                """
            )
        )
        rows = result.mappings().all()
    return [
        {
            "conversation_id": row["conversation_id"],
            "title": row["title"] or "新会话",
            "updated_at": serialize_datetime(row["updated_at"]),
            "message_count": int(row["message_count"] or 0),
        }
        for row in rows
    ]


def serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def parse_datetime(value):
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.now()
