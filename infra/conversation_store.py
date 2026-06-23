import sqlite3
import uuid
import re
from datetime import datetime
from pathlib import Path


DATA_DIR = Path("data")
DATABASE_PATH = DATA_DIR / "app.db"
DEFAULT_CONVERSATION_ID = "default"
VALID_CONVERSATION_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_VISUAL_HISTORY_ITEMS = 20


def normalize_conversation_id(conversation_id):
    if not conversation_id:
        return DEFAULT_CONVERSATION_ID
    conversation_id = str(conversation_id).strip()
    if VALID_CONVERSATION_ID.fullmatch(conversation_id):
        return conversation_id
    return uuid.uuid4().hex


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    initialize_database(connection)
    return connection


def initialize_database(connection):
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '新会话',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            image_url TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id)
                ON DELETE CASCADE
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS visual_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            image_url TEXT NOT NULL DEFAULT '',
            user_input TEXT NOT NULL DEFAULT '',
            assistant_text TEXT NOT NULL DEFAULT '',
            image_prompt TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(conversation_id)
                ON DELETE CASCADE
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id
        ON messages(conversation_id, id)
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_visual_history_conversation_id_id
        ON visual_history(conversation_id, id)
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
        ON conversations(updated_at)
    """)
    connection.commit()


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


def ensure_conversation(connection, conversation_id, title="新会话"):
    safe_id = normalize_conversation_id(conversation_id)
    now = datetime.now().isoformat()
    connection.execute(
        """
        INSERT OR IGNORE INTO conversations (
            conversation_id,
            title,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (safe_id, title or "新会话", now, now),
    )
    return safe_id


def create_conversation(title="新会话"):
    conversation_id = uuid.uuid4().hex
    now = datetime.now().isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (
                conversation_id,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title or "新会话", now, now),
        )
    return load_conversation(conversation_id)


def load_conversation(conversation_id):
    safe_id = normalize_conversation_id(conversation_id)
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT conversation_id, title, created_at, updated_at
            FROM conversations
            WHERE conversation_id = ?
            """,
            (safe_id,),
        ).fetchone()

        if row is None:
            return create_empty_conversation(safe_id)

        messages = load_messages(connection, safe_id)
        visual_history = load_visual_history(connection, safe_id)
        return {
            "conversation_id": row["conversation_id"],
            "title": row["title"] or "新会话",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": messages,
            "visual_history": visual_history,
        }


def load_messages(connection, conversation_id):
    rows = connection.execute(
        """
        SELECT role, content, image_url
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    ).fetchall()
    messages = []
    for row in rows:
        message = {
            "role": row["role"],
            "content": row["content"] or "",
        }
        if row["image_url"]:
            message["image_url"] = row["image_url"]
        messages.append(message)
    return messages


def load_visual_history(connection, conversation_id):
    rows = connection.execute(
        """
        SELECT turn_id, image_url, user_input, assistant_text, image_prompt, created_at
        FROM visual_history
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [normalize_visual_record(dict(row)) for row in rows]


def save_conversation(conversation):
    conversation_id = normalize_conversation_id(conversation.get("conversation_id"))
    now = datetime.now().isoformat()
    messages = conversation.get("messages", [])
    visual_history = conversation.get("visual_history", [])
    title = conversation.get("title") or build_conversation_title(conversation)

    with get_connection() as connection:
        ensure_conversation(connection, conversation_id, title)
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (title or "新会话", now, conversation_id),
        )
        connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        connection.execute("DELETE FROM visual_history WHERE conversation_id = ?", (conversation_id,))
        insert_messages(connection, conversation_id, messages)
        insert_visual_history(connection, conversation_id, visual_history)
    return load_conversation(conversation_id)


def append_messages(conversation_id, messages):
    safe_id = normalize_conversation_id(conversation_id)
    now = datetime.now().isoformat()
    with get_connection() as connection:
        ensure_conversation(connection, safe_id)
        insert_messages(connection, safe_id, messages)
        maybe_update_title(connection, safe_id)
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (now, safe_id),
        )
    return load_conversation(safe_id)


def insert_messages(connection, conversation_id, messages):
    now = datetime.now().isoformat()
    for item in messages or []:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        image_url = str(item.get("image_url") or "").strip()
        if role not in ["user", "assistant"] or not content:
            continue
        connection.execute(
            """
            INSERT INTO messages (
                conversation_id,
                role,
                content,
                image_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, image_url, now),
        )


def append_visual_history(conversation_id, visual_record):
    safe_id = normalize_conversation_id(conversation_id)
    record = normalize_visual_record(visual_record)
    if not record:
        return load_conversation(safe_id)

    with get_connection() as connection:
        ensure_conversation(connection, safe_id)
        insert_visual_history(connection, safe_id, [record])
        trim_visual_history(connection, safe_id)
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (datetime.now().isoformat(), safe_id),
        )
    return load_conversation(safe_id)


def insert_visual_history(connection, conversation_id, visual_history):
    for item in visual_history or []:
        record = normalize_visual_record(item)
        if not record:
            continue
        connection.execute(
            """
            INSERT INTO visual_history (
                conversation_id,
                turn_id,
                image_url,
                user_input,
                assistant_text,
                image_prompt,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                record["turn_id"],
                record["image_url"],
                record["user_input"],
                record["assistant_text"],
                record["image_prompt"],
                record["created_at"],
            ),
        )


def trim_visual_history(connection, conversation_id):
    connection.execute(
        """
        DELETE FROM visual_history
        WHERE conversation_id = ?
          AND id NOT IN (
              SELECT id
              FROM visual_history
              WHERE conversation_id = ?
              ORDER BY id DESC
              LIMIT ?
          )
        """,
        (conversation_id, conversation_id, MAX_VISUAL_HISTORY_ITEMS),
    )


def maybe_update_title(connection, conversation_id):
    row = connection.execute(
        "SELECT title FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if row and row["title"] != "新会话":
        return

    first_user_message = connection.execute(
        """
        SELECT content
        FROM messages
        WHERE conversation_id = ? AND role = 'user'
        ORDER BY id ASC
        LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    if not first_user_message:
        return

    title = str(first_user_message["content"] or "").strip()[:24] or "新会话"
    connection.execute(
        "UPDATE conversations SET title = ? WHERE conversation_id = ?",
        (title, conversation_id),
    )


def delete_conversation(conversation_id):
    safe_id = normalize_conversation_id(conversation_id)
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (safe_id,),
        )
        return cursor.rowcount > 0


def get_history_messages(conversation_id):
    conversation = load_conversation(conversation_id)
    history = []
    for item in conversation.get("messages", []):
        role = item.get("role")
        content = item.get("content")
        if role in ["user", "assistant"] and isinstance(content, str) and content.strip():
            history.append({
                "role": role,
                "content": content.strip(),
            })
    return history


def get_recent_visual_history(conversation_id, limit=3):
    safe_id = normalize_conversation_id(conversation_id)
    with get_connection() as connection:
        ensure_conversation(connection, safe_id)
        rows = connection.execute(
            """
            SELECT turn_id, image_url, user_input, assistant_text, image_prompt, created_at
            FROM visual_history
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_id, limit),
        ).fetchall()
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
        "created_at": str(record.get("created_at") or datetime.now().isoformat()).strip(),
    }


def build_conversation_title(conversation):
    for item in conversation.get("messages", []):
        if item.get("role") == "user":
            content = str(item.get("content") or "").strip()
            if content:
                return content[:24]
    return "新会话"


def list_conversations():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.conversation_id,
                c.title,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m
                ON c.conversation_id = m.conversation_id
            GROUP BY c.conversation_id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()

    return [
        {
            "conversation_id": row["conversation_id"],
            "title": row["title"] or "新会话",
            "updated_at": row["updated_at"] or "",
            "message_count": int(row["message_count"] or 0),
        }
        for row in rows
    ]
