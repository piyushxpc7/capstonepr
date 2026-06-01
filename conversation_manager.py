"""Conversation history manager for ChatGPT-like multi-turn interactions."""
import json
import os
import sqlite3
from datetime import datetime
from typing import Optional
import uuid

# Get the project root directory
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "telecom_ops.db")


def _get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_conversation(title: Optional[str] = None) -> str:
    """Create a new conversation. Returns conversation_id."""
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations (conversation_id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (conv_id, title or "New Conversation", now, now)
    )
    conn.commit()
    conn.close()
    return conv_id


def get_conversations() -> list[dict]:
    """Get all active conversations, sorted by most recent first."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT conversation_id, title, created_at, updated_at, follow_up_count "
        "FROM conversations WHERE is_active = 1 "
        "ORDER BY updated_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_conversation_messages(conversation_id: str) -> list[dict]:
    """Get all messages in a conversation."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT message_id, conversation_id, role, content, final_response, execution_trace, created_at "
        "FROM conversation_messages "
        "WHERE conversation_id = ? "
        "ORDER BY message_id ASC",
        (conversation_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    final_response: Optional[str] = None,
    execution_trace: Optional[list] = None
) -> int:
    """Add a message to conversation. Returns message_id."""
    now = datetime.utcnow().isoformat()
    trace_json = json.dumps(execution_trace) if execution_trace else None

    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversation_messages "
        "(conversation_id, role, content, final_response, execution_trace, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (conversation_id, role, content, final_response, trace_json, now)
    )
    message_id = cur.lastrowid

    # Update conversation's updated_at
    cur.execute(
        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
        (now, conversation_id)
    )

    # Increment follow_up_count if role is user
    if role == "user":
        cur.execute(
            "UPDATE conversations SET follow_up_count = follow_up_count + 1 "
            "WHERE conversation_id = ?",
            (conversation_id,)
        )

    conn.commit()
    conn.close()
    return message_id


def get_conversation_info(conversation_id: str) -> Optional[dict]:
    """Get conversation metadata."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT conversation_id, title, created_at, updated_at, follow_up_count "
        "FROM conversations WHERE conversation_id = ?",
        (conversation_id,)
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_conversation(conversation_id: str) -> None:
    """Soft-delete a conversation (mark as inactive)."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE conversations SET is_active = 0 WHERE conversation_id = ?",
        (conversation_id,)
    )
    conn.commit()
    conn.close()


def get_conversation_context(conversation_id: str) -> str:
    """Build a context string from conversation history for the graph.

    Returns: "User: {msg1}\nAssistant: {response1}\nUser: {msg2}\n..." format
    """
    messages = get_conversation_messages(conversation_id)
    context_parts = []
    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        context_parts.append(f"{role_label}: {content}")
    return "\n".join(context_parts)
