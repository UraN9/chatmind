"""
Database access layer for chatmind.

Handles the connection to Postgres and provides simple functions
to save indexed media items and search for similar ones by embedding.
"""

from typing import Optional

import psycopg
from pgvector.psycopg import register_vector

from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection() -> psycopg.Connection:
    """
    Open a new connection to the database and register the pgvector
    type adapter, so Python lists/arrays can be sent and received
    as `vector` columns transparently.
    """
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    register_vector(conn)
    return conn


def save_item(
    chat_id: int,
    message_id: int,
    file_id: str,
    media_type: str,
    embedding: list[float],
    sender_id: Optional[int] = None,
    sender_name: Optional[str] = None,
    ocr_text: Optional[str] = None,
    caption: Optional[str] = None,
) -> int:
    """
    Insert a new media item into the database.
    Returns the id of the newly created row.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO media_items
                    (chat_id, message_id, sender_id, sender_name,
                     file_id, media_type, ocr_text, caption, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    chat_id,
                    message_id,
                    sender_id,
                    sender_name,
                    file_id,
                    media_type,
                    ocr_text,
                    caption,
                    embedding,
                ),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
    return new_id


def search_similar(
    query_embedding: list[float],
    chat_id: Optional[int] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Find the media items whose embedding is closest to query_embedding,
    using cosine distance. Optionally restrict the search to one chat.

    Returns a list of dicts, ordered from most to least similar.
    """
    sql = """
        SELECT
            id,
            file_id,
            media_type,
            ocr_text,
            caption,
            sender_name,
            created_at,
            1 - (embedding <=> %s::vector) AS similarity
        FROM media_items
        WHERE embedding IS NOT NULL
    """
    params: list = [query_embedding]

    if chat_id is not None:
        sql += " AND chat_id = %s"
        params.append(chat_id)

    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([query_embedding, limit])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]