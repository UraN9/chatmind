"""
Database access layer for chatmind.

Handles the connection to Postgres and provides simple functions
to save indexed media items and search for similar ones by embedding.
"""

from typing import Optional

import psycopg
from pgvector.psycopg import register_vector

import config


def get_connection() -> psycopg.Connection:
    """
    Open a new connection to the database and register the pgvector
    type adapter, so Python lists/arrays can be sent and received
    as `vector` columns transparently.

    Reads connection settings from the `config` module at call time
    (not at import time), so tests can point this at a different
    database by overriding config.DB_NAME before calling.
    """
    conn = psycopg.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
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
    offset: int = 0,
) -> list[dict]:
    """
    Find the media items whose embedding is closest to query_embedding,
    using cosine distance. Optionally restrict the search to one chat.

    `offset` skips the first N closest matches, useful for pagination
    (e.g. "show me the next batch of results").

    Returns a list of dicts, ordered from most to least similar.
    """
    sql = """
        SELECT
            id,
            file_id,
            media_type,
            ocr_text,
            caption,
            sender_id,
            sender_name,
            created_at,
            is_favorite,
            1 - (embedding <=> %s::vector) AS similarity
        FROM media_items
        WHERE embedding IS NOT NULL
    """
    params: list = [query_embedding]

    if chat_id is not None:
        sql += " AND chat_id = %s"
        params.append(chat_id)

    sql += " ORDER BY embedding <=> %s::vector LIMIT %s OFFSET %s"
    params.extend([query_embedding, limit, offset])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def count_similar(
    query_embedding: list[float],
    chat_id: Optional[int] = None,
) -> int:
    """
    Count how many media items would be returned by search_similar
    for the same chat filter (i.e. every embedded item in that chat,
    since cosine similarity ranks rather than thresholds). Used to
    show a "3/14" style counter in the results gallery.
    """
    sql = "SELECT COUNT(*) FROM media_items WHERE embedding IS NOT NULL"
    params: list = []

    if chat_id is not None:
        sql += " AND chat_id = %s"
        params.append(chat_id)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]


def count_by_ocr_text(
    query_text: str,
    chat_id: Optional[int] = None,
) -> int:
    """
    Count how many media items match query_text via full-text search,
    using the same WHERE clause as search_by_ocr_text. Used to show
    a "3/14" style counter in the results gallery.
    """
    sql = """
        SELECT COUNT(*)
        FROM media_items
        WHERE to_tsvector('simple', coalesce(ocr_text, ''))
              @@ plainto_tsquery('simple', %s)
    """
    params: list = [query_text]

    if chat_id is not None:
        sql += " AND chat_id = %s"
        params.append(chat_id)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]


def search_by_ocr_text(
    query_text: str,
    chat_id: Optional[int] = None,
    limit: int = 5,
    offset: int = 0,
) -> list[dict]:
    """
    Find media items whose OCR-extracted text matches query_text,
    using Postgres full-text search (the same GIN index used here is
    created in db/init.sql). Useful for finding a specific screenshot
    by text that literally appears on it (e.g. "invoice #4471"),
    which CLIP's visual similarity search isn't designed to catch.

    `offset` skips the first N matches, useful for pagination.

    Returns a list of dicts, ordered by text-match relevance.
    """
    sql = """
        SELECT
            id,
            file_id,
            media_type,
            ocr_text,
            caption,
            sender_id,
            sender_name,
            created_at,
            is_favorite,
            ts_rank(
                to_tsvector('simple', coalesce(ocr_text, '')),
                plainto_tsquery('simple', %s)
            ) AS rank
        FROM media_items
        WHERE to_tsvector('simple', coalesce(ocr_text, ''))
              @@ plainto_tsquery('simple', %s)
    """
    params: list = [query_text, query_text]

    if chat_id is not None:
        sql += " AND chat_id = %s"
        params.append(chat_id)

    sql += " ORDER BY rank DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def get_chat_stats(chat_id: int, top_senders_limit: int = 3) -> dict:
    """
    Aggregate stats for a chat, used by the /stats command:
    - total: how many media items are indexed
    - with_ocr: how many of those have non-empty OCR text (and are
      therefore also searchable via search_by_ocr_text)
    - first_photo_at / last_photo_at: created_at of the oldest/newest
      indexed item (None if the chat has nothing indexed yet)
    - top_senders: up to `top_senders_limit` (sender_name, count)
      pairs, ordered by how many photos they sent, ignoring items
      with no sender_name (e.g. forwarded/anonymous posts)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (
                        WHERE ocr_text IS NOT NULL AND ocr_text != ''
                    ) AS with_ocr,
                    MIN(created_at) AS first_photo_at,
                    MAX(created_at) AS last_photo_at
                FROM media_items
                WHERE chat_id = %s
                """,
                (chat_id,),
            )
            total, with_ocr, first_photo_at, last_photo_at = cur.fetchone()

            cur.execute(
                """
                SELECT sender_name, COUNT(*) AS count
                FROM media_items
                WHERE chat_id = %s AND sender_name IS NOT NULL
                GROUP BY sender_name
                ORDER BY count DESC
                LIMIT %s
                """,
                (chat_id, top_senders_limit),
            )
            top_senders = [
                {"sender_name": name, "count": count}
                for name, count in cur.fetchall()
            ]

    return {
        "total": total,
        "with_ocr": with_ocr,
        "first_photo_at": first_photo_at,
        "last_photo_at": last_photo_at,
        "top_senders": top_senders,
    }


def list_favorites(chat_id: int, limit: int = 5, offset: int = 0) -> list[dict]:
    """
    Fetch favorited media items for a chat, most recently indexed
    first (there's no separate "favorited at" timestamp, so this
    orders by created_at like everything else). Used by /favorites,
    which reuses the same one-at-a-time gallery as /find.
    """
    sql = """
        SELECT id, file_id, media_type, ocr_text, caption,
               sender_id, sender_name, created_at, is_favorite
        FROM media_items
        WHERE chat_id = %s AND is_favorite = TRUE
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (chat_id, limit, offset))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def count_favorites(chat_id: int) -> int:
    """Count favorited media items in a chat, for the /favorites
    gallery's "N/total" counter."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM media_items WHERE chat_id = %s AND is_favorite = TRUE",
                (chat_id,),
            )
            return cur.fetchone()[0]


def toggle_favorite(item_id: int) -> bool:
    """
    Flip is_favorite on a single media item and return the new value.
    Favorite is shared per-chat, not per-user: anyone who can see the
    photo can toggle it, same as anyone can already search for it.

    Raises ValueError if no item with that id exists.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE media_items
                SET is_favorite = NOT is_favorite
                WHERE id = %s
                RETURNING is_favorite
                """,
                (item_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"No media item with id={item_id}")
            new_value = row[0]
        conn.commit()

    return new_value


def delete_item(item_id: int) -> bool:
    """
    Remove a media item from the index (does NOT delete the underlying
    Telegram message -- the bot doesn't have permission for that, and
    it would be a different, riskier action anyway). Returns True if
    a row was actually deleted, False if no item with that id existed
    (e.g. someone else already deleted it a moment earlier).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM media_items WHERE id = %s", (item_id,))
            deleted = cur.rowcount > 0
        conn.commit()

    return deleted


def get_item_by_id(item_id: int) -> Optional[dict]:
    """
    Fetch a single media item by its id. Used when the person taps an
    inline button to pick one of several search results the bot
    presented, so the actual photo can be sent only then.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_id, media_type, ocr_text, caption,
                       sender_id, sender_name, created_at, is_favorite
                FROM media_items
                WHERE id = %s
                """,
                (item_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cur.description]

    return dict(zip(columns, row))