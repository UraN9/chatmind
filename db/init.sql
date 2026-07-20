-- This script runs once, when the Postgres container is first
-- initialized (empty volume). It sets up two databases with the
-- same schema:
--   - the main database (POSTGRES_DB, e.g. "chatmind")
--   - a dedicated test database ("chatmind_test"), used by pytest
--     so tests never touch real data.

-- --- Main database (already the active connection at this point) ---

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS media_items (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    sender_id BIGINT,
    sender_name TEXT,

    file_id TEXT NOT NULL,          -- Telegram file_id, no need to re-download the file
    media_type TEXT NOT NULL,       -- 'photo' | 'document' | 'video' etc.

    ocr_text TEXT,                  -- text extracted from the image (OCR)
    caption TEXT,                   -- original caption, if any

    -- CLIP ViT-B/32 produces a 512-dimensional vector.
    -- Update this dimension if you switch to a different model.
    embedding vector(512),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS media_items_embedding_idx
    ON media_items USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS media_items_chat_id_idx ON media_items (chat_id);
CREATE INDEX IF NOT EXISTS media_items_sender_id_idx ON media_items (sender_id);
CREATE INDEX IF NOT EXISTS media_items_created_at_idx ON media_items (created_at);

CREATE INDEX IF NOT EXISTS media_items_ocr_text_fts_idx
    ON media_items USING gin (to_tsvector('simple', coalesce(ocr_text, '')));

-- --- Test database (used exclusively by pytest) ---

CREATE DATABASE chatmind_test;

\connect chatmind_test

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS media_items (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    sender_id BIGINT,
    sender_name TEXT,
    file_id TEXT NOT NULL,
    media_type TEXT NOT NULL,
    ocr_text TEXT,
    caption TEXT,
    embedding vector(512),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS media_items_embedding_idx
    ON media_items USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS media_items_chat_id_idx ON media_items (chat_id);
CREATE INDEX IF NOT EXISTS media_items_sender_id_idx ON media_items (sender_id);
CREATE INDEX IF NOT EXISTS media_items_created_at_idx ON media_items (created_at);

CREATE INDEX IF NOT EXISTS media_items_ocr_text_fts_idx
    ON media_items USING gin (to_tsvector('simple', coalesce(ocr_text, '')));