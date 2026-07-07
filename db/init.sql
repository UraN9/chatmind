-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main Table of Media Elements (Photos, Screenshots, Documents)
CREATE TABLE IF NOT EXISTS media_items (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    sender_id BIGINT,
    sender_name TEXT,

    file_id TEXT NOT NULL,          -- Telegram file_id, You don't need to download the file again
    media_type TEXT NOT NULL,       -- 'photo' | 'document' | 'video' more

    ocr_text TEXT,                  -- text extracted from an image (OCR)
    caption TEXT,                   -- photo caption, if any

    -- CLIP ViT-B/32 outputs a 512-dimensional vector.
    embedding vector(512),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for fast vector search (cosine distance)
CREATE INDEX IF NOT EXISTS media_items_embedding_idx
    ON media_items USING hnsw (embedding vector_cosine_ops);

-- Indices for typical filters
CREATE INDEX IF NOT EXISTS media_items_chat_id_idx ON media_items (chat_id);
CREATE INDEX IF NOT EXISTS media_items_sender_id_idx ON media_items (sender_id);
CREATE INDEX IF NOT EXISTS media_items_created_at_idx ON media_items (created_at);

-- Full-text search on OCR text as a fallback option
-- (useful if semantic search doesn't find anything conclusive)
CREATE INDEX IF NOT EXISTS media_items_ocr_text_fts_idx
    ON media_items USING gin (to_tsvector('simple', coalesce(ocr_text, '')));