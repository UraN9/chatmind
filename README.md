# chatmind

A Telegram bot that indexes photos and screenshots shared in a chat and lets you find them later using a natural-language description (semantic search powered by CLIP + pgvector).
---
![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?style=flat&logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-In_Development-yellow?style=flat)

## How it works

- Send a photo to a chat the bot is in → it's embedded with CLIP and saved to Postgres.
- Send `/find <description>` → the bot embeds your query text, searches for the closest matching photos, and sends them back.

## Project structure

```
chatmind/
├── docker-compose.yml   # Postgres + pgvector
├── config.py            # centralized configuration
├── pytest.ini           # pytest configuration
├── db/
│   └── init.sql         # database schema (main + test databases)
├── bot/                  # Telegram-specific logic
│   ├── main.py            # entry point, starts polling
│   └── handlers.py        # photo indexing, /find command
├── processing/           # CLIP embeddings (OCR coming later)
├── storage/               # database access layer
├── tests/                 # pytest test suite (runs against chatmind_test)
└── requirements.txt
```

## Setup

### 1. Database

```bash
docker compose up -d
```

### 2. Python environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows (git bash)
pip install -r requirements.txt
```

## Running the bot

```bash
python -m bot.main
```

If it starts correctly, you'll see logs like:

```
[INFO] root: Starting chatmind bot...
[INFO] aiogram.dispatcher: Run polling for bot @your_bot_username
```

## Running tests

The test suite runs against a dedicated `chatmind_test` database (created automatically by `db/init.sql` on a fresh setup), so it never touches real data:

```bash
pytest -v
```

## Status

- [x] Database schema (pgvector, main + test databases)
- [x] Database access layer
- [x] Processing module (CLIP embeddings)
- [x] Telegram bot (photo indexing + `/find` search)
- [x] Pytest suite with isolated test database
- [ ] OCR for text inside screenshots
- [ ] Bot UX improvements (indexing confirmation, error handling)