# chatmind

A Telegram bot that indexes photos and screenshots shared in a chat and lets you find them later using a natural-language description (semantic search powered by CLIP + pgvector, with OCR full-text search as a first pass).

---

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat&logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)
[![codecov](https://codecov.io/gh/UraN9/chatmind/graph/badge.svg)](https://codecov.io/gh/UraN9/chatmind)

## Features

- **Photo indexing** — any photo sent to a chat is embedded with CLIP and OCR'd for visible text, then saved to Postgres. The bot reacts 👍 on success, ⚠️ if the file isn't readable as an image, 👎 on unexpected errors.
- **`/find <description>`** — searches OCR text first (exact/fuzzy text match), falling back to CLIP visual similarity if nothing matches. Results are shown one at a time in a single-photo gallery with ⬅️ / N of total / ➡️ navigation, so browsing candidates edits one message in place instead of flooding the chat.
- **❤️ Favorites** — toggle a favorite right from the gallery view; `/favorites` browses only favorited photos, same gallery UI. Favorites are shared per-chat, not per-user.
- **`/stats`** — how many photos are indexed, how many have OCR text, the date range, and (for group chats) top senders by photo count.
- **`/start` / `/help`** — onboarding with inline buttons (Search / Help) and a guided search prompt flow.

## How it works

- Send a photo to a chat the bot is in → it's embedded with CLIP, OCR'd, and saved to Postgres (`pgvector` for the embedding column).
- Send `/find <description>` → the bot searches OCR text via Postgres full-text search; if nothing matches, it embeds your query with CLIP and does a cosine-similarity search instead.
- Tap ❤️ on any result to save it to `/favorites`.

## Project structure

```
chatmind/
├── Dockerfile              # bot image: python:3.12-slim + tesseract-ocr + CPU-only torch
├── docker-compose.yml      # db (pgvector) + bot services
├── .env.docker.example     # template for container-specific env config
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # pytest on every PR/push; build+push image to GHCR on merge to main
├── config.py                # centralized configuration (reads .env)
├── pytest.ini
├── requirements.txt
├── db/
│   └── init.sql             # database schema (main + chatmind_test databases)
├── bot/                     # Telegram-specific logic
│   ├── main.py                # entry point, starts polling
│   └── handlers.py            # indexing, /find gallery, /favorites, /stats, /start, /help
├── processing/               # CLIP embeddings + OCR
│   ├── embeddings.py
│   └── ocr.py
├── storage/
│   └── db.py                  # database access layer
└── tests/                     # pytest suite (runs against chatmind_test, real CLIP/OCR)
    ├── conftest.py
    ├── test_db.py
    ├── test_embeddings.py
    ├── test_ocr.py
    ├── test_integration.py
    ├── test_handlers.py
    └── test_main.py
```

## Setup

There are two ways to run chatmind: in Docker (recommended, closest to how it'd run in production) or directly on the host with a virtualenv.

### Option A — Docker

1. Copy the env template and fill in your real values (`POSTGRES_PASSWORD`, `TELEGRAM_BOT_TOKEN`):
   ```bash
   cp .env.docker.example .env.docker
   ```
2. Build and start everything (Postgres + bot):
   ```bash
   docker compose up --build
   ```
   First run downloads CLIP weights (~600MB) into a persistent `hf_cache` volume, so subsequent restarts are fast.

### Option B — Local (venv)

1. Start just the database:
   ```bash
   docker compose up -d db
   ```
2. Set up the Python environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows (git bash)
   pip install -r requirements.txt
   ```
3. Copy `.env` and fill in your local values (host paths for `TESSERACT_CMD`/`HF_HOME` if needed, `POSTGRES_HOST=localhost`, `POSTGRES_PORT=5433` to match the host-mapped port in `docker-compose.yml`).
4. Run the bot:
   ```bash
   python -m bot.main
   ```

If it starts correctly, you'll see logs like:

```
[INFO] root: Starting chatmind bot...
[INFO] aiogram.dispatcher: Run polling for bot @your_bot_username
```

> **Note:** `.env` (host/local dev) and `.env.docker` (container config) are deliberately separate — see the comments in `.env.docker.example` for why (Windows paths for `TESSERACT_CMD`/`HF_HOME` don't resolve inside the Linux container, and `POSTGRES_HOST` differs between `localhost` and the `db` service name).

## Running tests

The test suite runs against a dedicated `chatmind_test` database (created automatically by `db/init.sql`), using real CLIP embeddings and real Tesseract OCR rather than mocks — only the Telegram API boundary (`Message`/`CallbackQuery`/`Bot`) is mocked in `test_handlers.py`/`test_main.py`, so it never touches real chat data:

```bash
pytest -v
```

Coverage reports (terminal, HTML, and XML for Codecov) are generated automatically via `pytest.ini`.

## CI/CD

- **CI** — every push and pull request runs the full test suite against a real `pgvector` service container in GitHub Actions, with coverage uploaded to [Codecov](https://codecov.io/gh/UraN9/chatmind).
- **CD** — after a merge to `main`, the Docker image is built and pushed to `ghcr.io/uran9/chatmind`.
- `main` is protected: direct pushes are blocked, and merging requires the `test` check to pass.

## Status

- [x] Database schema (pgvector, main + test databases)
- [x] Database access layer
- [x] CLIP embeddings + OCR
- [x] Photo indexing with reaction feedback (👍/⚠️/👎)
- [x] `/find` semantic + text search with single-photo gallery navigation
- [x] `/favorites` (❤️ toggle, shared per-chat)
- [x] `/stats`
- [x] Pytest suite with isolated test database (handlers + main included)
- [x] Docker + docker-compose
- [x] CI/CD (GitHub Actions, GHCR, branch protection)
- [x] Codecov coverage tracking
- [ ] `/delete` for removing an indexed photo
- [ ] Privacy policy

## License

MIT