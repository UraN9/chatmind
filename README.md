# chatmind

A Telegram bot that indexes photos and screenshots shared in a chat and lets you find them later using a natural-language description (semantic search powered by CLIP + pgvector).

## Project structure

```
chatmind/
├── docker-compose.yml   # Postgres + pgvector
├── .env                 # environment variables
├── db/
│   └── init.sql         # database schema
├── bot/                 # Telegram-specific logic
├── processing/          # OCR and CLIP embeddings
├── storage/             # database access layer
└── requirements.txt
```

## Running the database

```bash
docker compose up -d
```

## Status

- [x] Database schema (pgvector)
- [ ] Processing module (OCR + CLIP)
- [ ] Database access layer
- [ ] Telegram bot