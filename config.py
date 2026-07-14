"""
Centralized configuration for chatmind.

Loads environment variables from .env once, so every other module
can just import the values it needs instead of calling os.getenv()
directly.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Database
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "chatmind")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# Telegram bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Works around a known Windows issue where torch and other libraries
# (e.g. numpy) each bundle their own copy of the OpenMP runtime,
# which can cause silent crashes (segmentation faults) when both are
# loaded in the same process. Safe to leave enabled on Linux/Mac too.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")