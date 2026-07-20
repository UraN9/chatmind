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

# Dedicated database used by the pytest suite (tests/conftest.py points
# config.DB_NAME here for the duration of the test session), so tests
# never touch real data.
TEST_DB_NAME = os.getenv("POSTGRES_TEST_DB", "chatmind_test")

# Telegram bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Works around a known Windows issue where torch and other libraries
# (e.g. numpy) each bundle their own copy of the OpenMP runtime,
# which can cause silent crashes (segmentation faults) when both are
# loaded in the same process. Safe to leave enabled on Linux/Mac too.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Optional override for where Hugging Face / open_clip cache downloaded
# model weights. Useful on Windows if the default cache (inside the
# user profile) ends up inside a OneDrive-synced folder, which can
# cause crashes (access violations) when safetensors memory-maps the
# file while OneDrive is syncing it. Set HF_HOME in .env to a local,
# non-synced path (e.g. C:\hf_cache) if you hit that issue.
# (load_dotenv() above already puts it into os.environ if it's set.)