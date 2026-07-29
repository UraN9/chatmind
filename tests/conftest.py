"""
Shared pytest fixtures for the chatmind test suite.

Key idea: config.DB_NAME is overridden to point at a dedicated test
database (chatmind_test) for the whole test session, so tests never
touch real data. storage/db.py reads config.DB_NAME at call time
(not at import time), which is exactly what makes this override work.
"""

import pytest
from PIL import Image, ImageDraw, ImageFont

import config
from storage.db import get_connection


@pytest.fixture(scope="session", autouse=True)
def use_test_database():
    """Point every database call made during tests at chatmind_test."""
    original_db_name = config.DB_NAME

    # Safety guard: refuse to run the test suite at all if the test
    # database is somehow configured to be the same as the real one
    # (e.g. a misconfigured .env). Better to fail loudly here than to
    # silently wipe real data in clean_media_items below.
    if config.TEST_DB_NAME == original_db_name:
        raise RuntimeError(
            f"TEST_DB_NAME ({config.TEST_DB_NAME!r}) must be different "
            f"from DB_NAME ({original_db_name!r}). Refusing to run tests "
            "to avoid touching real data."
        )

    config.DB_NAME = config.TEST_DB_NAME

    # Pre-flight check: actually connect and ask Postgres which
    # database we're talking to, rather than trusting the config
    # values alone. This must pass BEFORE any other fixture or test
    # runs, so a misconfiguration can never let clean_media_items (or
    # any test) touch the wrong database.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            actual_db = cur.fetchone()[0]

    if actual_db != config.TEST_DB_NAME:
        config.DB_NAME = original_db_name
        raise RuntimeError(
            f"Expected to be connected to {config.TEST_DB_NAME!r}, "
            f"but Postgres reports {actual_db!r}. Refusing to run "
            "tests to avoid touching the wrong database."
        )

    print(f"\n[conftest] Verified: running tests against database {actual_db!r}")
    yield
    config.DB_NAME = original_db_name


@pytest.fixture(autouse=True)
def clean_media_items():
    """Wipe the media_items table before and after every test."""

    def _truncate():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE media_items RESTART IDENTITY")
            conn.commit()

    _truncate()
    yield
    _truncate()


@pytest.fixture
def sample_embedding():
    """A deterministic fake 512-dim embedding, useful when the actual
    values don't matter (e.g. testing that save/search round-trips
    correctly, without needing a real CLIP model loaded)."""
    return [0.01 * i for i in range(512)]


@pytest.fixture(scope="session")
def red_image():
    """A solid red square. Cheap to generate, no external file needed."""
    return Image.new("RGB", (224, 224), color=(220, 30, 30))


@pytest.fixture(scope="session")
def blue_image():
    """A solid blue square, visually distinct from red_image."""
    return Image.new("RGB", (224, 224), color=(30, 60, 220))


@pytest.fixture(scope="session")
def text_image():
    """A plain white image with rendered text, for OCR tests."""
    image = Image.new("RGB", (800, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=48)
    except TypeError:
        # Older Pillow versions don't support the `size` argument.
        font = ImageFont.load_default()
    draw.text((20, 60), "HELLO CHATMIND", fill=(0, 0, 0), font=font)
    return image