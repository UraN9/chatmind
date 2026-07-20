"""
Shared pytest fixtures for the chatmind test suite.

Key idea: config.DB_NAME is overridden to point at a dedicated test
database (chatmind_test) for the whole test session, so tests never
touch real data. storage/db.py reads config.DB_NAME at call time
(not at import time), which is exactly what makes this override work.
"""

import pytest

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