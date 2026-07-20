"""
Tests for storage/db.py: saving items and searching by similarity.

Uses fake (but deterministic) embeddings instead of real CLIP vectors,
so these tests are fast and don't require the model to be loaded.
"""

import random

import config
from storage.db import save_item, search_similar, get_connection


def _random_vector() -> list[float]:
    return [random.uniform(-1, 1) for _ in range(512)]


def test_save_item_returns_an_id(sample_embedding):
    item_id = save_item(
        chat_id=1,
        message_id=101,
        file_id="fake_file_id",
        media_type="photo",
        embedding=sample_embedding,
        caption="a test item",
    )

    assert isinstance(item_id, int)
    assert item_id > 0


def test_search_similar_returns_saved_items(sample_embedding):
    save_item(
        chat_id=1,
        message_id=101,
        file_id="fake_file_id",
        media_type="photo",
        embedding=sample_embedding,
        caption="a test item",
    )

    results = search_similar(query_embedding=sample_embedding, chat_id=1)

    assert len(results) == 1
    assert results[0]["caption"] == "a test item"


def test_search_similar_ranks_closer_vectors_higher():
    close_vector = [1.0] * 512
    far_vector = [-1.0] * 512

    save_item(
        chat_id=1,
        message_id=1,
        file_id="close_item",
        media_type="photo",
        embedding=close_vector,
        caption="close item",
    )
    save_item(
        chat_id=1,
        message_id=2,
        file_id="far_item",
        media_type="photo",
        embedding=far_vector,
        caption="far item",
    )

    results = search_similar(query_embedding=close_vector, chat_id=1, limit=2)

    assert results[0]["caption"] == "close item"
    assert results[1]["caption"] == "far item"
    assert results[0]["similarity"] > results[1]["similarity"]


def test_search_similar_respects_chat_id_filter(sample_embedding):
    save_item(
        chat_id=1,
        message_id=1,
        file_id="item_in_chat_1",
        media_type="photo",
        embedding=sample_embedding,
    )
    save_item(
        chat_id=2,
        message_id=1,
        file_id="item_in_chat_2",
        media_type="photo",
        embedding=sample_embedding,
    )

    results = search_similar(query_embedding=sample_embedding, chat_id=1)

    assert len(results) == 1
    assert results[0]["file_id"] == "item_in_chat_1"


def test_search_similar_respects_limit(sample_embedding):
    for i in range(5):
        save_item(
            chat_id=1,
            message_id=i,
            file_id=f"item_{i}",
            media_type="photo",
            embedding=_random_vector(),
        )

    results = search_similar(query_embedding=sample_embedding, chat_id=1, limit=3)

    assert len(results) == 3


def test_connection_points_at_test_database():
    """
    Explicit safety check: confirms every test in this suite is
    actually talking to chatmind_test, not the real database.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            db_name = cur.fetchone()[0]

    assert db_name == config.TEST_DB_NAME
    assert db_name != "chatmind"