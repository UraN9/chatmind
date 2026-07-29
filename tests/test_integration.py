"""
End-to-end integration tests: real CLIP embeddings (and OCR, where
available) combined with the actual database layer. Confirms the
full pipeline works together, not just each piece in isolation.

Uses the synthetic image fixtures from conftest.py, so no external
image files are needed to run these tests.
"""

import pytesseract
import pytest

from processing.embeddings import embed_image, embed_text
from processing.ocr import extract_text
from storage.db import save_item, search_similar, search_by_ocr_text

TEST_CHAT_ID = 999999


def test_saved_image_is_found_by_matching_text_query(red_image, blue_image):
    red_id = save_item(
        chat_id=TEST_CHAT_ID,
        message_id=1,
        file_id="red_image",
        media_type="photo",
        embedding=embed_image(red_image),
        caption="red square",
    )
    save_item(
        chat_id=TEST_CHAT_ID,
        message_id=2,
        file_id="blue_image",
        media_type="photo",
        embedding=embed_image(blue_image),
        caption="blue square",
    )

    query_embedding = embed_text("a photo of the color red")
    results = search_similar(
        query_embedding=query_embedding, chat_id=TEST_CHAT_ID, limit=1
    )

    assert len(results) == 1
    assert results[0]["id"] == red_id


def test_saved_image_is_found_by_ocr_text(text_image):
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip("Tesseract binary not found or not configured")

    ocr_text = extract_text(text_image)
    item_id = save_item(
        chat_id=TEST_CHAT_ID,
        message_id=3,
        file_id="text_image",
        media_type="photo",
        embedding=embed_image(text_image),
        ocr_text=ocr_text,
    )

    results = search_by_ocr_text(
        query_text="hello", chat_id=TEST_CHAT_ID, limit=1
    )

    assert len(results) == 1
    assert results[0]["id"] == item_id