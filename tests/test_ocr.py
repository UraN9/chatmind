"""
Tests for processing/ocr.py.

Uses a synthetic image with known rendered text (see the text_image
fixture in conftest.py), so these tests don't depend on any external
screenshot file.

If the Tesseract binary isn't installed/configured on this machine,
these tests are skipped rather than failing with a confusing error
(see the check at import time below).
"""

import pytesseract
import pytest

from processing.ocr import extract_text

try:
    pytesseract.get_tesseract_version()
except Exception:
    pytest.skip(
        "Tesseract binary not found or not configured (see TESSERACT_CMD "
        "in .env) - skipping OCR tests",
        allow_module_level=True,
    )


def test_extract_text_reads_rendered_text(text_image):
    result = extract_text(text_image)
    assert "HELLO" in result.upper()


def test_extract_text_returns_empty_string_for_image_without_text(red_image):
    result = extract_text(red_image)
    assert result == ""