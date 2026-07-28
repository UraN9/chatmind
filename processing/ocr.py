"""
OCR for chatmind.

Extracts text from screenshots and other images using Tesseract, so
photos can later be found by searching for text that appears on them
(e.g. "the screenshot where it said the invoice was overdue").

Requires the Tesseract binary to be installed separately on the
system (pytesseract is just a thin Python wrapper around it):
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Mac: brew install tesseract
- Linux: apt install tesseract-ocr

If Tesseract isn't on your system PATH, set TESSERACT_CMD in .env to
the full path of the executable (e.g. on Windows, something like
C:\\Program Files\\Tesseract-OCR\\tesseract.exe).
"""

from pathlib import Path
from typing import Union

import pytesseract
from PIL import Image

import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def extract_text(image: Union[str, Path, Image.Image]) -> str:
    """
    Run OCR on an image and return the recognized text (stripped of
    leading/trailing whitespace). Returns an empty string if nothing
    was recognized or the image contains no readable text.

    `image` can be a file path (str or Path) or an already-opened
    PIL.Image.Image.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    text = pytesseract.image_to_string(image)
    return text.strip()