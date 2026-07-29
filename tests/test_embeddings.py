"""
Tests for processing/embeddings.py: CLIP image and text embeddings.

Uses small synthetic images generated on the fly (solid colors), so
these tests don't depend on any external image file. Loading the CLIP
model is slow (a few seconds) but only happens once per test session,
since processing/embeddings.py loads it once at import time and every
test in this file reuses the same loaded model.
"""

import math

from processing.embeddings import embed_image, embed_text


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_embed_image_returns_a_512_dim_vector(red_image):
    embedding = embed_image(red_image)
    assert len(embedding) == 512


def test_embed_text_returns_a_512_dim_vector():
    embedding = embed_text("a red square")
    assert len(embedding) == 512


def test_embeddings_are_normalized(red_image):
    """embed_image/embed_text both normalize their output, so cosine
    similarity in storage/db.py's <=> operator behaves as expected."""
    image_embedding = embed_image(red_image)
    text_embedding = embed_text("a red square")

    assert math.isclose(_norm(image_embedding), 1.0, abs_tol=1e-3)
    assert math.isclose(_norm(text_embedding), 1.0, abs_tol=1e-3)


def test_embedding_is_deterministic(red_image):
    """The same image should always produce the same embedding (no
    randomness in inference), which matters for reproducible search."""
    first = embed_image(red_image)
    second = embed_image(red_image)

    assert first == second


def test_different_images_produce_different_embeddings(red_image, blue_image):
    red_embedding = embed_image(red_image)
    blue_embedding = embed_image(blue_image)

    assert red_embedding != blue_embedding


def test_text_query_matches_the_right_color_image(red_image, blue_image):
    """Sanity check for CLIP's basic color understanding: a solid red
    image should score higher against "the color red" than against
    "the color blue", and vice versa. This is the same mechanism
    /find relies on in the bot."""
    red_text_embedding = embed_text("a photo of the color red")
    blue_text_embedding = embed_text("a photo of the color blue")

    red_image_embedding = embed_image(red_image)
    blue_image_embedding = embed_image(blue_image)

    assert _cosine(red_image_embedding, red_text_embedding) > _cosine(
        red_image_embedding, blue_text_embedding
    )
    assert _cosine(blue_image_embedding, blue_text_embedding) > _cosine(
        blue_image_embedding, red_text_embedding
    )