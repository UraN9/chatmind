"""
Quick manual test for processing/embeddings.py.

Takes an image path and a few candidate text descriptions, computes
CLIP embeddings for all of them, and prints how well each description
matches the image (cosine similarity, since embeddings are normalized
this is just a dot product).

Run with:
    python test_embeddings.py path/to/your/image.jpg
"""

import sys

from processing.embeddings import embed_image, embed_text


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_embeddings.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    candidate_texts = [
        "a screenshot of a chat conversation",
        "a photo of a cat",
        "a picture of food",
        "a screenshot with a payment or invoice",
        "a photo of nature or landscape",
        "a meme or funny picture",
    ]

    print(f"Loading and embedding image: {image_path}")
    image_embedding = embed_image(image_path)

    print("\nComparing against candidate descriptions:\n")

    results = []
    for text in candidate_texts:
        text_embedding = embed_text(text)
        score = cosine_similarity(image_embedding, text_embedding)
        results.append((text, score))

    results.sort(key=lambda pair: pair[1], reverse=True)

    for text, score in results:
        print(f"  {score:.4f}  {text}")

    print(f"\nBest match: {results[0][0]!r} (score={results[0][1]:.4f})")


if __name__ == "__main__":
    main()