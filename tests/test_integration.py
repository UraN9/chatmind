"""
End-to-end integration test: real CLIP embeddings + Postgres/pgvector.

Saves a real image to the database using its actual CLIP embedding,
then searches for it using a plain text query, to confirm the full
pipeline works (not just each piece in isolation).

Run with:
    python test_integration.py path/to/your/image.jpg
"""

import sys

from processing.embeddings import embed_image, embed_text
from storage.db import save_item, search_similar

TEST_CHAT_ID = 999999  # dedicated fake chat_id, easy to clean up later


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_integration.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Embedding image: {image_path}")
    image_embedding = embed_image(image_path)

    print("Saving to the database...")
    item_id = save_item(
        chat_id=TEST_CHAT_ID,
        message_id=1,
        file_id="integration_test_file_id",
        media_type="photo",
        embedding=image_embedding,
        sender_name="integration_test",
        caption="integration test image",
    )
    print(f"Saved as id={item_id}")

    queries = [
        "a screenshot of a chat conversation",
        "a photo of a cat",
        "a picture of food",
    ]

    print("\nSearching with text queries:\n")
    for query in queries:
        query_embedding = embed_text(query)
        results = search_similar(
            query_embedding=query_embedding, chat_id=TEST_CHAT_ID, limit=1
        )
        top = results[0] if results else None
        if top:
            print(f"  Query: {query!r}")
            print(f"    -> id={top['id']} similarity={top['similarity']:.4f}")
        else:
            print(f"  Query: {query!r} -> no results")


if __name__ == "__main__":
    main()