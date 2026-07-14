"""
Quick manual test for storage/db.py.

Inserts a few fake items with random embeddings and runs a similarity
search, just to confirm the database layer works end to end before
plugging in real CLIP embeddings.

Run with:
    python test_db.py
"""

import random

from storage.db import save_item, search_similar

EMBEDDING_DIM = 512


def random_vector() -> list[float]:
    return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]


def main() -> None:
    print("Inserting fake items...")

    fake_items = [
        {
            "chat_id": 1,
            "message_id": 101,
            "file_id": "fake_file_id_1",
            "media_type": "photo",
            "sender_name": "Dima",
            "caption": "payment screenshot",
        },
        {
            "chat_id": 1,
            "message_id": 102,
            "file_id": "fake_file_id_2",
            "media_type": "photo",
            "sender_name": "Dima",
            "caption": "meeting screenshot",
        },
        {
            "chat_id": 1,
            "message_id": 103,
            "file_id": "fake_file_id_3",
            "media_type": "photo",
            "sender_name": "Dima",
            "caption": "random cat photo",
        },
    ]

    inserted_ids = []
    for item in fake_items:
        new_id = save_item(embedding=random_vector(), **item)
        inserted_ids.append(new_id)
        print(f"  Inserted id={new_id} ({item['caption']})")

    print("\nRunning similarity search with a random query vector...")
    results = search_similar(query_embedding=random_vector(), chat_id=1, limit=5)

    print(f"\nFound {len(results)} result(s):")
    for row in results:
        print(
            f"  id={row['id']} caption={row['caption']!r} "
            f"similarity={row['similarity']:.4f}"
        )


if __name__ == "__main__":
    main()