"""
Telegram handlers for chatmind.

Two responsibilities:
- When a photo is sent to the chat, embed it with CLIP and save it
  to the database (indexing).
- When the /find command is used, embed the query text and search
  for the most similar previously indexed photos.
"""

import io
import logging

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from PIL import Image

from processing.embeddings import embed_image, embed_text
from storage.db import save_item, search_similar

logger = logging.getLogger(__name__)

router = Router()


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    """Index a photo sent to the chat."""
    photo = message.photo[-1]  # highest resolution version available
    file = await message.bot.get_file(photo.file_id)
    file_bytes = await message.bot.download_file(file.file_path)

    image = Image.open(io.BytesIO(file_bytes.read())).convert("RGB")
    embedding = embed_image(image)

    item_id = save_item(
        chat_id=message.chat.id,
        message_id=message.message_id,
        file_id=photo.file_id,
        media_type="photo",
        embedding=embedding,
        sender_id=message.from_user.id if message.from_user else None,
        sender_name=message.from_user.full_name if message.from_user else None,
        caption=message.caption,
    )

    logger.info("Indexed photo id=%s from chat=%s", item_id, message.chat.id)


@router.message(Command("find"))
async def handle_find(message: Message) -> None:
    """Search for a photo matching the given text description."""
    query = (message.text or "").removeprefix("/find").strip()

    if not query:
        await message.reply("Usage: /find <description>")
        return

    query_embedding = embed_text(query)
    results = search_similar(
        query_embedding=query_embedding, chat_id=message.chat.id, limit=3
    )

    if not results:
        await message.reply("Nothing found yet.")
        return

    for item in results:
        caption_lines = [f"similarity: {item['similarity']:.2f}"]
        if item["caption"]:
            caption_lines.append(item["caption"])
        await message.answer_photo(
            item["file_id"], caption="\n".join(caption_lines)
        )