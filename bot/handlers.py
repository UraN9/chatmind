"""
Telegram handlers for chatmind.

Responsibilities:
- /start and /help: explain what the bot does.
- When a photo is sent to the chat, embed it with CLIP (and run OCR
  on it) and save it to the database (indexing).
- When the /find command is used, search for the closest matching
  photos and send back a batch of candidates. If none of them are
  the right one, tapping "Show more" swaps those same photo messages
  in place for the next batch (edits them), instead of sending new
  messages and cluttering the chat.
"""

import io
import logging
import uuid

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
    ReactionTypeEmoji,
)
from PIL import Image, UnidentifiedImageError

from processing.embeddings import embed_image, embed_text
from processing.ocr import extract_text
from storage.db import save_item, search_by_ocr_text, search_similar

logger = logging.getLogger(__name__)

router = Router()

WELCOME_TEXT = (
    "🧠 Welcome to ChatMind\n\n"
    "Your AI-powered photo memory.\n\n"
    "Send me photos and I'll remember them for you.\n"
    "Later, just describe what you're looking for, and I'll find it in seconds.\n\n"
    "✨ Try searching for:\n"
    "• Red Ferrari\n"
    "• Amazon receipt\n"
    "• Cat sleeping on a sofa\n"
    "• Sunset over the mountains"
)

HELP_TEXT = (
    "📖 How ChatMind works\n\n"
    "📸 Send a photo\n"
    "I'll analyze it, extract visible text, and save it for future searches.\n\n"
    "🔍 Find it later\n"
    "Use /find <description> to search by objects, scenes, or text inside the image.\n\n"
    "💡 Tips\n"
    "• English descriptions usually give the best results.\n"
    "• Be as specific as possible.\n"
    "• If the first results aren't right, use the navigation buttons to browse more."
)

SEARCH_PROMPT_TEXT = "🔍 Describe the photo you're looking for.\n\n"

NOTHING_FOUND_TEXT = (
    "😕 No matching photos found.\n\n"
    "Try describing:\n"
    "• objects or people\n"
    "• colors\n"
    "• places or events\n"
    "• visible text\n\n"
    "💡 The more specific your description, the better the results."
)

RESULTS_PER_PAGE = 3

# In-memory state for "Show more" pagination, keyed by a short random
# token embedded in each button's callback_data. This is intentionally
# simple (no persistence, no expiry) since chatmind runs as a single
# long-lived process for one chat; if the bot restarts, in-flight
# "Show more" buttons just stop working and people can run /find again.
_pending_searches: dict[str, dict] = {}


async def _react(message: Message, emoji: str) -> None:
    """Best-effort reaction on a message; failures here (e.g. missing
    permissions) shouldn't ever crash the handler that called this."""
    try:
        await message.bot.set_message_reaction(
            chat_id=message.chat.id,
            message_id=message.message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)],
        )
    except TelegramAPIError:
        logger.debug("Could not set reaction on message %s", message.message_id)


def _run_search(query: str, chat_id: int, kind: str, offset: int) -> list[dict]:
    """Run the given kind of search ("text match" or "visual
    similarity") at a specific offset, for pagination."""
    if kind == "text match":
        return search_by_ocr_text(
            query_text=query, chat_id=chat_id, limit=RESULTS_PER_PAGE, offset=offset
        )
    query_embedding = embed_text(query)
    return search_similar(
        query_embedding=query_embedding,
        chat_id=chat_id,
        limit=RESULTS_PER_PAGE,
        offset=offset,
    )


def _result_caption(item: dict, kind: str, index: int) -> str:
    score = item.get("similarity", item.get("rank"))
    lines = [f"#{index} \u2014 {kind}: {score:.2f}"]
    if item["caption"]:
        lines.append(item["caption"])
    return "\n".join(lines)


def _more_results_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="\U0001F504 Show more", callback_data=f"more:{token}")]
        ]
    )


def _welcome_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="\U0001F50D Search", callback_data="start_search"
                ),
                InlineKeyboardButton(
                    text="\u2139\uFE0F Help", callback_data="start_help"
                ),
            ]
        ]
    )


async def _perform_find(message: Message, query: str) -> None:
    """Shared search logic used by both /find and the "Search" button
    flow (reply to the "What are you looking for?" prompt)."""
    try:
        results = _run_search(query, message.chat.id, "text match", offset=0)
        kind = "text match"

        if not results:
            results = _run_search(
                query, message.chat.id, "visual similarity", offset=0
            )
            kind = "visual similarity"

    except Exception:
        logger.exception(
            "Search failed for query %r in chat %s", query, message.chat.id
        )
        await message.reply(
            "Something went wrong while searching. Please try again."
        )
        return

    if not results:
        await message.reply(NOTHING_FOUND_TEXT)
        return

    sent_message_ids = []
    for i, item in enumerate(results, start=1):
        try:
            sent = await message.answer_photo(
                item["file_id"], caption=_result_caption(item, kind, i)
            )
            sent_message_ids.append(sent.message_id)
        except TelegramAPIError:
            logger.warning(
                "Could not send photo file_id=%s (may be too old/invalid)",
                item["file_id"],
            )

    if len(results) == RESULTS_PER_PAGE and sent_message_ids:
        token = uuid.uuid4().hex[:12]
        _pending_searches[token] = {
            "query": query,
            "chat_id": message.chat.id,
            "kind": kind,
            "offset": len(results),
            "message_ids": sent_message_ids,
        }
        await message.answer(
            "None of these the right one?",
            reply_markup=_more_results_keyboard(token),
        )


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(WELCOME_TEXT, reply_markup=_welcome_keyboard())


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.reply(HELP_TEXT)


@router.callback_query(F.data == "start_help")
async def handle_start_help_button(callback: CallbackQuery) -> None:
    await callback.message.answer(HELP_TEXT)
    await callback.answer()


@router.callback_query(F.data == "start_search")
async def handle_start_search_button(callback: CallbackQuery) -> None:
    await callback.message.answer(
        SEARCH_PROMPT_TEXT,
        reply_markup=ForceReply(
            input_field_placeholder="e.g. red car, invoice, cat..."
        ),
    )
    await callback.answer()


@router.message(F.reply_to_message, F.text)
async def handle_search_prompt_reply(message: Message) -> None:
    """Treat a reply to the "What are you looking for?" prompt as a
    search query, so people can search via the Search button without
    needing to type /find manually."""
    replied_to = message.reply_to_message
    is_reply_to_bot = (
        replied_to.from_user and replied_to.from_user.is_bot
    )
    if not (is_reply_to_bot and replied_to.text == SEARCH_PROMPT_TEXT):
        return

    await _perform_find(message, message.text.strip())


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    """Index a photo sent to the chat."""
    try:
        photo = message.photo[-1]  # highest resolution version available
        file = await message.bot.get_file(photo.file_id)
        file_bytes = await message.bot.download_file(file.file_path)

        image = Image.open(io.BytesIO(file_bytes.read())).convert("RGB")
        embedding = embed_image(image)
        ocr_text = extract_text(image)

        item_id = save_item(
            chat_id=message.chat.id,
            message_id=message.message_id,
            file_id=photo.file_id,
            media_type="photo",
            embedding=embedding,
            sender_id=message.from_user.id if message.from_user else None,
            sender_name=(
                message.from_user.full_name if message.from_user else None
            ),
            caption=message.caption,
            ocr_text=ocr_text or None,
        )

        logger.info(
            "Indexed photo id=%s from chat=%s (ocr_text: %s chars)",
            item_id,
            message.chat.id,
            len(ocr_text),
        )
        await _react(message, "\U0001F44D")  # 👍

    except UnidentifiedImageError:
        logger.warning(
            "Could not read image for message %s in chat %s",
            message.message_id,
            message.chat.id,
        )
        await _react(message, "\u26A0")  # ⚠️

    except Exception:
        logger.exception(
            "Failed to index photo from message %s in chat %s",
            message.message_id,
            message.chat.id,
        )
        await _react(message, "\U0001F44E")  # 👎


@router.message(Command("find"))
async def handle_find(message: Message) -> None:
    """Search for photos matching the given text description and
    send back the first batch of candidates, with a "Show more"
    button that swaps them in place for the next batch."""
    query = (message.text or "").removeprefix("/find").strip()

    if not query:
        await message.reply("Usage: /find <description>")
        return

    await _perform_find(message, query)


@router.callback_query(F.data.startswith("more:"))
async def handle_more_results(callback: CallbackQuery) -> None:
    """Fetch the next batch of results and swap them into the
    existing photo messages in place, instead of sending new ones."""
    token = callback.data.removeprefix("more:")
    state = _pending_searches.get(token)

    if state is None:
        await callback.answer(
            "This search has expired, try /find again.", show_alert=True
        )
        return

    try:
        results = _run_search(
            state["query"], state["chat_id"], state["kind"], state["offset"]
        )
    except Exception:
        logger.exception(
            "Pagination search failed for query %r in chat %s",
            state["query"],
            state["chat_id"],
        )
        await callback.answer("Something went wrong, please try again.")
        return

    if not results:
        await callback.message.edit_text("No more matches found.")
        await callback.answer()
        _pending_searches.pop(token, None)
        return

    message_ids = state["message_ids"]
    bot = callback.bot
    chat_id = state["chat_id"]
    start_index = state["offset"] + 1

    for i, message_id in enumerate(message_ids):
        if i < len(results):
            item = results[i]
            caption = _result_caption(item, state["kind"], start_index + i)
            try:
                await bot.edit_message_media(
                    chat_id=chat_id,
                    message_id=message_id,
                    media=InputMediaPhoto(media=item["file_id"], caption=caption),
                )
            except TelegramAPIError:
                logger.warning(
                    "Could not edit photo message %s with file_id=%s",
                    message_id,
                    item["file_id"],
                )
        else:
            # Fewer results this time than photo messages we have to
            # fill, so remove the leftover ones.
            try:
                await bot.delete_message(chat_id=chat_id, message_id=message_id)
            except TelegramAPIError:
                pass

    state["message_ids"] = message_ids[: len(results)]
    state["offset"] += len(results)
    await callback.answer()

    if len(results) < RESULTS_PER_PAGE:
        # Fewer than a full page came back, so there's nothing left.
        await callback.message.edit_text("No more matches found.")
        _pending_searches.pop(token, None)