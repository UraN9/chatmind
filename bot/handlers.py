"""
Telegram handlers for chatmind.

Responsibilities:
- /start and /help: explain what the bot does.
- When a photo is sent to the chat, embed it with CLIP (and run OCR
  on it) and save it to the database (indexing).
- When the /find command is used, search for the closest matching
  photo and send it back as a single-photo gallery with
  \u2b05\ufe0f / \u27a1\ufe0f navigation buttons and a "3/14" counter, so browsing
  candidates edits one message in place instead of cluttering the
  chat with a batch of photos.
"""

import io
import logging
import uuid
from typing import Optional

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
from storage.db import (
    count_by_ocr_text,
    count_favorites,
    count_similar,
    delete_item,
    get_chat_stats,
    get_item_by_id,
    list_favorites,
    save_item,
    search_by_ocr_text,
    search_similar,
    toggle_favorite,
)

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
    "❤️ Save favorites\n"
    "Tap the heart on any result to save it, then browse them with /favorites.\n\n"
    "🗑️ Remove a photo\n"
    "Tap Delete on a result to remove it from the index (only the person who sent it can).\n\n"
    "📊 Check your stats\n"
    "Use /stats to see how many photos are indexed in this chat.\n\n"
    "💡 Tips\n"
    "• English descriptions usually give the best results.\n"
    "• Be as specific as possible.\n"
    "• If the first results aren't right, use the navigation buttons to browse more."
)

SEARCH_PROMPT_TEXT = "🔍 Describe the photo you're looking for."

NOTHING_FOUND_TEXT = (
    "😕 No matching photos found.\n\n"
    "Try describing:\n"
    "• objects or people\n"
    "• colors\n"
    "• places or events\n"
    "• visible text\n\n"
    "💡 The more specific your description, the better the results."
)

STATS_EMPTY_TEXT = (
    "📊 No photos indexed in this chat yet.\n\n"
    "Send me a few photos and check back with /stats!"
)

FAVORITES_EMPTY_TEXT = (
    "🤍 No favorites yet.\n\n"
    "Tap ❤️ on a photo from /find to add it here."
)

# In-memory state for gallery navigation, keyed by a short random
# token embedded in each nav button's callback_data. This is
# intentionally simple (no persistence, no expiry) since chatmind runs
# as a single long-lived process for one chat; if the bot restarts,
# in-flight galleries just stop working and people can run /find again.
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


def _fetch_at(query: str, chat_id: int, kind: str, index: int) -> list[dict]:
    """Fetch the single result at `index` for the given kind of search
    ("text match", "visual similarity", or "favorites"), for gallery
    navigation."""
    if kind == "text match":
        return search_by_ocr_text(query_text=query, chat_id=chat_id, limit=1, offset=index)
    if kind == "favorites":
        return list_favorites(chat_id=chat_id, limit=1, offset=index)
    query_embedding = embed_text(query)
    return search_similar(
        query_embedding=query_embedding, chat_id=chat_id, limit=1, offset=index
    )


def _count_results(query: str, chat_id: int, kind: str) -> int:
    """Count the total number of results for the given kind of
    search, used for the "3/14" gallery counter. Only called for
    "text match" / "visual similarity" -- /favorites counts its own
    total directly via count_favorites(), since it doesn't need the
    text-match-then-fallback logic _perform_find does."""
    if kind == "text match":
        return count_by_ocr_text(query_text=query, chat_id=chat_id)
    query_embedding = embed_text(query)
    return count_similar(query_embedding=query_embedding, chat_id=chat_id)


def _result_caption(item: dict, kind: str, index: int, total: int) -> str:
    if kind == "favorites":
        header = f"{index + 1}/{total} \u2014 \u2764\ufe0f Favorite"
    else:
        score = item.get("similarity", item.get("rank"))
        header = f"{index + 1}/{total} \u2014 {kind}: {score:.2f}"
    lines = [header]
    if item["caption"]:
        lines.append(item["caption"])
    return "\n".join(lines)


def _gallery_keyboard(
    token: str, index: int, total: int, item_id: int, is_favorite: bool
) -> InlineKeyboardMarkup:
    """Build the gallery keyboard: a \u2b05\ufe0f N/total \u27a1\ufe0f nav row (arrows
    omitted at either end, since Telegram has no disabled-button
    concept), a row to toggle favorite status, and a delete row.
    Delete is shown to everyone (Telegram can't render different
    keyboards per viewer on one message) -- the actual permission
    check (only the original uploader can delete) happens when the
    button is tapped, in handle_delete."""
    nav_row = []
    if index > 0:
        nav_row.append(InlineKeyboardButton(text="\u2b05\ufe0f", callback_data=f"nav:{token}:-1"))
    nav_row.append(
        InlineKeyboardButton(text=f"{index + 1}/{total}", callback_data=f"noop:{token}")
    )
    if index < total - 1:
        nav_row.append(InlineKeyboardButton(text="\u27a1\ufe0f", callback_data=f"nav:{token}:1"))

    fav_text = "\U0001F494 Remove favorite" if is_favorite else "\u2764\uFE0F Add favorite"
    fav_row = [InlineKeyboardButton(text=fav_text, callback_data=f"fav:{token}:{item_id}")]

    del_row = [
        InlineKeyboardButton(text="\U0001F5D1\uFE0F Delete", callback_data=f"del:{token}:{item_id}")
    ]

    return InlineKeyboardMarkup(inline_keyboard=[nav_row, fav_row, del_row])


def _format_stats(stats: dict) -> str:
    total = stats["total"]
    with_ocr = stats["with_ocr"]
    ocr_pct = round(100 * with_ocr / total) if total else 0

    lines = [
        "📊 Chat stats",
        "",
        f"📸 Photos indexed: {total}",
        f"🔤 With readable text (OCR): {with_ocr} ({ocr_pct}%)",
        f"📅 First photo: {stats['first_photo_at']:%b %d, %Y}",
        f"📅 Latest photo: {stats['last_photo_at']:%b %d, %Y}",
    ]

    if stats["top_senders"]:
        lines.append("")
        lines.append("🏆 Top senders:")
        medals = ["🥇", "🥈", "🥉"]
        for i, sender in enumerate(stats["top_senders"]):
            medal = medals[i] if i < len(medals) else "•"
            lines.append(f"{medal} {sender['sender_name']} — {sender['count']}")

    return "\n".join(lines)


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


async def _send_gallery(
    message: Message, kind: str, query: Optional[str], results: list[dict], total: int
) -> None:
    """Send the first result as a single-photo gallery message with
    nav + favorite-toggle buttons, and register it in
    _pending_searches so later button taps know what to do. Shared by
    /find (and its Search-button reply flow) and /favorites."""
    item = results[0]
    token = uuid.uuid4().hex[:12]

    try:
        sent = await message.answer_photo(
            item["file_id"],
            caption=_result_caption(item, kind, index=0, total=total),
            reply_markup=_gallery_keyboard(
                token, index=0, total=total, item_id=item["id"], is_favorite=item["is_favorite"]
            ),
        )
    except TelegramAPIError:
        logger.warning(
            "Could not send photo file_id=%s (may be too old/invalid)",
            item["file_id"],
        )
        await message.reply(NOTHING_FOUND_TEXT)
        return

    _pending_searches[token] = {
        "query": query,
        "chat_id": message.chat.id,
        "kind": kind,
        "index": 0,
        "total": total,
        "message_id": sent.message_id,
    }


async def _perform_find(message: Message, query: str) -> None:
    """Shared search logic used by both /find and the "Search" button
    flow (reply to the "What are you looking for?" prompt)."""
    try:
        kind = "text match"
        total = _count_results(query, message.chat.id, kind)

        if total == 0:
            kind = "visual similarity"
            total = _count_results(query, message.chat.id, kind)

        results = _fetch_at(query, message.chat.id, kind, index=0) if total else []

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

    await _send_gallery(message, kind, query, results, total)


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
    # Compare with .strip() on both sides: Telegram trims leading/
    # trailing whitespace from message text, so a naive `==` against
    # the SEARCH_PROMPT_TEXT constant can silently never match.
    if not (
        is_reply_to_bot
        and (replied_to.text or "").strip() == SEARCH_PROMPT_TEXT.strip()
    ):
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


@router.message(Command("favorites"))
async def handle_favorites(message: Message) -> None:
    """Browse favorited photos as the same one-at-a-time gallery used
    by /find, most recently indexed first."""
    try:
        total = count_favorites(message.chat.id)
        results = list_favorites(message.chat.id, limit=1, offset=0) if total else []
    except Exception:
        logger.exception("Failed to fetch favorites for chat %s", message.chat.id)
        await message.reply(
            "Something went wrong while fetching favorites. Please try again."
        )
        return

    if not results:
        await message.reply(FAVORITES_EMPTY_TEXT)
        return

    await _send_gallery(message, "favorites", None, results, total)


@router.message(Command("stats"))
async def handle_stats(message: Message) -> None:
    """Show aggregate stats for this chat: how many photos are
    indexed, how many have OCR text, the date range, and (if
    available) who's sent the most."""
    try:
        stats = get_chat_stats(message.chat.id)
    except Exception:
        logger.exception("Failed to fetch stats for chat %s", message.chat.id)
        await message.reply("Something went wrong while fetching stats. Please try again.")
        return

    if stats["total"] == 0:
        await message.reply(STATS_EMPTY_TEXT)
        return

    await message.reply(_format_stats(stats))


@router.callback_query(F.data.startswith("fav:"))
async def handle_favorite(callback: CallbackQuery) -> None:
    """Toggle favorite status on the currently shown item and refresh
    just the keyboard \u2014 the photo and caption stay the same. Works
    even if the pending-search state has expired (e.g. bot restart),
    since item_id is embedded in the callback_data itself; it just
    falls back to a single-item (1/1) keyboard in that case."""
    _, token, item_id_str = callback.data.split(":", maxsplit=2)
    item_id = int(item_id_str)

    try:
        is_favorite = toggle_favorite(item_id)
    except Exception:
        logger.exception("Failed to toggle favorite for item %s", item_id)
        await callback.answer("Something went wrong, please try again.")
        return

    state = _pending_searches.get(token)
    index = state["index"] if state else 0
    total = state["total"] if state else 1

    try:
        await callback.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=callback.message.message_id,
            reply_markup=_gallery_keyboard(token, index, total, item_id, is_favorite),
        )
    except TelegramAPIError:
        logger.warning(
            "Could not update favorite keyboard for message %s",
            callback.message.message_id,
        )

    await callback.answer(
        "Added to favorites \u2764\uFE0F" if is_favorite else "Removed from favorites"
    )


@router.callback_query(F.data.startswith("del:"))
async def handle_delete(callback: CallbackQuery) -> None:
    """Remove an item from the index -- only the person who originally
    uploaded it is allowed to. This only unindexes it (so it stops
    showing up in /find and /favorites); the actual Telegram message
    with the photo is untouched, since the bot has no business (or
    permission) deleting other people's messages."""
    _, token, item_id_str = callback.data.split(":", maxsplit=2)
    item_id = int(item_id_str)

    item = get_item_by_id(item_id)
    if item is None:
        await callback.answer("Already removed.", show_alert=True)
        return

    if item["sender_id"] != callback.from_user.id:
        await callback.answer(
            "Only the person who uploaded this photo can delete it.",
            show_alert=True,
        )
        return

    try:
        deleted = delete_item(item_id)
    except Exception:
        logger.exception("Failed to delete item %s", item_id)
        await callback.answer("Something went wrong, please try again.")
        return

    if not deleted:
        await callback.answer("Already removed.", show_alert=True)
        return

    try:
        await callback.bot.edit_message_caption(
            chat_id=callback.message.chat.id,
            message_id=callback.message.message_id,
            caption="\U0001F5D1\uFE0F Removed from the index \u2014 it won't show up in "
            "/find or /favorites anymore.",
            reply_markup=None,
        )
    except TelegramAPIError:
        logger.warning(
            "Could not update message %s after deleting item %s",
            callback.message.message_id,
            item_id,
        )

    _pending_searches.pop(token, None)
    await callback.answer("Removed.")


@router.callback_query(F.data.startswith("noop:"))
async def handle_noop(callback: CallbackQuery) -> None:
    """The middle "3/14" button is just a counter, not an action."""
    await callback.answer()


@router.callback_query(F.data.startswith("nav:"))
async def handle_nav(callback: CallbackQuery) -> None:
    """Step the gallery forward or backward one result, editing the
    existing photo message in place instead of sending a new one."""
    _, token, delta_str = callback.data.split(":", maxsplit=2)
    state = _pending_searches.get(token)

    if state is None:
        await callback.answer(
            "This search has expired, try /find again.", show_alert=True
        )
        return

    new_index = state["index"] + int(delta_str)
    if not (0 <= new_index < state["total"]):
        await callback.answer()
        return

    try:
        results = _fetch_at(state["query"], state["chat_id"], state["kind"], new_index)
    except Exception:
        logger.exception(
            "Gallery navigation search failed for query %r in chat %s",
            state["query"],
            state["chat_id"],
        )
        await callback.answer("Something went wrong, please try again.")
        return

    if not results:
        # The underlying data changed since we counted (e.g. an item
        # was deleted); just stop at the last position we know is good.
        await callback.answer("No more matches found.", show_alert=True)
        state["total"] = max(state["index"] + 1, 1)
        return

    item = results[0]
    caption = _result_caption(item, state["kind"], index=new_index, total=state["total"])

    try:
        await callback.bot.edit_message_media(
            chat_id=state["chat_id"],
            message_id=state["message_id"],
            media=InputMediaPhoto(media=item["file_id"], caption=caption),
            reply_markup=_gallery_keyboard(
                token,
                new_index,
                state["total"],
                item_id=item["id"],
                is_favorite=item["is_favorite"],
            ),
        )
    except TelegramAPIError:
        logger.warning(
            "Could not edit photo message %s with file_id=%s",
            state["message_id"],
            item["file_id"],
        )
        await callback.answer("Something went wrong, please try again.")
        return

    state["index"] = new_index
    await callback.answer()