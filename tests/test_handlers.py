"""
Tests for bot/handlers.py.

Following the same philosophy as the rest of the suite: real
chatmind_test database, real CLIP embeddings / OCR where relevant
(via processing/embeddings.py, processing/ocr.py, storage/db.py).
The only thing mocked is the Telegram boundary itself (Message,
CallbackQuery, Bot) -- there's no way to test against a live bot, and
mocking it is the standard approach for aiogram handler tests.

asyncio_mode = auto (see pytest.ini) means every `async def test_...`
here just works, no @pytest.mark.asyncio needed.
"""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramAPIError

from bot import handlers
from processing.embeddings import embed_image
from storage.db import get_item_by_id, save_item, toggle_favorite

CHAT_ID = 555555


@pytest.fixture(autouse=True)
def clean_pending_searches():
    """handlers._pending_searches is in-memory module state, not
    covered by the clean_media_items DB-truncation fixture in
    conftest.py, so reset it around every test too."""
    handlers._pending_searches.clear()
    yield
    handlers._pending_searches.clear()


@pytest.fixture
def fake_bot():
    """A stand-in aiogram Bot: only the async methods handlers.py
    actually calls are mocked, so assertions can check exactly how
    they were invoked without a real Telegram connection."""
    bot = MagicMock()
    bot.set_message_reaction = AsyncMock()
    bot.get_file = AsyncMock()
    bot.download_file = AsyncMock()
    bot.edit_message_media = AsyncMock()
    bot.edit_message_reply_markup = AsyncMock()
    bot.edit_message_caption = AsyncMock()
    return bot


@pytest.fixture
def make_message(fake_bot):
    """Factory for a fake aiogram Message covering just the surface
    handlers.py touches: chat/from_user attributes, plus AsyncMock
    stand-ins for reply/answer/answer_photo so we can assert on what
    was sent without a live Telegram connection."""

    def _make(
        text=None,
        photo=None,
        reply_to_message=None,
        chat_id=CHAT_ID,
        message_id=1,
        user_id=1001,
        user_name="Test User",
        sent_photo_message_id=999,
    ):
        message = MagicMock()
        message.chat = SimpleNamespace(id=chat_id)
        message.message_id = message_id
        message.text = text
        message.caption = None
        message.photo = photo
        message.reply_to_message = reply_to_message
        message.from_user = SimpleNamespace(
            id=user_id, full_name=user_name, is_bot=False
        )
        message.bot = fake_bot
        message.reply = AsyncMock()
        message.answer = AsyncMock()
        message.answer_photo = AsyncMock(
            return_value=SimpleNamespace(message_id=sent_photo_message_id)
        )
        return message

    return _make


@pytest.fixture
def make_callback(fake_bot):
    """Factory for a fake aiogram CallbackQuery: .data, a minimal
    .message (with chat/message_id/answer), .bot, .from_user, and
    .answer()."""

    def _make(data, chat_id=CHAT_ID, message_id=999, user_id=1001):
        callback = MagicMock()
        callback.data = data
        callback.bot = fake_bot
        callback.from_user = SimpleNamespace(id=user_id)
        callback.message = MagicMock()
        callback.message.chat = SimpleNamespace(id=chat_id)
        callback.message.message_id = message_id
        callback.message.answer = AsyncMock()
        callback.answer = AsyncMock()
        return callback

    return _make


def _image_bytes(image, fmt="PNG"):
    """A file-like object with the image's raw bytes, matching what
    bot.download_file(...) returns in real aiogram usage."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf


# --- /start, /help ---------------------------------------------------


async def test_handle_start_sends_welcome_with_keyboard(make_message):
    message = make_message()
    await handlers.handle_start(message)

    message.answer.assert_awaited_once()
    args, kwargs = message.answer.call_args
    assert args[0] == handlers.WELCOME_TEXT
    assert kwargs["reply_markup"] is not None


async def test_handle_help_replies_with_help_text(make_message):
    message = make_message()
    await handlers.handle_help(message)

    message.reply.assert_awaited_once_with(handlers.HELP_TEXT)


async def test_start_help_button_sends_help_and_answers_callback(make_callback):
    callback = make_callback("start_help")
    await handlers.handle_start_help_button(callback)

    callback.message.answer.assert_awaited_once_with(handlers.HELP_TEXT)
    callback.answer.assert_awaited_once()


async def test_start_search_button_sends_prompt_with_force_reply(make_callback):
    callback = make_callback("start_search")
    await handlers.handle_start_search_button(callback)

    callback.message.answer.assert_awaited_once()
    args, kwargs = callback.message.answer.call_args
    assert args[0] == handlers.SEARCH_PROMPT_TEXT
    assert kwargs["reply_markup"] is not None
    callback.answer.assert_awaited_once()


# --- handle_photo (indexing) ------------------------------------------


async def test_handle_photo_indexes_and_reacts_thumbs_up(
    make_message, fake_bot, red_image
):
    fake_bot.get_file.return_value = SimpleNamespace(file_path="fake/path.jpg")
    fake_bot.download_file.return_value = _image_bytes(red_image)

    message = make_message(photo=[SimpleNamespace(file_id="photo123")])
    await handlers.handle_photo(message)

    fake_bot.set_message_reaction.assert_awaited_once()
    _, kwargs = fake_bot.set_message_reaction.call_args
    assert kwargs["reaction"][0].emoji == "\U0001F44D"  # 👍

    # confirm the item was actually written to the (real, test) database
    from storage.db import search_similar

    results = search_similar(
        query_embedding=embed_image(red_image), chat_id=CHAT_ID, limit=1
    )
    assert len(results) == 1
    assert results[0]["file_id"] == "photo123"
    assert results[0]["sender_name"] == "Test User"


async def test_handle_photo_reacts_warning_for_unreadable_image(
    make_message, fake_bot
):
    fake_bot.get_file.return_value = SimpleNamespace(file_path="fake/path.jpg")
    fake_bot.download_file.return_value = io.BytesIO(b"not an image")

    message = make_message(photo=[SimpleNamespace(file_id="bad_photo")])
    await handlers.handle_photo(message)

    fake_bot.set_message_reaction.assert_awaited_once()
    _, kwargs = fake_bot.set_message_reaction.call_args
    assert kwargs["reaction"][0].emoji == "\u26A0"  # ⚠


async def test_handle_photo_reacts_thumbs_down_on_unexpected_error(
    make_message, fake_bot, red_image, monkeypatch
):
    fake_bot.get_file.return_value = SimpleNamespace(file_path="fake/path.jpg")
    fake_bot.download_file.return_value = _image_bytes(red_image)

    def _boom(*args, **kwargs):
        raise RuntimeError("db is down")

    monkeypatch.setattr(handlers, "save_item", _boom)

    message = make_message(photo=[SimpleNamespace(file_id="photo123")])
    await handlers.handle_photo(message)

    fake_bot.set_message_reaction.assert_awaited_once()
    _, kwargs = fake_bot.set_message_reaction.call_args
    assert kwargs["reaction"][0].emoji == "\U0001F44E"  # 👎


async def test_react_swallows_telegram_api_errors(make_message):
    """_react is best-effort: a TelegramAPIError while setting the
    reaction (e.g. missing permissions) must not propagate and crash
    the calling handler."""
    message = make_message()
    message.bot.set_message_reaction = AsyncMock(
        side_effect=TelegramAPIError(method=MagicMock(), message="boom")
    )

    await handlers._react(message, "\U0001F44D")  # should not raise


# --- /find --------------------------------------------------------------


async def test_find_returns_first_result_by_text_match(make_message, sample_embedding):
    save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="ocr_item",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="invoice number 4471",
        caption="an invoice",
    )

    message = make_message(text="/find invoice")
    await handlers.handle_find(message)

    message.answer_photo.assert_awaited_once()
    args, kwargs = message.answer_photo.call_args
    assert args[0] == "ocr_item"
    assert "text match" in kwargs["caption"]
    assert "1/1" in kwargs["caption"]
    assert kwargs["reply_markup"] is not None


async def test_find_falls_back_to_visual_similarity_when_no_text_match(
    make_message, red_image
):
    save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="red_item",
        media_type="photo",
        embedding=embed_image(red_image),
        caption=None,
        ocr_text=None,
    )

    message = make_message(text="/find the color red")
    await handlers.handle_find(message)

    message.answer_photo.assert_awaited_once()
    args, kwargs = message.answer_photo.call_args
    assert args[0] == "red_item"
    assert "visual similarity" in kwargs["caption"]


async def test_find_replies_nothing_found_when_no_results(make_message):
    message = make_message(text="/find something that definitely does not exist")
    await handlers.handle_find(message)

    message.reply.assert_awaited_once_with(handlers.NOTHING_FOUND_TEXT)
    message.answer_photo.assert_not_called()


async def test_find_without_query_shows_usage(make_message):
    message = make_message(text="/find")
    await handlers.handle_find(message)

    message.reply.assert_awaited_once_with("Usage: /find <description>")


# --- reply-to-search-prompt flow ----------------------------------------


async def test_search_prompt_reply_triggers_find(make_message, sample_embedding):
    save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="ocr_item",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="red car",
    )
    prompt = SimpleNamespace(
        text=handlers.SEARCH_PROMPT_TEXT, from_user=SimpleNamespace(is_bot=True)
    )
    message = make_message(text="red car", reply_to_message=prompt)
    await handlers.handle_search_prompt_reply(message)

    message.answer_photo.assert_awaited_once()


async def test_search_prompt_reply_ignores_replies_to_non_bot_messages(make_message):
    prompt = SimpleNamespace(
        text=handlers.SEARCH_PROMPT_TEXT, from_user=SimpleNamespace(is_bot=False)
    )
    message = make_message(text="red car", reply_to_message=prompt)
    await handlers.handle_search_prompt_reply(message)

    message.answer_photo.assert_not_called()
    message.reply.assert_not_called()


async def test_search_prompt_reply_tolerates_trailing_whitespace(
    make_message, sample_embedding
):
    """Regression test: Telegram trims trailing whitespace from sent
    messages, so replied_to.text may not exactly equal the constant
    with its own formatting -- the comparison must tolerate that."""
    save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="ocr_item",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="red car",
    )
    prompt = SimpleNamespace(
        text=handlers.SEARCH_PROMPT_TEXT + "\n\n",
        from_user=SimpleNamespace(is_bot=True),
    )
    message = make_message(text="red car", reply_to_message=prompt)
    await handlers.handle_search_prompt_reply(message)

    message.answer_photo.assert_awaited_once()


# --- gallery navigation ---------------------------------------------------


async def test_nav_moves_to_next_result(make_message, make_callback, sample_embedding):
    save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="item_1",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="shared text",
    )
    save_item(
        chat_id=CHAT_ID,
        message_id=2,
        file_id="item_2",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="shared text",
    )

    message = make_message(text="/find shared")
    await handlers.handle_find(message)
    token = next(iter(handlers._pending_searches))

    callback = make_callback(f"nav:{token}:1")
    await handlers.handle_nav(callback)

    callback.bot.edit_message_media.assert_awaited_once()
    callback.answer.assert_awaited_once()
    assert handlers._pending_searches[token]["index"] == 1


async def test_nav_does_nothing_past_the_last_result(
    make_message, make_callback, sample_embedding
):
    save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="item_1",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="shared text",
    )

    message = make_message(text="/find shared")
    await handlers.handle_find(message)
    token = next(iter(handlers._pending_searches))

    callback = make_callback(f"nav:{token}:1")
    await handlers.handle_nav(callback)

    callback.bot.edit_message_media.assert_not_called()
    callback.answer.assert_awaited_once()


async def test_nav_shows_alert_for_expired_token(make_callback):
    callback = make_callback("nav:doesnotexist:1")
    await handlers.handle_nav(callback)

    callback.answer.assert_awaited_once()
    _, kwargs = callback.answer.call_args
    assert kwargs.get("show_alert") is True


async def test_noop_just_answers_callback(make_callback):
    callback = make_callback("noop:sometoken")
    await handlers.handle_noop(callback)

    callback.answer.assert_awaited_once()


# --- favorites --------------------------------------------------------


async def test_favorite_toggle_marks_item_as_favorite(
    make_message, make_callback, sample_embedding
):
    item_id = save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="item_1",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="find me",
    )
    message = make_message(text="/find find me")
    await handlers.handle_find(message)
    token = next(iter(handlers._pending_searches))

    callback = make_callback(f"fav:{token}:{item_id}")
    await handlers.handle_favorite(callback)

    assert get_item_by_id(item_id)["is_favorite"] is True
    callback.bot.edit_message_reply_markup.assert_awaited_once()
    callback.answer.assert_awaited_once()


async def test_favorite_toggle_twice_unmarks_it(make_callback, sample_embedding):
    item_id = save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="item_1",
        media_type="photo",
        embedding=sample_embedding,
    )
    callback = make_callback(f"fav:sometoken:{item_id}")

    await handlers.handle_favorite(callback)
    await handlers.handle_favorite(callback)

    assert get_item_by_id(item_id)["is_favorite"] is False


async def test_favorites_command_lists_favorited_items(make_message, sample_embedding):
    item_id = save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="fav_item",
        media_type="photo",
        embedding=sample_embedding,
    )
    toggle_favorite(item_id)

    message = make_message(text="/favorites")
    await handlers.handle_favorites(message)

    message.answer_photo.assert_awaited_once()
    args, _ = message.answer_photo.call_args
    assert args[0] == "fav_item"


async def test_favorites_command_shows_empty_message_when_none(make_message):
    message = make_message(text="/favorites")
    await handlers.handle_favorites(message)

    message.reply.assert_awaited_once_with(handlers.FAVORITES_EMPTY_TEXT)


# --- /stats ---------------------------------------------------------------


async def test_stats_shows_empty_message_for_chat_with_nothing_indexed(make_message):
    message = make_message(text="/stats")
    await handlers.handle_stats(message)

    message.reply.assert_awaited_once_with(handlers.STATS_EMPTY_TEXT)


async def test_stats_summarizes_indexed_photos(make_message, sample_embedding):
    save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="a",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="some text",
        sender_name="Dima",
    )
    save_item(
        chat_id=CHAT_ID,
        message_id=2,
        file_id="b",
        media_type="photo",
        embedding=sample_embedding,
        sender_name="Dima",
    )
    save_item(
        chat_id=CHAT_ID,
        message_id=3,
        file_id="c",
        media_type="photo",
        embedding=sample_embedding,
        sender_name="Olena",
    )

    message = make_message(text="/stats")
    await handlers.handle_stats(message)

    message.reply.assert_awaited_once()
    (text,), _ = message.reply.call_args
    assert "Photos indexed: 3" in text
    assert "With readable text (OCR): 1" in text
    assert "Dima" in text
    assert "Olena" in text


# --- /delete (via the 🗑️ gallery button) ---------------------------------


async def test_delete_removes_item_when_uploader_taps_it(
    make_message, make_callback, sample_embedding
):
    item_id = save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="item_1",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="find me",
        sender_id=1001,
    )
    message = make_message(text="/find find me", user_id=1001)
    await handlers.handle_find(message)
    token = next(iter(handlers._pending_searches))

    # Same user_id (1001) as the uploader -- default for both fixtures.
    callback = make_callback(f"del:{token}:{item_id}", user_id=1001)
    await handlers.handle_delete(callback)

    assert get_item_by_id(item_id) is None
    callback.bot.edit_message_caption.assert_awaited_once()
    callback.answer.assert_awaited_once()
    _, kwargs = callback.answer.call_args
    assert kwargs.get("show_alert") is not True


async def test_delete_refuses_when_a_different_user_taps_it(
    make_message, make_callback, sample_embedding
):
    item_id = save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="item_1",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="find me",
        sender_id=1001,
    )
    message = make_message(text="/find find me", user_id=1001)
    await handlers.handle_find(message)
    token = next(iter(handlers._pending_searches))

    # Different user_id than the uploader (1001).
    callback = make_callback(f"del:{token}:{item_id}", user_id=9999)
    await handlers.handle_delete(callback)

    # Item must survive -- only the uploader may delete it.
    assert get_item_by_id(item_id) is not None
    callback.bot.edit_message_caption.assert_not_called()
    callback.answer.assert_awaited_once()
    _, kwargs = callback.answer.call_args
    assert kwargs.get("show_alert") is True


async def test_delete_shows_alert_for_already_removed_item(make_callback):
    callback = make_callback("del:sometoken:999999")  # no such item id
    await handlers.handle_delete(callback)

    callback.answer.assert_awaited_once()
    _, kwargs = callback.answer.call_args
    assert kwargs.get("show_alert") is True


async def test_deleted_item_no_longer_shows_up_in_find(
    make_message, make_callback, sample_embedding
):
    item_id = save_item(
        chat_id=CHAT_ID,
        message_id=1,
        file_id="item_1",
        media_type="photo",
        embedding=sample_embedding,
        ocr_text="unique phrase",
        sender_id=1001,
    )
    message = make_message(text="/find unique phrase", user_id=1001)
    await handlers.handle_find(message)
    token = next(iter(handlers._pending_searches))

    callback = make_callback(f"del:{token}:{item_id}", user_id=1001)
    await handlers.handle_delete(callback)

    second_search = make_message(text="/find unique phrase")
    await handlers.handle_find(second_search)

    second_search.reply.assert_awaited_once_with(handlers.NOTHING_FOUND_TEXT)
    second_search.answer_photo.assert_not_called()