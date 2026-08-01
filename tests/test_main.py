"""
Tests for bot/main.py.

main() ends in dp.start_polling(bot), a real infinite polling loop --
there's nothing meaningful to unit-test there directly. Instead we
monkeypatch Dispatcher.include_router / Dispatcher.start_polling (the
aiogram class methods, not an instance -- main() creates its own
Dispatcher() internally, so there's no instance to patch beforehand)
so main() runs to completion without ever touching the network or
blocking forever, and we can assert it wired things up correctly.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Dispatcher

from bot import main as main_module
from bot.handlers import router as handlers_router

FAKE_TOKEN = "123456789:AAFakeTokenForTestsXXXXXXXXXXXXXXX"


async def test_main_raises_when_token_missing(monkeypatch):
    monkeypatch.setattr(main_module, "TELEGRAM_BOT_TOKEN", "")

    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        await main_module.main()


async def test_main_registers_the_handlers_router(monkeypatch):
    monkeypatch.setattr(main_module, "TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(Dispatcher, "start_polling", AsyncMock())
    include_router_mock = MagicMock()
    monkeypatch.setattr(Dispatcher, "include_router", include_router_mock)

    await main_module.main()

    include_router_mock.assert_called_once_with(handlers_router)


async def test_main_starts_polling_with_a_bot_using_the_configured_token(
    monkeypatch,
):
    monkeypatch.setattr(main_module, "TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(Dispatcher, "include_router", MagicMock())
    start_polling_mock = AsyncMock()
    monkeypatch.setattr(Dispatcher, "start_polling", start_polling_mock)

    await main_module.main()

    start_polling_mock.assert_awaited_once()
    (bot_arg,), _ = start_polling_mock.call_args
    assert bot_arg.token == FAKE_TOKEN