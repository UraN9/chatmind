"""
Entry point for the chatmind Telegram bot.

Run with:
    python -m bot.main
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher

from config import TELEGRAM_BOT_TOKEN
from bot.handlers import router


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Add it to your .env file."
        )

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    logging.info("Starting chatmind bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())