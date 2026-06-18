"""
Telegram-бот поиска фильмов по описанию сюжета.
Работает через polling.
"""

import logging
import os
import asyncio

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
load_dotenv()

from movie_finder import MovieFinder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Настройки из переменных окружения ---
BOT_TOKEN = os.environ["BOT_TOKEN"]
TOP_K = int(os.environ.get("TOP_K", "5"))

router = Router()
finder = MovieFinder()


@router.message(CommandStart())
@router.message(Command("help"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! 🎬 Я ищу фильмы по описанию сюжета или сцены.\n\n"
        "Просто напиши, о чём фильм — на русском или английском, неважно.\n\n"
        "Например:\n"
        "<i>«человек теряет память и пытается найти убийцу своей жены»</i>"
    )


@router.message(F.text)
async def handle_query(message: Message):
    query = message.text.strip()

    if len(query) < 3:
        await message.answer("Опиши сюжет немного подробнее, пожалуйста 🙂")
        return

    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        results = finder.search(query, top_k=TOP_K)
    except Exception as e:
        logger.exception("Ошибка при поиске")
        await message.answer("Что-то пошло не так при поиске 😕 Попробуй ещё раз.")
        return

    if not results:
        await message.answer("Ничего не нашлось 😕 Попробуй описать иначе.")
        return

    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"<b>{i}. {r['title']}</b> (совпадение {r['score']:.0%})\n"
            f"<i>{r['genres_ru']}</i>\n"
            f"{r['overview_ru']}"
        )
    await message.answer("\n\n".join(blocks))


async def main():
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Бот запущен (polling)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())