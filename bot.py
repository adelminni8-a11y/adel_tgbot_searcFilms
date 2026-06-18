"""
Telegram-бот поиска фильмов по описанию сюжета.
Работает через webhook (не polling) — для деплоя на VPS за nginx с SSL.
"""

import logging
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from dotenv import load_dotenv
load_dotenv()

from movie_finder import MovieFinder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Настройки из переменных окружения ---
BOT_TOKEN = os.environ["BOT_TOKEN"]
WEBHOOK_HOST = os.environ["WEBHOOK_HOST"]            # например: https://yourdomain.com
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/webhook")
WEBHOOK_URL = WEBHOOK_HOST.rstrip("/") + WEBHOOK_PATH

WEBAPP_HOST = os.environ.get("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.environ.get("WEBAPP_PORT", "8080"))

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


async def health(request: web.Request) -> web.Response:
    return web.Response(text="OK")


async def on_startup(bot: Bot):
    await bot.set_webhook(WEBHOOK_URL)
    logger.info(f"Webhook установлен: {WEBHOOK_URL}")


async def on_shutdown(bot: Bot):
    await bot.delete_webhook()
    logger.info("Webhook удалён")


def main():
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    app = web.Application()
    app.router.add_get("/", health)  # для проверки, что контейнер живой

    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)

    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)


if __name__ == "__main__":
    main()
