from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message

from app.core.config import settings
from app.services.media import generate_video_note, generate_voice_reply
from app.services.rag import RagService, parse_mode


dp = Dispatcher()
rag: RagService | None = None
user_modes: dict[int, str] = {}
user_pixel_mode: dict[int, bool] = {}



def get_rag() -> RagService:
    global rag
    if rag is None:
        rag = RagService()
    return rag


@dp.message(CommandStart())
async def start_handler(message: Message) -> None:
    uid = message.from_user.id
    user_modes[uid] = "text"
    user_pixel_mode[uid] = False
    await message.answer(
        "Avatar bot ready. Commands: /mode text|voice|video, /pixel on|off, /sources"
    )


@dp.message(Command("sources"))
async def sources_handler(message: Message) -> None:
    await message.answer(
        "Sources are available after ingest. Ask any question to get matching fragments."
    )


@dp.message(F.text.startswith("/mode"))
async def mode_handler(message: Message) -> None:
    mode = parse_mode(message.text)
    user_modes[message.from_user.id] = mode
    await message.answer(f"Mode switched to: {mode}")


@dp.message(F.text.startswith("/pixel"))
async def pixel_handler(message: Message) -> None:
    uid = message.from_user.id
    text = (message.text or "").lower().strip()
    enabled = text.endswith("on")
    if text.endswith("off"):
        enabled = False
    user_pixel_mode[uid] = enabled
    await message.answer(f"Pixel mode: {'on' if enabled else 'off'}")


@dp.message(F.text)
async def text_handler(message: Message) -> None:
    uid = message.from_user.id
    mode = user_modes.get(uid, "text")

    if rag is None:
        await message.answer("Первый запрос может занять до минуты: загружаю модель поиска...")
    answer = get_rag().answer(message.text, mode=mode)

    if mode == "text":
        await message.answer(answer)
        return

    if mode == "voice":
        await message.answer("Генерирую голосовой ответ...")
        try:
            voice_path = generate_voice_reply(answer, uid)
            await message.answer_voice(voice=FSInputFile(str(voice_path)))
        except Exception as exc:
            await message.answer(f"Ошибка генерации voice: {exc}")
        return

    await message.answer("Генерирую видео-кружок...")
    try:
        video_path = generate_video_note(answer, uid, pixel_mode=user_pixel_mode.get(uid, False))
        await message.answer_video_note(video_note=FSInputFile(str(video_path)))
    except Exception as exc:
        await message.answer(f"Ошибка генерации video note: {exc}")


async def run() -> None:
    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)
