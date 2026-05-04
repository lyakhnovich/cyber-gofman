import asyncio
import random
from contextlib import suppress

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.core.config import settings
from app.services.media import generate_video_note, generate_voice_reply, load_video_profiles
from app.services.rag import RagService, parse_mode


dp = Dispatcher()
rag: RagService | None = None
user_modes: dict[int, str] = {}
user_pixel_mode: dict[int, bool] = {}
user_video_profile: dict[int, dict] = {}
video_profiles = load_video_profiles()
PIXEL_ON_CB = "pixel_on"
PIXEL_OFF_CB = "pixel_off"
ABOUT_CB = "about"
TEXT_MODE_CB = "mode_text"
VIDEO_MODE_CB = "mode_video"



def _format_error(prefix: str, exc: Exception, limit: int = 900) -> str:
    # Telegram has a hard message-length limit; long URLs in exceptions can exceed it.
    raw = f"{prefix}: {exc}".replace("\n", " ").strip()
    if len(raw) <= limit:
        return raw
    return f"{raw[: limit - 3]}..."


async def _animate_loading(status_msg: Message, base_text: str = "Игорь Гофман думает") -> None:
    step = 0
    while True:
        dots = "." * (step % 4)
        with suppress(Exception):
            await status_msg.edit_text(f"{base_text}{dots}")
        step += 1
        await asyncio.sleep(0.7)


async def _start_loading(message: Message, base_text: str = "Игорь Гофман думает") -> tuple[Message, asyncio.Task]:
    status_msg = await message.answer(base_text)
    task = asyncio.create_task(_animate_loading(status_msg, base_text=base_text))
    return status_msg, task


async def _stop_loading(status_msg: Message | None, task: asyncio.Task | None) -> None:
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    if status_msg is not None:
        with suppress(Exception):
            await status_msg.delete()


def _main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Текстом", callback_data=TEXT_MODE_CB),
                InlineKeyboardButton(text="Видео", callback_data=VIDEO_MODE_CB),
            ],
            [
                InlineKeyboardButton(text="Pixel on", callback_data=PIXEL_ON_CB),
                InlineKeyboardButton(text="Pixel off", callback_data=PIXEL_OFF_CB),
            ],
            [InlineKeyboardButton(text="Что это?", callback_data=ABOUT_CB)],
        ],
    )


def get_rag() -> RagService:
    global rag
    if rag is None:
        rag = RagService()
    return rag


def _pick_video_profile(uid: int, reset: bool = False) -> dict | None:
    if not video_profiles:
        return None
    if reset or uid not in user_video_profile:
        user_video_profile[uid] = random.choice(video_profiles)
    return user_video_profile[uid]


@dp.message(CommandStart())
async def start_handler(message: Message) -> None:
    uid = message.from_user.id
    user_modes[uid] = "text"
    user_pixel_mode[uid] = False
    _pick_video_profile(uid, reset=True)
    await message.answer(
        "Игорь Гофман на связи. По умолчанию отвечаю текстом; кругляшки — кнопка «Видео».",
        reply_markup=_main_keyboard(),
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


@dp.callback_query(F.data == ABOUT_CB)
async def about_handler(callback: CallbackQuery) -> None:
    await callback.answer()
    variants = (
        "Это Игорь Гофман. С 1990 года занимаюсь самообразованием, чем восполнил пробелы официального образования.",
        "Это Игорь. Высшее образование: инженер по автоматизации. Второе высшее - программист.",
        "Я Игорь Авраамович Гофман. Занимаюсь разработками в области автоматизации, науками близкими к Каббале, историей, семантическим анализом по четырем языкам и наукой о человеке. Веду блог: httpigal-igal.blogspot.com",
    )
    await callback.message.answer(random.choice(variants))


@dp.callback_query(F.data == TEXT_MODE_CB)
async def text_mode_button_handler(callback: CallbackQuery) -> None:
    uid = callback.from_user.id
    user_modes[uid] = "text"
    await callback.answer("Режим: text")
    await callback.message.answer("Переключил в текстовый режим. Теперь отвечаю текстом быстрее.")


@dp.callback_query(F.data == VIDEO_MODE_CB)
async def video_mode_button_handler(callback: CallbackQuery) -> None:
    uid = callback.from_user.id
    user_modes[uid] = "video"
    await callback.answer("Режим: video")
    await callback.message.answer("Переключил в видео-режим. Снова отвечаю кругляшами.")


@dp.callback_query(F.data.in_({PIXEL_ON_CB, PIXEL_OFF_CB}))
async def quick_buttons_handler(callback: CallbackQuery) -> None:
    uid = callback.from_user.id
    enabled = callback.data == PIXEL_ON_CB
    user_pixel_mode[uid] = enabled
    await callback.answer("Pixel mode: on" if enabled else "Pixel mode: off")


@dp.message(F.text)
async def text_handler(message: Message) -> None:
    uid = message.from_user.id
    mode = user_modes.get(uid, "text")
    loading_msg: Message | None = None
    loading_task: asyncio.Task | None = None
    if mode == "video":
        loading_msg, loading_task = await _start_loading(message, base_text="Игорь Гофман думает")
    try:
        answer = await asyncio.to_thread(get_rag().answer, message.text, mode, uid)
    except Exception:
        await _stop_loading(loading_msg, loading_task)
        raise

    if mode == "text":
        await _stop_loading(loading_msg, loading_task)
        await message.answer(answer)
        return

    if mode == "voice":
        try:
            voice_path = await asyncio.to_thread(generate_voice_reply, answer, uid)
            await _stop_loading(loading_msg, loading_task)
            await message.answer_voice(voice=FSInputFile(str(voice_path)))
        except Exception as exc:
            await _stop_loading(loading_msg, loading_task)
            await message.answer(_format_error("Ошибка генерации voice", exc))
        return

    try:
        profile = _pick_video_profile(uid)
        video_path = await asyncio.to_thread(
            generate_video_note,
            answer,
            uid,
            user_pixel_mode.get(uid, False),
            profile,
        )
        await _stop_loading(loading_msg, loading_task)
        await message.answer_video_note(video_note=FSInputFile(str(video_path)))
    except Exception as exc:
        await _stop_loading(loading_msg, loading_task)
        await message.answer(_format_error("Ошибка генерации video note", exc))


async def run() -> None:
    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)
