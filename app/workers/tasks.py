from celery import Celery

from app.core.config import settings


celery_app = Celery("avatar", broker=settings.redis_url, backend=settings.redis_url)


@celery_app.task
def generate_voice_task(text: str, user_id: int) -> str:
    # TODO: integrate XTTS/OpenVoice here.
    return f"voice://user-{user_id}-placeholder.ogg"


@celery_app.task
def generate_video_task(text: str, user_id: int) -> str:
    # TODO: integrate Wav2Lip/LivePortrait + Telegram video-note formatting.
    return f"video://user-{user_id}-placeholder.mp4"


def enqueue_voice(text: str, user_id: int) -> str:
    res = generate_voice_task.delay(text, user_id)
    return res.id


def enqueue_video(text: str, user_id: int) -> str:
    res = generate_video_task.delay(text, user_id)
    return res.id
