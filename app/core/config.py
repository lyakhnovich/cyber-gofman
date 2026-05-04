from dataclasses import dataclass
from os import getenv


def _env_bool(name: str, default: str = "false") -> bool:
    return getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    telegram_bot_token: str = getenv("TELEGRAM_BOT_TOKEN", "")
    openai_api_key: str = getenv("OPENAI_API_KEY", "")
    qdrant_url: str = getenv("QDRANT_URL", "http://localhost:6333")
    redis_url: str = getenv("REDIS_URL", "redis://localhost:6379/0")
    llm_model: str = getenv("LLM_MODEL", "gpt-4o-mini")
    avatar_name: str = getenv("AVATAR_NAME", "Digital Avatar")
    qdrant_collection: str = getenv("QDRANT_COLLECTION", "avatar_chunks")
    embed_model: str = getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    reference_video_path: str = getenv("REFERENCE_VIDEO_PATH", "app/data/reference_circle.mp4")
    reference_profiles_path: str = getenv("REFERENCE_PROFILES_PATH", "app/data/video_profiles.json")
    reference_still_path: str = getenv("REFERENCE_STILL_PATH", "")
    reference_voice_path: str = getenv("REFERENCE_VOICE_PATH", "app/data/reference.wav")
    xtts_model: str = getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    xtts_language: str = getenv("XTTS_LANGUAGE", "ru")
    fallback_still_second: float = float(getenv("FALLBACK_STILL_SECOND", "1.0"))
    fallback_clip_seconds: float = float(getenv("FALLBACK_CLIP_SECONDS", "1.8"))
    wav2lip_repo_path: str = getenv("WAV2LIP_REPO_PATH", "")
    wav2lip_checkpoint_path: str = getenv("WAV2LIP_CHECKPOINT_PATH", "")
    # Local QLoRA (Qwen + PEFT adapter). See README.
    local_llm_enabled: bool = _env_bool("LOCAL_LLM_ENABLED", "true")
    local_llm_base_model: str = getenv("LOCAL_LLM_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    local_llm_adapter_path: str = getenv("LOCAL_LLM_ADAPTER_PATH", "app/data/models/igor-lora/adapter")
    local_llm_max_new_tokens: int = int(getenv("LOCAL_LLM_MAX_NEW_TOKENS", "240"))


settings = Settings()
