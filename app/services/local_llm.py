from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.core.config import settings
from app.services.rag import RagService

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MODEL: Any = None
_TOKENIZER: Any = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _project_root() / p


def _load_model() -> tuple[Any, Any]:
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    adapter = _resolve(settings.local_llm_adapter_path)
    if not adapter.is_dir() or not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter not found at {adapter}")

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading tokenizer from %s", adapter)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model %s (4-bit)", settings.local_llm_base_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.local_llm_base_model,
        torch_dtype=torch.float16,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    logger.info("Loading LoRA adapter from %s", adapter)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    _MODEL = model
    _TOKENIZER = tokenizer
    return _MODEL, _TOKENIZER


def _truncate_context(blocks: list[str], max_chars: int = 12000) -> str:
    out: list[str] = []
    total = 0
    for b in blocks:
        if total >= max_chars:
            break
        piece = b[: max_chars - total]
        out.append(piece)
        total += len(piece)
    return "\n---\n".join(out)


def _build_instruction(rag: Any, user_text: str, hits: list[dict[str, Any]]) -> str:
    ctx_parts: list[str] = []
    for h in hits[:6]:
        t = str(h.get("text", "")).strip()
        if t:
            ctx_parts.append(t)
    context = _truncate_context(ctx_parts) if ctx_parts else "(В базе пока нет близких фрагментов.)"

    quirk = (
        "Черта характера: окружающих нередко воспринимаешь как подставных актёров сценария (включая близких); "
        "в речи уместно «так называемые», в письменной форме — сокращение «т.н.». "
        "Не перегружай ответ этим, используй по уместности."
    )

    brief = "Ответ строго из двух или трёх предложений; без списков и без длинных абзацев."

    if rag._is_identity_query(user_text):
        persona = rag._persona_identity()
        return (
            f"{persona}\n\n{quirk}\n\n"
            f"{brief}\n\n"
            "Ответь от первого лица, по существу. "
            "Можешь опереться на фрагменты ниже, если они уместны.\n\n"
            f"Фрагменты:\n{context}\n\n"
            f"Вопрос собеседника:\n{user_text.strip()}"
        )

    return (
        "Ты отвечаешь от лица Игоря Гофмана (Игала Авраамовича). "
        "Стиль: интеллигентно, спокойно, иногда лёгкие одесские обороты. "
        f"{quirk} "
        f"{brief} "
        "Опирайся на фрагменты ниже; не выдумывай конкретные факты, которых нет в тексте. "
        "Если фрагментов мало, отвечай осторожно, в характере персонажа.\n\n"
        f"Фрагменты:\n{context}\n\n"
        f"Вопрос собеседника:\n{user_text.strip()}"
    )


def generate_reply(rag: Any, user_text: str, hits: list[dict[str, Any]]) -> str:
    """
    Return generated text, or "" to signal fallback to rule+RAG path.
    """
    if not settings.local_llm_enabled:
        return ""
    if not torch.cuda.is_available():
        logger.warning("LOCAL_LLM_ENABLED but CUDA unavailable; using RAG fallback")
        return ""

    instruction = _build_instruction(rag, user_text, hits)
    prompt = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )

    with _LOCK:
        model, tokenizer = _load_model()
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(settings.local_llm_max_new_tokens),
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not text:
        return ""
    return RagService.clip_reply(text)
