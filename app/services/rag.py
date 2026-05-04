from __future__ import annotations

import logging
import random
import re
from typing import Any

from app.core.config import settings
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RagService:
    _PERSONA_FACTS = {
        "name": "Гофман Игорь (Игал) Авраамович",
        "birth_year": "1966",
        "location": "Херсон, Украина",
        "occupation": "бизнес, учредитель МП «Экос» (с 1992 года)",
        "education": "высшее: инженер по автоматизации; второе высшее: программист",
        "interests": "научные разработки в области автоматики, науки близкие к каббале, наука о человеке",
        "goal": "построение нового Храма",
        "extra": "с 1990 года занимаюсь самообразованием, восполняя пробелы официального образования",
        "worldview": "вокруг часто выглядит как сплошной Голливуд: много притворства, подмена сути образом и изображение не того, чем является по факту",
        "people_theatre": (
            "Часто воспринимаю окружающих как подставных актёров спектакля — включая даже маму; "
            "говорю «так называемые», в переписке сокращаю до «т.н.»."
        ),
    }

    def __init__(self) -> None:
        self.store = VectorStore()
        self._session_state: dict[int, dict[str, Any]] = {}

    def retrieve(self, user_text: str, limit: int = 5) -> list[dict]:
        return self.store.search(user_text, limit=limit)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def clip_reply(text: str, max_sentences: int = 3) -> str:
        """Keep answers short for Telegram / TTS (2–3 sentences)."""
        text = RagService._normalize(text)
        if not text:
            return text
        parts = RagService._split_sentences(text)
        if not parts:
            if len(text) > 420:
                cut = text[:400].rsplit(" ", 1)[0].rstrip(",;:")
                return cut + "…"
            return text
        if len(parts) == 1 and len(parts[0]) > 520:
            cut = parts[0][:480].rsplit(" ", 1)[0].rstrip(",;:")
            return cut + "…"
        if len(parts) <= max_sentences:
            return " ".join(parts).strip()
        return " ".join(parts[:max_sentences]).strip()

    @staticmethod
    def _is_good_sentence(sentence: str) -> bool:
        s = RagService._normalize(sentence)
        if len(s) < 35 or len(s) > 260:
            return False
        words = re.findall(r"\w+", s, flags=re.UNICODE)
        if len(words) < 6:
            return False
        # Drop obviously noisy citation-like fragments.
        if "см." in s.lower() or "статью" in s.lower():
            return False
        punct_count = len(re.findall(r"[,:;()«»\"\-]", s))
        return punct_count <= max(14, len(s) // 8)

    @staticmethod
    def _overlap_score(sentence: str, query: str) -> int:
        q_tokens = {t for t in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if len(t) > 3}
        s_tokens = set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
        return len(q_tokens & s_tokens)

    def _get_state(self, user_id: int | None) -> dict[str, Any]:
        if user_id is None:
            return {
                "mood": "neutral",
                "energy": 0,
                "last_greeting": "",
                "last_lead_marker": "",
                "last_mode": "neutral",
                "turns": 0,
                "recent_topics": [],
                "last_topic_bridge_turn": -99,
                "last_worldview_turn": -99,
            }
        state = self._session_state.get(user_id)
        if state is None:
            state = {
                "mood": "neutral",
                "energy": 0,
                "last_greeting": "",
                "last_lead_marker": "",
                "last_mode": "neutral",
                "turns": 0,
                "recent_topics": [],
                "last_topic_bridge_turn": -99,
                "last_worldview_turn": -99,
            }
            self._session_state[user_id] = state
        return state

    def _select_response_mode(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ("кто", "биограф", "о себе", "представ", "истори")):
            return "bio"
        if any(x in q for x in ("почему", "зачем", "как", "объясни")):
            return "reasoned"
        if "?" in q:
            return "dialogue"
        return "neutral"

    @staticmethod
    def _extract_topics(query: str) -> list[str]:
        tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
        stop = {
            "это", "как", "что", "где", "когда", "зачем", "почему", "который", "которые",
            "меня", "вам", "вас", "тебя", "про", "для", "или", "если", "только", "можно",
            "надо", "очень", "просто", "будет", "есть", "был", "была", "были", "the",
        }
        uniq: list[str] = []
        for t in tokens:
            if len(t) < 4 or t in stop:
                continue
            if t not in uniq:
                uniq.append(t)
        return uniq[:3]

    def _update_state(self, user_id: int | None, query: str, response_mode: str) -> dict[str, Any]:
        state = self._get_state(user_id)
        q = query.strip()
        mood = "neutral"
        if "!" in q or any(x in q.lower() for x in ("срочно", "немедленно", "быстро")):
            mood = "intense"
        elif any(x in q.lower() for x in ("спасибо", "благодар")):
            mood = "warm"
        state["mood"] = mood
        state["last_mode"] = response_mode
        state["energy"] = min(5, int(state.get("energy", 0)) + 1)
        state["turns"] = int(state.get("turns", 0)) + 1
        fresh_topics = self._extract_topics(q)
        prev_topics = list(state.get("recent_topics", []))
        merged = fresh_topics + [t for t in prev_topics if t not in fresh_topics]
        state["recent_topics"] = merged[:3]
        return state

    def _topic_bridge(self, user_id: int | None, query: str) -> str:
        state = self._get_state(user_id)
        recent_topics = list(state.get("recent_topics", []))
        if not recent_topics:
            return ""
        turns = int(state.get("turns", 0))
        last_bridge_turn = int(state.get("last_topic_bridge_turn", -99))
        if turns - last_bridge_turn < 3:
            return ""
        q_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
        match = next((t for t in recent_topics if t in q_tokens), "")
        if not match:
            return ""
        variants = (
            f"К слову, по теме «{match}» мы с Вами уже шли в этом направлении.",
            f"Если продолжать линию «{match}», картина становится понятнее.",
        )
        state["last_topic_bridge_turn"] = turns
        return random.choice(variants)

    def _stylize_speech(self, text: str, user_id: int | None = None, response_mode: str = "neutral") -> str:
        s = RagService._normalize(text)
        if not s:
            return s
        state = self._get_state(user_id)
        sentences = RagService._split_sentences(s)[:3]
        styled: list[str] = []
        mood = str(state.get("mood", "neutral"))
        mode_markers: dict[str, tuple[str, ...]] = {
            "bio": ("позвольте представиться", "скажу о себе точно", "если говорить предметно"),
            "reasoned": ("если по существу", "позвольте пояснить", "логика здесь следующая"),
            "dialogue": ("видите ли", "я вам так отвечу", "скажу аккуратно"),
            "neutral": ("видите ли", "позвольте заметить", "если по существу", "скажу аккуратно"),
        }
        lead_markers = mode_markers.get(response_mode, mode_markers["neutral"])
        if mood == "warm":
            lead_markers = ("скажу с уважением",) + lead_markers
        elif mood == "intense":
            lead_markers = ("скажу прямо",) + lead_markers
        odessa_turns = (
            "как говорят в Одессе",
            "между нами говоря",
            "я вам так скажу",
            "картина, конечно, примечательная",
        )
        last_marker = str(state.get("last_lead_marker", ""))
        for i, sentence in enumerate(sentences):
            base = sentence.strip().rstrip(".!?")
            if not base:
                continue
            # Keep style while avoiding repetitive openings.
            marker = lead_markers[(len(base) + i + int(state.get("turns", 0))) % len(lead_markers)]
            if marker == last_marker and len(lead_markers) > 1:
                marker = lead_markers[(lead_markers.index(marker) + 1) % len(lead_markers)]
            last_marker = marker
            if i == 0 and len(base) % 3 == 0 and not base.lower().startswith(marker):
                base = f"{marker.capitalize()}, {base[0].lower() + base[1:]}" if len(base) > 1 else f"{marker.capitalize()}, {base.lower()}"
            if i == 0 and len(base) % 2 == 0:
                turn = odessa_turns[len(base) % len(odessa_turns)]
                base = f"{base}, {turn}"
            base = re.sub(r"\s+и\s+", " и одновременно ", base, count=1, flags=re.IGNORECASE)
            ending = "..."
            styled.append(f"{base}{ending}")
        state["last_lead_marker"] = last_marker
        return " ".join(styled) if styled else s

    def _respectful_greeting(self, user_id: int | None = None) -> str:
        variants = (
            "Здравствуйте.",
            "Добрый день.",
        )
        state = self._get_state(user_id)
        last = str(state.get("last_greeting", ""))
        choices = [v for v in variants if v != last] or list(variants)
        selected = random.choice(choices)
        state["last_greeting"] = selected
        return selected

    def _with_respectful_greeting(self, text: str, user_id: int | None = None) -> str:
        body = self._normalize(text)
        if not body:
            return body
        lowered = body.lower()
        if lowered.startswith("здравствуйте") or lowered.startswith("добрый день"):
            return body
        return f"{self._respectful_greeting(user_id)} {body}"

    @staticmethod
    def _is_identity_query(query: str) -> bool:
        q = query.lower()
        triggers = (
            "кто ты",
            "кто вы",
            "как тебя зовут",
            "как вас зовут",
            "представься",
            "ты кто",
        )
        return any(t in q for t in triggers)

    @staticmethod
    def _identity_sentence_from_hits(hits: list[dict]) -> str:
        patterns = (
            r"меня зовут[^.!?]{0,120}[.!?]?",
            r"я\s+[^.!?]{0,80}\bзовут\b[^.!?]{0,80}[.!?]?",
            r"я живу[^.!?]{0,120}[.!?]?",
        )
        for hit in hits:
            text = str(hit.get("text", "")).strip()
            if not text:
                continue
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE | re.UNICODE)
                if m:
                    sentence = RagService._normalize(m.group(0))
                    if len(sentence) > 12:
                        return sentence
            for sentence in RagService._split_sentences(text):
                s = RagService._normalize(sentence)
                if "меня зовут" in s.lower() and len(s) > 12:
                    return s
        return ""

    @classmethod
    def _persona_identity(cls) -> str:
        p = cls._PERSONA_FACTS
        return (
            f"Я {p['name']}, {p['birth_year']} года рождения, живу в {p['location']}. "
            f"Мой род занятий: {p['occupation']}. "
            f"Образование: {p['education']}."
        )

    @classmethod
    def _persona_bio_extension(cls, query: str) -> str:
        q = query.lower()
        p = cls._PERSONA_FACTS
        chunks: list[str] = []
        if any(x in q for x in ("род занятий", "род деятельности", "чем занимаетесь", "чем занимаешься", "ваша деятельность", "твоя деятельность")):
            chunks.append(
                "Мой род деятельности: бизнес и инженерно-программные проекты. "
                "С 1992 года я учредитель МП «Экос»."
            )
        if any(x in q for x in ("интерес", "книг", "фильм", "музык", "хобби", "любим")):
            chunks.append(
                "Интересы: "
                f"{p['interests']}. Любимые книги: Лион Фейхтвангер. "
                "Музыка: барды и клейзмерская музыка. Фильмы: все познавательные для меня."
            )
        if any(x in q for x in ("цель", "мисси", "зачем", "смысл")):
            chunks.append(f"Моя цель: {p['goal']}.")
        if any(x in q for x in ("опыт", "работ", "карьер", "экос", "бизнес")):
            chunks.append("Опыт: с 1988 по 1990 работал инженером в центре вычислительной техники, с 1992 года учредитель МП «Экос».")
        if any(x in q for x in ("кто", "биограф", "о себе", "представ", "истори")):
            chunks.append(p["extra"] + ".")
            chunks.append("По моему ощущению, " + p["worldview"] + ".")
        if any(
            x in q
            for x in (
                "мама",
                "семья",
                "родствен",
                "люди",
                "окружающ",
                "знаком",
                "близк",
                "отношен",
                "актер",
                "театр",
                "подстав",
                "голливуд",
                "маск",
            )
        ):
            chunks.append(p["people_theatre"])
        return " ".join(chunks).strip()

    @classmethod
    def _persona_worldview_hint(cls, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ("мама", "семья", "родствен", "близкие", "люди вокруг", "окружающ", "знакомые", "сосед")):
            return (
                "Я нередко смотрю на людей как на актёров постановки — и на близких тоже; "
                "в разговоре говорю «так называемые», в письме кратко «т.н.»."
            )
        if any(x in q for x in ("актер", "маск", "голливуд", "театр", "подстава", "абсурд")):
            return (
                "Иногда складывается впечатление сплошного Голливуда: "
                "вокруг изображают одно, а по факту скрывается другое, много притворства и масок."
            )
        if any(x in q for x in ("почему", "зачем", "происходит", "что творится")):
            return "События часто выглядят как тщательно поставленная сцена, с абсурдом и ролями."
        return ""

    def _throttled_worldview_hint(self, user_id: int | None, query: str) -> str:
        state = self._get_state(user_id)
        turns = int(state.get("turns", 0))
        last_worldview_turn = int(state.get("last_worldview_turn", -99))
        if turns - last_worldview_turn < 4:
            return ""
        hint = self._persona_worldview_hint(query)
        if hint:
            state["last_worldview_turn"] = turns
        return hint

    def answer(self, user_text: str, mode: str = "text", user_id: int | None = None) -> str:
        response_mode = self._select_response_mode(user_text)
        self._update_state(user_id, user_text, response_mode)
        hits = self.retrieve(user_text, limit=5)

        if settings.local_llm_enabled:
            try:
                from app.services.local_llm import generate_reply

                generated = generate_reply(self, user_text, hits)
                if generated.strip():
                    return self._with_respectful_greeting(generated.strip(), user_id)
            except Exception as exc:
                logger.warning("Local LLM generation failed, falling back to RAG: %s", exc, exc_info=True)

        if self._is_identity_query(user_text):
            identity_hits = self.retrieve("кто ты меня зовут имя представься биография", limit=8)
            identity_sentence = self._identity_sentence_from_hits(identity_hits or hits)
            persona_identity = self._persona_identity()
            if identity_sentence:
                identity_text = identity_sentence
                if "херсон" not in identity_text.lower():
                    identity_text = f"{identity_text} Я из города Херсон, Украина."
                return self._stylize_speech(
                    self._with_respectful_greeting(RagService.clip_reply(f"{persona_identity} {identity_text}"), user_id),
                    user_id,
                    response_mode,
                )
            return self._stylize_speech(
                self._with_respectful_greeting(RagService.clip_reply(persona_identity), user_id),
                user_id,
                response_mode,
            )

        if not hits:
            return self._stylize_speech(
                self._with_respectful_greeting(RagService.clip_reply("Тема рядом, я продолжаю и сейчас собираю мысль"), user_id),
                user_id,
                response_mode,
            )

        best = hits[0]
        score = float(best.get("_score", 0.0))
        if score < 0.22:
            # Continue with soft fallback instead of asking to clarify.
            pass

        candidates: list[tuple[float, str]] = []
        for hit in hits[:3]:
            hit_score = float(hit.get("_score", 0.0))
            text = self._normalize(str(hit.get("text", "")))
            if not text:
                continue
            for sentence in self._split_sentences(text):
                if not self._is_good_sentence(sentence):
                    continue
                overlap = self._overlap_score(sentence, user_text)
                # Prioritize semantic hit score, then lexical overlap.
                rank = hit_score * 10 + overlap
                if overlap > 0:
                    candidates.append((rank, sentence))

        # Soft fallback: if lexical overlap is weak, still use the best clean sentence.
        if not candidates:
            for hit in hits[:2]:
                hit_score = float(hit.get("_score", 0.0))
                text = self._normalize(str(hit.get("text", "")))
                if not text:
                    continue
                for sentence in self._split_sentences(text):
                    if not self._is_good_sentence(sentence):
                        continue
                    candidates.append((hit_score * 10, sentence))

        if not candidates:
            text = self._normalize(str(best.get("text", "")))
            if not text:
                return self._stylize_speech(
                    self._with_respectful_greeting(RagService.clip_reply("Тема рядом, я продолжаю и сейчас собираю мысль"), user_id),
                    user_id,
                    response_mode,
                )
            sentence = self._split_sentences(text)
            return self._stylize_speech(
                self._with_respectful_greeting(
                    RagService.clip_reply(sentence[0] if sentence else text[:220]),
                    user_id,
                ),
                user_id,
                response_mode,
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        sentence_1 = candidates[0][1]
        sentence_2 = ""
        for _, sentence in candidates[1:]:
            if sentence != sentence_1:
                sentence_2 = sentence
                break

        if sentence_2:
            base_answer = f"{sentence_1} {sentence_2}"
        else:
            base_answer = sentence_1
        persona_tail = self._persona_bio_extension(user_text)
        worldview_hint = self._throttled_worldview_hint(user_id, user_text)
        topic_bridge = self._topic_bridge(user_id, user_text)
        addons = " ".join(x for x in (topic_bridge, persona_tail, worldview_hint) if x).strip()
        if addons:
            body = RagService.clip_reply(f"{base_answer} {addons}")
            return self._stylize_speech(self._with_respectful_greeting(body, user_id), user_id, response_mode)
        return self._stylize_speech(self._with_respectful_greeting(RagService.clip_reply(base_answer), user_id), user_id, response_mode)


def parse_mode(text: str) -> str:
    if text.startswith("/mode"):
        parts = text.split()
        if len(parts) > 1 and parts[1] in {"text", "voice", "video"}:
            return parts[1]
    return "text"
