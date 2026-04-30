from __future__ import annotations

import re

from app.services.vector_store import VectorStore


class RagService:
    def __init__(self) -> None:
        self.store = VectorStore()

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

    @staticmethod
    def _stylize_speech(text: str) -> str:
        s = RagService._normalize(text)
        if not s:
            return s
        sentences = RagService._split_sentences(s)[:2]
        styled: list[str] = []
        lead_markers = ("значит", "ну", "в общем", "смотри")
        for i, sentence in enumerate(sentences):
            base = sentence.strip().rstrip(".!?")
            if not base:
                continue
            # Keep style, but avoid constant "значит" repetition.
            marker = lead_markers[(len(base) + i) % len(lead_markers)]
            if i == 0 and len(base) % 3 == 0 and not base.lower().startswith(marker):
                base = f"{marker.capitalize()}, {base[0].lower() + base[1:]}" if len(base) > 1 else f"{marker.capitalize()}, {base.lower()}"
            base = re.sub(r"\s+и\s+", " и одновременно ", base, count=1, flags=re.IGNORECASE)
            ending = "..."
            styled.append(f"{base}{ending}")
        return " ".join(styled) if styled else s

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

    def answer(self, user_text: str, mode: str = "text") -> str:
        hits = self.retrieve(user_text, limit=5)
        if self._is_identity_query(user_text):
            identity_hits = self.retrieve("кто ты меня зовут имя представься", limit=8)
            identity_sentence = self._identity_sentence_from_hits(identity_hits or hits)
            if identity_sentence:
                identity_text = identity_sentence
                if "херсон" not in identity_text.lower():
                    identity_text = f"{identity_text} Я из города Херсон, Украина."
                return self._stylize_speech(identity_text)

        if not hits:
            return self._stylize_speech("Тема рядом, я продолжаю и сейчас собираю мысль")

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
                return self._stylize_speech("Тема рядом, я продолжаю и сейчас собираю мысль")
            sentence = self._split_sentences(text)
            return self._stylize_speech(sentence[0] if sentence else text[:220])

        candidates.sort(key=lambda x: x[0], reverse=True)
        sentence_1 = candidates[0][1]
        sentence_2 = ""
        for _, sentence in candidates[1:]:
            if sentence != sentence_1:
                sentence_2 = sentence
                break

        if sentence_2:
            return self._stylize_speech(f"{sentence_1} {sentence_2}")
        return self._stylize_speech(sentence_1)


def parse_mode(text: str) -> str:
    if text.startswith("/mode"):
        parts = text.split()
        if len(parts) > 1 and parts[1] in {"text", "voice", "video"}:
            return parts[1]
    return "text"
