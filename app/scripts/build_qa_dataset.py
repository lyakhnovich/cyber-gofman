from __future__ import annotations

import json
import random
import re
from pathlib import Path


TRANSCRIPTS_DIR = Path("app/data/transcripts")
OUT_DIR = Path("app/data/datasets")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", _normalize(text)) if s.strip()]


def _safe_summary(text: str, max_words: int = 14) -> str:
    words = re.findall(r"\w+|[^\w\s]", _normalize(text), flags=re.UNICODE)
    if not words:
        return "этот вопрос"
    short = " ".join(words[:max_words]).strip(" ,;:-")
    return short or "этот вопрос"


def _assemble_passages(segments: list[dict], min_chars: int = 220, max_chars: int = 700) -> list[dict]:
    passages: list[dict] = []
    buf: list[str] = []
    start = 0.0
    end = 0.0
    for seg in segments:
        text = _normalize(str(seg.get("text", "")))
        if not text:
            continue
        if not buf:
            start = float(seg.get("start", 0.0))
        buf.append(text)
        end = float(seg.get("end", start))
        joined = " ".join(buf)
        if len(joined) >= max_chars:
            passages.append({"text": joined, "start": start, "end": end})
            buf = []
    if buf:
        joined = " ".join(buf)
        if len(joined) >= min_chars:
            passages.append({"text": joined, "start": start, "end": end})
    return passages


def _question_variants(topic: str) -> tuple[str, str]:
    return (
        f"Как Вы это объясняете: {topic}?",
        f"Что Вы имеете в виду, когда говорите про {topic}?",
    )


def _make_pairs_from_passage(text: str, style_prefix: str) -> list[tuple[str, str]]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    topic = _safe_summary(sentences[0])
    q1, q2 = _question_variants(topic)
    answer = _normalize(text)
    if style_prefix:
        answer = f"{style_prefix} {answer}"
    return [(q1, answer), (q2, answer)]


def build_dataset(
    transcripts_dir: Path,
    out_file: Path,
    max_files: int | None,
    seed: int,
    style_prefix: str,
) -> tuple[int, int]:
    random.seed(seed)
    files = sorted(transcripts_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if max_files is not None:
        files = files[:max_files]

    pairs_written = 0
    files_used = 0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            segments = payload.get("segments", [])
            if not isinstance(segments, list) or not segments:
                continue
            passages = _assemble_passages(segments)
            if not passages:
                continue
            random.shuffle(passages)
            for p in passages:
                text = str(p.get("text", ""))
                for q, a in _make_pairs_from_passage(text, style_prefix):
                    row = {
                        "question": q,
                        "answer": a,
                        "meta": {
                            "video": payload.get("video", ""),
                            "start": float(p.get("start", 0.0)),
                            "end": float(p.get("end", 0.0)),
                            "source_transcript": path.name,
                        },
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pairs_written += 1
            files_used += 1
    return files_used, pairs_written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts-dir", default=str(TRANSCRIPTS_DIR))
    parser.add_argument("--out", default=str(OUT_DIR / "igor_qa_pairs.jsonl"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--style-prefix",
        default="Здравствуйте. Скажу по существу.",
        help="Optional prefix added to each answer for stable style.",
    )
    args = parser.parse_args()

    files_used, pairs_written = build_dataset(
        transcripts_dir=Path(args.transcripts_dir),
        out_file=Path(args.out),
        max_files=args.max_files,
        seed=args.seed,
        style_prefix=args.style_prefix.strip(),
    )
    print(f"DONE: files_used={files_used}, qa_pairs={pairs_written}, out={args.out}")


if __name__ == "__main__":
    main()
