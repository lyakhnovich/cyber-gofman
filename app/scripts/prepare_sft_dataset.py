from __future__ import annotations

import json
import random
import re
from pathlib import Path


IN_PATH = Path("app/data/datasets/igor_qa_pairs.jsonl")
OUT_DIR = Path("app/data/datasets")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _is_valid_pair(question: str, answer: str) -> bool:
    q = (question or "").strip()
    a = (answer or "").strip()
    if len(q) < 12 or len(a) < 40:
        return False
    if len(q) > 240 or len(a) > 2200:
        return False
    if q.count("?") > 3:
        return False
    return True


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        key = f"{_normalize(q)}|||{_normalize(a)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        if not _is_valid_pair(q, a):
            continue
        rows.append({"question": q, "answer": a, "meta": row.get("meta", {})})
    return _dedupe_rows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=str(IN_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(Path(args.in_path))
    random.seed(args.seed)
    random.shuffle(rows)

    train_ratio = min(0.98, max(0.5, float(args.train_ratio)))
    split = int(len(rows) * train_ratio)
    train_rows = rows[:split]
    val_rows = rows[split:]

    out_dir = Path(args.out_dir)
    train_path = out_dir / "igor_sft_train.jsonl"
    val_path = out_dir / "igor_sft_val.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print(
        "DONE: "
        f"clean_rows={len(rows)}, "
        f"train={len(train_rows)}, "
        f"val={len(val_rows)}, "
        f"train_path={train_path}, "
        f"val_path={val_path}"
    )


if __name__ == "__main__":
    main()
