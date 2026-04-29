import argparse
from pathlib import Path

from app.services.vector_store import Chunk, VectorStore


def chunk_text(text: str, source_name: str, chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_chars - overlap_chars)
    idx = 0
    start_pos = 0

    while start_pos < len(clean):
        end_pos = min(len(clean), start_pos + chunk_chars)
        piece = clean[start_pos:end_pos].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    video=source_name,
                    start=float(idx),
                    end=float(idx + 1),
                )
            )
            idx += 1
        if end_pos >= len(clean):
            break
        start_pos += step

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to source text file")
    parser.add_argument("--source-name", default="text_dataset", help="Source label in payload")
    parser.add_argument("--chunk-chars", type=int, default=900, help="Chunk size in characters")
    parser.add_argument("--overlap-chars", type=int, default=150, help="Chunk overlap in characters")
    args = parser.parse_args()

    file_path = Path(args.file)
    text = file_path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text=text,
        source_name=args.source_name,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )

    store = VectorStore()
    upserted = store.upsert_chunks(chunks)
    print(f"DONE: source={args.source_name}, chunks={upserted}")


if __name__ == "__main__":
    main()
