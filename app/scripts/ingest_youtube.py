import json
import subprocess
import sys
import re
from pathlib import Path

import librosa
import numpy as np
from faster_whisper import WhisperModel

from app.services.vector_store import Chunk, VectorStore

RAW_DIR = Path("app/data/raw_videos")
TRANSCRIPTS_DIR = Path("app/data/transcripts")


class SpeakerFilter:
    def __init__(self, reference_wav: Path, threshold: float = 0.62) -> None:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
        except ImportError as exc:
            raise RuntimeError(
                "Speaker filtering requires optional dependency 'resemblyzer'. "
                "Install it separately only if you need --target-reference."
            ) from exc
        self.threshold = threshold
        self.encoder = VoiceEncoder()
        ref_wav = preprocess_wav(str(reference_wav))
        self.ref_embedding = self.encoder.embed_utterance(ref_wav)

    def similarity(self, audio: np.ndarray) -> float:
        if len(audio) < 16000:
            return 0.0
        emb = self.encoder.embed_utterance(audio)
        num = float(np.dot(self.ref_embedding, emb))
        den = float(np.linalg.norm(self.ref_embedding) * np.linalg.norm(emb))
        if den == 0:
            return 0.0
        return num / den



def download_from_youtube(url: str) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in RAW_DIR.glob("*.mp4")}
    out_template = str(RAW_DIR / "%(uploader)s__%(id)s.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "mp4/bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "-o",
        out_template,
        url,
    ]
    subprocess.run(cmd, check=True)

    # Prefer exact file match by YouTube video id (works for "already downloaded" too).
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if m:
        yt_id = m.group(1)
        exact = sorted(
            RAW_DIR.glob(f"*__{yt_id}.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if exact:
            return [exact[0]]

    after = sorted(RAW_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    new_files = [p for p in after if p.name not in before]
    return new_files if new_files else after



def load_mono_16k(video_path: Path) -> np.ndarray:
    wav, sr = librosa.load(str(video_path), sr=16000, mono=True)
    return wav



def transcribe(video_path: Path, model: WhisperModel, speaker_filter: SpeakerFilter | None = None) -> Path:
    segments, info = model.transcribe(str(video_path), vad_filter=True)
    audio = load_mono_16k(video_path) if speaker_filter else None

    out_segments = []
    kept = 0
    total = 0

    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        total += 1
        keep = True

        if speaker_filter and audio is not None:
            start = max(0, int(s.start * 16000))
            end = min(len(audio), int(s.end * 16000))
            snippet = audio[start:end]
            sim = speaker_filter.similarity(snippet)
            keep = sim >= speaker_filter.threshold

        if keep:
            kept += 1
            out_segments.append({"start": s.start, "end": s.end, "text": text})

    payload = {
        "video": str(video_path),
        "language": info.language,
        "duration": info.duration,
        "segments": out_segments,
    }

    out = TRANSCRIPTS_DIR / f"{video_path.stem}.json"
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Speaker filter kept {kept}/{total} segments")
    return out



def chunk_transcript(transcript_path: Path, max_chars: int = 900) -> list[Chunk]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    buf: list[str] = []
    start = 0.0
    end = 0.0

    for seg in payload.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if not buf:
            start = float(seg.get("start", 0.0))
        buf.append(text)
        end = float(seg.get("end", start))

        if len(" ".join(buf)) >= max_chars:
            chunks.append(Chunk(text=" ".join(buf), video=payload["video"], start=start, end=end))
            buf = []

    if buf:
        chunks.append(Chunk(text=" ".join(buf), video=payload["video"], start=start, end=end))

    return chunks



def read_urls(args_url: str | None, urls_file: str | None) -> list[str]:
    urls: list[str] = []
    if args_url:
        urls.append(args_url)
    if urls_file:
        path = Path(urls_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                urls.append(clean)
    return urls



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="YouTube video or playlist URL")
    parser.add_argument("--urls-file", help="Text file with one YouTube URL per line")
    parser.add_argument("--max-per-url", type=int, default=3, help="How many downloaded videos to process per URL")
    parser.add_argument("--model", default="medium", help="Whisper model size")
    parser.add_argument("--target-reference", help="Path to WAV sample of target speaker")
    parser.add_argument("--speaker-threshold", type=float, default=0.62, help="Cosine threshold for speaker match")
    args = parser.parse_args()

    urls = read_urls(args.url, args.urls_file)
    if not urls:
        raise ValueError("Provide at least one source via --url or --urls-file")

    # Try GPU first, but gracefully fall back to CPU when CUDA runtime is missing.
    try:
        whisper = WhisperModel(args.model, device="cuda", compute_type="float16")
    except Exception:
        whisper = WhisperModel(args.model, device="cpu", compute_type="int8")

    speaker_filter = None
    if args.target_reference:
        speaker_filter = SpeakerFilter(Path(args.target_reference), threshold=args.speaker_threshold)

    store = VectorStore()

    total_videos = 0
    total_chunks = 0

    for url in urls:
        videos = download_from_youtube(url)[: args.max_per_url]
        print(f"SOURCE: {url} -> {len(videos)} video(s) selected", flush=True)
        for video in videos:
            try:
                print(f"TRANSCRIBE_START: {video.name}", flush=True)
                transcript = transcribe(video, whisper, speaker_filter=speaker_filter)
                print(f"TRANSCRIBE_DONE: {transcript.name}", flush=True)
            except RuntimeError as exc:
                if "cublas64_12.dll" not in str(exc):
                    raise
                print("CUDA runtime not found, retrying transcription on CPU...")
                whisper = WhisperModel(args.model, device="cpu", compute_type="int8")
                print(f"TRANSCRIBE_START_CPU: {video.name}", flush=True)
                transcript = transcribe(video, whisper, speaker_filter=speaker_filter)
                print(f"TRANSCRIBE_DONE_CPU: {transcript.name}", flush=True)
            chunks = chunk_transcript(transcript)
            upserted = store.upsert_chunks(chunks)
            total_videos += 1
            total_chunks += upserted
            print(f"OK: {video.name} -> {transcript.name} -> {upserted} chunks")

    print(f"DONE: videos={total_videos}, chunks={total_chunks}")


if __name__ == "__main__":
    main()
