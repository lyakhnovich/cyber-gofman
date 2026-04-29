import os
import subprocess
import sys
import uuid
from pathlib import Path

from app.core.config import settings

OUT_DIR = Path("app/data/outputs")
_XTTS = None
_COQUI_DOWNLOAD_PATCHED = False


def _hf_coqui_resolve_url(scarf_url: str) -> str:
    marker = "/hf-coqui/"
    if marker not in scarf_url or "coqui.gateway.scarf.sh" not in scarf_url:
        return scarf_url
    tail = scarf_url.split(marker, 1)[1]
    parts = tail.split("/")
    if len(parts) < 3:
        return scarf_url
    repo, revision = parts[0], parts[1]
    rest = "/".join(parts[2:])
    return f"https://huggingface.co/coqui/{repo}/resolve/{revision}/{rest}"


def _parse_hf_resolve_url(url: str) -> tuple[str, str, str] | None:
    """Return (repo_id, revision, filename) for huggingface.co/.../resolve/... URLs."""
    base = url.split("?", 1)[0]
    marker = "huggingface.co/"
    if marker not in base:
        return None
    path = base.split(marker, 1)[1]
    parts = path.split("/")
    if len(parts) < 5 or parts[2] != "resolve":
        return None
    org, repo = parts[0], parts[1]
    revision = parts[3]
    filename = "/".join(parts[4:])
    if not filename:
        return None
    return f"{org}/{repo}", revision, filename


def _patch_coqui_scarf_to_huggingface() -> None:
    """Coqui uses Scarf (often blocked) and plain requests for HF URLs (redirects to xethub).
    We map Scarf → HF and download via huggingface_hub with XET disabled (see app.main HF_HUB_DISABLE_XET).
    Set COQUI_USE_SCARF=1 to keep Scarf URLs (still uses patched downloader for HF resolve).
    """
    global _COQUI_DOWNLOAD_PATCHED
    if _COQUI_DOWNLOAD_PATCHED:
        return
    if os.environ.get("COQUI_USE_SCARF", "").lower() in ("1", "true", "yes"):
        _COQUI_DOWNLOAD_PATCHED = True
        return
    try:
        from TTS.utils import manage as tts_manage
    except ImportError:
        return

    orig = tts_manage.ModelManager._download_model_files

    def _download_model_files_wrapped(file_urls, output_folder, progress_bar):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        mapped = [_hf_coqui_resolve_url(u) for u in file_urls]
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return orig(mapped, output_folder, progress_bar)

        for file_url in mapped:
            parsed = _parse_hf_resolve_url(file_url)
            if parsed is None:
                orig([file_url], output_folder, progress_bar)
                continue
            repo_id, revision, filename = parsed
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=output_folder,
                local_dir_use_symlinks=False,
            )

    tts_manage.ModelManager._download_model_files = staticmethod(_download_model_files_wrapped)
    _COQUI_DOWNLOAD_PATCHED = True


def _find_ffmpeg() -> str:
    candidates = [
        "ffmpeg",
        str(Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"),
    ]
    for candidate in candidates:
        try:
            proc = subprocess.run([candidate, "-version"], capture_output=True, text=True)
            if proc.returncode == 0:
                return candidate
        except Exception:
            continue

    winget_dir = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget_dir.exists():
        for exe in winget_dir.glob("Gyan.FFmpeg*/**/ffmpeg.exe"):
            return str(exe)

    raise RuntimeError("ffmpeg executable not found")


def _get_xtts():
    global _XTTS
    if _XTTS is not None:
        return _XTTS

    _patch_coqui_scarf_to_huggingface()

    try:
        import torch
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError(
            "XTTS is not installed. Run: pip install TTS==0.22.0"
        ) from exc

    model = TTS(settings.xtts_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _XTTS = model.to(device)
    return _XTTS


def _tts_to_wav(text: str, wav_path: Path) -> None:
    reference_wav = Path(settings.reference_voice_path)
    if not reference_wav.exists():
        raise RuntimeError(f"Reference voice WAV not found: {reference_wav}")

    tts = _get_xtts()
    tts.tts_to_file(
        text=text,
        file_path=str(wav_path),
        speaker_wav=str(reference_wav),
        language=settings.xtts_language,
    )


def _to_telegram_voice(wav_path: Path, ogg_path: Path) -> None:
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-c:a",
        "libopus",
        "-b:a",
        "64k",
        str(ogg_path),
    ]
    subprocess.run(cmd, check=True)


def generate_voice_reply(text: str, user_id: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"u{user_id}_{uuid.uuid4().hex[:8]}"
    wav_path = OUT_DIR / f"{base}.wav"
    ogg_path = OUT_DIR / f"{base}.ogg"

    _tts_to_wav(text, wav_path)
    _to_telegram_voice(wav_path, ogg_path)
    return ogg_path


def _run_wav2lip(face_video: Path, wav_audio: Path, out_video: Path) -> None:
    repo = Path(settings.wav2lip_repo_path)
    checkpoint = Path(settings.wav2lip_checkpoint_path)
    if not repo.exists() or not checkpoint.exists():
        raise RuntimeError(
            "Wav2Lip is not configured. Set WAV2LIP_REPO_PATH and WAV2LIP_CHECKPOINT_PATH in .env"
        )

    inference_py = repo / "inference.py"
    if not inference_py.exists():
        raise RuntimeError(f"Wav2Lip inference script not found: {inference_py}")

    cmd = [
        sys.executable,
        str(inference_py),
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(face_video),
        "--audio",
        str(wav_audio),
        "--outfile",
        str(out_video),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo))


def _postprocess_video_note(src: Path, dst: Path, pixel_mode: bool) -> None:
    ffmpeg = _find_ffmpeg()
    vf = "crop='min(iw,ih)':'min(iw,ih)',scale=640:640"
    if pixel_mode:
        vf = (
            "crop='min(iw,ih)':'min(iw,ih)',"
            "scale=96:96:flags=neighbor,"
            "scale=640:640:flags=neighbor,"
            "fps=12"
        )

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def generate_video_note(text: str, user_id: int, pixel_mode: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"u{user_id}_{uuid.uuid4().hex[:8]}"
    wav_path = OUT_DIR / f"{base}.wav"
    lip_path = OUT_DIR / f"{base}_lip.mp4"
    out_path = OUT_DIR / f"{base}.mp4"

    _tts_to_wav(text, wav_path)

    reference_video = Path(settings.reference_video_path)
    if not reference_video.exists():
        raise RuntimeError(f"Reference video not found: {reference_video}")

    _run_wav2lip(reference_video, wav_path, lip_path)
    _postprocess_video_note(lip_path, out_path, pixel_mode=pixel_mode)
    return out_path
