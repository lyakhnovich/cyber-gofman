import os
import subprocess
import sys
import uuid
import json
from datetime import datetime
from pathlib import Path

from app.core.config import settings

OUT_DIR = Path("app/data/outputs")
WAV2LIP_LOG_DIR = OUT_DIR / "wav2lip_logs"
_XTTS = None
_COQUI_DOWNLOAD_PATCHED = False


def _youtube_id_from_text(text: str) -> str | None:
    import re

    m = re.search(r"([A-Za-z0-9_-]{11})", text or "")
    return m.group(1) if m else None


def _resolve_reference_video(selected: dict) -> Path:
    raw_value = str(selected.get("video", settings.reference_video_path))
    direct = Path(raw_value)
    if direct.exists():
        return direct

    # If exact path is absent, try matching by YouTube id in raw_videos filenames.
    yt_id = _youtube_id_from_text(raw_value) or _youtube_id_from_text(str(selected.get("id", "")))
    if yt_id:
        raw_dir = Path("app/data/raw_videos")
        for candidate in raw_dir.glob(f"*{yt_id}*.mp4"):
            return candidate

    return direct


def load_video_profiles() -> list[dict]:
    path = Path(settings.reference_profiles_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        profiles = payload.get("profiles", [])
        if isinstance(profiles, list):
            return [p for p in profiles if isinstance(p, dict)]
    return []


def _square_crop_filter(crop_anchor: str | float | int | None) -> str:
    if isinstance(crop_anchor, (int, float)):
        ratio = min(1.0, max(0.0, float(crop_anchor)))
        x_expr = f"(iw-min(iw\\,ih))*{ratio:.3f}"
    else:
        anchor = str(crop_anchor or "left").lower()
        if anchor == "center":
            x_expr = "(iw-min(iw\\,ih))/2"
        elif anchor == "right":
            x_expr = "(iw-min(iw\\,ih))"
        else:
            x_expr = "0"
    return f"crop='min(iw\\,ih)':'min(iw\\,ih)':{x_expr}:(ih-min(iw\\,ih))/2"


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
    orig_tos_agreed = tts_manage.ModelManager.tos_agreed
    orig_ask_tos = tts_manage.ModelManager.ask_tos

    def _download_model_files_wrapped(file_urls, output_folder, progress_bar):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        mapped = [_hf_coqui_resolve_url(u) for u in file_urls]
        try:
            from huggingface_hub import constants as hf_constants
            from huggingface_hub import hf_hub_download
        except ImportError:
            return orig(mapped, output_folder, progress_bar)
        hf_constants.HF_HUB_DISABLE_XET = True

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
            )

    def _tos_agreed_wrapped(model_item, model_full_path):
        if os.environ.get("COQUI_TOS_AGREED") == "1":
            return True
        return orig_tos_agreed(model_item, model_full_path)

    def _ask_tos_wrapped(model_full_path):
        if os.environ.get("COQUI_TOS_AGREED") == "1":
            tos_path = Path(model_full_path) / "tos_agreed.txt"
            tos_path.write_text(
                "I have read, understood and agreed to the Terms and Conditions.",
                encoding="utf-8",
            )
            return True
        return orig_ask_tos(model_full_path)

    tts_manage.ModelManager._download_model_files = staticmethod(_download_model_files_wrapped)
    tts_manage.ModelManager.tos_agreed = staticmethod(_tos_agreed_wrapped)
    tts_manage.ModelManager.ask_tos = staticmethod(_ask_tos_wrapped)
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


def _find_ffprobe() -> str:
    ffmpeg = _find_ffmpeg()
    ffprobe_candidate = str(Path(ffmpeg).with_name("ffprobe.exe"))
    if Path(ffprobe_candidate).exists():
        return ffprobe_candidate
    return "ffprobe"


def _audio_duration_seconds(audio_path: Path) -> float:
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout or "{}")
    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    return max(0.1, duration)


def _get_xtts():
    global _XTTS
    if _XTTS is not None:
        return _XTTS

    _patch_coqui_scarf_to_huggingface()
    os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

    try:
        import torch
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError(
            "XTTS is not installed. Run: pip install TTS==0.22.0"
        ) from exc

    # torchaudio 2.9+ routes load() through TorchCodec, which can fail on Windows
    # due to DLL/version issues. XTTS only needs WAV loading for speaker reference,
    # so patch torchaudio.load with a stable soundfile-based implementation.
    try:
        import soundfile as sf
        import torchaudio
        import numpy as np
    except ImportError:
        sf = None
        torchaudio = None
        np = None
    if torchaudio is not None and sf is not None and not getattr(torchaudio, "_xtts_soundfile_patch", False):
        _orig_torchaudio_load = torchaudio.load

        def _torchaudio_load_soundfile(path, *args, **kwargs):
            if kwargs.get("format") not in (None, "wav"):
                return _orig_torchaudio_load(path, *args, **kwargs)
            frame_offset = int(kwargs.get("frame_offset", 0))
            num_frames = int(kwargs.get("num_frames", -1))
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            if frame_offset > 0:
                audio = audio[frame_offset:]
            if num_frames > 0:
                audio = audio[:num_frames]
            tensor = torch.from_numpy(np.asarray(audio).T)
            return tensor, sr

        torchaudio.load = _torchaudio_load_soundfile
        torchaudio._xtts_soundfile_patch = True

    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # XTTS checkpoints include full config objects and fail to load with that default.
    if not getattr(torch.load, "_xtts_weights_compat", False):
        _orig_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)

        _torch_load_compat._xtts_weights_compat = True
        torch.load = _torch_load_compat

    model = TTS(settings.xtts_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _XTTS = model.to(device)
    return _XTTS


def _xtts_max_input_tokens(tts) -> int:
    """XTTS asserts len(token_ids) < gpt_max_text_tokens (default 402)."""
    try:
        tm = tts.synthesizer.tts_model
        return max(1, int(getattr(tm.args, "gpt_max_text_tokens", 402)) - 1)
    except Exception:
        return 399


def _xtts_num_tokens(tts, fragment: str, language: str) -> int:
    lang = (language or "en").split("-")[0]
    tm = tts.synthesizer.tts_model
    return len(tm.tokenizer.encode(fragment, lang=lang))


def _truncate_xtts_fragment(tts, fragment: str, language: str) -> str:
    """One synthesis segment must stay under XTTS token cap (inference lowercases per sentence)."""
    fragment = fragment.strip()
    if not fragment:
        return fragment
    max_tok = _xtts_max_input_tokens(tts)
    if _xtts_num_tokens(tts, fragment, language) <= max_tok:
        return fragment
    lo, hi = 1, len(fragment)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = fragment[:mid].strip()
        if not cand:
            hi = mid - 1
            continue
        if _xtts_num_tokens(tts, cand, language) <= max_tok:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    if not best:
        return fragment[:80].rstrip() + "…"
    trimmed = best.rstrip(" ,;:")
    return trimmed + ("…" if trimmed != fragment else "")


def _clip_text_for_xtts(tts, text: str) -> str:
    """Shorten RAG/LLM replies so XTTS never hits the 400-token wall on a single segment."""
    text = (text or "").strip()
    if not text:
        return text
    language = settings.xtts_language
    try:
        sentences = tts.synthesizer.split_into_sentences(text)
    except Exception:
        sentences = [text]
    if not sentences:
        return text
    parts: list[str] = []
    for sent in sentences:
        s = sent.strip()
        if s:
            parts.append(_truncate_xtts_fragment(tts, s, language))
    return " ".join(p for p in parts if p).strip()


def _tts_to_wav(text: str, wav_path: Path) -> None:
    reference_wav = Path(settings.reference_voice_path)
    if not reference_wav.exists():
        raise RuntimeError(f"Reference voice WAV not found: {reference_wav}")

    try:
        tts = _get_xtts()
        text = _clip_text_for_xtts(tts, text)
        tts.tts_to_file(
            text=text,
            file_path=str(wav_path),
            speaker_wav=str(reference_wav),
            language=settings.xtts_language,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "torchcodec" not in msg and "libtorchcodec" not in msg:
            raise
        # Emergency fallback: keep bot usable when local TorchCodec stack is broken.
        try:
            import pyttsx3
        except ImportError as import_exc:
            raise RuntimeError("XTTS failed and pyttsx3 fallback is unavailable") from import_exc
        engine = pyttsx3.init()
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()


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


def _run_wav2lip(face_video: Path, wav_audio: Path, out_video: Path, profile: dict | None = None) -> None:
    repo = Path(settings.wav2lip_repo_path)
    checkpoint = Path(settings.wav2lip_checkpoint_path)
    if not repo.exists() or not checkpoint.exists():
        raise RuntimeError(
            "Wav2Lip is not configured. Set WAV2LIP_REPO_PATH and WAV2LIP_CHECKPOINT_PATH in .env"
        )

    inference_py = repo / "inference.py"
    if not inference_py.exists():
        raise RuntimeError(f"Wav2Lip inference script not found: {inference_py}")

    face_video = face_video.resolve()
    wav_audio = wav_audio.resolve()
    out_video = out_video.resolve()

    base_cmd = [
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
        "--pads",
        "0",
        "20",
        "0",
        "0",
        "--nosmooth",
    ]
    box = (profile or {}).get("box")
    if isinstance(box, list) and len(box) == 4:
        base_cmd += ["--box", *(str(int(v)) for v in box)]
    env = os.environ.copy()
    ffmpeg = _find_ffmpeg()
    ffmpeg_bin = str(Path(ffmpeg).parent)
    env["PATH"] = ffmpeg_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")
    def _run_cmd_with_log(cmd: list[str], attempt_name: str) -> None:
        proc = subprocess.run(
            cmd,
            check=False,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            return

        WAV2LIP_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = WAV2LIP_LOG_DIR / f"wav2lip_{stamp}_{uuid.uuid4().hex[:8]}.log"
        log_path.write_text(
            "\n".join(
                [
                    f"attempt={attempt_name}",
                    f"returncode={proc.returncode}",
                    f"cwd={repo}",
                    "command=" + " ".join(cmd),
                    "",
                    "----- stdout -----",
                    proc.stdout or "<empty>",
                    "",
                    "----- stderr -----",
                    proc.stderr or "<empty>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        # Keep full primary traceback visible in terminal logs as requested.
        print(
            f"[Wav2Lip:{attempt_name}] failed with code {proc.returncode}. "
            f"Full log: {log_path}",
            file=sys.stderr,
        )
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Wav2Lip {attempt_name} failed (code {proc.returncode}). "
            f"See log: {log_path}"
        )

    try:
        _run_cmd_with_log(base_cmd, "default")
    except RuntimeError:
        # Retry with static face for unstable frame-by-frame detection on noisy clips.
        static_cmd = base_cmd + ["--static", "True"]
        _run_cmd_with_log(static_cmd, "static")


def _postprocess_video_note(src: Path, dst: Path, pixel_mode: bool) -> None:
    ffmpeg = _find_ffmpeg()
    vf = "crop='min(iw\\,ih)':'min(iw\\,ih)':0:(ih-min(iw\\,ih))/2,scale=640:640,setsar=1"
    if pixel_mode:
        vf = (
            "crop='min(iw\\,ih)':'min(iw\\,ih)':0:(ih-min(iw\\,ih))/2,"
            "scale=96:96:flags=neighbor,"
            "scale=640:640:flags=neighbor,"
            "fps=12,"
            "setsar=1"
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


def _compose_fallback_video_with_audio(face_video: Path, wav_audio: Path, out_video: Path, profile: dict | None = None) -> None:
    """Fallback when Wav2Lip stack fails: use short live clip or still + generated audio."""
    ffmpeg = _find_ffmpeg()
    still_path = OUT_DIR / f"{out_video.stem}_still.jpg"
    motion_path = OUT_DIR / f"{out_video.stem}_motion.mp4"
    custom_still = Path(settings.reference_still_path) if settings.reference_still_path else None
    crop_filter = _square_crop_filter((profile or {}).get("crop_anchor"))
    use_motion_clip = not (custom_still and custom_still.exists())
    audio_seconds = _audio_duration_seconds(wav_audio)
    if custom_still and custom_still.exists():
        still_input = custom_still
    else:
        start_cfg = (profile or {}).get("start")
        end_cfg = (profile or {}).get("end")
        if start_cfg is not None and end_cfg is not None and float(end_cfg) > float(start_cfg):
            clip_start = max(0.0, float(start_cfg))
            clip_seconds = max(0.8, min(audio_seconds, float(end_cfg) - float(start_cfg)))
        else:
            still_second = max(0.0, float((profile or {}).get("still_second", settings.fallback_still_second)))
            clip_seconds = max(0.8, audio_seconds)
            clip_start = max(0.0, still_second - clip_seconds / 2)
        extract_motion_cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{clip_start:.2f}",
            "-t",
            f"{clip_seconds:.2f}",
            "-i",
            str(face_video),
            "-an",
            "-vf",
            f"{crop_filter},scale=640:640,setsar=1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            str(motion_path),
        ]
        subprocess.run(extract_motion_cmd, check=True)

    cmd = [
        ffmpeg,
        "-y",
        *([] if use_motion_clip else ["-stream_loop", "-1"]),
        "-i",
        str(motion_path if use_motion_clip else still_input),
        "-i",
        str(wav_audio),
        "-shortest",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-vf",
        "setsar=1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_video),
    ]
    subprocess.run(cmd, check=True)


def _prepare_wav2lip_face_video(reference_video: Path, profile: dict, wav_audio: Path, out_face: Path) -> Path:
    ffmpeg = _find_ffmpeg()
    start = max(0.0, float(profile.get("start", 0.0) or 0.0))
    end_raw = profile.get("end")
    clip_max = None
    if end_raw is not None:
        try:
            end = float(end_raw)
            if end > start:
                clip_max = end - start
        except Exception:
            clip_max = None

    audio_seconds = _audio_duration_seconds(wav_audio)
    clip_seconds = audio_seconds if clip_max is None else min(audio_seconds, clip_max)
    clip_seconds = max(0.8, clip_seconds)

    # Keep full frame for detection stability; framing is applied in postprocess.
    vf = "scale=640:-2,setsar=1"
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{clip_seconds:.2f}",
        "-i",
        str(reference_video),
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        str(out_face),
    ]
    subprocess.run(cmd, check=True)
    return out_face


def generate_video_note(text: str, user_id: int, pixel_mode: bool = False, profile: dict | None = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"u{user_id}_{uuid.uuid4().hex[:8]}"
    wav_path = OUT_DIR / f"{base}.wav"
    face_path = OUT_DIR / f"{base}_face.mp4"
    lip_path = OUT_DIR / f"{base}_lip.mp4"
    out_path = OUT_DIR / f"{base}.mp4"

    _tts_to_wav(text, wav_path)

    selected = profile or {}
    reference_video = _resolve_reference_video(selected)
    if not reference_video.exists():
        raise RuntimeError(f"Reference video not found: {reference_video}")
    prepared_face = _prepare_wav2lip_face_video(reference_video, selected, wav_path, face_path)

    try:
        _run_wav2lip(prepared_face, wav_path, lip_path, profile=selected)
    except Exception:
        # Keep feature usable even if Wav2Lip/codec stack fails on a specific clip.
        _compose_fallback_video_with_audio(reference_video, wav_path, lip_path, profile=selected)
    _postprocess_video_note(lip_path, out_path, pixel_mode=pixel_mode)
    return out_path
