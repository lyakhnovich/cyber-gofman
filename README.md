# Cyber Gofman Avatar Bot (Stage 1/2 bridge)

Current stack:
- Retrieval-based text responses (offline)
- YouTube/text ingest to Qdrant
- Character voice generation via XTTS (local)
- Video note generation via Wav2Lip + reference video

## Setup

1. Create `.env` from template.
2. Fill Telegram token and paths.
3. Install Python deps:

```bash
python -m pip install -r app/requirements.txt
```

4. Install XTTS runtime (one-time):

```bash
python -m pip install TTS==0.22.0
```

## Required media

- `REFERENCE_VOICE_PATH` -> clean voice sample WAV (10-30s)
- `REFERENCE_VIDEO_PATH` -> square face reference video (`reference_circle.mp4`)

## Wav2Lip setup

Set in `.env`:
- `WAV2LIP_REPO_PATH` (local Wav2Lip repo)
- `WAV2LIP_CHECKPOINT_PATH` (path to `wav2lip_gan.pth` or similar)

## Bot modes

- `/mode text`
- `/mode voice` (XTTS voice clone)
- `/mode video` (XTTS audio + Wav2Lip video note)
- `/pixel on|off`

## Ingest

```bash
python -m app.scripts.ingest_youtube --url "https://www.youtube.com/watch?v=..." --max-per-url 2
python -m app.scripts.ingest_text --file dataset_site_source.txt --source-name site_text
```
