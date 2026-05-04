# Cyber Gofman Bot (Stage 1/2 bridge)

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

- `/mode text` (default after `/start`)
- `/mode voice` (XTTS voice clone)
- `/mode video` (XTTS audio + Wav2Lip video note)
- `/pixel on|off`

## Ingest

```bash
python -m app.scripts.ingest_youtube --url "https://www.youtube.com/watch?v=..." --max-per-url 2
python -m app.scripts.ingest_text --file dataset_site_source.txt --source-name site_text
```

## LoRA / SFT training

Dataset files produced in this project:
- `app/data/datasets/igor_sft_train.jsonl`
- `app/data/datasets/igor_sft_val.jsonl`

Install training deps:

```bash
python -m pip install -r app/requirements.txt
```

Run LoRA SFT (RTX 3060 friendly defaults):

```bash
python -m app.scripts.train_lora_sft \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --train-file "app/data/datasets/igor_sft_train.jsonl" \
  --val-file "app/data/datasets/igor_sft_val.jsonl" \
  --out-dir "app/data/models/igor-lora"
```

Result adapter is saved to:
- `app/data/models/igor-lora/adapter`

## Local LLM in the bot (QLoRA adapter)

With `LOCAL_LLM_ENABLED=true` (see `.env.example`), the bot builds the same `### Instruction` / `### Response` prompt as in training, fills **Instruction** with RAG snippets + user message, and generates the reply with **Qwen2.5-7B-Instruct (4-bit) + your LoRA**. If CUDA is missing or generation fails, it falls back to the previous offline RAG path.

Paths:

- `LOCAL_LLM_BASE_MODEL` — Hugging Face id of the base model (must match training).
- `LOCAL_LLM_ADAPTER_PATH` — folder with `adapter_model.safetensors` and tokenizer (default: `app/data/models/igor-lora/adapter`).
