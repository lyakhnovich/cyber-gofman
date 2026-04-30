$ErrorActionPreference = "Stop"

Set-Location "D:\dev\projects\cyber-gofman"

# Put one URL per line into scripts\youtube_urls.txt
$urlsFile = "scripts\youtube_urls.txt"
if (-not (Test-Path $urlsFile)) {
    throw "Create $urlsFile with one YouTube URL per line."
}

# Tune these if needed:
$maxPerUrl = 1
$whisperModel = "medium"

.\.venv\Scripts\python.exe -m app.scripts.ingest_youtube `
  --urls-file $urlsFile `
  --max-per-url $maxPerUrl `
  --model $whisperModel
