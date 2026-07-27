$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv311\Scripts\python.exe"
$app = Join-Path $projectRoot "YouTubeLiveTranscribeUI.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Compatible environment not found at $python"
}

Set-Location -LiteralPath $projectRoot

try {
    $status = Invoke-RestMethod -Uri "http://127.0.0.1:7860/status" -TimeoutSec 2
    Write-Host "YouTube Live Transcribe is already running at http://localhost:7860"
    Write-Host "Status: $($status.status)"
    exit 0
} catch {
    # No healthy instance is answering; the Python preflight performs the
    # authoritative port check before allocating CUDA memory.
}

& $python $app
