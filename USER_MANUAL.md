# YouTube Live Transcribe — User Manual

## What It Does

This tool transcribes any YouTube live stream in real time using
**Mistral Voxtral-Mini-4B-Realtime-2602**, an open-weights speech-to-text
model that runs entirely on your local machine. No audio is ever sent to
any external API. Everything happens on your GPU.

It captures the YouTube audio stream, splits it into small chunks, feeds
each chunk through the Voxtral model, and streams the transcript to a
web browser UI as fast as the model can process it.

---

## System Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| OS | Ubuntu 20.04+ / Windows 10+ | Ubuntu 24.04 |
| Python | 3.11 | 3.11.14 |
| GPU | NVIDIA GPU with 6 GB VRAM | RTX series |
| CUDA | 11.8+ | 12.6 |
| RAM | 8 GB | 16 GB+ |
| Disk | ~10 GB (model weights) | SSD recommended |

**CPU-only mode** works but is very slow (10–30s per 2s chunk).

---

## Installation

### Step 1 — System packages

```bash
sudo apt update
sudo apt install ffmpeg git curl
```

> `ffmpeg` is required to decode YouTube audio streams.

### Step 2 — Python environment

Create a Python 3.11 virtual environment (conda or venv):

```bash
# Option A: conda
conda create -n voxtralenv python=3.11
conda activate voxtralenv

# Option B: venv
python3.11 -m venv voxtralenv
source voxtralenv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you do **not** have a CUDA GPU, edit `requirements.txt` first:
> - Remove the `--extra-index-url` line
> - Change `torch==2.7.1` to just `torch`

### Step 4 — Download the model (first run only)

The model (~8 GB) downloads automatically on first run from HuggingFace.
Make sure you have a stable internet connection. Subsequent runs load from
the local cache instantly.

Default cache location:
- **Linux/Mac:** `~/.local/share/Antix Digital/AICS Service/model_cache`
- **Windows:** `%PROGRAMDATA%\Antix Digital\AICS Service\model_cache`

---

## Running the App

### Web UI (recommended)

```bash
python YouTubeLiveTranscribeUI.py
```

Then open your browser at: **http://localhost:7860**

The UI loads and shows "Loading model…" in the top-right corner.
Wait for it to change to **"System Ready"** before starting.

### Command Line (no UI)

```bash
python YouTubeLiveTranscribe.py "https://www.youtube.com/live/VIDEO_ID"
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-seconds` | `2.0` | Audio chunk size in seconds. Smaller = lower latency, larger = more accurate |
| `--language` | `en` | Transcription language (en, fr, de, es, ar, zh, ja, ko, ru, hi, …) |
| `--silence-rms` | `200` | Skip chunks quieter than this RMS value (saves GPU) |
| `--model-cache` | auto | Custom path to store model weights |

Example:
```bash
python YouTubeLiveTranscribe.py "https://www.youtube.com/live/ABC123" \
    --chunk-seconds 3.0 \
    --language ar
```

---

## Using the Web UI

1. Paste a YouTube live stream URL into the input box
2. Adjust parameters if needed (see below)
3. Click **Transcribe**
4. The video thumbnail appears with a live indicator; transcription starts streaming on the right panel
5. Click **Stop** to end the session
6. Click **Export .txt** to save the transcript

### UI Parameters

| Parameter | What It Controls |
|-----------|-----------------|
| **Language** | Language the speaker is using |
| **Chunk Seconds** | How many seconds of audio per inference call. Lower (1–2s) gives faster output but may miss words at chunk boundaries. Higher (4–10s) is more accurate but adds latency. |
| **Silence Threshold (RMS)** | Chunks below this energy level are skipped. Increase if you see empty transcriptions during silence. Decrease if speech is being skipped. |

---

## How It Works (Technical)

```
YouTube URL
    │
    ▼
yt-dlp  ──────► resolves CDN stream URL
    │
    ▼
ffmpeg  ──────► decodes audio → 16kHz mono PCM chunks
    │
    ▼
Silence check  ── silent? → skip chunk
    │
    ▼
Overlap buffer ── prepends 0.3s of previous chunk
    │              (prevents boundary words being cut)
    ▼
Voxtral model ── runs on local GPU (CUDA)
    │
    ▼
SSE stream ───► browser (Flask Server-Sent Events)
    │
    ▼
Live transcript displayed in browser
```

The video in the UI is also proxied through the local Flask server
(to bypass YouTube's CORS restriction), re-encoded as fragmented MP4
by ffmpeg, and played by the browser's native `<video>` element.

---

## Files

| File | Purpose |
|------|---------|
| `YouTubeLiveTranscribeUI.py` | Flask web server — run this for the browser UI |
| `YouTubeLiveTranscribe.py` | Core logic — model loading, audio capture, transcription |
| `templates/index.html` | Browser UI (served by Flask) |
| `requirements.txt` | Python dependencies |

---

## Troubleshooting

**"Loading model…" never changes to "System Ready"**
- Check terminal output for error messages
- Make sure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

**"yt-dlp failed" error**
- Update yt-dlp: `pip install -U yt-dlp`
- Make sure the URL is a valid live stream (not a regular video or a stream that ended)

**Video not playing / stuck buffering**
- ffmpeg must be installed: `ffmpeg -version`
- The stream proxy adds latency; wait 10–15 seconds for the buffer to fill

**Transcription is empty or missing words**
- Lower the Silence Threshold (RMS) slider
- Increase Chunk Seconds for more context per inference

**Out of GPU memory**
- The model requires ~6 GB VRAM. Close other GPU applications.
- CPU fallback works but is slow.

**Permission denied on cache directory**
- Run: `sudo chown -R $USER:$USER ~/.cache/huggingface`
- Or run: `sudo chown -R $USER:$USER ~/.local/share`

---

## Supported YouTube URL Formats

```
https://www.youtube.com/live/VIDEO_ID
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
```

Any URL supported by yt-dlp will work (Twitch, etc. can also be tried).

---

## License & Model

The Voxtral-Mini-4B-Realtime-2602 model is provided by Mistral AI.
Check the [model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
for license terms before commercial use.
