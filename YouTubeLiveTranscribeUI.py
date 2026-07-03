#!/usr/bin/env python3
"""
YouTubeLiveTranscribeUI.py — Flask web UI for real-time YouTube live stream
transcription using mistralai/Voxtral-Mini-4B-Realtime-2602.

Usage:
    python YouTubeLiveTranscribeUI.py
    Then open http://localhost:7860 in your browser.
"""

import json
import queue
import sys
import subprocess
import platform
import threading
import time

import psutil
import torch
import logging
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from YouTubeLiveTranscribe import (
    CHANNELS,
    DEFAULT_CACHE_DIR,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    TAIL_SECONDS,
    get_audio_stream_url,
    is_silent,
    load_voxtral_model,
    pcm_to_wav,
    stream_audio,
    strip_overlap,
    transcribe_wav_bytes,
)

app = Flask(__name__)

# Hide noisy per-request logs (e.g. /metrics polling). We'll print our own high-signal messages instead.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ── Global model state ────────────────────────────────────────────────────────
_model = _processor = _device = None
_model_ready  = threading.Event()
_model_error: str | None = None
_model_loading_state: dict = {
    "attempt": 0,
    "max_attempts": 8,
    "last_error": None,
    "next_retry_in_s": None,
}
_current_model_id: str | None = "mistralai/Voxtral-Mini-4B-Realtime-2602"

from collections import deque

_log_buf: deque[str] = deque(maxlen=4000)


class _LogTee:
    def __init__(self, original):
        self._original = original

    def write(self, s):
        try:
            self._original.write(s)
        except Exception:
            pass
        try:
            # Normalize carriage returns from progress bars and keep only useful lines.
            text = str(s).replace("\r", "\n")
            for line in text.splitlines():
                line = line.strip("\n")
                if not line.strip():
                    continue
                # Drop very noisy tqdm frames; keep the final 100% line.
                if "Loading weights:" in line and "100%|" not in line:
                    continue
                _log_buf.append(line)
        except Exception:
            pass

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass


# Mirror backend prints into an in-memory console for the UI
sys.stdout = _LogTee(sys.stdout)
sys.stderr = _LogTee(sys.stderr)

# ── Active session state ──────────────────────────────────────────────────────
_session_stop   = threading.Event()
_session_active = False
_transcript_q: queue.Queue = queue.Queue(maxsize=500)

_SYSTEM_SPECS = {
    "os": platform.platform(),
    "python_version": sys.version.split()[0],
    "cpu_model": platform.processor() or platform.uname().machine,
    "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda or None,
    "device_type": "cuda" if torch.cuda.is_available() else "cpu",
}

if torch.cuda.is_available():
    try:
        _SYSTEM_SPECS["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        _SYSTEM_SPECS["gpu_name"] = "Unknown"
    try:
        _SYSTEM_SPECS["gpu_count"] = torch.cuda.device_count()
    except Exception:
        _SYSTEM_SPECS["gpu_count"] = 1


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (background at startup)
# ─────────────────────────────────────────────────────────────────────────────
def _load_model_bg():
    global _model, _processor, _device, _model_error, _model_loading_state

    # Retry transient HuggingFace/CDN failures (e.g. 504) with backoff.
    max_attempts = int(_model_loading_state.get("max_attempts", 8) or 8)
    base_sleep_s = 5

    for attempt in range(1, max_attempts + 1):
        _model_loading_state["attempt"] = attempt
        _model_loading_state["next_retry_in_s"] = None
        try:
            _model, _processor, _device = load_voxtral_model(DEFAULT_CACHE_DIR, model_id=_current_model_id or None)
            _model_error = None
            _model_ready.set()
            return
        except Exception as exc:
            msg = str(exc)
            _model_loading_state["last_error"] = msg

            # If it's the last attempt, surface the error and stop.
            if attempt >= max_attempts:
                _model_error = msg
                _model_ready.set()
                return

            # Exponential backoff with a small cap.
            sleep_s = min(60, base_sleep_s * (2 ** (attempt - 1)))
            _model_loading_state["next_retry_in_s"] = sleep_s
            time.sleep(sleep_s)


threading.Thread(target=_load_model_bg, daemon=True, name="ModelLoader").start()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    if not _model_ready.is_set():
        return jsonify({"status": "loading", "loading": _model_loading_state})
    if _model_error:
        return jsonify({"status": "error", "message": _model_error, "loading": _model_loading_state})
    return jsonify({"status": "ready", "session_active": _session_active, "model_id": _current_model_id})


@app.route("/models")
def models():
    """
    Simple local model catalog for UI selection. Requirements are estimates.
    """
    return jsonify(
        {
            "current_model_id": _current_model_id,
            "models": [
                {
                    "id": "mistralai/Voxtral-Mini-3B-Realtime-2602",
                    "label": "Voxtral Mini 3B Realtime (26.02)",
                    "recommended_vram_gb_bf16": 5,
                    "notes": "Smaller/faster; less VRAM than 4B (if available in your environment).",
                },
                {
                    "id": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                    "label": "Voxtral Mini 4B Realtime (26.02)",
                    "recommended_vram_gb_bf16": 6,
                    "notes": "Fast, low-latency realtime transcription.",
                },
                {
                    "id": "mistralai/Voxtral-Small-24B-2507",
                    "label": "Voxtral Small 24B (25.07)",
                    "recommended_vram_gb_bf16": 55,
                    "notes": "Much larger; may not fit on most GPUs.",
                },
            ],
        }
    )


@app.route("/load_model", methods=["POST"])
def load_model():
    global _current_model_id, _model_error, _model, _processor, _device
    data = request.get_json(force=True)
    model_id = (data.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "No model_id provided"}), 400

    # If already loaded, no-op.
    if _current_model_id == model_id and _model_ready.is_set() and not _model_error:
        return jsonify({"ok": True, "status": "ready", "model_id": _current_model_id})

    # Reset state and start background loader
    _current_model_id = model_id
    print(f"[ui] Loading model requested: {_current_model_id}")
    _model_error = None
    _model = _processor = _device = None
    _model_ready.clear()
    _model_loading_state["attempt"] = 0
    _model_loading_state["last_error"] = None
    _model_loading_state["next_retry_in_s"] = None

    threading.Thread(target=_load_model_bg, daemon=True, name="ModelLoader").start()
    return jsonify({"ok": True, "status": "loading", "model_id": _current_model_id})


@app.route("/proxy_video")
def proxy_video():
    """
    Proxy the YouTube live stream through ffmpeg → fragmented MP4 → browser.
    Bypasses YouTube CDN CORS restrictions.
    """
    yt_url = request.args.get("url", "").strip()
    if not yt_url:
        return "No URL", 400

    # Resolve direct stream URL
    res = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "-f", "best", "--get-url", yt_url],
        capture_output=True, text=True, timeout=30,
    )
    lines = res.stdout.strip().splitlines()
    if not lines or not lines[0]:
        return "Could not resolve stream URL", 500
    stream_url = lines[0]

    # ffmpeg reads the stream and outputs fragmented MP4 to stdout
    ffmpeg_cmd = [
        "ffmpeg",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", stream_url,
        "-c:v", "copy", "-c:a", "aac",
        "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-loglevel", "quiet",
        "pipe:1",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def generate():
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.kill()
            proc.wait()

    return Response(
        stream_with_context(generate()),
        mimetype="video/mp4",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/start", methods=["POST"])
def start():
    global _session_active, _current_model_id

    if not _model_ready.is_set():
        return jsonify({"error": "Model is still loading — please wait"}), 503
    if _model_error:
        return jsonify({"error": _model_error}), 500
    if _session_active:
        return jsonify({"error": "A session is already active"}), 400

    data           = request.get_json(force=True)
    url            = data.get("url", "").strip()
    chunk_seconds  = float(data.get("chunk_seconds", 2.0))
    language       = data.get("language", "en").strip()
    silence_rms    = float(data.get("silence_rms", 200.0))
    max_new_tokens = int(data.get("max_new_tokens", 256))
    temperature    = float(data.get("temperature", 0.0))
    tail_seconds   = float(data.get("tail_seconds", TAIL_SECONDS))
    target_delay_ms = data.get("target_delay_ms", None)
    target_delay_ms = int(target_delay_ms) if target_delay_ms is not None and str(target_delay_ms).strip() != "" else None
    requested_model_id = (data.get("model_id") or _current_model_id or "").strip()

    print(
        "[ui] Start session params: "
        f"model={requested_model_id or _current_model_id} "
        f"lang={language} chunk={chunk_seconds}s rms={silence_rms} "
        f"max_new_tokens={max_new_tokens} temp={temperature} "
        f"tail={tail_seconds}s target_delay_ms={target_delay_ms}"
    )

    # If UI requests a different model, kick off loading and ask user to wait.
    if requested_model_id and requested_model_id != _current_model_id:
        _transcript_q.put({"type": "error", "message": f"Loading model {requested_model_id}…"})
        # fire-and-forget model load
        _current_model_id = requested_model_id
        _model_ready.clear()
        threading.Thread(target=_load_model_bg, daemon=True, name="ModelLoader").start()
        return jsonify({"error": "Selected model is loading — please wait"}), 503

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Flush old messages
    while not _transcript_q.empty():
        try:
            _transcript_q.get_nowait()
        except queue.Empty:
            break

    _session_stop.clear()
    _session_active = True

    threading.Thread(
        target=_run_session,
        args=(url, chunk_seconds, language, silence_rms, max_new_tokens, temperature, tail_seconds, target_delay_ms),
        daemon=True,
        name="TranscribeSession",
    ).start()

    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    _session_stop.set()
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — pushes transcript chunks to the browser."""
    def _generate():
        while True:
            try:
                msg = _transcript_q.get(timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") == "stopped":
                    break
            except queue.Empty:
                # keepalive ping
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/console_stream")
def console_stream():
    """SSE console stream for frontend log viewer."""
    since = int(request.args.get("since", 0) or 0)

    def _gen():
        last_idx = since
        while True:
            try:
                buf = list(_log_buf)
                # If client index is out of range (e.g. server restarted but browser kept old index),
                # rewind so the user still sees recent startup/model logs.
                if last_idx > len(buf):
                    last_idx = max(len(buf) - 200, 0)
                if last_idx < 0:
                    last_idx = 0
                if last_idx < len(buf):
                    for i in range(last_idx, len(buf)):
                        yield f"data: {json.dumps({'i': i + 1, 'line': buf[i]})}\n\n"
                    last_idx = len(buf)
                else:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                time.sleep(0.5)
            except GeneratorExit:
                break
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                time.sleep(1)

    return Response(_gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.route("/metrics")
def metrics():
    """
    Realtime system metrics for the dashboard.

    Returned fields:
      - cpu_percent (0-100)
      - ram_used_gb / ram_total_gb / ram_percent
      - vram_* (if CUDA available)
      - system_specs (static, filled once at import time)
    """
    vm = psutil.virtual_memory()
    # `interval=None` can return 0.0 on first call; use a tiny interval for a stable reading.
    cpu_percent = psutil.cpu_percent(interval=0.05)

    resp = {
        "cpu_percent": cpu_percent,
        "ram_used_gb": round((vm.total - vm.available) / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_percent": vm.percent,
        "system_specs": _SYSTEM_SPECS,
        "vram": None,
    }

    if torch.cuda.is_available():
        device_idx = 0
        try:
            device_idx = torch.cuda.current_device()
        except Exception:
            device_idx = 0

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            allocated_bytes = torch.cuda.memory_allocated(device_idx)
            reserved_bytes = torch.cuda.memory_reserved(device_idx)
            peak_bytes = torch.cuda.max_memory_allocated(device_idx)

            resp["vram"] = {
                "gpu_device_index": device_idx,
                "vram_free_gb": round(free_bytes / (1024**3), 2),
                "vram_total_gb": round(total_bytes / (1024**3), 2),
                "vram_allocated_gb": round(allocated_bytes / (1024**3), 2),
                "vram_reserved_gb": round(reserved_bytes / (1024**3), 2),
                "vram_peak_allocated_gb": round(peak_bytes / (1024**3), 2),
                # "used" based on reserved, so it better reflects caching/cudnn allocations.
                "vram_reserved_percent": round((reserved_bytes / total_bytes) * 100.0, 2)
                if total_bytes
                else None,
            }
        except Exception as exc:
            resp["vram"] = {"error": str(exc)}

    return jsonify(resp)


# ─────────────────────────────────────────────────────────────────────────────
# Session worker
# ─────────────────────────────────────────────────────────────────────────────
def _run_session(
    url: str,
    chunk_seconds: float,
    language: str,
    silence_rms: float,
    max_new_tokens: int,
    temperature: float,
    tail_seconds: float,
    target_delay_ms: int | None,
):
    global _session_active
    try:
        # Resolve YouTube stream URL
        try:
            stream_url = get_audio_stream_url(url)
        except Exception as exc:
            _transcript_q.put({"type": "error", "message": f"yt-dlp failed: {exc}"})
            return

        _transcript_q.put({"type": "started"})

        # Start audio capture
        audio_q = queue.Queue(maxsize=20)
        capture_thread = threading.Thread(
            target=stream_audio,
            args=(stream_url, chunk_seconds, audio_q, _session_stop),
            daemon=True,
        )
        capture_thread.start()

        tail_bytes      = int(SAMPLE_RATE * tail_seconds * SAMPLE_WIDTH * CHANNELS)
        prev_tail_pcm   = b""
        prev_tail_words = []
        chunk_num       = 0

        while not _session_stop.is_set():
            try:
                pcm_data = audio_q.get(timeout=5)
            except queue.Empty:
                if not capture_thread.is_alive():
                    break
                continue

            if is_silent(pcm_data, silence_rms):
                prev_tail_pcm   = b""
                prev_tail_words = []
                continue

            combined_pcm  = prev_tail_pcm + pcm_data
            prev_tail_pcm = pcm_data[-tail_bytes:]
            wav_bytes     = pcm_to_wav(combined_pcm)
            chunk_num    += 1
            t0            = time.time()

            try:
                text = transcribe_wav_bytes(
                    wav_bytes,
                    _model,
                    _processor,
                    _device,
                    language,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    target_delay_ms=target_delay_ms,
                )
            except Exception as exc:
                _transcript_q.put({"type": "error", "message": str(exc)})
                continue

            elapsed = time.time() - t0
            text    = strip_overlap(text, prev_tail_words)

            if text:
                prev_tail_words = text.split()[-3:]
                ts = time.strftime("%H:%M:%S")
                _transcript_q.put({
                    "type":    "transcript",
                    "timestamp": ts,
                    "text":    text,
                    "elapsed": round(elapsed, 2),
                })

    except Exception as exc:
        _transcript_q.put({"type": "error", "message": str(exc)})
    finally:
        _session_active = False
        _transcript_q.put({"type": "stopped"})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  YouTube Live Transcribe — Web UI")
    print("  Open http://localhost:7860 in your browser")
    print("  Model loading in background …")
    print("=" * 55)
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
