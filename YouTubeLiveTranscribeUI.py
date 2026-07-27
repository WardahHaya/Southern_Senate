#!/usr/bin/env python3
"""
YouTubeLiveTranscribeUI.py — Flask web UI for real-time YouTube live stream
transcription using mistralai/Voxtral-Mini-4B-Realtime-2602.

Usage:
    python YouTubeLiveTranscribeUI.py
    Then open http://localhost:7860 in your browser.
"""

import json
import gc
import math
import queue
import sys
import subprocess
import platform
import threading
import time
import os
import shutil
import socket
import uuid
from collections import deque
from pathlib import Path

# Must be set before transformers/huggingface_hub is imported. Windows
# Application Control blocks hf_xet's native DLL on this workstation.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from whisper_live_stream import (
    WhisperCaptionBuffer,
    WhisperSpeakerUnit,
    remove_repeated_window_prefix,
    remove_timestamp_covered_words,
    split_unit_at_speaker_boundaries,
    transcribe_whisper_window,
)
import psutil
import torch
import logging
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from speaker_diarization import (
    DiarizationContext,
    LiveSpeakerTracker,
    LiveTurnCommitter,
    SpeakerRegistry,
    env_flag,
    load_hf_token_from_env_file,
)
from audio_spool import AudioSpool
from voxtral_live_stream import stream_voxtral_text

from YouTubeLiveTranscribe import (
    CHANNELS,
    DEFAULT_CACHE_DIR,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    TAIL_SECONDS,
    get_audio_stream_url,
    get_video_stream_url,
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

_hf_token = os.getenv("HF_TOKEN") or load_hf_token_from_env_file()
_diary = DiarizationContext(
    enabled=env_flag("DIARIZATION_ENABLED", default=bool(_hf_token)),
    hf_token=_hf_token,
)

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
# The browser explicitly requests its selected backend after connecting.
# Starting a default Voxtral load here races that request (usually Whisper),
# temporarily retaining both models and starving a 16 GB host of RAM.
_current_model_id: str | None = None
_model_switch_lock = threading.Lock()
FASTER_WHISPER_MODEL_ID = "faster-whisper/large-v3-turbo"
_dll_directory_handles = []


def _is_faster_whisper_model(model_id: str | None) -> bool:
    return bool(model_id and model_id.startswith("faster-whisper/"))


def _configure_faster_whisper_cuda_runtime() -> None:
    """Expose pip-installed CUDA 12 libraries to CTranslate2 on Windows."""
    if os.name != "nt":
        return
    site_packages = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    candidates = (
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
        site_packages / "ctranslate2",
    )
    current_path = os.environ.get("PATH", "")
    for directory in candidates:
        if not directory.is_dir():
            continue
        directory_text = str(directory)
        if directory_text.lower() not in current_path.lower():
            current_path = f"{directory_text};{current_path}"
        # Keep each handle alive for the lifetime of the process. Releasing an
        # add_dll_directory handle removes the directory from DLL resolution.
        if hasattr(os, "add_dll_directory"):
            _dll_directory_handles.append(os.add_dll_directory(directory_text))
    os.environ["PATH"] = current_path

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
# Never block audio/ASR because a browser tab is refreshing or briefly
# disconnected. Transcript events are small and must remain lossless.
_transcript_q: queue.Queue = queue.Queue()
_live_telemetry = {
    "last_audio_received": None,
    "source_ended": False,
    "processing": False,
    "backlog_seconds": 0.0,
    "last_inference_seconds": 0.0,
    "last_audio_seconds": 0.0,
    "realtime_factor": 0.0,
    "average_realtime_factor": 0.0,
    "processed_audio_seconds": 0.0,
    "inference_seconds": 0.0,
    "diarization_seconds": 0.0,
    "diarization_passes": 0,
    "last_diarization_seconds": 0.0,
    "diarization_speaker_count": 0,
    "speaker_update_count": 0,
}

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
def _release_loaded_model():
    """Release model tensors before loading a replacement on the same GPU."""
    global _model, _processor, _device
    old_model = _model
    _model = _processor = _device = None
    if old_model is not None:
        # faster-whisper wraps a CTranslate2 Whisper instance whose CUDA
        # allocator is independent of torch.cuda.empty_cache(). Explicitly
        # unload it before switching back to a Voxtral/PyTorch backend.
        ctranslate_model = getattr(old_model, "model", None)
        unload_ctranslate = getattr(ctranslate_model, "unload_model", None)
        if callable(unload_ctranslate):
            try:
                unload_ctranslate()
            except Exception as exc:
                print(f"[model] CTranslate2 unload warning: {exc}")
        del old_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _cuda_allocated_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        # CTranslate2 allocates outside PyTorch's caching allocator. For the
        # faster-whisper backend, device-used memory is the only meaningful
        # figure available from this process.
        if _is_faster_whisper_model(_current_model_id):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return round((total_bytes - free_bytes) / (1024 ** 3), 2)
        return round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
    except Exception:
        return 0.0


def _load_model_bg(model_id: str | None = None):
    global _model, _processor, _device, _model_error, _model_loading_state

    requested_model_id = model_id or _current_model_id
    with _model_switch_lock:
        # Retry transient HuggingFace/CDN failures (e.g. 504) with backoff.
        max_attempts = int(_model_loading_state.get("max_attempts", 8) or 8)
        base_sleep_s = 5

        for attempt in range(1, max_attempts + 1):
            if requested_model_id != _current_model_id:
                return
            _model_loading_state["attempt"] = attempt
            _model_loading_state["next_retry_in_s"] = None
            try:
                if _is_faster_whisper_model(requested_model_id):
                    _configure_faster_whisper_cuda_runtime()
                    from faster_whisper import WhisperModel

                    whisper_name = requested_model_id.split("/", 1)[1]
                    print(
                        f"[model] Loading faster-whisper {whisper_name} "
                        "(Stable Broadcast, int8_float16)"
                    )
                    loaded_model = WhisperModel(
                        whisper_name,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        compute_type=(
                            "int8_float16"
                            if torch.cuda.is_available()
                            else "int8"
                        ),
                        download_root=str(
                            Path(DEFAULT_CACHE_DIR) / "faster_whisper"
                        ),
                    )
                    loaded_processor = None
                    loaded_device = (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    print("[model] faster-whisper Ready")
                else:
                    loaded_model, loaded_processor, loaded_device = load_voxtral_model(
                        DEFAULT_CACHE_DIR,
                        model_id=requested_model_id or None,
                    )
                # A different model may have been selected while this slow
                # load was in progress. Never let a stale loader overwrite
                # the selected backend or mark it ready under the wrong name.
                if requested_model_id != _current_model_id:
                    del loaded_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return
                _model, _processor, _device = (
                    loaded_model,
                    loaded_processor,
                    loaded_device,
                )
                _model_error = None
                _model_ready.set()
                return
            except Exception as exc:
                if requested_model_id != _current_model_id:
                    return
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


def _begin_model_load(model_id: str) -> None:
    """Start one memory-safe asynchronous model replacement."""
    global _current_model_id, _model_error
    _current_model_id = model_id
    _model_error = None
    _model_ready.clear()
    _model_loading_state["attempt"] = 0
    _model_loading_state["last_error"] = None
    _model_loading_state["next_retry_in_s"] = None
    _release_loaded_model()
    threading.Thread(
        target=_load_model_bg,
        args=(model_id,),
        daemon=True,
        name="ModelLoader",
    ).start()


def _preload_diarization_bg():
    """Warm speaker models before the first live audio reaches the pipeline."""
    if not _diary.enabled:
        return
    _model_ready.wait()
    if _model_error:
        return
    print("[diarization] Preloading speaker detection model ...")
    pipeline = _diary.initialize()
    if pipeline is None:
        print(f"[diarization] Preload failed: {_diary.pipeline_error}")
    else:
        print("[diarization] Ready")


_background_services_started = False


def _start_background_services() -> None:
    """Start model services once, only after the UI port is known to be free."""
    global _background_services_started
    if _background_services_started:
        return
    _background_services_started = True
    if _current_model_id is not None:
        threading.Thread(
            target=_load_model_bg,
            args=(_current_model_id,),
            daemon=True,
            name="ModelLoader",
        ).start()
    threading.Thread(
        target=_preload_diarization_bg,
        daemon=True,
        name="DiarizationPreloader",
    ).start()


def _port_is_available(host: str, port: int) -> bool:
    """Refuse duplicate launches before either process allocates GPU memory."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    response = app.make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/status")
def status():
    if _current_model_id is None and _model is None:
        return jsonify({
            "status": "unloaded",
            "session_active": _session_active,
            "model_id": None,
            "diarization": {
                "enabled": _diary.enabled,
                "ready": _diary.is_ready(),
                "error": _diary.pipeline_error,
                "model_id": _diary.model_id,
            },
        })
    if not _model_ready.is_set():
        return jsonify({"status": "loading", "loading": _model_loading_state})
    if _model_error:
        return jsonify({"status": "error", "message": _model_error, "loading": _model_loading_state})
    return jsonify({
        "status": "ready",
        "session_active": _session_active,
        "model_id": _current_model_id,
        "diarization": {
            "enabled": _diary.enabled,
            "ready": _diary.is_ready(),
            "error": _diary.pipeline_error,
            "model_id": _diary.model_id,
        },
    })


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
                    "recommended_vram_gb_bf16": 9,
                    "notes": "Verified locally; the current model is unloaded before switching.",
                },
                {
                    "id": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                    "label": "Voxtral Mini 4B Realtime (26.02)",
                    "recommended_vram_gb_bf16": 9,
                    "notes": "Current default; verified locally on this 12 GB GPU.",
                },
                {
                    "id": FASTER_WHISPER_MODEL_ID,
                    "label": "faster-whisper Large v3 Turbo (Stable Broadcast)",
                    "recommended_vram_gb_bf16": 4,
                    "notes": (
                        "Word timestamps + pyannote fusion; int8_float16 "
                        "for long-session stability."
                    ),
                },
            ],
        }
    )


@app.route("/load_model", methods=["POST"])
def load_model():
    data = request.get_json(force=True)
    model_id = (data.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "No model_id provided"}), 400

    # If already loaded, no-op.
    if _current_model_id == model_id and _model_ready.is_set() and not _model_error:
        return jsonify({"ok": True, "status": "ready", "model_id": _current_model_id})
    # Ignore duplicate clicks/requests while this exact model is already loading.
    if _current_model_id == model_id and not _model_ready.is_set():
        return jsonify({"ok": True, "status": "loading", "model_id": _current_model_id})

    print(f"[ui] Loading model requested: {model_id}")
    _begin_model_load(model_id)
    return jsonify({"ok": True, "status": "loading", "model_id": _current_model_id})


@app.route("/unload_model", methods=["POST"])
def unload_model():
    """Stop live work and release the transcription model's CUDA allocation."""
    global _current_model_id, _model_error

    if _session_active:
        print("[ui] Stopping active session before model unload ...")
        _session_stop.set()
        deadline = time.monotonic() + 10.0
        while _session_active and time.monotonic() < deadline:
            time.sleep(0.05)
        if _session_active:
            return jsonify({
                "error": "The active session is still stopping. Retry unload in a few seconds."
            }), 409

    before_gb = _cuda_allocated_gb()
    with _model_switch_lock:
        _release_loaded_model()
        _current_model_id = None
        _model_error = None
        _model_ready.clear()
        _model_loading_state["attempt"] = 0
        _model_loading_state["last_error"] = None
        _model_loading_state["next_retry_in_s"] = None
    after_gb = _cuda_allocated_gb()
    freed_gb = round(max(0.0, before_gb - after_gb), 2)
    print(
        f"[model] Unloaded; CUDA allocated {before_gb:.2f} GB -> "
        f"{after_gb:.2f} GB (freed {freed_gb:.2f} GB)"
    )
    return jsonify({
        "ok": True,
        "status": "unloaded",
        "freed_vram_gb": freed_gb,
        "allocated_vram_gb": after_gb,
    })


@app.route("/proxy_video")
def proxy_video():
    """
    Proxy the YouTube live stream through ffmpeg → fragmented MP4 → browser.
    Bypasses YouTube CDN CORS restrictions.
    """
    yt_url = request.args.get("url", "").strip()
    cookie_browser = request.args.get("cookie_browser", "").strip().lower() or None
    if not yt_url:
        return "No URL", 400

    try:
        stream_url = get_video_stream_url(yt_url, cookie_browser=cookie_browser)
    except Exception as exc:
        return f"Could not resolve authenticated video: {exc}", 502

    # ffmpeg reads the stream and outputs fragmented MP4 to stdout
    ffmpeg_cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-live_start_index", "-1",
        "-probesize", "32768",
        "-analyzeduration", "1000000",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", stream_url,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
        "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-frag_duration", "500000",
        "-flush_packets", "1",
        "-loglevel", "quiet",
        "pipe:1",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def generate():
        try:
            while True:
                chunk = proc.stdout.read(16384)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.kill()
            proc.wait()

    return Response(
        stream_with_context(generate()),
        mimetype="video/mp4",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/start", methods=["POST"])
def start():
    global _session_active, _current_model_id

    if _current_model_id is None or _model is None:
        return jsonify({"error": "No model is loaded. Select a model and click Load model."}), 503
    if not _model_ready.is_set():
        return jsonify({"error": "Model is still loading — please wait"}), 503
    if _model_error:
        return jsonify({"error": _model_error}), 500
    if _session_active:
        return jsonify({"error": "A session is already active"}), 400

    data           = request.get_json(force=True)
    url            = data.get("url", "").strip()
    requested_chunk_seconds = max(
        1.0,
        min(float(data.get("chunk_seconds", 1.0)), 4.0),
    )
    # The live speaker profile uses one-second attribution units. Keeping this
    # server-side prevents an old browser tab from restoring the former 2s
    # behavior and coloring most of a new speaker's sentence incorrectly.
    chunk_seconds = 1.0
    language       = data.get("language", "en").strip()
    # Enforce the verified live profile even if a stale browser tab submits
    # the former 256-token/200-RMS settings.
    silence_rms    = max(0.0, min(float(data.get("silence_rms", 20.0)), 20.0))
    max_new_tokens = max(32, min(int(data.get("max_new_tokens", 64)), 64))
    temperature    = float(data.get("temperature", 0.0))
    # A short overlap can cut consonants/words that straddle consecutive
    # one-second capture chunks. Keep enough preceding speech for Voxtral to
    # reconstruct that boundary; duplicate text is removed after decoding.
    tail_seconds = max(
        0.4,
        min(float(data.get("tail_seconds", TAIL_SECONDS)), 1.0),
    )
    target_delay_ms = data.get("target_delay_ms", None)
    target_delay_ms = int(target_delay_ms) if target_delay_ms is not None and str(target_delay_ms).strip() != "" else None
    if target_delay_ms is not None:
        target_delay_ms = max(0, min(target_delay_ms, 120))
    requested_model_id = (data.get("model_id") or _current_model_id or "").strip()
    cookie_browser = (data.get("cookie_browser") or "").strip().lower()
    if cookie_browser not in {"", "edge", "chrome", "firefox", "brave"}:
        return jsonify({"error": "Unsupported YouTube sign-in browser"}), 400

    print(
        "[ui] Start session params: "
        f"model={requested_model_id or _current_model_id} "
        f"lang={language} chunk={chunk_seconds}s"
        f"(requested={requested_chunk_seconds}s) rms={silence_rms} "
        f"max_new_tokens={max_new_tokens} temp={temperature} "
        f"tail={tail_seconds}s target_delay_ms={target_delay_ms} "
        f"youtube_auth={cookie_browser or 'public'}"
    )

    # If UI requests a different model, kick off loading and ask user to wait.
    if requested_model_id and requested_model_id != _current_model_id:
        _transcript_q.put({"type": "error", "message": f"Loading model {requested_model_id}…"})
        _begin_model_load(requested_model_id)
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
        args=(
            url,
            chunk_seconds,
            language,
            silence_rms,
            max_new_tokens,
            temperature,
            tail_seconds,
            target_delay_ms,
            cookie_browser or None,
        ),
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
        "live": {
            "session_active": _session_active,
            "source_ended": bool(_live_telemetry.get("source_ended")),
            "source_idle_seconds": (
                round(time.monotonic() - _live_telemetry["last_audio_received"], 1)
                if _live_telemetry.get("last_audio_received") is not None
                else None
            ),
            "processing": bool(_live_telemetry.get("processing")),
            "backlog_seconds": round(float(_live_telemetry.get("backlog_seconds", 0.0)), 1),
            "last_inference_seconds": round(float(_live_telemetry.get("last_inference_seconds", 0.0)), 2),
            "last_audio_seconds": round(float(_live_telemetry.get("last_audio_seconds", 0.0)), 2),
            "realtime_factor": round(float(_live_telemetry.get("realtime_factor", 0.0)), 3),
            "average_realtime_factor": round(
                float(_live_telemetry.get("average_realtime_factor", 0.0)), 3
            ),
            "last_diarization_seconds": round(
                float(_live_telemetry.get("last_diarization_seconds", 0.0)), 2
            ),
            "average_diarization_seconds": round(
                (
                    float(_live_telemetry.get("diarization_seconds", 0.0))
                    / int(_live_telemetry.get("diarization_passes", 0))
                )
                if int(_live_telemetry.get("diarization_passes", 0))
                else 0.0,
                2,
            ),
            "diarization_speaker_count": int(
                _live_telemetry.get("diarization_speaker_count", 0)
            ),
            "speaker_update_count": int(
                _live_telemetry.get("speaker_update_count", 0)
            ),
        },
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
            ctranslate_backend = _is_faster_whisper_model(_current_model_id)
            effective_used_bytes = (
                total_bytes - free_bytes
                if ctranslate_backend
                else allocated_bytes
            )
            effective_reserved_bytes = (
                effective_used_bytes
                if ctranslate_backend
                else reserved_bytes
            )

            resp["vram"] = {
                "gpu_device_index": device_idx,
                "vram_free_gb": round(free_bytes / (1024**3), 2),
                "vram_total_gb": round(total_bytes / (1024**3), 2),
                "vram_allocated_gb": round(effective_used_bytes / (1024**3), 2),
                "vram_reserved_gb": round(effective_reserved_bytes / (1024**3), 2),
                "vram_peak_allocated_gb": round(peak_bytes / (1024**3), 2),
                "vram_measurement": (
                    "CUDA device used (CTranslate2)"
                    if ctranslate_backend
                    else "PyTorch allocator"
                ),
                "vram_reserved_percent": round((effective_reserved_bytes / total_bytes) * 100.0, 2)
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
    cookie_browser: str | None,
):
    global _session_active
    try:
        _live_telemetry.update({
            "last_audio_received": None,
            "source_ended": False,
            "processing": False,
            "backlog_seconds": 0.0,
            "last_inference_seconds": 0.0,
            "last_audio_seconds": 0.0,
            "realtime_factor": 0.0,
            "average_realtime_factor": 0.0,
            "processed_audio_seconds": 0.0,
            "inference_seconds": 0.0,
            "diarization_seconds": 0.0,
            "diarization_passes": 0,
            "last_diarization_seconds": 0.0,
            "diarization_speaker_count": 0,
            "speaker_update_count": 0,
        })
        # Resolve YouTube stream URL
        try:
            stream_url = get_audio_stream_url(url, cookie_browser=cookie_browser)
        except Exception as exc:
            _transcript_q.put({"type": "error", "message": f"yt-dlp failed: {exc}"})
            return

        _transcript_q.put({"type": "started"})

        # Start audio capture
        # Lossless, disk-backed capture. A crash or inference failure leaves
        # unacknowledged PCM chunks on disk instead of losing testimony.
        spool_root = Path(__file__).resolve().parent / "runtime_spool"
        session_spool = spool_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        audio_q = AudioSpool(session_spool)
        # Twelve seconds of half-second slices absorbs temporary CUDA
        # contention without dropping the audio used for speaker decisions.
        live_diarization_q = queue.Queue(maxsize=24) if _diary.enabled else None
        capture_thread = threading.Thread(
            target=stream_audio,
            args=(
                stream_url,
                chunk_seconds,
                audio_q,
                _session_stop,
                _live_telemetry,
                live_diarization_q,
            ),
            daemon=True,
        )
        capture_thread.start()

        if shutil.disk_usage(spool_root).free < 2 * 1024**3:
            _transcript_q.put({
                "type": "error",
                "message": "Less than 2 GB free for the lossless audio spool; session not started.",
            })
            return

        tail_bytes      = int(SAMPLE_RATE * tail_seconds * SAMPLE_WIDTH * CHANNELS)
        prev_tail_pcm   = b""
        prev_tail_words = []
        diarization_buffer = bytearray()
        diarization_segments = []
        diarization_lock = threading.Lock()
        diarization_jobs: queue.Queue = queue.Queue(maxsize=1)
        diarization_tracker = LiveSpeakerTracker()
        # Keep the newest edge revisable across at least one additional
        # pyannote pass. Tentative changes are still published immediately;
        # transcript rows are finalized from this settled timeline.
        turn_committer = LiveTurnCommitter(commit_lag=0.5, merge_gap=0.4)
        _diary.registry = SpeakerRegistry()
        recent_transcripts = deque(maxlen=40)
        # Pyannote clustering needs enough voice context to distinguish a
        # real broadcast turn. Latest-only scheduling keeps this longer window
        # responsive without processing stale queued audio.
        # Broadcast packages often contain a very short guest between an
        # anchor and reporter. Community-1 needs enough surrounding voice
        # context to avoid merging that guest into a dominant cluster.
        # Ten seconds gives Community-1 enough material to separate short news
        # hand-offs and translated/original voice mixtures. The 0.75s stride
        # keeps the live edge responsive; latest-only scheduling prevents the
        # larger overlapping window from creating a processing backlog.
        diarization_window_seconds = 10.0
        diarization_run_interval = 0.75
        diarization_warmup_seconds = 2.0
        max_diarization_bytes = int(SAMPLE_RATE * diarization_window_seconds * SAMPLE_WIDTH * CHANNELS)
        last_diarization_submit = 0.0
        diarization_audio_seconds = 0.0
        # Native Voxtral text becomes visible slightly after the causal audio
        # that produced it. Attribute text against an earlier diarization
        # interval without delaying display; otherwise a turn at the end of a
        # sentence can recolor that whole sentence as the next speaker.
        speaker_alignment_delay_seconds = 0.75

        def _aligned_speaker_interval(
            start_seconds: float,
            end_seconds: float,
            delay_seconds: float | None = None,
        ) -> tuple[float, float]:
            delay = (
                speaker_alignment_delay_seconds
                if delay_seconds is None
                else max(0.0, float(delay_seconds))
            )
            aligned_start = max(
                0.0,
                float(start_seconds) - delay,
            )
            aligned_end = max(
                aligned_start + 0.08,
                float(end_seconds) - delay,
            )
            return aligned_start, aligned_end

        def _speaker_from_segments(
            segments: list[dict[str, object]],
            start_seconds: float,
            end_seconds: float,
        ) -> str | None:
            """Choose the voice with the greatest overlap in a provisional window."""
            scores: dict[str, float] = {}
            latest: dict[str, float] = {}
            for segment in segments:
                overlap = max(
                    0.0,
                    min(float(segment["end"]), end_seconds)
                    - max(float(segment["start"]), start_seconds),
                )
                if overlap <= 0:
                    continue
                speaker = str(segment["speaker"])
                scores[speaker] = scores.get(speaker, 0.0) + overlap
                latest[speaker] = max(latest.get(speaker, 0.0), float(segment["end"]))
            if not scores:
                return None
            return max(scores, key=lambda speaker: (scores[speaker], latest[speaker]))

        def _publish_speaker_corrections(
            latest_end: float | None = None,
            provisional_segments: list[dict[str, object]] | None = None,
        ):
            """Color covered rows immediately, then settle them from committed turns."""
            for item in recent_transcripts:
                if latest_end is not None and float(item["audio_start"]) > latest_end + 0.15:
                    continue
                aligned_start, aligned_end = _aligned_speaker_interval(
                    float(item["audio_start"]),
                    float(item["audio_end"]),
                    float(
                        item.get(
                            "speaker_alignment_delay",
                            speaker_alignment_delay_seconds,
                        )
                    ),
                )
                corrected_label = turn_committer.speaker_for_interval(
                    aligned_start,
                    aligned_end,
                )
                # Provisional turns are useful only before the settled
                # timeline covers a new caption. Never let a later rolling
                # pass overwrite an already committed color and make it
                # flicker back and forth.
                if corrected_label is None and provisional_segments:
                    provisional_label = _speaker_from_segments(
                        provisional_segments,
                        aligned_start,
                        aligned_end,
                    )
                    if provisional_label:
                        corrected_label = provisional_label
                if not corrected_label or corrected_label == item.get("speaker_label"):
                    continue
                item["speaker_label"] = corrected_label
                item["speaker_revisions"] = int(item.get("speaker_revisions", 0)) + 1
                corrected_id, corrected_color = _diary.registry.register_label(
                    corrected_label
                )
                _transcript_q.put({
                    "type": "speaker_update",
                    "entry_id": item["entry_id"],
                    "speaker_id": corrected_id,
                    "speaker_color": corrected_color,
                })
                _live_telemetry["speaker_update_count"] += 1

        def _diarization_worker():
            nonlocal diarization_segments
            consecutive_failures = 0
            previous_signature = None
            last_announced_speaker = None
            last_committed_speaker = None
            while not _session_stop.is_set():
                try:
                    snapshot, window_start_seconds = diarization_jobs.get(timeout=1)
                except queue.Empty:
                    continue
                try:
                    # Let pyannote infer however many voices are actually
                    # present. Session-wide identities remain unlimited.
                    diarization_t0 = time.perf_counter()
                    # Diarize each window unconstrained, as in the proven live
                    # engine. The session registry, not Pyannote clustering,
                    # applies the ten-speaker presentation ceiling.
                    segments = _diary.diarize_pcm(
                        snapshot,
                        SAMPLE_RATE,
                        # Local constraint only: a ten-second news window
                        # almost never contains more than four voices. The
                        # session registry remains open-ended across windows.
                        max_speakers=4,
                    )
                    window_embeddings = dict(_diary.last_window_embeddings)
                    if segments and not window_embeddings:
                        # Some Community-1 builds do not expose pass-native
                        # speaker_embeddings. Without a fallback every
                        # non-overlapping news window becomes a brand-new
                        # speaker/color even when an anchor returns.
                        window_embeddings = _diary.embed_speakers_pcm(
                            snapshot,
                            SAMPLE_RATE,
                            segments,
                        )
                    diarization_elapsed = time.perf_counter() - diarization_t0
                    _live_telemetry["last_diarization_seconds"] = diarization_elapsed
                    _live_telemetry["diarization_seconds"] += diarization_elapsed
                    _live_telemetry["diarization_passes"] += 1
                    with diarization_lock:
                        reconciled = diarization_tracker.update(
                            segments,
                            window_start_seconds,
                            window_embeddings=window_embeddings or None,
                        )
                        diarization_segments = reconciled
                        snapshot_seconds = len(snapshot) / (
                            SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
                        )
                        stream_end_seconds = window_start_seconds + snapshot_seconds
                        # Publish the live-edge decision immediately. This is
                        # intentionally provisional: the longer committed
                        # timeline below can still correct transcript rows.
                        live_edge_time = max(
                            window_start_seconds,
                            stream_end_seconds - 0.25,
                        )
                        provisional_speaker = diarization_tracker.speaker_at(
                            live_edge_time,
                            fallback_to_latest=False,
                        )
                        if (
                            provisional_speaker
                            and provisional_speaker != last_announced_speaker
                        ):
                            last_announced_speaker = provisional_speaker
                            provisional_id, provisional_color = (
                                _diary.registry.register_label(provisional_speaker)
                            )
                            provisional_start = next(
                                (
                                    float(segment["start"])
                                    for segment in reversed(reconciled)
                                    if str(segment["speaker"]) == provisional_speaker
                                ),
                                live_edge_time,
                            )
                            print(
                                "[diarization] Live turn "
                                f"{provisional_start:.2f}s -> {provisional_id}"
                            )
                            _transcript_q.put({
                                "type": "speaker_turn",
                                "speaker_id": provisional_id,
                                "speaker_color": provisional_color,
                                "audio_time": round(provisional_start, 2),
                                "provisional": True,
                            })
                        newly_committed = turn_committer.commit(
                            reconciled,
                            stream_end_seconds,
                        )
                        for committed_turn in newly_committed:
                            committed_speaker = str(committed_turn["speaker"])
                            if committed_speaker == last_committed_speaker:
                                continue
                            last_committed_speaker = committed_speaker
                            turn_id, turn_color = _diary.registry.register_label(
                                committed_speaker
                            )
                            print(
                                "[diarization] Turn "
                                f"{float(committed_turn['start']):.2f}s -> "
                                f"{turn_id}"
                            )
                        stable_labels = list(dict.fromkeys(
                            str(segment["speaker"]) for segment in reconciled
                        ))
                        _live_telemetry["diarization_speaker_count"] = max(
                            int(_live_telemetry.get("diarization_speaker_count", 0)),
                            len(stable_labels),
                        )
                        signature = (
                            len(stable_labels),
                            stable_labels[-1] if stable_labels else None,
                        )
                        if signature != previous_signature:
                            print(
                                "[diarization] Speaker state: "
                                f"window_speakers={len(stable_labels)} "
                                f"latest={signature[1] or 'none'} "
                                f"embeddings={len(window_embeddings)}"
                            )
                            previous_signature = signature
                        latest_end = max(
                            (float(segment["end"]) for segment in turn_committer.turns),
                            default=0.0,
                        )
                        _publish_speaker_corrections(
                            max(latest_end, stream_end_seconds),
                            provisional_segments=reconciled,
                        )
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    print(f"[diarization] Pass {consecutive_failures} failed: {exc}")
                    if consecutive_failures >= 3:
                        reason = f"Diarization disabled after 3 failures: {exc}"
                        _diary.disable(reason)
                        _transcript_q.put({"type": "warning", "message": reason})
                        return

        def _diarization_feeder():
            """Feed Pyannote from capture, independently of Voxtral latency."""
            nonlocal diarization_audio_seconds, last_diarization_submit
            while not _session_stop.is_set():
                try:
                    pcm_slice, capture_end_seconds = live_diarization_q.get(timeout=1)
                except queue.Empty:
                    continue
                diarization_buffer.extend(pcm_slice)
                diarization_audio_seconds = float(capture_end_seconds)
                if len(diarization_buffer) > max_diarization_bytes:
                    del diarization_buffer[:-max_diarization_bytes]
                if (
                    time.monotonic() - last_diarization_submit < diarization_run_interval
                    or len(diarization_buffer)
                    < int(
                        SAMPLE_RATE
                        * diarization_warmup_seconds
                        * SAMPLE_WIDTH
                        * CHANNELS
                    )
                ):
                    continue
                last_diarization_submit = time.monotonic()
                snapshot_seconds = len(diarization_buffer) / (
                    SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
                )
                window_start_seconds = max(
                    0.0,
                    diarization_audio_seconds - snapshot_seconds,
                )
                try:
                    while True:
                        diarization_jobs.get_nowait()
                except queue.Empty:
                    pass
                try:
                    diarization_jobs.put_nowait(
                        (bytes(diarization_buffer), window_start_seconds)
                    )
                except queue.Full:
                    pass

        if _diary.enabled:
            threading.Thread(
                target=_diarization_worker,
                daemon=True,
                name="DiarizationWorker",
            ).start()
            threading.Thread(
                target=_diarization_feeder,
                daemon=True,
                name="DiarizationAudioFeeder",
            ).start()

        def _run_faster_whisper_stream() -> None:
            """Long-running word-timestamp ASR with exact speaker fusion."""
            print(
                "[transcription] faster-whisper Stable Broadcast enabled "
                "(1s analysis, 6s rolling context, late-word recovery)"
            )
            pending = []
            previous_tail = b""
            tail_bytes_whisper = int(
                SAMPLE_RATE * 5.0 * SAMPLE_WIDTH * CHANNELS
            )
            stream_started = time.perf_counter()
            emitted_audio_end = 0.0
            stream_primed = False
            caption_buffer = WhisperCaptionBuffer(
                # Length is deliberately variable: a confirmed speaker turn
                # flushes immediately, sentence/clause punctuation gives
                # readable lines, and these bounds only protect long monologues.
                max_words=20,
                max_duration=4.0,
                min_sentence_words=4,
                min_clause_words=10,
            )
            recent_whisper_tokens = []
            covered_whisper_intervals = []
            ready_caption_units = deque()
            latest_inference_seconds = 0.0
            latest_batch_audio_seconds = 1.0

            def _settled_caption_parts(unit):
                def resolve(start: float, end: float) -> str | None:
                    with diarization_lock:
                        return turn_committer.speaker_for_interval(start, end)

                # Preserve every timestamp-backed pyannote boundary. Smoothing
                # short caption parts here swallowed genuine live hand-offs.
                return split_unit_at_speaker_boundaries(unit, resolve)

            def _queue_settled_parts(parts) -> None:
                """Coalesce adjacent settled parts without crossing a voice."""
                for part in parts:
                    if (
                        ready_caption_units
                        and ready_caption_units[-1].speaker == part.speaker
                        and part.start - ready_caption_units[-1].end <= 0.8
                        and (
                            len(ready_caption_units[-1].text.split())
                            + len(part.text.split())
                            <= 24
                        )
                    ):
                        previous = ready_caption_units.pop()
                        ready_caption_units.append(
                            WhisperSpeakerUnit(
                                text=f"{previous.text} {part.text}".strip(),
                                start=previous.start,
                                end=part.end,
                                speaker=part.speaker,
                                words=previous.words + part.words,
                            )
                        )
                    else:
                        ready_caption_units.append(part)

            def _emit_whisper_unit(
                unit,
                elapsed: float,
                inference_seconds: float,
                batch_audio_seconds: float,
            ) -> None:
                final_speaker = unit.speaker
                entry_id = uuid.uuid4().hex
                with diarization_lock:
                    recent_transcripts.append({
                        "entry_id": entry_id,
                        "audio_start": unit.start,
                        "audio_end": unit.end,
                        "speaker_label": final_speaker,
                        "speaker_revisions": 0,
                        "speaker_alignment_delay": 0.0,
                    })
                speaker_id, color = _diary.registry.register_label(final_speaker)
                _transcript_q.put({
                    "type": "transcript",
                    "entry_id": entry_id,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "text": unit.text,
                    "elapsed": round(elapsed, 2),
                    "audio_seconds": round(unit.end - unit.start, 2),
                    "realtime_factor": round(
                        inference_seconds / batch_audio_seconds,
                        3,
                    ),
                    "speaker_id": speaker_id,
                    "speaker_color": color,
                    "diarization_enabled": _diary.enabled,
                })

            while not _session_stop.is_set():
                try:
                    item = audio_q.get(timeout=1)
                    pending.append(item)
                except queue.Empty:
                    if capture_thread.is_alive():
                        continue
                    if not pending:
                        break

                # Prime with two seconds so the first decode has lexical
                # context. Afterwards decode every captured second with the
                # preceding two seconds attached. This avoids two-second UI
                # bursts while retaining enough context for complete words.
                if (
                    not stream_primed
                    and len(pending) < 2
                    and capture_thread.is_alive()
                ):
                    continue

                batch_size = 2 if not stream_primed else 1
                batch = pending[:batch_size]
                del pending[: len(batch)]
                stream_primed = True
                final_source_batch = (
                    not capture_thread.is_alive()
                    and not pending
                    and audio_q.empty()
                )
                pcm = b"".join(chunk.data for chunk in batch)
                context_pcm = previous_tail + pcm
                context_tail_seconds = len(previous_tail) / (
                    SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
                )
                context_start = max(
                    0.0,
                    float(batch[0].sequence) * chunk_seconds
                    - context_tail_seconds,
                )
                batch_end = float(batch[-1].sequence + 1) * chunk_seconds
                def _resolve_whisper_speaker(
                    word_start: float,
                    word_end: float,
                ) -> str | None:
                    # pyannote and Whisper share the absolute audio timeline.
                    # Greatest-overlap assignment catches changes inside an ASR
                    # window instead of applying one speaker to the whole text.
                    with diarization_lock:
                        speaker_label = turn_committer.speaker_for_interval(
                            word_start,
                            word_end,
                        )
                        if speaker_label is None and diarization_segments:
                            speaker_label = diarization_tracker.speaker_at(
                                (word_start + word_end) / 2.0
                            )
                    return speaker_label

                inference_t0 = time.perf_counter()
                try:
                    units = transcribe_whisper_window(
                        _model,
                        context_pcm,
                        context_start=context_start,
                        emitted_audio_end=emitted_audio_end,
                        late_recovery_seconds=0.35,
                        commit_before=(
                            None
                            if final_source_batch
                            else max(0.0, batch_end - 0.45)
                        ),
                        language=language or "en",
                        resolve_speaker=_resolve_whisper_speaker,
                    )
                    units, recent_whisper_tokens = remove_repeated_window_prefix(
                        units,
                        recent_whisper_tokens,
                        last_emitted_time=emitted_audio_end,
                    )
                    units, covered_whisper_intervals = (
                        remove_timestamp_covered_words(
                            units,
                            covered_whisper_intervals,
                        )
                    )
                except Exception:
                    # Leave these files unacknowledged for recovery.
                    pending = batch + pending
                    raise

                inference_seconds = time.perf_counter() - inference_t0
                latest_inference_seconds = inference_seconds
                batch_audio_seconds = max(
                    0.001,
                    batch_end - float(batch[0].sequence) * chunk_seconds,
                )
                latest_batch_audio_seconds = batch_audio_seconds
                _live_telemetry["processing"] = True
                _live_telemetry["last_inference_seconds"] = inference_seconds
                _live_telemetry["inference_seconds"] += inference_seconds
                _live_telemetry["processed_audio_seconds"] = batch_end
                _live_telemetry["last_audio_seconds"] = batch_end
                _live_telemetry["realtime_factor"] = (
                    inference_seconds / batch_audio_seconds
                )
                elapsed = max(0.001, time.perf_counter() - stream_started)
                _live_telemetry["average_realtime_factor"] = (
                    _live_telemetry["inference_seconds"] / max(batch_end, 0.001)
                )
                _live_telemetry["backlog_seconds"] = (
                    audio_q.qsize() * chunk_seconds
                )

                for buffered_unit in caption_buffer.push(units):
                    _queue_settled_parts(
                        _settled_caption_parts(buffered_unit)
                    )
                # The ASR may finish several internal segments in one fast GPU
                # pass. Publish at most one caption per one-second analysis
                # stride so the operator sees a live sequence, not a burst.
                if ready_caption_units:
                    _emit_whisper_unit(
                        ready_caption_units.popleft(),
                        elapsed,
                        inference_seconds,
                        batch_audio_seconds,
                    )

                # Never mark the full capture window as emitted. Whisper may
                # defer a boundary word until the following context arrives;
                # advancing to batch_end here permanently discarded it.
                if units:
                    emitted_audio_end = max(
                        emitted_audio_end,
                        max(unit.end for unit in units),
                    )
                previous_tail = context_pcm[-tail_bytes_whisper:]
                audio_q.ack(batch)

            for buffered_unit in caption_buffer.flush():
                _queue_settled_parts(
                    _settled_caption_parts(buffered_unit)
                )
            while ready_caption_units:
                _emit_whisper_unit(
                    ready_caption_units.popleft(),
                    max(0.001, time.perf_counter() - stream_started),
                    latest_inference_seconds,
                    latest_batch_audio_seconds,
                )
            _live_telemetry["processing"] = False
            _live_telemetry["backlog_seconds"] = (
                audio_q.qsize() * chunk_seconds
            )

        if _is_faster_whisper_model(_current_model_id):
            _run_faster_whisper_stream()
            return

        native_replay = deque()

        def _run_native_voxtral_stream(initial_replay: deque) -> tuple[list, bool]:
            """Run one bounded causal segment and return its unprocessed edge."""
            consumed = deque()
            consumed_lock = threading.Lock()
            segment_base_seconds = None
            final_global_cursor = 0.0
            phrase_words = []
            phrase_start = None
            last_phrase_end = 0.0
            stream_started = time.perf_counter()
            emitted_any = False

            def pcm_chunks():
                nonlocal segment_base_seconds
                while not _session_stop.is_set():
                    if initial_replay:
                        item = initial_replay.popleft()
                    else:
                        try:
                            item = audio_q.get(timeout=1)
                        except queue.Empty:
                            if not capture_thread.is_alive():
                                return
                            continue
                    if segment_base_seconds is None:
                        segment_base_seconds = float(item.sequence) * chunk_seconds
                    with consumed_lock:
                        consumed.append(item)
                    yield item.data

            def on_native_audio_progress(local_audio_cursor: float) -> None:
                """Track causal progress even while no complete text is emitted."""
                nonlocal final_global_cursor
                base = float(segment_base_seconds or 0.0)
                audio_cursor = base + float(local_audio_cursor)
                final_global_cursor = max(final_global_cursor, audio_cursor)
                _live_telemetry["processing"] = True
                _live_telemetry["processed_audio_seconds"] = audio_cursor
                _live_telemetry["last_audio_seconds"] = audio_cursor
                _live_telemetry["backlog_seconds"] = (
                    audio_q.qsize() * chunk_seconds
                )
                safe_audio_end = max(0.0, audio_cursor - 2.0)
                ready_to_ack = []
                with consumed_lock:
                    while (
                        consumed
                        and float(consumed[0].sequence + 1) * chunk_seconds
                        <= safe_audio_end
                    ):
                        ready_to_ack.append(consumed.popleft())
                if ready_to_ack:
                    audio_q.ack(ready_to_ack)

            def emit_phrase(audio_end: float) -> None:
                nonlocal phrase_words, phrase_start, last_phrase_end, emitted_any
                if not phrase_words:
                    return
                audio_start = max(
                    last_phrase_end,
                    float(phrase_start if phrase_start is not None else audio_end),
                )
                audio_end = max(audio_start + 0.08, float(audio_end))
                with diarization_lock:
                    current_segments = list(diarization_segments)
                    aligned_start, aligned_end = _aligned_speaker_interval(
                        audio_start,
                        audio_end,
                    )
                    speaker_label = turn_committer.speaker_for_interval(
                        aligned_start,
                        aligned_end,
                    )
                    if speaker_label is None and current_segments:
                        speaker_label = diarization_tracker.speaker_at(
                            (aligned_start + aligned_end) / 2.0
                        )
                    entry_id = uuid.uuid4().hex
                    recent_transcripts.append({
                        "entry_id": entry_id,
                        "audio_start": audio_start,
                        "audio_end": audio_end,
                        "speaker_label": speaker_label,
                        "speaker_revisions": 0,
                        "speaker_alignment_delay": (
                            speaker_alignment_delay_seconds
                        ),
                    })
                speaker_id, color = _diary.registry.register_label(speaker_label)
                elapsed_wall = max(1e-3, time.perf_counter() - stream_started)
                _transcript_q.put({
                    "type": "transcript",
                    "entry_id": entry_id,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "text": " ".join(phrase_words),
                    "elapsed": round(elapsed_wall, 2),
                    "audio_seconds": round(audio_end - audio_start, 2),
                    "realtime_factor": round(elapsed_wall / audio_end, 3),
                    "speaker_id": speaker_id,
                    "speaker_color": color,
                    "diarization_enabled": _diary.enabled,
                })
                phrase_words = []
                phrase_start = None
                last_phrase_end = audio_end
                emitted_any = True

            try:
                print("[transcription] Native stateful Voxtral streaming enabled")
                for text_piece, audio_cursor in stream_voxtral_text(
                    pcm_chunks(),
                    _model,
                    _processor,
                    _device,
                    _session_stop,
                    on_audio_progress=on_native_audio_progress,
                    max_audio_seconds=90.0,
                ):
                    audio_cursor = float(segment_base_seconds or 0.0) + audio_cursor
                    _live_telemetry["processing"] = True
                    _live_telemetry["processed_audio_seconds"] = audio_cursor
                    wall_elapsed = max(
                        1e-3,
                        time.perf_counter() - stream_started,
                    )
                    _live_telemetry["inference_seconds"] = wall_elapsed
                    _live_telemetry["average_realtime_factor"] = (
                        wall_elapsed / audio_cursor if audio_cursor else 0.0
                    )
                    _live_telemetry["realtime_factor"] = (
                        wall_elapsed / audio_cursor if audio_cursor else 0.0
                    )
                    _live_telemetry["last_audio_seconds"] = audio_cursor
                    _live_telemetry["backlog_seconds"] = (
                        audio_q.qsize() * chunk_seconds
                    )

                    # Once native output has covered a chunk, its spool file
                    # can be acknowledged. Retain two seconds for safe replay
                    # if the experimental stream fails at its live edge.
                    safe_audio_end = max(0.0, audio_cursor - 2.0)
                    on_native_audio_progress(audio_cursor)

                    words = text_piece.strip().split()
                    if not words:
                        continue
                    if phrase_start is None:
                        phrase_start = max(last_phrase_end, audio_cursor - 0.4)
                    phrase_words.extend(words)
                    sentence_end = phrase_words[-1].endswith((".", "?", "!"))
                    if sentence_end or len(phrase_words) >= 14:
                        emit_phrase(audio_cursor)
                emit_phrase(float(_live_telemetry["processed_audio_seconds"]))
                processed_chunks = []
                unprocessed_chunks = []
                with consumed_lock:
                    while consumed:
                        item = consumed.popleft()
                        item_end = float(item.sequence + 1) * chunk_seconds
                        if item_end <= final_global_cursor + 1e-3:
                            processed_chunks.append(item)
                        else:
                            unprocessed_chunks.append(item)
                audio_q.ack(processed_chunks)
                _live_telemetry["processing"] = False
                return unprocessed_chunks, True
            except Exception as exc:
                _live_telemetry["processing"] = False
                print(f"[transcription] Native stream fallback: {exc}")
                _transcript_q.put({
                    "type": "warning",
                    "message": (
                        "Native streaming was unavailable; continuing with "
                        "the proven buffered decoder."
                    ),
                })
                # Before the first emitted phrase every consumed spool item is
                # still present. A later failure replays only the retained
                # two-second safety edge to avoid duplicating old captions.
                with consumed_lock:
                    return list(consumed), False

        if env_flag("VOXTRAL_NATIVE_STREAMING", default=True):
            native_ok = True
            while native_ok and not _session_stop.is_set():
                replay_items, native_ok = _run_native_voxtral_stream(native_replay)
                native_replay = deque(replay_items)
                if (
                    native_ok
                    and not native_replay
                    and not capture_thread.is_alive()
                    and audio_q.empty()
                ):
                    _live_telemetry["backlog_seconds"] = 0.0
                    return
                if native_ok:
                    print(
                        "[transcription] Rolling native context at 90s "
                        f"(replay={len(native_replay)} chunks)"
                    )
            if native_ok:
                _live_telemetry["backlog_seconds"] = 0.0
                return

        while not _session_stop.is_set():
            if native_replay:
                first_audio_item = native_replay.popleft()
            else:
                try:
                    first_audio_item = audio_q.get(timeout=5)
                except queue.Empty:
                    if not capture_thread.is_alive():
                        break
                    continue

            # Keep one transcript decision per chunk while fully caught up.
            # ``qsize`` is measured *after* taking the first item, so a value
            # of one already means two seconds of audio are ready. React at
            # that first queued second; waiting for qsize >= 2 made a model
            # running near RTF 1.0 hold a permanent 1-2 second backlog.
            audio_items = [first_audio_item]
            queued_items = audio_q.qsize()
            # At RTF close to 1.0, a three-chunk ceiling cannot erase an
            # existing backlog because per-call setup consumes the small
            # realtime margin. Larger lossless batches are used only while
            # behind; output returns to two-second cadence once caught up.
            catchup_limit = (
                8 if queued_items >= 7
                else 6 if queued_items >= 5
                else 4 if queued_items >= 3
                else 3 if queued_items >= 2
                else 2 if queued_items >= 1
                else 1
            )
            while len(audio_items) < catchup_limit:
                try:
                    audio_items.append(audio_q.get_nowait())
                except queue.Empty:
                    break
            pcm_data = b"".join(item.data for item in audio_items)
            effective_chunk_seconds = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
            _live_telemetry["processing"] = True
            _live_telemetry["backlog_seconds"] = audio_q.qsize() * chunk_seconds

            if is_silent(pcm_data, silence_rms):
                prev_tail_pcm   = b""
                prev_tail_words = []
                audio_q.ack(audio_items)
                _live_telemetry["processing"] = False
                _live_telemetry["backlog_seconds"] = audio_q.qsize() * chunk_seconds
                continue

            combined_pcm  = prev_tail_pcm + pcm_data
            prev_tail_pcm = pcm_data[-tail_bytes:]
            wav_bytes     = pcm_to_wav(combined_pcm)
            t0            = time.time()
            # A catch-up batch contains more speech than a normal live unit.
            # Scale the output budget so fast continuous speech is not cut at
            # the old fixed 64-token ceiling.
            inference_max_new_tokens = min(
                128,
                max(
                    max_new_tokens,
                    int(math.ceil(effective_chunk_seconds * 12.0)),
                ),
            )

            text = None
            for attempt in range(1, 4):
                try:
                    text = transcribe_wav_bytes(
                        wav_bytes,
                        _model,
                        _processor,
                        _device,
                        language,
                        max_new_tokens=inference_max_new_tokens,
                        temperature=temperature,
                        target_delay_ms=target_delay_ms,
                    )
                    break
                except Exception as exc:
                    if attempt >= 3:
                        _transcript_q.put({
                            "type": "error",
                            "message": (
                                f"Transcription failed after 3 attempts; audio retained at "
                                f"{session_spool}: {exc}"
                            ),
                        })
                    else:
                        time.sleep(0.5 * attempt)
            if text is None:
                _live_telemetry["processing"] = False
                break

            elapsed = time.time() - t0
            audio_q.ack(audio_items)
            _live_telemetry["last_inference_seconds"] = elapsed
            _live_telemetry["last_audio_seconds"] = effective_chunk_seconds
            _live_telemetry["realtime_factor"] = (
                elapsed / effective_chunk_seconds if effective_chunk_seconds else 0.0
            )
            _live_telemetry["processed_audio_seconds"] += effective_chunk_seconds
            _live_telemetry["inference_seconds"] += elapsed
            _live_telemetry["average_realtime_factor"] = (
                _live_telemetry["inference_seconds"]
                / _live_telemetry["processed_audio_seconds"]
                if _live_telemetry["processed_audio_seconds"]
                else 0.0
            )
            _live_telemetry["processing"] = False
            _live_telemetry["backlog_seconds"] = audio_q.qsize() * chunk_seconds
            text    = strip_overlap(text, prev_tail_words)

            if text:
                text_words = text.split()
                prev_tail_words = text_words[-3:]
                ts = time.strftime("%H:%M:%S")
                attribution_units = []
                with diarization_lock:
                    current_segments = list(diarization_segments)
                    # Inference may combine chunks to remain realtime. Map the
                    # decoded words back to each original capture unit so one
                    # catch-up batch never becomes a long single-color turn.
                    unit_count = len(audio_items)
                    for unit_index, audio_item in enumerate(audio_items):
                        unit_start = float(audio_item.sequence) * chunk_seconds
                        unit_end = float(audio_item.sequence + 1) * chunk_seconds
                        unit_midpoint = (unit_start + unit_end) / 2.0
                        speaker_label = turn_committer.speaker_for_interval(
                            unit_start,
                            unit_end,
                        )
                        if speaker_label is None and current_segments:
                            speaker_label = diarization_tracker.speaker_at(
                                unit_midpoint
                            )
                        word_start = round(unit_index * len(text_words) / unit_count)
                        word_end = round((unit_index + 1) * len(text_words) / unit_count)
                        unit_words = text_words[word_start:word_end]
                        if not unit_words:
                            continue
                        if (
                            attribution_units
                            and attribution_units[-1]["speaker_label"] == speaker_label
                            and len(attribution_units[-1]["words"]) + len(unit_words) <= 16
                        ):
                            attribution_units[-1]["words"].extend(unit_words)
                            attribution_units[-1]["audio_end"] = unit_end
                        else:
                            attribution_units.append({
                                "words": list(unit_words),
                                "audio_start": unit_start,
                                "audio_end": unit_end,
                                "speaker_label": speaker_label,
                            })

                    for unit in attribution_units:
                        entry_id = uuid.uuid4().hex
                        recent_transcripts.append({
                            "entry_id": entry_id,
                            "audio_start": unit["audio_start"],
                            "audio_end": unit["audio_end"],
                            "speaker_label": unit["speaker_label"],
                            "speaker_revisions": 0,
                        })
                        unit["entry_id"] = entry_id

                for unit in attribution_units:
                    unit_text = " ".join(unit["words"])
                    speaker_id, color = _diary.registry.register_label(
                        unit["speaker_label"]
                    )
                    _transcript_q.put({
                        "type": "transcript",
                        "entry_id": unit["entry_id"],
                        "timestamp": ts,
                        "text": unit_text,
                        "elapsed": round(elapsed, 2),
                        "audio_seconds": round(
                            float(unit["audio_end"]) - float(unit["audio_start"]),
                            2,
                        ),
                        "realtime_factor": round(
                            elapsed / effective_chunk_seconds
                            if effective_chunk_seconds else 0.0,
                            3,
                        ),
                        "speaker_id": speaker_id,
                        "speaker_color": color,
                        "diarization_enabled": _diary.enabled,
                    })

    except Exception as exc:
        _transcript_q.put({"type": "error", "message": str(exc)})
    finally:
        _session_active = False
        # Stop capture before removing an empty spool directory. Otherwise the
        # ffmpeg reader can race shutdown and try to write its final PCM slice
        # after AudioSpool.close() has removed that directory.
        _session_stop.set()
        if "capture_thread" in locals() and capture_thread.is_alive():
            capture_thread.join(timeout=3.0)
        _live_telemetry["processing"] = False
        _live_telemetry["backlog_seconds"] = 0.0
        if "audio_q" in locals():
            audio_q.close()
        _transcript_q.put({"type": "stopped"})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _port_is_available("127.0.0.1", 7860):
        print(
            "[ui] Port 7860 is already in use. The transcription UI is likely "
            "already running at http://localhost:7860; refusing a duplicate "
            "launch before allocating GPU memory."
        )
        raise SystemExit(2)
    _start_background_services()
    print("=" * 55)
    print("  YouTube Live Transcribe — Web UI")
    print("  Open http://localhost:7860 in your browser")
    print("  Model loading in background …")
    print("=" * 55)
    # .env is loaded explicitly by speaker_diarization.py; prevent Flask from
    # suggesting an unnecessary python-dotenv dependency.
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True, load_dotenv=False)
