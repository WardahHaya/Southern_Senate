#!/usr/bin/env python3
"""
YouTubeLiveTranscribe.py — Real-time YouTube live stream transcription
using mistralai/Voxtral-Mini-4B-Realtime-2602.

Usage:
    python YouTubeLiveTranscribe.py <youtube_live_url>
    python YouTubeLiveTranscribe.py <youtube_live_url> --chunk-seconds 2.0
    python YouTubeLiveTranscribe.py <youtube_live_url> --language fr
    python YouTubeLiveTranscribe.py <youtube_live_url> --no-model-load  (uses HTTP API on port 8002)

Requirements:
    pip install yt-dlp
    sudo apt install ffmpeg        # or brew install ffmpeg on Mac
    # All other deps already in requirements-voxtral.txt
"""

import sys
import os
import argparse
import subprocess
import shutil
import io
import wave
import time
import threading
import queue
import signal
import logging
import tempfile
import importlib.util
from pathlib import Path

import torch
import numpy as np
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from transformers import modeling_utils
from transformers.processing_utils import ProcessorMixin
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_ID     = "mistralai/Voxtral-Mini-4B-Realtime-2602"
SAMPLE_RATE  = 16000
CHANNELS     = 1
SAMPLE_WIDTH = 2          # int16 → 2 bytes per sample
TAIL_SECONDS = 0.3        # audio overlap kept from end of previous chunk

# Reuse the same cache dir as the main Voxtral server
if os.name == "nt":
    # Keep multi-gigabyte model weights on the project drive. The Windows
    # system drive may not have enough working space for both weights and the
    # page file during model initialization.
    DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
else:
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.local/share/Antix Digital/AICS Service/model_cache")

logging.getLogger("transformers").setLevel(logging.ERROR)


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_voxtral_model(cache_dir: str, model_id: str = MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16   # bfloat16 required by Voxtral

    os.environ["HF_HOME"] = cache_dir
    print(f"[model] Loading {model_id}")
    print(f"[model] Cache  : {cache_dir}")
    print(f"[model] Device : {device}")

    # Monkey-patch to prevent deepcopy crash inside transformers logger
    original_repr = ProcessorMixin.__repr__
    ProcessorMixin.__repr__ = lambda self: f"{self.__class__.__name__}(loaded)"
    try:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    finally:
        ProcessorMixin.__repr__ = original_repr

    # Transformers normally makes one model-sized allocator warm-up before
    # loading weights. Windows WDDM can reject that single large allocation
    # even when the tensors themselves fit. Skip only that optional warm-up
    # and continue streaming the checkpoint directly to the selected device.
    original_allocator_warmup = modeling_utils.caching_allocator_warmup
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            cache_dir=cache_dir,
        )
    finally:
        modeling_utils.caching_allocator_warmup = original_allocator_warmup
    model.eval()
    print("[model] Ready\n")
    return model, processor, device


# ─────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────
def _prepare_voxtral_inputs(
    audio_array,
    processor,
    language: str | None,
    target_delay_ms: int | None,
):
    """
    Build inputs via `mistral_common` transcription request so we can pass language/delay.
    """
    # The processor's feature extractor expects 16kHz audio arrays.
    audio = Audio(audio_array=audio_array, sampling_rate=processor.feature_extractor.sampling_rate, format="wav")
    req = TranscriptionRequest(
        audio=RawAudio.from_audio(audio),
        streaming=StreamingMode.OFFLINE,
        language=language or None,
        target_streaming_delay_ms=target_delay_ms,
    )
    tokenized = processor.tokenizer.tokenizer.encode_transcription(req)
    # `tokenized.audios` are serialized chunks; we want the decoded audio arrays for feature extraction.
    audio_arrays = [el.audio_array for el in tokenized.audios]

    # Ensure we return torch tensors (model expects `.shape`).
    text_encoding = processor.tokenizer(
        [tokenized.tokens],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    audio_encoding = processor.feature_extractor(
        audio_arrays,
        center=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
    )
    return {**text_encoding, **audio_encoding, "num_delay_tokens": processor.num_delay_tokens}


def transcribe_wav_bytes(
    wav_bytes: bytes,
    model,
    processor,
    device: str,
    language: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    target_delay_ms: int | None = None,
) -> str:
    """Write WAV bytes to a temp file and run Voxtral Realtime transcription."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)

        audio = Audio.from_file(tmp_path)
        audio.resample(processor.feature_extractor.sampling_rate)

        # Use a request-based path so language & delay can be respected.
        inputs = _prepare_voxtral_inputs(
            audio.audio_array,
            processor,
            language=language,
            target_delay_ms=target_delay_ms,
        )

        # Move tensors to the correct device.
        # Only cast floating tensors to the model dtype (keep input_ids as int64).
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    moved[k] = v.to(device, dtype=model.dtype)
                else:
                    moved[k] = v.to(device)
            else:
                moved[k] = v
        inputs = moved

        # Inference mode removes autograd version tracking in addition to
        # gradients and is measurably cheaper in the repeated live loop.
        with torch.inference_mode():
            gen_kwargs = {"max_new_tokens": int(max_new_tokens)}
            # Temperature only matters when sampling is enabled.
            if temperature and float(temperature) > 0:
                gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
            outputs = model.generate(**inputs, **gen_kwargs)

        text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return text
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# PCM → WAV helper
# ─────────────────────────────────────────────
def pcm_to_wav(pcm_data: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def _yt_dlp_base_cmd() -> list[str]:
    """
    Return the best way to invoke yt-dlp.

    On Windows, the venv-installed `yt-dlp.exe` may not be on PATH for child
    processes, so we fall back to `python -m yt_dlp` using the current venv.
    """
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def _find_deno() -> str | None:
    """Find Deno, including a fresh winget install not yet visible on PATH."""
    deno = shutil.which("deno") or shutil.which("deno.exe")
    if deno:
        return deno
    if os.name != "nt":
        return None
    package_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if not package_root.is_dir():
        return None
    matches = sorted(package_root.glob("DenoLand.Deno_*/*deno.exe"))
    return str(matches[-1]) if matches else None


def validate_runtime_environment() -> dict:
    """Check whether the required runtime tools are available before workflow startup."""
    issues: list[str] = []
    ffmpeg_path = shutil.which("ffmpeg")
    yt_dlp_path = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    deno_path = _find_deno()
    python_path = sys.executable
    yt_dlp_module_available = importlib.util.find_spec("yt_dlp") is not None

    if not ffmpeg_path:
        issues.append("ffmpeg is not installed or not on PATH")
    if not yt_dlp_path and not yt_dlp_module_available:
        issues.append("yt-dlp is not installed or not available via python -m yt_dlp")
    if not deno_path:
        issues.append("Deno is not installed; YouTube JavaScript challenge solving may fail")

    return {
        "ffmpeg_available": bool(ffmpeg_path),
        "yt_dlp_available": bool(yt_dlp_path or yt_dlp_module_available),
        "deno_available": bool(deno_path),
        "deno_path": deno_path,
        "python_executable": python_path,
        "issues": issues,
    }


# ─────────────────────────────────────────────
# YouTube stream URL resolution
# ─────────────────────────────────────────────
def _get_stream_url(
    youtube_url: str,
    format_selector: str,
    cookie_browser: str | None = None,
    stream_kind: str = "stream",
) -> str:
    """
    Use yt-dlp to resolve the best audio stream URL.
    For live streams this returns the HLS / DASH manifest URL — ffmpeg handles both.
    """
    cmd = _yt_dlp_base_cmd() + [
        "--no-playlist",
        "--no-warnings",
        "-f", format_selector,
        "--get-url",
    ]
    deno_path = _find_deno()
    if deno_path:
        cmd.extend(["--js-runtimes", f"deno:{deno_path}"])
    allowed_browsers = {"edge", "chrome", "firefox", "brave"}
    selected_browser = (cookie_browser or "").strip().lower()
    if selected_browser:
        if selected_browser not in allowed_browsers:
            raise RuntimeError(f"Unsupported cookie browser: {selected_browser}")
        cmd.extend(["--cookies-from-browser", selected_browser])
    cmd.append(youtube_url)
    auth_mode = f" using {selected_browser} sign-in" if selected_browser else ""
    print(f"[yt-dlp] Resolving {stream_kind} URL{auth_mode} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        details = result.stderr.strip()
        if "Could not copy Chrome cookie database" in details:
            raise RuntimeError(
                "Chrome is still using its cookie database. Fully exit Chrome "
                "(including background processes), then retry, or select Edge/Firefox."
            )
        if "Sign in to confirm you’re not a bot" in details and not selected_browser:
            raise RuntimeError(
                "YouTube requires authentication for this stream. "
                "Select your signed-in browser under YouTube Sign-in and retry."
            )
        raise RuntimeError(f"yt-dlp failed:\n{details}")

    # yt-dlp may return multiple lines (audio + video); take the first
    url = result.stdout.strip().splitlines()[0]
    if not url:
        raise RuntimeError("yt-dlp returned an empty URL")
    print(f"[yt-dlp] {stream_kind.capitalize()} URL resolved\n")
    return url


def get_audio_stream_url(youtube_url: str, cookie_browser: str | None = None) -> str:
    """Resolve the best audio-only stream for transcription."""
    return _get_stream_url(
        youtube_url,
        "bestaudio/best",
        cookie_browser=cookie_browser,
        stream_kind="audio stream",
    )


def get_video_stream_url(youtube_url: str, cookie_browser: str | None = None) -> str:
    """Resolve a muxed live format suitable for the authenticated UI preview."""
    return _get_stream_url(
        youtube_url,
        "best[height<=720]/best",
        cookie_browser=cookie_browser,
        stream_kind="video stream",
    )


# ─────────────────────────────────────────────
# ffmpeg audio capture thread
# ─────────────────────────────────────────────
def stream_audio(stream_url: str, chunk_seconds: float,
                 audio_queue: queue.Queue, stop_event: threading.Event,
                 telemetry: dict | None = None,
                 live_audio_queue: queue.Queue | None = None):
    """
    Run ffmpeg to pull audio from `stream_url`, decode to 16kHz mono PCM,
    then push fixed-size chunks into `audio_queue`.
    """
    chunk_bytes = int(SAMPLE_RATE * chunk_seconds * SAMPLE_WIDTH * CHANNELS)

    ffmpeg_cmd = [
        "ffmpeg",
        # HLS servers may initially expose several buffered segments. Read at
        # media speed so a fast connection cannot dump them into the caption
        # UI in a burst before reaching the live edge.
        "-re",
        # FFmpeg otherwise starts a few HLS segments behind the live edge.
        # Begin with the newest available segment for minimum broadcast lag.
        "-live_start_index",       "-1",
        "-reconnect",              "1",
        "-reconnect_streamed",     "1",
        "-reconnect_delay_max",    "5",
        "-i",                      stream_url,
        "-vn",                     # drop video
        "-ar",                     str(SAMPLE_RATE),
        "-ac",                     str(CHANNELS),
        "-f",                      "s16le",   # raw signed 16-bit little-endian PCM
        "-loglevel",               "quiet",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    buf = b""
    live_buf = b""
    live_slice_bytes = int(SAMPLE_RATE * 0.5 * SAMPLE_WIDTH * CHANNELS)
    captured_live_seconds = 0.0
    try:
        while not stop_event.is_set():
            data = proc.stdout.read(4096)
            if not data:
                print("[ffmpeg] Stream ended.")
                if telemetry is not None:
                    telemetry["source_ended"] = True
                break
            if telemetry is not None:
                telemetry["last_audio_received"] = time.monotonic()
                telemetry["source_ended"] = False
            buf += data
            if live_audio_queue is not None:
                live_buf += data
                while len(live_buf) >= live_slice_bytes:
                    live_slice = live_buf[:live_slice_bytes]
                    live_buf = live_buf[live_slice_bytes:]
                    captured_live_seconds += 0.5
                    item = (live_slice, captured_live_seconds)
                    try:
                        live_audio_queue.put_nowait(item)
                    except queue.Full:
                        # Diarization is a live side-channel: discard only its
                        # oldest unprocessed slice and keep capture/transcription
                        # lossless and unblocked.
                        try:
                            live_audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            live_audio_queue.put_nowait(item)
                        except queue.Full:
                            pass
            while len(buf) >= chunk_bytes:
                chunk = buf[:chunk_bytes]
                buf   = buf[chunk_bytes:]
                try:
                    # Lossless mode: wait for the consumer instead of dropping
                    # any captured speech. With an unbounded live-session queue
                    # this returns immediately under normal operation.
                    audio_queue.put(chunk)
                except queue.Full:
                    # Retained for callers that deliberately provide a bounded
                    # queue: apply backpressure until space becomes available.
                    audio_queue.put(chunk)
    finally:
        proc.kill()
        proc.wait()


# ─────────────────────────────────────────────
# Fallback: send chunk to running HTTP server
# ─────────────────────────────────────────────
def transcribe_via_api(wav_bytes: bytes, language: str,
                        api_url: str = "http://localhost:8002/transcribe") -> str:
    import requests
    files  = {"audio": ("chunk.wav", wav_bytes, "audio/wav")}
    data   = {
        "stream_id": "yt_live",
        "audio_id":  str(time.time()),
        "enable_caps": "false",
        "language": language,
    }
    resp = requests.post(api_url, files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json().get("result", {}).get("text", "").strip()


# ─────────────────────────────────────────────
# Silence check (simple energy threshold)
# ─────────────────────────────────────────────
def is_silent(pcm_data: bytes, threshold: float = 200.0) -> bool:
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples ** 2)) if len(samples) else 0.0
    return rms < threshold


# ─────────────────────────────────────────────
# Boundary word deduplication
# ─────────────────────────────────────────────
def strip_overlap(new_text: str, prev_tail_words: list) -> str:
    """
    Remove words at the start of new_text that were already emitted in the
    previous chunk (they came from the overlap audio that was prepended).
    Comparison is case- and punctuation-insensitive.
    """
    if not prev_tail_words or not new_text:
        return new_text

    def normalize(w: str) -> str:
        return w.lower().strip(".,!?;:\"'")

    new_words = new_text.split()
    max_check = min(len(prev_tail_words), len(new_words))
    for length in range(max_check, 0, -1):
        if [normalize(w) for w in new_words[:length]] == \
           [normalize(w) for w in prev_tail_words[-length:]]:
            return " ".join(new_words[length:]).strip()
    return new_text


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Real-time YouTube live stream transcription with Voxtral-Mini-4B-Realtime-2602",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("url",
                        help="YouTube live stream URL (or any yt-dlp-supported URL)")
    parser.add_argument("--chunk-seconds", type=float, default=2.0,
                        help="Audio chunk length in seconds (shorter = more latency, longer = better accuracy)")
    parser.add_argument("--language", default="en",
                        help="Transcription language code (e.g. en, fr, de, es, ...)")
    parser.add_argument("--model-cache", default=DEFAULT_CACHE_DIR,
                        help="Directory to cache / load the Voxtral model weights")
    parser.add_argument("--use-api", action="store_true",
                        help="Send chunks to the running HTTP server (port 8002) instead of loading the model locally")
    parser.add_argument("--api-url", default="http://localhost:8002/transcribe",
                        help="HTTP transcription API endpoint (used with --use-api)")
    parser.add_argument("--silence-rms", type=float, default=200.0,
                        help="RMS energy below which a chunk is treated as silence and skipped")
    args = parser.parse_args()

    runtime_status = validate_runtime_environment()
    if runtime_status["issues"]:
        print("[runtime] Environment issues detected:")
        for issue in runtime_status["issues"]:
            print(f"  - {issue}")

    # ── Load model (unless using HTTP API) ──────────────────────────────────
    model = processor = device = None
    if not args.use_api:
        model, processor, device = load_voxtral_model(args.model_cache)

    # ── Resolve YouTube stream URL ───────────────────────────────────────────
    stream_url = get_audio_stream_url(args.url)

    # ── Start audio capture thread ───────────────────────────────────────────
    audio_queue = queue.Queue(maxsize=20)
    stop_event  = threading.Event()

    capture_thread = threading.Thread(
        target=stream_audio,
        args=(stream_url, args.chunk_seconds, audio_queue, stop_event),
        daemon=True,
        name="AudioCapture",
    )
    capture_thread.start()

    # ── Handle Ctrl-C cleanly ────────────────────────────────────────────────
    def _shutdown(sig, frame):
        print("\n[info] Stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  YouTube Live Stream — Real-time Transcription")
    print(f"  Model    : {MODEL_ID}")
    print(f"  Chunk    : {args.chunk_seconds}s  |  Language: {args.language}")
    print(f"  Backend  : {'HTTP API @ ' + args.api_url if args.use_api else 'Local GPU/CPU'}")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    chunk_num = 0
    tail_bytes     = int(SAMPLE_RATE * TAIL_SECONDS * SAMPLE_WIDTH * CHANNELS)
    prev_tail_pcm  : bytes = b""   # last TAIL_SECONDS of previous speech chunk
    prev_tail_words: list  = []    # last few words already printed (for dedup)

    while not stop_event.is_set():
        try:
            pcm_data = audio_queue.get(timeout=5)
        except queue.Empty:
            if not capture_thread.is_alive():
                print("[info] Audio stream finished.")
                break
            print("[info] Waiting for audio ...")
            continue

        # Skip silent chunks to save GPU time
        if is_silent(pcm_data, args.silence_rms):
            # Silence = natural boundary; reset overlap state
            prev_tail_pcm   = b""
            prev_tail_words = []
            continue

        # ── Prepend tail of previous chunk to preserve boundary words ────
        combined_pcm  = prev_tail_pcm + pcm_data
        prev_tail_pcm = pcm_data[-tail_bytes:]   # save for next iteration

        wav_bytes = pcm_to_wav(combined_pcm)
        chunk_num += 1
        ts = time.strftime("%H:%M:%S")
        t0 = time.time()

        try:
            if args.use_api:
                text = transcribe_via_api(wav_bytes, args.language, args.api_url)
            else:
                text = transcribe_wav_bytes(wav_bytes, model, processor, device, args.language)
        except Exception as exc:
            print(f"[{ts}] ERROR transcribing chunk {chunk_num}: {exc}")
            continue

        elapsed = time.time() - t0

        # Remove words already printed from the overlap region
        text = strip_overlap(text, prev_tail_words)

        if text:
            prev_tail_words = text.split()[-3:]   # keep last 3 words for next dedup
            print(f"[{ts}] {text}  ({elapsed:.2f}s)")
        # Silently discard empty outputs — no need to clutter the terminal

    stop_event.set()
    print("\nDone.")


if __name__ == "__main__":
    main()
