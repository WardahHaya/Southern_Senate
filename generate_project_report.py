"""Generate the client-ready project report as a styled PDF."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "deliverables"
PDF_PATH = OUT_DIR / "AI_Live_Transcription_Project_Report.pdf"
PREVIEW_DIR = OUT_DIR / "report_previews"

NAVY = "#0B1526"
PANEL = "#142238"
BLUE = "#3B82F6"
CYAN = "#2DD4BF"
RED = "#EF4444"
GOLD = "#F59E0B"
GREEN = "#22C55E"
INK = "#172033"
MUTED = "#526078"
LIGHT = "#F4F7FB"
LINE = "#D8E0EC"
WHITE = "#FFFFFF"


def canvas(title: str | None = None, subtitle: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0.075, 0.935, title, fontsize=19, fontweight="bold", color=NAVY, va="top")
        if subtitle:
            ax.text(0.075, 0.902, subtitle, fontsize=9.5, color=MUTED, va="top")
        ax.plot([0.075, 0.925], [0.875, 0.875], color=BLUE, linewidth=2)
    return fig, ax


def footer(ax, page_no: int):
    ax.plot([0.075, 0.925], [0.055, 0.055], color=LINE, linewidth=0.8)
    ax.text(0.075, 0.035, "AI-Powered Live Transcription & Speaker Diarization", fontsize=7.5, color=MUTED)
    ax.text(0.925, 0.035, str(page_no), fontsize=8, color=MUTED, ha="right")


def wrapped(ax, text: str, x: float, y: float, width: int = 92, size: float = 9.2,
            color: str = INK, weight: str = "normal", line_height: float = 0.019):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=False))
    ax.text(x, y, "\n".join(lines), fontsize=size, color=color, va="top",
            fontweight=weight, linespacing=1.35)
    return y - line_height * len(lines)


def heading(ax, text: str, y: float, color: str = NAVY):
    ax.text(0.075, y, text, fontsize=12.5, fontweight="bold", color=color, va="top")
    return y - 0.032


def bullets(ax, items: list[str], y: float, width: int = 85, size: float = 9.0,
            color: str = INK, bullet_color: str = BLUE):
    for item in items:
        lines = textwrap.wrap(item, width=width, break_long_words=False)
        ax.text(0.083, y, "•", fontsize=size + 2, color=bullet_color, va="top", fontweight="bold")
        ax.text(0.105, y, "\n".join(lines), fontsize=size, color=color, va="top", linespacing=1.3)
        y -= 0.019 * len(lines) + 0.010
    return y


def info_card(ax, x, y, w, h, title, value, accent=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=LIGHT, edgecolor=LINE, linewidth=0.8))
    ax.add_patch(FancyBboxPatch((x, y), 0.008, h, boxstyle="round,pad=0,rounding_size=0.004",
                                facecolor=accent, edgecolor=accent))
    ax.text(x + 0.025, y + h - 0.024, title.upper(), fontsize=7.2, color=MUTED, fontweight="bold")
    ax.text(x + 0.025, y + 0.026, value, fontsize=12, color=NAVY, fontweight="bold")


def table(ax, headers, rows, x=0.075, y=0.84, widths=None, row_h=0.043, font_size=7.7):
    if widths is None:
        widths = [0.85 / len(headers)] * len(headers)
    total = sum(widths)
    widths = [w * 0.85 / total for w in widths]
    cx = x
    for header, w in zip(headers, widths):
        ax.add_patch(plt.Rectangle((cx, y - row_h), w, row_h, facecolor=NAVY, edgecolor=WHITE, linewidth=0.5))
        ax.text(cx + 0.008, y - row_h / 2, header, fontsize=font_size, color=WHITE,
                fontweight="bold", va="center")
        cx += w
    y -= row_h
    for ridx, row in enumerate(rows):
        height = row_h
        wrapped_cells = []
        max_lines = 1
        for value, w in zip(row, widths):
            chars = max(12, int(w * 120))
            cell_lines = textwrap.wrap(str(value), width=chars, break_long_words=False) or [""]
            wrapped_cells.append(cell_lines)
            max_lines = max(max_lines, len(cell_lines))
        height = max(row_h, 0.017 * max_lines + 0.015)
        cx = x
        bg = WHITE if ridx % 2 == 0 else LIGHT
        for cell_lines, w in zip(wrapped_cells, widths):
            ax.add_patch(plt.Rectangle((cx, y - height), w, height, facecolor=bg, edgecolor=LINE, linewidth=0.5))
            ax.text(cx + 0.008, y - 0.012, "\n".join(cell_lines), fontsize=font_size, color=INK,
                    va="top", linespacing=1.25)
            cx += w
        y -= height
    return y


def box(ax, x, y, w, h, title, text="", color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                facecolor=WHITE, edgecolor=color, linewidth=1.4))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top",
            fontsize=9, color=color, fontweight="bold")
    if text:
        ax.text(x + w / 2, y + h / 2 - 0.012, text, ha="center", va="center",
                fontsize=7.2, color=INK, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.2))


def add_cover(pdf):
    fig, ax = canvas()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=NAVY))
    ax.add_patch(plt.Rectangle((0, 0.77), 1, 0.23, facecolor="#101F36"))
    ax.add_patch(plt.Circle((0.83, 0.87), 0.17, facecolor=BLUE, alpha=0.12, edgecolor="none"))
    ax.add_patch(plt.Circle((0.88, 0.82), 0.09, facecolor=CYAN, alpha=0.18, edgecolor="none"))
    ax.text(0.085, 0.82, "PROJECT REPORT", fontsize=10, color=CYAN, fontweight="bold")
    ax.text(0.085, 0.72, "AI-Powered Real-Time\nLive Transcription &\nSpeaker Diarization", fontsize=30,
            color=WHITE, fontweight="bold", va="top", linespacing=1.05)
    ax.text(0.085, 0.49, "A locally hosted, GPU-accelerated monitoring platform for\n"
            "YouTube live streams and US Senate proceedings.", fontsize=13,
            color="#C8D4E6", va="top", linespacing=1.45)
    ax.plot([0.085, 0.44], [0.405, 0.405], color=CYAN, linewidth=3)
    ax.text(0.085, 0.35, "Prepared for", fontsize=8, color="#8FA3C0", fontweight="bold")
    ax.text(0.085, 0.318, "US Client — Controlled Pilot Evaluation", fontsize=12, color=WHITE)
    ax.text(0.085, 0.255, "Prepared by", fontsize=8, color="#8FA3C0", fontweight="bold")
    ax.text(0.085, 0.223, "Southern Senate Project Team", fontsize=12, color=WHITE)
    ax.text(0.085, 0.12, f"Version 1.1  |  {date.today().strftime('%B %d, %Y')}",
            fontsize=9, color="#AFC0D8")
    ax.text(0.915, 0.06, "CONFIDENTIAL — PILOT PROJECT", fontsize=7.5, color="#7E93B0", ha="right")
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def add_document_control(pdf, page):
    fig, ax = canvas("Document Control", "Project identity, status, audience, and purpose")
    rows = [
        ("Document title", "AI-Powered Real-Time Live Transcription & Speaker Diarization"),
        ("Version", "1.1"),
        ("Status", "Functional prototype — ready for controlled pilot testing"),
        ("Primary audience", "Executive sponsors, product owners, technical reviewers, and pilot operators"),
        ("Target use case", "US Senate hearings, speeches, committee sessions, interviews, and public proceedings"),
        ("Deployment model", "Local Windows workstation with NVIDIA GPU; browser dashboard bound to localhost"),
        ("Classification", "Client project report; operational details should be handled according to client policy"),
    ]
    table(ax, ["Field", "Details"], rows, y=0.83, widths=[0.23, 0.62], row_h=0.052, font_size=8.3)
    y = 0.39
    y = heading(ax, "Purpose of this report", y)
    wrapped(ax, "This report documents the business need, implemented system, end-to-end workflow, "
            "technical architecture, operational safeguards, validation evidence, limitations, and "
            "recommended roadmap. It is intended to support a client demonstration and a structured "
            "decision on controlled pilot deployment.", 0.075, y, width=100, size=9.4)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_exec_summary(pdf, page):
    fig, ax = canvas("Executive Summary", "A near-live monitoring platform designed for accountable public proceedings")
    info_card(ax, 0.075, 0.77, 0.25, 0.075, "Current state", "Pilot-ready prototype", GREEN)
    info_card(ax, 0.375, 0.77, 0.25, 0.075, "Models", "Voxtral Mini 3B + 4B", BLUE)
    info_card(ax, 0.675, 0.77, 0.25, 0.075, "Validation", "35 tests passed", CYAN)
    y = 0.72
    y = wrapped(ax, "The project is a locally hosted, GPU-accelerated system that monitors a live YouTube "
                "broadcast, continuously captures its audio, produces a near-live transcript, identifies "
                "speaker turns, assigns stable anonymous speaker labels, and presents video, text, colors, "
                "console output, and operational telemetry in one browser dashboard.", 0.075, y, width=104, size=10.2)
    y -= 0.018
    y = heading(ax, "Business value", y)
    y = bullets(ax, [
        "Accelerates monitoring and review of Senate hearings, floor speeches, confirmation hearings, briefings, and interviews.",
        "Keeps inference local, reducing reliance on paid cloud transcription APIs and improving operational control over media processing.",
        "Preserves queued audio through a disk-backed first-in, first-out spool when processing temporarily falls behind.",
        "Makes different voices easier to follow through stable labels, consistent colors, and retrospective correction of delayed diarization decisions.",
        "Gives operators real-time visibility into source health, transcript backlog, real-time factor, CPU, RAM, and GPU memory.",
    ], y, width=90)
    y -= 0.005
    y = heading(ax, "Positioning", y)
    wrapped(ax, "The current build is a functional prototype ready for controlled pilot testing. It is not "
            "an official record-generation platform and does not replace certified transcripts, stenographers, "
            "or human review. The design balances low delay, transcription completeness, and speaker accuracy; "
            "none of these can be treated as absolute under variable live-stream conditions.", 0.075, y, width=101, size=9.3)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_need_scope(pdf, page):
    fig, ax = canvas("Business Need, Objectives & Scope")
    y = 0.84
    y = heading(ax, "The need", y)
    y = wrapped(ax, "Public proceedings often contain hours of fast-moving testimony, questioning, and prepared "
                "remarks. Manual monitoring is expensive, difficult to search, and vulnerable to missed details. "
                "A near-live transcript with visible speaker turns provides a faster operational view while preserving "
                "the source video for context.", 0.075, y, width=101, size=9.5)
    y -= 0.016
    y = heading(ax, "Project objectives", y)
    y = bullets(ax, [
        "Transcribe continuous live speech with minimal practical delay and without intentionally discarding queued content.",
        "Detect speaker changes and maintain stable Speaker 1, Speaker 2, and subsequent identities across a session.",
        "Recognize a returning questioner in an A → B → A exchange instead of creating a false third identity.",
        "Synchronize authenticated video and transcription against the same source family.",
        "Provide an operator-friendly dashboard, export capability, console visibility, and resource telemetry.",
        "Support safe model loading, switching, and unloading on a 12 GB-class GPU.",
    ], y, width=91)
    y -= 0.008
    table(ax, ["In scope", "Out of scope for current prototype"], [
        ("YouTube live acquisition with browser-cookie authentication", "Certified or legally authoritative transcripts"),
        ("Local Voxtral transcription and Pyannote diarization", "Automatic verified naming of senators and witnesses"),
        ("Anonymous stable speaker labels and colors", "Perfect word-level speaker boundaries"),
        ("Authenticated video preview and transcript export", "Public internet deployment and multi-tenant access"),
        ("Operational telemetry and crash safeguards", "Formal WER/DER benchmark results"),
    ], y=y, widths=[0.425, 0.425], row_h=0.047, font_size=7.7)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_architecture(pdf, page):
    fig, ax = canvas("System Architecture", "Modular local processing with separate media, AI, reliability, and presentation layers")
    levels = [
        (0.77, "Presentation Layer", "Browser dashboard • authenticated video • transcript • controls • telemetry", BLUE),
        (0.64, "Application & API Layer", "Flask routes • session lifecycle • model lifecycle • Server-Sent Events", CYAN),
        (0.51, "Media Ingestion Layer", "yt-dlp • Firefox cookies • Deno/EJS • FFmpeg audio/video pipelines", GOLD),
        (0.38, "AI Inference Layer", "Voxtral transcription • Pyannote diarization • voice embeddings • identity reconciliation", RED),
        (0.25, "Reliability Layer", "Disk spool • retries • backlog metrics • duplicate-instance guard • CUDA cleanup", GREEN),
        (0.12, "Hardware Layer", "Windows • Python 3.11 • RTX 5070 • CUDA • local RAM and disk", "#7C3AED"),
    ]
    for y, title, text, color in levels:
        ax.add_patch(FancyBboxPatch((0.1, y), 0.8, 0.088, boxstyle="round,pad=0.012,rounding_size=0.015",
                                    facecolor=LIGHT, edgecolor=color, linewidth=1.4))
        ax.text(0.13, y + 0.058, title, fontsize=11, color=color, fontweight="bold", va="center")
        ax.text(0.13, y + 0.027, text, fontsize=8.1, color=INK, va="center")
    for idx in range(len(levels) - 1):
        arrow(ax, 0.5, levels[idx][0], 0.5, levels[idx + 1][0] + 0.09, MUTED)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_workflow(pdf, page):
    fig, ax = canvas("End-to-End Workflow", "Audio continuity and asynchronous speaker analysis converge in the live dashboard")
    y = 0.755
    xs = [0.07, 0.27, 0.47, 0.67]
    titles = ["YouTube Live", "Authenticated\nResolution", "FFmpeg Capture", "Disk Audio Spool"]
    texts = ["Public proceeding", "Firefox cookies\nDeno + EJS", "16 kHz mono PCM", "Ordered, lossless FIFO"]
    for x, title, text in zip(xs, titles, texts):
        box(ax, x, y, 0.16, 0.10, title, text, BLUE)
    for x in [0.23, 0.43, 0.63]:
        arrow(ax, x, y + 0.05, x + 0.04, y + 0.05)
    ax.text(0.075, 0.70, "TRANSCRIPTION BRANCH", fontsize=8, color=BLUE, fontweight="bold")
    txs = [0.10, 0.34, 0.58, 0.79]
    t_titles = ["Chunk + Tail", "Voxtral STT", "Text Deduplication", "Transcript Event"]
    t_texts = ["Boundary overlap", "CUDA inference", "Remove repeated words", "Time + entry ID"]
    for x, title, text in zip(txs, t_titles, t_texts):
        box(ax, x, 0.56, 0.15 if x < 0.79 else 0.14, 0.095, title, text, BLUE)
    for x1, x2 in [(0.25, 0.34), (0.49, 0.58), (0.73, 0.79)]:
        arrow(ax, x1, 0.607, x2, 0.607)
    arrow(ax, 0.75, y, 0.18, 0.655)
    ax.text(0.075, 0.48, "SPEAKER BRANCH", fontsize=8, color=RED, fontweight="bold")
    dxs = [0.10, 0.34, 0.58, 0.79]
    d_titles = ["8s Rolling Window", "Pyannote", "Stable Identity", "Speaker Update"]
    d_texts = ["Latest window ~0.75s", "Turn segmentation", "Overlap + voiceprint", "Label + color revision"]
    for x, title, text in zip(dxs, d_titles, d_texts):
        box(ax, x, 0.34, 0.15 if x < 0.79 else 0.14, 0.095, title, text, RED)
    for x1, x2 in [(0.25, 0.34), (0.49, 0.58), (0.73, 0.79)]:
        arrow(ax, x1, 0.387, x2, 0.387)
    arrow(ax, 0.75, y, 0.18, 0.435)
    box(ax, 0.29, 0.14, 0.42, 0.105, "Flask + Server-Sent Events",
        "Transcript and speaker-update events → browser dashboard → export", CYAN)
    arrow(ax, 0.86, 0.56, 0.58, 0.245, BLUE)
    arrow(ax, 0.86, 0.34, 0.62, 0.245, RED)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_features(pdf, page):
    fig, ax = canvas("Implemented Feature Matrix")
    rows = [
        ("Live acquisition", "Browser-cookie authentication, Deno/EJS challenge solving, authenticated audio and video resolution", "Implemented"),
        ("Speech-to-text", "Local Voxtral Mini 3B/4B selection, deterministic decoding, configurable chunk/tail/delay controls", "Implemented"),
        ("Content continuity", "Disk-backed FIFO spool, ordered processing, acknowledgement after successful inference", "Implemented"),
        ("Speaker diarization", "Latest-only overlapping Pyannote analysis with up to ten stable session speakers", "Implemented"),
        ("Stable identity", "Time-overlap reconciliation and asynchronous voiceprint matching for returning speakers", "Implemented"),
        ("Visual speaker tracking", "Speaker labels, stable colors, colored transcript borders, retrospective row correction", "Implemented"),
        ("Operator dashboard", "Authenticated video, live transcript, console, controls, status indicators, clear and export", "Implemented"),
        ("Telemetry", "CPU, RAM, GPU, VRAM, words, chunks, source state, backlog, and real-time factor", "Implemented"),
        ("Model lifecycle", "Safe load, switch, unload, CUDA cleanup, duplicate request coalescing", "Implemented"),
        ("Named public figures", "Automatic mapping of anonymous voices to verified senator/witness names", "Proposed"),
        ("Formal benchmark", "WER, DER, change-delay, and end-to-end latency evaluation on Senate material", "Pending"),
    ]
    table(ax, ["Capability", "Current implementation", "Status"], rows, y=0.84,
          widths=[0.19, 0.52, 0.14], row_h=0.041, font_size=7.1)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_stt(pdf, page):
    fig, ax = canvas("Speech-to-Text Pipeline", "Near-live processing designed to preserve audio continuity")
    y = 0.84
    y = bullets(ax, [
        "FFmpeg decodes the resolved live stream to 16 kHz, mono, signed 16-bit PCM audio.",
        "Audio is divided into configurable chunks; the verified live profile normally uses approximately two-second chunks.",
        "A short tail, normally 0.3 seconds, is prepended to the next chunk to reduce words being clipped at chunk boundaries.",
        "Text-level overlap removal eliminates repeated words introduced by the audio tail.",
        "Temperature defaults to 0.0 and output tokens are limited to reduce hallucination, rambling, and unnecessary latency.",
        "Silence-threshold processing avoids spending GPU inference time on sufficiently quiet chunks.",
        "Each chunk is written to a disk-backed spool and acknowledged only after processing, protecting continuity during transient inference slowdown.",
        "Each transcript decision remains tied to a bounded audio interval; oversized ten-second catch-up batches were removed because they obscured speaker changes.",
    ], y, width=91, size=9.2)
    y -= 0.01
    y = heading(ax, "Operational latency model", y)
    wrapped(ax, "Observed end-to-end timing is the sum of upstream YouTube broadcast delay, stream acquisition, "
            "chunk duration, model inference, browser delivery, and any backlog. The dashboard therefore reports "
            "real-time factor and backlog rather than claiming impossible zero latency. When the real-time factor "
            "remains below 1.0, inference is faster than incoming audio.", 0.075, y, width=102, size=9.3)
    info_card(ax, 0.075, 0.12, 0.25, 0.075, "Sample rate", "16 kHz mono", BLUE)
    info_card(ax, 0.375, 0.12, 0.25, 0.075, "Default tail", "0.3 seconds", CYAN)
    info_card(ax, 0.675, 0.12, 0.25, 0.075, "Decoding", "Temperature 0.0", GREEN)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_diarization(pdf, page):
    fig, ax = canvas("Speaker Diarization & Identity Management")
    rows = [
        ("Speaker-change detection", "Determines that the active voice has changed."),
        ("Diarization", "Segments audio by anonymous speaker turns."),
        ("Stable identity", "Maps window-local labels to persistent session identities."),
        ("Speaker labeling", "Displays Speaker 1, Speaker 2, and subsequent anonymous labels."),
        ("Color assignment", "Maintains a consistent visual color for each stable identity."),
    ]
    table(ax, ["Concept", "Role"], rows, y=0.84, widths=[0.27, 0.58], row_h=0.046, font_size=8.0)
    y = 0.54
    y = heading(ax, "Implemented method", y)
    y = bullets(ax, [
        "Pyannote analyzes the newest overlapping eight-second rolling window approximately every 0.75 seconds once enough speech is available.",
        "Window-local speaker labels are reconciled against the previous window using absolute timeline overlap.",
        "Voice embeddings are calculated asynchronously and compared with session profiles to support A → B → A identity reuse.",
        "The live registry supports up to ten stable speaker identities and reuses returning voice identities instead of inventing a new speaker for each turn.",
        "Transcription does not wait for diarization. Recent transcript rows retain entry IDs and audio positions.",
        "When diarization finishes later, a speaker-update event revises the affected row’s label, color, and border.",
    ], y, width=90, size=8.8)
    y -= 0.005
    y = heading(ax, "Important limitation", y)
    wrapped(ax, "Reliable voice identification requires a short evidence window. Rapid interruptions, overlapping speech, "
            "similar voices, broadcast mixing, and poor microphones remain difficult. Without word-level alignment, a "
            "speaker change inside one transcription chunk cannot always be placed at the exact word boundary.",
            0.075, y, width=101, size=8.8, color=MUTED)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_youtube_ui(pdf, page):
    fig, ax = canvas("Authenticated YouTube Integration & Operator Experience")
    y = 0.84
    y = heading(ax, "Authentication design", y)
    y = wrapped(ax, "YouTube may require “Sign in to confirm you’re not a bot.” The application does not request "
                "the operator’s username or password. Instead, yt-dlp reads cookies from an explicitly selected "
                "browser profile. Firefox was verified locally because Chrome and Edge can lock or encrypt their "
                "cookie databases through Windows protections.", 0.075, y, width=101, size=9.2)
    y -= 0.016
    y = bullets(ax, [
        "yt-dlp 2026.07.04 performs media extraction.",
        "yt-dlp-ejs and Deno 2.9.3 solve current YouTube JavaScript challenges.",
        "The video preview uses an authenticated server-side proxy instead of an iframe with a separate cookie context.",
        "Low-buffer FFmpeg flags, smaller proxy writes, and no-cache response headers reduce unnecessary preview drift.",
        "The preview displays CONNECTING, LIVE, or DELAYED status and offers a Jump to live control when browser playback falls behind.",
    ], y, width=91, size=8.9)
    y -= 0.005
    y = heading(ax, "Dashboard controls and feedback", y)
    controls = [
        ("Session", "URL input, Start/Stop, source state, authenticated preview"),
        ("Transcript", "Live rows, labels, colors, clear, TXT export"),
        ("Model", "Select, load, unload, safe switching"),
        ("Tuning", "Language, chunk, RMS, tokens, temperature, overlap, delay"),
        ("Operations", "Console, CPU/RAM/GPU, backlog, RTF, words, chunks"),
    ]
    table(ax, ["Area", "Functions"], controls, y=y, widths=[0.21, 0.64], row_h=0.045, font_size=8.0)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_stack(pdf, page):
    fig, ax = canvas("Technology Stack")
    rows = [
        ("Python 3.11", "Backend and AI orchestration", "Mature ecosystem and direct integration with model tooling"),
        ("Flask + SSE", "Local API and live events", "Simple threaded local server and low-overhead one-way updates"),
        ("HTML/CSS/JavaScript", "Operator dashboard", "Portable browser experience without a heavy frontend build"),
        ("PyTorch + CUDA", "GPU inference", "Local acceleration on NVIDIA RTX hardware"),
        ("Transformers", "Voxtral loading and inference", "Model configuration and generation runtime"),
        ("Mistral Common", "Audio protocol", "Voxtral streaming/transcription request structures"),
        ("Voxtral Mini Realtime", "Speech-to-text", "Open-weights, local near-live transcription"),
        ("Pyannote Audio", "Speaker diarization", "Speaker-turn segmentation and embedding support"),
        ("FFmpeg 8.1.2", "Media processing", "Resampling, decoding, reconnection, and video proxying"),
        ("yt-dlp + EJS", "YouTube extraction", "Cookie authentication and current challenge support"),
        ("Deno 2.9.3", "JavaScript runtime", "Recommended runtime for YouTube challenge solving"),
        ("NumPy / psutil", "Signal and telemetry", "Embedding math and system resource reporting"),
        ("PowerShell", "Windows launcher", "Environment checks and duplicate-instance protection"),
        ("unittest", "Automated validation", "Fast repeatable contract and behavior checks"),
    ]
    table(ax, ["Technology", "Role", "Rationale"], rows, y=0.84,
          widths=[0.22, 0.24, 0.39], row_h=0.036, font_size=6.7)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_reliability(pdf, page):
    fig, ax = canvas("Reliability, GPU Safety & Failure Recovery")
    y = 0.84
    y = heading(ax, "Runtime safeguards", y)
    y = bullets(ax, [
        "Disk-backed FIFO buffering preserves order and retains unacknowledged chunks through transient inference failure.",
        "FFmpeg reconnect options handle recoverable streamed-media interruptions.",
        "Model downloads and loading use retry/backoff behavior for transient hosting failures.",
        "Diarization requires three consecutive failures before it is disabled; transcription remains independent.",
        "Voice embedding is treated as an enhancement and cannot terminate transcription.",
        "A minimum two-gigabyte disk reserve is required before the live spool proceeds.",
        "Source idle time, processing state, real-time factor, and backlog are exposed to the operator.",
    ], y, width=90, size=8.8)
    y -= 0.005
    y = heading(ax, "Memory-safe model lifecycle", y)
    y = bullets(ax, [
        "Only one transcription model is intentionally held in memory.",
        "Switching clears model and processor references, runs garbage collection, empties CUDA cache, and attempts CUDA IPC cleanup.",
        "Duplicate model-load requests are coalesced to prevent accidental double loading.",
        "Unload stops active work first and reports freed/remaining PyTorch VRAM.",
        "The server checks port 7860 before background model loading. A duplicate app launch is rejected before CUDA allocation.",
        "The PowerShell launcher detects a healthy existing instance and exits safely.",
    ], y, width=90, size=8.8)
    info_card(ax, 0.075, 0.11, 0.25, 0.075, "Loaded model", "~8.25 GB allocated", RED)
    info_card(ax, 0.375, 0.11, 0.25, 0.075, "After unload", "0.00 GB allocated", GREEN)
    info_card(ax, 0.675, 0.11, 0.25, 0.075, "Duplicate launch", "Rejected pre-CUDA", BLUE)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_security(pdf, page):
    fig, ax = canvas("Security, Privacy & Data Governance")
    y = 0.84
    y = heading(ax, "Current posture", y)
    y = bullets(ax, [
        "Speech recognition and speaker analysis run locally on the workstation.",
        "The application does not collect browser usernames or passwords.",
        "yt-dlp accesses cookies only from the operator-selected local browser profile.",
        "Resolved media URLs are temporary and are not displayed in application logs.",
        "Flask binds to 127.0.0.1, debug mode is disabled, and the dashboard is not publicly exposed by default.",
        "Transcript exports and runtime audio spools remain local and should be governed by an agreed retention policy.",
    ], y, width=91, size=9.0)
    y -= 0.015
    y = heading(ax, "Production controls recommended", y)
    rows = [
        ("Identity and access", "Authenticated users, role-based permissions, session timeout"),
        ("Transport", "HTTPS/TLS and secure reverse proxy"),
        ("Storage", "Encryption at rest, protected export folders, managed retention"),
        ("Auditability", "Operator actions, corrections, exports, model/configuration history"),
        ("Secrets", "Managed handling of tokens and browser-auth alternatives"),
        ("Network", "Firewall policy, allowlisted sources, monitored outbound access"),
        ("Governance", "Privacy review, records policy, incident response, data classification"),
    ]
    table(ax, ["Control area", "Recommended production measure"], rows, y=y,
          widths=[0.25, 0.60], row_h=0.044, font_size=7.8)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_validation(pdf, page):
    fig, ax = canvas("Testing & Validation Evidence", "Final presentation-readiness checks completed on the target workstation")
    rows = [
        ("Automated test suite", "35 tests passed", "PASS"),
        ("Python compilation", "Core backend, UI, diarization, and spool modules", "PASS"),
        ("Python dependency integrity", "No broken requirements reported", "PASS"),
        ("PowerShell launcher", "Parser completed with zero errors", "PASS"),
        ("Dashboard/API", "/, /status, /models, and /metrics returned HTTP 200", "PASS"),
        ("YouTube authentication", "Exact live audio and video URLs resolved with Firefox cookies", "PASS"),
        ("Media ingestion", "Five-second FFmpeg audio and video decode smoke tests", "PASS"),
        ("Model runtime", "Voxtral Mini 3B and 4B selectable with safe CUDA switching", "PASS"),
        ("Diarization runtime", "Pyannote initialized and reported ready", "PASS"),
        ("Model lifecycle", "Unload freed ~8.25 GB; reload completed successfully", "PASS"),
        ("Duplicate launch", "Rejected before GPU model allocation", "PASS"),
        ("Disk capacity", "~513 GB free on project drive during validation", "PASS"),
    ]
    table(ax, ["Check", "Evidence", "Result"], rows, y=0.84,
          widths=[0.22, 0.50, 0.13], row_h=0.041, font_size=7.3)
    ax.text(0.075, 0.16, "Metrics not yet formally measured", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(0.075, 0.125, "Word error rate (WER), diarization error rate (DER), speaker-change delay, "
            "and 95th-percentile end-to-end latency remain pending a representative Senate benchmark dataset.",
            fontsize=8.5, color=MUTED, va="top", wrap=True)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_risks(pdf, page):
    fig, ax = canvas("Risks & Mitigations")
    rows = [
        ("YouTube authentication changes", "Stream resolution may fail", "Keep yt-dlp/EJS/Deno current; support alternate official feeds", "Medium"),
        ("Network or source interruption", "Temporary missing live input", "FFmpeg reconnect, source-idle telemetry, disk spool", "Medium"),
        ("GPU out-of-memory", "Process crash or failed model load", "3B/4B selection, unload control, duplicate guards, one-model lifecycle", "Low"),
        ("Speaker overlap/cross-talk", "Incorrect speaker boundaries", "Retrospective updates, operator correction roadmap, human review", "High"),
        ("Similar voices", "False identity merge or split", "Voice profiles, overlap reconciliation, controlled benchmark tuning", "Medium"),
        ("Upstream live delay", "Video/text timing mismatch", "Authenticated proxy, same source family, live-edge status and jump control", "Medium"),
        ("Transcript hallucination/error", "Incorrect record", "Temperature 0, token bounds, source video, human verification", "Medium"),
        ("Local data retention", "Privacy or records exposure", "Formal retention, encryption, controlled exports", "Medium"),
        ("Single workstation failure", "Loss of service", "Production service supervision, health checks, redundant deployment", "Medium"),
    ]
    table(ax, ["Risk", "Impact", "Mitigation", "Residual"], rows, y=0.84,
          widths=[0.20, 0.19, 0.38, 0.08], row_h=0.043, font_size=6.8)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_roadmap(pdf, page):
    fig, ax = canvas("Recommended Roadmap for a US Senate Pilot")
    phases = [
        ("Phase 1 — Current Prototype", "Completed", [
            "Local live transcription and authenticated video",
            "Anonymous stable speakers and colors",
            "Operational telemetry and export",
            "Crash and GPU lifecycle safeguards",
        ], GREEN),
        ("Phase 2 — Pilot Validation", "Next", [
            "Build representative Senate evaluation set",
            "Measure WER, DER, latency, and turn-delay KPIs",
            "Run multi-hour endurance and interruption tests",
            "Collect operator usability feedback",
        ], BLUE),
        ("Phase 3 — Senate Enhancement", "Planned", [
            "Committee rosters and hearing metadata",
            "Operator rename/merge/split controls",
            "Word-level alignment and citation timestamps",
            "Entities, bills, topics, alerts, and Q&A grouping",
        ], GOLD),
        ("Phase 4 — Production Hardening", "Future", [
            "Authentication, RBAC, HTTPS, and audit trails",
            "Service supervision and centralized monitoring",
            "Retention, encryption, and records governance",
            "Redundancy and approved deployment architecture",
        ], RED),
    ]
    y = 0.80
    for title, status, items, color in phases:
        ax.add_patch(FancyBboxPatch((0.08, y - 0.14), 0.84, 0.145,
                                    boxstyle="round,pad=0.01,rounding_size=0.014",
                                    facecolor=LIGHT, edgecolor=color, linewidth=1.3))
        ax.text(0.105, y - 0.025, title, fontsize=10.5, color=color, fontweight="bold")
        ax.text(0.885, y - 0.025, status.upper(), fontsize=7.5, color=color, fontweight="bold", ha="right")
        for idx, item in enumerate(items):
            col = 0 if idx < 2 else 1
            row = idx if idx < 2 else idx - 2
            ax.text(0.115 + col * 0.41, y - 0.065 - row * 0.035, "• " + item,
                    fontsize=7.7, color=INK, va="top")
        y -= 0.175
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_metrics(pdf, page):
    fig, ax = canvas("Pilot Success Metrics")
    rows = [
        ("Transcript latency", "Median and 95th percentile from source audio to displayed text", "Pending benchmark"),
        ("Transcription quality", "Word error rate by hearing type, speaker, and audio condition", "Pending benchmark"),
        ("Diarization quality", "Diarization error rate and speaker confusion rate", "Pending benchmark"),
        ("Turn responsiveness", "Median speaker-change detection and correction delay", "Pending benchmark"),
        ("Identity stability", "A → B → A returning-speaker accuracy", "Pending benchmark"),
        ("Continuity", "Percentage of captured audio processed; dropped-audio seconds", "Architecture targets zero intentional drops"),
        ("Throughput", "Average/maximum backlog and real-time factor", "Available in dashboard"),
        ("Reliability", "Session uptime and recovery time after source interruption", "Pending endurance test"),
        ("Resource stability", "Peak VRAM/RAM and absence of growth over multi-hour runs", "Pending endurance test"),
        ("Operator quality", "Correction rate, task completion time, and usability score", "Pending pilot feedback"),
    ]
    table(ax, ["KPI", "Definition", "Current state"], rows, y=0.84,
          widths=[0.22, 0.43, 0.20], row_h=0.043, font_size=7.2)
    y = 0.26
    y = heading(ax, "Recommended acceptance approach", y)
    wrapped(ax, "Agree target thresholds with the client before pilot scoring. Benchmark against manually reviewed "
            "Senate material covering prepared remarks, rapid questioning, interruptions, remote witnesses, accents, "
            "poor microphones, and overlapping speech. Report aggregate results and failure-case slices.",
            0.075, y, width=100, size=8.8)
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_limitations_conclusion(pdf, page):
    fig, ax = canvas("Limitations & Conclusion")
    y = 0.84
    y = heading(ax, "Known limitations", y)
    y = bullets(ax, [
        "Zero latency is physically impossible; the product is near-live and inherits YouTube’s upstream broadcast delay.",
        "YouTube authentication and anti-bot mechanisms can change without notice.",
        "Cross-talk, interruptions, similar voices, noise, and broadcast mixing reduce diarization accuracy.",
        "Anonymous speaker numbers are not verified personal identities.",
        "Exact word-level speaker boundaries require an additional alignment stage.",
        "A 12 GB-class GPU limits safe model size and available concurrency.",
        "Official records, quotations, legal use, and publication require human review against the source.",
    ], y, width=91, size=8.9)
    y -= 0.015
    y = heading(ax, "Conclusion", y)
    y = wrapped(ax, "The project has reached a strong prototype milestone: authenticated live media acquisition, local "
                "GPU transcription, continuous audio protection, stable anonymous speaker handling, retrospective color "
                "correction, operator telemetry, model lifecycle controls, and presentation-focused crash safeguards are "
                "integrated and locally validated.", 0.075, y, width=101, size=9.5)
    y -= 0.015
    wrapped(ax, "The recommended next step is a controlled Senate pilot centered on measurable transcription quality, "
            "speaker accuracy, end-to-end delay, multi-hour stability, and operator correction workflows. With those "
            "results, the client can make an evidence-based decision on Senate-specific enhancement and production hardening.",
            0.075, y, width=101, size=9.5, weight="bold")
    ax.add_patch(FancyBboxPatch((0.075, 0.11), 0.85, 0.085, boxstyle="round,pad=0.012,rounding_size=0.015",
                                facecolor=NAVY, edgecolor=NAVY))
    ax.text(0.5, 0.153, "CURRENT STATUS: FUNCTIONAL PROTOTYPE — READY FOR CONTROLLED PILOT TESTING",
            ha="center", va="center", fontsize=9.5, color=WHITE, fontweight="bold")
    footer(ax, page)
    pdf.savefig(fig)
    plt.close(fig)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_PATH) as pdf:
        add_cover(pdf)
        pages = [
            add_document_control,
            add_exec_summary,
            add_need_scope,
            add_architecture,
            add_workflow,
            add_features,
            add_stt,
            add_diarization,
            add_youtube_ui,
            add_stack,
            add_reliability,
            add_security,
            add_validation,
            add_risks,
            add_roadmap,
            add_metrics,
            add_limitations_conclusion,
        ]
        for page_no, fn in enumerate(pages, start=2):
            fn(pdf, page_no)
        metadata = pdf.infodict()
        metadata["Title"] = "AI-Powered Real-Time Live Transcription & Speaker Diarization"
        metadata["Author"] = "Southern Senate Project Team"
        metadata["Subject"] = "Client-ready project report and controlled pilot proposal"
        metadata["Keywords"] = "live transcription, speaker diarization, Voxtral, Pyannote, US Senate"
    print(PDF_PATH)


class _PreviewSink:
    def __init__(self, name: str):
        self.name = name

    def savefig(self, fig, **_kwargs):
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_DIR / f"{self.name}.png", dpi=130, facecolor=fig.get_facecolor())


def render_previews():
    add_cover(_PreviewSink("01_cover"))
    add_architecture(_PreviewSink("05_architecture"), 5)
    add_workflow(_PreviewSink("06_workflow"), 6)
    add_validation(_PreviewSink("14_validation"), 14)


if __name__ == "__main__":
    build()
    render_previews()
