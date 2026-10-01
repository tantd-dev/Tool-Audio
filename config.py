"""
Configuration and constants for Local Audio Model Benchmark.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

APP_TITLE = "Local Audio Model Benchmark"
DEFAULT_MODEL_DIR = Path.home() / ".lmstudio" / "models"
DEFAULT_SERVER = "http://localhost:1234"

MODE_AUTO = "Auto"
MODE_API = "LM Studio API only"
MODE_CLI = "llama-mtmd-cli only"

WHISPER_HINT = (
    "whisper-cli (whisper.cpp) loads ggml Whisper models, usually files named ggml-*.bin "
    "(e.g. ggml-large-v3-turbo-q8_0.bin from huggingface.co/ggerganov/whisper.cpp). "
    "A Whisper .gguf converted for another runtime may not load in whisper-cli."
)

# whisper.cpp reads its own ggml format (magic "lmgg"); a real GGUF file (magic "GGUF") gives "bad magic".
WHISPER_DOWNLOADS = {
    "ggml-large-v3-turbo-q8_0.bin  (~0.9 GB, best quality/size)": "ggml-large-v3-turbo-q8_0.bin",
    "ggml-large-v3-turbo-q5_0.bin  (~0.6 GB, smaller)": "ggml-large-v3-turbo-q5_0.bin",
    "ggml-large-v3-turbo.bin  (~1.6 GB, F16)": "ggml-large-v3-turbo.bin",
    "ggml-large-v3-q5_0.bin  (~1.1 GB, non-turbo)": "ggml-large-v3-q5_0.bin",
    "ggml-small.bin  (~0.5 GB)": "ggml-small.bin",
    "ggml-base.bin  (~0.15 GB)": "ggml-base.bin",
}
WHISPER_DOWNLOAD_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}"
WHISPER_DOWNLOAD_DIR = DEFAULT_MODEL_DIR / "ggerganov" / "whisper.cpp"

MMPROJ_PREFERENCE = ("f16", "bf16", "f32")


def utc_now() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def find_exe(names: list[str]) -> str:
    """Find the first existing executable name in system PATH."""
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    return ""
