"""
Audio inspection, validation, and conversion helpers.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


def get_audio_duration(path: str) -> float | None:
    """
    Get audio duration in seconds.
    Tries Python standard library wave first, then ffprobe, and finally ffmpeg.
    """
    # 1. WAV with stdlib
    try:
        with wave.open(path, "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except Exception:
        pass

    # 2. ffprobe
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=20)
            if proc.returncode == 0:
                return float(proc.stdout.strip())
        except Exception:
            pass

    # 3. ffmpeg as fallback (works for mp3, m4a, flac, ogg, etc.)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            proc = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True, timeout=20)
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
            if m:
                return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        except Exception:
            pass

    return None


def is_wav_16k_mono(path: str) -> bool:
    """Check if the given file is already a 16kHz 16-bit mono WAV file."""
    try:
        with wave.open(path, "rb") as w:
            return w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    except Exception:
        return False


def convert_to_16k_mono_wav(src_path: str, dst_path: str) -> bool:
    """Convert audio file to 16kHz mono 16-bit PCM WAV using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst_path],
            capture_output=True, text=True, timeout=60
        )
        return proc.returncode == 0 and Path(dst_path).is_file()
    except Exception:
        return False


def prepare_api_audio(path: str) -> tuple[str, str]:
    """
    Prepare audio for OpenAI / LM Studio chat API.
    Directly returns base64 and format for wav / mp3.
    Converts other formats to 16kHz wav using ffmpeg.
    """
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in ("wav", "mp3"):
        return base64.b64encode(Path(path).read_bytes()).decode("ascii"), ext

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f".{ext} needs ffmpeg for conversion; use .wav/.mp3 or install ffmpeg.")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "converted.wav"
        proc = subprocess.run([ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", "16000", str(out)],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"ffmpeg conversion failed:\n{(proc.stderr or '')[-800:]}")
        return base64.b64encode(out.read_bytes()).decode("ascii"), "wav"
