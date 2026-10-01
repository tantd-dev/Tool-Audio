"""
Whisper engine executing ASR models via whisper-cli (whisper.cpp).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from core.audio import convert_to_16k_mono_wav, is_wav_16k_mono
from core.exceptions import MethodError
from core.models import whisper_hint


def _exe_ok(exe: str) -> bool:
    return bool(exe) and (Path(exe).exists() or bool(shutil.which(exe)))


def run_whisper(exe: str, profile: dict, audio: str, lang: str, threads: int) -> tuple[str, dict[str, Any]]:
    """Execute whisper-cli (whisper.cpp) for ASR transcription."""
    if not _exe_ok(exe):
        raise MethodError(
            "whisper-cli not found. Download whisper.cpp binaries and set the whisper-cli path in Tools."
        )
    model = profile["model"]
    if not Path(model).is_file():
        raise MethodError(f"Model file does not exist: {model}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = audio
        converted = False
        if not is_wav_16k_mono(audio):
            out_wav = str(Path(tmp) / "input_16k.wav")
            if convert_to_16k_mono_wav(audio, out_wav):
                wav = out_wav
                converted = True

        cmd = [
            exe,
            "-m", model,
            "-f", wav,
            "-l", (lang or "auto"),
            "-nt",
            "-t", str(threads),
        ]
        cwd = str(Path(exe).parent) if Path(exe).exists() else None
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd
        )

    raw = (proc.stdout or "").strip()
    extra = {
        "return_code": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": (proc.stderr or "").strip(),
        "command": cmd,
        "converted_to_16k_wav": converted,
    }
    if proc.returncode != 0 or not raw:
        extra["hint"] = whisper_hint(model)
        raise MethodError(f"whisper-cli failed (exit code {proc.returncode}) or returned no text.", extra)
    return raw, extra
