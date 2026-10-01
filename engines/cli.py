"""
CLI engine executing multimodal models via llama-mtmd-cli.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.exceptions import MethodError


def _exe_ok(exe: str) -> bool:
    return bool(exe) and (Path(exe).exists() or bool(shutil.which(exe)))


def run_cli(cli: str, profile: dict, audio: str, prompt: str, threads: int) -> tuple[str, dict[str, Any]]:
    """Execute llama-mtmd-cli with audio and mmproj projector."""
    if not _exe_ok(cli):
        raise MethodError("llama-mtmd-cli not found. Set its path in the Tools section.")
    if not Path(profile["model"]).is_file():
        raise MethodError(f'Model file does not exist: {profile["model"]}')
    if not profile.get("mmproj"):
        raise MethodError(
            "This model has no mmproj file; llama-mtmd-cli requires one for audio. "
            "Please download or configure the corresponding mmproj GGUF file."
        )
    if not Path(profile["mmproj"]).is_file():
        raise MethodError(f'mmproj file does not exist: {profile["mmproj"]}')

    cmd = [
        cli,
        "-m", profile["model"],
        "--mmproj", profile["mmproj"],
        "--audio", audio,
        "-ngl", "999",
        "--ctx-size", "8192",
        "--threads", str(threads),
        "--temp", "0.1",
        "-p", prompt,
    ]
    cwd = str(Path(cli).parent) if Path(cli).exists() else None
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
    }
    if proc.returncode != 0:
        raise MethodError(f"llama-mtmd-cli exited with code {proc.returncode}", extra)
    if not raw or "Experimental CLI for multimodal" in raw:
        raise MethodError("llama-mtmd-cli printed no answer (only its usage text).", extra)
    return raw, extra
