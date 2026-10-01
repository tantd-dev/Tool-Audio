"""
Model discovery, magic header inspection, and projector matching.
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path
from typing import Callable

from config import MMPROJ_PREFERENCE, WHISPER_HINT


def model_magic(path: str | Path) -> str:
    """Determine file magic type: 'ggml' (whisper.cpp format), 'gguf' (real GGUF), 'unknown' or 'missing'."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return "missing"
    if head == b"GGUF":
        return "gguf"
    if head == b"lmgg":
        return "ggml"
    return "unknown"


def whisper_hint(model_path: str | Path) -> str:
    """Return tailored diagnostic guidance when loading a Whisper model."""
    magic = model_magic(model_path)
    if magic == "gguf":
        return (
            "This file is a real GGUF (magic 'GGUF'), but whisper-cli only loads whisper.cpp's own ggml "
            "format (the 'bad magic' error). Use 'Download Whisper model' to get a ggml-*.bin file "
            "(from huggingface.co/ggerganov/whisper.cpp), e.g. ggml-large-v3-turbo-q8_0.bin."
        )
    if magic == "missing":
        return "The model file could not be read."
    return WHISPER_HINT


def pick_mmproj(candidates: list[Path]) -> str:
    """Select the best matching mmproj file based on preferred precision (f16, bf16, f32)."""
    if not candidates:
        return ""
    for tag in MMPROJ_PREFERENCE:
        for c in candidates:
            if tag in c.name.lower():
                return str(c)
    return str(sorted(candidates)[0])


def is_whisper(profile: dict) -> bool:
    """Check if model profile represents a Whisper ASR model."""
    return "whisper" in (profile.get("name", "") + " " + profile.get("model", "")).lower()


def scan_lmstudio_models(root: str | Path) -> list[dict]:
    """Scan LM Studio folder for every GGUF model (and ggml-*.bin Whisper files) paired with mmproj."""
    root = Path(root)
    profiles = []
    if not root.is_dir():
        return profiles

    def is_model(p: Path) -> bool:
        n = p.name.lower()
        if "mmproj" in n:
            return False
        return n.endswith(".gguf") or (n.startswith("ggml-") and n.endswith(".bin"))

    all_files = [p for p in root.rglob("*") if p.is_file() and is_model(p)]
    mmproj_by_folder: dict[Path, list[Path]] = {}
    for p in root.rglob("*.gguf"):
        if "mmproj" in p.name.lower():
            mmproj_by_folder.setdefault(p.parent, []).append(p)

    for f in sorted(all_files):
        try:
            label = "/".join(f.parent.relative_to(root).parts[-2:])
        except ValueError:
            label = f.parent.name
        profiles.append({
            "name": f"{label} / {f.stem}",
            "model": str(f),
            "mmproj": pick_mmproj(mmproj_by_folder.get(f.parent, [])),
            "api_id": "",
            "api_manual": False,
        })
    return profiles


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_api_id(profile: dict, api_models: list[dict]) -> str:
    """Find the LM Studio API identifier corresponding to this model file."""
    hay = _norm(profile["model"])
    best = ""
    for m in api_models:
        n = _norm(m.get("id", ""))
        if n and n in hay and len(n) > len(_norm(best)):
            best = m["id"]
    return best


def download_file(url: str, dest: str | Path, progress_cb: Callable[[str], None] | None = None) -> str:
    """Download file with chunking and progress reporting."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tool-audio-benchmark"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress_cb and time.time() - last > 0.5:
                last = time.time()
                pct = f" ({done * 100 // total}%)" if total else ""
                progress_cb(f"Downloading {dest.name}: {done / 1048576:.0f} MB{pct}")
    part.replace(dest)
    return str(dest)
