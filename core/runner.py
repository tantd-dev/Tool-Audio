"""
Benchmark orchestrator executing runs, monitoring resources, and exporting JSON results.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import MODE_API, MODE_CLI, utc_now
from core.audio import get_audio_duration
from core.exceptions import MethodError
from core.models import is_whisper
from core.monitor import SystemMonitor
from engines.api import run_api
from engines.cli import run_cli
from engines.whisper import run_whisper


def run_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Execute benchmark on the given audio and model profile with fallback and resource tracking.
    Tracks peak CPU, RAM, and GPU VRAM during execution.
    Exports result to a JSON file and returns the result dictionary.
    """
    profile = cfg["profile"]
    audio = cfg["audio"]
    threads = max(1, os.cpu_count() or 8)
    duration = get_audio_duration(audio)

    # Initialize and start high-frequency resource monitoring
    monitor = SystemMonitor(sample_interval=0.1)
    monitor.start()

    # Determine execution plan
    if is_whisper(profile):
        plan = ["whisper"]
    elif cfg["mode"] == MODE_API:
        plan = ["api"]
    elif cfg["mode"] == MODE_CLI:
        plan = ["cli"]
    else:
        plan = ["api", "cli"] if profile.get("api_id") else ["cli"]

    result: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "model": profile["name"],
        "model_file": profile["model"],
        "mmproj_file": profile.get("mmproj") or None,
        "audio_file": audio,
        "audio_duration_sec": round(duration, 4) if duration is not None else None,
        "method_plan": plan,
        "prompt": cfg["prompt"] if not is_whisper(profile) else None,
    }

    attempts = []
    t0 = time.perf_counter()
    raw = ""
    used = None

    for method in plan:
        m0 = time.perf_counter()
        try:
            if method == "whisper":
                raw, extra = run_whisper(cfg["whisper"], profile, audio, cfg["lang"], threads)
            elif method == "api":
                raw, extra = run_api(cfg["server"], profile.get("api_id", ""), audio,
                                     cfg["prompt"], cfg["max_tokens"])
            else:
                raw, extra = run_cli(cfg["cli"], profile, audio, cfg["prompt"], threads)

            used = method
            result.update(extra)
            attempts.append({
                "method": method,
                "status": "success",
                "time_sec": round(time.perf_counter() - m0, 4)
            })
            break
        except MethodError as exc:
            attempts.append({
                "method": method,
                "status": "error",
                "error": str(exc),
                "time_sec": round(time.perf_counter() - m0, 4)
            })
            result["error"] = str(exc)
            if exc.extra:
                result.update(exc.extra)
        except Exception as exc:
            attempts.append({
                "method": method,
                "status": "error",
                "error": repr(exc),
                "time_sec": round(time.perf_counter() - m0, 4)
            })
            result["error"] = repr(exc)

    elapsed = round(time.perf_counter() - t0, 4)
    res_stats = monitor.stop()

    ok = used is not None
    if ok:
        result.pop("error", None)

    result.update({
        "finished_utc": utc_now(),
        "status": "success" if ok else "error",
        "method_used": used,
        "fell_back": ok and len(attempts) > 1,
        "attempts": attempts,
        "processing_time_sec": elapsed,
        "rtf": round(elapsed / duration, 6) if duration else None,
        # CPU statistics
        "cpu_percent_before": res_stats["cpu_percent_before"],
        "cpu_percent_peak": res_stats["cpu_percent_peak"],
        "cpu_percent_avg": res_stats["cpu_percent_avg"],
        "cpu_percent_after": res_stats["cpu_percent_after"],
        # RAM statistics
        "ram_mb_before": res_stats["ram_mb_before"],
        "ram_mb_peak": res_stats["ram_mb_peak"],
        "ram_mb_after": res_stats["ram_mb_after"],
        # GPU VRAM statistics
        "gpu_vram_mb_before": res_stats["gpu_vram_mb_before"],
        "gpu_vram_mb_peak": res_stats["gpu_vram_mb_peak"],
        "gpu_vram_mb_after": res_stats["gpu_vram_mb_after"],
        "raw_response": raw if ok else "",
    })

    # Save JSON result to disk beside audio file
    clean_model_name = re.sub(r'[^A-Za-z0-9_-]+', '_', profile['name'])
    out_path = Path(audio).with_name(
        f"{Path(audio).stem}_{clean_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    try:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["result_json_path"] = str(out_path)
    except Exception as exc:
        result["export_error"] = repr(exc)

    return result
