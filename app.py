#!/usr/bin/env python3
"""
Benchmark script for LM Studio audio analysis.
- Lists available models and lets user select with ↑/↓ arrow keys
- Asks for audio file path
- Auto-detects audio duration with ffprobe
- Saves detailed processing info to JSON
"""

import time
import json
import sys
import tty
import termios
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests
import subprocess

LM_STUDIO_BASE = "http://localhost:1234"
CHAT_URL = f"{LM_STUDIO_BASE}/v1/chat/completions"
MODELS_URL = f"{LM_STUDIO_BASE}/v1/models"


def get_gpu_vram():
    """Return used GPU VRAM in MB. Returns None if nvidia-smi is unavailable."""
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(result.strip().split("\n")[0])
    except Exception:
        return None


def get_audio_duration(file_path: str) -> float | None:
    """Get duration of audio/video file in seconds using ffprobe. Returns None on failure."""
    try:
        result = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(result.strip())
    except Exception:
        return None


def get_available_models():
    """Fetch list of model IDs from LM Studio OpenAI-compatible API."""
    try:
        resp = requests.get(MODELS_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m["id"] for m in data.get("data", []) if "id" in m]
        return sorted(models)
    except Exception as e:
        print(f"[Lỗi] Không thể lấy danh sách model từ LM Studio: {e}")
        print("Hãy chắc chắn LM Studio đang chạy và Local Server đã bật.")
        sys.exit(1)


def _get_key():
    """Read a single key / escape sequence from stdin (Unix)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # ESC sequence
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return ch + ch2 + ch3
            return ch + ch2
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_model(models: list[str]) -> str:
    """
    Interactive menu: use ↑/↓ (or k/j) to move, Enter to select, q/Esc to quit.
    Returns the selected model id.
    """
    if not models:
        print("Không có model nào. Hãy load model trong LM Studio trước.")
        sys.exit(1)

    idx = 0
    n = len(models)

    def render():
        # Move cursor up to redraw
        sys.stdout.write(f"\033[{n + 3}A")
        sys.stdout.write("\033[J")  # clear from cursor down
        print("╔══════════════════════════════════════════════════════╗")
        print("║  Chọn model (↑/↓ hoặc k/j, Enter để chọn, q thoát)  ║")
        print("╚══════════════════════════════════════════════════════╝")
        for i, m in enumerate(models):
            if i == idx:
                print(f"  \033[1;36m❯ {m}\033[0m")
            else:
                print(f"    {m}")
        sys.stdout.flush()

    # Initial draw (reserve space)
    print("\n" * (n + 3), end="")
    render()

    while True:
        key = _get_key()

        if key in ("\x1b[A", "k", "K"):  # Up
            idx = (idx - 1) % n
            render()
        elif key in ("\x1b[B", "j", "J"):  # Down
            idx = (idx + 1) % n
            render()
        elif key in ("\r", "\n"):  # Enter
            # Clear menu area
            sys.stdout.write(f"\033[{n + 3}A\033[J")
            sys.stdout.flush()
            return models[idx]
        elif key in ("q", "Q", "\x1b", "\x03"):  # q / Esc / Ctrl-C
            sys.stdout.write(f"\033[{n + 3}A\033[J")
            sys.stdout.flush()
            print("Đã hủy chọn model.")
            sys.exit(0)


def ask_audio_path() -> str:
    """Ask user for audio file path and validate it exists."""
    print("Nhập đường dẫn file audio (kéo thả file vào terminal cũng được):")
    while True:
        path = input("> ").strip().strip("'\"")  # remove quotes if drag-drop
        if not path:
            print("Đường dẫn không được để trống. Thử lại:")
            continue
        p = Path(path).expanduser().resolve()
        if not p.exists():
            print(f"[Lỗi] File không tồn tại: {p}")
            print("Thử lại:")
            continue
        if not p.is_file():
            print(f"[Lỗi] Không phải là file: {p}")
            print("Thử lại:")
            continue
        return str(p)


def main():
    print("=" * 60)
    print("  LM Studio Audio Benchmark")
    print("=" * 60)

    # 1. Lấy đường dẫn file audio
    print()
    audio_path = ask_audio_path()
    print(f"✓ File audio: \033[1;32m{audio_path}\033[0m")

    # Lấy thời lượng thật bằng ffprobe
    duration = get_audio_duration(audio_path)
    if duration is not None and duration > 0:
        audio_duration = duration
        print(f"✓ Thời lượng audio: \033[1;32m{audio_duration:.2f} giây\033[0m\n")
    else:
        print("[Cảnh báo] Không đọc được thời lượng bằng ffprobe.")
        try:
            audio_duration = float(input("Nhập thời lượng audio (giây): ").strip())
        except ValueError:
            audio_duration = 600.0
            print(f"Dùng giá trị mặc định: {audio_duration} giây\n")
        else:
            print()

    # 2. Lấy danh sách model và cho người dùng chọn
    print("Đang lấy danh sách model từ LM Studio...")
    models = get_available_models()
    print(f"Tìm thấy {len(models)} model.\n")

    selected_model = select_model(models)
    print(f"✓ Đã chọn model: \033[1;32m{selected_model}\033[0m\n")

    # 3. Chuẩn bị đo đạc
    process = psutil.Process()
    ram_before = process.memory_info().rss / 1024 / 1024
    vram_before = get_gpu_vram()

    start_wall = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()

    # 4. Gọi API
    # Lưu ý: Hiện tại gửi prompt text. Nếu model hỗ trợ multimodal audio,
    # bạn có thể mở rộng phần content để gửi base64 audio sau.
    user_prompt = f"Analyze this audio file: {audio_path}"

    print("Đang gửi request tới LM Studio (có thể mất thời gian)...")
    try:
        response = requests.post(
            CHAT_URL,
            json={
                "model": selected_model,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                "temperature": 0,
            },
            timeout=1800,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[Lỗi] Request thất bại: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - start
    end_wall = datetime.now(timezone.utc).isoformat()

    data = response.json()
    usage = data.get("usage", {})

    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    ram_after = process.memory_info().rss / 1024 / 1024
    vram_after = get_gpu_vram()

    # 5. Tạo kết quả chi tiết
    result = {
        "timestamp_utc": start_wall,
        "finished_utc": end_wall,
        "model": selected_model,
        "audio_file": audio_path,
        "status": "success",
        "http_status": response.status_code,
        # Thời gian
        "processing_time_sec": round(elapsed, 3),
        "audio_duration_sec": round(audio_duration, 3),
        "rtf": round(rtf, 4),  # Real-Time Factor
        # Token usage
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        # Tài nguyên
        "ram_mb_before": round(ram_before, 2),
        "ram_mb_after": round(ram_after, 2),
        "ram_mb_delta": round(ram_after - ram_before, 2),
        "gpu_vram_mb_before": vram_before,
        "gpu_vram_mb_after": vram_after,
        "gpu_vram_mb_delta": (
            round(vram_after - vram_before, 2)
            if vram_before is not None and vram_after is not None
            else None
        ),
        # Prompt & Output
        "prompt": user_prompt,
        "output": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
        # Raw usage
        "raw_usage": usage,
    }

    # 6. In ra console
    print("\n" + "=" * 60)
    print("KẾT QUẢ BENCHMARK")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)

    # 7. Lưu file JSON chi tiết
    out_file = "benchmark_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Đã lưu kết quả chi tiết vào: \033[1;33m{os.path.abspath(out_file)}\033[0m")


if __name__ == "__main__":
    main()