"""
Local Audio Model Benchmark GUI
- Windows-friendly Tkinter GUI (pure stdlib with optional psutil/nvml)
- Scans LM Studio models folder and pairs GGUF with mmproj
- Auto / API / CLI execution with automatic fallback
- High-frequency resource monitoring: Peak CPU %, Peak RAM MB, Peak GPU VRAM MB
- Exports structured benchmark JSON results

Run:
    py app.py
"""

from __future__ import annotations

from ui.app_window import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()