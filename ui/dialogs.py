"""
Tkinter dialog windows for adding/editing model profiles and downloading Whisper models.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from config import (
    WHISPER_DOWNLOAD_DIR,
    WHISPER_DOWNLOAD_URL,
    WHISPER_DOWNLOADS,
)
from core.models import download_file


class ProfileDialog(tk.Toplevel):
    """Dialog to add or edit a model profile."""
    def __init__(self, parent: tk.Widget, title: str = "Add model", profile: dict | None = None):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, False)
        self.result: dict | None = None
        profile = profile or {"name": "", "model": "", "mmproj": "", "api_id": ""}
        self.columnconfigure(1, weight=1)

        self.name_var = tk.StringVar(value=profile.get("name", ""))
        self.model_var = tk.StringVar(value=profile.get("model", ""))
        self.mmproj_var = tk.StringVar(value=profile.get("mmproj", ""))
        self.api_var = tk.StringVar(value=profile.get("api_id", ""))

        rows = [
            ("Display name", self.name_var, None),
            ("Model file (.gguf / ggml .bin)", self.model_var, self.pick_model),
            ("mmproj GGUF (audio LLMs)", self.mmproj_var, self.pick_mmproj),
            ("LM Studio API id (optional)", self.api_var, None),
        ]
        for r, (label, var, cmd) in enumerate(rows):
            ttk.Label(self, text=label).grid(row=r, column=0, padx=10, pady=7, sticky="w")
            ttk.Entry(self, textvariable=var, width=65).grid(row=r, column=1, padx=10, pady=7, sticky="ew")
            if cmd:
                ttk.Button(self, text="Browse…", command=cmd).grid(row=r, column=2, padx=10, pady=7)

        buttons = ttk.Frame(self)
        buttons.grid(row=len(rows), column=0, columnspan=3, pady=12)
        ttk.Button(buttons, text="Save", command=self.save).pack(side="left", padx=5)
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=5)
        self.transient(parent)
        self.grab_set()

    def pick_model(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.gguf *.bin"), ("All files", "*.*")])
        if p:
            self.model_var.set(p)
            if not self.name_var.get().strip():
                self.name_var.set(Path(p).stem)

    def pick_mmproj(self):
        p = filedialog.askopenfilename(filetypes=[("GGUF projector", "*.gguf"), ("All files", "*.*")])
        if p:
            self.mmproj_var.set(p)

    def save(self):
        name = self.name_var.get().strip()
        model = self.model_var.get().strip()
        if not name or not model:
            messagebox.showerror("Missing information", "Please enter a model name and file path.", parent=self)
            return
        api_id = self.api_var.get().strip()
        self.result = {
            "name": name,
            "model": model,
            "mmproj": self.mmproj_var.get().strip(),
            "api_id": api_id,
            "api_manual": bool(api_id),
        }
        self.destroy()


class WhisperDownloadDialog(tk.Toplevel):
    """Dialog to download ggml Whisper models from HuggingFace."""
    def __init__(self, parent: tk.Widget, events: queue.Queue):
        super().__init__(parent)
        self.events = events
        self.title("Download Whisper model (ggml, for whisper-cli)")
        self.transient(parent)
        self.grab_set()

        ttk.Label(self, text=f"Saved to: {WHISPER_DOWNLOAD_DIR}").grid(
            row=0, column=0, columnspan=2, padx=12, pady=(12, 4), sticky="w"
        )
        self.choice = tk.StringVar(value=list(WHISPER_DOWNLOADS)[0])
        ttk.Combobox(
            self,
            textvariable=self.choice,
            values=list(WHISPER_DOWNLOADS),
            state="readonly",
            width=60
        ).grid(row=1, column=0, columnspan=2, padx=12, pady=6)

        ttk.Button(self, text="Download", command=self.go).grid(row=2, column=0, padx=12, pady=12, sticky="e")
        ttk.Button(self, text="Cancel", command=self.destroy).grid(row=2, column=1, padx=12, pady=12, sticky="w")

    def go(self):
        fname = WHISPER_DOWNLOADS[self.choice.get()]
        self.destroy()
        dest = WHISPER_DOWNLOAD_DIR / fname
        if dest.is_file():
            self.events.put(("dl_done", str(dest)))
            return
        self.events.put(("status", f"Downloading {fname}…"))

        def work():
            try:
                path = download_file(
                    WHISPER_DOWNLOAD_URL.format(fname),
                    dest,
                    lambda t: self.events.put(("status", t))
                )
                self.events.put(("dl_done", path))
            except Exception as exc:
                self.events.put(("dl_error", f"Download failed: {exc!r}"))

        threading.Thread(target=work, daemon=True).start()
