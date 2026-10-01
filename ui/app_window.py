"""
Main application window and event loop for Local Audio Model Benchmark GUI.
"""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import (
    APP_TITLE,
    DEFAULT_MODEL_DIR,
    DEFAULT_SERVER,
    MODE_API,
    MODE_AUTO,
    MODE_CLI,
    find_exe,
)
from core.models import (
    is_whisper,
    match_api_id,
    model_magic,
    scan_lmstudio_models,
)
from core.runner import run_benchmark
from engines.api import fetch_api_models
from ui.dialogs import ProfileDialog, WhisperDownloadDialog


class App(tk.Tk):
    """Main benchmark application GUI."""
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1040x860")
        self.minsize(860, 700)
        self.events: queue.Queue = queue.Queue()
        self.profiles: list[dict] = scan_lmstudio_models(DEFAULT_MODEL_DIR)
        self.api_models: list[dict] = []

        self._build()
        self._refresh_profiles()
        self.after(100, self._poll_events)
        self.after(400, lambda: self.check_api(silent=True))

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        # --- tools & server -------------------------------------------------
        tools = ttk.LabelFrame(self, text="Tools, LM Studio folder and server")
        tools.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        tools.columnconfigure(1, weight=1)

        self.cli_var = tk.StringVar(value=find_exe(["llama-mtmd-cli", "llama-mtmd-cli.exe"]))
        self.whisper_var = tk.StringVar(value=find_exe(["whisper-cli", "whisper-cli.exe"]))
        self.models_dir_var = tk.StringVar(value=str(DEFAULT_MODEL_DIR))
        self.server_var = tk.StringVar(value=DEFAULT_SERVER)

        def row(r, label, var, browse=None, extra_btn=None):
            ttk.Label(tools, text=label).grid(row=r, column=0, padx=(8, 6), pady=5, sticky="w")
            ttk.Entry(tools, textvariable=var).grid(row=r, column=1, padx=4, pady=5, sticky="ew")
            if browse:
                ttk.Button(tools, text="Browse…", command=browse).grid(row=r, column=2, padx=4, pady=5)
            if extra_btn:
                ttk.Button(tools, text=extra_btn[0], command=extra_btn[1]).grid(row=r, column=3, padx=(4, 8), pady=5)

        row(0, "llama-mtmd-cli", self.cli_var, lambda: self._pick_file(self.cli_var))
        row(1, "whisper-cli", self.whisper_var, lambda: self._pick_file(self.whisper_var))
        row(2, "LM Studio models folder", self.models_dir_var, self.pick_models_dir,
            ("⟳ Scan models", self.scan_models))
        row(3, "LM Studio server URL", self.server_var, None, ("Check API", self.check_api))

        # --- model list -----------------------------------------------------
        model_frame = ttk.LabelFrame(self, text="Models — select one; use ↑ / ↓ to reorder")
        model_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        model_frame.columnconfigure(0, weight=1)
        self.model_list = tk.Listbox(model_frame, height=6, exportselection=False)
        self.model_list.grid(row=0, column=0, rowspan=3, padx=8, pady=8, sticky="ew")
        self.model_list.bind("<<ListboxSelect>>", self.on_model_select)

        controls = ttk.Frame(model_frame)
        controls.grid(row=0, column=1, padx=8, pady=8, sticky="ns")
        ttk.Button(controls, text="↑ Move up", command=lambda: self.move_profile(-1)).pack(fill="x", pady=3)
        ttk.Button(controls, text="↓ Move down", command=lambda: self.move_profile(1)).pack(fill="x", pady=3)
        ttk.Button(controls, text="Add", command=self.add_profile).pack(fill="x", pady=3)
        ttk.Button(controls, text="Edit", command=self.edit_profile).pack(fill="x", pady=3)
        ttk.Button(controls, text="Remove", command=self.remove_profile).pack(fill="x", pady=3)
        ttk.Button(controls, text="Download Whisper model", command=self.download_whisper).pack(fill="x", pady=3)

        self.selected_info = tk.StringVar(value="Select a model.")
        ttk.Label(model_frame, textvariable=self.selected_info, wraplength=820, justify="left").grid(
            row=3, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="w"
        )

        # --- audio ----------------------------------------------------------
        audio_frame = ttk.LabelFrame(self, text="Audio input")
        audio_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        audio_frame.columnconfigure(0, weight=1)
        self.audio_var = tk.StringVar()
        ttk.Entry(audio_frame, textvariable=self.audio_var).grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        ttk.Button(audio_frame, text="Choose audio…", command=self.pick_audio).grid(row=0, column=1, padx=8, pady=8)

        # --- prompt & options -----------------------------------------------
        prompt_frame = ttk.LabelFrame(self, text="Prompt (ignored for Whisper) and options")
        prompt_frame.grid(row=3, column=0, sticky="ew", padx=12, pady=6)
        prompt_frame.columnconfigure(0, weight=1)
        self.prompt_text = tk.Text(prompt_frame, height=6, wrap="word")
        self.prompt_text.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        self.prompt_text.insert("1.0", (
            "Analyze the provided audio directly. Return ONLY valid JSON with keys: "
            "language, audio_type, summary, speakers, main_topics, key_information, "
            "background_sounds, important_events, uncertainties. "
            "Do not invent information. If uncertain, say so."
        ))

        opts = ttk.Frame(prompt_frame)
        opts.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="w")
        ttk.Label(opts, text="Method").pack(side="left")
        self.mode_var = tk.StringVar(value=MODE_AUTO)
        ttk.Combobox(opts, textvariable=self.mode_var, state="readonly", width=22,
                     values=[MODE_AUTO, MODE_API, MODE_CLI]).pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="Max tokens").pack(side="left")
        self.max_tokens_var = tk.StringVar(value="2048")
        ttk.Entry(opts, textvariable=self.max_tokens_var, width=7).pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="Whisper language").pack(side="left")
        self.lang_var = tk.StringVar(value="auto")
        ttk.Entry(opts, textvariable=self.lang_var, width=7).pack(side="left", padx=4)

        # --- run ------------------------------------------------------------
        run_frame = ttk.Frame(self)
        run_frame.grid(row=4, column=0, sticky="nsew", padx=12, pady=6)
        run_frame.columnconfigure(0, weight=1)
        run_frame.rowconfigure(1, weight=1)
        self.run_button = ttk.Button(run_frame, text="Run test and export JSON", command=self.start_run)
        self.run_button.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="e", pady=(0, 6))
        self.output = tk.Text(run_frame, wrap="word", state="disabled")
        self.output.grid(row=1, column=0, sticky="nsew")

    # ------------------------------------------------------------ helpers
    def log(self, text: str):
        self.output.configure(state="normal")
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)
        self.output.configure(state="disabled")

    def _pick_file(self, var: tk.StringVar):
        p = filedialog.askopenfilename(filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if p:
            var.set(p)

    def pick_audio(self):
        p = filedialog.askopenfilename(filetypes=[
            ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"), ("All files", "*.*")
        ])
        if p:
            self.audio_var.set(p)

    def pick_models_dir(self):
        d = filedialog.askdirectory(initialdir=self.models_dir_var.get() or str(Path.home()))
        if d:
            self.models_dir_var.set(d)
            self.scan_models()

    # ----------------------------------------------------------- profiles
    def _refresh_profiles(self, selected: int = 0):
        self.model_list.delete(0, tk.END)
        for p in self.profiles:
            if is_whisper(p):
                tag = "ASR ⚠ GGUF-not-ggml" if model_magic(p["model"]) == "gguf" else "ASR"
            else:
                tag = "API" if p.get("api_id") else "CLI"
            self.model_list.insert(tk.END, f'[{tag}] {p["name"]}')
        if self.profiles:
            selected = max(0, min(selected, len(self.profiles) - 1))
            self.model_list.selection_set(selected)
            self.model_list.activate(selected)
            self.on_model_select()
        else:
            self.selected_info.set("No models found. Use Scan models or Add.")

    def selected_index(self) -> int | None:
        sel = self.model_list.curselection()
        return sel[0] if sel else (0 if self.profiles else None)

    def on_model_select(self, _event=None):
        i = self.selected_index()
        if i is None:
            return
        p = self.profiles[i]
        if is_whisper(p):
            method = "Whisper detected → ASR via whisper-cli"
            if model_magic(p["model"]) == "gguf":
                method += ("\n⚠ This is a real GGUF; whisper-cli needs a ggml .bin (bad magic). "
                           "Click 'Download Whisper model'.")
        elif p.get("api_id"):
            method = f'LM Studio API id "{p["api_id"]}" (falls back to llama-mtmd-cli if audio is rejected)'
        else:
            method = "No LM Studio API id matched → llama-mtmd-cli"
        self.selected_info.set(
            f'Model: {p["model"]}\nmmproj: {p["mmproj"] or "(none)"}\nMethod: {method}'
        )

    def move_profile(self, delta: int):
        i = self.selected_index()
        if i is None:
            return
        j = i + delta
        if 0 <= j < len(self.profiles):
            self.profiles[i], self.profiles[j] = self.profiles[j], self.profiles[i]
            self._refresh_profiles(j)

    def add_profile(self):
        dlg = ProfileDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            self.profiles.append(dlg.result)
            self._apply_api_matches()
            self._refresh_profiles(len(self.profiles) - 1)

    def edit_profile(self):
        i = self.selected_index()
        if i is None:
            return
        dlg = ProfileDialog(self, "Edit model", self.profiles[i])
        self.wait_window(dlg)
        if dlg.result:
            self.profiles[i] = dlg.result
            self._apply_api_matches()
            self._refresh_profiles(i)

    def remove_profile(self):
        i = self.selected_index()
        if i is not None and messagebox.askyesno("Remove model", f'Remove "{self.profiles[i]["name"]}" from this list?'):
            self.profiles.pop(i)
            self._refresh_profiles(max(0, i - 1))

    def scan_models(self):
        found = scan_lmstudio_models(self.models_dir_var.get().strip())
        if not found:
            messagebox.showwarning("No models", "No model files found in that folder.")
            return
        self.profiles = found
        self._apply_api_matches()
        self._refresh_profiles(0)
        self.status_var.set(f"Found {len(found)} model(s).")

    # ---------------------------------------------------- whisper download
    def download_whisper(self):
        WhisperDownloadDialog(self, self.events)

    # ---------------------------------------------------------------- API
    def check_api(self, silent: bool = False):
        base = self.server_var.get().strip()
        self.status_var.set("Checking LM Studio server…")

        def work():
            try:
                self.events.put(("api_models", fetch_api_models(base)))
            except Exception as exc:
                self.events.put(("api_error", (str(exc), silent)))

        threading.Thread(target=work, daemon=True).start()

    def _apply_api_matches(self):
        for p in self.profiles:
            if not p.get("api_manual") and not is_whisper(p):
                p["api_id"] = match_api_id(p, self.api_models) if self.api_models else p.get("api_id", "")

    # ---------------------------------------------------------------- run
    def start_run(self):
        i = self.selected_index()
        audio = self.audio_var.get().strip()
        prompt = self.prompt_text.get("1.0", "end").strip()
        if i is None:
            messagebox.showerror("No model", "Scan or add a model first.")
            return
        if not Path(audio).is_file():
            messagebox.showerror("Audio not found", "Choose an existing audio file.")
            return
        profile = self.profiles[i].copy()
        if not is_whisper(profile) and not prompt:
            messagebox.showerror("Empty prompt", "Enter a prompt.")
            return
        try:
            max_tokens = int(self.max_tokens_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid value", "Max tokens must be an integer.")
            return

        cfg = {
            "profile": profile,
            "audio": audio,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "cli": self.cli_var.get().strip(),
            "whisper": self.whisper_var.get().strip(),
            "server": self.server_var.get().strip(),
            "mode": self.mode_var.get(),
            "lang": self.lang_var.get().strip() or "auto",
        }
        self.run_button.configure(state="disabled")
        self.status_var.set("Running benchmark & monitoring resources…")
        self.log(f"Starting: {profile['name']}")

        def worker():
            res = run_benchmark(cfg)
            self.events.put(("done", res))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- events
    def _poll_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "api_models":
                    self.api_models = data
                    self._apply_api_matches()
                    sel = self.selected_index()
                    self._refresh_profiles(sel or 0)
                    self.status_var.set(f"LM Studio API: {len(data)} model(s) available.")
                elif kind == "status":
                    self.status_var.set(data)
                elif kind == "dl_done":
                    self.models_dir_var.set(str(DEFAULT_MODEL_DIR))
                    keep = [p for p in self.profiles if Path(p["model"]) != Path(data)]
                    self.profiles = keep + [{
                        "name": "ggerganov/whisper.cpp / " + Path(data).stem,
                        "model": data,
                        "mmproj": "",
                        "api_id": "",
                        "api_manual": False
                    }]
                    self._refresh_profiles(len(self.profiles) - 1)
                    self.status_var.set(f"Ready: {Path(data).name}")
                    self.log(f"Whisper model ready: {data}")
                elif kind == "dl_error":
                    self.status_var.set("Download failed.")
                    messagebox.showerror("Download", data)
                elif kind == "api_error":
                    msg, silent = data
                    self.status_var.set("LM Studio API not reachable.")
                    self.log(msg)
                    if not silent:
                        messagebox.showwarning("LM Studio API", msg)
                elif kind == "done":
                    self.run_button.configure(state="normal")
                    used_str = data.get("method_used") or "no method succeeded"
                    self.status_var.set(f"Finished: {data.get('status')} ({used_str})")

                    # Highlight Peak metrics in logs
                    cpu_peak = data.get("cpu_percent_peak")
                    ram_peak = data.get("ram_mb_peak")
                    vram_peak = data.get("gpu_vram_mb_peak")
                    peak_summary = []
                    if cpu_peak is not None:
                        peak_summary.append(f"Peak CPU: {cpu_peak}%")
                    if ram_peak is not None:
                        peak_summary.append(f"Peak RAM: {ram_peak} MB")
                    if vram_peak is not None:
                        peak_summary.append(f"Peak VRAM: {vram_peak} MB")
                    if peak_summary:
                        self.log(" | ".join(peak_summary))

                    self.log(json.dumps(data, ensure_ascii=False, indent=2))
                    if data.get("result_json_path"):
                        peak_msg = ("\n" + " | ".join(peak_summary)) if peak_summary else ""
                        messagebox.showinfo(
                            "Test finished",
                            f"Status: {data['status']}\nMethod: {data.get('method_used')}"
                            f"{peak_msg}\n\nJSON saved:\n{data['result_json_path']}"
                        )
                    else:
                        messagebox.showerror("Export issue", data.get("export_error", "Could not save JSON."))
        except queue.Empty:
            pass
        self.after(150, self._poll_events)
