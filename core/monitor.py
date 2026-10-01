"""
Resource monitor for tracking CPU, RAM, and GPU VRAM usage and peak values.
Works on Windows using standard library ctypes (zero dependencies required)
with transparent fallback/acceleration via psutil and nvml/nvidia-smi.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

# Try importing psutil if installed
try:
    import psutil
except ImportError:
    psutil = None

# Windows ctypes definitions for RAM and CPU tracking without extra dependencies
is_windows = os.name == "nt"
if is_windows:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.c_uint),
            ("dwHighDateTime", ctypes.c_uint),
        ]


class _WindowsCPUSampler:
    """Calculates system CPU usage percentage using kernel32.GetSystemTimes."""
    def __init__(self):
        self.prev_idle = 0
        self.prev_total = 0
        self.reset()

    @staticmethod
    def _filetime_to_int(ft: FILETIME) -> int:
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    def reset(self):
        if not is_windows:
            return
        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            self.prev_idle = self._filetime_to_int(idle)
            self.prev_total = self._filetime_to_int(kernel) + self._filetime_to_int(user)

    def sample(self) -> float:
        if not is_windows:
            return 0.0
        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return 0.0

        i = self._filetime_to_int(idle)
        t = self._filetime_to_int(kernel) + self._filetime_to_int(user)

        d_idle = i - self.prev_idle
        d_total = t - self.prev_total

        self.prev_idle = i
        self.prev_total = t

        if d_total <= 0:
            return 0.0
        pct = (1.0 - (d_idle / d_total)) * 100.0
        return max(0.0, min(100.0, round(pct, 2)))


class _NvmlHandler:
    """Direct NVML (NVIDIA Management Library) bindings via ctypes for instant, low-overhead VRAM query."""
    def __init__(self):
        self.initialized = False
        self.handle = None
        self.nvml = None
        if not is_windows:
            return
        try:
            self.nvml = ctypes.CDLL("nvml.dll")
            if self.nvml.nvmlInit_v2() == 0:
                dev = ctypes.c_void_p()
                if self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)) == 0:
                    self.handle = dev
                    self.initialized = True
        except Exception:
            self.initialized = False

    def get_memory(self) -> tuple[float | None, float | None]:
        """Return (used_mb, total_mb) or (None, None)."""
        if not self.initialized or not self.handle:
            return None, None
        try:
            class c_nvmlMemory_t(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]
            mem = c_nvmlMemory_t()
            if self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)) == 0:
                return round(mem.used / (1024 * 1024), 2), round(mem.total / (1024 * 1024), 2)
        except Exception:
            pass
        return None, None


_windows_cpu = _WindowsCPUSampler() if is_windows else None
_nvml = _NvmlHandler() if is_windows else None


def get_ram_info() -> tuple[float | None, float | None]:
    """Get system RAM as (used_mb, total_mb)."""
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return round(vm.used / (1024 * 1024), 2), round(vm.total / (1024 * 1024), 2)
        except Exception:
            pass

    if is_windows:
        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                used = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 * 1024)
                total = stat.ullTotalPhys / (1024 * 1024)
                return round(used, 2), round(total, 2)
        except Exception:
            pass

    return None, None


def get_ram_mb() -> float | None:
    """Get currently used RAM in MB."""
    used, _ = get_ram_info()
    return used


def get_cpu_percent() -> float:
    """Get current instantaneous CPU percent."""
    if psutil is not None:
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            pass

    if _windows_cpu is not None:
        return _windows_cpu.sample()

    return 0.0


def get_gpu_vram_info() -> tuple[float | None, float | None]:
    """Get GPU VRAM info as (used_mb, total_mb)."""
    if _nvml is not None and _nvml.initialized:
        used, total = _nvml.get_memory()
        if used is not None:
            return used, total

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                line = proc.stdout.splitlines()[0]
                parts = [float(x.strip()) for x in line.split(",") if x.strip()]
                if len(parts) >= 2:
                    return parts[0], parts[1]
                elif len(parts) == 1:
                    return parts[0], None
        except Exception:
            pass
    return None, None


def get_gpu_vram_mb() -> float | None:
    """Get current GPU VRAM used in MB."""
    used, _ = get_gpu_vram_info()
    return used


class SystemMonitor:
    """
    Background sampler that monitors CPU, RAM, and GPU VRAM at high frequency (100ms)
    during benchmark execution and records peak (maximum) usage values.
    """
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Snapshot before
        self.cpu_before: float | None = None
        self.ram_before: float | None = None
        self.ram_total: float | None = None
        self.vram_before: float | None = None
        self.vram_total: float | None = None

        # Peak values
        self.cpu_peak: float = 0.0
        self.ram_peak: float = 0.0
        self.vram_peak: float | None = None

        # Cumulative stats
        self.cpu_samples: list[float] = []
        self.ram_samples: list[float] = []
        self.vram_samples: list[float] = []

        # Snapshot after
        self.cpu_after: float | None = None
        self.ram_after: float | None = None
        self.vram_after: float | None = None

    def start(self):
        """Start monitoring and sampling."""
        # Warm up CPU tracker
        if _windows_cpu is not None:
            _windows_cpu.reset()
        time.sleep(0.02)

        self.cpu_before = get_cpu_percent()
        ram_u, ram_t = get_ram_info()
        self.ram_before = ram_u
        self.ram_total = ram_t
        vram_u, vram_t = get_gpu_vram_info()
        self.vram_before = vram_u
        self.vram_total = vram_t

        self.cpu_peak = self.cpu_before or 0.0
        self.ram_peak = self.ram_before or 0.0
        self.vram_peak = self.vram_before

        if self.cpu_before is not None:
            self.cpu_samples.append(self.cpu_before)
        if self.ram_before is not None:
            self.ram_samples.append(self.ram_before)
        if self.vram_before is not None:
            self.vram_samples.append(self.vram_before)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_sampling, daemon=True)
        self._thread.start()

    def _run_sampling(self):
        while not self._stop_event.is_set():
            time.sleep(self.sample_interval)
            if self._stop_event.is_set():
                break

            # CPU
            c = get_cpu_percent()
            self.cpu_samples.append(c)
            if c > self.cpu_peak:
                self.cpu_peak = c

            # RAM
            r = get_ram_mb()
            if r is not None:
                self.ram_samples.append(r)
                if r > self.ram_peak:
                    self.ram_peak = r

            # VRAM
            v = get_gpu_vram_mb()
            if v is not None:
                self.vram_samples.append(v)
                if self.vram_peak is None or v > self.vram_peak:
                    self.vram_peak = v

    def stop(self) -> dict[str, Any]:
        """Stop monitoring and return full resource metrics dict."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self.cpu_after = get_cpu_percent()
        self.ram_after = get_ram_mb()
        self.vram_after = get_gpu_vram_mb()

        if self.cpu_after is not None:
            self.cpu_samples.append(self.cpu_after)
            if self.cpu_after > self.cpu_peak:
                self.cpu_peak = self.cpu_after

        if self.ram_after is not None and self.ram_after > self.ram_peak:
            self.ram_peak = self.ram_after

        if self.vram_after is not None and (self.vram_peak is None or self.vram_after > self.vram_peak):
            self.vram_peak = self.vram_after

        avg_cpu = round(sum(self.cpu_samples) / len(self.cpu_samples), 2) if self.cpu_samples else None

        return {
            "cpu_percent_before": round(self.cpu_before, 2) if self.cpu_before is not None else None,
            "cpu_percent_peak": round(self.cpu_peak, 2),
            "cpu_percent_avg": avg_cpu,
            "cpu_percent_after": round(self.cpu_after, 2) if self.cpu_after is not None else None,
            "ram_mb_before": self.ram_before,
            "ram_mb_peak": round(self.ram_peak, 2) if self.ram_peak else None,
            "ram_mb_after": self.ram_after,
            "ram_total_mb": self.ram_total,
            "gpu_vram_mb_before": self.vram_before,
            "gpu_vram_mb_peak": self.vram_peak,
            "gpu_vram_mb_after": self.vram_after,
            "gpu_vram_total_mb": self.vram_total,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
