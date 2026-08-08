"""Work out what this machine can run, and pick a vision engine to match.

The hybrid pipeline has two halves with very different appetites. OCR and the
OpenCV work are CPU jobs that run anywhere. The vision-language model that
assigns a role to each piece of text is the part that wants a GPU — so the
engine profile is really a question about VRAM.

Profiles degrade honestly rather than pretending: on a machine with no GPU the
answer is "OCR and CV only, roles assigned by rule", which is a genuinely
weaker reading of a label, and the profile says so instead of quietly producing
worse records.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict, field

# VRAM thresholds in GB for the 7-8B class of vision-language model.
_VRAM_FULL_PRECISION = 15.0
_VRAM_QUANTISED = 7.0

PROFILE_GPU_FULL = "gpu-full"
PROFILE_GPU_QUANTISED = "gpu-quantised"
PROFILE_CPU = "cpu-ocr-only"

_PROFILE_NOTES = {
    PROFILE_GPU_FULL: (
        "Comfortable for a 7-8B vision-language model at full precision, batched. "
        "Run the whole corpus locally; there is headroom for a LoRA later."
    ),
    PROFILE_GPU_QUANTISED: (
        "Enough for a 4-bit quantised 7-8B vision-language model. Slower per label, "
        "but the corpus is only read once, so this is a throughput question, not a "
        "quality one."
    ),
    PROFILE_CPU: (
        "No usable GPU found. OCR and the OpenCV measurements still work, and still "
        "give real millimetre type sizes -- but nothing assigns a role to each text "
        "block, so element classification falls back to rules and will be weaker. "
        "Options: add a GPU, or route only the role-assignment step to a hosted "
        "model (artwork would then leave this machine)."
    ),
}


@dataclass
class GPU:
    name: str
    vram_gb: float


@dataclass
class Probe:
    platform: str
    python: str
    cpu_count: int
    ram_gb: float | None
    gpus: list[GPU] = field(default_factory=list)
    profile: str = PROFILE_CPU
    notes: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def max_vram_gb(self) -> float:
        return max((g.vram_gb for g in self.gpus), default=0.0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["max_vram_gb"] = self.max_vram_gb
        return data

    def summary(self) -> str:
        lines = [
            f"platform    {self.platform}",
            f"python      {self.python}",
            f"cpu cores   {self.cpu_count}",
            f"ram         {f'{self.ram_gb:.1f} GB' if self.ram_gb else 'unknown'}",
        ]
        if self.gpus:
            for gpu in self.gpus:
                lines.append(f"gpu         {gpu.name} ({gpu.vram_gb:.1f} GB)")
        else:
            lines.append("gpu         none detected")
        lines += ["", f"profile     {self.profile}", "", self.notes]
        for warning in self.warnings:
            lines.append(f"\n!  {warning}")
        return "\n".join(lines)


def _nvidia_gpus() -> tuple[list[GPU], list[str]]:
    """Query nvidia-smi. Absent driver simply means no NVIDIA GPU here."""
    warnings: list[str] = []
    if not shutil.which("nvidia-smi"):
        return [], warnings
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], [f"nvidia-smi present but failed: {exc}"]

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        return [], [f"nvidia-smi failed: {detail[0] if detail else result.returncode}"]

    gpus: list[GPU] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        name, _, memory = line.partition(",")
        try:
            gpus.append(GPU(name=name.strip(), vram_gb=float(memory.strip()) / 1024))
        except ValueError:
            warnings.append(f"could not read GPU memory from {line!r}")
    return gpus, warnings


def _total_ram_gb() -> float | None:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except (OSError, ValueError):
            pass
    if platform.system() == "Windows":  # pragma: no cover - platform specific
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1024**3
        except (AttributeError, OSError):
            pass
    return None


def choose_profile(max_vram_gb: float) -> str:
    if max_vram_gb >= _VRAM_FULL_PRECISION:
        return PROFILE_GPU_FULL
    if max_vram_gb >= _VRAM_QUANTISED:
        return PROFILE_GPU_QUANTISED
    return PROFILE_CPU


def probe() -> Probe:
    gpus, warnings = _nvidia_gpus()
    result = Probe(
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=_total_ram_gb(),
        gpus=gpus,
        warnings=warnings,
    )
    result.profile = choose_profile(result.max_vram_gb)
    result.notes = _PROFILE_NOTES[result.profile]

    if result.gpus and result.profile == PROFILE_CPU:
        result.warnings.append(
            f"A GPU was found ({result.max_vram_gb:.1f} GB) but it is below the "
            f"{_VRAM_QUANTISED:.0f} GB needed for a quantised 7-8B model."
        )
    return result


def probe_json() -> str:
    return json.dumps(probe().to_dict(), indent=2)
