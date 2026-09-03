"""Work out what this machine is, so ``--hardware auto`` means something.

Two paths, because two kinds of box matter: a rented Linux instance with NVIDIA GPUs
(NVML for the name, the memory and the device count) and an Apple Silicon Mac (sysctl
for the chip and its unified memory). Anything else raises, and the user passes a named
profile instead -- guessing a bandwidth for an unknown accelerator would quietly corrupt
every roofline the diagnosis rules compute.

Peak numbers come from lookup tables rather than from the driver because no driver
reports them: NVML gives a name and a capacity, not HBM bandwidth or dense tensor-core
throughput. The tables are vendor specs; the Apple entries are approximations and are
marked as such.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
from typing import Any, Literal, NamedTuple

from infervolt.core.types import HardwareProfile


class _Peak(NamedTuple):
    slug: str
    hbm_bw_gbs: float
    peak_tflops: float
    compute_capability: float


NVIDIA_TABLE: tuple[tuple[tuple[str, ...], _Peak], ...] = (
    # Every substring must appear in "<NVML name> <rounded GiB>GB", so the two A100
    # capacities are told apart whether or not the card's own name mentions it.
    (("A100", "80GB"), _Peak("a100-80", 2039, 312, 8.0)),
    (("A100", "40GB"), _Peak("a100-40", 1555, 312, 8.0)),
    (("H200",), _Peak("h200", 4800, 989, 9.0)),
    (("H100",), _Peak("h100", 3350, 989, 9.0)),
    # L40 is checked before L4: "L40" contains "L4".
    (("L40",), _Peak("l40", 864, 181, 8.9)),
    (("L4",), _Peak("l4", 300, 121, 8.9)),
    (("A6000",), _Peak("a6000", 768, 155, 8.6)),
    (("5090",), _Peak("rtx5090", 1792, 210, 12.0)),
    (("4090",), _Peak("rtx4090", 1008, 165, 8.9)),
    # A10 is checked after A100, which contains it.
    (("A10",), _Peak("a10", 600, 125, 8.6)),
)
"""Dense BF16 TFLOPs and HBM bandwidth per NVIDIA device, by name substrings."""

APPLE_TABLE: tuple[tuple[str, _Peak], ...] = (
    # Longest variants first, so "M3 MAX" is not matched as "M3".
    ("M1 ULTRA", _Peak("m1-ultra", 800, 21.0, 0.0)),
    ("M1 MAX", _Peak("m1-max", 400, 10.4, 0.0)),
    ("M1 PRO", _Peak("m1-pro", 200, 5.2, 0.0)),
    ("M1", _Peak("m1", 68, 2.6, 0.0)),
    ("M2 ULTRA", _Peak("m2-ultra", 800, 27.2, 0.0)),
    ("M2 MAX", _Peak("m2-max", 400, 13.6, 0.0)),
    ("M2 PRO", _Peak("m2-pro", 200, 6.8, 0.0)),
    ("M2", _Peak("m2", 100, 3.6, 0.0)),
    ("M3 ULTRA", _Peak("m3-ultra", 800, 28.4, 0.0)),
    ("M3 MAX", _Peak("m3-max", 400, 14.2, 0.0)),
    ("M3 PRO", _Peak("m3-pro", 150, 7.1, 0.0)),
    ("M3", _Peak("m3", 100, 4.1, 0.0)),
    ("M4 ULTRA", _Peak("m4-ultra", 1092, 36.8, 0.0)),
    ("M4 MAX", _Peak("m4-max", 546, 18.4, 0.0)),
    ("M4 PRO", _Peak("m4-pro", 273, 9.2, 0.0)),
    ("M4", _Peak("m4", 120, 4.6, 0.0)),
)
"""Apple Silicon, **approximate**.

Bandwidth is the published unified-memory figure. The TFLOPs column is an estimate of
sustained FP16 GPU throughput -- Apple publishes no dense tensor number, and the GPU
core count (which the base-die entries vary by, and which sysctl does not expose)
changes it by tens of percent. Treat these as the right order of magnitude, and prefer
a measured baseline over the roofline they imply.
"""

_BYTES_PER_GIB = 1024**3


def _match_nvidia(name: str, mem_gb: float) -> _Peak:
    # NVML reports usable memory (an "80GB" A100 shows ~79.4 GiB), so snap to the
    # nearest marketed size before matching.
    sizes = (16, 24, 32, 40, 48, 80, 96, 141, 192)
    marketed = min(sizes, key=lambda s: abs(s - mem_gb))
    haystack = f"{name.upper()} {marketed}GB"
    for needles, peak in NVIDIA_TABLE:
        if all(needle in haystack for needle in needles):
            return peak
    known = sorted({needles[0] for needles, _ in NVIDIA_TABLE})
    raise RuntimeError(
        f"no peak-performance entry for GPU {name!r}; add one to NVIDIA_TABLE or pass "
        f"--hardware <profile>. Known: {known}"
    )


def _decode(value: Any) -> str:
    """NVML returns ``bytes`` on older bindings and ``str`` on newer ones."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _interconnect(name: str, count: int) -> Literal["single", "nvlink", "pcie"]:
    """How the cards in this box talk to each other.

    SXM boards are the ones with NVLink between them; a multi-GPU PCIe box has to move
    tensors over the root complex, which is an order of magnitude slower and is what the
    communication diagnosis rule is looking for.
    """
    if count <= 1:
        return "single"
    return "nvlink" if "SXM" in name.upper() else "pcie"


def _detect_nvidia() -> HardwareProfile:
    nvml = importlib.import_module("pynvml")
    nvml.nvmlInit()
    try:
        count = int(nvml.nvmlDeviceGetCount())
        if count < 1:
            raise RuntimeError("NVML reports no devices")
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        name = _decode(nvml.nvmlDeviceGetName(handle))
        mem_gb = float(nvml.nvmlDeviceGetMemoryInfo(handle).total) / _BYTES_PER_GIB
    finally:
        nvml.nvmlShutdown()

    peak = _match_nvidia(name, mem_gb)
    return HardwareProfile(
        name=peak.slug,
        gpu=name,
        count=count,
        mem_gb=mem_gb,
        hbm_bw_gbs=peak.hbm_bw_gbs,
        peak_tflops=peak.peak_tflops,
        compute_capability=peak.compute_capability,
        interconnect=_interconnect(name, count),
    )


def sysctl(key: str) -> str:
    """One sysctl value as text. Monkeypatched in tests; not available off macOS."""
    out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _detect_apple() -> HardwareProfile:
    brand = sysctl("machdep.cpu.brand_string")
    upper = brand.upper()
    match = next((peak for prefix, peak in APPLE_TABLE if prefix in upper), None)
    if match is None:
        raise RuntimeError(
            f"unrecognised Apple chip {brand!r}; add it to APPLE_TABLE or pass --hardware <profile>"
        )
    mem_gb = float(sysctl("hw.memsize")) / _BYTES_PER_GIB
    return HardwareProfile(
        name=match.slug,
        gpu=brand,
        count=1,
        # Unified memory: the whole machine's RAM is the GPU's, less whatever macOS and
        # the rest of the system are holding. Sizing weights against all of it will be
        # optimistic; the launch that fails is the correction.
        mem_gb=mem_gb,
        hbm_bw_gbs=match.hbm_bw_gbs,
        peak_tflops=match.peak_tflops,
        compute_capability=match.compute_capability,
        interconnect="single",
    )


def detect_hardware() -> HardwareProfile:
    """Describe this machine, or say why it cannot be described.

    ``usd_per_hour`` is left at zero: what a box costs is a property of who rented it,
    not of the silicon, so the provider fills it in.
    """
    system = platform.system()
    if system == "Linux":
        try:
            return _detect_nvidia()
        except ImportError as e:
            raise RuntimeError(
                "hardware auto-detection on Linux needs pynvml and an NVIDIA driver; "
                "pass --hardware <profile> instead"
            ) from e
        except RuntimeError:
            raise  # an unrecognised card: the message already says what to do
        except Exception as e:  # NVML raises its own error type for a missing driver
            raise RuntimeError(f"NVML could not describe this machine: {e}") from e
    if system == "Darwin" and platform.machine() == "arm64":
        return _detect_apple()
    raise RuntimeError(
        f"no hardware auto-detection for {system}/{platform.machine()}; pass --hardware <profile>"
    )
