"""Named hardware profiles.

Peak numbers are dense BF16 TFLOPs and HBM bandwidth from vendor specs.
"""

from __future__ import annotations

from infervolt.core.types import HardwareProfile

PROFILES: dict[str, HardwareProfile] = {
    "a100-80": HardwareProfile(
        name="a100-80",
        gpu="NVIDIA A100-SXM4-80GB",
        mem_gb=80,
        hbm_bw_gbs=2039,
        peak_tflops=312,
        compute_capability=8.0,
        usd_per_hour=1.5,
    ),
    "h100-80": HardwareProfile(
        name="h100-80",
        gpu="NVIDIA H100 80GB HBM3",
        mem_gb=80,
        hbm_bw_gbs=3350,
        peak_tflops=989,
        compute_capability=9.0,
        usd_per_hour=3.0,
    ),
    "rtx4090-24": HardwareProfile(
        name="rtx4090-24",
        gpu="NVIDIA GeForce RTX 4090",
        mem_gb=24,
        hbm_bw_gbs=1008,
        peak_tflops=165,
        compute_capability=8.9,
        usd_per_hour=0.5,
    ),
    "l4-24": HardwareProfile(
        name="l4-24",
        gpu="NVIDIA L4",
        mem_gb=24,
        hbm_bw_gbs=300,
        peak_tflops=121,
        compute_capability=8.9,
        usd_per_hour=0.6,
    ),
    "m3-8": HardwareProfile(
        name="m3-8",
        gpu="Apple M3 (10-core GPU)",
        mem_gb=8,
        hbm_bw_gbs=100,
        peak_tflops=3.5,
        compute_capability=0.0,
        usd_per_hour=0.0,
    ),
}


def get_profile(name: str) -> HardwareProfile:
    return PROFILES[name]
