"""Hardware auto-detection, driven by a fake NVML module and a fake sysctl.

Nothing here touches a real driver: ``pynvml`` is injected into ``sys.modules`` and
``platform`` is monkeypatched, so the Linux path is exercised on a Mac and the Apple
path is exercised for chips this machine is not.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from infervolt.hardware import detect
from infervolt.hardware.detect import detect_hardware
from infervolt.hardware.profiles import get_profile

GIB = 1024**3


class FakeNvml:
    """The five NVML calls detect.py makes, and a record of the lifecycle ones."""

    def __init__(self, name: str, mem_gb: float, count: int = 1) -> None:
        self.name = name
        self.mem_bytes = int(mem_gb * GIB)
        self.count = count
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self) -> None:  # noqa: N802
        self.init_calls += 1

    def nvmlShutdown(self) -> None:  # noqa: N802
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:  # noqa: N802
        return index

    def nvmlDeviceGetName(self, handle: int) -> str:  # noqa: N802
        return self.name

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> Any:  # noqa: N802
        return types.SimpleNamespace(total=self.mem_bytes)


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect.platform, "system", lambda: "Linux")


@pytest.fixture
def darwin_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detect.platform, "machine", lambda: "arm64")


def install(monkeypatch: pytest.MonkeyPatch, nvml: FakeNvml) -> FakeNvml:
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    return nvml


def fake_sysctl(monkeypatch: pytest.MonkeyPatch, brand: str, mem_gb: float) -> None:
    values = {"machdep.cpu.brand_string": brand, "hw.memsize": str(int(mem_gb * GIB))}
    monkeypatch.setattr(detect, "sysctl", lambda key: values[key])


# ---------------------------------------------------------------- NVIDIA


def test_a100_80gb(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    nvml = install(monkeypatch, FakeNvml("NVIDIA A100-SXM4-80GB", 79.2))
    hw = detect_hardware()
    assert hw.name == "a100-80"
    assert hw.gpu == "NVIDIA A100-SXM4-80GB"
    assert hw.hbm_bw_gbs == 2039
    assert hw.peak_tflops == 312
    assert hw.compute_capability == 8.0
    assert hw.count == 1
    assert hw.interconnect == "single"
    assert hw.mem_gb == pytest.approx(79.2, abs=0.1)
    assert hw.usd_per_hour == 0.0
    assert (nvml.init_calls, nvml.shutdown_calls) == (1, 1)


def test_a100_40gb_is_told_apart_by_capacity(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeNvml("NVIDIA A100-PCIE-40GB", 39.6))
    hw = detect_hardware()
    assert hw.name == "a100-40"
    assert hw.hbm_bw_gbs == 1555


@pytest.mark.parametrize(
    ("name", "mem_gb", "slug", "bw", "tflops", "cc"),
    [
        ("NVIDIA H100 80GB HBM3", 80, "h100", 3350, 989, 9.0),
        ("NVIDIA H200", 141, "h200", 4800, 989, 9.0),
        ("NVIDIA L40S", 48, "l40", 864, 181, 8.9),
        ("NVIDIA L4", 24, "l4", 300, 121, 8.9),
        ("NVIDIA RTX A6000", 48, "a6000", 768, 155, 8.6),
        ("NVIDIA GeForce RTX 4090", 24, "rtx4090", 1008, 165, 8.9),
        ("NVIDIA GeForce RTX 5090", 32, "rtx5090", 1792, 210, 12.0),
        ("NVIDIA A10", 24, "a10", 600, 125, 8.6),
    ],
)
def test_the_gpu_table(
    linux: None,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mem_gb: float,
    slug: str,
    bw: float,
    tflops: float,
    cc: float,
) -> None:
    install(monkeypatch, FakeNvml(name, mem_gb))
    hw = detect_hardware()
    assert (hw.name, hw.hbm_bw_gbs, hw.peak_tflops, hw.compute_capability) == (slug, bw, tflops, cc)


def test_nvml_byte_names_are_decoded(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    nvml = FakeNvml("NVIDIA L4", 24)
    nvml.nvmlDeviceGetName = lambda handle: b"NVIDIA L4"
    install(monkeypatch, nvml)
    assert detect_hardware().gpu == "NVIDIA L4"


def test_multiple_sxm_cards_are_nvlinked(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeNvml("NVIDIA A100-SXM4-80GB", 80, count=8))
    hw = detect_hardware()
    assert hw.count == 8
    assert hw.interconnect == "nvlink"


def test_multiple_pcie_cards_are_not(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeNvml("NVIDIA A100-PCIE-40GB", 40, count=2))
    assert detect_hardware().interconnect == "pcie"


def test_an_unknown_card_says_what_to_do(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeNvml("NVIDIA Tesla K80", 12))
    with pytest.raises(RuntimeError, match="Tesla K80"):
        detect_hardware()


def test_no_devices_is_an_error(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeNvml("NVIDIA L4", 24, count=0))
    with pytest.raises(RuntimeError, match="no devices"):
        detect_hardware()


def test_a_driver_failure_is_reported_not_raised_raw(
    linux: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvml = FakeNvml("NVIDIA L4", 24)

    def boom() -> None:
        raise OSError("libnvidia-ml.so.1: cannot open shared object file")

    nvml.nvmlInit = boom
    install(monkeypatch, nvml)
    with pytest.raises(RuntimeError, match="NVML could not describe"):
        detect_hardware()


def test_a_missing_pynvml_says_so(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules is what an unimportable module looks like to import_module.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with pytest.raises(RuntimeError, match="needs pynvml"):
        detect_hardware()


# ---------------------------------------------------------------- Apple


@pytest.mark.parametrize(
    ("brand", "slug", "bw"),
    [
        ("Apple M1", "m1", 68),
        ("Apple M1 Pro", "m1-pro", 200),
        ("Apple M2 Max", "m2-max", 400),
        ("Apple M2 Ultra", "m2-ultra", 800),
        ("Apple M3", "m3", 100),
        ("Apple M3 Max", "m3-max", 400),
        ("Apple M4 Pro", "m4-pro", 273),
    ],
)
def test_the_apple_table(
    darwin_arm: None, monkeypatch: pytest.MonkeyPatch, brand: str, slug: str, bw: float
) -> None:
    fake_sysctl(monkeypatch, brand, 36)
    hw = detect_hardware()
    assert hw.name == slug
    assert hw.gpu == brand
    assert hw.hbm_bw_gbs == bw
    assert hw.compute_capability == 0.0
    assert hw.interconnect == "single"
    assert hw.count == 1
    assert hw.mem_gb == pytest.approx(36)


def test_a_base_chip_is_not_matched_as_a_pro(
    darwin_arm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_sysctl(monkeypatch, "Apple M3 Max", 48)
    assert detect_hardware().name == "m3-max"


def test_an_unknown_apple_chip_says_what_to_do(
    darwin_arm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_sysctl(monkeypatch, "Apple M9 Ultra", 64)
    with pytest.raises(RuntimeError, match="unrecognised Apple chip"):
        detect_hardware()


# ---------------------------------------------------------------- elsewhere


def test_intel_macs_and_other_platforms_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(detect.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="no hardware auto-detection"):
        detect_hardware()

    monkeypatch.setattr(detect.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="no hardware auto-detection"):
        detect_hardware()


# ---------------------------------------------------------------- wiring


def test_get_profile_auto_detects(linux: None, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeNvml("NVIDIA H100 80GB HBM3", 80))
    assert get_profile("auto").name == "h100"


def test_named_profiles_still_come_from_the_table() -> None:
    assert get_profile("a100-80").gpu == "NVIDIA A100-SXM4-80GB"
    with pytest.raises(KeyError, match="unknown hardware profile"):
        get_profile("no-such-card")
