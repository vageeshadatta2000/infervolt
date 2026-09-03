"""VllmAdapter: flag mapping, validation, launch/ready/stop, scraping, recipe rendering.

Nothing here installs or starts vLLM. The server process is a fake ``Popen`` and the HTTP
endpoints are an in-process :class:`~tests.fake_http.FakeServer` replaying recorded
``/metrics`` text, so the whole lifecycle runs on a laptop with no GPU.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from infervolt.core.types import (
    EngineConfig,
    KnobValue,
    LoadResult,
    RequestRecord,
    RunContext,
    Workload,
)
from infervolt.engines.base import ExitInfo, LaunchError, ServerHandle
from infervolt.engines.vllm import adapter as va
from infervolt.engines.vllm.adapter import VllmAdapter
from infervolt.hardware.profiles import get_profile
from infervolt.models.catalog import get_model_info
from infervolt.workloads.presets import get_workload, parse_slo
from tests.fake_http import FakeServer

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_ctx(
    tmp_path: Path,
    hardware: str = "rtx4090-24",
    model: str = "mock/qwen3-8b",
    workload: str = "chat-4k-512",
    gpu_count: int = 1,
) -> RunContext:
    hw = get_profile(hardware).model_copy(update={"count": gpu_count})
    return RunContext(
        run_id="run-test",
        run_dir=str(tmp_path),
        hw=hw,
        model=get_model_info(model),
        workload=get_workload(workload),
        slo=parse_slo("ttft=600ms,itl=30ms"),
        seed=7,
    )


def cfg_of(**knobs: KnobValue) -> EngineConfig:
    return EngineConfig(engine="vllm", knobs={**va.DEFAULT_KNOBS, **knobs})


class FakeProc:
    """Just enough of ``subprocess.Popen`` for launch/ready/stop.

    ``exit_after`` polls return None until that many polls have happened, which is how a
    server that dies while loading weights looks to :meth:`VllmAdapter.ready`.
    """

    def __init__(self, cmd: list[str], exit_after: int | None = None, code: int = 0) -> None:
        self.cmd = cmd
        self.pid = 999_999_999  # no such process: os.getpgid will refuse it
        self.returncode: int | None = None
        self.signals: list[int] = []
        self._polls = 0
        self._exit_after = exit_after
        self._code = code

    def poll(self) -> int | None:
        self._polls += 1
        if self._exit_after is not None and self._polls > self._exit_after:
            self.returncode = self._code
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = self._code
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = -sig


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[FakeProc]:
    """Replace ``subprocess.Popen`` with a recorder and hand back what was 'started'."""
    procs: list[FakeProc] = []

    def fake_popen(cmd: list[str], **kwargs: Any) -> FakeProc:
        proc = FakeProc(cmd)
        procs.append(proc)
        return proc

    monkeypatch.setattr(va.subprocess, "Popen", fake_popen)
    return procs


class StubLoadGen:
    """A load generator that records where it was pointed and returns one canned result."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def run(self, workload: Workload, concurrency: int, num_requests: int, seed: int) -> LoadResult:
        return LoadResult(
            concurrency=concurrency,
            duration_s=1.0,
            requests=[
                RequestRecord(ttft_s=0.1, itl_s=[0.01, 0.01], output_tokens=2)
                for _ in range(num_requests)
            ],
        )


# ---------------------------------------------------------------- knob space


def test_knob_space_mirrors_the_mock_knobs(tmp_path: Path) -> None:
    space = VllmAdapter().knob_space(make_ctx(tmp_path))
    assert space.names() == [
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "max_model_len",
        "enable_prefix_caching",
        "kv_cache_dtype",
        "enforce_eager",
        "speculative",
        "quantization",
        "tensor_parallel_size",
    ]
    assert set(space.groups()) == {"kv", "decode", "sched", "prefill", "parallel"}


def test_context_lengths_cover_the_workload_and_stop_at_the_model_cap(tmp_path: Path) -> None:
    # chat-4k-512 needs p99 ISL 6000 + p50 OSL 512 = 6512 tokens.
    space = VllmAdapter().knob_space(make_ctx(tmp_path))
    assert space.get("max_model_len").choices == [8192, 16384, 32768]
    assert space.get("max_model_len").default == 8192


def test_a_model_with_a_short_context_gets_one_capped_choice(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, workload="rag-16k-64")
    ctx = ctx.model_copy(update={"model": ctx.model.model_copy(update={"max_pos": 8192})})
    choices = VllmAdapter().knob_space(ctx).get("max_model_len").choices
    assert choices == [8192]
    # ...and the config it forces is then rejected, with the reason spelled out.
    errs = VllmAdapter().validate(cfg_of(max_model_len=8192), ctx)
    assert "max_model_len shorter than workload p99 ISL + OSL" in errs


def test_fp8_choices_are_gated_on_compute_capability(tmp_path: Path) -> None:
    adapter = VllmAdapter()
    ada = adapter.knob_space(make_ctx(tmp_path, hardware="rtx4090-24"))  # cc 8.9
    ampere = adapter.knob_space(make_ctx(tmp_path, hardware="a100-80"))  # cc 8.0
    apple = adapter.knob_space(make_ctx(tmp_path, hardware="m3-8"))  # cc 0.0
    hopper = adapter.knob_space(make_ctx(tmp_path, hardware="h100-80"))  # cc 9.0
    # fp8 KV needs FlashAttention 3 (Hopper); on Ada/Ampere vLLM silently falls back to
    # the slower Triton backend, so the knob is withheld there.
    assert hopper.get("kv_cache_dtype").choices == ["auto", "fp8"]
    assert ada.get("kv_cache_dtype").choices == ["auto"]
    assert ada.get("quantization").choices == ["none", "fp8"]
    assert ampere.get("kv_cache_dtype").choices == ["auto"]
    assert ampere.get("quantization").choices == ["none"]
    assert apple.get("kv_cache_dtype").choices == ["auto"]
    assert apple.get("quantization").choices == ["none"]


def test_eagle3_is_not_offered_without_a_draft_model(tmp_path: Path) -> None:
    assert VllmAdapter().knob_space(make_ctx(tmp_path)).get("speculative").choices == [
        "none",
        "ngram",
    ]


def test_tensor_parallel_choices_divide_both_the_gpus_and_the_kv_heads(tmp_path: Path) -> None:
    adapter = VllmAdapter()
    # mock/qwen3-8b has 8 KV heads.
    assert adapter.knob_space(make_ctx(tmp_path, gpu_count=4)).get(
        "tensor_parallel_size"
    ).choices == [1, 2, 4]
    # 6 GPUs: 3 and 6 divide the node but not the 8 KV heads.
    assert adapter.knob_space(make_ctx(tmp_path, gpu_count=6)).get(
        "tensor_parallel_size"
    ).choices == [1, 2]
    assert adapter.knob_space(make_ctx(tmp_path)).get("tensor_parallel_size").choices == [1]


# ---------------------------------------------------------------- validation


def test_validate_accepts_the_defaults(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    space = VllmAdapter().knob_space(ctx)
    assert VllmAdapter().validate(EngineConfig(engine="vllm", knobs=space.defaults()), ctx) == []


@pytest.mark.parametrize(
    ("knobs", "fragment"),
    [
        ({"kv_cache_dtype": "int8"}, "is not one of"),
        ({"enable_prefix_caching": "maybe"}, "is not a bool"),
        ({"max_num_seqs": 4096}, "outside [8, 1024]"),
        ({"max_num_seqs": "many"}, "is not a number"),
        ({"max_num_seqs": 12.5}, "is not an int"),
        ({"gpu_memory_utilization": 0.99}, "outside [0.7, 0.95]"),
    ],
)
def test_validate_reports_bad_knobs_instead_of_raising(
    tmp_path: Path, knobs: dict[str, KnobValue], fragment: str
) -> None:
    errs = VllmAdapter().validate(cfg_of(**knobs), make_ctx(tmp_path))
    assert any(fragment in e for e in errs), errs


def test_validate_rejects_fp8_weights_on_ampere(tmp_path: Path) -> None:
    """knob_space already withholds the choice; a recipe or a YAML can still ask for it."""
    errs = VllmAdapter().validate(
        cfg_of(quantization="fp8"), make_ctx(tmp_path, hardware="a100-80")
    )
    assert any("fp8 quantization needs compute capability >= 8.9" in e for e in errs)


def test_validate_rejects_fp8_kv_cache_below_ampere(tmp_path: Path) -> None:
    errs = VllmAdapter().validate(cfg_of(kv_cache_dtype="fp8"), make_ctx(tmp_path, hardware="m3-8"))
    assert any("fp8 kv_cache_dtype needs compute capability >= 9.0" in e for e in errs)


def test_a_prompt_must_fit_one_batch_when_chunked_prefill_is_off(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    bad = cfg_of(enable_chunked_prefill=False, max_model_len=8192, max_num_batched_tokens=2048)
    assert any(
        "max_num_batched_tokens must be >= max_model_len" in e
        for e in VllmAdapter().validate(bad, ctx)
    )
    good = bad.with_knobs(max_num_batched_tokens=8192)
    assert VllmAdapter().validate(good, ctx) == []
    # With chunking on, the same small token budget is exactly the point of the knob.
    assert VllmAdapter().validate(bad.with_knobs(enable_chunked_prefill=True), ctx) == []


def test_validate_rejects_a_tensor_parallel_size_that_does_not_shard(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, gpu_count=4)
    assert any(
        "does not divide gpu count 4" in e
        for e in VllmAdapter().validate(cfg_of(tensor_parallel_size=3), ctx)
    )
    heads = ctx.model_copy(
        update={"model": ctx.model.model_copy(update={"num_kv_heads": 6}), "hw": ctx.hw}
    )
    assert any(
        "does not divide num_kv_heads 6" in e
        for e in VllmAdapter().validate(cfg_of(tensor_parallel_size=4), heads)
    )


# ---------------------------------------------------------------- flag mapping


def test_to_args_renders_the_default_config(tmp_path: Path) -> None:
    args = VllmAdapter().to_args(cfg_of(max_model_len=8192))
    assert args == [
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "8192",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--kv-cache-dtype",
        "auto",
        "--tensor-parallel-size",
        "1",
    ]


def test_false_booleans_render_as_the_negative_flag_not_as_a_value() -> None:
    """``--enable-prefix-caching false`` would be read by vLLM as a positional argument."""
    args = VllmAdapter().to_args(cfg_of(enable_prefix_caching=False, enable_chunked_prefill="no"))
    assert "--no-enable-prefix-caching" in args
    assert "--no-enable-chunked-prefill" in args
    assert "false" not in args and "no" not in args


def test_enforce_eager_is_a_presence_flag() -> None:
    assert "--enforce-eager" not in VllmAdapter().to_args(cfg_of(enforce_eager=False))
    assert "--enforce-eager" in VllmAdapter().to_args(cfg_of(enforce_eager=True))


def test_ngram_speculation_is_one_json_argument() -> None:
    args = VllmAdapter().to_args(cfg_of(speculative="ngram"))
    payload = args[args.index("--speculative-config") + 1]
    assert json.loads(payload) == {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
    assert "--speculative-config" not in VllmAdapter().to_args(cfg_of(speculative="none"))


def test_quantization_none_emits_no_flag() -> None:
    assert "--quantization" not in VllmAdapter().to_args(cfg_of(quantization="none"))
    args = VllmAdapter().to_args(cfg_of(quantization="fp8"))
    assert args[args.index("--quantization") + 1] == "fp8"


def test_unknown_knobs_are_ignored_by_the_flag_mapping() -> None:
    args = VllmAdapter().to_args(EngineConfig(engine="vllm", knobs={"not_a_vllm_flag": 3}))
    assert args == []


# ---------------------------------------------------------------- launch / ready / stop


def test_launch_builds_the_venv_command_and_logs_it(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    ctx = make_ctx(tmp_path)
    adapter = VllmAdapter(vllm_bin="/opt/venv/bin/vllm", port=8123)
    handle = adapter.launch(cfg_of(max_model_len=8192), ctx)
    cmd = spawned[0].cmd
    assert cmd[:3] == ["/opt/venv/bin/vllm", "serve", "mock/qwen3-8b"]
    assert cmd[3:7] == ["--host", "127.0.0.1", "--port", "8123"]
    assert "--disable-log-requests" not in cmd
    assert "--max-num-seqs" in cmd
    assert handle.url == "http://127.0.0.1:8123"
    log = va._state(handle).log_path
    assert log.parent == tmp_path
    assert "vllm serve" in log.read_text()


def test_launch_uses_an_explicit_model_path_when_given(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    adapter = VllmAdapter(model_path="/workspace/models/qwen3-8b", port=8124)
    adapter.launch(cfg_of(), make_ctx(tmp_path))
    assert spawned[0].cmd[2] == "/workspace/models/qwen3-8b"


def test_docker_launch_names_the_container_and_maps_the_cache(
    tmp_path: Path, spawned: list[FakeProc], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", "/workspace/hf")
    adapter = VllmAdapter(install="docker", docker_image="vllm/vllm-openai:v0.11.0", port=8125)
    handle = adapter.launch(cfg_of(max_model_len=8192), make_ctx(tmp_path))
    cmd = spawned[0].cmd
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd and "--ipc=host" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "all"
    assert cmd[cmd.index("-p") + 1] == "8125:8000"
    assert cmd[cmd.index("-v") + 1] == "/workspace/hf:/root/.cache/huggingface"
    # Passed by name so the token never lands in the container's argv.
    assert cmd[cmd.index("-e") + 1] == "HF_TOKEN"
    assert cmd[cmd.index("--model") + 1] == "mock/qwen3-8b"
    assert cmd.index("vllm/vllm-openai:v0.11.0") < cmd.index("--model")
    container = va._state(handle).container
    assert container is not None and container.startswith("infervolt-vllm-run-test")


def test_launch_reports_a_missing_binary_as_a_launch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(cmd: list[str], **kwargs: Any) -> Any:
        raise FileNotFoundError("vllm")

    monkeypatch.setattr(va.subprocess, "Popen", boom)
    with pytest.raises(LaunchError) as ei:
        VllmAdapter().launch(cfg_of(), make_ctx(tmp_path))
    assert ei.value.exit.code == 127


def test_ready_polls_health_until_the_server_answers(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    with FakeServer(unhealthy_polls=2) as server:
        adapter = VllmAdapter(port=_port_of(server), ready_poll_s=0.01)
        handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
        assert adapter.ready(handle, timeout_s=5.0)
    assert server.health_polls == 3


def test_ready_times_out_without_raising(tmp_path: Path, spawned: list[FakeProc]) -> None:
    with FakeServer(unhealthy_polls=1000) as server:
        adapter = VllmAdapter(port=_port_of(server), ready_poll_s=0.01)
        handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
        assert adapter.ready(handle, timeout_s=0.2) is False


def test_ready_raises_with_the_log_tail_when_the_server_dies_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crash = (
        "ValueError: The model's max seq len (32768) is larger than the maximum number "
        "of tokens that can be stored in KV cache (12480).\n"
    )

    def fake_popen(cmd: list[str], **kwargs: Any) -> FakeProc:
        proc = FakeProc(cmd, exit_after=0, code=1)
        Path(kwargs["stdout"].name).write_text(crash)
        return proc

    monkeypatch.setattr(va.subprocess, "Popen", fake_popen)
    adapter = VllmAdapter(port=8199, ready_poll_s=0.01)
    handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
    with pytest.raises(LaunchError) as ei:
        adapter.ready(handle, timeout_s=5.0)
    assert ei.value.exit.code == 1
    assert "maximum number of tokens" in ei.value.exit.log_tail
    assert adapter.classify_crash(ei.value.exit) == "oom"


@pytest.mark.parametrize(
    "message",
    [
        "ValueError: No available memory for the cache blocks.",
        "The model's max seq len is larger than the maximum number of tokens that can be "
        "stored in KV cache",
        "torch.OutOfMemoryError: CUDA out of memory.",
    ],
)
def test_vllm_oom_messages_classify_as_oom(message: str) -> None:
    assert VllmAdapter().classify_crash(ExitInfo(code=1, log_tail=message)) == "oom"


def test_a_clean_exit_is_not_a_crash() -> None:
    assert VllmAdapter().classify_crash(ExitInfo(code=0)) == "none"
    assert VllmAdapter().classify_crash(ExitInfo(code=1, log_tail="bad flag")) == "runtime"


def test_stop_signals_the_whole_process_group(
    tmp_path: Path, spawned: list[FakeProc], monkeypatch: pytest.MonkeyPatch
) -> None:
    """vLLM's EngineCore workers are children; signalling only the leader leaks the GPU."""
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(va.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(va.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    adapter = VllmAdapter(port=8126)
    handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
    exit_info = adapter.stop(handle)
    assert killed == [(4242, signal.SIGTERM)]
    assert exit_info.code == 0
    assert "vllm serve" in exit_info.log_tail


def test_stop_falls_back_to_the_process_when_the_group_is_gone(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    adapter = VllmAdapter(port=8127)
    handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
    adapter.stop(handle)
    assert spawned[0].signals == [signal.SIGTERM]


def test_stop_removes_the_docker_container(
    tmp_path: Path, spawned: list[FakeProc], monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(
        va.subprocess,
        "run",
        lambda cmd, **kw: ran.append(cmd) or subprocess.CompletedProcess(cmd, 0, b"", b""),
    )
    adapter = VllmAdapter(install="docker", port=8128)
    handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
    container = va._state(handle).container
    adapter.stop(handle)
    assert ran == [["docker", "rm", "-f", container]]


def test_stop_escalates_to_sigkill_after_the_grace_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Stubborn(FakeProc):
        def __init__(self, cmd: list[str]) -> None:
            super().__init__(cmd)
            self.waits = 0

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(self.cmd, timeout or 0)
            self.returncode = -9
            return -9

    procs: list[Stubborn] = []
    monkeypatch.setattr(
        va.subprocess, "Popen", lambda cmd, **kw: procs.append(Stubborn(cmd)) or procs[-1]
    )
    sent: list[int] = []
    monkeypatch.setattr(va.os, "getpgid", lambda pid: 4243)
    monkeypatch.setattr(va.os, "killpg", lambda pgid, sig: sent.append(sig))
    monkeypatch.setattr(va, "STOP_GRACE_S", 0.01)
    adapter = VllmAdapter(port=8129)
    handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
    exit_info = adapter.stop(handle)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert exit_info.code == -9


# ---------------------------------------------------------------- observation


def _port_of(server: FakeServer) -> int:
    return int(server.url.rsplit(":", 1)[1])


def _run_one_load_point(
    adapter: VllmAdapter, handle: ServerHandle, ctx: RunContext, server: FakeServer
) -> dict[str, float]:
    """One load point's worth of sampling, with exactly two snapshots in the window.

    The adapter is built with a sampling interval far longer than the test, so the
    background thread scrapes once; ``scrape`` then takes the closing sample itself. That
    makes the window deterministic -- otherwise the number of repeats of the last fixture
    would move every mean and percentile the assertions below pin down.
    """
    adapter.loadgen(handle, ctx)
    deadline = time.monotonic() + 5.0
    while server.scrapes < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    engine = adapter.scrape(handle)
    assert server.scrapes == 2
    return engine


def test_scrape_returns_the_canonical_keys_for_the_load_window(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    texts = [fixture("metrics_v0.11.txt"), fixture("metrics_v0.11_later.txt")]
    ctx = make_ctx(tmp_path)
    cfg = cfg_of(max_model_len=8192, max_num_seqs=128, kv_cache_dtype="fp8")
    with FakeServer(metrics=texts) as server:
        adapter = VllmAdapter(
            port=_port_of(server),
            loadgen_factory=StubLoadGen,
            sample_interval_s=30.0,
            gpu_probe=lambda: (80.0, 60.0),
        )
        handle = adapter.launch(cfg, ctx)
        handle.config = cfg
        engine = _run_one_load_point(adapter, handle, ctx, server)
        gpu = adapter.gpu_stats(handle)
    assert engine["num_running"] == 64.0
    assert engine["num_waiting"] == pytest.approx(10.0)
    assert engine["kv_usage_p95"] == pytest.approx(0.867, abs=1e-3)
    assert engine["preemptions_per_s"] > 0
    assert engine["queue_time_p90_s"] == pytest.approx(0.65)
    assert engine["prefill_time_p50_s"] == pytest.approx(0.4)
    assert engine["prefill_share"] == pytest.approx(0.3)
    assert engine["prefix_hit_rate"] == pytest.approx(0.62)
    # Not scraped: what the config asked for.
    assert engine["max_num_seqs"] == 128.0
    assert engine["kv_dtype_bytes"] == 1.0
    assert gpu == {"sm_active": 0.8, "dram_active": 0.6}


def test_the_resolved_names_go_to_the_log_not_into_the_observation(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    """``_resolved`` is provenance; a stray key would reach the rules as a measurement."""
    ctx = make_ctx(tmp_path)
    with FakeServer(metrics=[fixture("metrics_v0.9_gpu_cache.txt")]) as server:
        adapter = VllmAdapter(
            port=_port_of(server),
            loadgen_factory=StubLoadGen,
            sample_interval_s=30.0,
            gpu_probe=lambda: None,
        )
        handle = adapter.launch(cfg_of(), ctx)
        handle.config = cfg_of()
        engine = _run_one_load_point(adapter, handle, ctx, server)
    assert "_resolved" not in engine
    state = va._state(handle)
    assert state.resolved["kv_usage"] == "vllm:gpu_cache_usage_perc"
    log = state.log_path.read_text()
    assert "resolved metrics" in log
    assert '"kv_usage": "vllm:gpu_cache_usage_perc"' in log


def test_loadgen_is_pointed_at_the_server_and_the_model(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    adapter = VllmAdapter(
        port=8130, loadgen_factory=StubLoadGen, model_path="/models/q", gpu_probe=lambda: None
    )
    handle = adapter.launch(cfg_of(), make_ctx(tmp_path))
    gen = adapter.loadgen(handle, make_ctx(tmp_path))
    adapter.scrape(handle)
    assert isinstance(gen, StubLoadGen)
    assert gen.base_url == "http://127.0.0.1:8130"
    assert gen.model == "/models/q"


def test_scrape_without_a_reachable_server_reports_only_the_config_keys(
    tmp_path: Path, spawned: list[FakeProc]
) -> None:
    adapter = VllmAdapter(
        port=1, loadgen_factory=StubLoadGen, sample_interval_s=0.01, gpu_probe=lambda: None
    )
    ctx = make_ctx(tmp_path)
    handle = adapter.launch(cfg_of(), ctx)
    handle.config = cfg_of()
    adapter.loadgen(handle, ctx)
    engine = adapter.scrape(handle)
    assert engine == {"max_num_seqs": 256.0, "kv_dtype_bytes": 2.0}
    assert adapter.gpu_stats(handle) == {}


def test_a_foreign_handle_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        VllmAdapter().scrape(ServerHandle(url="http://x", state=object()))


# ---------------------------------------------------------------- recipe and version


def test_to_recipe_block_renders_a_pasteable_serve_command(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    cfg = cfg_of(max_model_len=8192, kv_cache_dtype="fp8", speculative="ngram", enforce_eager=True)
    knobs, cmd = VllmAdapter(vllm_bin="vllm").to_recipe_block(cfg, ctx)
    assert knobs == dict(cfg.knobs)
    assert cmd.startswith("vllm serve mock/qwen3-8b ")
    assert "--kv-cache-dtype fp8" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--enforce-eager" in cmd
    # The JSON argument is quoted, so the line survives a copy into a shell.
    assert '--speculative-config \'{"method": "ngram"' in cmd


def test_version_reads_the_binary_for_a_venv_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        va.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "0.11.0\n", ""),
    )
    assert VllmAdapter().version().version == "0.11.0"


def test_version_reads_the_image_digest_for_a_docker_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(
        va.subprocess,
        "run",
        lambda cmd, **kw: (
            seen.append(cmd)
            or subprocess.CompletedProcess(cmd, 0, "vllm/vllm-openai@sha256:abc\n", "")
        ),
    )
    version = VllmAdapter(install="docker").version()
    assert version.image_digest == "vllm/vllm-openai@sha256:abc"
    assert seen[0][:2] == ["docker", "image"]


def test_version_is_empty_when_vllm_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cmd: list[str], **kw: Any) -> Any:
        raise FileNotFoundError("vllm")

    monkeypatch.setattr(va.subprocess, "run", boom)
    assert VllmAdapter().version() == va.EngineVersion(name="vllm")


def test_the_adapter_is_registered_under_the_vllm_entry_point() -> None:
    from infervolt.engines.registry import available_engines, get_adapter

    assert "vllm" in available_engines()
    assert isinstance(get_adapter("vllm"), VllmAdapter)


def test_free_port_hands_out_a_bindable_port() -> None:
    port = va.free_port()
    assert 1024 < port < 65536


def test_as_bool_reads_cli_spellings() -> None:
    assert va.as_bool("false") is False and va.as_bool("YES") is True
    assert va.as_bool(0) is False and va.as_bool(True) is True
    with pytest.raises(ValueError, match="cannot read"):
        va.as_bool("perhaps")


def test_launch_log_names_are_unique_per_trial(tmp_path: Path, spawned: list[FakeProc]) -> None:
    adapter = VllmAdapter(port=8131)
    ctx = make_ctx(tmp_path)
    first = va._state(adapter.launch(cfg_of(), ctx)).log_path
    second = va._state(adapter.launch(cfg_of(), ctx)).log_path
    assert first != second
    assert {p.name for p in tmp_path.iterdir() if p.suffix == ".log"} == {first.name, second.name}
