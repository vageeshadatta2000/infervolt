"""Real vLLM adapter: launches ``vllm serve`` in a venv or the official container.

The knob space mirrors the mock's -- the mock was modelled on vLLM in the first place --
but the bounds and the choices are the real ones, gated on what the card can actually do:
fp8 KV cache needs Ampere, fp8 weights need Ada, and tensor parallelism has to divide
both the GPU count and the model's KV heads.

Nothing in this module imports vLLM. The adapter shells out to an executable, so
infervolt keeps installing on a laptop with no CUDA, and the tests drive it with fake
subprocesses and recorded ``/metrics`` text.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from infervolt.core.types import (
    EngineConfig,
    Knob,
    KnobSpace,
    KnobValue,
    RunContext,
)
from infervolt.engines.base import EngineAdapter, EngineVersion, ExitInfo, LaunchError, ServerHandle
from infervolt.engines.vllm import metrics as m
from infervolt.loadgen.base import LoadGenerator

Install = Literal["venv", "docker"]
LoadGenFactory = Callable[[str, str], LoadGenerator]
"""(base_url, model) -> a load generator pointed at that OpenAI-compatible server."""

MAX_MODEL_LEN_CHOICES: tuple[int, ...] = (4096, 8192, 16384, 32768)
"""Context lengths offered to the search, before the workload and the model cap them."""

INT_KNOBS = frozenset({"max_num_seqs", "max_num_batched_tokens"})

FP8_KV_COMPUTE_CAPABILITY = (
    9.0  # FlashAttention fp8 KV needs Hopper; Ampere silently falls back to Triton
)
"""Ampere and later. vLLM's fp8 KV cache is a storage format, not a GEMM, so it does not
need fp8 tensor cores -- but it does need the conversion kernels, which start at sm80."""

FP8_QUANT_COMPUTE_CAPABILITY = 8.9
"""Ada and later. fp8 *weights* run on fp8 tensor cores; Ampere has none, so the config
does not run slowly there, it does not run."""

NGRAM_SPECULATIVE_CONFIG: dict[str, Any] = {
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 4,
}
"""vLLM V1 spelling of n-gram speculative decoding, passed as one JSON argument."""

DEFAULT_KNOBS: dict[str, KnobValue] = {
    "max_num_seqs": 256,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.9,
    "max_model_len": MAX_MODEL_LEN_CHOICES[0],
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "kv_cache_dtype": "auto",
    "enforce_eager": False,
    "speculative": "none",
    "quantization": "none",
    "tensor_parallel_size": 1,
}

STOP_GRACE_S = 30.0
"""How long SIGTERM (or ``docker rm -f``) gets before SIGKILL."""

LOG_TAIL_BYTES = 8192
READY_POLL_S = 0.5
SAMPLE_INTERVAL_S = 0.5

_TRUE_WORDS = frozenset({"true", "1", "yes"})
_FALSE_WORDS = frozenset({"false", "0", "no"})


def as_bool(value: KnobValue) -> bool:
    """Read a knob as a bool the way a CLI flag would.

    ``bool()`` is wrong: every non-empty string is truthy, so ``bool("false")`` is True
    and a knob set from YAML or a command line silently inverts.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ValueError(f"cannot read {value!r} as a bool; use true/false, 1/0 or yes/no")


def as_number(value: KnobValue) -> float | None:
    """Read a knob as a number, or None if it is not one. Bools are rejected outright."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def free_port() -> int:
    """An ephemeral port the OS has just confirmed is free.

    Racy in principle -- something else can claim it between the close and the server's
    bind -- but the alternative is a fixed port that collides with the *previous* trial's
    server, which is not a race but a certainty when a teardown is slow.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _fmt(value: KnobValue) -> str:
    """Render a knob value for a command line, without Python's float noise."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass
class VllmState:
    """Everything :class:`VllmAdapter` needs to talk to, sample, and kill one server."""

    port: int
    log_path: Path
    sampler: m.MetricsSampler
    proc: subprocess.Popen[bytes] | None = None
    container: str | None = None
    resolved: dict[str, str] = field(default_factory=dict)
    window: m.Window = field(default_factory=m.Window)
    _resolution_logged: bool = False


def _state(handle: ServerHandle) -> VllmState:
    """Narrow ``ServerHandle.state``, which is deliberately ``Any`` in the base contract."""
    st = handle.state
    if not isinstance(st, VllmState):
        raise TypeError(f"handle was not produced by VllmAdapter.launch: {st!r}")
    return st


class VllmAdapter(EngineAdapter):
    name = "vllm"

    def __init__(
        self,
        install: Install = "venv",
        vllm_bin: str = "vllm",
        docker_image: str = "vllm/vllm-openai:latest",
        model_path: str | None = None,
        loadgen_factory: LoadGenFactory | None = None,
        port: int | None = None,
        gpu_probe: m.GpuProbe | None = None,
        sample_interval_s: float = SAMPLE_INTERVAL_S,
        ready_poll_s: float = READY_POLL_S,
    ) -> None:
        self.install = install
        self.vllm_bin = vllm_bin
        self.docker_image = docker_image
        self.model_path = model_path
        self.loadgen_factory = loadgen_factory or _default_loadgen
        self.port = port
        self.gpu_probe = gpu_probe
        self.sample_interval_s = sample_interval_s
        self.ready_poll_s = ready_poll_s
        self._launches = 0

    # ---------------------------------------------------------------- identity

    def version(self) -> EngineVersion:
        """``vllm --version`` for a venv install, the image digest for a container one.

        Best effort: a controller that has never installed vLLM (the Mac) still has to be
        able to render a recipe, so a missing binary yields an empty version rather than
        an exception.
        """
        if self.install == "docker":
            digest = self._capture(
                ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}"]
                + [self.docker_image]
            )
            return EngineVersion(name="vllm", image_digest=digest)
        return EngineVersion(name="vllm", version=self._capture([self.vllm_bin, "--version"]))

    @staticmethod
    def _capture(cmd: list[str]) -> str:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def model_ref(self, ctx: RunContext) -> str:
        """What to hand ``vllm serve``: the explicit local path if given, else the HF id."""
        return self.model_path or ctx.model.id

    # ---------------------------------------------------------------- knobs

    def _len_choices(self, ctx: RunContext) -> list[int]:
        """Offered context lengths: long enough for the workload, short enough for the model.

        A length below the workload is rejected by ``validate`` anyway, so offering it
        would only spend a trial; a length above ``model.max_pos`` is refused by vLLM
        itself at startup. When nothing in the table fits between the two, the single
        best available length is offered so the space is never empty -- ``validate`` then
        says why the workload cannot run here.
        """
        need = ctx.workload.isl.p99 + ctx.workload.osl.p50
        cap = ctx.model.max_pos
        choices = [c for c in MAX_MODEL_LEN_CHOICES if need <= c <= cap]
        return choices or [min(max(need, MAX_MODEL_LEN_CHOICES[0]), cap)]

    @staticmethod
    def _tp_choices(ctx: RunContext) -> list[int]:
        """Tensor-parallel sizes that divide both the GPU count and the model's KV heads.

        vLLM shards KV heads across ranks, so a TP that does not divide ``num_kv_heads``
        either fails at startup or silently replicates heads; a TP above the GPU count
        has nowhere to run.
        """
        heads = divisors(max(1, ctx.model.num_kv_heads))
        return [d for d in divisors(max(1, ctx.hw.count)) if d in heads] or [1]

    def knob_space(self, ctx: RunContext) -> KnobSpace:
        d = DEFAULT_KNOBS
        len_choices = self._len_choices(ctx)
        cc = ctx.hw.compute_capability
        kv_dtypes: list[KnobValue] = ["auto"]
        if cc >= FP8_KV_COMPUTE_CAPABILITY:
            kv_dtypes.append("fp8")
        quant: list[KnobValue] = ["none"]
        if cc >= FP8_QUANT_COMPUTE_CAPABILITY:
            quant.append("fp8")
        tp_choices = self._tp_choices(ctx)
        return KnobSpace(
            knobs=[
                Knob(
                    name="max_num_seqs",
                    kind="int",
                    groups=["kv", "decode", "sched"],
                    default=d["max_num_seqs"],
                    low=8,
                    high=1024,
                    log=True,
                ),
                Knob(
                    name="max_num_batched_tokens",
                    kind="int",
                    groups=["prefill"],
                    default=d["max_num_batched_tokens"],
                    low=512,
                    high=16384,
                    log=True,
                ),
                Knob(
                    name="gpu_memory_utilization",
                    kind="float",
                    groups=["kv"],
                    default=d["gpu_memory_utilization"],
                    low=0.7,
                    high=0.95,
                    step=0.05,
                ),
                Knob(
                    name="max_model_len",
                    kind="cat",
                    groups=["kv"],
                    default=len_choices[0],
                    choices=list(len_choices),
                ),
                Knob(
                    name="enable_prefix_caching",
                    kind="bool",
                    groups=["kv"],
                    default=d["enable_prefix_caching"],
                ),
                Knob(
                    name="kv_cache_dtype",
                    kind="cat",
                    groups=["kv", "decode"],
                    default=d["kv_cache_dtype"],
                    choices=kv_dtypes,
                ),
                Knob(
                    name="enforce_eager",
                    kind="bool",
                    groups=["sched"],
                    default=d["enforce_eager"],
                ),
                # eagle3 is deliberately absent: it needs a draft checkpoint configured
                # alongside the target model, and offering it without one only buys a
                # trial that cannot launch.
                Knob(
                    name="speculative",
                    kind="cat",
                    groups=["decode"],
                    default=d["speculative"],
                    choices=["none", "ngram"],
                ),
                Knob(
                    name="quantization",
                    kind="cat",
                    groups=["prefill", "decode"],
                    default=d["quantization"],
                    choices=quant,
                ),
                # Also in "kv": sharding the model across ranks is one of the few ways to
                # buy KV capacity, so a kv_capacity diagnosis should be allowed to reach
                # for it, not only a communication one.
                Knob(
                    name="tensor_parallel_size",
                    kind="cat",
                    groups=["parallel", "kv"],
                    default=tp_choices[0],
                    choices=list(tp_choices),
                ),
            ]
        )

    # ---------------------------------------------------------------- validation

    @staticmethod
    def _numeric_errors(knob: Knob, value: KnobValue) -> list[str]:
        num = as_number(value)
        if num is None:
            return [f"{knob.name}={value!r} is not a number"]
        if knob.name in INT_KNOBS and not float(num).is_integer():
            return [f"{knob.name}={value!r} is not an int"]
        if knob.low is not None and knob.high is not None and not knob.low <= num <= knob.high:
            return [f"{knob.name}={value} outside [{knob.low}, {knob.high}]"]
        return []

    def validate(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        """Every reason this config must not be launched, as strings. Never raises.

        The hardware gates repeat what ``knob_space`` already withheld, because a config
        can arrive from a user's YAML or a cached recipe without passing through the
        space at all.
        """
        errs: list[str] = []
        space = self.knob_space(ctx)
        for knob in space.knobs:
            if knob.name not in cfg.knobs:
                continue
            value = cfg.knobs[knob.name]
            if knob.kind == "cat":
                if not _in_choices(value, knob.choices):
                    errs.append(f"{knob.name}={value!r} is not one of {knob.choices!r}")
            elif knob.kind == "bool":
                try:
                    as_bool(value)
                except ValueError:
                    errs.append(
                        f"{knob.name}={value!r} is not a bool; "
                        f"choices: ['true', 'false', '1', '0', 'yes', 'no']"
                    )
            else:
                errs.extend(self._numeric_errors(knob, value))
        errs.extend(self._static_constraints(cfg, ctx))
        return errs

    def _static_constraints(self, cfg: EngineConfig, ctx: RunContext) -> list[str]:
        """vLLM's own refusals, the ones it makes before it ever allocates anything."""
        errs: list[str] = []
        cc = ctx.hw.compute_capability
        if cfg.knobs.get("quantization") == "fp8" and cc < FP8_QUANT_COMPUTE_CAPABILITY:
            errs.append(
                f"fp8 quantization needs compute capability >= {FP8_QUANT_COMPUTE_CAPABILITY}"
            )
        if cfg.knobs.get("kv_cache_dtype") == "fp8" and cc < FP8_KV_COMPUTE_CAPABILITY:
            errs.append(
                f"fp8 kv_cache_dtype needs compute capability >= {FP8_KV_COMPUTE_CAPABILITY}"
            )
        max_len = as_number(cfg.knobs.get("max_model_len", 0))
        need = ctx.workload.isl.p99 + ctx.workload.osl.p50
        if max_len is not None and max_len > 0:
            if max_len < need:
                errs.append("max_model_len shorter than workload p99 ISL + OSL")
            if max_len > ctx.model.max_pos:
                errs.append(f"max_model_len above the model's max_pos ({ctx.model.max_pos})")
            # Without chunked prefill a prompt has to fit in one batch, so vLLM refuses a
            # token budget below the context length outright rather than at the first
            # long request.
            chunked = cfg.knobs.get("enable_chunked_prefill", True)
            batched = as_number(cfg.knobs.get("max_num_batched_tokens", 0))
            if not _safe_bool(chunked) and batched is not None and 0 < batched < max_len:
                errs.append(
                    "max_num_batched_tokens must be >= max_model_len when chunked prefill is off"
                )
        tp = as_number(cfg.knobs.get("tensor_parallel_size", 1))
        if tp is not None and tp >= 1 and float(tp).is_integer():
            tp_int = int(tp)
            if ctx.hw.count % tp_int:
                errs.append(
                    f"tensor_parallel_size {tp_int} does not divide gpu count {ctx.hw.count}"
                )
            elif ctx.model.num_kv_heads % tp_int:
                errs.append(
                    f"tensor_parallel_size {tp_int} does not divide "
                    f"num_kv_heads {ctx.model.num_kv_heads}"
                )
        return errs

    # ---------------------------------------------------------------- flags

    def to_args(self, cfg: EngineConfig) -> list[str]:
        """The knobs as ``vllm serve`` flags, in a fixed order.

        Booleans render as vLLM's paired flags (``--enable-x`` / ``--no-enable-x``) rather
        than as ``--enable-x false``, which vLLM would read as a positional argument;
        ``--enforce-eager`` has no negative form, so it is emitted only when true.
        """
        k = cfg.knobs
        args: list[str] = []
        for name, flag in (
            ("max_num_seqs", "--max-num-seqs"),
            ("max_num_batched_tokens", "--max-num-batched-tokens"),
            ("gpu_memory_utilization", "--gpu-memory-utilization"),
            ("max_model_len", "--max-model-len"),
        ):
            if name in k:
                args += [flag, _fmt(k[name])]
        for name, flag in (
            ("enable_prefix_caching", "enable-prefix-caching"),
            ("enable_chunked_prefill", "enable-chunked-prefill"),
        ):
            if name in k:
                args.append(f"--{flag}" if as_bool(k[name]) else f"--no-{flag}")
        if "kv_cache_dtype" in k:
            args += ["--kv-cache-dtype", _fmt(k["kv_cache_dtype"])]
        if "enforce_eager" in k and as_bool(k["enforce_eager"]):
            args.append("--enforce-eager")
        if str(k.get("speculative", "none")) == "ngram":
            args += ["--speculative-config", json.dumps(NGRAM_SPECULATIVE_CONFIG)]
        quantization = str(k.get("quantization", "none"))
        if quantization != "none":
            args += ["--quantization", quantization]
        if "tensor_parallel_size" in k:
            args += ["--tensor-parallel-size", _fmt(k["tensor_parallel_size"])]
        return args

    # ---------------------------------------------------------------- lifecycle

    def launch(self, cfg: EngineConfig, ctx: RunContext) -> ServerHandle:
        port = self.port or free_port()
        self._launches += 1
        run_dir = Path(ctx.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"vllm-{self._launches:03d}.log"
        name = f"infervolt-vllm-{ctx.run_id}-{self._launches:03d}"
        container = name if self.install == "docker" else None
        cmd = (
            self._docker_cmd(cfg, ctx, port, container or "")
            if self.install == "docker"
            else self._venv_cmd(cfg, ctx, port)
        )
        url = f"http://127.0.0.1:{port}"
        sampler = m.MetricsSampler(
            f"{url}/metrics", interval_s=self.sample_interval_s, gpu_probe=self.gpu_probe
        )
        state = VllmState(port=port, log_path=log_path, sampler=sampler, container=container)
        log_path.write_text(f"$ {shlex.join(cmd)}\n", encoding="utf-8")
        try:
            with log_path.open("ab") as log:
                state.proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as e:
            # A missing binary is this trial's launch failure, not the run's: the runner
            # contracts ``launch`` to signal every failure as a LaunchError.
            raise LaunchError(ExitInfo(code=127, log_tail=f"{type(e).__name__}: {e}")) from e
        return ServerHandle(url=url, state=state)

    def _venv_cmd(self, cfg: EngineConfig, ctx: RunContext, port: int) -> list[str]:
        return [
            self.vllm_bin,
            "serve",
            self.model_ref(ctx),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *self.to_args(cfg),
        ]

    def _docker_cmd(
        self, cfg: EngineConfig, ctx: RunContext, port: int, container: str
    ) -> list[str]:
        """``docker run`` in the foreground, so the log file and the exit code work as usual.

        ``HF_TOKEN`` is passed by name, not by value: docker reads it from our own
        environment, so the token never appears in the container's argv where any
        ``ps`` or ``docker inspect`` would show it.
        """
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--gpus",
            "all",
            "--ipc=host",
            "-p",
            f"{port}:8000",
            "-v",
            f"{hf_home}:/root/.cache/huggingface",
            "-e",
            "HF_TOKEN",
            self.docker_image,
            "--model",
            self.model_ref(ctx),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            *self.to_args(cfg),
        ]

    def ready(self, handle: ServerHandle, timeout_s: float) -> bool:
        """Poll ``/health`` until it answers 200, the process dies, or the clock runs out.

        A server that exits during startup raises rather than returning False: the exit
        code and the log tail are what tell an OOM apart from a bad flag, and reporting
        the death as a plain timeout would throw both away.
        """
        st = _state(handle)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code = st.proc.poll() if st.proc is not None else None
            if code is not None:
                raise LaunchError(ExitInfo(code=code, log_tail=self._log_tail(st)))
            got = m.http_get(f"{handle.url}/health", timeout_s=5.0)
            if got is not None and got[0] == 200:
                return True
            time.sleep(self.ready_poll_s)
        return False

    def loadgen(self, handle: ServerHandle, ctx: RunContext) -> LoadGenerator:
        """Hand out the generator and start sampling: the window is the load point."""
        st = _state(handle)
        st.sampler.start()
        return self.loadgen_factory(handle.url, self.model_ref(ctx))

    # ---------------------------------------------------------------- observation

    def scrape(self, handle: ServerHandle) -> dict[str, float]:
        """Canonical engine metrics for the load point that just finished.

        Stops the sampler started by :meth:`loadgen`, resolves the metric names against
        what this server actually exposes (once, then remembered on the handle), and
        reduces the window. The resolved mapping goes to the log rather than into the
        returned dict: the diagnosis rules read canonical keys only, and a stray
        ``_resolved`` entry would land in the observation as if it were a measurement.
        """
        st = _state(handle)
        st.window = st.sampler.stop()
        snaps = st.window.snapshots
        if snaps and not st.resolved:
            st.resolved = m.resolve_names(snaps[-1].families())
            self._log_resolution(st)
        out = m.aggregate(snaps, st.resolved)
        cfg = handle.config
        if cfg is not None:
            # Not scraped: the scheduler's admission limit and the KV element width are
            # what we *asked* for, and the rules compare the measured occupancy to them.
            seqs = as_number(cfg.knobs.get("max_num_seqs", 0))
            if seqs:
                out["max_num_seqs"] = seqs
            out["kv_dtype_bytes"] = 1.0 if cfg.knobs.get("kv_cache_dtype") == "fp8" else 2.0
        return out

    def _log_resolution(self, st: VllmState) -> None:
        if st._resolution_logged:
            return
        st._resolution_logged = True
        line = f"infervolt: resolved metrics {json.dumps(st.resolved, sort_keys=True)}\n"
        # A log we cannot write to must not fail the trial.
        with contextlib.suppress(OSError), st.log_path.open("a", encoding="utf-8") as log:
            log.write(line)

    def gpu_stats(self, handle: ServerHandle) -> dict[str, float]:
        """Mean SM and DRAM utilization over the same window ``scrape`` reduced.

        Empty when no GPU probe answered -- there is no NVML on the machines CI runs on,
        and a fabricated 0.0 would tell the rules the card was idle.
        """
        return m.gpu_stats(_state(handle).window)

    # ---------------------------------------------------------------- teardown

    def stop(self, handle: ServerHandle) -> ExitInfo:
        """SIGTERM the process group (or ``docker rm -f``), then SIGKILL after the grace period."""
        st = _state(handle)
        if st.sampler.running:
            st.window = st.sampler.stop()
        proc = st.proc
        if proc is None:
            return ExitInfo(code=0, log_tail=self._log_tail(st))
        if st.container:
            self._run_quiet(["docker", "rm", "-f", st.container])
        else:
            self._signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._signal_group(proc, signal.SIGKILL)
            # A process that ignores SIGKILL is stuck in an uninterruptible driver call;
            # nothing here can reap it, and the trial's result is still worth returning.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=STOP_GRACE_S)
        code = proc.returncode
        return ExitInfo(code=0 if code is None else int(code), log_tail=self._log_tail(st))

    @staticmethod
    def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
        """Signal the whole session we started, so vLLM's worker processes go too.

        ``start_new_session=True`` put the server in its own process group; signalling
        only the leader leaves the EngineCore workers holding the GPU, and the next
        trial then OOMs for reasons that have nothing to do with its config.
        """
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            # The group is already gone, or was never ours; the leader is the best we can do.
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.send_signal(sig)

    @staticmethod
    def _run_quiet(cmd: list[str]) -> None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(cmd, capture_output=True, timeout=STOP_GRACE_S, check=False)

    @staticmethod
    def _log_tail(st: VllmState) -> str:
        try:
            with st.log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - LOG_TAIL_BYTES))
                return log.read().decode("utf-8", "replace")
        except OSError:
            return ""

    # ---------------------------------------------------------------- recipe

    def to_recipe_block(
        self, cfg: EngineConfig, ctx: RunContext
    ) -> tuple[dict[str, KnobValue], str]:
        args = self.to_args(cfg)
        cmd = shlex.join([self.vllm_bin, "serve", self.model_ref(ctx), *args])
        return dict(cfg.knobs), cmd


def _in_choices(value: KnobValue, choices: list[KnobValue]) -> bool:
    """Membership that survives a round trip through YAML, where 4096 may arrive as "4096"."""
    if value in choices:
        return True
    return any(str(value) == str(c) for c in choices)


def _safe_bool(value: KnobValue) -> bool:
    """``as_bool`` for the constraint checks, where an unreadable value is already reported."""
    try:
        return as_bool(value)
    except ValueError:
        return True


def _default_loadgen(base_url: str, model: str) -> LoadGenerator:
    """The real HTTP generator, imported lazily so this module stays importable without it.

    Lazy because ``infervolt.loadgen.http`` is a heavier import than an adapter that is
    only being asked to render a recipe should have to pay, and because the adapter has
    to construct on a controller that never runs a load point at all.
    """
    from infervolt.loadgen.http import HttpLoadGenerator

    # ignore_eos pins the output length to the workload OSL; vLLM accepts it as a
    # first-class field, so every request generates exactly max_tokens tokens.
    gen: LoadGenerator = HttpLoadGenerator(
        base_url=base_url, model=model, extra_body={"ignore_eos": True}
    )
    return gen
