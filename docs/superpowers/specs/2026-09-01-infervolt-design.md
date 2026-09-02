# infervolt: design spec (self-learning inference-optimization agent)

> Status: research complete, all user decisions recorded, plan ready for approval.
> **Name: `infervolt`** (chosen 2026-09-01; verified free on PyPI, npm, GitHub handle/repos, and .com/.dev/.ai DNS). Repo: `~/infervolt`, package `infervolt`, CLI `infervolt`, env prefix `INFERVOLT_`. Register the PyPI name in M0 (two similar zero-star projects, `servetune` and `goodputlab`, appeared in the last weeks).

## Context

The user wants to build and publish on GitHub an AI agent that continuously learns how to run open-source LLMs faster, and that produces a **recipe**: what the bottleneck is, what change removes it, and the measured before/after. It must be a proper open-source project (repo hygiene, docs, CI, community recipes).

### User decisions (2026-09-01)
- **Engines v1:** vLLM first, llama.cpp second. SGLang / TensorRT-LLM later as adapters.
- **Agent brain:** Claude API as reference provider, behind a pluggable LLM-client interface (OpenAI-compatible endpoints incl. local models must work).
- **GPU:** Thunder Compute on demand via the already-authenticated `tnr` CLI. Local machine is an Apple M3, 8 GB RAM: fine for llama.cpp/MLX smoke tests and for writing the harness, not for vLLM.
- **Name/license:** `infervolt` (user picked from a vetted, fully-clear shortlist). License: Apache-2.0 (norm for vLLM/SGLang/lm-eval; patent grant).

### Local assets found
- `~/.unsloth/llama.cpp` full checkout, already built, with `llama-bench`, `batched-bench`, `quantize`.
- `~/.lmstudio` has MLX models (Qwen3-4B-MLX-4bit) and Metal llama.cpp backends.
- `~/.thunder` Thunder CLI configured (3 SSH keys). `cli_config.json` holds a plaintext token: never copy it into the repo.
- `uv` 0.9.2, Python 3.14 (bare) and Anaconda 3.11 with torch; Docker CLI present but daemon not running.
- No vLLM/SGLang/TRT-LLM anywhere locally: greenfield.

## Research findings (Sept 2026)

### Landscape: what exists, and the gap
| Category | Examples | What they lack |
|---|---|---|
| Load generators | GuideLLM (vllm-project, very active), NVIDIA AIPerf (genai-perf successor; has client-side Bayesian sweeps), inference-perf (k8s-sigs), `vllm bench` | Measure only. No server-config search, no diagnosis. |
| Blind auto-tuners | vLLM `benchmarks/auto_tune` (grid over 2 knobs; README says it does not explain why), BentoML llm-optimizer (dormant since 2025-09), openshift-psap/auto-tuning-vllm (Optuna+Ray+GuideLLM, alive), novitalabs/autotuner (15 stars), SLO-Guard (arXiv 2604.17627: crash-aware TPE, OOM as constraint), llm-tuna (WWW 2026) | Scalar-objective search; no bottleneck attribution; no cross-run memory; no recipe output; vLLM-only. |
| Offline estimators | NVIDIA aiconfigurator (arXiv 2601.06288; analytic per-op DB, 30 s search, emits Dynamo/llm-d manifests; admits optimism), DynoSim/Spica, LLM-Viewer & llm-analysis (stale since 2024) | Never run anything; no data for consumer GPUs / Apple. |
| Read-only diagnoser | vLLM Doctor (~9 Prometheus rules → "likely bottleneck + evidence + next checks"; 16 stars) | Never acts, never verifies. |
| Agentic/evolutionary | OpenEvolve, ShinkaEvolve, KernelAgent (Meta: NCU → roofline → LLM root-cause → fix → verify), VibeServe (UW, arXiv 2605.06068: agent writes whole serving systems with nsys feedback), Kernel Foundry (experience library, no code) | Kernel-level or whole-system codegen; none tune existing engines' configs end-to-end. |
| Recipes | vllm-project/recipes (1k stars, YAML per model, schema-validated, hand-written, no evidence attached), llm-d well-lit paths, vllm-skills (deploy/bench only) | No tool generates recipes with measurements + rationale. |

**Gap (nobody does):** measure → attribute bottleneck with evidence → targeted fix → re-measure → confirm → emit recipe with provenance, with memory that transfers across models/hardware/engine versions. Nearest neighbours to cite and complement: aiconfigurator (modeled), vLLM Doctor (read-only), auto-tuning-vllm (blind BO), VibeServe (codegen).

### Optimization levers the agent must know (with sources)
- Decode is HBM-bandwidth-bound (~0.9 FLOP/byte at batch 1); prefill is compute-bound. Roofline first (arXiv 2402.16363, 2512.01644).
- Primary vLLM search space: `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `max_model_len`, chunked prefill, prefix caching, CUDA graphs vs `enforce_eager`, quantization, KV dtype, TP/PP. Many configs OOM: crashes are constraints (SLO-Guard).
- Speculative decoding: EAGLE-3 (2503.01840), DFlash (2602.06036), native MTP heads (DeepSeek V4, Qwen3.6, GLM-5.x), ngram fallback. Largest single decode win; acceptance is workload-dependent, always measure.
- KV: FP8 KV (~54% bytes/token, break-even ≥4-7k ctx), LMCache, SGLang HiCache, Mooncake store; prefix cache hit-rate dominates agentic workloads.
- Weights: FP8 (Hopper+), W4A16 for memory-bound small batch, W8A8 for compute-bound prefill, NVFP4/MXFP4 on Blackwell (~1.6x over BF16). Quality guard via accuracy recovery ≥99% (FP8) / ≥97% (INT4) on a fixed lm-eval task set (Red Hat 500k-eval study).
- Disaggregated P/D (DistServe, Mooncake, Dynamo, llm-d), Wide-EP for MoE: out of scope for v1, but the recipe schema must be able to describe them.
- Kernels: FA3 (Hopper), FA4 (Blackwell, 2603.05451), FlashInfer, FlashMLA. torch.compile + piecewise CUDA graphs in vLLM V1.
- llama.cpp: `--fit` (memory only), `-ngl`, `-t`, batch/ubatch, KV cache type q8_0/q4_0, `--spec-type mtp`, Metal/CUDA backends; `llama-bench` is the measurement tool.

### Bottleneck signatures (diagnosis rules seed)
| Bottleneck | Evidence | Targeted fixes |
|---|---|---|
| Prefill/compute-bound | TTFT grows with ISL and concurrency; SM util high | chunked prefill budget, FP8/W8A8, disaggregate |
| Decode/bandwidth-bound | ITL flat until BW saturates; low SM util, high HBM BW | spec decoding, W4A16, FP8 KV, larger batch |
| KV-capacity-bound | preemptions/recomputes, `max_num_seqs` never reached, low cache hit | FP8 KV, offload, prefix cache, lower `max_model_len`, higher `gpu_memory_utilization` |
| Scheduler/CPU-bound | GPU idle gaps between steps in nsys; ITL constant floor | CUDA graphs, async scheduling, fewer small requests |
| Communication-bound | all-reduce/all-to-all dominate profile at TP>1 | reduce TP, PP, DeepEP, overlap |
| Client-bound (benchmark artifact) | single-process asyncio load gen inflates TTFT (arXiv 2605.24217) | multi-process load generator |

### Measurement standards
- Metrics: TTFT, TPOT/ITL, E2E, throughput, **goodput under SLO** (the objective), tok/s/GPU, $/M tokens.
- Tools: GuideLLM (sweep modes, warmup/cooldown, over-saturation detection, JSON out), `vllm bench serve`, AIPerf, lm-evaluation-harness for quality.
- Reproducibility: warmup, repeats with mean ± SD, image pinned by digest, engine version + commit recorded, GPUs not bit-reproducible.
- Reference leaderboard to cite: InferenceX (SemiAnalysis), nightly, fixed configs.

### Agent-loop design patterns (from AlphaEvolve/OpenEvolve/ShinkaEvolve/AI-Scientist-v2/Voyager/LLAMBO)
- Single record type `Candidate → Result(metrics, artifacts, status)`; cheap-to-expensive evaluation cascade; novelty filter before GPU spend; LLM-maintained insight scratchpad injected into prompts.
- A 2026 study (arXiv 2603.24647) found TPE/CMA-ES beat pure LLM agents inside a fixed search space; the hybrid ("Centaur": share optimizer state with the LLM) won. Design rule: **LLM diagnoses, picks the sub-space, seeds priors, and explains; a classical optimizer (Optuna TPE + ASHA pruning) searches inside it.**
- Warm-start via meta-features (model family, params, MoE?, GPU SKU/HBM, ISL/OSL, rate) → nearest prior runs (Feurer 2015; Perrone NeurIPS 2019 learned search spaces).
- Crash-safe: subprocess per trial, container pinned by digest, hard wall-clock timeout, kill process group, `NCCL_ASYNC_ERROR_HANDLING=1`, classify OOM by exit code + log regex, record crash as constraint violation.

### OSS project practices
- `src/` layout, `pyproject.toml`, uv + lock, ruff, mypy, pytest, pre-commit; adapters via entry-points; pydantic-settings config; SQLite ledger + Parquet artifacts (+ DuckDB for analytics); mkdocs-material; GitHub-hosted CI for unit + mock adapters + CPU smoke (tiny model), GPU CI as scheduled/self-hosted; Trusted Publishing to PyPI; CONTRIBUTING/CODE_OF_CONDUCT/SECURITY/CODEOWNERS; `recipes/` as community cookbook in the `vllm-project/recipes` YAML schema plus extensions.
- Emulate: guidellm (SLO profiles, outputs), optimum-benchmark (backend/launcher/scenario split), lm-eval (task registry), DSPy (docs-first), OpenEvolve (config-driven, checkpoints).

## Environment findings that affect milestones
- `~/.unsloth/llama.cpp/build/bin` has only `llama-server` and `llama-quantize`; `llama-bench` is not built. M2 starts with `cmake --build build --target llama-bench llama-batched-bench`. `llama-server --metrics` exposes Prometheus, so llama.cpp reuses the same server + loadgen + scrape path as vLLM.
- `tnr` is v2.0.65; minimum required is v2.0.71 and auto-update needs sudo. **User must reinstall `tnr` before M3.** Calls the provisioner wraps: `tnr create --gpu {a100,h100} --mode development --num-gpus N --json -y`, `tnr connect <id> -t 8000`, `tnr scp`, `tnr delete`.
- No runnable GGUF locally (only vocab files). M2 downloads Qwen3-1.7B Q8_0; CI uses a ~1 MB stories-class GGUF.

## Positioning (one paragraph for the README)
"The real-measurement, explanation-emitting, engine-agnostic complement to NVIDIA aiconfigurator (modeled, never runs), vLLM Doctor (diagnoses, never acts), and auto-tuning-vllm (searches, never explains)." Core loop: **measure → attribute bottleneck with evidence → targeted fix → re-measure → verify → emit recipe with provenance → remember.**

## Architecture

### Design rules
1. Deterministic rules produce the evidence; the LLM ranks, explains, chooses the sub-space, seeds priors, and writes the narrative. It never emits CLI flags, only knob names from the `KnobSpace` it was shown. All LLM outputs are pydantic-validated.
2. A classical optimizer (Optuna TPE, `multivariate=True`, ASHA pruning) searches inside the LLM-chosen sub-space (the "Centaur" hybrid, arXiv 2603.24647).
3. Crashes are constraints, not errors (SLO-Guard): OOM tightens bounds and is recorded on an "OOM frontier".
4. Every state transition is checkpointed in SQLite; `optimize --resume <run_id>` continues after the agent itself dies.
5. Goodput under SLO is the objective; a config only "wins" if the interleaved 3x re-run shows a CI-separated improvement.

### Package layout (`src/infervolt/`)
| Module | Responsibility |
|---|---|
| `cli/` | Typer: `optimize`, `watch`, `report`, `recipe validate`, `memory stats`, `hw detect` |
| `config.py` | pydantic-settings, `infervolt_` env prefix: API keys, budgets, paths, remote defaults |
| `core/types.py` | All shared pydantic models (below). Single import point. |
| `engines/base.py`, `engines/{mock,llamacpp,vllm}/`, `engines/registry.py` | `EngineAdapter` protocol; discovery via `[project.entry-points."infervolt.engines"]` |
| `hardware/` | Detect (NVML / `sysctl`+Metal), roofline table per SKU (peak TFLOPs, HBM GB/s, mem GB) → `HardwareProfile` |
| `workloads/` | Presets `chat-4k-512`, `rag-8k-256`, `agentic-prefix-16k-512`; token-length distributions; prompt synthesis with controllable shared prefix |
| `loadgen/` | `LoadGenerator` protocol; `builtin` (multi-process, one asyncio loop per process, per-token timestamps, client-health counters); `guidellm` adapter; percentile/goodput analysis |
| `metrics/` | Prometheus text scraper (1 s poll), GPU sampler (`nvidia-smi dmon` / DCGM), evidence assembly into `Observation` |
| `diagnose/rules.py`, `diagnose/ranker.py` | Deterministic rules R0–R6 → `Finding`s; LLM ranks/explains → `Diagnosis` |
| `search/` | Knob-space DSL, Optuna study factory (TPE + ASHA), constraint handling, novelty filter, bound tightening on crash |
| `runner/` | One `Candidate`: launch → warm-up → stage-1 short load → prune? → stage-2 full load → scrape → stop. Time-boxed, process-group kill, crash classification |
| `verify/` | SLO goodput, repeat policy (n=3 interleaved, CI test vs baseline), quality guard (lm-eval subprocess) |
| `store/` | SQLite ledger (runs, trials, evidence, notes, watch_state) + Parquet artifacts; DuckDB views |
| `memory/` | Meta-features, kNN warm start, insight notes, learning-curve metrics |
| `recipes/` | Recipe schema, YAML emitter, Markdown report (Jinja2), JSON-schema export for CI validation |
| `llm/` | `LLMClient` protocol; `anthropic.py` (tool-use forced JSON), `openai_compat.py` (json_schema response_format, prompt+repair fallback), `replay.py` (cassettes for CI), `prompts/` (versioned Jinja2) |
| `agent/planner.py`, `agent/budget.py` | Explicit state machine PREPARE→BASELINE→DIAGNOSE→PLAN→SEARCH→VERIFY→EMIT→LEARN; wall-clock / $ / trial budgets |
| `remote/thunder.py`, `remote/ssh.py` | `tnr` wrapper (JSON mode only, timeouts, retries), SSH tunnel to 8000, `docker run` by image digest, log tail |
| `continuous/` | Watchers (vLLM releases/GHCR tags, llama.cpp tags, HF model list, workload drift), trigger → warm-started `optimize`, recipe versioning + diff, optional PR opening |

### Key interfaces (minimal)
```python
class EngineAdapter(Protocol):
    name: str
    def version(self) -> EngineVersion                          # name, version, commit, image_digest
    def knob_space(self, hw: HardwareProfile, model: ModelInfo) -> KnobSpace
    def validate(self, cfg: EngineConfig, hw, model) -> list[str]   # static rejections, never launched
    async def launch(self, cfg: EngineConfig, ctx: LaunchContext) -> ServerHandle
    async def ready(self, h: ServerHandle, timeout_s: float) -> bool
    async def scrape(self, h: ServerHandle) -> dict[str, float]
    async def stop(self, h: ServerHandle) -> ExitInfo
    def classify_crash(self, exit: ExitInfo, log_tail: str) -> CrashKind   # OOM|STARTUP|RUNTIME|TIMEOUT|NONE
    def to_recipe_block(self, cfg: EngineConfig) -> dict

class LoadGenerator(Protocol):
    async def run(self, url, workload: Workload, *, duration_s, workers) -> LoadResult  # per-request records + client_health

class LLMClient(Protocol):
    model_id: str
    async def structured(self, *, system, user, schema: type[T], max_tokens=4096) -> T
    async def text(self, *, system, user, max_tokens) -> str

class Store(Protocol):   save_run/save_trial/save_observation/save_note; trials(run_id); runs(filter); artifact_path(run_id, name)
class Memory(Protocol):  features(spec, hw, model) -> MetaFeatures; nearest(f, k, engine) -> list[PriorRun]; notes(f, tags, k) -> list[Insight]; record_learning(run_id, curve)
```
Data models in `core/types.py`: `Workload{name, isl: TokenDist, osl: TokenDist, load: LoadSpec, prefix_share, num_requests, seed}`, `SLO{ttft_ms, itl_ms, e2e_ms, percentile=0.9}`, `Metrics` (ttft/itl/e2e p50/p90/p99, output_tps, req_per_s, goodput_rps, goodput_frac, error_rate, tokens_per_s_per_gpu, usd_per_m_tokens), `Evidence{source, key, value, unit, window, note}`, `Observation{metrics, evidence, series, config, load_point}`, `Finding{rule_id, bottleneck, score, evidence, fixes}`, `Diagnosis{primary, ranked, rationale, confidence, subspaces, caveats}`, `Candidate{id, config, parent_id, origin: baseline|warm_start|llm_prior|tpe|manual, hypothesis}`, `Trial{id, run_id, candidate, status: pending|running|ok|infeasible_oom|crash|timeout|pruned|rejected, result, cost_usd}`, `Result{observations, objective, feasible, slo_violations, repeats, quality, artifacts}`.

### Data flow of one `optimize` run
| Step | Component | What happens | LLM call |
|---|---|---|---|
| 0 Prepare | cli → planner | Parse spec; detect hardware (or `tnr create` + probe); `ModelInfo` from HF `config.json`; pick adapter; `memory.features()` + `memory.nearest(k=5)`; open run | – |
| 1 Baseline | runner | Engine defaults (or upstream recipe if one exists) → warm-up 32 req → concurrency sweep {1,4,16,64,…} until goodput drops → `Observation` per load point with Prometheus + GPU series; client-artifact check (R6) | – |
| 2 Diagnose | rules | Baseline observations + roofline expectations → scored `Finding`s (score <0.3 dropped) | – |
| 3 Rank | ranker | Given findings, per-load-point metrics, roofline numbers, workload, SLO, static config → `Diagnosis`. Must cite only rule ids it was given; else reject, retry once, then top rule score wins | **1** |
| 4 Plan | agent | Given `Diagnosis`, `KnobSpace`, k nearest prior runs, insight notes, budget → `SearchPlan{subspace, fixed, priors(≤4, each with hypothesis), stage1_duration_s, max_trials, stop_rule}`. Priors clamped via `validate()`; unknown knobs dropped + logged | **2** |
| 5 Search | search + runner | Optuna study over `subspace`; enqueue warm-starts then LLM priors; novelty filter (L∞ <0.05 normalized → skip). Per trial: stage-1 60 s at baseline's best load point → ASHA prune if goodput < median → stage-2 full sweep 180 s. Crash → infeasible + bound tightening. Save every trial before the next | – |
| 6 Replan | agent | At most once, on plateau (5 trials) or 3 consecutive infeasible: trial table summary → updated `SearchPlan` or stop | **3** |
| 7 Verify | verify | Best vs baseline, 3x interleaved (B,C,B,C,B,C) at SLO load point; accept only if CI of goodput delta excludes 0. Quant/KV-dtype/spec-decode changes → quality guard: lm-eval (`gsm8k`, `arc_challenge`, mmlu subset) recovery ≥99% (fp8) / ≥97% (int4); spec-decode → greedy-equivalence on 20 prompts | – |
| 8 Emit | recipes | Given before/after metrics, diagnosis, winning knob deltas, evidence → `rationale` (≤200 words), `next_steps`, ≤5 insight notes citing trial ids. Writes `recipe.yaml`, `report.md`, Parquet | **4** |
| 9 Learn | memory | Store meta-features, best config, `trials_to_target`, notes; teardown (`tnr delete`) if `--remote-teardown` | – |

### Diagnosis rules v1 (thresholds config-tunable; metric names marked (?) resolved by regex against live `/metrics` at startup and recorded)
| Rule | Bottleneck | Evidence | Sub-space |
|---|---|---|---|
| R0 | Under-loaded | `vllm:num_requests_waiting`≈0, running ≪ `max_num_seqs`, SM active <30%, SLOs met | extend sweep; no tuning |
| R1 | KV-capacity | rate(`vllm:num_preemptions_total`)>0; or p95(`vllm:kv_cache_usage_perc`)>0.90 with waiting>0 while running never reaches `max_num_seqs`; p90(`vllm:request_queue_time_seconds`) > 0.5×TTFT SLO; low prefix hit rate (`vllm:prefix_cache_hits`/`vllm:prefix_cache_queries` (?)) when prefix_share>0.3; roofline: KV capacity from `vllm:cache_config_info` vs concurrency×(ISL+OSL) | `kv`: gpu_memory_utilization, max_model_len, kv_cache_dtype, enable_prefix_caching, max_num_seqs |
| R2 | Decode-bandwidth | p50 ITL within ±30% of floor `(weight_bytes + batch×kv_bytes/token×ctx)/HBM_BW`; DRAM active >60% and SM active <50% (DCGM `DCGM_FI_PROF_DRAM_ACTIVE`/`SM_ACTIVE`, fallback `nvidia-smi dmon`); ITL sub-linear in concurrency | `decode`: speculative_config, quantization, kv_cache_dtype, max_num_seqs↑ |
| R3 | Prefill-compute | p90 TTFT ≥ linear in concurrency; p50 prefill time ≥ 0.5× inference time (`vllm:request_prefill_time_seconds`, `vllm:request_inference_time_seconds`); ITL p99/p50 > 3; tensor-pipe active >70% | `prefill`: max_num_batched_tokens, long_prefill_token_threshold, max_num_partial_prefills, quantization |
| R4 | Scheduler/CPU | ITL(c=1) ≈ ITL(c=8) within 15% with SM<40% and DRAM<40%; step time > 2× roofline at batch 1; `enforce_eager=True` (static); optional torch-profiler trace with inter-kernel gaps >30% | `sched`: enforce_eager, cudagraph mode, async_scheduling, max_num_seqs |
| R5 | Communication | TP>1 and PCIe-only topology (`nvidia-smi topo -m`) or NVLink/PCIe TX high; step ≫ roofline/TP; low-confidence unless profiler evidence | `parallel`: tensor_parallel_size, pipeline_parallel_size, disable_custom_all_reduce |
| R6 | Client artifact (invalidates) | loadgen worker CPU >80%; loop lag p99 >5 ms; client TTFT p50 − server TTFT p50 > 20%; error rate >1% | re-run with more workers; observation marked invalid |

llama.cpp mapping: `llama-bench -o json` fields (`avg_ts`, `stddev_ts`, `n_prompt`, `n_gen`, `n_batch`, `n_ubatch`, `n_threads`, `type_k/v`, `n_gpu_layers`, `flash_attn`, `model_size`, `build_commit`) and `llama-server --metrics` (`llamacpp:prompt_tokens_seconds`, `llamacpp:predicted_tokens_seconds`, `llamacpp:kv_cache_usage_ratio`, `llamacpp:requests_processing`, `llamacpp:requests_deferred` (?)). R1 ← kv usage>0.9 or deferred>0; R2 ← tg t/s within 30% of `model_size/mem_bw`; R3 ← pp t/s far below compute roofline; R4 ← tg not scaling with `-t`.

### Search spaces
**vLLM** (adapter owns knob→flag mapping): `max_num_seqs` int log 8..1024 (≤ KV capacity/(ISL+OSL)×1.5); `max_num_batched_tokens` int log 512..16384 (≥ max_num_seqs); `gpu_memory_utilization` 0.70..0.95 step 0.05 (tightened on OOM); `max_model_len` cat {p99 ISL+OSL, 8k, 16k, 32k, model max}; `enable_prefix_caching` (forced on if prefix_share>0.3); `enable_chunked_prefill`; `kv_cache_dtype` {auto, fp8} (SM≥8.9; quality guard); `enforce_eager`; `cudagraph_mode` via `--compilation-config` (enum names per pinned version); `async_scheduling` (version-gated vs spec-decode); `speculative` {none, ngram(k,n), eagle3 if draft known, mtp if model has heads} (greedy-equivalence guard); `quantization` {none, fp8 dynamic, pre-quantized alias} (quality guard); `tensor_parallel_size` divisors(num_gpus)∩divisors(num_kv_heads). Phase 2: attention backend, block_size, LMCache.

**llama.cpp** (M3: model+KV ≤ 5.5 GB enforced statically): `-t` 1..8; `-b` {256..2048}; `-ub` {64..512} ≤ b; `-ctk/-ctv` {f16,q8_0,q4_0} (q4_0 V needs `-fa on`); `-fa`; `-ngl` {0, all} on Metal / 0..N CUDA; `-c`; server `-np` 1..8, `--cache-reuse` {0,256}, `--spec-type` {none, ngram-simple, ngram-map-k, draft-mtp} with `--spec-draft-n-max` 2..8; `--fit`; `--mlock`.

### Crash handling (runner policy + adapter regexes)
| Event | Detection | Action |
|---|---|---|
| OOM at startup | exit≠0 + regex (`CUDA out of memory`, `OutOfMemoryError`, `No available memory for the cache blocks`, `larger than the maximum number of tokens that can be stored in KV cache`; llama.cpp `failed to allocate`, `ggml_metal.*alloc`, `kv_cache_init: failed`, SIGKILL) | `infeasible_oom`; objective = worst; tighten `gpu_memory_utilization.max = v−0.05`, `max_model_len.max = v`, `max_num_seqs.max = v`; record OOM frontier |
| OOM under load | server dies / 5xx spike | same + mark load point; continue |
| Startup timeout | no `/health` 200 within 15 min | `timeout`; retry once only if log shows weight download |
| Runtime crash | exit≠0 other | `crash`; 3 consecutive → abort run, log tail explained in report |
| Static rejection | `validate()` non-empty | `rejected`, never launched, not budgeted |
| Dirty GPU | after stop, nvidia-smi used >1 GB | killpg, `docker rm -f`, wait 30 s, re-check; abort if still dirty |
| Hard time-box | trial > stage budget ×1.5 | kill, `timeout` |
Isolation: one subprocess per trial (local) or `docker run --rm --gpus all --ipc=host <image@sha256>` (remote); `NCCL_ASYNC_ERROR_HANDLING=1`; `start_new_session=True`; `os.killpg` on stop.

### Recipe YAML (top-level keys mirror `vllm-project/recipes`; confirm exact upstream key names against its schema in M0 and lock in `recipes/schema.py`)
```yaml
schema_version: 1
model: {id: Qwen/Qwen3-8B, revision: <sha>, params_b: 8.2, arch: qwen3, moe: false}
hardware: {gpu: NVIDIA A100-SXM4-80GB, count: 1, driver: "580.xx", topology: single, provider: thunder-compute}
engine: {name: vllm, version: 0.11.x, image: vllm/vllm-openai@sha256:..., commit: <sha>}
workload: {name: chat-4k-512, isl: {p50: 4096, p99: 6000}, osl: {p50: 512}, prefix_share: 0.1, load: {mode: sweep, concurrency: [1,4,16,64]}}
slo: {ttft_ms: 500, itl_ms: 30, percentile: 0.9}
serve:
  args: {max-num-seqs: 128, max-num-batched-tokens: 4096, gpu-memory-utilization: 0.90, kv-cache-dtype: fp8, enable-prefix-caching: true}
  env: {}
  command: "vllm serve Qwen/Qwen3-8B --max-num-seqs 128 ..."
baseline: {serve_args: {...}, metrics: {goodput_rps: 3.1, ttft_p90_ms: 812, itl_p90_ms: 27.4, output_tps: 1420}}
result:   {metrics: {goodput_rps: 5.6, ttft_p90_ms: 430, itl_p90_ms: 24.9, output_tps: 2210}, repeats: 3,
           improvement: {goodput_rps: "+81% (95% CI +62..+97)"}, quality: {guard: lm-eval, tasks: [gsm8k, arc_challenge], recovery: 0.996}}
infervolt:
  run_id: 2026-09-03T14-02-11Z-7f3a
  diagnosis: {primary: kv_capacity, confidence: 0.82, findings: [{rule: R1, score: 0.9, evidence: [{key: "vllm:num_preemptions_total/s", value: 2.3}, {key: "kv_cache_usage_perc.p95", value: 0.97}]}]}
  rationale: "Preemptions at c=16 show KV exhaustion at default 0.9 util with fp16 KV; fp8 KV doubles token capacity ..."
  search: {trials: 14, infeasible: 3, subspace: [gpu_memory_utilization, kv_cache_dtype, max_num_seqs, max_model_len], optimizer: optuna-tpe, seed: 7}
  trials_to_target: 6
  warm_start: {from_runs: [...], notes_used: [...]}
  next_steps: ["Try EAGLE-3 draft: decode is now bandwidth-bound (R2 score rose to 0.7 after fix)"]
  artifacts: {report: report.md, trials: trials.parquet, requests: requests.parquet}
  provenance: {tool_version: 0.1.0, llm: claude-..., prompts_sha: ..., created: 2026-09-03}
```
`report.md`: summary table, diagnosis with evidence table, before/after chart (PNG), trial table, reproduction command, caveats.

### Memory and measurable learning
- Per trial (SQLite + Parquet): run_id, trial_id, engine+version, meta-feature vector, full config, status, objective, all `Metrics`, key evidence, stage reached, duration, cost. Per run: spec, best, baseline, diagnosis, learning curve, recipe path.
- Meta-features (standardized): log10 params, log10 active params, is_moe, num_layers, KV bytes/token, weight_bits, hbm_gb, hbm_bw, peak_tflops, num_gpus, sm_major, log ISL p50, log OSL p50, prefix_share, log target concurrency, slo_ttft, slo_itl, engine major.minor.
- Warm start: `nearest(k=5)` same engine; each neighbour's best config (clamped via `validate`) enqueued as an Optuna trial; its bottleneck label is a prior for the ranker. Far neighbours contribute notes only.
- Insight notes: ≤5 per run, `{text, tags, cites: [trial_ids], confidence}`; retrieval by tag overlap + kNN; auto-demoted when a later near-identical trial contradicts the cited direction.
- **Learning metric:** `trials_to_target` = trials until objective ≥ 0.95× final best (or SLO first met). `memory stats` prints it per (engine, model family, GPU) over time. **Test (mock engine):** 12 sequential runs over 4 synthetic hardware × 3 models, memory on vs off, seed-fixed; assert median `trials_to_target` with memory ≤ 0.7× without. Same figure goes in the README.

## Milestones
| Milestone | Scope | Verification | Size |
|---|---|---|---|
| **M0 Scaffold** | pyproject (uv, ruff, mypy strict, pytest), `src/` layout, `core/types.py`, CLI skeleton, config, store (SQLite+Parquet), recipe schema + validator, LICENSE/README/CONTRIBUTING/CoC/SECURITY, CI (lint+type+unit), pre-commit, `.gitleaks` hook | `uv run infervolt recipe validate examples/recipe.yaml` passes; CI green on a PR | 2–3 days |
| **M1 Mock loop end-to-end (first demo)** | Mock adapter with roofline latency model (step = max(compute, memory) + sched overhead; KV capacity → preemption/queueing; OOM when weights+min KV > mem×util; spec-decode divides ITL by (1+acceptance); prefix cache scales prefill; seeded 3% noise; fake `/metrics`); builtin loadgen; rules R0–R6; Optuna search with crash tightening; verify; recipe+report; `LLMClient` with replay cassettes + rule-based fake for CI; Anthropic + OpenAI-compatible clients | `optimize --engine mock --hardware a100-80 --model mock/qwen3-8b --workload chat-4k-512 --slo ttft=500ms,itl=30ms` finishes <60 s on a GitHub runner; primary diagnosis matches the injected bottleneck in each of 4 mock scenarios (parametrized test); recipe validates; report renders | 1.5–2 weeks |
| **M2 llama.cpp local** | Build `llama-bench`; adapter (llama-bench stage 1, `llama-server --metrics` stage 2 with same loadgen); Metal detect; llama.cpp knob space; CPU smoke test in CI with ~1 MB GGUF; local run on Qwen3-1.7B Q8_0 | CI CPU smoke passes on ubuntu + macos runners; local M3 run yields a recipe where `-ctk`/`-fa`/`-t` change tg t/s beyond noise (3 repeats) | 1 week |
| **M3 vLLM on Thunder** | Thunder provisioner (`tnr` JSON mode; needs `tnr` ≥ 2.0.71), SSH tunnel, docker-by-digest launcher, vLLM adapter, metric-name resolution, GPU sampler, quality guard (lm-eval), budget guard ($ + wall-clock), teardown in `finally` + `atexit`; `gpu.yml` workflow_dispatch + weekly cron | One full run on A100 for Qwen3-8B with the CLI command above; CI-separated improvement over vLLM defaults; instance deleted at end (asserted via `tnr status --json`); cost logged in recipe | 2 weeks |
| **M4 Memory + continuous** | Meta-features, kNN warm start, insight notes, learning metrics + mock learning test; `watch` with release/model/drift watchers; recipe versioning + diff; GitHub Actions cron template; optional PR opening | Mock learning test passes (≥30% fewer trials); `watch --once` detects a fixture "new vLLM tag" and triggers a warm-started run whose `trials_to_target` < the cold run's | 1.5 weeks |
| **M5 Docs + launch** | mkdocs-material (concepts, quickstarts per engine, recipe schema, adding an adapter / rule), 3 committed recipes with reports, README with learning-curve figure + 60 s asciinema, PyPI trusted publishing, v0.1.0 | Fresh clone: mock quickstart works as documented; `pip install infervolt` works; docs deploy green | 1 week |

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Reward hacking / benchmark artifacts (Sakana CUDA Engineer lesson) | Goodput from client-observed per-token timestamps with fixed `max_tokens` + `ignore_eos`; R6 invalidates client-bound runs; 3x interleaved verify with CI-separated delta; output token count must match OSL within 5%; spec-decode requires greedy equivalence |
| GPU cost blowout | `budget.py` hard caps (wall-clock, trials, $ = price × elapsed); ASHA prune at 60 s; novelty filter; teardown in `finally`/`atexit`; `watch` daily cap; runs resumable |
| Flaky measurements | Warm-up, fixed seeds/prompts, repeats mean ± SD, GPU clocks/temp recorded, cool-down between trials, image by digest, engine commit in every record; noise floor from baseline repeats = minimum detectable improvement |
| LLM hallucinated flags | LLM chooses only from shown `KnobSpace`; pydantic validation; unknown knobs dropped + logged; `validate()` before any launch; rules provide evidence and LLM must cite rule ids |
| Metric-name drift across vLLM versions | Regex resolution at startup, mapping recorded; `/metrics` fixtures per supported version in tests |
| Thunder / `tnr` flakiness | JSON mode only, timeouts + retries; provisioning failure is a run-level error with a clear message |
| Quality regressions | Mandatory quality guard for quant/KV-dtype/spec knobs; recovery recorded in recipe |
| Scope creep (P/D disaggregation, Wide-EP, SGLang, TRT-LLM) | Schema can describe them (`topology`, `serve.roles`); adapters not implemented in v1; listed in ROADMAP.md |

## Repo hygiene for launch
Apache-2.0 `LICENSE` + `NOTICE`; README (problem, 60 s demo, learning-curve figure, comparison table vs aiconfigurator / vLLM Doctor / auto-tuning-vllm, "what it does not do"); CONTRIBUTING (add adapter / rule / workload / recipe), CODE_OF_CONDUCT, SECURITY, CODEOWNERS, issue/PR templates, CHANGELOG; `uv.lock` committed; ruff + mypy strict + pytest coverage gate + pre-commit + gitleaks; CI `ci.yml` / `gpu.yml` (teardown job `if: always()`) / `docs.yml` / `release.yml` (Trusted Publishing on `v*`); `.env.example`; `recipes/` cookbook with JSON-schema validation and `recipes/README.md`; docs site; SemVer; prompts versioned and hashed into provenance; Discussions on; labels `good first issue`, `adapter`, `rule`, `recipe`; `ROADMAP.md`; citations to the papers above.

## Critical files (first to write)
- `src/infervolt/core/types.py`: every shared model; everything depends on it.
- `src/infervolt/engines/base.py`: `EngineAdapter` protocol + `KnobSpace` DSL; the contract all adapters implement.
- `src/infervolt/agent/planner.py`: checkpointed state machine and the four LLM call sites.
- `src/infervolt/diagnose/rules.py`: R0–R6; what makes this an attributor rather than a blind tuner.
- `src/infervolt/engines/mock/model.py`: roofline-based synthetic engine; makes the loop CI-testable and is the M1 demo.

## Verification (end-to-end)
1. M1: `uv run pytest -m integration` runs the mock loop for 4 injected bottlenecks and asserts diagnosis + recipe validity; CLI demo <60 s.
2. M2: `uv run infervolt optimize --engine llamacpp --model Qwen/Qwen3-1.7B-GGUF:Q8_0 --workload chat-1k-128` on the M3 produces `recipe.yaml` + `report.md` with a beyond-noise win.
3. M3: full Thunder A100 run for Qwen3-8B; check recipe `result.improvement` CI excludes 0, `quality.recovery` present if quant knobs changed, and `tnr status --json` shows no live instance afterwards.
4. M4: `uv run pytest -m learning` shows median `trials_to_target` drop ≥30% with memory; `watch --once` fixture test.
5. M5: fresh-clone quickstart + `pip install` smoke on a clean venv.
