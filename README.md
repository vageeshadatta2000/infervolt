# infervolt

**Measure → diagnose the bottleneck → targeted fix → re-measure → verify → emit a recipe → remember.**

infervolt is an open-source agent that optimizes inference for open-source LLMs on *your* hardware and
explains *why* the winning configuration wins. It is the real-measurement, explanation-emitting,
engine-agnostic complement to tools that only model (NVIDIA aiconfigurator), only diagnose
(vLLM Doctor), or only search (auto-tuning-vllm).

Status: **pre-alpha**. The loop is complete and tested against a roofline-based mock engine.
llama.cpp and vLLM adapters are next (see ROADMAP.md).

## 60-second demo (no GPU, no API key)

```bash
uv sync --extra dev
uv run infervolt optimize --engine mock --model mock/qwen3-8b --hardware rtx4090-24 \
  --workload chat-4k-512 --slo ttft=600ms,itl=30ms --llm fake --max-trials 8
```

You get a `recipe.yaml` (engine args + before/after metrics + evidence + rationale) and a `report.md`.
Swap `--llm anthropic` (set `ANTHROPIC_API_KEY`) or `--llm openai` (any OpenAI-compatible endpoint,
including a local vLLM) for real reasoning.

## How it works

1. **Baseline sweep** at rising concurrency, recording client-side TTFT/ITL per token, engine counters, and GPU activity.
2. **Rules** (R0–R6) score bottleneck signatures from evidence: KV capacity, prefill compute, decode bandwidth, scheduler/CPU, communication, client artifact.
3. **LLM ranks and explains** the findings and picks the knob sub-space and a few prior candidates. It can only name knobs it was shown; every reply is schema-validated.
4. **Optuna TPE** searches inside that sub-space. OOMs are constraints that tighten bounds. Cheap stage-1 runs prune weak candidates before the full sweep.
5. **Verify**: 3 interleaved baseline/candidate repeats; a win needs a confidence interval on goodput that excludes zero, plus a quality guard when numerics change.
6. **Recipe + report** with provenance. Cross-run memory and continuous re-tuning arrive in M4.

## Compared to

| Tool | Runs real benchmarks | Attributes the bottleneck | Emits a recipe with evidence | Learns across runs |
|---|---|---|---|---|
| NVIDIA aiconfigurator | no (analytic) | no | manifests, no evidence | no |
| vLLM Doctor | reads metrics only | rules | no | local history only |
| auto-tuning-vllm / llm-optimizer | yes | no | config only | per-study |
| **infervolt** | yes | rules + LLM, evidence-backed | yes | M4 |

## Contributing

See CONTRIBUTING.md. Adapters, rules, workloads, and recipes are the four extension points.

## License

Apache-2.0.
