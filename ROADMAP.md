# Roadmap

- **M1 (now):** mock engine loop end to end, CPU-only CI. Interfaces are synchronous; artifacts are JSONL.
- **M2:** llama.cpp adapter (llama-bench stage 1, `llama-server --metrics` stage 2), multi-process HTTP load generator (async), Apple Silicon detection.
- **M3:** vLLM adapter on Thunder Compute (docker by digest, Prometheus name resolution, lm-eval quality guard, budget guard, teardown).
- **M4:** cross-run memory (meta-features, kNN warm start, insight notes), `--resume`, `watch` mode, Parquet + DuckDB `memory stats`.
- **M5:** docs site, committed recipes, PyPI release.
- **Later:** SGLang and TensorRT-LLM adapters, disaggregated prefill/decode and wide-EP recipes.

## Known follow-ups from the M1 review

- Do not bill statically-rejected trials against `max_trials`; unify with the novelty-skip path.
- Enforce `--max-usd` inside `run_search`, not only before verify (blocking for M3 real-GPU spend).
- Report budget exhaustion as its own outcome instead of "no feasible candidate".
- Split the recipe's `infeasible` count into OOM / rejected / crashed / timeout.
- One "usable trial" predicate shared by search and planner (`status == ok and result.feasible`).
- Drop or consume the unread canonical keys (`queue_time_p90_s`, `prefill_time_p50_s`, `prefix_hit_rate`).
- Canonical boolean flag rendering in `to_recipe_block` before the llama.cpp/vLLM adapters copy the pattern.
- Lock `trials.jsonl` appends before the async load generator lands.
- `extra="forbid"` on `Recipe`; keep `examples/recipe.yaml` in step with the schema.
- Dependabot and action version bumps in CI.
- Mock realism: per-run goodput noise is 0.5%; real GPUs show 1-3%.
