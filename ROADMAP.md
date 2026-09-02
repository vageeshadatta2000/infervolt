# Roadmap

- **M1 (now):** mock engine loop end to end, CPU-only CI. Interfaces are synchronous; artifacts are JSONL.
- **M2:** llama.cpp adapter (llama-bench stage 1, `llama-server --metrics` stage 2), multi-process HTTP load generator (async), Apple Silicon detection.
- **M3:** vLLM adapter on Thunder Compute (docker by digest, Prometheus name resolution, lm-eval quality guard, budget guard, teardown).
- **M4:** cross-run memory (meta-features, kNN warm start, insight notes), `--resume`, `watch` mode, Parquet + DuckDB `memory stats`.
- **M5:** docs site, committed recipes, PyPI release.
- **Later:** SGLang and TensorRT-LLM adapters, disaggregated prefill/decode and wide-EP recipes.
