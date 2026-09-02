# Contributing to infervolt

Thanks for helping. infervolt is pre-alpha; the fastest way to contribute is to run the mock loop,
file issues with reproductions, and send small PRs.

## Dev setup

```bash
git clone https://github.com/infervolt/infervolt && cd infervolt
uv sync --extra dev
uv run pre-commit install
uv run pytest -q
```

## Lint gate (CI runs exactly this)

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

## How to add things

- **An engine adapter:** implement `EngineAdapter` from `src/infervolt/engines/base.py`, register it under
  `[project.entry-points."infervolt.engines"]`, add fixtures for its `/metrics` output, and a knob space.
- **A diagnosis rule:** add a function in `src/infervolt/diagnose/rules.py` returning a `Finding`,
  register it in `RULES`, and add a mock scenario that triggers it.
- **A workload preset:** add it to `src/infervolt/workloads/presets.py` with a test.
- **A recipe:** run `infervolt optimize`, then open a PR adding `recipes/<engine>/<model>/<hardware>/`
  with `recipe.yaml`, `report.md`, and `trials.jsonl`. CI validates the YAML.

## Commit style

Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`). Every PR needs tests.
