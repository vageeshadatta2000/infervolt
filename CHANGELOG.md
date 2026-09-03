# Changelog

All notable changes are documented here. Format: Keep a Changelog. Versioning: SemVer.

## [Unreleased]

### Added
- Mock engine loop: baseline sweep, rule-based diagnosis, LLM ranking/planning, Optuna search, verify, recipe emission.
- CLI: `infervolt optimize`, `infervolt report`, `infervolt recipe validate`.
- Fake, Anthropic, OpenAI-compatible, and replay LLM clients.
- Mock engine run-to-run noise (`RUN_NOISE`, 0.5% on the load phase's wall clock): without it every
  repeat measures the same goodput to twelve digits, the paired CI collapses onto the mean, and
  `verify` accepts any config a hair above the baseline. The simulator exercises the statistics
  rather than defeating them, so the band sits under the effects the scenarios are built around.
- Minimum effect size in `verify` (`MIN_EFFECT_FRAC`, 1%): separation from zero is a statement about
  confidence, not size. With enough repeats a reproducible 0.1% clears the interval test and is
  still not worth rewriting a production config for, so a CI-separated win below the floor is
  reported as such and rejected.
- `--baseline` knob validation: unknown knobs and out-of-choices values fail the run before the
  baseline sweep instead of being silently measured as the default.
- Any LLM failure now degrades rather than ending a run: ranking falls back to rule order, planning
  to the diagnosis sub-spaces with no priors, and the narrative to a template.
- Recipes carry the diagnosis caveats, each arm's load point in the summary table, and count a
  timed-out trial as infeasible.

### Changed
- `VerifyResult` exposes `baseline_mean` and `comparable`, so callers word a win against a baseline
  that served nothing in absolute rps without re-deriving the condition.
- `infervolt optimize` rejects an unknown `--llm` and `--hardware auto` as usage errors; `infervolt
  report` looks the run up in the ledger and distinguishes an unknown run from a missing report.
