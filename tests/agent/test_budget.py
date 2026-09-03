"""What the tracker calls "spent", branch by branch."""

from __future__ import annotations

from infervolt.agent.budget import BudgetTracker
from infervolt.core.types import Budget


def test_a_fresh_tracker_has_budget_left() -> None:
    t = BudgetTracker(Budget(max_trials=4, max_wall_s=3600.0, max_usd=10.0))
    assert t.exhausted() is None
    assert t.deadline == t.start + 3600.0


def test_wall_clock_exhaustion_wins_over_everything_else() -> None:
    t = BudgetTracker(Budget(max_trials=4, max_wall_s=1.0, max_usd=10.0))
    t.start -= 10.0  # a tracker that started ten seconds ago with a one-second budget
    assert t.exhausted() == "wall-clock budget exhausted"
    # Wall clock is gone whether or not trials remain, so the trial opt-out cannot mask it.
    assert t.exhausted(count_trials=False) == "wall-clock budget exhausted"


def test_cost_exhaustion_reports_what_was_spent() -> None:
    t = BudgetTracker(Budget(max_trials=4, max_usd=1.0))
    t.charge(0.75)
    assert t.exhausted() is None
    t.charge(0.75)
    assert t.exhausted() == "cost budget exhausted ($1.50)"
    assert t.exhausted(count_trials=False) == "cost budget exhausted ($1.50)"


def test_zero_max_usd_means_unlimited() -> None:
    """Also what an engine that reports no cost at all looks like."""
    t = BudgetTracker(Budget(max_trials=100, max_usd=0.0))
    t.charge(1_000_000.0)
    assert t.exhausted() is None


def test_trial_exhaustion_is_the_only_branch_the_opt_out_hides() -> None:
    t = BudgetTracker(Budget(max_trials=2, max_wall_s=3600.0))
    t.charge(0.0)
    assert t.exhausted() is None
    t.charge(0.0)
    assert t.exhausted() == "trial budget exhausted"
    # Spending every trial is what the search is for: it must not veto the verification
    # that search earned.
    assert t.exhausted(count_trials=False) is None
    assert t.trials == 2
