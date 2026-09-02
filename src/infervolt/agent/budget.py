"""What the run is allowed to spend, and whether it has spent it.

The tracker is advisory rather than enforcing: it converts a :class:`Budget` into an
absolute deadline the search can stop before crossing, and answers "is there anything
left" for the caller's own checks. Nothing here cancels work already in flight.
"""

from __future__ import annotations

import time

from infervolt.core.types import Budget


class BudgetTracker:
    """Wall-clock, cost and trial counters for one run."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.start = time.time()
        self.spent_usd = 0.0
        self.trials = 0

    @property
    def deadline(self) -> float:
        """Absolute ``time.time()`` after which no new trial may start."""
        return self.start + self.budget.max_wall_s

    def charge(self, usd: float) -> None:
        self.spent_usd += usd
        self.trials += 1

    def exhausted(self, *, count_trials: bool = True) -> str | None:
        """Why the budget is spent, or ``None`` while it is not.

        ``count_trials=False`` asks only about the resources a run can still *waste*.
        Spending every trial is what the search is for, so a caller deciding whether the
        verification it already earned may go ahead asks without the trial counter --
        wall-clock and money are gone whether or not the work was worth it, but a search
        that used its whole trial budget is a search that finished.
        """
        if time.time() > self.deadline:
            return "wall-clock budget exhausted"
        # ``max_usd`` of 0 means unlimited, which is also what an untracked engine reports.
        if self.budget.max_usd and self.spent_usd > self.budget.max_usd:
            return f"cost budget exhausted (${self.spent_usd:.2f})"
        if count_trials and self.trials >= self.budget.max_trials:
            return "trial budget exhausted"
        return None
