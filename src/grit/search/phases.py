"""The shared validation-only phase order, independent of dataset artifact types."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

RunT = TypeVar("RunT")
FinalistsT = TypeVar("FinalistsT")
WinnersT = TypeVar("WinnersT")
SummaryT = TypeVar("SummaryT")
StatusT = TypeVar("StatusT")


def run_selection_phases(
    *,
    tuning: Callable[[], tuple[RunT, ...] | None],
    finalists: Callable[[tuple[RunT, ...]], FinalistsT],
    confirmation: Callable[[FinalistsT], tuple[RunT, ...] | None] | None,
    winners: Callable[[FinalistsT, tuple[RunT, ...]], WinnersT],
    final: Callable[[WinnersT], tuple[RunT, ...] | None],
    summarize: Callable[[FinalistsT, WinnersT, tuple[RunT, ...]], SummaryT],
    status: Callable[[], StatusT],
    stop_after_tuning: bool = False,
) -> SummaryT | StatusT:
    """Incomplete stages stop advancement; final access follows winner freezing."""
    tuning_runs = tuning()
    if tuning_runs is None or stop_after_tuning:
        return status()
    selected = finalists(tuning_runs)
    confirmation_runs = () if confirmation is None else confirmation(selected)
    if confirmation_runs is None:
        return status()
    frozen = winners(selected, confirmation_runs)
    final_runs = final(frozen)
    if final_runs is None:
        return status()
    return summarize(selected, frozen, final_runs)
