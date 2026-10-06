"""Small spawned-process checks for scheduling, isolation, and failure handling."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from grit.search.execution import ExecutionOptions, execution_scope, run_parallel


def _worker_factory() -> Callable[[int], tuple[int, int, int]]:
    def execute(value: int) -> tuple[int, int, int]:
        time.sleep(0.03 * (3 - value))
        return value, os.getpid(), torch.get_num_threads()

    return execute


@dataclass(frozen=True)
class _FailureFactory:
    root: Path

    def __call__(self) -> Callable[[int], int]:
        def execute(value: int) -> int:
            (self.root / f"{value}.started").touch()
            if value == 0:
                raise RuntimeError("intentional fit failure")
            deadline = time.monotonic() + 10
            while not (self.root / "0.started").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("first worker did not start")
                time.sleep(0.01)
            time.sleep(0.2)
            (self.root / f"{value}.finished").touch()
            return value

        return execute


def test_spawned_workers_preserve_order_and_restore_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    before = torch.get_num_threads()
    with execution_scope(tmp_path, ExecutionOptions(2)):
        results = run_parallel((0, 1, 2, 3), _worker_factory, ExecutionOptions(2))
        assert tuple(row[0] for row in results) == (0, 1, 2, 3)
        assert all(pid != os.getpid() and threads == 1 for _, pid, threads in results)
        assert os.environ["OMP_NUM_THREADS"] == "3"
        with pytest.raises(ValueError, match="already running"):
            with execution_scope(tmp_path, ExecutionOptions()):
                pytest.fail("second coordinator acquired the output lock")
    assert torch.get_num_threads() == before
    with execution_scope(tmp_path, ExecutionOptions()):
        pass


def test_worker_failure_stops_dispatch_and_drains_active_fits(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="intentional fit failure"):
        run_parallel(tuple(range(8)), _FailureFactory(tmp_path), ExecutionOptions(2))
    assert {p.stem for p in tmp_path.glob("*.started")} <= {"0", "1"}
    if (tmp_path / "1.started").exists():
        assert (tmp_path / "1.finished").exists()


@pytest.mark.parametrize("workers,threads", [(0, None), (-1, 1), (True, 1), (2, 0)])
def test_invalid_execution_options_are_rejected(
    workers: int, threads: int | None
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ExecutionOptions(workers, threads)
