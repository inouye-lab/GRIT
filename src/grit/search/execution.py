"""Bounded, optional process execution; no dataset or selection semantics."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import signal
import sys
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import torch

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ExecutionOptions:
    workers: int = 1
    threads_per_worker: int | None = None

    def __post_init__(self) -> None:
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        for name, value in (
            ("workers", self.workers),
            ("threads_per_worker", self.threads_per_worker),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")

    @property
    def effective_threads(self) -> int | None:
        if self.threads_per_worker is not None:
            return self.threads_per_worker
        return 1 if self.workers > 1 else None


SEQUENTIAL_EXECUTION = ExecutionOptions()
_options = SEQUENTIAL_EXECUTION
_execute: Callable[[object], object] | None = None


def current_execution_options() -> ExecutionOptions:
    return _options


@contextmanager
def output_lock(path: Path, *, blocking: bool = False) -> Generator[None, None, None]:
    """Keep the lock inode: unlinking it could admit a second coordinator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(stream, flags)
        except BlockingIOError as error:
            raise ValueError(
                f"this search is already running: {path.parent}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def execution_scope(
    root: Path, options: ExecutionOptions
) -> Generator[None, None, None]:
    global _options
    with output_lock(root / ".run.lock"):
        previous, old_threads = _options, torch.get_num_threads()
        _options = options
        try:
            if options.effective_threads is not None:
                torch.set_num_threads(options.effective_threads)
            yield
        finally:
            _options = previous
            if torch.get_num_threads() != old_threads:
                torch.set_num_threads(old_threads)


def execution_metadata() -> dict[str, int | str]:
    return {
        "workers": _options.workers,
        "threads_per_worker": torch.get_num_threads(),
        "start_method": "spawn" if _options.workers > 1 else "direct",
    }


def record_execution(root: Path) -> None:
    """Operational provenance, separate from candidate/config identity."""
    temporary = root / "execution.tmp"
    temporary.write_text(json.dumps(execution_metadata(), sort_keys=True) + "\n")
    temporary.replace(root / "execution.json")


def _initialize(factory: object, options: ExecutionOptions) -> None:
    global _execute, _options
    # The coordinator drains active work on Ctrl-C; workers finish their own files.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _options = options
    if options.effective_threads is not None:
        torch.set_num_threads(options.effective_threads)
    _execute = cast(Callable[[], Callable[[object], object]], factory)()


def _dispatch(task: object) -> object:
    if _execute is None:
        raise RuntimeError("worker was not initialized")
    return _execute(task)


def run_parallel(
    tasks: Sequence[TaskT],
    factory: Callable[[], Callable[[TaskT], ResultT]],
    options: ExecutionOptions,
) -> tuple[ResultT, ...]:
    """At most workers tasks in flight; return canonical, not completion, order.

    Factories are importable functions or picklable objects. They may construct
    closures inside the child, so feature tensors and closures never cross IPC.
    """
    if not tasks:
        return ()
    if options.workers < 2:
        raise ValueError("parallel dispatch requires at least two workers")
    # Set before spawn imports torch/BLAS, then restore the caller's environment.
    previous = {
        key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    }
    for key in previous:
        os.environ[key] = str(options.effective_threads)
    pool: ProcessPoolExecutor | None = None
    pending: dict[Future[object], int] = {}
    results: dict[int, ResultT] = {}
    indexed = iter(enumerate(tasks))

    def submit_one() -> None:
        assert pool is not None
        item = next(indexed, None)
        if item is not None:
            index, task = item
            pending[pool.submit(_dispatch, task)] = index

    try:
        pool = ProcessPoolExecutor(
            max_workers=min(options.workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(factory, options),
        )
        for _ in range(min(options.workers, len(tasks))):
            submit_one()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                results[index] = cast(ResultT, future.result())
            for _ in done:
                submit_one()
    except BaseException:
        for future in pending:
            future.cancel()
        print(
            "Search stopped; waiting for active fits to finish safely.", file=sys.stderr
        )
        raise
    finally:
        # Do not release the search lock while children can still write results.
        handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            signal.signal(signal.SIGINT, handler)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return tuple(results[index] for index in range(len(tasks)))
