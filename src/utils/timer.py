"""Simple context manager timer for profiling code blocks."""

from __future__ import annotations

import time


class Timer:
    """Context manager for timing code blocks.

    Usage::

        with Timer("forward pass") as t:
            model(x)
        print(t.elapsed)  # seconds as float

        # Or without a name:
        with Timer() as t:
            expensive_op()
        print(f"Took {t.elapsed:.3f}s")
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start: float = 0.0
        self._end: float | None = None

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.4f}s")

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds.

        If the timer is still running (inside the context), returns time
        since ``__enter__``.  After the context exits, returns the final
        measured duration.
        """
        if self._end is not None:
            return self._end - self._start
        return time.perf_counter() - self._start
