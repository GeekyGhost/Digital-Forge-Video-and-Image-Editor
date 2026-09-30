"""Where a worker's wall-clock actually went, split by stage.

The panel shows one elapsed number and `.aiv/worker.log` records peak VRAM, which together
answer "how long did that take" and nothing else. Every question worth asking about performance
needs the split instead:

  * Is a resident worker worth building? Only if load is a large share of a run. A 7-minute run
    that is 5 minutes of load and 2 of denoising says yes loudly; the same 7 minutes at 30
    seconds of load says spend the effort elsewhere.
  * Did Sage help? Attention lives in denoise. Comparing totals buries a 20% denoise improvement
    under an unchanged four-minute load.
  * How does this compare to another implementation? Their number may or may not include load,
    and comparing a load-inclusive total against a load-exclusive one is how you conclude the
    opposite of the truth.

Deliberately stdlib-only and torch-free, like `ltx_geometry` and `ltx_offload`, so the adapters
can import it before torch exists and the tests can exercise it anywhere.
"""

from __future__ import annotations

import sys
import time
from typing import Callable


def _humanise(seconds: float) -> str:
    """`431.3s` reads worse than `7:11`, and both matter -- so print both."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}:{remainder:04.1f}"


class StageTimer:
    """Accumulates named stages in the order they finish.

    Usage is one call per boundary rather than a context manager per stage, because the stages
    here are not nested blocks -- they are points in a long linear function, and wrapping each in
    `with` would mean restructuring the adapter around the instrumentation.
    """

    def __init__(self, tag: str, clock: Callable[[], float] = time.perf_counter) -> None:
        self._tag = tag
        self._clock = clock
        self._start = clock()
        self._mark = self._start
        self._stages: list[tuple[str, float]] = []

    def stage(self, name: str) -> float:
        """Close the current stage under `name` and start the next. Returns its duration."""
        now = self._clock()
        elapsed = now - self._mark
        self._mark = now
        self._stages.append((name, elapsed))
        return elapsed

    @property
    def total(self) -> float:
        return self._clock() - self._start

    @property
    def stages(self) -> tuple[tuple[str, float], ...]:
        return tuple(self._stages)

    def share(self, name: str) -> float:
        """What fraction of the total a named stage took, 0.0 when it was never recorded."""
        total = sum(duration for _, duration in self._stages)
        if total <= 0:
            return 0.0
        for stage, duration in self._stages:
            if stage == name:
                return duration / total
        return 0.0

    def summary(self) -> str:
        """One line naming every stage, its duration, and its share of the run."""
        if not self._stages:
            return f"[{self._tag}] timing: nothing recorded"
        measured = sum(duration for _, duration in self._stages)
        parts = [
            f"{name} {_humanise(duration)} ({duration / measured:.0%})"
            for name, duration in self._stages
        ]
        return f"[{self._tag}] timing: " + " · ".join(parts) + f" · measured {_humanise(measured)}"

    def report(self, stream=None) -> str:
        """Print the summary where the worker's other diagnostics go, and return it."""
        line = self.summary()
        print(line, file=stream if stream is not None else sys.stderr, flush=True)
        return line
