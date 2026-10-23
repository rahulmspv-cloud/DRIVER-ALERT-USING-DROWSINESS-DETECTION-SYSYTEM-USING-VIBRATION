"""Time-weighted rolling PERCLOS (percentage of eyelid closure)."""

from __future__ import annotations

from collections import deque


class RollingPerclos:
    """Fraction of *valid* time the eyes were closed inside a sliding window.

    Each sample carries a weight ``dt`` (seconds it represents), so the result
    does not depend on the frame rate or on dropped frames. Frames where no
    face was found are simply never added, so they neither raise nor lower
    the value.
    """

    def __init__(self, window_s: float = 60.0) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = window_s
        self._samples: deque[tuple[float, float, bool]] = deque()
        self._valid = 0.0
        self._closed = 0.0

    def add(self, t: float, dt: float, closed: bool) -> None:
        self._samples.append((t, dt, closed))
        self._valid += dt
        if closed:
            self._closed += dt
        while self._samples and t - self._samples[0][0] > self.window_s:
            _, old_dt, old_closed = self._samples.popleft()
            self._valid -= old_dt
            if old_closed:
                self._closed -= old_dt
        # guard against tiny negative values from float drift
        self._valid = max(self._valid, 0.0)
        self._closed = max(self._closed, 0.0)

    @property
    def valid_time(self) -> float:
        return self._valid

    @property
    def value(self) -> float:
        """PERCLOS in [0, 1]; 0.0 when there is no data yet."""
        return self._closed / self._valid if self._valid > 1e-9 else 0.0
