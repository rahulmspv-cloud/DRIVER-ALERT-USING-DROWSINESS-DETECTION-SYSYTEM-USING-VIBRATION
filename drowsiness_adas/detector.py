"""Drowsiness decision logic: calibration -> EAR -> PERCLOS -> alert level.

The detector is pure logic (no camera, no OpenCV) so it can be unit-tested,
replayed on recorded EAR traces, and later ported to C++ / AUTOSAR if needed.

Alert levels
------------
NORMAL    nothing to report
ADVISORY  early warning       (PERCLOS >= perclos_advisory, or face lost for a long time)
WARNING   driver is drowsy    (PERCLOS >= perclos_warning, or one eye closure >= closure_warning_s)
CRITICAL  microsleep          (closure >= closure_critical_s, or a repeated long closure
                               within repeat_window_s) -> request ADAS hand-off

Escalation is immediate; de-escalation needs the lower level to persist for
``deescalate_hold_s`` so the alert does not flicker around a threshold.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

import numpy as np

from .config import Config
from .perclos import RollingPerclos


class AlertLevel(IntEnum):
    NORMAL = 0
    ADVISORY = 1
    WARNING = 2
    CRITICAL = 3


class Status(str, Enum):
    CALIBRATING = "calibrating"
    MONITORING = "monitoring"
    DEGRADED = "degraded"  # face not visible: monitoring unavailable


@dataclass(frozen=True)
class Output:
    t: float
    status: Status
    level: AlertLevel
    ear: float
    closed: bool
    perclos: float          # NaN until perclos_min_fill_s of valid data exists
    closure_s: float        # duration of the current uninterrupted eye closure
    baseline: Optional[float]
    handoff_request: bool   # True at CRITICAL: ask the ADAS stack for a safe-state manoeuvre
    reason: str


class DrowsinessDetector:
    """Feed one ``update(t, ear)`` per frame; ``ear`` is NaN when no face was found."""

    def __init__(self, config: Optional[Config] = None, baseline_ear: Optional[float] = None) -> None:
        self.cfg = config or Config()
        self.baseline: Optional[float] = baseline_ear
        self._perclos = RollingPerclos(self.cfg.perclos_window_s)
        self._calib: list[float] = []
        self._calib_time = 0.0
        self._last_t: Optional[float] = None
        self._face_lost_for = 0.0
        self._closure_start: Optional[float] = None
        self._long_fired = False
        self._repeat_flag = False
        self._long_events: deque[float] = deque()
        self._level = AlertLevel.NORMAL
        self._below_since: Optional[float] = None
        self._last_ear = float("nan")

    # ------------------------------------------------------------------ API
    def update(self, t: float, ear: Optional[float]) -> Output:
        cfg = self.cfg
        raw_dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        dt = min(raw_dt, cfg.max_dt_s)

        face = ear is not None and not math.isnan(ear)
        if not face:
            self._face_lost_for += raw_dt
            self._closure_start = None
            self._long_fired = False
            return self._face_lost_output(t)
        self._face_lost_for = 0.0
        self._last_ear = float(ear)

        if self.baseline is None:
            return self._calibrate(t, float(ear), dt)
        return self._monitor(t, float(ear), dt)

    # ----------------------------------------------------------- internals
    def _calibrate(self, t: float, ear: float, dt: float) -> Output:
        cfg = self.cfg
        self._calib.append(ear)
        self._calib_time += dt
        reason = "calibrating"
        if self._calib_time >= cfg.calibration_s and len(self._calib) >= cfg.calibration_min_samples:
            candidate = float(np.percentile(self._calib, cfg.open_percentile))
            if candidate < cfg.min_baseline_ear:
                self._calib.clear()
                self._calib_time = 0.0
                reason = f"calibration rejected (baseline EAR {candidate:.2f} < {cfg.min_baseline_ear})"
            else:
                self.baseline = candidate
                reason = f"baseline EAR {candidate:.3f}"
        status = Status.MONITORING if self.baseline is not None else Status.CALIBRATING
        return Output(t, status, AlertLevel.NORMAL, ear, False, float("nan"), 0.0,
                      self.baseline, False, reason)

    def _monitor(self, t: float, ear: float, dt: float) -> Output:
        cfg = self.cfg
        assert self.baseline is not None
        closed = ear < cfg.closed_ratio * self.baseline
        self._perclos.add(t, dt, closed)

        if closed:
            if self._closure_start is None:
                self._closure_start = t
                self._long_fired = False
                self._repeat_flag = False
            closure_s = t - self._closure_start
        else:
            self._closure_start = None
            self._long_fired = False
            closure_s = 0.0

        perclos = (self._perclos.value
                   if self._perclos.valid_time >= cfg.perclos_min_fill_s else float("nan"))

        candidate, reason = AlertLevel.NORMAL, ""

        # PERCLOS rules, with hysteresis relative to the current level
        if not math.isnan(perclos):
            adv_t = cfg.perclos_advisory * (cfg.deescalate_factor if self._level >= AlertLevel.ADVISORY else 1.0)
            war_t = cfg.perclos_warning * (cfg.deescalate_factor if self._level >= AlertLevel.WARNING else 1.0)
            if perclos >= war_t:
                candidate, reason = AlertLevel.WARNING, f"PERCLOS {perclos:.2f} >= {war_t:.2f}"
            elif perclos >= adv_t:
                candidate, reason = AlertLevel.ADVISORY, f"PERCLOS {perclos:.2f} >= {adv_t:.2f}"

        # Single long closure rules
        if closure_s >= cfg.closure_warning_s:
            if not self._long_fired:
                self._long_fired = True
                while self._long_events and t - self._long_events[0] > cfg.repeat_window_s:
                    self._long_events.popleft()
                self._repeat_flag = len(self._long_events) > 0
                self._long_events.append(t)
            if self._repeat_flag and candidate < AlertLevel.CRITICAL:
                candidate, reason = AlertLevel.CRITICAL, f"repeated long closure ({closure_s:.1f} s)"
            elif candidate < AlertLevel.WARNING:
                candidate, reason = AlertLevel.WARNING, f"eyes closed {closure_s:.1f} s"
        if closure_s >= cfg.closure_critical_s and candidate < AlertLevel.CRITICAL:
            candidate, reason = AlertLevel.CRITICAL, f"microsleep: eyes closed {closure_s:.1f} s"

        self._apply_level(t, candidate)
        if self._level > candidate and not reason:
            reason = "holding level (hysteresis)"
        return Output(t, Status.MONITORING, self._level, ear, closed, perclos, closure_s,
                      self.baseline, self._level == AlertLevel.CRITICAL, reason)

    def _apply_level(self, t: float, candidate: AlertLevel) -> None:
        if candidate >= self._level:
            self._level = candidate
            self._below_since = None
            return
        if self._below_since is None:
            self._below_since = t
        if t - self._below_since >= self.cfg.deescalate_hold_s:
            self._level = candidate
            self._below_since = None

    def _face_lost_output(self, t: float) -> Output:
        cfg = self.cfg
        status = Status.MONITORING if self.baseline is not None else Status.CALIBRATING
        reason = "face not visible"
        if self._face_lost_for >= cfg.face_lost_degraded_s:
            status = Status.DEGRADED
            reason = f"face lost {self._face_lost_for:.0f} s: monitoring unavailable"
        # Hold the current level: a driver whose head dropped out of view must not
        # silence an alert that is already active.
        level = self._level
        if cfg.face_lost_advisory_s is not None and self._face_lost_for >= cfg.face_lost_advisory_s:
            level = max(level, AlertLevel.ADVISORY)
        return Output(t, status, level, self._last_ear, False, float("nan"), 0.0,
                      self.baseline, level == AlertLevel.CRITICAL, reason)
