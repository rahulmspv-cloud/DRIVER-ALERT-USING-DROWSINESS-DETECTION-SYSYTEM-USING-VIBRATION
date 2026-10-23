"""Unit tests for EAR, PERCLOS and the alert state machine (pure logic, no camera)."""

import math

import numpy as np
import pytest

from drowsiness_adas import AlertLevel, Config, DrowsinessDetector, RollingPerclos, Status, eye_aspect_ratio
from drowsiness_adas.actuators import MockActuator

FPS = 30.0
OPEN = 0.30
CLOSED = 0.08


def eye(width=30.0, height=9.0):
    """Six landmarks p1..p6 of a symmetric eye of the given pixel size."""
    h = height / 2.0
    return [(0, 0), (width * 0.3, -h), (width * 0.7, -h), (width, 0), (width * 0.7, h), (width * 0.3, h)]


# ----------------------------------------------------------------------- EAR
def test_ear_matches_hand_calculation():
    # vertical gaps are 9 each, width 30 -> (9 + 9) / (2 * 30) = 0.3
    assert eye_aspect_ratio(eye(30, 9)) == pytest.approx(0.3)


def test_ear_is_zero_for_closed_eye_and_scale_invariant():
    assert eye_aspect_ratio(eye(30, 0)) == pytest.approx(0.0)
    assert eye_aspect_ratio(eye(60, 18)) == pytest.approx(eye_aspect_ratio(eye(30, 9)))


def test_ear_degenerate_width_is_nan_and_bad_shape_raises():
    assert math.isnan(eye_aspect_ratio(eye(0, 5)))
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0)] * 5)


# -------------------------------------------------------------------- PERCLOS
def test_perclos_fraction_and_window_eviction():
    p = RollingPerclos(window_s=10.0)
    for i in range(100):                      # 10 s, half closed
        p.add(i * 0.1, 0.1, closed=(i % 2 == 0))
    assert p.value == pytest.approx(0.5, abs=0.02)
    for i in range(100, 300):                 # 20 more seconds, all open -> closed share must vanish
        p.add(i * 0.1, 0.1, closed=False)
    assert p.value == pytest.approx(0.0, abs=1e-6)
    assert p.valid_time == pytest.approx(10.0, abs=0.2)


def test_perclos_empty_is_zero():
    assert RollingPerclos(5.0).value == 0.0


# ------------------------------------------------------------------- detector
class Driver:
    """Drives the detector with a synthetic EAR signal at a fixed frame rate."""

    def __init__(self, cfg=None):
        self.det = DrowsinessDetector(cfg)
        self.t = 0.0
        self.outputs = []

    def run(self, seconds, ear=OPEN):
        for _ in range(int(round(seconds * FPS))):
            fn = ear if callable(ear) else (lambda t, e=ear: e)
            out = self.det.update(self.t, fn(self.t))
            self.outputs.append(out)
            self.t += 1.0 / FPS
        return self.outputs[-1]

    def blink(self, seconds):
        return self.run(seconds, CLOSED)

    def max_level(self, since=0):
        return max(o.level for o in self.outputs[since:])


def calibrated(cfg=None):
    d = Driver(cfg)
    last = d.run(20.0)
    assert last.status == Status.MONITORING
    return d


def test_calibration_learns_open_eye_baseline():
    d = Driver()
    out = d.run(5.0)
    assert out.status == Status.CALIBRATING and out.level == AlertLevel.NORMAL
    out = d.run(15.0)
    assert out.status == Status.MONITORING
    assert out.baseline == pytest.approx(OPEN, abs=0.01)


def test_normal_blinking_never_alerts():
    d = calibrated()
    start = len(d.outputs)
    for _ in range(180):                       # 3 minutes at ~15 blinks/min, 0.2 s each
        d.run(3.8)
        d.blink(0.2)
    assert d.max_level(start) == AlertLevel.NORMAL


def test_one_second_closure_is_warning_and_two_seconds_is_critical():
    d = calibrated()
    start = len(d.outputs)
    d.blink(1.2)
    assert d.max_level(start) == AlertLevel.WARNING
    d.run(10.0)  # recover
    start = len(d.outputs)
    d.blink(2.3)
    assert d.max_level(start) == AlertLevel.CRITICAL
    assert any(o.handoff_request for o in d.outputs[start:])


def test_repeated_long_closure_within_window_is_critical():
    d = calibrated()
    d.blink(1.2)       # first long closure: WARNING
    d.run(8.0)
    start = len(d.outputs)
    d.blink(1.2)       # second one within repeat_window_s: CRITICAL
    assert d.max_level(start) == AlertLevel.CRITICAL


def test_slow_blinks_raise_perclos_to_warning_without_any_single_long_closure():
    d = calibrated()
    start = len(d.outputs)
    for _ in range(40):                        # 0.6 s closed every 2.4 s -> PERCLOS 0.25
        d.run(1.8)
        d.blink(0.6)
    assert d.max_level(start) >= AlertLevel.WARNING
    assert max(o.closure_s for o in d.outputs[start:]) < Config().closure_warning_s


def test_hysteresis_holds_level_then_releases():
    d = calibrated()
    d.blink(1.2)
    just_after = d.run(1.0)
    assert just_after.level >= AlertLevel.WARNING          # still held right after recovery
    final = d.run(80.0)                                      # PERCLOS window fully refreshed
    assert final.level == AlertLevel.NORMAL


def test_face_lost_degrades_but_does_not_silence_active_alert():
    d = calibrated()
    d.blink(2.3)                                             # CRITICAL
    out = d.run(4.0, ear=float("nan"))
    assert out.status == Status.DEGRADED
    assert out.level == AlertLevel.CRITICAL


def test_face_lost_long_enough_gives_advisory():
    d = calibrated()
    out = d.run(9.0, ear=float("nan"))
    assert out.status == Status.DEGRADED and out.level >= AlertLevel.ADVISORY


def test_face_loss_is_not_counted_as_closed_eyes():
    d = calibrated()
    start = len(d.outputs)
    d.run(2.0, ear=float("nan"))                             # head turn, short
    d.run(5.0)
    assert d.max_level(start) == AlertLevel.NORMAL


def test_low_baseline_calibration_is_rejected():
    d = Driver()
    out = d.run(20.0, ear=0.08)                              # e.g. eyes closed/squinting during calibration
    assert out.status == Status.CALIBRATING and out.baseline is None


# ------------------------------------------------------------------ actuators
def test_mock_actuator_acts_only_on_change():
    d = calibrated()
    act = MockActuator()
    for o in d.outputs:
        act.apply(o)
    # one event for CALIBRATING and one when monitoring starts; steady frames add nothing
    assert [e[1] for e in act.events] == [AlertLevel.NORMAL, AlertLevel.NORMAL]
    d.blink(2.3)
    for o in d.outputs[-70:]:
        act.apply(o)
    levels = [e[1] for e in act.events]
    assert AlertLevel.CRITICAL in levels
    assert levels.count(AlertLevel.CRITICAL) == 1
