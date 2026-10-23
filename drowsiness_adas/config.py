"""All tunable thresholds in one place.

Every value below is an INITIAL value taken from common practice in the
drowsiness literature. None of them has been validated by this project yet.
They are meant to be tuned on training folds of a public dataset
(see docs/DATASETS.md) and then frozen before measuring on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- per-driver calibration -------------------------------------------
    calibration_s: float = 15.0          # seconds of valid face data used to learn the open-eye EAR
    calibration_min_samples: int = 60    # and at least this many frames
    open_percentile: float = 75.0        # robust "eyes open" level (blinks pull the low tail down)
    min_baseline_ear: float = 0.15       # below this the calibration is rejected and restarted

    # --- eye state ----------------------------------------------------------
    closed_ratio: float = 0.55           # eye counts as closed when EAR < closed_ratio * baseline

    # --- PERCLOS ------------------------------------------------------------
    perclos_window_s: float = 60.0
    perclos_min_fill_s: float = 10.0     # do not trust PERCLOS before this much valid data
    perclos_advisory: float = 0.15
    perclos_warning: float = 0.25

    # --- single long closures (microsleep) ------------------------------------
    closure_warning_s: float = 1.0
    closure_critical_s: float = 2.0
    repeat_window_s: float = 30.0        # a 2nd long closure inside this window is CRITICAL

    # --- alert hysteresis -------------------------------------------------------
    deescalate_hold_s: float = 5.0       # candidate level must stay lower this long before we step down
    deescalate_factor: float = 0.7       # PERCLOS must fall to factor * threshold to leave a level

    # --- face lost -------------------------------------------------------------
    face_lost_degraded_s: float = 3.0    # status becomes DEGRADED (monitoring unavailable)
    face_lost_advisory_s: float = 8.0    # ...and level rises to at least ADVISORY.
                                         # With a vehicle-speed signal this should be gated on
                                         # "vehicle moving"; set to None to disable.

    # --- time handling -------------------------------------------------------------
    max_dt_s: float = 0.5                # a gap longer than this counts as only this much valid time
