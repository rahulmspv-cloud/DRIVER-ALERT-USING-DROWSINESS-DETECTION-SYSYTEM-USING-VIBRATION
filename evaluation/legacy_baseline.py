"""The ORIGINAL project's drowsiness logic, re-implemented on recorded traces.

The original code (report, chapter 11) does this per frame:
    detect a face with a Haar cascade;
    detect eyes inside the face box with a Haar eye cascade;
    if no eye is found  -> "sleeping" -> buzzer on;
    if the face itself is not found -> nothing changes.

``haar_eyes`` in a trace is therefore -1 (no face), 0 (face but no eye) or >= 1.
Two variants are scored so the comparison is fair to the original design:

* legacy_raw      alarm on the very first frame with no eye (what the code literally does)
* legacy_4frames  alarm after 4 consecutive no-eye frames (the debounce the report text describes;
                  note this is frame-rate dependent: 4 frames is 0.13 s at 30 fps)
"""

from __future__ import annotations

from typing import List

import numpy as np

LEGACY_VARIANTS = {"legacy_raw": 1, "legacy_4frames": 4}


def legacy_alarm_onsets(t: np.ndarray, haar_eyes: np.ndarray, min_run: int) -> List[float]:
    """Times at which the legacy logic starts a new alarm."""
    onsets: List[float] = []
    run = 0
    for ti, eyes in zip(t, haar_eyes):
        if eyes == 0:
            run += 1
            if run == min_run:
                onsets.append(float(ti))
        else:  # eyes found, or face lost: the run is broken
            run = 0
    return onsets
