"""Per-frame signals computed from facial landmarks."""

from __future__ import annotations

import math

import numpy as np

NAN = float("nan")


def eye_aspect_ratio(points) -> float:
    """Eye Aspect Ratio (Soukupova & Cech, 2016) from six eye landmarks.

    ``points`` is a 6x2 array ordered p1..p6 where
    p1 / p4 are the horizontal corners, p2 / p3 the upper lid and
    p6 / p5 the lower lid (p2 above p6, p3 above p5)::

        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    Returns NaN when the eye has zero width (bad landmarks).
    """
    p = np.asarray(points, dtype=float)
    if p.shape != (6, 2):
        raise ValueError(f"expected 6x2 landmark array, got {p.shape}")
    horizontal = float(np.linalg.norm(p[0] - p[3]))
    if horizontal < 1e-9:
        return NAN
    vertical = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    return vertical / (2.0 * horizontal)


def mean_ear(left_points, right_points) -> float:
    """Average EAR of both eyes, ignoring an eye whose landmarks are degenerate."""
    values = [eye_aspect_ratio(left_points), eye_aspect_ratio(right_points)]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else NAN
