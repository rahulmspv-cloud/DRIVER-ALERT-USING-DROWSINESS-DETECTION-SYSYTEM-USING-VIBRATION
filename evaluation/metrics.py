"""Small, dependency-free metric helpers."""

from __future__ import annotations

import math
from typing import Tuple


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion. Honest about small samples."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def f1(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)
