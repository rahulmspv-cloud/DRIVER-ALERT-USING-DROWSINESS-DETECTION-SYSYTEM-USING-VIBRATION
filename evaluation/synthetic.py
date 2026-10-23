"""Generate SYNTHETIC traces to smoke-test the evaluation harness.

Everything here is invented (blink rates, durations, Haar miss rates). The output proves that the
code paths run and behave sensibly. It says NOTHING about accuracy on real drivers, and every trace
is tagged synthetic=True so the report prints a warning banner.

    python -m evaluation.synthetic --out traces_synthetic/ --per-class 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def make_trace(rng: np.random.Generator, drowsy: bool, minutes: float = 10.0, fps: float = 30.0):
    n = int(minutes * 60 * fps)
    t = np.arange(n) / fps
    base = float(np.clip(rng.normal(0.30, 0.025), 0.22, 0.38))
    ear = base + rng.normal(0.0, 0.01, n)

    # (events per minute, min duration s, max duration s)
    blink = (22, 0.3, 0.9) if drowsy else (15, 0.10, 0.30)
    events = [(blink, None)]
    if drowsy:
        events.append(((1.5, 1.0, 2.5), None))  # microsleeps

    for (rate, dmin, dmax), _ in events:
        s = 0.0
        while True:
            s += rng.exponential(60.0 / rate)
            if s >= minutes * 60:
                break
            d = rng.uniform(dmin, dmax)
            i0, i1 = int(s * fps), min(n, int((s + d) * fps) + 1)
            if i1 - i0 < 2:
                continue
            # eyelid trajectory: close in ~0.08 s, hold, reopen in ~0.12 s
            tau = np.arange(i1 - i0) / fps
            depth = np.clip(np.minimum(tau / 0.08, (d - tau) / 0.12), 0.0, 1.0)
            ear[i0:i1] = np.minimum(ear[i0:i1], base - (base - 0.08) * depth)

    closed = ear < 0.55 * base

    # Haar eye count: open eyes are found imperfectly, closed eyes are mostly not found.
    open_draw = rng.choice([2, 1, 0], size=n, p=[0.75, 0.12, 0.13])
    closed_draw = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    haar = np.where(closed, closed_draw, open_draw).astype(np.int16)

    # occasional short head turns: face lost
    for _ in range(int(minutes / 3)):
        i0 = int(rng.uniform(30, minutes * 60 - 5) * fps)
        i1 = i0 + int(rng.uniform(1.0, 2.0) * fps)
        ear[i0:i1] = np.nan
        haar[i0:i1] = -1

    proc_ms = rng.lognormal(np.log(8.0), 0.25, n)
    haar_ms = rng.lognormal(np.log(5.0), 0.25, n)
    return t, ear, haar, proc_ms, haar_ms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, drowsy in (("alert", False), ("drowsy", True)):
        for k in range(args.per_class):
            t, ear, haar, pm, hm = make_trace(rng, drowsy, args.minutes)
            name = f"synthetic_{label}_{k:02d}.npz"
            np.savez_compressed(args.out / name, t=t, ear=ear, haar_eyes=haar, proc_ms=pm, haar_ms=hm,
                                fps=30.0, synthetic=True, host="synthetic")
            rows.append({"trace": name, "label": label, "subject": f"s{k}", "video": "synthetic"})
    with open(args.out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trace", "label", "subject", "video"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic traces to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
