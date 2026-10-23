"""Score the EAR+PERCLOS detector against the original Haar logic on recorded traces.

    python -m evaluation.evaluate --traces traces/ --out results/
    python -m evaluation.evaluate --traces traces/ --out results/ --sweep

Video-level task: a video labelled drowsy should raise at least one alert at or
above --alert-level (default 2 = WARNING); a video labelled alert should not.

Reported per method:
    recall / specificity (with 95% Wilson intervals), precision, F1,
    false alarms per hour on alert videos, median time to first alert on drowsy videos.
Videos the EAR detector could not calibrate on (face rarely visible) are counted
as UNSCORED and excluded for ALL methods, and the count is reported.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from drowsiness_adas import AlertLevel, Config, DrowsinessDetector

from .legacy_baseline import LEGACY_VARIANTS, legacy_alarm_onsets
from .metrics import f1, safe_div, wilson_interval


@dataclass
class Trace:
    name: str
    label: str
    t: np.ndarray
    ear: np.ndarray
    haar_eyes: np.ndarray
    proc_ms: np.ndarray
    haar_ms: np.ndarray
    synthetic: bool
    host: str

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


@dataclass
class MethodResult:
    scored: bool
    onsets: List[float]
    monitored_s: float


def load_traces(trace_dir: Path) -> List[Trace]:
    traces: List[Trace] = []
    with open(trace_dir / "index.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            z = np.load(trace_dir / row["trace"], allow_pickle=False)
            traces.append(Trace(
                name=row["trace"], label=row["label"], t=z["t"], ear=z["ear"], haar_eyes=z["haar_eyes"],
                proc_ms=z["proc_ms"], haar_ms=z["haar_ms"],
                synthetic=bool(z["synthetic"]) if "synthetic" in z else False,
                host=str(z["host"]) if "host" in z else "unknown",
            ))
    return traces


# ------------------------------------------------------------------ methods
def run_ear_perclos(trace: Trace, cfg: Config, min_level: AlertLevel = AlertLevel.WARNING) -> MethodResult:
    det = DrowsinessDetector(cfg)
    onsets: List[float] = []
    prev = AlertLevel.NORMAL
    monitor_start: Optional[float] = None
    for ti, ear in zip(trace.t, trace.ear):
        out = det.update(float(ti), float(ear))
        # monitoring starts when a baseline exists (status alone is not enough: a driver whose face
        # was never found is reported DEGRADED without ever having been calibrated)
        if monitor_start is None and det.baseline is not None:
            monitor_start = float(ti)
        if out.level >= min_level and prev < min_level:
            onsets.append(float(ti))
        prev = out.level
    if monitor_start is None:
        return MethodResult(False, [], 0.0)
    return MethodResult(True, onsets, float(trace.t[-1]) - monitor_start)


def run_legacy(trace: Trace, min_run: int) -> MethodResult:
    return MethodResult(True, legacy_alarm_onsets(trace.t, trace.haar_eyes, min_run), trace.duration_s)


# ------------------------------------------------------------------ scoring
def score_method(rows: Sequence[tuple], positives: set, negatives: set) -> dict:
    """rows: (trace, MethodResult). Only traces whose label is positive/negative are used."""
    tp = fp = tn = fn = 0
    neg_onsets = 0
    neg_hours = 0.0
    first_alert_times: List[float] = []
    for trace, res in rows:
        fired = len(res.onsets) > 0
        if trace.label in positives:
            if fired:
                tp += 1
                first_alert_times.append(res.onsets[0] - float(trace.t[0]))
            else:
                fn += 1
        elif trace.label in negatives:
            if fired:
                fp += 1
            else:
                tn += 1
            neg_onsets += len(res.onsets)
            neg_hours += res.monitored_s / 3600.0
    recall = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    prec = safe_div(tp, tp + fp)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "recall_ci95": wilson_interval(tp, tp + fn),
        "specificity": spec, "specificity_ci95": wilson_interval(tn, tn + fp),
        "precision": prec, "f1": f1(prec, recall),
        "false_alarms_per_hour": safe_div(neg_onsets, neg_hours),
        "median_time_to_first_alert_s": statistics.median(first_alert_times) if first_alert_times else float("nan"),
    }


def evaluate(traces: List[Trace], cfg: Config, positives: set, negatives: set,
             min_level: AlertLevel = AlertLevel.WARNING) -> dict:
    relevant = [t for t in traces if t.label in positives | negatives]
    ear_runs = [(t, run_ear_perclos(t, cfg, min_level)) for t in relevant]
    scored = [(t, r) for t, r in ear_runs if r.scored]
    unscored = [t.name for t, r in ear_runs if not r.scored]
    methods = {"ear_perclos": score_method(scored, positives, negatives)}
    scored_traces = [t for t, _ in scored]
    for name, run in LEGACY_VARIANTS.items():
        methods[name] = score_method([(t, run_legacy(t, run)) for t in scored_traces], positives, negatives)

    lat = np.concatenate([t.proc_ms for t in relevant]) if relevant else np.array([])
    lat = lat[~np.isnan(lat)]
    return {
        "n_videos": len(relevant), "n_scored": len(scored), "unscored": unscored,
        "synthetic": any(t.synthetic for t in relevant),
        "extraction_hosts": sorted({t.host for t in relevant}),
        "latency_ms_ear_path": {q: float(np.percentile(lat, p)) for q, p in (("p50", 50), ("p95", 95), ("p99", 99))}
        if lat.size else {},
        "methods": methods,
    }


# ------------------------------------------------------------------ output
def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _ci(x: float, ci) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}% ({100 * ci[0]:.0f}-{100 * ci[1]:.0f})"


def _num(x: float, fmt: str = "{:.1f}", suffix: str = "") -> str:
    return "n/a" if x != x else fmt.format(x) + suffix


def to_markdown(res: dict, cfg: Config, min_level: AlertLevel) -> str:
    lines: List[str] = []
    if res["synthetic"]:
        lines += ["> **SYNTHETIC DATA. These numbers only prove the code runs end to end. They are not evidence",
                  "> that the detector works on real drivers.**", ""]
    lines += [f"Videos scored: {res['n_scored']} of {res['n_videos']} (alert level >= {min_level.name})", ""]
    if res["unscored"]:
        lines += [f"Unscored (calibration never completed): {', '.join(res['unscored'])}", ""]
    lines += ["| Method | Recall (95% CI) | Specificity (95% CI) | Precision | F1 | False alarms / hour | Median time to alert |",
              "|---|---|---|---|---|---|---|"]
    for name, m in res["methods"].items():
        lines.append(f"| {name} | {_ci(m['recall'], m['recall_ci95'])} | {_ci(m['specificity'], m['specificity_ci95'])} | "
                     f"{_pct(m['precision'])} | {_num(m['f1'], '{:.2f}')} | "
                     f"{_num(m['false_alarms_per_hour'])} | {_num(m['median_time_to_first_alert_s'], '{:.0f}', ' s')} |")
    lat = res["latency_ms_ear_path"]
    if lat:
        lines += ["", f"Per-frame latency of the landmark+EAR path (p50 / p95 / p99): "
                      f"{lat['p50']:.1f} / {lat['p95']:.1f} / {lat['p99']:.1f} ms, measured on: {', '.join(res['extraction_hosts'])}.",
                  "This is the extraction host, not the target embedded hardware."]
    lines += ["", "Thresholds used: " + json.dumps(dataclasses.asdict(cfg))]
    return "\n".join(lines) + "\n"


def sweep(traces: List[Trace], base: Config, positives: set, negatives: set, min_level: AlertLevel) -> str:
    rows = ["| closed_ratio | perclos_warning | Recall | Specificity | False alarms / hour |", "|---|---|---|---|---|"]
    for cr in (0.45, 0.50, 0.55, 0.60, 0.65):
        for pw in (0.15, 0.20, 0.25, 0.30):
            cfg = dataclasses.replace(base, closed_ratio=cr, perclos_warning=pw,
                                      perclos_advisory=min(base.perclos_advisory, pw))
            m = evaluate(traces, cfg, positives, negatives, min_level)["methods"]["ear_perclos"]
            rows.append(f"| {cr:.2f} | {pw:.2f} | {_pct(m['recall'])} | {_pct(m['specificity'])} | "
                        f"{_num(m['false_alarms_per_hour'])} |")
    return ("\n".join(rows) + "\n\nTune on training folds only, freeze the thresholds, then report on held-out subjects. "
            "Choosing the best row on the same videos you report is overfitting.\n")


def _csv_set(s: str) -> set:
    return {x.strip() for x in s.split(",") if x.strip()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--positive", default="drowsy", help="comma-separated labels counted as drowsy")
    ap.add_argument("--negative", default="alert", help="comma-separated labels counted as alert")
    ap.add_argument("--alert-level", type=int, default=int(AlertLevel.WARNING), choices=[1, 2, 3])
    ap.add_argument("--closed-ratio", type=float)
    ap.add_argument("--perclos-warning", type=float)
    ap.add_argument("--perclos-advisory", type=float)
    ap.add_argument("--sweep", action="store_true", help="also print a threshold sweep table")
    args = ap.parse_args(argv)

    overrides = {k: v for k, v in {"closed_ratio": args.closed_ratio, "perclos_warning": args.perclos_warning,
                                   "perclos_advisory": args.perclos_advisory}.items() if v is not None}
    cfg = dataclasses.replace(Config(), **overrides)
    min_level = AlertLevel(args.alert_level)
    positives, negatives = _csv_set(args.positive), _csv_set(args.negative)

    traces = load_traces(args.traces)
    res = evaluate(traces, cfg, positives, negatives, min_level)
    md = to_markdown(res, cfg, min_level)
    if args.sweep:
        md += "\n## Threshold sweep (EAR + PERCLOS)\n\n" + sweep(traces, cfg, positives, negatives, min_level)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=2))
    (args.out / "results.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
