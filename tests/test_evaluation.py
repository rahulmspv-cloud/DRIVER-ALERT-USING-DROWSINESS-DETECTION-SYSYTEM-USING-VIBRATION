"""Harness tests. The synthetic traces are invented: these tests check plumbing, not real-world accuracy."""

import math
from pathlib import Path

import numpy as np
import pytest

from drowsiness_adas import AlertLevel, Config
from evaluation import evaluate as ev
from evaluation import synthetic
from evaluation.legacy_baseline import legacy_alarm_onsets
from evaluation.metrics import f1, wilson_interval


def test_wilson_interval_is_wide_for_small_samples_and_contains_estimate():
    lo, hi = wilson_interval(5, 5)
    assert hi == pytest.approx(1.0) and 0.4 < lo < 0.7        # 5/5 is not "100% sure"
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25
    assert all(math.isnan(x) for x in wilson_interval(0, 0))


def test_f1_edge_cases():
    assert f1(1.0, 1.0) == 1.0
    assert math.isnan(f1(float("nan"), 0.5))
    assert math.isnan(f1(0.0, 0.0))


def test_legacy_onsets_raw_vs_debounced():
    t = np.arange(10) / 10.0
    eyes = np.array([2, 0, 2, 0, 0, 0, 0, 2, -1, 0])
    assert legacy_alarm_onsets(t, eyes, 1) == pytest.approx([0.1, 0.3, 0.9])
    assert legacy_alarm_onsets(t, eyes, 4) == pytest.approx([0.6])


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    synthetic.main(["--out", str(d), "--per-class", "4", "--minutes", "6", "--seed", "1"])
    return d


def test_end_to_end_on_synthetic_traces(synth_dir):
    traces = ev.load_traces(synth_dir)
    assert len(traces) == 8 and all(t.synthetic for t in traces)
    res = ev.evaluate(traces, Config(), {"drowsy"}, {"alert"})
    assert res["synthetic"] is True and res["n_scored"] == 8

    m = res["methods"]["ear_perclos"]
    assert m["recall"] >= 0.75 and m["specificity"] >= 0.75   # separable by construction

    # the original "no eye found" logic fires on nearly every alert video, because Haar misses open eyes
    assert res["methods"]["legacy_raw"]["specificity"] <= 0.25
    assert res["methods"]["legacy_raw"]["false_alarms_per_hour"] > 10 * (m["false_alarms_per_hour"] + 1e-9)


def test_markdown_report_flags_synthetic_data(synth_dir):
    traces = ev.load_traces(synth_dir)
    res = ev.evaluate(traces, Config(), {"drowsy"}, {"alert"})
    md = ev.to_markdown(res, Config(), AlertLevel.WARNING)
    assert "SYNTHETIC DATA" in md and "ear_perclos" in md and "legacy_raw" in md


def test_unscored_videos_are_reported_not_hidden(synth_dir, tmp_path):
    # a trace where the face is never found cannot be calibrated -> must be counted as unscored
    t = np.arange(0, 60, 1 / 30)
    np.savez_compressed(tmp_path / "noface.npz", t=t, ear=np.full_like(t, np.nan),
                        haar_eyes=np.full(len(t), -1, dtype=np.int16), proc_ms=np.zeros(len(t)),
                        haar_ms=np.zeros(len(t)), fps=30.0, synthetic=False, host="test")
    (tmp_path / "index.csv").write_text("trace,label,subject,video\nnoface.npz,drowsy,s0,x\n")
    res = ev.evaluate(ev.load_traces(tmp_path), Config(), {"drowsy"}, {"alert"})
    assert res["n_videos"] == 1 and res["n_scored"] == 0 and res["unscored"] == ["noface.npz"]


def test_cli_writes_results_files(synth_dir, tmp_path):
    out = tmp_path / "res"
    assert ev.main(["--traces", str(synth_dir), "--out", str(out), "--sweep"]) == 0
    assert (out / "results.json").is_file()
    assert "Threshold sweep" in (out / "results.md").read_text()
