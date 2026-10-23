# Driver Alert using Drowsiness Detection + Vibration

Camera-based driver drowsiness detection that escalates through **advisory, warning and critical** alerts,
using seat/steering **vibration** before (and together with) a buzzer, and requests an ADAS hand-off on microsleep.

Originally a B.E. Automobile Engineering final-year project (PSG College of Technology, 2022). This branch rebuilds
the detection logic and adds an evaluation harness, so the claim "this works" can be backed with numbers.

## Status, stated plainly

| | |
|---|---|
| Detector logic (EAR, PERCLOS, microsleep, escalation) | implemented, 23 automated tests pass |
| Evaluation harness with the original logic as baseline | implemented, tested on synthetic data only |
| Results on real drivers | **none yet.** Run the baseline on RLDD / NTHU-DDD (see `docs/DATASETS.md`) |
| Landmark extraction (`landmarks.py`) | written against the MediaPipe Tasks API; **not yet run on real faces** (the model file could not be downloaded in the authoring environment) |
| Raspberry Pi vibration/buzzer output (`GpioActuator`) | written, **not tested on hardware** |
| Haptic intensities / 5 Hz pulse | placeholders pending a user study |

Do not quote any number from `evaluation.synthetic`: it is invented data used only to test the code.

## How it works

1. **Landmarks.** MediaPipe FaceLandmarker gives 6 points per eye.
2. **EAR (Eye Aspect Ratio).** `(|p2-p6| + |p3-p5|) / (2 |p1-p4|)`: eyelid gap divided by eye width, about 0.3 open and near 0.1 closed.
3. **Calibration.** The first 15 s of valid face data set a per-driver open-eye level; "closed" means EAR below 55% of it.
4. **PERCLOS.** Share of time the eyes are closed over a rolling 60 s window (time-weighted, so frame drops do not bias it).
5. **Alert levels.**

| Level | Trigger | Output (placeholder values) |
|---|---|---|
| ADVISORY | PERCLOS >= 0.15, or face lost >= 8 s | soft vibration |
| WARNING | PERCLOS >= 0.25, or eyes closed >= 1 s | stronger vibration + buzzer |
| CRITICAL | eyes closed >= 2 s, or a 2nd 1 s closure within 30 s | maximum vibration + buzzer + `handoff_request` |

Escalation is immediate; stepping down needs the lower level to hold for 5 s. If the face disappears the
detector reports `DEGRADED` and **keeps** any active alert (a head-down driver must not silence it).
All thresholds are in `drowsiness_adas/config.py`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                                   # 23 tests, no camera needed

# Smoke-test the evaluation harness on SYNTHETIC data (plumbing only, not evidence)
python -m evaluation.synthetic --out traces_synthetic/
python -m evaluation.evaluate --traces traces_synthetic/ --out results_synthetic/

# Live / video run (download the model first, see drowsiness_adas/landmarks.py)
python -m drowsiness_adas.run --model face_landmarker.task --source 0
python -m drowsiness_adas.run --model face_landmarker.task --actuator gpio     # on the Pi
```

Real baseline: follow "Running the baseline" in `docs/DATASETS.md`.

## Layout

```
drowsiness_adas/   detector (pure logic), signals, perclos, config, landmarks (MediaPipe), actuators, run (CLI)
evaluation/        extract_features, evaluate (+ threshold sweep), legacy_baseline, make_manifest, synthetic
tests/             unit tests + harness tests
docs/              DATASETS.md, ROADMAP.md
```

## Issues found in the original report (fix before showing it to a company)

These are from the 2022 report; reviewers will spot them.

- **Hardware inconsistent.** The abstract says Raspberry Pi 3 Model B with 1 GB RAM; chapter 6.4 lists a BCM2835 / ARM11 700 MHz / 512 MB board (the original-generation spec) under the "Pi 3 B+" name.
- **Method inconsistent.** The abstract and section 4.2 describe Circular Hough Transform eye tracking; the code in chapter 11 uses the Haar eye cascade only.
- **Code does not implement the flowchart.** The flowchart describes PERCLOS with a 0.25 s blink rule; the code treats "no eye found" in a single frame as sleeping, which also fires on a head turn, glare or glasses. Thresholds are also stated three ways (4 frames, X >= 5, "more than a second").
- **Unsupported accuracy claim.** "99.0% eye tracking on some test videos" has no dataset, count or definition.
- **Standard misused.** ISO 2631-1 defines how to measure whole-body vibration exposure; it does not prescribe a 5 Hz awakening signal.
- **Pins.** The code drives the buzzer and motor from GPIO2/GPIO3, which carry the I2C pull-up resistors; use a MOSFET driver and other pins (`actuators.py` defaults to GPIO18/23).
- **Python 2.** The code uses `print` statements; this branch is Python 3.

## Roadmap

See `docs/ROADMAP.md`: real-data baseline, robustness (night, sunglasses), Raspberry Pi latency, haptic user study, CAN fusion and safety-standard mapping, pitch pack.
