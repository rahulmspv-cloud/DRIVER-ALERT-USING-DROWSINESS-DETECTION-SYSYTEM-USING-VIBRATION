# Roadmap: from final-year prototype to something an OEM would pilot

The pitch is not "camera detects drowsiness, buzzer sounds" (series vehicles already ship driver monitoring).
The defensible contribution is the **response side**: a faster, validated, escalating alert (haptic first)
and a clean hand-off to the ADAS stack. Every phase below ends in evidence the customer can check.

Targets marked *(proposed)* are my suggestions: agree them with your supervisor and write them down
**before** looking at results, so nobody can say they were chosen to fit.

## Phase 0: done in this branch

- Pure-logic detector: calibration, EAR, time-weighted PERCLOS, microsleep rule, 4-level escalation with hysteresis, face-lost handling.
- Evaluation harness with the original report's logic as the baseline, confidence intervals, unscored-video reporting.
- 23 tests; synthetic smoke test (plumbing only).
- **Not done:** nothing has been run on real video, on a Raspberry Pi, or on a vibration motor.

## Phase 1: real-data baseline (weeks 1-2)

1. Get RLDD (and apply for NTHU-DDD). Download the MediaPipe model.
2. Extract features, run `evaluation.evaluate`. Sweep thresholds on 4 folds, freeze, score the 5th.
3. Deliverable: one table (EAR+PERCLOS vs legacy_raw vs legacy_4frames) with CIs and false alarms per hour.
- Gate *(proposed)*: at WARNING level, specificity >= 90% on alert videos and clearly better recall than both legacy variants. If it is not, fix detection before touching anything else: the rest of the roadmap depends on this.

## Phase 2: robustness (weeks 3-4)

- Score by NTHU scenario (glasses, night, sunglasses). Report where it fails.
- Near-infrared camera + IR illuminator for night and sunglasses; re-measure.
- Add head pose and the mouth aspect ratio (yawning, using YawDD) as extra features; add a "head down / looking away" state instead of treating it as closed eyes.
- Gate *(proposed)*: no scenario where specificity collapses without being reported and explained.

## Phase 3: embedded port and latency (weeks 5-6)

- Run on Raspberry Pi 5 (and a Jetson Orin Nano if a small neural model is needed). Measure **end-to-end latency**: eye closure at the camera to vibration at the seat, including actuator rise time, with a high-speed camera or logic analyser.
- Move the actuator wiring to a MOSFET driver; stop using GPIO2/3.
- Target *(proposed)*: >= 15 fps sustained, eye-closure-to-alert under 500 ms for the microsleep path. Report the measured number whatever it is.

## Phase 4: haptic user study (weeks 6-9), the differentiator

- Question: does seat/steering vibration produce faster driver recovery than a buzzer, without startling?
- Conditions: buzzer only, haptic only, both. Within-subject, counterbalanced, driving simulator or a safe static rig. Measure time to hand-on-wheel / eyes-on-road, plus Karolinska Sleepiness Scale before and after.
- Test frequency, amplitude and pulse pattern, with staged intensity (ADVISORY soft, WARNING stronger, CRITICAL maximum). The 5 Hz pulse and the duty values in `actuators.py` are placeholders.
- Needs informed consent and your supervisor's approval; do not skip it for human subjects.
- Headline you want: "haptic alerts reduced time-to-recovery by X% (n = N, 95% CI ...)". Without N and a CI it is an anecdote.

## Phase 5: vehicle integration concept (weeks 8-10)

Here your chassis background matters, because camera-only systems cannot do this:
- Fuse driver state with vehicle signals over CAN: steering-angle reversal rate, lane-position variability, lane-keep status, vehicle speed (gates the face-lost advisory).
- Define the hand-off: CRITICAL sets `handoff_request`; specify what the ADAS stack does (reduce speed, hazard lights, controlled stop) and what it must **not** do.
- Safety and compliance map (a document, not a certification): ISO 26262 (functional safety, hazard analysis, an ASIL estimate), ISO 21448 SOTIF (known limits: sunglasses, glare, camera blocked), ISO/SAE 21434 (cybersecurity), GDPR (process on device, store no video). Check the current EU and Euro NCAP driver-monitoring requirements before citing them.

## Phase 6: pitch pack (weeks 10-12)

1. Problem and why a single buzzer fails.
2. Concept: detect, escalate with haptics, hand off.
3. Architecture diagram.
4. Baseline results (Phase 1-2) with CIs, honest limits.
5. Latency (Phase 3) and user-study result (Phase 4).
6. Integration and compliance concept (Phase 5).
7. Ask: a pilot, a thesis/alternance collaboration, or feedback from an ADAS / HMI team.

## Known weaknesses to fix or disclose

- Video-level evaluation on self-reported labels; no per-frame ground truth.
- No sensor-based ground truth (EEG/EOG); KSS is subjective.
- A visible-light camera fails in the conditions where drowsiness is most likely, until Phase 2.
- Thresholds in `config.py` are initial values, not validated ones.
