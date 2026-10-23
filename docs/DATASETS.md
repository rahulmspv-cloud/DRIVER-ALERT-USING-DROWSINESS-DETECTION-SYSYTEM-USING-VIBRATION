# Datasets for the baseline

Facts below were checked against the dataset papers/pages on 2026-10-23. Access terms change: confirm them
yourself before downloading, and respect each dataset's licence (for example, YawDD restricts publishing
screenshots of some participants).

## Which dataset answers which question

| Question the customer will ask | Dataset | Use |
|---|---|---|
| "Does it separate alert from drowsy drivers?" | **RLDD** | Primary video-level baseline |
| "Does it work at night, with glasses, with sunglasses?" | **NTHU-DDD** | Robustness by condition |
| "Can it also catch yawning?" | **YawDD** | Mouth module (later phase) |

## RLDD (Real-Life Drowsiness Dataset)

Ghoddoosian, Galib, Athitsos, *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection*,
CVPR Workshops 2019 ([arXiv 1904.07312](https://arxiv.org/abs/1904.07312)).

- 60 participants (51 men, 9 women, aged 20 to 59); 180 RGB videos, three per participant (one per state),
  about 10 minutes each, roughly 30 hours and 111 GB in total.
- Classes: alert, low vigilant, drowsy. **Labels are self-reported** by the participant (guided by the Karolinska
  Sleepiness Scale); the authors state there was no objective alertness sensor and that they screened out
  recordings that looked faked. Say this in the pitch before the customer finds it.
- Recorded by participants on their own phone or webcam, indoors, with varying lighting and backgrounds;
  frame rate below 30 fps.
- Five cross-validation folds of 12 participants. **Use the folds**: tune thresholds on 4 folds, freeze, report on
  the 5th. Never report numbers from thresholds tuned on the same subjects.
- Access: the paper points to a project website; it does not describe the licence. Check there.
- Binary task used here: alert vs drowsy. Low vigilant is left out because the labels are fuzzy.
  `evaluation.make_manifest --mode rldd` assumes files are named 0 / 5 / 10 inside per-subject folders.
  **That naming is an assumption, so check it against the dataset's own README.**

## NTHU-DDD (NTHU Driver Drowsiness Detection)

Dataset details below are from [arXiv 1801.02325](https://arxiv.org/abs/1801.02325) (checked). The original dataset
reference is, from memory and **not verified**, Weng, Lai, Lai, ACCV 2016 workshop: confirm the exact citation before using it.

- Training set: 18 volunteers; evaluation set: 4 volunteers.
- Five scenarios: glasses, no glasses, glasses at night, no glasses at night, sunglasses. Night footage is near-infrared.
- Four annotation files: drowsiness (normal / drowsy), eyes (normal / sleepy), head (normal / nodding / looking
  aside), mouth (normal / yawning / talking-laughing).
- Frame-level labelling is implied by the paper but not stated outright: inspect the annotation files first.
- Provided for the ACCV 2016 driver drowsiness workshop challenge; the papers do not say how to obtain it now.
  Contact the NTHU lab / paper authors. It is not in the harness's manifest format: write a small loader that
  converts the per-frame annotations into per-video or per-segment labels and then reuses `evaluation/`.
- Best use: score by scenario to show where the system breaks (sunglasses, night).

## YawDD (Yawning Detection Dataset)

Hosted on [IEEE DataPort](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) (checked). The
authors' citation (I recall Abtahi et al., 2014) is from memory: take the official one from that page.

- Set 1: 322 videos (3 or 4 per subject), camera under the front mirror. Set 2: 29 videos (one per subject), dash camera.
- 640x480, 30 fps AVI, no audio. **Unlabelled**: normal driving, talking/singing, silent, and yawning, separated by video (Set 1)
  or within one video (Set 2).
- Listed as open access with a free IEEE account, but the page is inconsistent about whether a subscription is needed. Check.
- This is a **yawning** dataset, not a drowsiness one: it cannot validate eye-closure/PERCLOS. Use it for the later mouth-aspect-ratio module.

## Running the baseline

```bash
# 1. Build a manifest (video,label[,subject])
python -m evaluation.make_manifest --root data/RLDD --mode rldd --out manifest.csv

# 2. Extract features once (slow: face landmarks + the original Haar logic on every frame)
python -m evaluation.extract_features --manifest manifest.csv --model face_landmarker.task --out traces/ --stride 2

# 3. Score as often as you like (fast: tune thresholds on training folds, then freeze)
python -m evaluation.evaluate --traces traces/ --out results/ --sweep
```

## What counts as a credible result

- Report recall and specificity **with confidence intervals**, false alarms per hour on alert drivers, and how many videos could not be scored.
- Show the comparison against the original logic (`legacy_raw`, `legacy_4frames`) on the same videos.
- State the label caveat (self-reported) and that results are video-level, not per-frame.
- Do not quote the synthetic-data numbers anywhere. They only prove the code runs.
