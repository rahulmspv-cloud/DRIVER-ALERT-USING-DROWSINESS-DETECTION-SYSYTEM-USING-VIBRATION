"""Turn videos into feature traces (one .npz per video) so metrics can be recomputed
and thresholds tuned without re-running the slow face-landmark step.

    python -m evaluation.extract_features --manifest manifest.csv --model face_landmarker.task --out traces/

manifest.csv columns:  video,label[,subject]
Each trace stores per-frame: t (s), ear (NaN = no face), haar_eyes (-1 no face, else eye count from the
ORIGINAL Haar logic), proc_ms (landmark+EAR time), haar_ms (Haar time).

Large datasets (RLDD is >100 GB): use --stride 2 or 3 and --max-seconds to cut work.
The detector is time-weighted, so a stride changes frame spacing but not the PERCLOS definition.
Keep stride * (1/fps) below Config.max_dt_s (0.5 s).
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np


def haar_eye_count(gray, face_cascade, eye_cascade) -> int:
    """Faithful copy of the original report's per-frame logic (chapter 11)."""
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return -1
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
    return int(len(eyes))


def process_video(path: Path, extractor, face_cascade, eye_cascade, stride: int, max_seconds: float):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t_l, ear_l, haar_l, pm_l, hm_l = [], [], [], [], []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / fps
        if max_seconds and t > max_seconds:
            break
        if idx % stride:
            continue

        t0 = time.perf_counter()
        ear = extractor.ear(frame, t)
        t1 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = haar_eye_count(gray, face_cascade, eye_cascade)
        t2 = time.perf_counter()

        t_l.append(t)
        ear_l.append(ear)
        haar_l.append(eyes)
        pm_l.append((t1 - t0) * 1000.0)
        hm_l.append((t2 - t1) * 1000.0)
    cap.release()
    return (np.array(t_l), np.array(ear_l), np.array(haar_l, dtype=np.int16),
            np.array(pm_l), np.array(hm_l), fps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--model", required=True, help="path to face_landmarker.task")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = whole video")
    args = ap.parse_args(argv)

    import cv2

    from drowsiness_adas.landmarks import FaceLandmarkExtractor

    if not hasattr(cv2, "CascadeClassifier"):
        raise SystemExit(
            f"OpenCV {cv2.__version__} has no CascadeClassifier (removed from the main module in OpenCV 5), "
            "which the legacy Haar baseline needs. Install:  pip install 'opencv-python<5'"
        )
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    host = f"{platform.platform()} / {platform.processor() or platform.machine()}"

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="") as fh:
        manifest = list(csv.DictReader(fh))

    index_rows = []
    for i, row in enumerate(manifest):
        video = Path(row["video"])
        name = f"{i:05d}_{video.stem}.npz"
        # a fresh extractor per video: MediaPipe VIDEO mode requires timestamps to restart from zero
        extractor = FaceLandmarkExtractor(args.model)
        try:
            t, ear, haar, pm, hm, fps = process_video(video, extractor, face_cascade, eye_cascade,
                                                      max(1, args.stride), args.max_seconds)
        finally:
            extractor.close()
        np.savez_compressed(args.out / name, t=t, ear=ear, haar_eyes=haar, proc_ms=pm, haar_ms=hm,
                            fps=fps, synthetic=False, host=host)
        index_rows.append({"trace": name, "label": row["label"], "subject": row.get("subject", ""),
                           "video": str(video)})
        face_pct = 100.0 * float(np.mean(~np.isnan(ear))) if len(ear) else 0.0
        print(f"[{i + 1}/{len(manifest)}] {video.name}: {len(t)} frames, face found in {face_pct:.0f}%")

    with open(args.out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trace", "label", "subject", "video"])
        w.writeheader()
        w.writerows(index_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
