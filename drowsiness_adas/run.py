"""Run the detector on a webcam or a video file.

    python -m drowsiness_adas.run --model face_landmarker.task --source 0
    python -m drowsiness_adas.run --model face_landmarker.task --source clip.mp4 --no-display --log out.csv
    python -m drowsiness_adas.run --model face_landmarker.task --actuator gpio     # on the Raspberry Pi
"""

from __future__ import annotations

import argparse
import csv
import time

from .actuators import GpioActuator, MockActuator
from .config import Config
from .detector import DrowsinessDetector
from .landmarks import FaceLandmarkExtractor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to face_landmarker.task")
    ap.add_argument("--source", default="0", help="camera index or video file path")
    ap.add_argument("--actuator", choices=["mock", "gpio"], default="mock")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--log", help="write a per-frame CSV here")
    args = ap.parse_args(argv)

    import cv2

    live = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if live else args.source)
    if not cap.isOpened():
        print(f"cannot open source {args.source!r}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    extractor = FaceLandmarkExtractor(args.model)
    detector = DrowsinessDetector(Config())
    actuator = GpioActuator() if args.actuator == "gpio" else MockActuator(verbose=True)

    log_file = open(args.log, "w", newline="") if args.log else None
    writer = csv.writer(log_file) if log_file else None
    if writer:
        writer.writerow(["t", "ear", "closed", "perclos", "closure_s", "level", "status", "proc_ms"])

    t0 = time.monotonic()
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (time.monotonic() - t0) if live else idx / fps
            idx += 1

            started = time.perf_counter()
            ear = extractor.ear(frame, t)
            out = detector.update(t, ear)
            proc_ms = (time.perf_counter() - started) * 1000.0

            actuator.apply(out)
            if writer:
                writer.writerow([f"{t:.3f}", f"{out.ear:.4f}", int(out.closed), f"{out.perclos:.4f}",
                                 f"{out.closure_s:.2f}", int(out.level), out.status.value, f"{proc_ms:.1f}"])

            if not args.no_display:
                label = f"{out.level.name} | {out.status.value} | EAR {out.ear:.2f} | PERCLOS {out.perclos:.2f}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("drowsiness_adas", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Esc
                    break
    finally:
        actuator.close()
        extractor.close()
        cap.release()
        if log_file:
            log_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
