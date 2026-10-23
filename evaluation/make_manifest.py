"""Build manifest.csv (video,label,subject) from a folder of videos.

    # label = name of the parent folder (alert/ and drowsy/ subfolders)
    python -m evaluation.make_manifest --root data/mine --mode parent --out manifest.csv

    # RLDD-style layout (ASSUMPTION, verify against the dataset's own README before trusting it):
    # one folder per subject, videos named 0 / 5 / 10 for alert / low-vigilant / drowsy
    python -m evaluation.make_manifest --root data/RLDD --mode rldd --out manifest.csv

NTHU-DDD has per-frame annotation files and needs its own loader; this helper only does video-level labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
RLDD_LABELS = {"0": "alert", "10": "drowsy"}  # 5 (low vigilant) is skipped: ambiguous for a binary task


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--mode", choices=["parent", "rldd"], default="parent")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    rows, skipped = [], 0
    for p in sorted(args.root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        if args.mode == "parent":
            rows.append({"video": str(p), "label": p.parent.name, "subject": ""})
        else:
            label = RLDD_LABELS.get(p.stem)
            if label is None:
                skipped += 1
                continue
            rows.append({"video": str(p), "label": label, "subject": p.parent.name})

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "label", "subject"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out} ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
