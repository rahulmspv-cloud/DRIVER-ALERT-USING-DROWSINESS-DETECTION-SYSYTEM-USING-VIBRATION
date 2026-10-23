"""Camera frame -> EAR, using MediaPipe's FaceLandmarker (Tasks API).

Setup: download the model bundle once and pass its path::

    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.float16.task

Landmark indices are those of the 468/478-point MediaPipe face mesh topology.
Coordinates are converted to PIXELS before computing EAR, because the
normalised x/y values have different scales and would distort the ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from .signals import NAN, mean_ear

# p1 (corner), p2, p3 (upper lid), p4 (corner), p5, p6 (lower lid)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)


class FaceLandmarkExtractor:
    def __init__(self, model_path: Union[str, Path]) -> None:
        import mediapipe as mp  # imported lazily so the core package works without it
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"FaceLandmarker model not found: {model_path}. See the module docstring for the download URL."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def ear(self, frame_bgr: np.ndarray, t_seconds: float) -> float:
        """Mean EAR for the first face in the frame, or NaN if no face is found."""
        import cv2

        # MediaPipe VIDEO mode needs strictly increasing integer millisecond timestamps.
        ts_ms = max(int(t_seconds * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks:
            return NAN

        h, w = frame_bgr.shape[:2]
        lm = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=float)
        return mean_ear(pts[list(LEFT_EYE)], pts[list(RIGHT_EYE)])

    def close(self) -> None:
        self._landmarker.close()
