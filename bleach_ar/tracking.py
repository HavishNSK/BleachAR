"""Face, eye, hand and body tracking (MediaPipe Holistic).

MediaPipe runs on a downscaled copy of the frame; landmarks come back
normalised, so they are scaled straight to render-resolution pixels. Each
part (face, pose, hands) is smoothed with an exponential moving average and
held for a short grace period when tracking drops out, with a ``confidence``
that fades from 1 to 0 so effects can fade instead of popping.
"""

from __future__ import annotations

import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import cv2
import numpy as np

# Pose landmark indices (MediaPipe PoseLandmark).
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 13, 14, 15, 16
L_HIP, R_HIP = 23, 24

# Face mesh indices (with refine_landmarks / refine_face_landmarks).
LEFT_IRIS, RIGHT_IRIS = 468, 473  # subject's right / left in the mirrored image
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


@dataclass
class Part:
    """Smoothed landmark set for one tracked part, in render pixels (N, 3)."""

    points: Optional[np.ndarray] = None
    confidence: float = 0.0
    last_seen: float = 0.0
    visible: bool = False

    @property
    def ok(self) -> bool:
        return self.points is not None and self.confidence > 0.0


@dataclass
class TrackingState:
    width: int = 0
    height: int = 0
    face: Part = field(default_factory=Part)
    pose: Part = field(default_factory=Part)
    # Hands keyed by the side of the *image* they appear on after mirroring:
    # "right" is the user's right hand when the camera feed is mirrored.
    hands: Dict[str, Part] = field(default_factory=lambda: {"left": Part(), "right": Part()})
    mask: Optional[np.ndarray] = None  # float32 0..1 at tracking resolution
    inference_ms: float = 0.0

    def snapshot(self) -> "TrackingState":
        """Shallow copy safe to hand to another thread: Part objects are copied,
        and their arrays are never mutated in place (only replaced)."""
        return TrackingState(self.width, self.height, replace(self.face), replace(self.pose),
                             {k: replace(v) for k, v in self.hands.items()}, self.mask, self.inference_ms)

    def status(self) -> Dict[str, bool]:
        return {
            "face": self.face.visible,
            "body": self.pose.visible,
            "hands": self.hands["left"].visible or self.hands["right"].visible,
        }


_POSE_MODELS = {0: "pose_landmark_lite.tflite", 2: "pose_landmark_heavy.tflite"}


def ensure_pose_model(mp, model_complexity: int) -> int:
    """Make sure the pose model for ``model_complexity`` is on disk.

    MediaPipe only bundles the complexity-1 model and downloads the others on
    first use with ``urllib``, which fails on python.org macOS builds that
    have no root certificates installed. Download it here using certifi's
    certificates instead; if that fails, fall back to the bundled model.
    Returns the complexity that can actually be used.
    """
    name = _POSE_MODELS.get(model_complexity)
    if name is None:
        return model_complexity
    path = os.path.join(os.path.dirname(mp.__file__), "modules", "pose_landmark", name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return model_complexity
    url = "https://storage.googleapis.com/mediapipe-assets/" + name
    print(f"Downloading MediaPipe {name} (one-time, a few MB)...")
    try:
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urllib.request.urlopen(url, context=context, timeout=30) as resp:
            data = resp.read()
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        return model_complexity
    except (OSError, urllib.error.URLError) as exc:
        print(f"[tracking] could not download {name} ({exc}); using the bundled full pose model instead.")
        return 1


def _landmarks_to_array(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks.landmark], dtype=np.float32)


class Tracker:
    def __init__(self, model_complexity=0, segmentation=True, tracking_width=640,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 smoothing=0.45, hold_seconds=0.35):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise SystemExit(
                "MediaPipe is not installed. Run: pip install -r requirements.txt\n"
                "(MediaPipe 0.10.21 needs Python 3.9-3.12.)"
            ) from exc
        if not hasattr(mp, "solutions"):
            raise SystemExit(
                f"mediapipe {mp.__version__} no longer ships the 'solutions' API this app uses. "
                "Install the pinned version: pip install mediapipe==0.10.21"
            )
        self.tracking_width = tracking_width
        self.segmentation = segmentation
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.hold_seconds = hold_seconds
        model_complexity = ensure_pose_model(mp, model_complexity)
        self.model_complexity = model_complexity
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            refine_face_landmarks=True,
            enable_segmentation=segmentation,
            smooth_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.state = TrackingState()

    def _update_part(self, part: Part, new: Optional[np.ndarray], now: float) -> None:
        if new is not None and not np.isfinite(new).all():
            new = None  # treat corrupt landmarks as a dropped frame
        if new is not None:
            if part.points is None or part.points.shape != new.shape or not part.visible:
                part.points = new
            else:
                a = self.smoothing
                # Move faster when the jump is large so fast swings are not smeared.
                jump = float(np.median(np.linalg.norm(new[:, :2] - part.points[:, :2], axis=1)))
                if jump > 25:
                    a *= 0.35
                part.points = part.points * a + new * (1.0 - a)
            part.visible = True
            part.last_seen = now
            part.confidence = 1.0
        else:
            part.visible = False
            if part.points is not None:
                age = now - part.last_seen
                part.confidence = max(0.0, 1.0 - age / self.hold_seconds) if self.hold_seconds > 0 else 0.0
                if part.confidence <= 0.0:
                    part.points = None

    def process(self, frame_bgr: np.ndarray) -> TrackingState:
        """Run tracking on a render-resolution BGR frame."""
        h, w = frame_bgr.shape[:2]
        tw = min(self.tracking_width, w)
        th = int(round(h * tw / w))
        small = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA) if tw != w else frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t0 = time.perf_counter()
        results = self._holistic.process(rgb)
        now = time.perf_counter()

        st = self.state
        st.width, st.height = w, h
        st.inference_ms = (now - t0) * 1000.0
        self._update_part(st.face, _landmarks_to_array(results.face_landmarks, w, h)
                          if results.face_landmarks else None, now)
        self._update_part(st.pose, _landmarks_to_array(results.pose_landmarks, w, h)
                          if results.pose_landmarks else None, now)
        # Holistic's left/right labels flip depending on whether the image is
        # mirrored, so hands are keyed by which side of the screen they are on.
        found = [_landmarks_to_array(lms, w, h) for lms in
                 (results.left_hand_landmarks, results.right_hand_landmarks) if lms]
        sides = {"left": None, "right": None}
        if len(found) == 2:
            found.sort(key=lambda pts: pts[0, 0])
            sides["left"], sides["right"] = found
        elif len(found) == 1:
            pts = found[0]
            if st.pose.ok:
                mid = (st.pose.points[L_SHOULDER, 0] + st.pose.points[R_SHOULDER, 0]) / 2.0
            else:
                mid = w / 2.0
            sides["right" if pts[0, 0] >= mid else "left"] = pts
        for side, pts in sides.items():
            self._update_part(st.hands[side], pts, now)
        if self.segmentation and results.segmentation_mask is not None and st.pose.visible:
            st.mask = results.segmentation_mask.copy()  # MediaPipe reuses this buffer
        elif not st.pose.ok:
            st.mask = None
        return st

    def close(self) -> None:
        self._holistic.close()


class AsyncTracker:
    """Runs a Tracker on a worker thread so inference overlaps rendering.

    ``submit`` hands over the newest frame (older pending frames are dropped);
    ``state`` is whatever the latest finished inference produced. Landmarks
    can trail the displayed image by about one inference (~25 ms), which is
    the price for not blocking the render loop.
    """

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self._cond = threading.Condition()
        self._pending: Optional[np.ndarray] = None
        self._running = True
        self.last_ms = 0.0
        self._fresh = False
        self._published = TrackingState()
        self._thread = threading.Thread(target=self._loop, name="tracking", daemon=True)
        self._thread.start()

    @property
    def state(self) -> TrackingState:
        with self._cond:
            return self._published

    @property
    def model_complexity(self) -> int:
        return self.tracker.model_complexity

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._cond:
            self._pending = frame_bgr
            self._cond.notify()

    def take_timing(self) -> Optional[float]:
        """Inference time of a result finished since the last call, else None."""
        with self._cond:
            if not self._fresh:
                return None
            self._fresh = False
            return self.last_ms

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and self._running:
                    self._cond.wait()
                if not self._running:
                    return
                frame, self._pending = self._pending, None
            snap = self.tracker.process(frame).snapshot()
            with self._cond:
                self._published = snap
                self.last_ms = snap.inference_ms
                self._fresh = True

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            # Still inside process(); closing the graph now could crash natively.
            print("[tracking] worker did not stop in time; leaving it to exit with the process.")
            return
        self.tracker.close()
