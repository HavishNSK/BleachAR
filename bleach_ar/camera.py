"""Webcam input.

Frames are read on a background thread so the render loop always gets the
newest frame instead of a stale one from the driver's buffer. ``source`` can
be a camera index, a video file path, or ``"synthetic"`` (an animated test
pattern used for headless testing when no camera is available).
"""

from __future__ import annotations

import platform
import threading
import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


def _open_capture(source):
    if isinstance(source, int):
        if platform.system() == "Darwin":
            return cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


def camera_help(source) -> str:
    msg = [f"Could not read frames from camera {source!r}."]
    if platform.system() == "Darwin":
        msg.append(
            "macOS: allow camera access for your terminal app in System Settings > "
            "Privacy & Security > Camera, then restart the terminal."
        )
    elif platform.system() == "Windows":
        msg.append("Windows: check Settings > Privacy > Camera and close other apps using the webcam.")
    other = 1 if source == 0 else 0
    msg.append(f"Try another index with --camera {other}, or test without a camera using --demo.")
    return "\n".join(msg)


class SyntheticSource:
    """Moving gradient test pattern; no people in it, so tracking finds nothing."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self._xx, self._yy = xx / width, yy / height
        self._t0 = time.perf_counter()

    def read(self):
        t = time.perf_counter() - self._t0
        base = 40 + 30 * np.sin(6.0 * self._xx + t) * np.cos(4.0 * self._yy - 0.7 * t)
        frame = np.empty((self.height, self.width, 3), np.uint8)
        frame[..., 0] = np.clip(base + 30, 0, 255)
        frame[..., 1] = np.clip(base + 10, 0, 255)
        frame[..., 2] = np.clip(base, 0, 255)
        return True, frame

    def release(self):
        pass


class Camera:
    def __init__(self, source=0, width=1280, height=720, fps=30, mirror=True):
        self.source = source
        self.mirror = mirror
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self._thread = None
        self.failures = 0

        if source == "synthetic":
            self.cap = SyntheticSource(width, height)
            self.is_file = False
        else:
            self.cap = _open_capture(source)
            if not self.cap.isOpened():
                raise CameraError(camera_help(source))
            self.is_file = isinstance(source, str)
            if not self.is_file:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._file_fps = fps
        if self.is_file:
            self._file_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps

        # Fail fast with a helpful message instead of showing a black window.
        ok, frame = False, None
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok or frame is None:
            self.cap.release()
            raise CameraError(camera_help(source))
        self.height, self.width = frame.shape[:2]
        self._store(frame)

    def _store(self, frame):
        # Never publish the driver's buffer: some backends reuse it on the next read.
        frame = cv2.flip(frame, 1) if self.mirror else frame.copy()
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def start(self) -> "Camera":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        period = 1.0 / max(1.0, self._file_fps) if (self.is_file or self.source == "synthetic") else 0.0
        while self._running:
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
                    continue
                self.failures += 1
                time.sleep(0.02)
                continue
            self.failures = 0
            self._store(frame)
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def read(self):
        """Returns (frame_id, frame). frame_id increases when a new frame arrives."""
        with self._lock:
            return self._frame_id, self._frame

    @property
    def lost(self) -> bool:
        return self.failures > 30

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                # Blocked in read(); releasing now could crash in the driver.
                print("[camera] capture thread did not stop in time; leaving it to exit with the process.")
                return
        self.cap.release()
