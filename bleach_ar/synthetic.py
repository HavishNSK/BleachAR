"""Synthetic person for headless tests and screenshots (no webcam needed).

``make_tracking`` builds a plausible TrackingState: a face mesh with the
landmarks the effects use (oval, eye contours, irises), a pose skeleton and
one or two hands in a chosen gesture. ``make_frame`` draws a matching grey
figure so screenshots show where effects sit relative to the "body".
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .tracking import EYE_A, EYE_B, FACE_OVAL, Part, TrackingState


def _face_points(cx: float, cy: float, fw: float, fh: float) -> np.ndarray:
    pts = np.zeros((478, 3), np.float32)
    pts[:, 0], pts[:, 1] = cx, cy
    for i, idx in enumerate(FACE_OVAL):
        a = -math.pi / 2 + 2 * math.pi * i / len(FACE_OVAL)
        pts[idx, :2] = (cx + math.cos(a) * fw / 2, cy + math.sin(a) * fh / 2)
    eye_dx, eye_y = fw * 0.2, cy - fh * 0.08
    ew, eh = fw * 0.16, fh * 0.055
    # Index sets ordered around each eye; EYE_A sits on screen-left.
    for contour, ex in ((EYE_A, cx - eye_dx), (EYE_B, cx + eye_dx)):
        for i, idx in enumerate(contour):
            a = math.pi + 2 * math.pi * i / len(contour)
            pts[idx, :2] = (ex + math.cos(a) * ew / 2, eye_y + math.sin(a) * eh / 2)
    # Eyelid landmarks used for blink detection.
    for top, bot, inner, outer, ex in ((159, 145, 133, 33, cx - eye_dx), (386, 374, 362, 263, cx + eye_dx)):
        pts[top, :2] = (ex, eye_y - eh / 2)
        pts[bot, :2] = (ex, eye_y + eh / 2)
        pts[inner, :2] = (ex + (ew / 2 if ex < cx else -ew / 2), eye_y)
        pts[outer, :2] = (ex - (ew / 2 if ex < cx else -ew / 2), eye_y)
    ir = eh * 0.45
    for base, ex in ((468, cx - eye_dx), (473, cx + eye_dx)):
        pts[base, :2] = (ex, eye_y)
        for k, (dx, dy) in enumerate(((1, 0), (0, -1), (-1, 0), (0, 1))):
            pts[base + 1 + k, :2] = (ex + dx * ir, eye_y + dy * ir)
    pts[10, :2] = (cx, cy - fh / 2)  # forehead
    pts[152, :2] = (cx, cy + fh / 2)  # chin
    pts[1, :2] = (cx, cy + fh * 0.08)  # nose tip
    pts[13, :2] = (cx, cy + fh * 0.25)  # upper lip
    pts[14, :2] = (cx, cy + fh * 0.29)  # lower lip
    return pts


def _hand_points(wrist, angle_deg: float, size: float, gesture: str) -> np.ndarray:
    """21 hand landmarks. ``angle_deg`` is the wrist->middle-knuckle direction."""
    a = math.radians(angle_deg)
    fwd = np.array([math.cos(a), math.sin(a)], np.float32)
    side = np.array([-fwd[1], fwd[0]], np.float32)
    w = np.asarray(wrist, np.float32)
    pts = np.zeros((21, 3), np.float32)
    pts[0, :2] = w
    extended = {
        "fist": (False, False, False, False),
        "open_palm": (True, True, True, True),
        "two_fingers": (True, True, False, False),
        "point": (True, False, False, False),
    }[gesture]
    offsets = (-0.35, -0.1, 0.12, 0.32)  # index..pinky across the palm
    for f, (mcp, off, ext) in enumerate(zip((5, 9, 13, 17), offsets, extended)):
        base = w + fwd * size + side * off * size
        pts[mcp, :2] = base
        if ext:
            for j in range(1, 4):
                pts[mcp + j, :2] = base + fwd * size * 0.38 * j
        else:  # curled back toward the palm
            pts[mcp + 1, :2] = base + fwd * size * 0.3
            pts[mcp + 2, :2] = base + fwd * size * 0.1 - side * 0.02 * size
            pts[mcp + 3, :2] = base - fwd * size * 0.15
    thumb_dir = fwd * 0.4 - side * 0.9
    for j, k in enumerate((1, 2, 3, 4), start=1):
        pts[k, :2] = w + thumb_dir * size * 0.35 * j
    return pts


def make_tracking(w: int = 960, h: int = 540, t: float = 0.0, gesture: str = "fist",
                  second_hand: Optional[str] = None, swing: float = 0.5,
                  hand_angle: Optional[float] = None) -> TrackingState:
    """``swing`` in 0..1 sweeps the right hand in an arc (for slash tests)."""
    st = TrackingState(width=w, height=h)
    cx = w * 0.5 + 10 * math.sin(t)
    fw, fh = h * 0.24, h * 0.32
    fy = h * 0.3
    st.face = Part(_face_points(cx, fy, fw, fh), 1.0, t, True)
    pose = np.zeros((33, 3), np.float32)
    sh_y, sh_dx = fy + fh * 0.85, fw * 1.1
    pose[0, :2] = (cx, fy)
    pose[11, :2] = (cx + sh_dx, sh_y)  # subject's left shoulder = screen right (mirrored)
    pose[12, :2] = (cx - sh_dx, sh_y)
    pose[23, :2] = (cx + sh_dx * 0.6, h * 1.05)
    pose[24, :2] = (cx - sh_dx * 0.6, h * 1.05)
    sweep = -60 + 120 * swing
    ang = math.radians(sweep - 90)
    wrist_r = np.array([cx + sh_dx + 40 + math.cos(ang) * 150, sh_y - 20 + math.sin(ang) * 150 + 120])
    pose[13, :2] = (cx + sh_dx + 30, sh_y + 110)
    pose[15, :2] = wrist_r
    pose[14, :2] = (cx - sh_dx - 30, sh_y + 110)
    pose[16, :2] = (cx - sh_dx - 60, sh_y + 60)
    st.pose = Part(pose, 1.0, t, True)
    size = fw * 0.42
    st.hands["right"] = Part(_hand_points(wrist_r, hand_angle if hand_angle is not None else -100 + sweep * 0.5,
                                          size, gesture), 1.0, t, True)
    if second_hand:
        st.hands["left"] = Part(_hand_points(pose[16, :2], -80, size, second_hand), 1.0, t, True)
    mask = np.zeros((h // 4, w // 4), np.float32)
    s = 0.25
    cv2.ellipse(mask, (int(cx * s), int(fy * s)), (int(fw * s / 2), int(fh * s / 2)), 0, 0, 360, 1.0, -1)
    body = np.array([[cx - sh_dx, sh_y], [cx + sh_dx, sh_y], [cx + sh_dx * 0.8, h], [cx - sh_dx * 0.8, h]]) * s
    cv2.fillPoly(mask, [body.astype(np.int32)], 1.0)
    cv2.rectangle(mask, (int((cx - fw * 0.18) * s), int((fy + fh * 0.4) * s)),
                  (int((cx + fw * 0.18) * s), int(sh_y * s)), 1.0, -1)
    st.mask = mask
    return st


def make_frame(st: TrackingState) -> np.ndarray:
    """A dim room with a grey figure matching ``st`` (BGR uint8)."""
    w, h = st.width, st.height
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = (52, 48, 44)
    cv2.rectangle(frame, (0, int(h * 0.78)), (w, h), (38, 40, 42), -1)
    if st.mask is not None:
        m = cv2.resize(st.mask, (w, h))
        frame[m > 0.5] = (120, 125, 135)
    if st.face.ok:
        pts = st.face.points
        for idx in (468, 473):
            cv2.circle(frame, (int(pts[idx, 0]), int(pts[idx, 1])), 6, (40, 30, 30), -1)
    for p in st.hands.values():
        if p.ok:
            for a, b in ((0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (5, 8), (9, 12), (13, 16), (17, 20), (1, 4)):
                cv2.line(frame, tuple(p.points[a, :2].astype(int)), tuple(p.points[b, :2].astype(int)),
                         (140, 160, 190), 6)
    return frame
