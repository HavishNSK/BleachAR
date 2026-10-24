"""Rule-based gestures from MediaPipe hand landmarks.

No model training is involved: each finger is classified as extended or
curled by comparing the wrist-to-tip distance with the wrist-to-PIP distance,
which is scale and rotation invariant. Static poses must be stable for
``stable_frames`` frames and then held for ``hold_seconds`` before they fire,
and every gesture trigger respects a global cooldown, so holding a pose does
not retrigger every frame.

Gestures produced (all names usable in ``config.gestures.bindings``):
  fist          - all four fingers curled (also used as the sword grip)
  open_palm     - all fingers extended, palm toward camera (hold to fire Cero)
  two_fingers   - index + middle extended, others curled (hold for Shikai)
  point         - only index extended
  both_palms_up - both hands open above the shoulders (hold for Bankai)
  swing         - fast sword-tip motion (fires Getsuga Tensho)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .tracking import L_SHOULDER, R_SHOULDER, TrackingState

WRIST = 0
FINGERS = {  # (mcp, pip, tip)
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
THUMB_TIP, THUMB_IP = 4, 3

HELD_GESTURES = ("open_palm", "two_fingers", "both_palms_up", "point", "fist")


def finger_states(hand: np.ndarray, extend_ratio: float = 1.12) -> Dict[str, bool]:
    wrist = hand[WRIST, :2]
    out = {}
    for name, (_, pip, tip) in FINGERS.items():
        d_tip = np.linalg.norm(hand[tip, :2] - wrist)
        d_pip = np.linalg.norm(hand[pip, :2] - wrist)
        out[name] = bool(d_tip > d_pip * extend_ratio)
    # Thumb: extended when its tip is far from the index MCP relative to hand size.
    palm = np.linalg.norm(hand[9, :2] - wrist) + 1e-6
    out["thumb"] = bool(np.linalg.norm(hand[THUMB_TIP, :2] - hand[5, :2]) / palm > 0.75)
    return out


def classify_hand(hand: np.ndarray, extend_ratio: float = 1.12) -> str:
    f = finger_states(hand, extend_ratio)
    four = [f["index"], f["middle"], f["ring"], f["pinky"]]
    if not any(four):
        return "fist"
    if all(four):
        return "open_palm"
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "two_fingers"
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        return "point"
    return "other"


def hand_size(hand: np.ndarray) -> float:
    return float(np.linalg.norm(hand[9, :2] - hand[WRIST, :2]))


@dataclass
class _Stable:
    name: Optional[str] = None
    count: int = 0
    since: float = 0.0
    fired: bool = False


@dataclass
class GestureFrame:
    """What the recogniser saw this frame (also shown in calibration mode)."""

    per_hand: Dict[str, str] = field(default_factory=dict)
    fingers: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    held: Dict[str, float] = field(default_factory=dict)  # gesture -> hold progress 0..1
    triggered: List[str] = field(default_factory=list)


class GestureRecognizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self._stable: Dict[str, _Stable] = {}
        self._last_fire = -1e9
        self.last = GestureFrame()

    def _global_ready(self, now: float) -> bool:
        return now - self._last_fire >= self.cfg["global_cooldown"]

    def _track_held(self, key: str, name: Optional[str], now: float, frame: GestureFrame) -> Optional[str]:
        """Debounce a static pose; returns its name once when the hold completes."""
        st = self._stable.setdefault(key, _Stable())
        if name != st.name:
            st.name, st.count, st.since, st.fired = name, 1, now, False
            return None
        st.count += 1
        if name is None or st.count < self.cfg["stable_frames"]:
            return None
        if st.count == self.cfg["stable_frames"]:
            st.since = now
        progress = min(1.0, (now - st.since) / max(1e-3, self.cfg["hold_seconds"]))
        frame.held[name] = max(frame.held.get(name, 0.0), progress)
        if progress >= 1.0 and not st.fired:
            st.fired = True  # must release the pose before it can fire again
            return name
        return None

    def update(self, tracking: TrackingState, swing_speed: float, now: Optional[float] = None) -> GestureFrame:
        now = time.perf_counter() if now is None else now
        frame = GestureFrame()
        ratio = self.cfg["finger_extend_ratio"]
        candidates: List[str] = []

        # ``ok`` includes the short hold after a dropout, so one missed frame
        # does not reset a nearly finished hold.
        visible = {side: p.points for side, p in tracking.hands.items() if p.ok}
        for side, pts in visible.items():
            frame.per_hand[side] = classify_hand(pts, ratio)
            frame.fingers[side] = finger_states(pts, ratio)

        # Two-handed pose: both palms open with wrists above shoulder line.
        both_up = None
        if len(visible) == 2 and all(g == "open_palm" for g in frame.per_hand.values()):
            if tracking.pose.ok:
                shoulder_y = min(tracking.pose.points[L_SHOULDER, 1], tracking.pose.points[R_SHOULDER, 1])
            else:
                shoulder_y = tracking.height * 0.45
            if all(pts[WRIST, 1] < shoulder_y for pts in visible.values()):
                both_up = "both_palms_up"
        fired = self._track_held("both", both_up, now, frame)
        if fired:
            candidates.append(fired)

        for side in ("left", "right"):
            g = frame.per_hand.get(side)
            if both_up:
                g = None  # don't also fire single-hand open_palm
            fired = self._track_held(side, g if g in HELD_GESTURES else None, now, frame)
            if fired:
                candidates.append(fired)

        if swing_speed >= self.cfg["swing_speed"]:
            candidates.append("swing")

        self.last = frame
        if not self.enabled:
            return frame

        bound = set(self.cfg["bindings"].values())
        if self._global_ready(now):
            frame.triggered = [name for name in candidates if name in bound]
        return frame

    def confirm(self, now: Optional[float] = None) -> None:
        """Start the global cooldown; call only when a gesture actually fired an ability."""
        self._last_fire = time.perf_counter() if now is None else now

    def ability_for(self, gesture: str) -> Optional[str]:
        for ability, g in self.cfg["bindings"].items():
            if g == gesture:
                return ability
        return None

    def reset(self) -> None:
        self._stable.clear()
        self._last_fire = -1e9
