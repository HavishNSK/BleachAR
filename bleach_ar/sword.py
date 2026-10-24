"""The Zanpakuto: a 2D sword pose estimated from one hand, and its rendering.

Summoning: in sword mode the blade appears when a hand has been classified
"fist" for ``SUMMON_FRAMES`` frames and stays while that hand remains a fist.
It is only dismissed after ``DISMISS_FRAMES`` consecutive non-fist frames
spanning at least ``DISMISS_SECONDS`` (one misclassified frame never makes it
flicker), and it fades in/out over ``FADE_SECONDS`` through
``ctx.sword.confidence``. In "hand" mode any visible hand qualifies.

Pose heuristic (an honest 2D approximation, no depth is inferred):

* Grip: the centroid of the four finger knuckles (MCP 5, 9, 13, 17), pulled
  30% of the way toward the wrist, i.e. the channel inside a closed fist.
* Direction for a fist: a held handle runs diagonally across the palm, from
  the heel of the hand (pinky side, near the wrist) out past the index
  knuckle. In 2D that is the knuckle axis (pinky MCP -> index MCP) tilted
  toward the forearm by ``FOREARM_TILT`` (about 40 degrees). The forearm is
  the pose elbow -> wrist of the matching arm (pose wrist nearest the hand
  wrist) blended with the hand's own wrist -> middle-MCP vector, which keeps
  the wrist bend. When the knuckle span collapses in 2D (fist seen edge-on)
  the knuckle axis is unreliable, so the direction slides to the forearm.
  When the forearm is foreshortened (pointing at the camera) its weight
  drops and the knuckle axis dominates.
* Direction for any other hand (and in "hand" mode): wrist -> middle MCP.
* The unit direction goes through a One-Euro filter: heavy smoothing when
  the hand is still, little lag when it rotates fast. The grip is not
  filtered (the tracker already smooths landmarks), so the handle stays
  locked in the fist and swing speed is not damped.

Rendering: per-style RGBA art (``assets.sword_images[style]``, then legacy
``assets.sword_image``) warped into an ROI, else procedural shapes. The
fingers are composited back over the handle from the camera pixels (hand
mask from the 21 landmarks), and ``ctx.sword_sink`` lowers the blade into
the ground while fading and narrowing it.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import fx
from .abilities import Context, SwordPose
from .config import PROJECT_ROOT
from .gestures import classify_hand, hand_size
from .tracking import Part

MCPS = (5, 9, 13, 17)
FINGERTIPS = (4, 8, 12, 16, 20)
# Weight of the forearm direction against the (unit) knuckle axis for a
# face-on fist: atan(0.85) ~ 40 degrees, the diagonal of a real grip.
FOREARM_TILT = 0.85
SUMMON_FRAMES = 2
SUMMON_SECONDS = 2 / 30  # and at least this long, so the delay doesn't depend on FPS
DISMISS_FRAMES = 3
DISMISS_SECONDS = 0.12
FADE_SECONDS = 0.12
# Landmark chains used for the finger-occlusion mask.
_FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
_PALM = (0, 1, 5, 9, 13, 17)
SINK_DROP = 0.6  # fraction of the frame height the blade descends at sink = 1
MOTE_COLOR = (235, 210, 255)  # pink-white (BGR)

_warned: set = set()


def _finite(part: Part) -> bool:
    return part.points is not None and bool(np.isfinite(part.points[:, :2]).all())


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(msg)


# --- shared blade geometry (blade-local units: u along blade, v across, x length) ---
# Also used by tools/make_assets.py so the art lines up with the energy glow.
# Image convention: blade up, the image spans u in [IMG_U0, IMG_U1] bottom to
# top with square pixels, and v = 0 is the horizontal centre.
IMG_U0, IMG_U1 = -0.2, 1.0


def katana_profile(n: int = 18):
    """(blade outline, shinogi outline, edge polyline, hamon polyline) in uv."""
    s = np.linspace(0.0, 1.0, n)
    u = 0.11 + 0.89 * s
    curve = 0.035 * s ** 2
    half = 0.017 - 0.005 * s
    back = np.stack([u, curve + half], 1)
    edge = np.stack([u, curve - half], 1)
    # Kissaki: the edge sweeps up to meet the back at the tip.
    k = s > 0.9
    edge[k, 1] = (curve - half + 2 * half * np.clip((s - 0.9) / 0.1, 0.0, 1.0) ** 2.2)[k]
    tip = np.array([[1.0, curve[-1] + half[-1] * 0.6]])
    blade = np.concatenate([back, tip, edge[::-1]], 0)
    shinogi = np.concatenate([back[:-2], (np.stack([u, curve + half * 0.2], 1))[:-2][::-1]], 0)
    hamon_v = curve - half + 2 * half * (0.32 + 0.12 * np.sin(s * 37.0) * (1 - s))
    hamon = np.stack([u, hamon_v], 1)[:-2]
    return blade, shinogi, edge[:-1], hamon


CLEAVER_BACK_V, CLEAVER_EDGE_V = 0.06, -0.115
CLEAVER_BLADE = np.array([[0.07, CLEAVER_BACK_V], [0.985, CLEAVER_BACK_V], [1.0, CLEAVER_BACK_V - 0.02],
                          [0.9, CLEAVER_EDGE_V * 0.55], [0.8, CLEAVER_EDGE_V], [0.07, CLEAVER_EDGE_V]])
CLEAVER_EDGE = np.array([[0.07, CLEAVER_EDGE_V], [0.8, CLEAVER_EDGE_V], [0.9, CLEAVER_EDGE_V * 0.55],
                         [1.0, CLEAVER_BACK_V - 0.02]])


def dark_profile(n: int = 10) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    u = 0.12 + 0.86 * s
    half = 0.02 - 0.006 * s
    return np.concatenate([np.stack([u, half], 1), [[1.0, 0.004]], np.stack([u, -half], 1)[::-1]], 0)


class OneEuro:
    """One-Euro filter for small vectors (Casiez et al. 2012).

    Cutoff = ``min_cutoff + beta * |dx/dt|`` (Hz): smooth when slow, nearly
    lag-free when fast. For a unit direction |dx/dt| is the angular speed
    in rad/s.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.x is None or dt <= 1e-5:
            self.x = x.copy()
            self.dx = np.zeros_like(x)
            return self.x
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        return self.x


class Zanpakuto:
    """Estimates ``ctx.sword`` from the tracked hands and draws the blade."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        g = cfg.get("gestures", {})
        self.extend_ratio = float(g.get("finger_extend_ratio", 1.12))
        # Sparks fire a bit below the Getsuga swing threshold so a slash that
        # fires the ability also gets its own clash of sparks.
        self.spark_speed = 0.75 * float(g.get("swing_speed", 2.4))
        self.trail = fx.Trail(duration=0.2, max_samples=20)
        assets = cfg.get("assets", {}) or {}
        # Per-style art first; legacy ``sword_image`` (unknown layout) is the
        # fallback for any style without its own; else procedural.
        self.images: Dict[str, List[np.ndarray]] = {}
        for style, raw in (assets.get("sword_images") or {}).items():
            img = self._load_image(raw)
            if img is not None:
                self.images[str(style)] = self._mips(img)
        legacy = self._load_image(assets.get("sword_image", ""))
        self.image: Optional[List[np.ndarray]] = self._mips(legacy) if legacy is not None else None
        self.dir_filter = OneEuro(min_cutoff=1.0, beta=1.5, d_cutoff=1.0)
        self._vscale = 1.0
        self.reset()

    # --- setup --------------------------------------------------------------
    @staticmethod
    def _load_image(raw: str) -> Optional[np.ndarray]:
        if not raw:
            return None
        path = raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
        if img is None or img.ndim != 3 or img.shape[2] not in (3, 4):
            _warn_once(f"[sword] could not read sword image {path!r}; using the procedural blade.")
            return None
        if img.shape[2] == 3:
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        return img

    @staticmethod
    def _mips(img: np.ndarray) -> List[np.ndarray]:
        """Premultiplied-alpha mip chain (full, 1/2, 1/4 ...) so warps never minify much."""
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        pm = np.dstack([(img[:, :, :3].astype(np.float32) * a).round().astype(np.uint8), img[:, :, 3]])
        out = [pm]
        while out[-1].shape[0] >= 96 and out[-1].shape[1] >= 16:
            h, w = out[-1].shape[:2]
            out.append(cv2.resize(out[-1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA))
        return out

    def reset(self) -> None:
        self.hand: Optional[str] = None
        self.summoned = False
        self.alpha = 0.0
        self.fist_frames: Dict[str, int] = {}
        self.fist_time: Dict[str, float] = {}
        self.miss_frames = 0
        self.miss_time = 0.0
        self.direction = np.array([0.0, -1.0], np.float32)
        self.length = 0.0
        self.prev_tip: Optional[np.ndarray] = None
        self.vel = np.zeros(2, np.float32)
        self.armed = True
        self.dir_filter.reset()
        self._last_conf = 0.0
        self.trail.clear()

    # --- summon / dismiss ---------------------------------------------------
    def _qualifies(self, part: Optional[Part], hand_mode: bool) -> bool:
        if part is None or not part.visible or not _finite(part):
            return False
        return hand_mode or classify_hand(part.points, self.extend_ratio) == "fist"

    def _update_summon(self, ctx: Context, hand_mode: bool) -> None:
        hands = ctx.tracking.hands
        if self.summoned:
            part = hands.get(self.hand) if self.hand else None
            if part is None or not part.ok or not _finite(part):
                self.summoned = False  # hand left (or garbage landmarks); fade out in place
            elif part.visible:  # held (invisible) landmarks neither confirm nor dismiss
                if self._qualifies(part, hand_mode):
                    self.miss_frames, self.miss_time = 0, 0.0
                else:
                    self.miss_frames += 1
                    self.miss_time += max(0.0, ctx.dt)
                    if self.miss_frames >= DISMISS_FRAMES and self.miss_time >= DISMISS_SECONDS:
                        self.summoned = False
            if self.summoned:
                return
            self.fist_frames.clear()
            self.fist_time.clear()
        ready = []
        for side in ("right", "left"):
            if self._qualifies(hands.get(side), hand_mode):
                self.fist_frames[side] = self.fist_frames.get(side, 0) + 1
                self.fist_time[side] = self.fist_time.get(side, 0.0) + max(0.0, ctx.dt)
                if self.fist_frames[side] >= SUMMON_FRAMES and self.fist_time[side] >= SUMMON_SECONDS:
                    ready.append(side)
            else:
                self.fist_frames[side] = 0
                self.fist_time[side] = 0.0
        if not ready:
            return
        side = self.hand if self.hand in ready else ready[0]
        if side != self.hand or self.alpha <= 0.0:
            self.dir_filter.reset()
            self.length = 0.0
        self.hand = side
        self.summoned = True
        self.miss_frames, self.miss_time = 0, 0.0
        # No velocity carried over from before the summon: no fake swing.
        self.prev_tip = None
        self.vel[:] = 0
        self.trail.clear()
        self.armed = True

    # --- pose ---------------------------------------------------------------
    @staticmethod
    def _forearm(ctx: Context, pts: np.ndarray, hsize: float) -> Optional[Tuple[np.ndarray, float]]:
        """(unit elbow->wrist, reliability 0..1) of the pose arm whose wrist matches this hand."""
        pose = ctx.tracking.pose
        if not pose.ok or pose.points is None or len(pose.points) < 17:
            return None
        pp = pose.points
        wrist = pts[0, :2]
        best = min(((15, 13), (16, 14)), key=lambda we: float(np.linalg.norm(pp[we[0], :2] - wrist)))
        if float(np.linalg.norm(pp[best[0], :2] - wrist)) > max(1.0 * hsize, 0.04 * ctx.diag):
            return None
        if not np.all(np.isfinite(pp[list(best), :2])):
            return None
        vec = pp[best[0], :2] - pp[best[1], :2]
        flen = float(np.linalg.norm(vec)) / hsize
        if not np.isfinite(flen):
            return None
        # An in-plane forearm is ~2.5-3 hand sizes; shorter means it points at the camera.
        return fx.unit(vec), float(np.clip((flen - 0.8) / 1.2, 0.0, 1.0))

    def _raw_direction(self, ctx: Context, pts: np.ndarray, is_fist: bool, hsize: float,
                       hand_mode: bool) -> np.ndarray:
        fwd = fx.unit(pts[9, :2] - pts[0, :2])
        if hand_mode or not is_fist:
            return fwd
        fore = fwd
        fa = self._forearm(ctx, pts, hsize)
        if fa is not None and fa[1] > 0.0:
            fore = fx.unit(fwd + fa[1] * fa[0])
        axis = pts[5, :2] - pts[17, :2]
        span = float(np.linalg.norm(axis)) / max(hsize, 1e-3)
        # Face-on fists have a knuckle span of ~0.6-0.7 hand sizes; edge-on it collapses.
        wk = float(np.clip((span - 0.2) / 0.35, 0.0, 1.0))
        return fx.unit(wk * fx.unit(axis) + (FOREARM_TILT + (1.0 - wk)) * fore)

    def _sink_drop(self, ctx: Context, sink: float) -> np.ndarray:
        return np.array([0.0, SINK_DROP * ctx.canvas.h * sink * sink], np.float32)

    def update(self, ctx: Context) -> None:
        pose: SwordPose = ctx.sword
        hand_mode = ctx.sword_mode == "hand"
        self._update_summon(ctx, hand_mode)
        step = ctx.dt / FADE_SECONDS if ctx.dt > 0 else 0.0
        self.alpha = min(1.0, self.alpha + step) if self.summoned else max(0.0, self.alpha - step)

        part: Optional[Part] = ctx.tracking.hands.get(self.hand) if self.hand else None
        pose.speed = 0.0
        if part is None or not part.ok or self.hand is None or not _finite(part):
            # Hand gone: keep the last pose where it was and fade it out.
            pose.visible = False
            pose.confidence = self.alpha * self._last_conf
            self.prev_tip = None
            self.vel[:] = 0
            if self.alpha <= 0.0:
                self.hand = None
            return
        pts = part.points
        hsize = max(1.0, hand_size(pts))
        diag = ctx.diag
        self._last_conf = float(part.confidence)

        knuckles = pts[list(MCPS), :2].mean(axis=0)
        grip = knuckles if hand_mode else knuckles + (pts[0, :2] - knuckles) * 0.3
        raw = self._raw_direction(ctx, pts, True, hsize, hand_mode)
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(grip)):
            return
        self.direction = fx.unit(self.dir_filter(raw, ctx.dt))

        if hand_mode:
            target_len = float(np.clip(hsize * 2.6, 0.08 * diag, 0.3 * diag))
        else:
            target_len = float(np.clip(hsize * 5.5, 0.18 * diag, 0.55 * diag))
        self.length = target_len if self.length <= 0 or self.prev_tip is None else (
            self.length * 0.7 + target_len * 0.3)
        tip = grip + self.direction * self.length

        if self.summoned and self.prev_tip is not None and ctx.dt > 1e-4:
            v = (tip - self.prev_tip) / ctx.dt
            self.vel = self.vel * 0.35 + v.astype(np.float32) * 0.65
        else:
            self.vel[:] = 0
        self.prev_tip = tip.astype(np.float32) if self.summoned else None

        pose.visible = bool(self.summoned and part.visible)
        pose.hand = self.hand
        pose.grip = grip.astype(np.float32)
        pose.tip = tip.astype(np.float32)
        pose.direction = self.direction.astype(np.float32)
        pose.length = self.length
        spd = float(np.linalg.norm(self.vel))
        pose.speed = spd / diag if self.summoned else 0.0
        if spd > 1e-3:
            pose.motion = (self.vel / spd).astype(np.float32)
        pose.confidence = float(part.confidence) * self.alpha
        if self.alpha <= 0.0 and not self.summoned:
            self.hand = None
            return

        if self.summoned:
            base = grip + (tip - grip) * (0.6 if hand_mode else 0.62)
            self.trail.add(ctx.now, base, tip)
            # Edge-triggered swing sparks: fire once on crossing, re-arm below 60%.
            if self.armed and pose.speed >= self.spark_speed and pose.confidence > 0.3:
                self.armed = False
                self._clash(ctx, pose)
            elif not self.armed and pose.speed < self.spark_speed * 0.6:
                self.armed = True

        sink = float(np.clip(ctx.sword_sink, 0.0, 1.0))
        if 0.02 < sink < 0.97 and not hand_mode and pose.confidence > 0.1 and ctx.show_sword:
            self._sink_motes(ctx, pose, sink)

    def _sink_motes(self, ctx: Context, pose: SwordPose, sink: float) -> None:
        # ~75-90 motes/s regardless of frame rate.
        n = int(np.random.poisson((75.0 * (1.0 - sink) + 15.0) * max(0.0, ctx.dt)))
        if n <= 0:
            return
        drop = self._sink_drop(ctx, sink)
        k = np.random.uniform(0.1, 1.0, (n, 1)).astype(np.float32)
        pos = pose.grip + drop + (pose.tip - pose.grip) * k
        L = max(pose.length, 1.0)
        ctx.particles.emit(n, pos, np.array([0.0, -0.15 * L], np.float32), spread_pos=L * 0.02,
                           spread_vel=L * 0.12, life=(0.4, 0.9), size=(1.0, 2.2), color=MOTE_COLOR, drag=1.5)

    def _clash(self, ctx: Context, pose: SwordPose) -> None:
        col = ctx.energy_color()
        L = max(pose.length, 1.0)
        tip_v = pose.motion * L * 2.2
        ctx.particles.emit(26, pose.tip, tip_v, spread_pos=L * 0.04, spread_vel=L * 1.1,
                           life=(0.15, 0.4), size=(1.0, 2.4), color=(255, 255, 255), drag=4.0, gravity=300.0)
        mid = pose.grip + (pose.tip - pose.grip) * 0.7
        ctx.particles.emit(18, mid, tip_v * 0.6, spread_pos=L * 0.08, spread_vel=L * 0.7,
                           life=(0.2, 0.5), size=(1.5, 3.0), color=col, drag=3.0)
        ctx.canvas.flash = max(ctx.canvas.flash, 0.06)
        ctx.shaker.kick(4.0)

    # --- rendering ----------------------------------------------------------
    def render(self, ctx: Context) -> None:
        pose = ctx.sword
        a = float(np.clip(pose.confidence, 0.0, 1.0))
        sink = float(np.clip(ctx.sword_sink, 0.0, 1.0))
        if a < 0.02 or pose.length <= 1.0 or sink >= 0.995 or ctx.canvas.frame is None:
            self.trail.prune(ctx.now)
            return
        # Sinking: ease-in descent toward the ground, fading and narrowing.
        a *= (1.0 - sink) ** 1.2
        self._vscale = 1.0 - 0.65 * sink
        if sink > 0.0:
            drop = self._sink_drop(ctx, sink)
            pose = replace(pose, grip=pose.grip + drop, tip=pose.tip + drop)
        col = ctx.energy_color()
        boost = 1.0 + max(0.0, ctx.aura_boost)
        hand_mode = ctx.sword_mode == "hand"
        occ = None if hand_mode else self._occlusion_begin(ctx)
        swing = float(np.clip((ctx.sword.speed - 0.5) / 1.5, 0.0, 1.0))
        if swing > 0.0 and sink < 0.5:
            self._render_trail(ctx, a * (0.2 + 0.5 * swing), col)
        else:
            self.trail.prune(ctx.now)

        if hand_mode:
            self._render_hand(ctx, pose, a, col, boost)
        else:
            style = ctx.current_weapon()
            img = self.images.get(style)
            if img is None and self.image is not None:
                self._render_image(ctx, pose, a, self.image)
                L = pose.length
                ctx.canvas.glow_line(pose.grip + pose.direction * 0.15 * L, pose.grip + pose.direction * L,
                                     col, L * 0.06 * self._vscale, 0.45 * a * boost)
            elif style == "cleaver":
                self._render_cleaver(ctx, pose, a, col, boost, img)
            elif style == "dark":
                self._render_dark(ctx, pose, a, col, boost, img)
            else:
                self._render_katana(ctx, pose, a, col, boost, img)
        if occ is not None:
            self._occlusion_end(ctx, occ)
        self._vscale = 1.0

    # Finger occlusion -----------------------------------------------------------
    def _occlusion_begin(self, ctx: Context):
        """Snapshot the camera/light/glow pixels around the hand and build a finger mask."""
        part = ctx.tracking.hands.get(self.hand) if self.hand else None
        if part is None or not part.ok or not _finite(part):
            return None
        canvas = ctx.canvas
        pts = part.points[:, :2]
        if not np.all(np.isfinite(pts)):
            return None
        hsize = max(1.0, hand_size(part.points))
        pad = int(hsize * 0.25) + 3
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(canvas.w, int(pts[:, 0].max()) + pad + 1)
        y1 = min(canvas.h, int(pts[:, 1].max()) + pad + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        p = np.round((pts - (x0, y0)) * 4).astype(np.int32)  # 2 fractional bits
        cv2.fillConvexPoly(m, cv2.convexHull(p[list(_PALM)]), 255, cv2.LINE_AA, shift=2)
        cv2.polylines(m, [p[list(_PALM)]], True, 255, max(2, int(hsize * 0.18)), cv2.LINE_AA, shift=2)
        thick = max(2, int(hsize * 0.24))
        cv2.polylines(m, [p[list(c)] for c in _FINGER_CHAINS], False, 255, thick, cv2.LINE_AA, shift=2)
        k = max(3, int(hsize * 0.05) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)
        gs = canvas.gs
        gx0, gy0 = int(x0 * gs), int(y0 * gs)
        gx1, gy1 = min(canvas.gw, int(math.ceil(x1 * gs))), min(canvas.gh, int(math.ceil(y1 * gs)))
        glow = canvas.glow[gy0:gy1, gx0:gx1].copy() if gx1 > gx0 and gy1 > gy0 else None
        return ((x0, y0, x1, y1), m, canvas.frame[y0:y1, x0:x1].copy(), canvas.light[y0:y1, x0:x1].copy(),
                (gx0, gy0, gx1, gy1), glow)

    @staticmethod
    def _occlusion_end(ctx: Context, occ) -> None:
        (x0, y0, x1, y1), m, frame0, light0, (gx0, gy0, gx1, gy1), glow0 = occ
        canvas = ctx.canvas
        w1 = m.astype(np.float32) * (1.0 / 255.0)
        w2 = 1.0 - w1
        for layer, orig in ((canvas.frame, frame0), (canvas.light, light0)):
            roi = layer[y0:y1, x0:x1]
            roi[:] = cv2.blendLinear(orig, roi, w1, w2)
        if glow0 is not None:
            gm = cv2.resize(w1, (gx1 - gx0, gy1 - gy0), interpolation=cv2.INTER_AREA)[:, :, None]
            g = canvas.glow[gy0:gy1, gx0:gx1]
            g[:] = g * (1.0 - gm) + glow0 * gm

    def _render_trail(self, ctx: Context, strength: float, col) -> None:
        """Slash arc: a faint ribbon (fx.Trail) plus a hot tapering line along the tip path."""
        canvas = ctx.canvas
        self.trail.render(canvas, ctx.now, col, strength=strength * 0.5)
        samples = self.trail.samples
        if len(samples) < 2:
            return
        tips = [b for _, _, b in samples]
        n = len(tips)
        for i in range(1, n):
            k = i / (n - 1)
            canvas.light_line(tips[i - 1], tips[i], (255, 255, 255), max(1, int(1 + 3 * k)), strength * k)
            canvas.glow_line(tips[i - 1], tips[i], col, 4 + 10 * k, strength * k)

    def _xf(self, pose: SwordPose, uv: np.ndarray) -> np.ndarray:
        """Blade-local (u along blade, v across) in units of length -> render pixels."""
        d = pose.direction
        n = fx.perp(d)
        L = pose.length
        uv = np.asarray(uv, np.float32)
        return (pose.grip[None, :] + uv[:, :1] * L * d[None, :]
                + uv[:, 1:2] * (L * self._vscale) * n[None, :]).astype(np.float32)

    @staticmethod
    def _roi(canvas: fx.Canvas, polys: List[np.ndarray], pad: int = 4) -> Optional[Tuple[int, int, int, int]]:
        allp = np.concatenate(polys, 0)
        x0 = max(0, int(allp[:, 0].min()) - pad)
        y0 = max(0, int(allp[:, 1].min()) - pad)
        x1 = min(canvas.w, int(allp[:, 0].max()) + pad + 1)
        y1 = min(canvas.h, int(allp[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _fill(frame: np.ndarray, pts: np.ndarray, color) -> None:
        cv2.fillPoly(frame, [np.round(pts * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)

    @staticmethod
    def _poly(frame: np.ndarray, pts: np.ndarray, color, thickness: int = 1, closed: bool = False) -> None:
        cv2.polylines(frame, [np.round(pts * 16).astype(np.int32)], closed, color, thickness, cv2.LINE_AA, shift=4)

    def _opaque(self, ctx: Context, polys: List[np.ndarray], a: float, draw) -> None:
        """Run ``draw(frame)`` and blend it over the camera image with opacity ``a``."""
        frame = ctx.canvas.frame
        box = self._roi(ctx.canvas, polys)
        if box is None:
            return
        x0, y0, x1, y1 = box
        orig = frame[y0:y1, x0:x1].copy() if a < 0.98 else None
        draw(frame)
        if orig is not None:
            roi = frame[y0:y1, x0:x1]
            cv2.addWeighted(roi, a, orig, 1.0 - a, 0, dst=roi)

    def _edge_glow(self, ctx: Context, edge: np.ndarray, col, strength: float, width: float) -> None:
        canvas = ctx.canvas
        for i in range(len(edge) - 1):
            canvas.glow_line(edge[i], edge[i + 1], col, width, strength)

    def _wrap_handle(self, pose: SwordPose, u0: float, u1: float, half_w: float, n_diamonds: int) -> List[np.ndarray]:
        out = []
        step = (u1 - u0) / n_diamonds
        for k in range(n_diamonds):
            uc = u0 + step * (k + 0.5)
            out.append(self._xf(pose, np.array([[uc - step * 0.42, 0], [uc, half_w * 0.7],
                                                [uc + step * 0.42, 0], [uc, -half_w * 0.7]])))
        return out

    # Katana --------------------------------------------------------------
    def _render_katana(self, ctx: Context, pose: SwordPose, a: float, col, boost: float,
                       img: Optional[List[np.ndarray]] = None) -> None:
        blade_uv, shinogi_uv, edge_uv, hamon_uv = katana_profile()
        blade = self._xf(pose, blade_uv)
        edge_px = self._xf(pose, edge_uv)
        L = pose.length
        if img is not None:
            self._render_image(ctx, pose, a, img)
        else:
            shinogi = self._xf(pose, shinogi_uv)
            hamon = self._xf(pose, hamon_uv)
            handle = self._xf(pose, np.array([[-0.18, -0.02], [0.085, -0.019], [0.085, 0.019], [-0.18, 0.02]]))
            pommel = self._xf(pose, np.array([[-0.2, -0.021], [-0.175, -0.022], [-0.175, 0.022], [-0.2, 0.021]]))
            habaki = self._xf(pose, np.array([[0.1, -0.017], [0.125, -0.016], [0.125, 0.019], [0.1, 0.019]]))
            diamonds = self._wrap_handle(pose, -0.17, 0.075, 0.02, 7)
            n = fx.perp(pose.direction)
            tsuba_c = self._xf(pose, np.array([[0.093, 0.0]]))[0]
            ang = math.degrees(math.atan2(n[1], n[0]))
            axes = (max(2, int(L * 0.052 * self._vscale)), max(1, int(L * 0.014)))

            def draw(frame: np.ndarray) -> None:
                self._fill(frame, handle, (38, 30, 34))
                for dm in diamonds:
                    self._fill(frame, dm, (175, 170, 160))
                self._fill(frame, pommel, (70, 140, 175))
                self._fill(frame, blade, (178, 174, 168))
                self._fill(frame, shinogi, (125, 122, 118))
                self._poly(frame, hamon, (238, 238, 236), 1)
                self._poly(frame, edge_px, (250, 250, 250), max(1, int(L * 0.004)))
                cv2.ellipse(frame, fx.ipt(tsuba_c), axes, ang, 0, 360, (40, 44, 48), -1, cv2.LINE_AA)
                cv2.ellipse(frame, fx.ipt(tsuba_c), axes, ang, 0, 360, (95, 105, 115), 1, cv2.LINE_AA)
                self._fill(frame, habaki, (60, 170, 215))

            self._opaque(ctx, [blade, handle, pommel], a, draw)
        g = a * boost
        self._edge_glow(ctx, edge_px, col, 0.55 * g, L * 0.03)
        canvas = ctx.canvas
        cv2.polylines(canvas.light, [edge_px.astype(np.int32)], False,
                      fx.scale_color(col, 0.35 * g * canvas.intensity), 1, cv2.LINE_AA)
        if boost > 1.05:
            canvas.glow_poly(blade, col, 0.5 * (boost - 1.0) * a)

    # Shikai cleaver ------------------------------------------------------------
    def _render_cleaver(self, ctx: Context, pose: SwordPose, a: float, col, boost: float,
                        img: Optional[List[np.ndarray]] = None) -> None:
        edge_v = CLEAVER_EDGE_V
        blade = self._xf(pose, CLEAVER_BLADE)
        edge_px = self._xf(pose, CLEAVER_EDGE)
        edge_fine = np.concatenate([np.linspace(edge_px[i], edge_px[i + 1], 6)[:-1] for i in range(3)]
                                   + [edge_px[-1:]], 0)
        L = pose.length
        # Loose cloth tail from the pommel, hanging under screen gravity with a little sway.
        base = self._xf(pose, np.array([[-0.2, 0.0]]))[0]
        sway = math.sin(ctx.now * 3.0) * 0.04
        cloth = [base]
        for j in range(1, 6):
            t = j / 5.0
            cloth.append(base - pose.direction * L * 0.05 * t
                         + np.array([sway * L * t, 0.0], np.float32)
                         + np.array([0.0, 1.0], np.float32) * L * 0.16 * t * t)
        cloth = np.array(cloth, np.float32)
        cloth_w = max(2, int(L * 0.012 * self._vscale))
        if img is not None:
            def draw_cloth(frame: np.ndarray) -> None:
                self._poly(frame, cloth, (150, 160, 170), cloth_w + 2)
                self._poly(frame, cloth, (212, 220, 226), cloth_w)

            self._opaque(ctx, [cloth], a, draw_cloth)
            self._render_image(ctx, pose, a, img)
        else:
            bevel = self._xf(pose, np.array([[0.07, edge_v + 0.035], [0.8, edge_v + 0.035], [0.885, edge_v * 0.25],
                                             [0.9, edge_v * 0.55], [0.8, edge_v], [0.07, edge_v]]))
            fuller = self._xf(pose, np.array([[0.1, CLEAVER_BACK_V - 0.018], [0.9, CLEAVER_BACK_V - 0.018]]))
            hilt = self._xf(pose, np.array([[-0.2, -0.022], [0.075, -0.024], [0.075, 0.024], [-0.2, 0.022]]))
            bands = []
            for k in range(8):
                u0 = -0.19 + k * 0.034
                bands.append(self._xf(pose, np.array([[u0, -0.024], [u0 + 0.02, 0.024]])))

            def draw(frame: np.ndarray) -> None:
                self._fill(frame, blade, (88, 86, 90))
                self._fill(frame, bevel, (190, 188, 192))
                self._poly(frame, fuller, (60, 58, 62), max(1, int(L * 0.006)))
                self._poly(frame, cloth, (215, 215, 210), cloth_w)
                self._fill(frame, hilt, (205, 205, 200))
                for b in bands:
                    self._poly(frame, b, (90, 90, 95), max(1, int(L * 0.006)))
                self._poly(frame, edge_fine, (255, 250, 240), max(1, int(L * 0.005)))

            self._opaque(ctx, [blade, hilt, cloth], a, draw)
        g = a * boost
        edge_col = tuple(min(255, int(c * 0.6 + 255 * 0.4)) for c in col)  # blue-white
        self._edge_glow(ctx, edge_fine, edge_col, 0.75 * g, L * 0.035)
        cv2.polylines(ctx.canvas.light, [edge_fine.astype(np.int32)], False,
                      fx.scale_color(edge_col, 0.55 * g * ctx.canvas.intensity), 2, cv2.LINE_AA)
        if boost > 1.05:
            ctx.canvas.glow_poly(blade, col, 0.45 * (boost - 1.0) * a)

    # Bankai dark blade -----------------------------------------------------------
    def _render_dark(self, ctx: Context, pose: SwordPose, a: float, col, boost: float,
                     img: Optional[List[np.ndarray]] = None) -> None:
        blade = self._xf(pose, dark_profile())
        guard = self._xf(pose, np.array([[0.095, -0.05], [0.105, -0.058], [0.12, -0.05], [0.12, 0.05],
                                         [0.105, 0.058], [0.095, 0.05]]))
        L = pose.length
        links = self._xf(pose, np.array([[-0.19, 0.0], [-0.225, 0.008], [-0.26, 0.02]]))
        ring_r = max(2, int(L * 0.013))
        if img is not None:
            def draw_links(frame: np.ndarray) -> None:
                for p in links[1:]:
                    cv2.circle(frame, fx.ipt(p), ring_r, (36, 32, 36), max(1, ring_r // 2), cv2.LINE_AA)

            self._opaque(ctx, [links], a, draw_links)
            self._render_image(ctx, pose, a, img)
        else:
            handle = self._xf(pose, np.array([[-0.17, -0.018], [0.095, -0.018], [0.095, 0.018], [-0.17, 0.018]]))
            diamonds = self._wrap_handle(pose, -0.16, 0.085, 0.018, 7)

            def draw(frame: np.ndarray) -> None:
                self._fill(frame, handle, (14, 12, 14))
                for dm in diamonds:
                    self._fill(frame, dm, (52, 48, 52))
                for p in links:
                    cv2.circle(frame, fx.ipt(p), ring_r, (24, 22, 24), max(1, ring_r // 2), cv2.LINE_AA)
                self._fill(frame, guard, (16, 14, 16))
                self._fill(frame, blade, (8, 6, 8))

            self._opaque(ctx, [blade, guard, handle, links], a, draw)
        g = a * boost
        canvas = ctx.canvas
        flick = 0.85 + 0.15 * math.sin(ctx.now * 23.0)
        outline = blade.astype(np.int32)
        cv2.polylines(canvas.light, [outline], True, fx.scale_color(col, 0.75 * g * flick * canvas.intensity),
                      max(1, int(L * 0.004)), cv2.LINE_AA)
        for i in range(len(outline)):
            canvas.glow_line(outline[i], outline[(i + 1) % len(outline)], col, L * 0.02, 0.5 * g * flick)
        canvas.glow_poly(guard, col, 0.35 * g)

    # Image asset ------------------------------------------------------------------
    def _render_image(self, ctx: Context, pose: SwordPose, a: float, mips: List[np.ndarray]) -> None:
        """Warp premultiplied RGBA art (blade up, u = IMG_U0..IMG_U1 bottom to top) into an ROI."""
        L = pose.length
        span = (IMG_U1 - IMG_U0) * L  # on-screen image height in px
        img = mips[0]
        for m in mips[1:]:  # smallest level that is still >= 85% of the on-screen size
            if m.shape[0] >= span * 0.85:
                img = m
        ih, iw = img.shape[:2]
        d = pose.direction
        sc = span / ih
        ex = fx.perp(d) * (sc * self._vscale)  # image +x per pixel
        ey = -d * sc  # image +y (image "up" is the blade direction)
        base = pose.grip + d * (IMG_U0 * L)  # bottom centre of the image
        t = base - ex * (iw / 2.0) - ey * ih
        corners = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], np.float32)
        world = t[None, :] + corners[:, :1] * ex[None, :] + corners[:, 1:2] * ey[None, :]
        box = self._roi(ctx.canvas, [world], pad=2)
        if box is None:
            return
        x0, y0, x1, y1 = box
        M = np.array([[ex[0], ey[0], t[0] - x0], [ex[1], ey[1], t[1] - y0]], np.float32)
        warped = cv2.warpAffine(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = warped[:, :, 3]
        if a < 0.995:
            alpha = cv2.convertScaleAbs(alpha, alpha=a)
            rgb = cv2.convertScaleAbs(warped[:, :, :3], alpha=a)
        else:
            rgb = warped[:, :, :3]
        roi = ctx.canvas.frame[y0:y1, x0:x1]
        inv = cv2.cvtColor(255 - alpha, cv2.COLOR_GRAY2BGR)
        cv2.add(cv2.multiply(roi, inv, scale=1.0 / 255.0), rgb, dst=roi)

    # Hand-only mode ---------------------------------------------------------------
    def _render_hand(self, ctx: Context, pose: SwordPose, a: float, col, boost: float) -> None:
        part = ctx.tracking.hands.get(self.hand) if self.hand else None
        canvas = ctx.canvas
        g = a * boost
        if part is not None and part.ok:
            pts = part.points[:, :2]
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
            c = hull.mean(axis=0)
            canvas.glow_poly(c + (hull - c) * 1.45, col, 0.45 * g)
            canvas.glow_poly(c + (hull - c) * 1.1, col, 0.4 * g)
            flick = 0.7 + 0.3 * math.sin(ctx.now * 19.0)
            canvas.light_poly(c + (hull - c) * 1.05, col, 0.18 * g * flick)
            hs = max(1.0, hand_size(part.points))
            for ti in FINGERTIPS:
                out = pts[ti] + fx.unit(pts[ti] - c) * hs * np.random.uniform(0.25, 0.5)
                line = fx.jagged_line(pts[ti], out, segments=3, jitter=hs * 0.06)
                cv2.polylines(canvas.light, [line.astype(np.int32)], False,
                              fx.scale_color(col, 0.6 * g * canvas.intensity), 1, cv2.LINE_AA)
            if np.random.random() < 0.6 * min(1.0, g):
                ctx.particles.emit(2, c, np.array([0.0, -60.0], np.float32), spread_pos=hs * 0.5,
                                   spread_vel=30.0, life=(0.3, 0.7), size=(1.0, 2.2), color=col, drag=1.0)
        # Energy blade extending past the fingertips (light only, no steel).
        s = np.linspace(0.0, 1.0, 8)
        half = 0.09 * (1.0 - s) ** 0.7 + 0.004
        u = 0.12 + 0.88 * s
        blade = self._xf(pose, np.concatenate([np.stack([u, half], 1), np.stack([u, -half], 1)[::-1]], 0))
        core = self._xf(pose, np.concatenate([np.stack([u, half * 0.35], 1),
                                              np.stack([u, -half * 0.35], 1)[::-1]], 0))
        canvas.glow_poly(blade, col, 0.8 * g)
        canvas.light_poly(blade, col, 0.45 * g)
        canvas.light_poly(core, (255, 255, 255), 0.75 * g)
