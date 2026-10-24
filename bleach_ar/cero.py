"""Cero: a charged sphere of crimson energy released as a beam.

Visually distinct from Getsuga Tensho (a travelling crescent): Cero is a
stationary orb that grows at the palm, then a straight beam from the hand to
the screen edge, or - when the arm points at the camera - a radial blast that
fills the screen.

Source point, in priority order: open palm centre, between the index and
middle fingertips of a "two_fingers" hand, any visible palm, the mouth, the
screen centre.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .abilities import Ability, Context, Phase
from .fx import ease_in, ease_out, jagged_line, perp, ring, scale_color, smoothstep, unit
from .gestures import classify_hand
from .tracking import L_ELBOW, L_WRIST, R_ELBOW, R_WRIST

PALM = [0, 5, 9, 13, 17]
GESTURE_RANK = {"open_palm": 0, "two_fingers": 1}


class Cero(Ability):
    name = "cero"
    label = "Cero"

    def __init__(self, params: dict):
        super().__init__(params)
        self.src = np.zeros(2, np.float32)
        self.dir = np.array([1.0, 0.0], np.float32)
        self.camera_mode = False
        self.end = np.zeros(2, np.float32)  # beam end (just past the screen edge)
        self.hit = np.zeros(2, np.float32)  # impact point (inside the screen)
        self._src_kind = "center"
        self._has_src = False

    # --- params -----------------------------------------------------------
    @property
    def size(self) -> float:
        return float(self.params.get("size", 1.0))

    @property
    def power(self) -> float:
        return float(self.params.get("intensity", 1.0))

    @property
    def color(self) -> Tuple[int, int, int]:
        return tuple(self.params.get("color", (60, 40, 255)))

    @property
    def core(self) -> Tuple[int, int, int]:
        return tuple(self.params.get("core_color", (235, 235, 255)))

    # --- tracking ---------------------------------------------------------
    def _find_source(self, ctx: Context) -> Tuple[np.ndarray, str]:
        ratio = float(ctx.cfg.get("gestures", {}).get("finger_extend_ratio", 1.12))
        best = None
        for part in ctx.tracking.hands.values():
            if not part.ok:
                continue
            pts = part.points
            g = classify_hand(pts, ratio)
            rank = GESTURE_RANK.get(g, 2)
            if g == "two_fingers":
                p = (pts[8, :2] + pts[12, :2]) * 0.5
            else:
                p = pts[PALM, :2].mean(axis=0)
            if best is None or rank < best[0]:
                best = (rank, p.astype(np.float32), g)
        if best is not None:
            return best[1], best[2]
        face = ctx.tracking.face
        if face.ok:
            pts = face.points
            return ((pts[13, :2] + pts[14, :2]) * 0.5).astype(np.float32), "mouth"
        return np.array([ctx.canvas.w * 0.5, ctx.canvas.h * 0.5], np.float32), "center"

    def _track_source(self, ctx: Context, follow: float = 14.0) -> None:
        p, kind = self._find_source(ctx)
        self._src_kind = kind
        if not self._has_src:
            self.src, self._has_src = p, True
        else:
            self.src += (p - self.src) * min(1.0, ctx.dt * follow)

    def _arm_direction(self, ctx: Context) -> Optional[np.ndarray]:
        """Forearm direction (elbow -> wrist) of the arm nearest the source, or None for camera mode."""
        pose = ctx.tracking.pose
        if not pose.ok or self._src_kind in ("mouth", "center"):
            return None
        pts = pose.points
        arms = ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST))
        elbow, wrist = min(arms, key=lambda ew: float(np.linalg.norm(pts[ew[1], :2] - self.src)))
        if np.linalg.norm(pts[wrist, :2] - self.src) > 0.2 * ctx.diag:
            return None
        v = pts[wrist, :2] - pts[elbow, :2]
        length = float(np.linalg.norm(v))
        if length < 0.06 * ctx.diag:
            return None
        # Foreshortened forearm (wrist much nearer the camera than the elbow).
        if -(float(pts[wrist, 2]) - float(pts[elbow, 2])) > 1.2 * length:
            return None
        d = unit(v)
        # An open palm raised with a vertical forearm is the "fire at the camera" pose.
        if self._src_kind == "open_palm" and d[1] < -0.9:
            return None
        return d

    def _edge_point(self, ctx: Context, margin: float) -> np.ndarray:
        w, h = ctx.canvas.w, ctx.canvas.h
        d = self.dir
        ts = []
        for axis, lim in ((0, w), (1, h)):
            if d[axis] > 1e-6:
                ts.append((lim + margin - self.src[axis]) / d[axis])
            elif d[axis] < -1e-6:
                ts.append((-margin - self.src[axis]) / d[axis])
        t = max(0.0, min(ts)) if ts else 0.0
        return (self.src + d * t).astype(np.float32)

    def _update_beam_geometry(self, ctx: Context) -> None:
        self.end = self._edge_point(ctx, 60.0)
        hit = self._edge_point(ctx, 0.0)
        self.hit = np.clip(hit - self.dir * 25.0, [0, 0], [ctx.canvas.w - 1, ctx.canvas.h - 1]).astype(np.float32)

    # --- hooks ------------------------------------------------------------
    def on_enter(self, phase: Phase, ctx: Context) -> None:
        cv = ctx.canvas
        d = ctx.diag
        if phase == Phase.CHARGING:
            self._has_src = False
            self._track_source(ctx)
        elif phase == Phase.ACTIVE:
            self._track_source(ctx)
            arm = self._arm_direction(ctx)
            self.camera_mode = arm is None
            if arm is not None:
                self.dir = arm
                self._update_beam_geometry(ctx)
            cv.flash = max(cv.flash, (0.65 if self.camera_mode else 0.45) * min(1.0, self.power))
            ctx.shaker.kick(20 * self.power)
            self._burst(ctx, self.src, 120, (0.15 * d, 0.6 * d), self.color)
            self._burst(ctx, self.src, 60, (0.1 * d, 0.4 * d), self.core, life=(0.3, 0.6))
        elif phase == Phase.IMPACT:
            self.hit = self.src.copy() if self.camera_mode else self.hit
            cv.flash = max(cv.flash, 0.35 * min(1.0, self.power))
            ctx.shaker.kick(16 * self.power)
            n = int(200 * self.size)
            self._burst(ctx, self.hit, n, (0.12 * d, 0.7 * d * self.size), self.color, life=(0.4, 1.1))
            self._burst(ctx, self.hit, n // 2, (0.05 * d, 0.35 * d * self.size), self.core, life=(0.3, 0.7))

    @staticmethod
    def _burst(ctx: Context, center, n: int, speed, color, life=(0.4, 1.0)) -> None:
        if n <= 0:
            return
        ang = np.random.uniform(0, 2 * math.pi, n)
        sp = np.random.uniform(speed[0], speed[1], n)
        vel = (np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None]).astype(np.float32)
        ctx.particles.emit(n, center, vel, spread_pos=4, life=life, size=(1.5, 4.0), color=color, drag=2.4)

    def on_update(self, ctx: Context) -> None:
        p = self.progress
        d = ctx.diag
        if self.phase == Phase.CHARGING:
            self._track_source(ctx)
            r = self._orb_radius(ctx, p)
            ctx.shaker.kick((1.0 + 6.0 * ease_in(p)) * self.power)
            ctx.aura_boost = max(ctx.aura_boost, 0.6 * p)
            n = int((5 + 12 * p) * self.size)
            ang = np.random.uniform(0, 2 * math.pi, n)
            rad = np.random.uniform(r * 3.5, r * 3.5 + 0.12 * d * self.size, n)
            dirs = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
            pos = self.src + dirs * rad[:, None]
            vel = -dirs * rad[:, None] * 1.4 + np.stack([-dirs[:, 1], dirs[:, 0]], 1) * rad[:, None] * 1.2
            col = self.color if np.random.rand() < 0.7 else self.core
            ctx.particles.emit(n, pos, vel.astype(np.float32), life=(0.35, 0.6), size=(1.2, 3.0), color=col,
                               drag=1.0)
            ctx.particles.attract(self.src, 1400 + 2200 * p, ctx.dt)
        elif self.phase == Phase.ACTIVE:
            self._track_source(ctx, follow=8.0)
            ctx.shaker.kick((9.0 + 4.0 * math.sin(self.t * 40)) * self.power)
            if self.t > 0.1 and int(self.t * 8) != int((self.t - ctx.dt) * 8):
                ctx.canvas.flash = max(ctx.canvas.flash, 0.12 * min(1.0, self.power))
            if self.camera_mode:
                self._emit_blast(ctx, p)
            else:
                arm = self._arm_direction(ctx)
                if arm is not None:
                    self.dir = unit(self.dir + (arm - self.dir) * min(1.0, ctx.dt * 6.0))
                self._update_beam_geometry(ctx)
                self._emit_spiral(ctx)

    def _emit_spiral(self, ctx: Context) -> None:
        """Two particle strands corkscrewing down the beam."""
        hw = self._beam_half_width(ctx)
        L = float(np.linalg.norm(self.end - self.src))
        n = 8
        s = np.random.uniform(0.02, 0.6, (n, 1)).astype(np.float32) * L
        ph = ctx.now * 18.0 + s[:, 0] * 0.03 + np.where(np.arange(n) % 2 == 0, 0.0, math.pi)
        nrm = perp(self.dir)
        off = np.sin(ph)[:, None] * hw * 1.5
        pos = self.src + self.dir * s + nrm * off
        speed = 0.9 * ctx.diag
        vel = self.dir * speed + nrm * (np.cos(ph)[:, None] * hw * 10.0)
        ctx.particles.emit(n, pos.astype(np.float32), vel.astype(np.float32), life=(0.25, 0.5),
                           size=(1.5, 3.0), color=(200, 190, 255), drag=0.5)

    def _emit_blast(self, ctx: Context, p: float) -> None:
        n = int(14 * self.size)
        d = ctx.diag
        self._burst(ctx, self.src, n, (0.4 * d, 1.1 * d), self.color, life=(0.3, 0.7))

    # --- geometry ---------------------------------------------------------
    def _orb_radius(self, ctx: Context, p: float) -> float:
        rmax = 0.075 * min(ctx.canvas.w, ctx.canvas.h) * self.size
        return rmax * (0.12 + 0.88 * ease_out(p))

    def _beam_half_width(self, ctx: Context) -> float:
        base = 0.06 * min(ctx.canvas.w, ctx.canvas.h) * self.size
        grow = ease_out(min(1.0, self.t / 0.14))
        fade = 1.0 - smoothstep((self.progress - 0.78) / 0.22)
        return base * grow * fade * (1.0 + 0.06 * math.sin(self.t * 55))

    # --- rendering --------------------------------------------------------
    def render(self, ctx: Context) -> None:
        if self.phase in (Phase.IDLE, Phase.COOLDOWN):
            return
        cv = ctx.canvas
        if self.phase == Phase.CHARGING:
            p = self.progress
            cv.dim = max(cv.dim, 0.3 * p)
            self._render_orb(ctx, self.src, self._orb_radius(ctx, p), p)
        elif self.phase == Phase.ACTIVE:
            cv.dim = max(cv.dim, 0.6 if self.camera_mode else 0.35)
            if self.camera_mode:
                self._render_blast(ctx)
            else:
                self._render_beam(ctx)
        elif self.phase == Phase.IMPACT:
            cv.dim = max(cv.dim, 0.35 * (1.0 - self.progress))
            self._render_impact(ctx)

    def _render_orb(self, ctx: Context, c, r: float, p: float) -> None:
        cv = ctx.canvas
        k = self.power
        col, core = self.color, self.core
        flick = 0.85 + 0.15 * np.random.rand()
        cv.glow_circle(c, r * 2.6, col, (0.25 + 0.4 * p) * k * flick)
        cv.glow_circle(c, r * 1.2, core, 0.2 * p * k)
        body = self._noisy_circle(c, r, 18, 0.12, ctx.now * 9.0)
        cv.light_poly(body, col, 0.95 * k * flick)
        cv.light_poly(self._noisy_circle(c, r * 0.72, 14, 0.08, -ctx.now * 7.0), scale_color(col, 1.4),
                      0.9 * k)
        cv.light_circle(c, r * 0.48, core, k)
        cv.light_circle(c, r * 0.28, (255, 255, 255), k)
        # Crackling arcs leaping off the orb.
        arcs = 2 + int(5 * p)
        for _ in range(arcs):
            a = np.random.uniform(0, 2 * math.pi)
            dv = np.array([math.cos(a), math.sin(a)], np.float32)
            reach = r * np.random.uniform(1.4, 2.6)
            pts = jagged_line(c + dv * r * 0.6, c + dv * reach, segments=5, jitter=r * 0.25 + 2)
            cv2.polylines(cv.light, [pts.astype(np.int32)], False, scale_color(col, 1.2 * k), 2, cv2.LINE_AA)
            cv2.polylines(cv.light, [pts[:3].astype(np.int32)], False, scale_color(core, k), 1, cv2.LINE_AA)
        # Rotating containment ring.
        if p > 0.3:
            cv2.ellipse(cv.light, (int(c[0]), int(c[1])), (max(1, int(r * 1.35)), max(1, int(r * 0.5))),
                        math.degrees(ctx.now * 3.0) % 360, 0, 360, scale_color(col, 0.8 * k), 1, cv2.LINE_AA)

    @staticmethod
    def _noisy_circle(c, r: float, n: int, amount: float, phase: float) -> np.ndarray:
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rr = r * (1.0 + amount * np.sin(a * 3 + phase) * 0.6 + amount * np.random.uniform(-0.5, 0.5, n))
        return (np.stack([np.cos(a), np.sin(a)], 1) * rr[:, None] + np.asarray(c, np.float32)).astype(np.float32)

    def _beam_poly(self, ctx: Context, hw: float, wobble: float, seed: float) -> np.ndarray:
        L = float(np.linalg.norm(self.end - self.src))
        n = max(8, int(L / 30))
        s = np.linspace(0.0, 1.0, n, dtype=np.float32)
        dist = s * L
        centre = self.src + self.dir * dist[:, None]
        t = ctx.now
        # Flared muzzle near the hand, then wobbling edges.
        flare = 1.0 + 0.5 * np.exp(-dist / (hw * 2.0 + 1.0))
        nl = 1.0 + wobble * (np.sin(dist * 0.045 - t * 38 + seed) * 0.6 + np.random.uniform(-0.4, 0.4, n))
        nr = 1.0 + wobble * (np.sin(dist * 0.05 - t * 33 + seed * 2.1) * 0.6 + np.random.uniform(-0.4, 0.4, n))
        nrm = perp(self.dir)
        left = centre + nrm * (hw * flare * nl)[:, None]
        right = centre - nrm * (hw * flare * nr)[:, None]
        return np.concatenate([left, right[::-1]], 0).astype(np.float32)

    def _render_beam(self, ctx: Context) -> None:
        cv = ctx.canvas
        hw = self._beam_half_width(ctx)
        if hw < 1.0:
            return
        k = self.power
        col, core = self.color, self.core
        cv.glow_poly(self._beam_poly(ctx, hw * 2.4, 0.15, 0.0), col, 0.55 * k)
        cv.light_poly(self._beam_poly(ctx, hw * 1.3, 0.25, 2.0), col, 0.35 * k)
        cv.light_poly(self._beam_poly(ctx, hw, 0.18, 3.0), col, 0.75 * k)
        cv.light_poly(self._beam_poly(ctx, hw * 0.45, 0.14, 4.0), core, 0.6 * k)
        cv.light_poly(self._beam_poly(ctx, hw * 0.2, 0.06, 5.0), (255, 255, 255), 0.8 * k)
        # Shock rings riding down the beam.
        L = float(np.linalg.norm(self.end - self.src))
        ang = math.degrees(math.atan2(self.dir[1], self.dir[0]))
        for j in range(4):
            s = ((self.t * 2.2 + j / 4.0) % 1.0)
            c = self.src + self.dir * (s * L)
            fade = 1.0 - s
            cv2.ellipse(cv.light, (int(c[0]), int(c[1])), (max(1, int(hw * 0.3)), int(hw * (1.5 + 0.7 * s))),
                        ang, 0, 360, scale_color(col, 0.7 * fade * k), 1, cv2.LINE_AA)
        # Edge lightning.
        nrm = perp(self.dir)
        for side in (-1.0, 1.0):
            a = self.src + nrm * side * hw * 1.3
            b = a + self.dir * (L * np.random.uniform(0.3, 0.7))
            pts = jagged_line(a, b, segments=10, jitter=hw * 0.5)
            cv2.polylines(cv.light, [pts.astype(np.int32)], False, scale_color(col, 1.3 * k), 1, cv2.LINE_AA)
        self._render_orb(ctx, self.src, hw * 1.15, 0.6)

    def _render_blast(self, ctx: Context) -> None:
        """Toward-the-camera mode: a sphere of energy swelling to fill the screen."""
        cv = ctx.canvas
        k = self.power
        col, core = self.color, self.core
        d = ctx.diag
        grow = ease_out(min(1.0, self.t / max(0.05, 0.4 * self.duration(Phase.ACTIVE))))
        fade = 1.0 - smoothstep((self.progress - 0.7) / 0.3)
        R = (0.05 + 0.5 * grow) * d * self.size
        c = self.src
        cv.glow_circle(c, R * 1.25, col, 0.55 * k * fade)
        cv.light_poly(self._noisy_circle(c, R * 0.85, 72, 0.05, ctx.now * 12), col, 0.2 * k * fade)
        cv.light_poly(self._noisy_circle(c, R * 0.55, 56, 0.06, -ctx.now * 10), col, 0.45 * k * fade)
        cv.light_poly(self._noisy_circle(c, R * 0.3, 40, 0.08, ctx.now * 15), col, 0.8 * k * fade)
        cv.light_poly(self._noisy_circle(c, R * 0.14, 28, 0.08, -ctx.now * 9), core, 0.6 * k * fade)
        cv.light_circle(c, R * 0.06, (255, 255, 255), 0.8 * k * fade)
        # Radial streaks rushing outward past the viewer.
        for _ in range(int(22 * self.size)):
            a = np.random.uniform(0, 2 * math.pi)
            dv = np.array([math.cos(a), math.sin(a)], np.float32)
            r0 = R * np.random.uniform(0.5, 0.9)
            r1 = r0 + d * np.random.uniform(0.1, 0.35)
            pts = jagged_line(c + dv * r0, c + dv * r1, segments=4, jitter=6)
            cv2.polylines(cv.light, [pts.astype(np.int32)], False,
                          scale_color(core if np.random.rand() < 0.3 else col, 0.9 * k * fade), 2, cv2.LINE_AA)
        ring(cv, c, R * (0.9 + 0.1 * math.sin(self.t * 30)), col, 3, 0.7 * k * fade)

    def _render_impact(self, ctx: Context) -> None:
        cv = ctx.canvas
        k = self.power
        col, core = self.color, self.core
        p = self.progress
        d = ctx.diag
        c = self.hit
        if not self.camera_mode and p < 0.35:
            # Beam remnant collapsing to a thread.
            hw = 0.06 * min(cv.w, cv.h) * self.size * (1.0 - p / 0.35) * 0.4
            if hw >= 1.0:
                cv.light_poly(self._beam_poly(ctx, hw, 0.2, 0.0), col, k * (1.0 - p / 0.35))
        q = (1.0 - p) ** 2
        cv.glow_circle(c, (0.06 + 0.14 * ease_out(p)) * d * self.size, col, 0.9 * k * q)
        cv.light_circle(c, 0.04 * d * self.size * q + 1, core, 0.8 * k * q)
        for delay, width, cc in ((0.0, 10, core), (0.12, 6, col), (0.25, 4, col)):
            a = (p - delay) / (1.0 - delay)
            if 0.0 <= a < 1.0:
                ring(cv, c, 10 + 0.32 * d * self.size * ease_out(a), cc, max(1, int(width * (1 - a))),
                     1.2 * k * (1 - a))
