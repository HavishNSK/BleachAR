"""Getsuga Tensho: a crescent of spiritual energy launched from the blade.

The projectile fakes depth: it travels along its launch direction while also
drifting toward a vanishing point a little above screen centre and shrinking
(1.0 -> 0.35 scale), so it reads as flying away from the user rather than
sliding across the glass. ``size`` scales the crescent, ``speed`` the travel
distance and ``intensity`` the brightness and particle counts.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from . import fx
from .abilities import BANKAI, Ability, Context, Phase

AFTERIMAGES = (0.06, 0.12, 0.18)  # seconds behind the head


class GetsugaTensho(Ability):
    name = "getsuga"
    label = "Getsuga Tensho"

    def __init__(self, params: dict):
        super().__init__(params)
        self._clear()

    def _clear(self) -> None:
        self.p0 = np.zeros(2, np.float32)
        self.dir = np.array([0.0, -1.0], np.float32)
        self.vanish = np.zeros(2, np.float32)
        self.dist = 0.0
        self.radius = 0.0
        self.end = np.zeros(2, np.float32)
        self.end_scale = 0.35
        self.dark = False
        self.swing_dir: Optional[np.ndarray] = None
        self.swing_peak = 0.0

    # --- params -------------------------------------------------------------
    def _p(self, key: str, default: float = 1.0) -> float:
        return float(self.params.get(key, default))

    def _colors(self, ctx: Context) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if self.dark or ctx.level == BANKAI:
            return tuple(ctx.energy_color()), (10, 6, 8)
        return tuple(self.params.get("color", (255, 230, 170))), tuple(self.params.get("core_color", (255, 255, 255)))

    # --- flight path -----------------------------------------------------------
    def _flight(self, t: float) -> Tuple[np.ndarray, float]:
        """Head position and perspective scale at normalised flight time ``t``."""
        t = float(np.clip(t, 0.0, 1.0))
        travel = fx.ease_out(t) * 0.6 + t * 0.4
        straight = self.p0 + self.dir * self.dist * travel
        pull = 0.55 * fx.smoothstep(t)
        pos = straight * (1.0 - pull) + self.vanish * pull
        scale = 1.0 - 0.65 * fx.smoothstep(t)
        return pos.astype(np.float32), scale

    def _shape(self, t: float):
        pos, sc = self._flight(t)
        d = self.dir
        r = self.radius * sc
        # The crescent's outer arc passes through the head position.
        center = pos - d * r * 0.85
        return pos, sc, d, r, center

    # --- state hooks -------------------------------------------------------------
    def _note_swing(self, ctx: Context) -> None:
        """Remember the fastest slash direction seen since the trigger."""
        sw = ctx.sword
        if sw.confidence > 0.2 and sw.speed > max(0.6, self.swing_peak):
            self.swing_peak = sw.speed
            self.swing_dir = fx.unit(sw.motion)

    def _origin(self, ctx: Context) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        sw = ctx.sword
        if sw.confidence > 0.2 and sw.length > 1.0:
            return (sw.grip + (sw.tip - sw.grip) * 0.8).astype(np.float32), sw.direction
        c = ctx.body_center()
        if c is None:
            c = np.array([ctx.canvas.w * 0.5, ctx.canvas.h * 0.6], np.float32)
        return np.asarray(c, np.float32), None

    def on_enter(self, phase: Phase, ctx: Context) -> None:
        canvas = ctx.canvas
        inten = self._p("intensity")
        if phase == Phase.CHARGING:
            self.dark = ctx.level == BANKAI
            self.swing_dir, self.swing_peak = None, 0.0
            self._note_swing(ctx)
        elif phase == Phase.ACTIVE:
            self.dark = ctx.level == BANKAI
            self._note_swing(ctx)
            origin, blade = self._origin(ctx)
            screen_c = np.array([canvas.w * 0.5, canvas.h * 0.5], np.float32)
            self.vanish = np.array([canvas.w * 0.5, canvas.h * 0.42], np.float32)
            if self.swing_dir is not None:
                # Follow the slash, nudged toward the vanishing point so it stays on screen.
                d = fx.unit(self.swing_dir * 0.75 + fx.unit(self.vanish - origin) * 0.25)
            elif blade is not None:
                d = fx.perp(blade)
                if float(np.dot(d, screen_c - origin)) < 0:
                    d = -d
            else:
                d = fx.unit(np.array([0.0, -1.0], np.float32))
            self.dir = d.astype(np.float32)
            self.p0 = origin
            self.dist = 0.45 * ctx.diag * self._p("speed")
            self.radius = 0.11 * ctx.diag * self._p("size")
            body, _ = self._colors(ctx)
            ctx.particles.emit(int(40 * inten), origin, d * self.radius * 3.0, spread_pos=self.radius * 0.15,
                               spread_vel=self.radius * 1.6, life=(0.2, 0.5), size=(1.5, 3.5),
                               color=body, drag=3.0)
            canvas.flash = max(canvas.flash, 0.12 * inten)
            ctx.shaker.kick(9.0 * inten)
        elif phase == Phase.IMPACT:
            self.end, self.end_scale = self._flight(1.0)
            body, _ = self._colors(ctx)
            r = self.radius * self.end_scale
            ctx.particles.emit(int(110 * inten), self.end, (0.0, 0.0), spread_pos=r * 0.2,
                               spread_vel=r * 5.0, life=(0.3, 0.8), size=(1.5, 4.0), color=body, drag=2.5)
            ctx.particles.emit(int(40 * inten), self.end, (0.0, 0.0), spread_pos=r * 0.1,
                               spread_vel=r * 3.0, life=(0.15, 0.4), size=(1.0, 2.5),
                               color=(255, 255, 255), drag=3.0)
            canvas.flash = max(canvas.flash, 0.35 * min(1.5, inten))
            ctx.shaker.kick(16.0 * inten)

    def on_update(self, ctx: Context) -> None:
        inten = self._p("intensity")
        body, _ = self._colors(ctx)
        if self.phase == Phase.CHARGING:
            self._note_swing(ctx)
            ctx.aura_boost = max(ctx.aura_boost, 1.6 * self.progress * inten)
            sw = ctx.sword
            if sw.confidence > 0.2 and sw.length > 1.0:
                k = np.random.uniform(0.2, 1.0, (int(10 * inten), 1)).astype(np.float32)
                on_blade = sw.grip + (sw.tip - sw.grip) * k
                n = fx.perp(sw.direction)
                side = np.random.choice([-1.0, 1.0], (len(k), 1)).astype(np.float32)
                start = on_blade + n * side * sw.length * np.random.uniform(0.2, 0.45, (len(k), 1))
                ctx.particles.emit(len(k), start, (on_blade - start) * 3.0, life=(0.2, 0.35),
                                   size=(1.2, 2.8), color=body, drag=0.5)
                ctx.particles.attract(sw.grip + (sw.tip - sw.grip) * 0.6, sw.length * 2.0, ctx.dt)
            else:
                origin, _ = self._origin(ctx)
                ang = np.random.uniform(0, 2 * math.pi, int(6 * inten))
                r = 0.12 * ctx.diag
                start = origin + np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32) * r
                ctx.particles.emit(len(ang), start, (origin - start) * 3.0, life=(0.2, 0.35),
                                   size=(1.2, 2.8), color=body, drag=0.5)
        elif self.phase == Phase.ACTIVE:
            pos, sc, d, r, center = self._shape(self.progress)
            pts = fx.crescent(center, d, r * 1.02, r * 0.3, span_deg=150, n=24)[:24]
            idx = np.random.randint(0, len(pts), int(10 * inten))
            back = -d * r * 2.5
            ctx.particles.emit(len(idx), pts[idx], back, spread_pos=r * 0.04, spread_vel=r * 1.2,
                               life=(0.15, 0.45), size=(1.0, 2.6 * max(0.5, sc)), color=body, drag=2.0)

    def reset(self, ctx: Context) -> None:
        super().reset(ctx)
        self._clear()

    # --- rendering -------------------------------------------------------------------
    def render(self, ctx: Context) -> None:
        if self.phase == Phase.CHARGING:
            self._render_charge(ctx)
        elif self.phase == Phase.ACTIVE:
            self._render_flight(ctx)
        elif self.phase == Phase.IMPACT:
            self._render_impact(ctx)

    def _render_charge(self, ctx: Context) -> None:
        body, _ = self._colors(ctx)
        k = (0.3 + 0.7 * fx.smoothstep(self.progress)) * self._p("intensity")
        k *= 0.85 + 0.15 * math.sin(ctx.now * 31.0)
        sw = ctx.sword
        canvas = ctx.canvas
        if sw.confidence > 0.2 and sw.length > 1.0:
            canvas.glow_line(sw.grip, sw.tip, body, sw.length * 0.12, 0.9 * k)
            canvas.light_line(sw.grip + (sw.tip - sw.grip) * 0.15, sw.tip, (255, 255, 255), 2, 0.5 * k)
        else:
            origin, _ = self._origin(ctx)
            canvas.glow_circle(origin, 0.05 * ctx.diag * (0.5 + k), body, 1.2 * k)

    def _crescent_layers(self, ctx: Context, t: float, strength: float, full: bool) -> None:
        canvas = ctx.canvas
        body, core = self._colors(ctx)
        pos, sc, d, r, center = self._shape(t)
        th = r * (0.42 if self.dark else 0.3)
        inten = self._p("intensity") * strength
        canvas.glow_poly(fx.crescent(center + d * r * 0.04, d, r * 1.08, th * 1.5, span_deg=158), body,
                         (0.3 if self.dark and full else 0.45) * inten)
        shape = fx.crescent(center, d, r, th, span_deg=150)
        canvas.light_poly(shape, body, 0.55 * inten)
        if not full:
            return
        if self.dark:
            core_pts = fx.crescent(center + d * r * 0.01, d, r * 0.975, th * 0.7, span_deg=140)
            # Bankai: black core with a crimson rim. Light and frame are both cut out.
            cv2.fillPoly(canvas.light, [core_pts.astype(np.int32)], (0, 0, 0), cv2.LINE_AA)
            cv2.fillPoly(canvas.frame, [core_pts.astype(np.int32)], core, cv2.LINE_AA)
            rim = fx.crescent(center, d, r * 1.01, th * 1.05, span_deg=152)
            cv2.polylines(canvas.light, [rim.astype(np.int32)], True,
                          fx.scale_color(body, 1.3 * inten * canvas.intensity), max(1, int(r * 0.03)),
                          cv2.LINE_AA)
            k = 0.9 * inten * canvas.intensity
            cv2.polylines(canvas.glow, [(rim * canvas.gs).astype(np.int32)], True,
                          (body[0] * k, body[1] * k, body[2] * k), max(1, int(r * 0.08 * canvas.gs)), cv2.LINE_AA)
        else:
            core_pts = fx.crescent(center + d * r * 0.02, d, r * 0.985, th * 0.38, span_deg=132)
            canvas.light_poly(core_pts, core, 0.95 * inten)
            # Hot leading edge.
            edge = shape[: len(shape) // 2]
            cv2.polylines(canvas.light, [edge.astype(np.int32)], False, (255, 255, 255),
                          max(1, int(r * 0.025)), cv2.LINE_AA)
        # Ragged energy wisps trailing from the inner arc (thin flickering triangles).
        inner = shape[len(shape) // 2:]
        rng = np.random.default_rng(int(ctx.now * 60) & 0xFFFF)
        side = fx.perp(d)
        m = len(inner)
        for i in rng.choice(np.arange(3, m - 3), size=min(7, m - 6), replace=False):
            ln = r * rng.uniform(0.06, 0.32) * math.sin(math.pi * i / m)
            a = inner[i]
            w = r * 0.035
            tipp = a - d * ln + side * rng.uniform(-0.25, 0.25) * ln
            tri = np.array([a + side * w, a - side * w, tipp], np.float32)
            canvas.light_poly(tri, body, 0.55 * inten)

    def _render_flight(self, ctx: Context) -> None:
        t = self.progress
        dur = max(1e-3, self.duration(Phase.ACTIVE))
        for i, lag in reversed(list(enumerate(AFTERIMAGES))):
            tt = t - lag / dur
            if tt > 0.0:
                self._crescent_layers(ctx, tt, 0.45 / (i + 1.5), full=False)
        fade_in = min(1.0, t / 0.08)
        self._crescent_layers(ctx, t, fade_in, full=True)

    def _render_impact(self, ctx: Context) -> None:
        canvas = ctx.canvas
        body, core = self._colors(ctx)
        hot = body if self.dark else core
        t = self.progress
        inten = self._p("intensity")
        r0 = self.radius * max(0.5, self.end_scale)
        fade = 1.0 - fx.ease_in(t)
        rad = r0 * (0.4 + 2.0 * fx.ease_out(t))
        fx.ring(canvas, self.end, rad, body, max(1, int(r0 * 0.1 * fade + 1)), 0.65 * fade * inten)
        fx.ring(canvas, self.end, rad * 0.8, hot, max(1, int(r0 * 0.03 * fade + 1)), 0.35 * fade * fade * inten)
        canvas.glow_circle(self.end, r0 * (0.6 + 0.8 * t), body, 0.7 * fade * fade * inten)
        # Radial streaks flying out of the blast.
        rng = np.random.default_rng(int(self.end[0] * 7 + self.end[1]) & 0xFFFF)
        angs = rng.uniform(0, 2 * math.pi, 14)
        lens = rng.uniform(0.6, 1.4, 14)
        for a, ln in zip(angs, lens):
            v = np.array([math.cos(a), math.sin(a)], np.float32)
            p0 = self.end + v * rad * (0.3 + 0.4 * t)
            p1 = self.end + v * rad * (0.6 + 0.45 * ln)
            canvas.light_line(p0, p1, body, max(1, int(r0 * 0.04)), 0.6 * fade * inten)
        if t < 0.3:
            k = 1.0 - t / 0.3
            canvas.light_circle(self.end, r0 * 0.25 * (1.0 + 2.0 * t), hot, k * inten)
