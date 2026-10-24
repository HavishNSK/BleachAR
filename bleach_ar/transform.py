"""Transformations: Shikai (sword release) and Bankai (full release).

Both are ``Ability`` state machines with two explicit modes chosen in
``trigger()``:

* ``RELEASE`` - power up (sealed -> shikai, or sealed/shikai -> bankai).
* ``SEAL``    - a short power-down played when the ability is triggered again
  while its level is active (shikai -> sealed, bankai -> sealed).

``duration()`` is mode-aware, so the base class still drives every phase and
zero-length phases are skipped as usual.

Bankai phase mapping (all seconds from ``cfg['abilities']['bankai']``):

    CHARGING = dim + charge   (first ``dim`` s darken, then energy gathers)
    ACTIVE   = release        (flash, burst, shockwaves; level -> BANKAI)
    IMPACT   = title          (title card, lingering energy, dim eases out)
    COOLDOWN = cooldown

``ctx.locked_by == "bankai"`` from CHARGING until COOLDOWN starts.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .abilities import BANKAI, SEALED, SHIKAI, Ability, Context, Phase
from .fx import Canvas, ease_in, ease_out, ipt, jagged_line, perp, ring, scale_color, smoothstep, unit

RELEASE, SEAL = "release", "seal"
WHITE = (255, 255, 255)


def _ring_points(center, radius: float, n: int, jitter: float = 0.0, phase: float = 0.0) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False) + phase
    r = radius + (np.random.uniform(-jitter, jitter, n) if jitter else 0.0)
    return (np.stack([np.cos(a), np.sin(a)], 1) * np.reshape(r, (-1, 1)) + np.asarray(center, np.float32)
            ).astype(np.float32)


def _spiral_in(ctx: Context, center, radius: Tuple[float, float], n: int, color, speed: float = 1.0,
               size=(1.5, 3.2), life=(0.5, 0.9), targets: Optional[np.ndarray] = None) -> None:
    """Emit ``n`` particles on a ring that swirl inward toward ``center`` (or ``targets``)."""
    if n <= 0:
        return
    c = np.asarray(center, np.float32)
    ang = np.random.uniform(0, 2 * math.pi, n)
    rad = np.random.uniform(radius[0], radius[1], n)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    pos = c + dirs * rad[:, None]
    tgt = c if targets is None else targets
    inward = (tgt - pos) / (0.5 * (life[0] + life[1]))
    tangent = np.stack([-dirs[:, 1], dirs[:, 0]], 1) * (rad[:, None] * 1.6)
    ctx.particles.emit(n, pos, (inward * 1.3 + tangent) * speed, life=life, size=size, color=color, drag=1.2)


def _burst(ctx: Context, center, n: int, speed: Tuple[float, float], color, life=(0.5, 1.1),
           size=(1.5, 4.0), drag: float = 2.2, gravity: float = 0.0, spread_pos: float = 0.0) -> None:
    if n <= 0:
        return
    ang = np.random.uniform(0, 2 * math.pi, n)
    sp = np.random.uniform(speed[0], speed[1], n)
    vel = np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None]
    ctx.particles.emit(n, center, vel.astype(np.float32), spread_pos=spread_pos, life=life, size=size,
                       color=color, drag=drag, gravity=gravity)


def _screen_center(ctx: Context) -> np.ndarray:
    return np.array([ctx.canvas.w * 0.5, ctx.canvas.h * 0.5], np.float32)


class Shikai(Ability):
    """Sealed <-> Shikai. Spiralling energy into the blade, then a release flash."""

    name = "shikai"
    label = "Shikai"

    def __init__(self, params: dict):
        super().__init__(params)
        self.mode = RELEASE
        self._a = np.zeros(2, np.float32)  # blade (or focus point) segment
        self._b = np.zeros(2, np.float32)

    # --- state machine ----------------------------------------------------
    def duration(self, phase: Phase) -> float:
        if self.mode == SEAL:
            return {Phase.CHARGING: 0.0, Phase.ACTIVE: float(self.params.get("seal", 0.6)),
                    Phase.IMPACT: 0.0, Phase.COOLDOWN: float(self.params.get("seal_cooldown", 0.5))}[phase]
        return super().duration(phase)

    def can_trigger(self, ctx: Context) -> bool:
        return super().can_trigger(ctx) and ctx.level in (SEALED, SHIKAI)

    def trigger(self, ctx: Context) -> bool:
        if not self.can_trigger(ctx):
            return False
        self.mode = SEAL if ctx.level == SHIKAI else RELEASE
        self._advance_from(-1, ctx)
        return True

    def reset(self, ctx: Context) -> None:
        super().reset(ctx)
        self.mode = RELEASE

    # --- helpers ----------------------------------------------------------
    def _track_blade(self, ctx: Context) -> None:
        s = ctx.sword
        if s.visible and s.length > 1.0:
            self._a, self._b = s.grip.astype(np.float32), s.tip.astype(np.float32)
        else:
            c = ctx.body_center()
            c = _screen_center(ctx) if c is None else c
            self._a = self._b = np.asarray(c, np.float32)

    def _blade_samples(self, n: int) -> np.ndarray:
        t = np.random.uniform(0.0, 1.0, (n, 1)).astype(np.float32)
        return self._a + (self._b - self._a) * t

    def _reach(self, ctx: Context) -> float:
        return max(float(np.linalg.norm(self._b - self._a)), 0.12 * ctx.diag)

    # --- hooks ------------------------------------------------------------
    def on_enter(self, phase: Phase, ctx: Context) -> None:
        color = tuple(self.params["color"])
        if phase in (Phase.CHARGING, Phase.ACTIVE):
            self._track_blade(ctx)
        if phase == Phase.ACTIVE and self.mode == RELEASE:
            if ctx.level != BANKAI:
                ctx.level = SHIKAI
            ctx.canvas.flash = max(ctx.canvas.flash, 0.35)
            ctx.shaker.kick(12)
            n = 160
            pts = self._blade_samples(n)
            d = unit(self._b - self._a)
            side = np.where(np.random.rand(n, 1) < 0.5, -1.0, 1.0).astype(np.float32)
            vel = perp(d) * side * np.random.uniform(250, 700, (n, 1)) + d * np.random.uniform(-120, 260, (n, 1))
            ctx.particles.emit(n, pts, vel.astype(np.float32), spread_pos=4, life=(0.4, 1.0),
                               size=(1.5, 4.0), color=color, drag=2.4)
            ctx.particles.emit(60, self._blade_samples(60), vel[:60] * 0.6, life=(0.3, 0.6),
                               size=(1.2, 2.5), color=WHITE, drag=2.8)
            _burst(ctx, self._b, 50, (150, 500), color, life=(0.4, 0.8))
            ctx.show_title(self.params.get("release_name", "SHIKAI"), sub=self.params.get("command", ""),
                           color=color, duration=max(1.2, self.duration(Phase.ACTIVE)))
        elif phase == Phase.ACTIVE and self.mode == SEAL:
            ctx.level = SEALED
            ctx.canvas.flash = max(ctx.canvas.flash, 0.15)
            ctx.shaker.kick(5)
            # Reverse burst: energy collapses back into the blade.
            r = self._reach(ctx)
            _spiral_in(ctx, (self._a + self._b) * 0.5, (r * 0.8, r * 1.3), 140, color, speed=1.4,
                       life=(0.3, 0.5), targets=self._blade_samples(140))
            ctx.show_title("SEALED", sub="", color=(200, 200, 200), duration=1.0)

    def on_update(self, ctx: Context) -> None:
        color = tuple(self.params["color"])
        if self.phase == Phase.CHARGING:
            self._track_blade(ctx)
            p = self.progress
            ctx.aura_boost = max(ctx.aura_boost, 1.5 * ease_out(p))
            ctx.shaker.kick(1.0 + 5.0 * ease_in(p))
            r = self._reach(ctx)
            n = int(4 + 14 * p)
            mid = (self._a + self._b) * 0.5
            _spiral_in(ctx, mid, (r * 0.7, r * 1.4), n, color, targets=self._blade_samples(n))
            _spiral_in(ctx, mid, (r * 0.5, r * 1.0), n // 3, WHITE, size=(1.0, 2.0),
                       targets=self._blade_samples(n // 3))
            ctx.particles.attract(mid, 260 * p, ctx.dt)
        elif self.phase == Phase.ACTIVE and self.mode == RELEASE:
            self._track_blade(ctx)
            ctx.aura_boost = max(ctx.aura_boost, 1.5 * (1.0 - self.progress))
            if np.random.rand() < 0.6:
                n = 3
                ctx.particles.emit(n, self._blade_samples(n), (0, -60), spread_vel=60, life=(0.4, 0.8),
                                   size=(1.0, 2.5), color=color, drag=1.0)
        elif self.phase == Phase.ACTIVE and self.mode == SEAL:
            self._track_blade(ctx)

    def render(self, ctx: Context) -> None:
        if self.phase == Phase.IDLE:
            return
        cv = ctx.canvas
        color = tuple(self.params["color"])
        a, b = self._a, self._b
        has_blade = float(np.linalg.norm(b - a)) > 1.0
        if self.phase == Phase.CHARGING:
            p = self.progress
            flick = 0.55 + 0.45 * np.random.rand()
            k = (0.3 + 0.9 * p) * flick
            if has_blade:
                self._draw_blade_energy(cv, a, b, color, k, 6 + 10 * p, crackle=p > 0.25)
            else:
                cv.glow_circle(a, 30 + 50 * p, color, 0.6 * k)
                cv.light_circle(a, 6 + 10 * p, WHITE, k)
            # Gathering ring closing in on the focus point.
            mid = (a + b) * 0.5
            rr = self._reach(ctx) * (1.25 - 0.9 * ((self.t * 1.6) % 1.0))
            ring(cv, mid, rr, color, 2, 0.5 * p)
        elif self.phase == Phase.ACTIVE and self.mode == RELEASE:
            p = self.progress
            age = self.t
            # Release shockwave from the blade centre.
            mid = (a + b) * 0.5
            wave = ease_out(min(1.0, age / 0.45))
            if wave < 1.0:
                ring(cv, mid, 20 + 0.35 * ctx.diag * wave, color, max(1, int(10 * (1 - wave))), 1.0 - wave)
                ring(cv, mid, 10 + 0.22 * ctx.diag * wave, WHITE, max(1, int(5 * (1 - wave))), 0.8 * (1 - wave))
            k = 1.2 * (1.0 - smoothstep(p)) + 0.15
            if has_blade:
                self._draw_blade_energy(cv, a, b, color, k, 14 * (1 - p) + 5, crackle=p < 0.4)
            else:
                cv.glow_circle(a, 70 * (1 - p) + 20, color, 0.8 * k)
        elif self.phase == Phase.ACTIVE and self.mode == SEAL:
            k = 1.0 - smoothstep(self.progress)
            if has_blade:
                self._draw_blade_energy(cv, a, b, color, 0.8 * k, 4 + 6 * k, crackle=False)
            mid = (a + b) * 0.5
            ring(cv, mid, self._reach(ctx) * (1.0 - ease_out(self.progress)) + 4, (200, 200, 200), 2, k)

    @staticmethod
    def _draw_blade_energy(cv: Canvas, a, b, color, k: float, width: float, crackle: bool) -> None:
        cv.glow_line(a, b, color, width * 4.0, 0.9 * k)
        cv.light_line(a, b, color, int(width), 0.55 * k)
        cv.light_line(a, b, WHITE, max(1, int(width * 0.3)), 0.8 * k)
        if crackle:
            for _ in range(2):
                pts = jagged_line(a, b, segments=10, jitter=width * 1.4)
                cv2.polylines(cv.light, [pts.astype(np.int32)], False, scale_color(color, k), 1, cv2.LINE_AA)


class Bankai(Ability):
    """The full release: darkening, gathering reiatsu, explosion, title card."""

    name = "bankai"
    label = "Bankai"
    N_EMBERS = 48

    def __init__(self, params: dict):
        super().__init__(params)
        self.mode = RELEASE
        self.center = np.zeros(2, np.float32)
        self._release_t: Optional[float] = None  # ctx.now when the explosion went off
        self._rays: Optional[np.ndarray] = None
        # Persistent ambience (embers + vignette), allocated per resolution.
        self._vig_size: Tuple[int, ...] = (0, 0, 0, 0)
        self._vig_dark: Optional[np.ndarray] = None
        self._vig_dark_tmp: Optional[np.ndarray] = None
        self._vig_edge: Optional[np.ndarray] = None
        self._ember_pos = np.zeros((self.N_EMBERS, 2), np.float32)
        self._ember_vel = np.zeros((self.N_EMBERS, 2), np.float32)
        self._ember_life = np.zeros(self.N_EMBERS, np.float32)
        self._ember_size = np.ones(self.N_EMBERS, np.float32)
        self._ambient = 0.0  # 0..1 strength of the persistent effect

    # --- timing -----------------------------------------------------------
    def _p(self, key: str, default: float) -> float:
        return float(self.params.get(key, default))

    @property
    def dim_time(self) -> float:
        return self._p("dim", 0.6)

    @property
    def dim_amount(self) -> float:
        return self._p("dim_amount", 0.65)

    def duration(self, phase: Phase) -> float:
        if self.mode == SEAL:
            return {Phase.CHARGING: 0.0, Phase.ACTIVE: self._p("seal", 0.9), Phase.IMPACT: 0.0,
                    Phase.COOLDOWN: self._p("seal_cooldown", 0.8)}[phase]
        return {Phase.CHARGING: self.dim_time + self._p("charge", 1.6), Phase.ACTIVE: self._p("release", 0.5),
                Phase.IMPACT: self._p("title", 1.6), Phase.COOLDOWN: self._p("cooldown", 2.0)}[phase]

    # --- state machine ----------------------------------------------------
    def trigger(self, ctx: Context) -> bool:
        if not self.can_trigger(ctx):
            return False
        self.mode = SEAL if ctx.level == BANKAI else RELEASE
        self._advance_from(-1, ctx)
        return True

    def reset(self, ctx: Context) -> None:
        super().reset(ctx)
        self.mode = RELEASE
        self._release_t = None
        self._ember_life.fill(0)
        self._ambient = 0.0
        if ctx.locked_by == self.name:
            ctx.locked_by = None

    def _unlock(self, ctx: Context) -> None:
        if ctx.locked_by == self.name:
            ctx.locked_by = None

    def _track_center(self, ctx: Context) -> None:
        c = ctx.body_center()
        self.center = _screen_center(ctx) if c is None else np.asarray(c, np.float32)

    # --- hooks ------------------------------------------------------------
    def on_enter(self, phase: Phase, ctx: Context) -> None:
        color = tuple(self.params["color"])
        if phase in (Phase.COOLDOWN, Phase.IDLE):
            self._unlock(ctx)
            return
        self._track_center(ctx)
        if self.mode == SEAL:
            if phase == Phase.ACTIVE:
                ctx.locked_by = self.name
                ctx.canvas.flash = max(ctx.canvas.flash, 0.2)
                ctx.shaker.kick(8)
                ctx.show_title("BANKAI RELEASED", sub="", color=(190, 190, 200), duration=1.4)
                _burst(ctx, self.center, 120, (60, 260), color, life=(0.6, 1.2), drag=1.0, gravity=-60,
                       spread_pos=40)
            elif phase == Phase.IMPACT:
                ctx.level = SEALED
            return
        if phase == Phase.CHARGING:
            ctx.locked_by = self.name
            self._release_t = None
        elif phase == Phase.ACTIVE:
            ctx.locked_by = self.name
            ctx.level = BANKAI
            ctx.canvas.flash = 1.0
            ctx.shaker.kick(34)
            self._release_t = ctx.now
            self._rays = np.random.uniform(0, 2 * math.pi, 28).astype(np.float32)
            d = ctx.diag
            _burst(ctx, self.center, 380, (0.35 * d, 1.3 * d), color, life=(0.5, 1.3), size=(2.0, 5.0),
                   drag=2.6)
            _burst(ctx, self.center, 160, (0.25 * d, 0.9 * d), WHITE, life=(0.3, 0.8), size=(1.5, 3.0),
                   drag=3.0)
            _burst(ctx, self.center, 90, (0.1 * d, 0.4 * d), (40, 30, 160), life=(0.8, 1.6), size=(2.5, 5.5),
                   drag=1.6, spread_pos=20)
        elif phase == Phase.IMPACT:
            ctx.locked_by = self.name
            ctx.show_title(self.params.get("release_name", "BANKAI"), sub=self.params.get("sub", ""),
                           color=color, duration=max(1.5, self.duration(Phase.IMPACT) + 0.6))
            ctx.shaker.kick(10)

    def on_update(self, ctx: Context) -> None:
        color = tuple(self.params["color"])
        d = ctx.diag
        if self.mode == SEAL:
            if self.phase == Phase.ACTIVE:
                self._track_center(ctx)
                k = 1.0 - self.progress
                n = int(10 * k) + 1
                ctx.particles.emit(n, self.center, (0, -120), spread_pos=0.08 * d, spread_vel=70,
                                   life=(0.6, 1.2), size=(1.5, 3.5), color=scale_color(color, 0.5 + 0.5 * k),
                                   drag=0.8, gravity=-80)
            return
        if self.phase == Phase.CHARGING:
            self._track_center(ctx)
            ct = self.t - self.dim_time
            if ct < 0:  # darkening: only a few wisps
                p = self.t / max(1e-3, self.dim_time)
                ctx.aura_boost = max(ctx.aura_boost, 0.6 * p)
                _spiral_in(ctx, self.center, (0.3 * d, 0.45 * d), int(2 + 4 * p), color, life=(0.8, 1.2))
                return
            p = min(1.0, ct / max(1e-3, self._p("charge", 1.6)))
            ctx.aura_boost = max(ctx.aura_boost, 0.6 + 2.4 * ease_out(p))
            ctx.shaker.kick(1.5 + 9.0 * ease_in(p))
            n = int(22 + 40 * p)
            _spiral_in(ctx, self.center, (0.25 * d, 0.6 * d), n, color, speed=1.0 + 0.5 * p, life=(0.5, 0.9),
                       size=(2.0, 4.5))
            _spiral_in(ctx, self.center, (0.15 * d, 0.35 * d), n // 3, WHITE, size=(1.2, 2.6), life=(0.4, 0.7))
            ctx.particles.attract(self.center, 900 + 2600 * p, ctx.dt)
            # Energy flowing up the column.
            col_x = self.center[0]
            m = int(3 + 8 * p)
            ys = np.random.uniform(0, ctx.canvas.h, (m, 1))
            xs = col_x + np.random.normal(0, 8 + 30 * p, (m, 1))
            ctx.particles.emit(m, np.hstack([xs, ys]).astype(np.float32), (0, -0.9 * d), spread_vel=40,
                               life=(0.3, 0.6), size=(1.0, 2.5), color=WHITE if np.random.rand() < 0.4 else color,
                               drag=0.5)
        elif self.phase == Phase.ACTIVE:
            ctx.aura_boost = max(ctx.aura_boost, 3.0)
            ctx.shaker.kick(20 * (1.0 - self.progress) + 6)
        elif self.phase == Phase.IMPACT:
            self._track_center(ctx)
            p = self.progress
            ctx.aura_boost = max(ctx.aura_boost, 3.0 * (1.0 - 0.6 * p))
            if np.random.rand() < 0.8:
                ctx.particles.emit(4, self.center, (0, -0.15 * d), spread_pos=0.12 * d, spread_vel=0.06 * d,
                                   life=(0.6, 1.2), size=(1.5, 3.5), color=color, drag=0.8, gravity=-50)

    # --- rendering --------------------------------------------------------
    def render(self, ctx: Context) -> None:
        cv = ctx.canvas
        self._render_ambient(ctx)
        if self.phase in (Phase.IDLE, Phase.COOLDOWN):
            return
        color = tuple(self.params["color"])
        if self.mode == SEAL:
            if self.phase == Phase.ACTIVE:
                k = 1.0 - self.progress
                cv.dim = max(cv.dim, 0.3 * k)
                ring(cv, self.center, 30 + 0.45 * ctx.diag * ease_out(self.progress), color, 3, k)
            return
        if self.phase == Phase.CHARGING:
            if self.t < self.dim_time:
                p = smoothstep(self.t / max(1e-3, self.dim_time))
                cv.dim = max(cv.dim, self.dim_amount * p)
                ring(cv, self.center, ctx.diag * (0.5 - 0.15 * p), color, 2, 0.3 * p)
                return
            cv.dim = max(cv.dim, self.dim_amount)
            p = min(1.0, (self.t - self.dim_time) / max(1e-3, self._p("charge", 1.6)))
            self._render_column(ctx, p)
            self._render_gather(ctx, p, color)
        elif self.phase == Phase.ACTIVE:
            cv.dim = max(cv.dim, self.dim_amount)
            self._render_column(ctx, 1.0 - self.progress, release=True)
        elif self.phase == Phase.IMPACT:
            p = self.progress
            cv.dim = max(cv.dim, self.dim_amount * (1.0 - smoothstep(p)))
            linger = 1.0 - smoothstep(p / 0.6)
            if linger > 0.01:
                self._render_column(ctx, 0.35, strength=linger, rim=False)
            cv.glow_circle(self.center, 0.25 * ctx.diag, color, 0.5 * (1.0 - p))
        if self._release_t is not None:
            self._render_explosion(ctx, ctx.now - self._release_t, color)

    def _render_column(self, ctx: Context, p: float, release: bool = False, strength: float = 1.0,
                       rim: bool = True) -> None:
        """Pillar of reiatsu rising through the body: dark rim, red body, white core."""
        cv = ctx.canvas
        color = tuple(self.params["color"])
        rim_color = tuple(self.params.get("rim_color", (30, 20, 20)))
        x = float(self.center[0])
        h = cv.h
        base_w = (0.025 + 0.11 * ease_out(p)) * cv.w * (1.6 if release else 1.0)
        n = 14
        ys = np.linspace(h + 10, -10, n).astype(np.float32)
        wob = 0.18 * np.sin(ys * 0.035 + ctx.now * 22.0) + 0.1 * np.random.uniform(-1, 1, n)
        taper = 0.75 + 0.25 * np.linspace(1, 0, n)
        half = base_w * 0.5 * taper * (1 + wob)

        def band(scale: float) -> np.ndarray:
            left = np.stack([x - half * scale, ys], 1)
            right = np.stack([x + half * scale, ys], 1)
            return np.concatenate([left, right[::-1]], 0).astype(np.float32)

        flick = (0.8 + 0.2 * np.random.rand()) * strength
        if rim:
            cv2.fillPoly(cv.frame, [band(1.25).astype(np.int32)], rim_color, cv2.LINE_AA)
        cv.glow_poly(band(2.2), color, 0.9 * flick * (0.4 + 0.6 * p))
        cv.light_poly(band(1.0), color, 0.75 * flick)
        cv.light_poly(band(0.35), WHITE, (0.5 + 0.5 * p) * flick)
        for _ in range(2):
            y0 = np.random.uniform(0, h)
            pts = jagged_line((x + np.random.uniform(-1, 1) * base_w, y0),
                              (x + np.random.uniform(-1, 1) * base_w * 1.6, y0 - np.random.uniform(40, 140)),
                              segments=5, jitter=10)
            cv2.polylines(cv.light, [pts.astype(np.int32)], False, scale_color(color, strength), 2, cv2.LINE_AA)

    def _render_gather(self, ctx: Context, p: float, color) -> None:
        cv = ctx.canvas
        c = self.center
        d = ctx.diag
        for k in range(3):
            ph = (self.t * 1.4 + k / 3.0) % 1.0
            r = d * 0.45 * (1.0 - ph) + 6
            ring(cv, c, r, color, max(1, int(1 + 3 * ph)), 0.25 + 0.6 * ph * p)
        cv.glow_circle(c, 40 + 120 * p, color, 0.5 + 0.8 * p)
        cv.light_circle(c, 8 + 26 * ease_in(p), WHITE, 0.6 + 0.4 * p)

    def _render_explosion(self, ctx: Context, age: float, color) -> None:
        cv = ctx.canvas
        c = self.center
        d = ctx.diag
        for k, (delay, width, col) in enumerate(((0.0, 22, WHITE), (0.08, 14, color), (0.2, 8, color))):
            a = (age - delay) / 1.1
            if 0.0 <= a < 1.0:
                e = ease_out(a)
                ring(cv, c, 20 + d * 0.95 * e, col, max(1, int(width * (1 - a))), 1.2 * (1 - a))
        a = age / 0.6
        if a < 1.0 and self._rays is not None:
            reach = d * (0.15 + 0.9 * ease_out(a))
            s = (1.0 - a) * 1.2
            for ang in self._rays:
                dv = np.array([math.cos(ang), math.sin(ang)], np.float32)
                pts = jagged_line(c + dv * reach * 0.15, c + dv * reach, segments=6, jitter=14)
                cv2.polylines(cv.light, [pts.astype(np.int32)], False, scale_color(color, s), 3, cv2.LINE_AA)
                cv2.polylines(cv.light, [pts[:4].astype(np.int32)], False, scale_color(WHITE, s), 1, cv2.LINE_AA)
            cv.glow_circle(c, d * 0.35 * (1 - a) + 30, color, 1.4 * (1 - a))

    # --- persistent Bankai ambience ----------------------------------------
    def _ensure_vignette(self, cv: Canvas) -> None:
        if self._vig_size == (cv.w, cv.h, cv.gw, cv.gh) and self._vig_dark is not None:
            return
        self._vig_size = (cv.w, cv.h, cv.gw, cv.gh)
        # Built at glow resolution and upsampled: the mask is smooth, so nothing is lost.
        yy, xx = np.mgrid[0:cv.gh, 0:cv.gw].astype(np.float32)
        r = np.sqrt(((xx - cv.gw / 2) / (cv.gw / 2)) ** 2 + ((yy - cv.gh / 2) / (cv.gh / 2)) ** 2) / math.sqrt(2)
        small = np.clip((r - 0.35) / 0.65, 0, 1) ** 1.6
        v = cv2.resize(small, (cv.w, cv.h), interpolation=cv2.INTER_LINEAR)
        self._vig_dark = cv2.merge([(v * 255).astype(np.uint8)] * 3)
        self._vig_dark_tmp = np.empty_like(self._vig_dark)
        small = small ** 2
        col = np.asarray(self.params["color"], np.float32)
        self._vig_edge = (small[..., None] * col).astype(np.float32)

    def _render_ambient(self, ctx: Context) -> None:
        """Dark pulsing vignette with red edges and falling black-red embers while in Bankai."""
        target = 0.0
        if ctx.level == BANKAI:
            target = 1.0
            if self.mode == RELEASE and self.phase == Phase.IMPACT:
                target = smoothstep(self.progress)
            elif self.mode == RELEASE and self.phase in (Phase.CHARGING, Phase.ACTIVE):
                target = 0.0
            elif self.mode == SEAL and self.phase == Phase.ACTIVE:
                target = 1.0 - self.progress
        self._ambient = target
        cv = ctx.canvas
        dt = ctx.dt
        if self._ambient <= 0.01:
            self._ember_life.fill(0)
            return
        self._ensure_vignette(cv)
        pulse = 0.5 + 0.5 * math.sin(ctx.now * 2.6)
        k = self._ambient
        cv2.convertScaleAbs(self._vig_dark, self._vig_dark_tmp, alpha=k * (0.55 + 0.15 * pulse))
        cv2.subtract(cv.frame, self._vig_dark_tmp, dst=cv.frame)
        cv2.scaleAdd(self._vig_edge, k * (0.25 + 0.35 * pulse), cv.glow, dst=cv.glow)
        self._update_embers(cv, dt, k)

    def _update_embers(self, cv: Canvas, dt: float, k: float) -> None:
        life = self._ember_life
        dead = np.nonzero(life <= 0)[0]
        spawn = np.random.poisson(6.0 * dt * 2.0)
        if spawn and dead.size:
            idx = dead[:spawn]
            n = idx.size
            self._ember_pos[idx, 0] = np.random.uniform(0, cv.w, n)
            self._ember_pos[idx, 1] = np.random.uniform(-20, cv.h * 0.3, n)
            self._ember_vel[idx, 0] = np.random.uniform(-25, 25, n)
            self._ember_vel[idx, 1] = np.random.uniform(35, 90, n)
            self._ember_life[idx] = np.random.uniform(2.5, 5.0, n)
            self._ember_size[idx] = np.random.uniform(1.5, 3.5, n)
        alive = np.nonzero(life > 0)[0]
        if alive.size == 0:
            return
        life[alive] -= dt
        self._ember_vel[alive, 0] += np.random.uniform(-30, 30, alive.size) * dt
        self._ember_pos[alive] += self._ember_vel[alive] * dt
        rim = scale_color(self.params["color"], 0.9 * k)
        for i in alive:
            p = ipt(self._ember_pos[i])
            if not (0 <= p[0] < cv.w and 0 <= p[1] < cv.h):
                life[i] = 0
                continue
            fade = min(1.0, life[i] / 1.0)
            r = max(1, int(self._ember_size[i]))
            cv2.circle(cv.frame, p, r + 1, (12, 8, 10), -1, cv2.LINE_AA)
            cv2.circle(cv.light, p, max(1, r - 1), scale_color(rim, fade), -1, cv2.LINE_AA)
