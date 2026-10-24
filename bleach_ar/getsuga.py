"""Getsuga Tensho: a crescent of spiritual energy launched from the blade.

Beats (the first three map onto the shared Ability phases):

* CHARGING - energy gathers on the blade: the edge glows, motes converge on
  it, a ring contracts onto the launch point (80% along grip -> tip).
* ACTIVE   - the first ``LAUNCH`` seconds are the launch beat (tip flare,
  cross streak, shock ring at the origin, the crescent growing out of the
  blade); then it travels, dragging a tapered energy ribbon and afterimages.
* IMPACT   - shockwave rings, radial streaks, a splash of the crescent and a
  particle burst where the flight ended.

The projectile fakes depth: it travels along its launch direction while also
drifting toward a vanishing point a little above screen centre and shrinking
(1.0 -> 0.35 scale), so it reads as flying away from the user rather than
sliding across the glass. ``size`` scales the crescent, ``speed`` the travel
distance and ``intensity`` the brightness and particle counts.

Everything derived from tracking goes through ``_finite`` before it is used,
and the flight path is a pure function of values frozen at launch, so a
tracking dropout can neither crash cv2 drawing nor make the slash teleport.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from . import fx
from .abilities import BANKAI, Ability, Context, Phase

AFTERIMAGES = (0.05, 0.10, 0.16)  # seconds behind the head
LAUNCH = 0.14  # seconds at the start of ACTIVE that play the launch beat
TRAIL = 0.34  # length of the energy ribbon, in normalised flight time
SWING_MIN = 0.6  # tip speed (frame diagonals / s) that counts as a slash
SWING_WINDOW = 0.25  # seconds of pre-trigger swing history considered
ANCHOR_TAU = 0.03  # seconds; EMA time constant for the charge anchor (~1 frame)
COORD_LIMIT = 1e4  # clamp for anything handed to cv2 (int32-safe, off-screen ok)

SRC_FALLBACK, SRC_HAND, SRC_SWORD = 0, 1, 2


def _finite(v, fallback=None) -> Optional[np.ndarray]:
    """``v`` as a clamped float32 2-vector, or ``fallback`` if it is not finite."""
    try:
        a = np.asarray(v, np.float32).reshape(-1)[:2]
    except (TypeError, ValueError):
        return fallback
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        return fallback
    return np.clip(a, -COORD_LIMIT, COORD_LIMIT).astype(np.float32)


class GetsugaTensho(Ability):
    name = "getsuga"
    label = "Getsuga Tensho"

    def __init__(self, params: dict):
        super().__init__(params)
        self._clear()

    def _clear(self) -> None:
        self._clear_shot()
        # (time, speed, unit motion) of recent fast tip motion; sampled even
        # while IDLE so the slash that fired the gesture is still known.
        self._swings: deque = deque(maxlen=24)

    def _clear_shot(self) -> None:
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
        self.trigger_time = 0.0
        # Last known good launch anchor (smoothed during CHARGING).
        self.anchor: Optional[np.ndarray] = None
        self.anchor_blade: Optional[np.ndarray] = None
        self.anchor_len = 0.0
        self.anchor_src = SRC_FALLBACK

    # --- params -------------------------------------------------------------
    def _p(self, key: str, default: float = 1.0) -> float:
        try:
            v = float(self.params.get(key, default))
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    def _colors(self, ctx: Context) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if self.dark or ctx.level == BANKAI:
            return tuple(ctx.energy_color()), (10, 6, 8)
        return tuple(self.params.get("color", (255, 230, 170))), tuple(self.params.get("core_color", (255, 255, 255)))

    @staticmethod
    def _deep(body) -> Tuple[float, float, float]:
        """More saturated version of ``body`` for the soft glow, so the halo
        reads as blue (or crimson) instead of bleaching to white under bloom."""
        c = np.asarray(body, np.float32)
        m = float(c.mean())
        return tuple(float(x) for x in np.clip((c - m) * 2.2 + m * 0.75, 0, 255))

    # --- tracking -> anchor ------------------------------------------------------
    @staticmethod
    def _sword_ok(ctx: Context) -> bool:
        sw = ctx.sword
        return (sw.confidence > 0.2 and math.isfinite(sw.length) and sw.length > 1.0
                and _finite(sw.grip) is not None and _finite(sw.tip) is not None)

    def _measure(self, ctx: Context):
        """Best anchor visible this frame: (point, blade unit dir or None, length, source) or None."""
        if self._sword_ok(ctx):
            sw = ctx.sword
            grip, tip = _finite(sw.grip), _finite(sw.tip)
            blade = tip - grip
            n = float(np.linalg.norm(blade))
            if n > 1.0:
                return grip + blade * 0.8, (blade / n).astype(np.float32), n, SRC_SWORD
        for side in ("right", "left"):
            part = ctx.tracking.hands.get(side)
            if part is not None and part.ok and len(part.points) > 9:
                p = _finite(part.points[9, :2])
                if p is not None:
                    return p, None, 0.0, SRC_HAND
        return None

    def _fallback_point(self, ctx: Context) -> np.ndarray:
        bc = ctx.body_center()
        c = _finite(bc) if bc is not None else None
        if c is None:
            c = np.array([ctx.canvas.w * 0.5, ctx.canvas.h * 0.6], np.float32)
        return c

    def _update_anchor(self, ctx: Context, snap: bool = False) -> None:
        """Track the launch point. Never downgrades the source (sword -> hand ->
        body) mid-charge: when tracking drops we keep the last good anchor."""
        m = self._measure(ctx)
        if m is None or (self.anchor is not None and m[3] < self.anchor_src):
            if self.anchor is None:
                self.anchor, self.anchor_src = self._fallback_point(ctx), SRC_FALLBACK
            return
        p, blade, length, src = m
        if self.anchor is None or snap:
            self.anchor = p
            self.anchor_blade = blade
            self.anchor_len = length
        else:
            # Light adaptive EMA: jitter-sized moves are smoothed (~1 frame time
            # constant), real moves pass through almost immediately.
            a = 1.0 - math.exp(-max(0.0, ctx.dt) / ANCHOR_TAU)
            moved = float(np.linalg.norm(p - self.anchor)) / (0.01 * ctx.diag)
            a += (1.0 - a) * min(1.0, moved)
            self.anchor = (self.anchor + (p - self.anchor) * a).astype(np.float32)
            if blade is not None:
                prev = self.anchor_blade if self.anchor_blade is not None else blade
                self.anchor_blade = fx.unit(prev + (blade - prev) * a)
                self.anchor_len += (length - self.anchor_len) * a
        self.anchor_src = src

    # --- swing history ------------------------------------------------------------
    def _sample_swing(self, ctx: Context) -> None:
        sw = ctx.sword
        keep = SWING_WINDOW + self.duration(Phase.CHARGING)
        while self._swings and not (0.0 <= ctx.now - self._swings[0][0] <= keep):
            self._swings.popleft()
        if not self._sword_ok(ctx) or not (math.isfinite(sw.speed) and sw.speed > SWING_MIN):
            return
        mv = _finite(sw.motion)
        if mv is None or float(np.linalg.norm(mv)) < 1e-3:
            return
        self._swings.append((ctx.now, float(sw.speed), fx.unit(mv)))

    def _best_swing(self) -> None:
        """Fastest slash from shortly before the trigger up to now."""
        self.swing_dir, self.swing_peak = None, 0.0
        for t, spd, mv in self._swings:
            if t >= self.trigger_time - SWING_WINDOW - 1e-6 and spd > self.swing_peak:
                self.swing_peak, self.swing_dir = spd, mv

    def update(self, ctx: Context) -> None:
        self._sample_swing(ctx)
        super().update(ctx)

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

    def _grow(self, t: float) -> float:
        """The crescent grows out of the blade during the launch beat."""
        lt = t * max(1e-3, self.duration(Phase.ACTIVE)) / LAUNCH
        return 0.5 + 0.5 * fx.ease_out(lt)

    def _shape(self, t: float):
        pos, sc = self._flight(t)
        d = self.dir
        r = self.radius * sc * self._grow(t)
        # The crescent's outer arc passes through the head position.
        center = pos - d * r * 0.85
        return pos, sc, d, r, center

    # --- state hooks -------------------------------------------------------------
    def on_enter(self, phase: Phase, ctx: Context) -> None:
        canvas = ctx.canvas
        inten = self._p("intensity")
        if phase == Phase.CHARGING:
            self._clear_shot()
            self.dark = ctx.level == BANKAI
            self.trigger_time = ctx.now
            self._update_anchor(ctx, snap=True)
        elif phase == Phase.ACTIVE:
            self.dark = ctx.level == BANKAI
            if self.anchor is None:  # zero-length charge
                self.trigger_time = ctx.now
            self._update_anchor(ctx, snap=self.anchor is None)
            self._best_swing()
            origin = self.anchor
            self.vanish = np.array([canvas.w * 0.5, canvas.h * 0.42], np.float32)
            self.p0 = origin.copy()
            self.dist = max(0.0, 0.45 * ctx.diag * self._p("speed"))
            self.radius = max(1.0, 0.11 * ctx.diag * self._p("size"))
            self.dir = self._keep_on_screen(ctx, self._launch_dir(ctx, origin))
            body, _ = self._colors(ctx)
            d = self.dir
            ctx.particles.emit(int(40 * inten), origin, d * self.radius * 3.0, spread_pos=self.radius * 0.15,
                               spread_vel=self.radius * 1.6, life=(0.2, 0.5), size=(1.5, 3.5),
                               color=body, drag=3.0)
            # Sideways spray along the blade edge: the energy "peels off" the sword.
            side = fx.perp(d)
            ctx.particles.emit(int(16 * inten), origin, side * self.radius * 2.5, spread_pos=self.radius * 0.1,
                               spread_vel=self.radius * 0.8, life=(0.12, 0.3), size=(1.0, 2.2),
                               color=(255, 255, 255), drag=4.0)
            ctx.particles.emit(int(16 * inten), origin, -side * self.radius * 2.5, spread_pos=self.radius * 0.1,
                               spread_vel=self.radius * 0.8, life=(0.12, 0.3), size=(1.0, 2.2),
                               color=(255, 255, 255), drag=4.0)
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
        elif phase == Phase.COOLDOWN:
            # Nothing of ours renders in cooldown; drop the shot (keep swing history).
            self._clear_shot()

    def _launch_dir(self, ctx: Context, origin: np.ndarray) -> np.ndarray:
        """Direction rule.

        1. A real slash wins: the crescent flies the way the tip was moving at
           its fastest (from just before the trigger to release).
        2. Otherwise fire edge-first, perpendicular to the blade, on the side
           facing the open screen. That is how the wave leaves the sword in the
           anime (the crescent forms along the edge) and it matches the
           crescent's own orientation (its chord lies along the blade). Firing
           along the blade's point was considered but rejected: with the usual
           upright grip it would just shoot off the top of the frame.
        3. No blade: toward the vanishing point (straight up if already there).

        ``_keep_on_screen`` then bends the result toward the vanishing point
        only as much as needed for the impact to land inside the frame.
        """
        to_vp = self.vanish - origin
        if self.swing_dir is not None:
            d = self.swing_dir
        elif self.anchor_blade is not None:
            d = fx.perp(self.anchor_blade)
            screen_c = np.array([ctx.canvas.w * 0.5, ctx.canvas.h * 0.5], np.float32)
            side = float(np.dot(d, screen_c - origin))
            if abs(side) < 1e-3 * ctx.diag:
                side = -float(d[1])  # dead centre: prefer the upward side
            if side < 0:
                d = -d
        else:
            d = fx.unit(to_vp) if float(np.linalg.norm(to_vp)) > 1.0 else np.array([0.0, -1.0], np.float32)
        d = _finite(d, np.array([0.0, -1.0], np.float32))
        return fx.unit(d)

    def _keep_on_screen(self, ctx: Context, d: np.ndarray) -> np.ndarray:
        """Smallest blend toward the vanishing point that keeps the impact in frame."""
        w, h = ctx.canvas.w, ctx.canvas.h
        if w <= 0 or h <= 0:
            return d
        m = 0.05 * min(w, h) + 0.25 * self.radius  # end-of-flight crescent mostly visible
        to_vp = fx.unit(self.vanish - self.p0) if float(np.linalg.norm(self.vanish - self.p0)) > 1.0 else d
        full = self.dist
        # Prefer keeping the user's direction and landing short; bend only if
        # even a short throw would leave the frame.
        for k in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            dk = fx.unit(d * (1.0 - k) + to_vp * k)
            self.dir = dk
            for f in (1.0, 0.8, 0.6):
                self.dist = full * f
                end, _ = self._flight(1.0)
                if m <= end[0] <= w - m and m <= end[1] <= h - m:
                    return dk
        self.dist = full
        return dk

    def on_update(self, ctx: Context) -> None:
        inten = self._p("intensity")
        body, _ = self._colors(ctx)
        if self.phase == Phase.CHARGING:
            self._update_anchor(ctx)
            ctx.aura_boost = max(ctx.aura_boost, 1.6 * self.progress * inten)
            origin = self.anchor
            if self.anchor_blade is not None:
                L = self.anchor_len
                grip = origin - self.anchor_blade * L * 0.8
                k = np.random.uniform(0.2, 1.0, (int(10 * inten), 1)).astype(np.float32)
                on_blade = grip + self.anchor_blade * L * k
                n = fx.perp(self.anchor_blade)
                side = np.random.choice([-1.0, 1.0], (len(k), 1)).astype(np.float32)
                start = on_blade + n * side * L * np.random.uniform(0.2, 0.45, (len(k), 1))
                ctx.particles.emit(len(k), start, (on_blade - start) * 3.0, life=(0.2, 0.35),
                                   size=(1.2, 2.8), color=body, drag=0.5)
                ctx.particles.attract(grip + self.anchor_blade * L * 0.6, L * 2.0, ctx.dt)
            else:
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
            idx = np.random.randint(0, len(pts), max(1, int(3 * inten)))
            ctx.particles.emit(len(idx), pts[idx], back * 0.5, spread_pos=r * 0.02, spread_vel=r * 0.6,
                               life=(0.08, 0.2), size=(0.8, 1.6), color=(255, 255, 255), drag=3.0)

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
        if self.anchor is None:
            return
        body, core = self._colors(ctx)
        hot = body if self.dark else core
        p = fx.smoothstep(self.progress)
        k = (0.3 + 0.7 * p) * self._p("intensity")
        k *= 0.85 + 0.15 * math.sin(ctx.now * 31.0)
        canvas = ctx.canvas
        origin = self.anchor
        if self.anchor_blade is not None:
            L = self.anchor_len
            grip = origin - self.anchor_blade * L * 0.8
            tip = origin + self.anchor_blade * L * 0.2
            canvas.glow_line(grip, tip, self._deep(body), L * (0.08 + 0.08 * p), 0.9 * k)
            canvas.light_line(grip + self.anchor_blade * L * 0.15, tip, body, max(2, int(L * 0.03)), 0.6 * k)
            canvas.light_line(grip + self.anchor_blade * L * 0.2, tip, (255, 255, 255), 2, 0.6 * k)
            orb = L * (0.06 + 0.1 * p)
        else:
            orb = 0.03 * ctx.diag * (0.5 + p)
        # Energy pooling at the launch point, with a ring contracting onto it.
        canvas.glow_circle(origin, orb * 1.8, self._deep(body), 0.8 * k)
        canvas.light_circle(origin, orb * 0.35, hot, 0.9 * k)
        ring_r = 0.09 * ctx.diag * self._p("size") * (1.25 - p)
        fx.ring(canvas, origin, ring_r, body, max(1, int(ring_r * 0.05)), 0.45 * k * (0.3 + p))

    # Ribbon: the swept path behind the head, tapering to a point at the tail.
    def _ribbon(self, t: float, length: float, width: float, back: float = 0.25, n: int = 9) -> Optional[np.ndarray]:
        t0 = max(0.0, t - length)
        if t - t0 < 1e-3:
            return None
        ts = np.linspace(t0, t, n)
        left, right = [], []
        for i, tt in enumerate(ts):
            pos, sc, d, r, _ = self._shape(tt)
            s = i / (n - 1)
            w = r * width * s ** 1.6  # concave taper: comet tail, not a wedge
            c = pos - d * r * back
            side = fx.perp(d) * w
            left.append(c + side)
            right.append(c - side)
        return np.array(left + right[::-1], np.float32)

    def _render_ribbon(self, ctx: Context, t: float, strength: float) -> None:
        canvas = ctx.canvas
        body, core = self._colors(ctx)
        inten = self._p("intensity") * strength
        if inten <= 0.01:
            return
        outer = self._ribbon(t, TRAIL, 0.6, back=0.35)
        if outer is None:
            return
        canvas.glow_poly(outer, self._deep(body), 0.45 * inten)
        canvas.light_poly(outer, body, 0.2 * inten)
        mid = self._ribbon(t, TRAIL * 0.6, 0.36, back=0.3)
        if mid is not None:
            canvas.light_poly(mid, body, 0.25 * inten)
        inner = self._ribbon(t, TRAIL * 0.32, 0.16, back=0.28)
        if inner is not None:
            if self.dark:
                canvas.light_poly(inner, body, 0.45 * inten)
            else:
                canvas.light_poly(inner, core, 0.32 * inten)
        # Energy filaments streaming along the path (one polylines call).
        lines = []
        n = 8
        t0 = max(0.0, t - TRAIL * 0.6)
        ts = np.linspace(t0, t, n)
        shapes = [self._shape(tt) for tt in ts]
        for j, off in enumerate((-0.5, -0.22, 0.22, 0.5)):
            pts = []
            for i, (pos, sc, d, r, _) in enumerate(shapes):
                s = i / (n - 1)
                wob = 0.06 * math.sin(ctx.now * 23.0 + i * 1.7 + j * 2.3)
                pts.append(pos - d * r * 0.3 + fx.perp(d) * r * (off * (0.35 + 0.65 * s ** 1.4) + wob * s))
            lines.append(np.array(pts, np.float32).astype(np.int32))
        th = max(1, int(self.radius * 0.012))
        cv2.polylines(canvas.light, lines, False, fx.scale_color(body, 0.75 * inten * canvas.intensity), th,
                      cv2.LINE_AA)

    def _crescent_layers(self, ctx: Context, t: float, strength: float, full: bool) -> None:
        canvas = ctx.canvas
        body, core = self._colors(ctx)
        pos, sc, d, r, center = self._shape(t)
        th = r * (0.42 if self.dark else 0.3)
        inten = self._p("intensity") * strength
        # Bankai keeps its halo tighter and less saturated so bloom doesn't fill the black core.
        canvas.glow_poly(fx.crescent(center + d * r * 0.04, d, r * 1.1, th * 1.6, span_deg=160),
                         body if self.dark else self._deep(body), (0.3 if self.dark and full else 0.5) * inten)
        shape = fx.crescent(center, d, r, th, span_deg=150)
        canvas.light_poly(shape, body, (0.55 if self.dark else 0.42) * inten)
        if not full:
            return
        if self.dark:
            # Tensa Bankai: a black core rimmed in crimson, glow pushed wider so
            # the darkness reads as compressed energy rather than a hole.
            core_pts = fx.crescent(center + d * r * 0.01, d, r * 0.975, th * 0.7, span_deg=140)
            ci = core_pts.astype(np.int32)
            cv2.fillPoly(canvas.light, [ci], (0, 0, 0), cv2.LINE_AA)
            cv2.fillPoly(canvas.frame, [ci], core, cv2.LINE_AA)
            rim = fx.crescent(center, d, r * 1.01, th * 1.05, span_deg=152)
            cv2.polylines(canvas.light, [rim.astype(np.int32)], True,
                          fx.scale_color(body, 1.4 * inten * canvas.intensity), max(1, int(r * 0.035)),
                          cv2.LINE_AA)
            k = 0.9 * inten * canvas.intensity
            cv2.polylines(canvas.glow, [(rim * canvas.gs).astype(np.int32)], True,
                          (body[0] * k, body[1] * k, body[2] * k), max(1, int(r * 0.08 * canvas.gs)), cv2.LINE_AA)
            # A hairline of white-hot crimson only on the leading edge.
            edge = rim[2: len(rim) // 2 - 2]
            cv2.polylines(canvas.light, [edge.astype(np.int32)], False, fx.scale_color((150, 140, 255), inten),
                          max(1, int(r * 0.012)), cv2.LINE_AA)
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
        tris = []
        for i in rng.choice(np.arange(3, m - 3), size=min(7, m - 6), replace=False):
            ln = r * rng.uniform(0.06, 0.32) * math.sin(math.pi * i / m)
            a = inner[i]
            w = r * 0.035
            tipp = a - d * ln + side * rng.uniform(-0.25, 0.25) * ln
            tris.append(np.array([a + side * w, a - side * w, tipp], np.float32).astype(np.int32))
        cv2.fillPoly(canvas.light, tris, fx.scale_color(body, 0.55 * inten * canvas.intensity), cv2.LINE_AA)

    def _render_launch(self, ctx: Context) -> None:
        """Launch beat: tip flare + cross streak + shock ring at the origin."""
        lt = self.t / LAUNCH
        if lt >= 1.0:
            return
        canvas = ctx.canvas
        body, core = self._colors(ctx)
        hot = (255, 255, 255) if not self.dark else (170, 150, 255)
        inten = self._p("intensity")
        r = self.radius
        fade = 1.0 - fx.ease_in(lt)
        o, d = self.p0, self.dir
        side = fx.perp(d)
        # Shock ring leaving the origin.
        rr = r * (0.25 + 0.9 * fx.ease_out(lt))
        fx.ring(canvas, o, rr, body, max(1, int(r * 0.06 * fade + 1)), 0.7 * fade * inten)
        # Flare: a hot streak shooting forward plus a thin perpendicular glint.
        ln = r * (0.6 + 1.6 * fx.ease_out(lt))
        canvas.glow_line(o - d * r * 0.2, o + d * ln, self._deep(body), r * 0.25, 0.9 * fade * inten)
        canvas.light_line(o, o + d * ln, hot, max(2, int(r * 0.04)), 0.9 * fade * inten)
        g = r * 1.1 * (1.0 - lt * 0.5)
        canvas.light_line(o - side * g, o + side * g, hot, max(1, int(r * 0.015)), 0.8 * fade * inten)
        canvas.glow_circle(o, r * 0.3 * (1.0 + 0.5 * lt), self._deep(body), 0.6 * fade * inten)
        canvas.light_circle(o, r * 0.12 * fade + 1, hot, fade * inten)

    def _render_flight(self, ctx: Context) -> None:
        t = self.progress
        dur = max(1e-3, self.duration(Phase.ACTIVE))
        self._render_ribbon(ctx, t, min(1.0, t / 0.12))
        for i, lag in reversed(list(enumerate(AFTERIMAGES))):
            tt = t - lag / dur
            if tt > 0.0:
                self._crescent_layers(ctx, tt, 0.45 / (i + 1.5), full=False)
        fade_in = min(1.0, 0.35 + self.t / LAUNCH)
        self._crescent_layers(ctx, t, fade_in, full=True)
        self._render_launch(ctx)

    def _render_impact(self, ctx: Context) -> None:
        canvas = ctx.canvas
        body, core = self._colors(ctx)
        hot = body if self.dark else core
        t = self.progress
        inten = self._p("intensity")
        r0 = self.radius * max(0.5, self.end_scale)
        fade = 1.0 - fx.ease_in(t)
        # The wake drains away and the crescent splashes against the "target".
        if t < 0.45:
            k = 1.0 - t / 0.45
            self._render_ribbon(ctx, 1.0, k * k)
            d = self.dir
            sr = r0 * (1.0 + 0.8 * fx.ease_out(t / 0.45))
            c = self.end - d * sr * 0.85
            canvas.glow_poly(fx.crescent(c, d, sr * 1.05, sr * 0.5 * k, span_deg=170), self._deep(body),
                             0.6 * k * inten)
            canvas.light_poly(fx.crescent(c, d, sr, sr * 0.22 * k + 1, span_deg=165), body, 0.6 * k * inten)
        rad = r0 * (0.4 + 2.0 * fx.ease_out(t))
        fx.ring(canvas, self.end, rad, body, max(1, int(r0 * 0.1 * fade + 1)), 0.65 * fade * inten)
        fx.ring(canvas, self.end, rad * 0.8, hot, max(1, int(r0 * 0.03 * fade + 1)), 0.35 * fade * fade * inten)
        t2 = max(0.0, t - 0.15) / 0.85
        if t2 > 0.0:
            fx.ring(canvas, self.end, r0 * (0.3 + 1.4 * fx.ease_out(t2)), body,
                    max(1, int(r0 * 0.05 * (1 - t2) + 1)), 0.4 * (1 - t2) * inten)
        canvas.glow_circle(self.end, r0 * (0.6 + 0.8 * t), self._deep(body), 0.7 * fade * fade * inten)
        # Radial streaks flying out of the blast (one polylines call).
        rng = np.random.default_rng(int(self.end[0] * 7 + self.end[1]) & 0xFFFF)
        angs = rng.uniform(0, 2 * math.pi, 14)
        lens = rng.uniform(0.6, 1.4, 14)
        v = np.stack([np.cos(angs), np.sin(angs)], 1).astype(np.float32)
        p0 = self.end + v * rad * (0.3 + 0.4 * t)
        p1 = self.end + v * (rad * (0.6 + 0.45 * lens))[:, None]
        segs = list(np.stack([p0, p1], 1).astype(np.int32))
        cv2.polylines(canvas.light, segs, False, fx.scale_color(body, 0.6 * fade * inten * canvas.intensity),
                      max(1, int(r0 * 0.04)), cv2.LINE_AA)
        if t < 0.3:
            k = 1.0 - t / 0.3
            canvas.light_circle(self.end, r0 * 0.25 * (1.0 + 2.0 * t), hot, k * inten)
