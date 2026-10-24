"""Cursed techniques: Lapse Blue, Reversal Red, Hollow Purple, Black Flash.

Four deliberately different effects sharing only plumbing (``_Technique``):

* **Lapse: Blue** - a point of *attraction*. A compact blue-black core with a
  bright rim, spiral streaks and a swirling ring of motes drawn inward,
  nearby particles accelerated into the core (and swallowed), the room
  pinched and swirled toward it. It ends in a controlled collapse: the core
  contracts to a pinpoint and releases a small implosion pop.
* **Reversal: Red** - a point of *repulsion*. A tiny, white-hot red core that
  compresses, then bursts: concentric shockwave rings, radial streaks,
  outward debris, a bulge + ripple distortion and nearby particles pushed
  away, all easing out smoothly (no lingering flash).
* **Hollow Purple** - a Blue source and a Red source appear at two points
  (each hand, else either side of the chest), stay separate, converge
  (orbiting) into a crackling violet-white core, which is then launched as a
  directional projectile: purple shell, white centre, bow shock, tapered
  wake, moving bulge/swirl distortion, ending in a ripple-ring impact.
* **Black Flash** - a ~0.12 s anticipation (the world greys out, a dark
  pinpoint, converging speed lines), then a black impact point with
  branching red-black lightning (black core lines, red rim glow), a fast
  shockwave, afterimage echoes of the bolts and a short recovery. It is a
  stylised impact at the fist; it does not try to detect "real" timing.

Direction / source rules are documented on each class. Everything derived
from tracking goes through ``_finite``; the anchor is held through tracking
loss; motion is dt-based; ``reset()`` clears all per-shot state.

Optional per-ability parameters (``cfg["abilities"][name]``, read with
``.get``): ``size``, ``intensity``, ``color``, ``core_color`` (all four),
``speed`` (hollow_purple travel distance), ``warp`` (distortion strength
multiplier, default 1; 0 disables warps), ``particles`` (particle-count
multiplier on top of the quality preset), ``shake`` (screen-shake
multiplier), ``flash`` (screen-flash multiplier).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import fx
from ..abilities import Ability, Context, Phase
from ..gestures import classify_hand

COORD_LIMIT = 1e4
PALM = [0, 5, 9, 13, 17]
MCP = [5, 9, 13, 17]
ANCHOR_TAU = 0.05  # s, EMA time constant for following the hand
UP = np.array([0.0, -1.0], np.float32)
TAU = 2.0 * math.pi
SWING_MIN = 0.5  # frame diagonals / s that count as a real swing / punch


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _finite(v, fallback=None) -> Optional[np.ndarray]:
    """``v`` as a clamped float32 2-vector, or ``fallback`` if not finite."""
    try:
        a = np.asarray(v, np.float32).reshape(-1)[:2]
    except (TypeError, ValueError):
        return fallback
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        return fallback
    return np.clip(a, -COORD_LIMIT, COORD_LIMIT).astype(np.float32)


def _pt(p) -> Tuple[int, int]:
    return (int(round(float(np.clip(p[0], -COORD_LIMIT, COORD_LIMIT)))),
            int(round(float(np.clip(p[1], -COORD_LIMIT, COORD_LIMIT)))))


def _ipoly(pts) -> np.ndarray:
    return np.clip(np.asarray(pts, np.float32), -COORD_LIMIT, COORD_LIMIT).round().astype(np.int32)


def _deep(c) -> Tuple[float, float, float]:
    """More saturated version of a colour for the soft glow layer."""
    c = np.asarray(c, np.float32)
    m = float(c.mean())
    return tuple(float(x) for x in np.clip((c - m) * 2.0 + m * 0.7, 0, 255))


def _mix(a, b, t: float) -> Tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(float(x) * (1 - t) + float(y) * t for x, y in zip(a, b))


def _scaled(color, k: float) -> Tuple[float, float, float]:
    return tuple(float(np.clip(c * k, 0, 255)) for c in color)


def field_force(ps: fx.ParticleSystem, center, radius: float, pull: float, swirl: float, dt: float,
                kill_radius: float = 0.0) -> None:
    """Local force field on live particles within ``radius`` of ``center``.

    ``pull`` > 0 attracts (px/s^2 at the centre, linear falloff), < 0 repels;
    ``swirl`` adds a tangential acceleration. Particles that reach
    ``kill_radius`` are swallowed (their life is cut to a few ms).
    Unlike ``ParticleSystem.attract`` this only affects *nearby* particles.
    """
    c = _finite(center)
    if c is None or radius <= 1.0 or dt <= 0.0:
        return
    alive = np.nonzero(ps.life > 0)[0]
    if alive.size == 0:
        return
    d = c - ps.pos[alive]
    dist = np.sqrt((d * d).sum(1)) + 1e-3
    near = dist < radius
    if not near.any():
        return
    idx, d, dist = alive[near], d[near], dist[near]
    n = d / dist[:, None]
    t = np.stack([-n[:, 1], n[:, 0]], 1)
    f = (1.0 - dist / radius)[:, None]
    ps.vel[idx] += (n * pull + t * swirl) * f * dt
    if kill_radius > 0:
        k = idx[dist < kill_radius]
        if k.size:
            ps.life[k] = np.minimum(ps.life[k], 0.03)


def branching_bolts(rng, origin, angle: float, length: float, depth: int = 2,
                    segments: int = 6, jag: float = 0.6, branch_p: float = 0.4
                    ) -> List[Tuple[np.ndarray, int]]:
    """Angular branching lightning from ``origin`` heading ``angle``.

    ``rng`` is a ``random.Random`` (or anything with uniform/random/choice);
    Python's scalar RNG is much faster than numpy's for this many draws.

    Returns ``[(points (k, 2) float32, level), ...]``; level 0 is the trunk,
    branches are shorter and thinner. Branch count is bounded by ``depth``.
    """
    out: List[Tuple[np.ndarray, int]] = []

    def walk(x, y, base, ln, lvl):
        pts = [(x, y)]
        seg = ln / segments
        for i in range(segments):
            a = base + rng.uniform(-jag, jag)  # sharp zig-zag around the heading
            s = seg * rng.uniform(0.7, 1.3)
            x, y = x + math.cos(a) * s, y + math.sin(a) * s
            pts.append((x, y))
            if lvl < depth and 0 < i < segments - 1 and rng.random() < branch_p:
                walk(x, y, base + rng.choice((-1.0, 1.0)) * rng.uniform(0.35, 0.95), ln * rng.uniform(0.3, 0.55),
                     lvl + 1)
        out.append((np.array(pts, np.float32), lvl))

    walk(float(origin[0]), float(origin[1]), float(angle), float(length), 0)
    return out


def _hand_aim(pts: np.ndarray, gesture: str):
    """(point, unit direction, hand size) for a 21-landmark hand."""
    size = max(1.0, float(np.linalg.norm(pts[9] - pts[0])))
    if gesture == "point":
        d = fx.unit(pts[8] - pts[5])
        return pts[8] + d * size * 0.25, d, size
    if gesture == "fist":
        return pts[MCP].mean(axis=0), fx.unit(pts[9] - pts[0]), size
    d = fx.unit(pts[9] - pts[0])
    return pts[PALM].mean(axis=0) + d * size * 0.35, d, size


# --------------------------------------------------------------------------
# shared base
# --------------------------------------------------------------------------
class _Technique(Ability):
    default_color: Tuple[int, int, int] = (255, 110, 150)
    default_core: Tuple[int, int, int] = (255, 255, 255)

    def __init__(self, params: dict):
        super().__init__(params)
        self._rng = np.random.default_rng(0)
        self._acc = 0.0  # fractional particle emission carry
        self._clear()

    def _clear(self) -> None:  # per-shot state
        pass

    def reset(self, ctx: Context) -> None:
        super().reset(ctx)
        self._acc = 0.0
        self._clear()

    # --- params ---------------------------------------------------------
    def _p(self, key: str, default: float = 1.0) -> float:
        try:
            v = float(self.params.get(key, default))
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    def _col(self, key: str, default) -> Tuple[float, float, float]:
        try:
            c = [float(x) for x in self.params.get(key, default)][:3]
        except (TypeError, ValueError):
            c = []
        if len(c) != 3 or not all(math.isfinite(x) for x in c):
            c = list(default)
        return tuple(float(np.clip(x, 0, 255)) for x in c)

    @property
    def body(self) -> Tuple[float, float, float]:
        return self._col("color", self.default_color)

    @property
    def core(self) -> Tuple[float, float, float]:
        return self._col("core_color", self.default_core)

    @property
    def inten(self) -> float:
        return float(np.clip(self._p("intensity"), 0.0, 3.0))

    @property
    def size(self) -> float:
        return float(np.clip(self._p("size"), 0.2, 4.0))

    @property
    def warp_k(self) -> float:
        return float(np.clip(self._p("warp"), 0.0, 3.0))

    def _q(self, ctx: Context) -> float:
        """Particle budget scale: quality preset (capacity) x intensity x ``particles``."""
        cap = float(getattr(ctx.particles, "capacity", 900))
        return float(np.clip(cap / 900.0, 0.4, 1.8)) * min(2.0, self.inten) * float(
            np.clip(self._p("particles"), 0.0, 3.0))

    def _shake(self, ctx: Context, px: float) -> None:
        ctx.shaker.kick(px * float(np.clip(self._p("shake"), 0.0, 3.0)))

    def _flash(self, ctx: Context, amount: float, color) -> None:
        a = amount * float(np.clip(self._p("flash"), 0.0, 3.0))
        if a > ctx.canvas.flash:
            ctx.canvas.flash = a
            ctx.canvas.flash_color = tuple(int(c) for c in color)

    def _emit(self, ctx: Context, rate: float) -> int:
        """Rate-based (per second) particle count for this frame, quality scaled."""
        self._acc += max(0.0, rate * self._q(ctx) * max(0.0, ctx.dt))
        n = int(self._acc)
        self._acc -= n
        return min(n, 200)

    # --- geometry ---------------------------------------------------------
    @staticmethod
    def _dims(ctx: Context) -> Tuple[int, int]:
        w = ctx.canvas.w or ctx.tracking.width or 960
        h = ctx.canvas.h or ctx.tracking.height or 540
        return int(w), int(h)

    def _diag(self, ctx: Context) -> float:
        w, h = self._dims(ctx)
        return float(math.hypot(w, h))

    def _chest(self, ctx: Context) -> np.ndarray:
        w, h = self._dims(ctx)
        try:
            bc = ctx.body_center()
        except Exception:  # malformed tracking arrays
            bc = None
        c = _finite(bc) if bc is not None else None
        if c is None:
            return np.array([w * 0.5, h * 0.6], np.float32)
        return _finite(c + np.array([0.0, 0.08 * h], np.float32))

    def _shoulder_half(self, ctx: Context) -> float:
        w, _ = self._dims(ctx)
        pose = ctx.tracking.pose
        if pose.ok and pose.points is not None and len(pose.points) > 12:
            a, b = _finite(pose.points[11, :2]), _finite(pose.points[12, :2])
            if a is not None and b is not None:
                return float(np.clip(np.linalg.norm(a - b) * 0.75, 0.08 * w, 0.25 * w))
        return 0.15 * w

    @staticmethod
    def _hands(ctx: Context):
        """[(side, (21, 2) points, gesture)] for visible, finite hands."""
        out = []
        for side in ("right", "left"):
            part = ctx.tracking.hands.get(side)
            if part is None or not part.ok:
                continue
            try:
                pts = np.asarray(part.points, np.float32)[:, :2]
            except (TypeError, ValueError, IndexError):
                continue
            if pts.shape[0] < 21 or not np.all(np.isfinite(pts[:21])):
                continue
            pts = np.clip(pts[:21], -COORD_LIMIT, COORD_LIMIT)
            try:
                g = classify_hand(pts)
            except Exception:
                g = "other"
            out.append((side, pts, g))
        return out

    @staticmethod
    def _follow(cur: Optional[np.ndarray], target: np.ndarray, dt: float, diag: float) -> np.ndarray:
        """Adaptive EMA: jitter is smoothed, real moves pass through quickly."""
        if cur is None:
            return target.copy()
        a = 1.0 - math.exp(-max(0.0, dt) / ANCHOR_TAU)
        moved = float(np.linalg.norm(target - cur)) / max(1.0, 0.02 * diag)
        a += (1.0 - a) * min(1.0, moved * 0.5)
        return _finite(cur + (target - cur) * a, cur)

    def _sword_ok(self, ctx: Context) -> bool:
        sw = ctx.sword
        return (math.isfinite(float(sw.confidence)) and sw.confidence > 0.2 and _finite(sw.grip) is not None
                and _finite(sw.direction) is not None)

    # --- frame ---------------------------------------------------------------
    def update(self, ctx: Context) -> None:
        super().update(ctx)
        if self.phase in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT):
            ctx.energy_override = tuple(int(c) for c in self.body)

    def _burst_time(self) -> float:
        """Seconds since ACTIVE began (continues through IMPACT)."""
        if self.phase == Phase.ACTIVE:
            return self.t
        if self.phase == Phase.IMPACT:
            return self.duration(Phase.ACTIVE) + self.t
        return 0.0

    def _burst_span(self) -> float:
        return max(1e-3, self.duration(Phase.ACTIVE) + self.duration(Phase.IMPACT))


# --------------------------------------------------------------------------
# Lapse: Blue
# --------------------------------------------------------------------------
class Blue(_Technique):
    """Source: a pointing index tip, else an open palm (slightly toward the
    fingers), else any visible hand, else in front of the chest. The anchor
    follows the hand while charging/active and is held through tracking loss.
    """

    name = "blue"
    label = "Lapse: Blue"
    default_color = (255, 150, 40)
    default_core = (255, 235, 200)
    COLLAPSE = 0.55  # fraction of IMPACT spent collapsing before the release pop

    def _clear(self) -> None:
        self.anchor: Optional[np.ndarray] = None
        self.radius = 0.0
        self._popped = False
        self._seed = 0

    def _measure(self, ctx: Context) -> Optional[np.ndarray]:
        best = None
        for _, pts, g in self._hands(ctx):
            rank = {"point": 0, "open_palm": 1}.get(g, 2)
            if best is None or rank < best[0]:
                best = (rank, _hand_aim(pts, g)[0])
        return None if best is None else _finite(best[1])

    def _track(self, ctx: Context, snap: bool = False) -> None:
        m = self._measure(ctx)
        if m is None:
            if self.anchor is None:
                self.anchor = self._chest(ctx)
            return
        self.anchor = m.copy() if snap or self.anchor is None else self._follow(self.anchor, m, ctx.dt,
                                                                                 self._diag(ctx))

    def on_enter(self, phase: Phase, ctx: Context) -> None:
        if phase == Phase.CHARGING:
            self._clear()
            self._seed = int(self._rng.integers(0, 1 << 30))
            self.radius = 0.03 * self._diag(ctx) * self.size
            self._track(ctx, snap=True)
        elif phase == Phase.ACTIVE:
            if self.anchor is None:
                self.on_enter(Phase.CHARGING, ctx)
                self.phase = Phase.ACTIVE
            self._shake(ctx, 3.0 * self.inten)
        elif phase == Phase.IMPACT:
            if self.anchor is None:
                self._track(ctx, snap=True)
                self.radius = 0.03 * self._diag(ctx) * self.size
        elif phase == Phase.COOLDOWN:
            self._clear()

    def _pop(self, ctx: Context) -> None:
        self._popped = True
        c, R, q = self.anchor, self.radius, self._q(ctx)
        diag = self._diag(ctx)
        n = int(30 * q)
        ang = self._rng.uniform(0, TAU, n)
        v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
        spd = self._rng.uniform(0.25, 0.6, (n, 1)).astype(np.float32) * diag
        ctx.particles.emit(n, c + v * R * 0.3, v * spd, life=(0.15, 0.4), size=(1.0, 2.4), color=self.core,
                           drag=5.0)
        self._flash(ctx, 0.07 * self.inten, self.core)
        self._shake(ctx, 7.0 * self.inten)

    def on_update(self, ctx: Context) -> None:
        if self.anchor is None:
            return
        if self.phase in (Phase.CHARGING, Phase.ACTIVE):
            self._track(ctx)
        c, R, diag, ps = self.anchor, self.radius, self._diag(ctx), ctx.particles
        if self.phase in (Phase.CHARGING, Phase.ACTIVE):
            lvl = fx.ease_out(self.progress) if self.phase == Phase.CHARGING else 1.0
            ctx.aura_boost = max(ctx.aura_boost, 1.2 * lvl * self.inten)
            n = self._emit(ctx, 110.0 * lvl)
            if n:
                ang = self._rng.uniform(0, TAU, n)
                rad = R * self._rng.uniform(3.0, 6.5, n)
                v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                tan = np.stack([-v[:, 1], v[:, 0]], 1)
                pos = c + v * rad[:, None]
                vel = tan * diag * 0.33 - v * diag * 0.1
                h = n // 2
                ps.emit(h, pos[:h], vel[:h], life=(0.5, 1.0), size=(1.0, 2.2), color=self.body, drag=0.6)
                ps.emit(n - h, pos[h:], vel[h:], life=(0.4, 0.8), size=(0.8, 1.6), color=self.core, drag=0.6)
            field_force(ps, c, R * 9.0, diag * 2.4 * lvl, diag * 0.8 * lvl, ctx.dt, kill_radius=R * 0.55)
        elif self.phase == Phase.IMPACT:
            if self.progress < self.COLLAPSE:
                field_force(ps, c, R * 10.0, diag * 5.0, diag * 1.2, ctx.dt, kill_radius=R * 0.5)
            elif not self._popped:
                self._pop(ctx)

    # --- render ----------------------------------------------------------
    def render(self, ctx: Context) -> None:
        if self.anchor is None or self.phase not in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT):
            return
        canvas = ctx.canvas
        c, R, now = self.anchor, self.radius, ctx.now
        body, core, deep = self.body, self.core, _deep(self.body)
        ik = self.inten
        collapse, pop = 0.0, -1.0
        if self.phase == Phase.CHARGING:
            lvl = fx.ease_out(self.progress)
            core_r = R * (0.25 + 0.75 * lvl)
        elif self.phase == Phase.ACTIVE:
            lvl = 1.0
            core_r = R * (1.0 + 0.07 * math.sin(now * 13.0))
        else:
            t = self.progress
            if t < self.COLLAPSE:
                collapse = t / self.COLLAPSE
                lvl = 1.0
                core_r = R * (1.0 - 0.8 * fx.ease_in(collapse))
            else:
                pop = (t - self.COLLAPSE) / (1.0 - self.COLLAPSE)
                lvl = 0.0
                core_r = 0.0
        if pop < 0:
            spin = 1.0 + 2.5 * collapse
            g = canvas.gs
            cv2.circle(canvas.glow, _pt(c * g), max(1, int(core_r * 1.5 * g)), _scaled(deep, 0.55 * lvl * ik),
                       max(1, int(core_r * 1.2 * g)), cv2.LINE_AA)
            canvas.add_sprite(c, R * (6.0 - 3.0 * collapse), deep, 0.22 * lvl * ik, layer="glow", falloff=1.0)
            self._streaks(ctx, c, R, lvl * ik, spin, collapse)
            # Accretion arcs whirling around the core.
            for j, (rr, sp, span) in enumerate(((1.55, 4.0, 110), (1.95, -2.7, 80), (2.4, 3.1, 60))):
                rad = max(2.0, core_r * rr * (1.0 - 0.3 * collapse))
                a0 = math.degrees(now * sp * spin) + j * 120
                for k in range(2):
                    cv2.ellipse(canvas.light, _pt(c), (int(rad), int(rad)), a0 + k * 180, 0, span,
                                fx.scale_color(_mix(body, core, 0.4 - j * 0.15), (0.75 - 0.18 * j) * lvl * ik),
                                max(1, int(R * 0.07)), cv2.LINE_AA)
            # Bright rim then the dark centre punched through light and camera.
            canvas.add_sprite(c, core_r * 1.9, body, 0.7 * lvl * ik, layer="light", falloff=1.4)
            fx.ring(canvas, c, core_r * 0.95, body, max(2, int(R * 0.16)), 0.8 * lvl * ik)
            canvas.light_circle(c, core_r * 0.9, core, 0.95 * lvl * ik, thickness=max(1, int(R * 0.06)))
            dark = max(1, int(core_r * 0.84))
            cv2.circle(canvas.light, _pt(c), dark, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas.frame, _pt(c), dark, (22, 6, 2), -1, cv2.LINE_AA)
            if collapse > 0:
                # Contracting implosion ring.
                rr = R * (5.0 - 4.4 * fx.ease_in(collapse))
                fx.ring(canvas, c, rr, _mix(body, core, 0.5), max(1, int(R * 0.06)), 0.5 * collapse * ik)
            wk = self.warp_k
            canvas.add_warp(c, R * 6.0, (0.28 + 0.35 * collapse) * lvl * wk, "pinch")
            canvas.add_warp(c, R * 5.0, (0.75 + 1.2 * collapse) * lvl * wk, "swirl")
        else:
            k = 1.0 - fx.ease_out(pop)
            canvas.add_sprite(c, R * (0.6 + 1.6 * pop), core, 1.1 * k * ik, layer="light", falloff=2.2)
            canvas.add_sprite(c, R * 3.0, deep, 0.6 * k * ik, layer="glow")
            rr = R * (0.5 + 4.5 * fx.ease_out(pop))
            fx.ring(canvas, c, rr, _mix(body, core, 0.5), max(1, int(R * 0.12 * k + 1)), 0.8 * k * ik)
            canvas.add_warp(c, rr, 0.7 * k * self.warp_k, "ripple", width=0.25)

    def _streaks(self, ctx: Context, c, R: float, k: float, spin: float, collapse: float) -> None:
        """Environmental streaks spiralling into the core (one polylines call per brightness)."""
        if k <= 0.01:
            return
        rng = np.random.default_rng(self._seed)
        n = int(np.clip(16 * min(1.5, self._q(ctx)), 6, 24))
        base = rng.uniform(0, TAU, n)
        off = rng.uniform(0, 1, n)
        rate = rng.uniform(0.9, 1.4, n)
        r_out, r_in = R * (6.5 - 2.5 * collapse), R * 0.9
        now = ctx.now
        bright, dim = [], []
        u = np.linspace(0.0, 0.24, 7)
        for i in range(n):
            s = (off[i] + now * rate[i] * 0.9 * spin) % 1.0
            uu = np.clip(s + u, 0.0, 1.0)
            rad = r_in + (r_out - r_in) * (1.0 - uu) ** 1.4
            th = base[i] + 2.6 * uu * uu + now * 0.7 * spin
            pts = c + np.stack([np.cos(th), np.sin(th)], 1) * rad[:, None]
            (bright if s > 0.45 else dim).append(_ipoly(pts))
        th_px = max(1, int(R * 0.05))
        col = _mix(self.body, self.core, 0.35)
        if dim:
            cv2.polylines(ctx.canvas.light, dim, False, fx.scale_color(col, 0.35 * k), th_px, cv2.LINE_AA)
        if bright:
            cv2.polylines(ctx.canvas.light, bright, False, fx.scale_color(col, 0.75 * k), th_px, cv2.LINE_AA)
            g = ctx.canvas.gs
            cv2.polylines(ctx.canvas.glow, [(b * g).astype(np.int32) for b in bright], False,
                          _scaled(_deep(self.body), 0.5 * k), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Reversal: Red
# --------------------------------------------------------------------------
class Red(_Technique):
    """Source: a pointing index tip, else an open palm, else any hand, else
    the chest. The burst point freezes at release (ACTIVE)."""

    name = "red"
    label = "Reversal: Red"
    default_color = (40, 40, 255)
    default_core = (220, 225, 255)

    def _clear(self) -> None:
        self.anchor: Optional[np.ndarray] = None
        self.radius = 0.0
        self._seed = 0

    _measure = Blue._measure
    _track = Blue._track

    def on_enter(self, phase: Phase, ctx: Context) -> None:
        if phase == Phase.CHARGING:
            self._clear()
            self._seed = int(self._rng.integers(0, 1 << 30))
            self.radius = 0.022 * self._diag(ctx) * self.size
            self._track(ctx, snap=True)
        elif phase == Phase.ACTIVE:
            if self.anchor is None:
                self._seed = int(self._rng.integers(0, 1 << 30))
                self.radius = 0.022 * self._diag(ctx) * self.size
                self._track(ctx, snap=True)
            c, R, q, diag = self.anchor, self.radius, self._q(ctx), self._diag(ctx)
            for n, col, spd, life, sz, drag in ((int(70 * q), self.body, 0.75, (0.35, 0.8), (1.4, 3.2), 2.6),
                                                (int(35 * q), self.core, 1.0, (0.15, 0.4), (1.0, 2.2), 3.5),
                                                (int(25 * q), (40, 120, 255), 0.5, (0.4, 0.9), (1.0, 2.6), 2.0)):
                if n <= 0:
                    continue
                ang = self._rng.uniform(0, TAU, n)
                v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                s = self._rng.uniform(0.3, 1.0, (n, 1)).astype(np.float32) * spd * diag
                ctx.particles.emit(n, c + v * R * 0.5, v * s, life=life, size=sz, color=col, drag=drag)
            self._flash(ctx, 0.1 * self.inten, (90, 90, 255))
            self._shake(ctx, 11.0 * self.inten)
        elif phase == Phase.COOLDOWN:
            self._clear()

    def on_update(self, ctx: Context) -> None:
        if self.phase == Phase.CHARGING:
            self._track(ctx)
            p = self.progress
            ctx.aura_boost = max(ctx.aura_boost, 1.4 * p * self.inten)
            n = self._emit(ctx, 60.0 * p)
            if n and self.anchor is not None:
                ang = self._rng.uniform(0, TAU, n)
                v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                ctx.particles.emit(n, self.anchor + v * self.radius * 0.6, v * self._diag(ctx) * 0.18,
                                   life=(0.08, 0.2), size=(0.8, 1.8), color=self.core, drag=3.0)
        elif self.phase in (Phase.ACTIVE, Phase.IMPACT) and self.anchor is not None:
            u = self._burst_time() / self._burst_span()
            front = self._ring_radius(u)
            field_force(ctx.particles, self.anchor, front * 1.3, -self._diag(ctx) * 3.0 * (1.0 - u), 0.0, ctx.dt)
            if self.phase == Phase.IMPACT:
                n = self._emit(ctx, 40.0 * (1.0 - self.progress))
                if n:
                    ctx.particles.emit(n, self.anchor, (0.0, -self.radius * 2.0), spread_pos=front * 0.35,
                                       spread_vel=self.radius * 2.0, life=(0.4, 0.8), size=(0.8, 1.8),
                                       color=(60, 110, 255), drag=1.0)

    def _ring_radius(self, u: float) -> float:
        return self.radius * (1.0 + 11.0 * fx.ease_out(min(1.0, u / 0.7)))

    def render(self, ctx: Context) -> None:
        if self.anchor is None or self.phase not in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT):
            return
        canvas = ctx.canvas
        c, R, now, ik = self.anchor, self.radius, ctx.now, self.inten
        body, core, deep = self.body, self.core, _deep(self.body)
        wk = self.warp_k
        if self.phase == Phase.CHARGING:
            p = self.progress
            k = (0.35 + 0.65 * fx.ease_out(p)) * ik
            flick = 0.85 + 0.15 * math.sin(now * 47.0)
            r = R * (0.45 + 0.55 * p) * flick
            canvas.add_sprite(c, r * 4.5, deep, 0.7 * k, layer="glow")
            canvas.add_sprite(c, r * 2.4, body, 1.0 * k, layer="light", falloff=1.8)
            canvas.add_sprite(c, r * 1.1, core, 1.2 * k, layer="light", falloff=1.2)
            canvas.light_circle(c, max(1.0, r * 0.4), (255, 255, 255), k)
            # Compression rings converging (the inverse of the coming burst).
            for j in range(2):
                f = (p * 1.6 + j * 0.5) % 1.0
                rr = R * (4.0 - 3.0 * fx.ease_in(f))
                fx.ring(canvas, c, rr, body, max(1, int(R * 0.08)), 0.5 * f * k)
            canvas.add_warp(c, R * 3.0, 0.15 * p * wk * flick, "bulge")
            return
        u = self._burst_time() / self._burst_span()
        fade = 1.0 - fx.smoothstep(u)
        # Core: bright, shrinking.
        cr = R * (1.25 - 0.9 * fx.ease_out(u))
        canvas.add_sprite(c, cr * 5.0, deep, 0.9 * fade * ik, layer="glow")
        canvas.add_sprite(c, cr * 2.2, body, 1.0 * fade * ik, layer="light", falloff=1.6)
        canvas.add_sprite(c, cr * 0.9, core, 1.0 * fade * ik, layer="light", falloff=1.1)
        # Pressure disc inside the front.
        front = self._ring_radius(u)
        canvas.glow_circle(c, front * 0.85, deep, 0.18 * fade * fade * ik)
        # Concentric shockwave rings, staggered.
        for j, (lag, col, th) in enumerate(((0.0, _mix(core, body, 0.6), 0.24), (0.1, body, 0.22), (0.22, deep, 0.16))):
            uj = (u - lag) / (1.0 - lag)
            if uj <= 0.0:
                continue
            rr = self._ring_radius(uj) * (1.0 - 0.12 * j)
            kj = (1.0 - fx.smoothstep(uj)) ** 1.3 * ik
            fx.ring(canvas, c, rr, col, max(1, int(R * th * (1.0 - uj) * 1.6 + 1)), (0.6 + 0.05 * j) * kj)
        # Radial streaks flung out (one polylines call).
        rng = np.random.default_rng(self._seed)
        m = 18
        ang = rng.uniform(0, TAU, m)
        ln = rng.uniform(0.25, 0.6, m)
        v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
        a = c + v * (front * (0.55 + 0.3 * u))
        b = a + v * (front * ln)[:, None] * (1.0 - u)
        segs = list(_ipoly(np.stack([a, b], 1)))
        cv2.polylines(canvas.light, segs, False, fx.scale_color(_mix(body, core, 0.5), 0.8 * fade * ik),
                      max(1, int(R * 0.1)), cv2.LINE_AA)
        # Short radial bulge + travelling ripple.
        bt = self._burst_time()
        if bt < 0.35:
            canvas.add_warp(c, R * 7.0, 0.5 * (1.0 - bt / 0.35) ** 2 * wk, "bulge")
        canvas.add_warp(c, front, 0.9 * fade * wk, "ripple", width=0.2)


# --------------------------------------------------------------------------
# Hollow Purple
# --------------------------------------------------------------------------
class HollowPurple(_Technique):
    """CHARGING: Blue at the screen-left hand (else left of the chest) and
    Red at the screen-right hand (else right of the chest) stay separate,
    then orbit inward and merge (distance 0 from 85% of the charge on).

    Launch direction (decided once at release, then frozen):
      1. a confident fist/"sword" (``ctx.sword``): its motion if swung fast,
         else its direction;
      2. a pointing hand: the index finger direction;
      3. otherwise "screen-forward": toward a vanishing point above screen
         centre with a strong perspective shrink.
    Directional shots are shortened to land inside the frame (min 15% of the
    diagonal; if even that is impossible the shot flies off-screen).
    """

    name = "hollow_purple"
    label = "Hollow Purple"
    default_color = (255, 60, 170)
    default_core = (255, 235, 250)
    BLUE = (255, 150, 40)
    RED = (40, 40, 255)
    SPLIT = 0.45  # charge fraction when convergence starts
    MERGE = 0.85  # ... and when it completes
    LAUNCH = 0.14  # s launch beat at the start of ACTIVE
    WAKE = 0.28  # normalised flight time covered by the wake

    def _clear(self) -> None:
        self.src_a: Optional[np.ndarray] = None  # blue source (raw anchor)
        self.src_b: Optional[np.ndarray] = None  # red source
        self.blue_pos: Optional[np.ndarray] = None  # rendered positions
        self.red_pos: Optional[np.ndarray] = None
        self.core_pos: Optional[np.ndarray] = None
        self.p0 = np.zeros(2, np.float32)
        self.dir = UP.copy()
        self.dist = 0.0
        self.end = np.zeros(2, np.float32)
        self.end_scale = 0.3
        self.forward = True
        self.R0 = 0.0
        self._seed = 0
        self.pos = np.zeros(2, np.float32)  # current projectile head

    # --- charge sources -------------------------------------------------
    def _measure(self, ctx: Context):
        hands = {side: _hand_aim(pts, g)[0] for side, pts, g in self._hands(ctx)}
        return _finite(hands.get("left")), _finite(hands.get("right"))

    def _track(self, ctx: Context, snap: bool = False) -> None:
        a, b = self._measure(ctx)
        chest = self._chest(ctx)
        half = self._shoulder_half(ctx)
        fa = chest + np.array([-half, 0.0], np.float32)
        fb = chest + np.array([half, 0.0], np.float32)
        diag = self._diag(ctx)
        if self.src_a is None or snap:
            self.src_a = a if a is not None else fa
        elif a is not None:
            self.src_a = self._follow(self.src_a, a, ctx.dt, diag)
        if self.src_b is None or snap:
            self.src_b = b if b is not None else fb
        elif b is not None:
            self.src_b = self._follow(self.src_b, b, ctx.dt, diag)
        # Both on one point (one hand reported twice): split them sideways.
        if float(np.linalg.norm(self.src_a - self.src_b)) < 0.05 * diag:
            m = (self.src_a + self.src_b) * 0.5
            self.src_a, self.src_b = m - np.array([half * 0.6, 0], np.float32), m + np.array([half * 0.6, 0],
                                                                                             np.float32)
        self._place()

    def _conv(self) -> float:
        if self.phase != Phase.CHARGING:
            return 1.0
        return fx.smoothstep((self.progress - self.SPLIT) / (self.MERGE - self.SPLIT))

    def _place(self) -> None:
        conv = self._conv()
        m = (self.src_a + self.src_b) * 0.5
        rot = conv * math.pi * 0.6  # orbit while closing in
        cr, sr = math.cos(rot), math.sin(rot)
        off = (self.src_a - m) * (1.0 - conv)
        off = np.array([off[0] * cr - off[1] * sr, off[0] * sr + off[1] * cr], np.float32)
        self.blue_pos = _finite(m + off)
        self.red_pos = _finite(m - off)
        self.core_pos = _finite(m)

    # --- flight -------------------------------------------------------------
    def _flight(self, t: float) -> Tuple[np.ndarray, float]:
        t = float(np.clip(t, 0.0, 1.0))
        travel = 0.35 * t + 0.65 * t * t  # heavy: builds speed
        pos = self.p0 + self.dir * self.dist * travel
        scale = 1.0 - (1.0 - self.end_scale) * fx.smoothstep(t)
        return pos.astype(np.float32), scale

    def _launch_dir(self, ctx: Context, origin: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self._sword_ok(ctx):
            sw = ctx.sword
            mv = _finite(sw.motion)
            spd = float(sw.speed) if math.isfinite(float(sw.speed)) else 0.0
            if spd > SWING_MIN and mv is not None and float(np.linalg.norm(mv)) > 1e-3:
                return fx.unit(mv), False
            d = _finite(sw.direction)
            if float(np.linalg.norm(d)) > 1e-3:
                return fx.unit(d), False
        for _, pts, g in self._hands(ctx):
            if g == "point":
                return _hand_aim(pts, g)[1], False
        return UP.copy(), True

    def _setup_flight(self, ctx: Context) -> None:
        w, h = self._dims(ctx)
        diag = self._diag(ctx)
        self.p0 = self.core_pos.copy()
        self.R0 = 0.055 * diag * self.size
        d, fwd = self._launch_dir(ctx, self.p0)
        self.forward = fwd
        if fwd:
            # Up and away, to the side of the head the core is on (never through the face).
            side = 1.0 if self.p0[0] >= w * 0.5 else -1.0
            vp = np.array([w * (0.5 + 0.3 * side), h * 0.22], np.float32)
            to = vp - self.p0
            if float(np.linalg.norm(to)) < 0.08 * diag:
                to = np.array([0.0, -0.12 * diag], np.float32)
            self.dir = fx.unit(to)
            self.dist = float(np.linalg.norm(to)) * min(1.5, max(0.3, self._p("speed")))
            self.end_scale = 0.22
        else:
            self.dir = fx.unit(d)
            full = 0.6 * diag * max(0.2, self._p("speed"))
            m = 0.06 * min(w, h)
            tmax = self._ray_to_box(self.p0, self.dir, m, m, w - m, h - m)
            self.dist = min(full, tmax) if tmax >= 0.15 * diag else full
            self.end_scale = 0.55
        self.end, _ = self._flight(1.0)

    @staticmethod
    def _ray_to_box(p, d, x0, y0, x1, y1) -> float:
        """Distance along ray p + s*d before leaving the box (0 if outside)."""
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            return 0.0
        s = float("inf")
        for i, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
            if d[i] > 1e-6:
                s = min(s, (hi - p[i]) / d[i])
            elif d[i] < -1e-6:
                s = min(s, (lo - p[i]) / d[i])
        return float(s) if math.isfinite(s) else 0.0

    # --- state hooks ----------------------------------------------------------
    def on_enter(self, phase: Phase, ctx: Context) -> None:
        q, ik = self._q(ctx), self.inten
        if phase == Phase.CHARGING:
            self._clear()
            self._seed = int(self._rng.integers(0, 1 << 30))
            self._track(ctx, snap=True)
        elif phase == Phase.ACTIVE:
            if self.src_a is None:
                self._seed = int(self._rng.integers(0, 1 << 30))
                self._track(ctx, snap=True)
            self._place()
            self._setup_flight(ctx)
            self.pos = self.p0.copy()
            n = int(40 * q)
            back = -self.dir
            ctx.particles.emit(n, self.p0, back * self.R0 * 4.0, spread_pos=self.R0 * 0.3,
                               spread_vel=self.R0 * 3.0, life=(0.2, 0.5), size=(1.2, 3.0), color=self.body, drag=3.0)
            ctx.particles.emit(n // 2, self.p0, (0, 0), spread_pos=self.R0 * 0.2, spread_vel=self.R0 * 5.0,
                               life=(0.1, 0.3), size=(0.8, 2.0), color=self.core, drag=4.0)
            self._flash(ctx, 0.14 * ik, (255, 160, 235))
            self._shake(ctx, 10.0 * ik)
        elif phase == Phase.IMPACT:
            if self.src_a is None:  # zero-length active
                self._track(ctx, snap=True)
                self._setup_flight(ctx)
            self.end, self.end_scale = self._flight(1.0)
            self.pos = self.end.copy()
            r = self.R0 * max(0.4, self.end_scale)
            ctx.particles.emit(int(90 * q), self.end, (0, 0), spread_pos=r * 0.3, spread_vel=r * 6.0,
                               life=(0.35, 0.9), size=(1.4, 3.6), color=self.body, drag=2.2)
            ctx.particles.emit(int(40 * q), self.end, (0, 0), spread_pos=r * 0.2, spread_vel=r * 4.0,
                               life=(0.2, 0.5), size=(1.0, 2.4), color=self.core, drag=3.0)
            self._flash(ctx, 0.3 * ik, (255, 190, 240))
            self._shake(ctx, 16.0 * ik)
        elif phase == Phase.COOLDOWN:
            self._clear()

    def on_update(self, ctx: Context) -> None:
        diag = self._diag(ctx)
        if self.phase == Phase.CHARGING:
            self._track(ctx)
            p = self.progress
            ctx.aura_boost = max(ctx.aura_boost, 2.0 * p * self.inten)
            ps = ctx.particles
            # Blue sucks motes in, Red spits sparks out.
            if p < self.MERGE:
                rb = 0.02 * diag * self.size
                n = self._emit(ctx, 70.0)
                if n:
                    ang = self._rng.uniform(0, TAU, n)
                    v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                    h = max(1, n // 2)
                    tan = np.stack([-v[:h, 1], v[:h, 0]], 1)
                    ps.emit(h, self.blue_pos + v[:h] * rb * 4.0, tan * diag * 0.2 - v[:h] * diag * 0.1,
                            life=(0.3, 0.6), size=(0.8, 1.8), color=self.BLUE, drag=0.8)
                    ps.emit(n - h, self.red_pos + v[h:] * rb * 0.6, v[h:] * diag * 0.22, life=(0.1, 0.25),
                            size=(0.8, 1.8), color=(120, 120, 255), drag=3.0)
                field_force(ps, self.blue_pos, rb * 7.0, diag * 1.6, diag * 0.6, ctx.dt, kill_radius=rb * 0.5)
            else:
                n = self._emit(ctx, 90.0)
                if n:
                    ang = self._rng.uniform(0, TAU, n)
                    v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                    r0 = 0.08 * diag * self.size
                    ps.emit(n, self.core_pos + v * r0, -v * diag * 0.35, life=(0.15, 0.3), size=(0.8, 2.0),
                            color=self.core if n % 2 else self.body, drag=0.5)
        elif self.phase == Phase.ACTIVE:
            t = self.progress
            self.pos, sc = self._flight(t)
            r = self.R0 * sc
            n = self._emit(ctx, 220.0)
            if n:
                ang = self._rng.uniform(0, TAU, n)
                v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                vel = -self.dir * r * 3.0 + v * r * 1.2
                h = n // 2
                ctx.particles.emit(h, self.pos + v[:h] * r, vel[:h], life=(0.2, 0.5), size=(1.0, 2.6 * sc + 0.5),
                                   color=self.body, drag=2.0)
                ctx.particles.emit(n - h, self.pos + v[h:] * r * 0.8, vel[h:] * 0.6, life=(0.1, 0.3),
                                   size=(0.8, 1.8), color=self.core, drag=3.0)
            # Space around it is dragged along: nearby particles swirl in its wake.
            field_force(ctx.particles, self.pos, r * 3.0, diag * 0.8, diag * 1.5, ctx.dt)

    # --- render -------------------------------------------------------------------
    def render(self, ctx: Context) -> None:
        if self.phase == Phase.CHARGING and self.core_pos is not None:
            self._render_charge(ctx)
        elif self.phase == Phase.ACTIVE:
            self._render_flight(ctx)
        elif self.phase == Phase.IMPACT:
            self._render_impact(ctx)

    def _blue_orb(self, ctx: Context, c, r: float, k: float) -> None:
        canvas, now = ctx.canvas, ctx.now
        canvas.add_sprite(c, r * 4.5, _deep(self.BLUE), 0.5 * k, layer="glow")
        canvas.add_sprite(c, r * 2.0, self.BLUE, 0.9 * k, layer="light", falloff=1.6)
        for j in range(2):
            cv2.ellipse(canvas.light, _pt(c), (int(r * 1.6), int(r * 1.6)), math.degrees(now * 5.0) + j * 180, 0, 100,
                        fx.scale_color((255, 200, 120), 0.7 * k), max(1, int(r * 0.12)), cv2.LINE_AA)
        fx.ring(canvas, c, r * 0.95, (255, 235, 200), max(1, int(r * 0.16)), 0.9 * k)
        cv2.circle(canvas.light, _pt(c), max(1, int(r * 0.8)), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas.frame, _pt(c), max(1, int(r * 0.8)), (34, 10, 4), -1, cv2.LINE_AA)

    def _red_orb(self, ctx: Context, c, r: float, k: float) -> None:
        canvas = ctx.canvas
        flick = 0.85 + 0.15 * math.sin(ctx.now * 43.0)
        canvas.add_sprite(c, r * 4.5, _deep(self.RED), 0.6 * k, layer="glow")
        canvas.add_sprite(c, r * 2.2 * flick, self.RED, 1.0 * k, layer="light", falloff=1.7)
        canvas.add_sprite(c, r * 1.1, (220, 225, 255), 1.2 * k, layer="light", falloff=1.1)
        canvas.light_circle(c, max(1.0, r * 0.35), (255, 255, 255), k)
        rr = r * (1.3 + 1.5 * ((ctx.now * 2.5) % 1.0))
        fx.ring(canvas, c, rr, self.RED, max(1, int(r * 0.1)), 0.5 * k * (1.0 - (ctx.now * 2.5) % 1.0))

    def _crackle(self, ctx: Context, c, r: float, n: int, k: float, color, reach: float = 1.0) -> None:
        rng = np.random.default_rng((self._seed + int(ctx.now * 30)) & 0x7FFFFFFF)
        lines = []
        for _ in range(n):
            a = rng.uniform(0, TAU)
            d = np.array([math.cos(a), math.sin(a)], np.float32)
            p0 = c + d * r * 0.8
            p1 = c + d * r * (1.3 + rng.uniform(0.2, 0.9) * reach)
            lines.append(_ipoly(fx.jagged_line(p0, p1, 4, r * 0.15)))
        cv2.polylines(ctx.canvas.light, lines, False, fx.scale_color(color, k), max(1, int(r * 0.05)), cv2.LINE_AA)

    def _render_charge(self, ctx: Context) -> None:
        canvas = ctx.canvas
        p, ik = self.progress, self.inten
        diag = self._diag(ctx)
        rb = 0.02 * diag * self.size
        appear = fx.ease_out(min(1.0, p / 0.22))
        conv = self._conv()
        merge = fx.smoothstep((p - 0.7) / 0.3)
        wk = self.warp_k
        dist = float(np.linalg.norm(self.blue_pos - self.red_pos))
        if merge < 1.0:
            k = appear * (1.0 - merge) * ik
            r = rb * (0.6 + 0.4 * appear) * (1.0 - 0.4 * conv)
            # Arcs bridging the two as they close in.
            if conv > 0.05 and dist > 2.0:
                rng = np.random.default_rng((self._seed + int(ctx.now * 30)) & 0x7FFFFFFF)
                for j in range(2):
                    pts = fx.jagged_line(self.blue_pos, self.red_pos, 7, min(dist * 0.15, rb * 1.5))
                    pts += rng.normal(0, 1, pts.shape).astype(np.float32)
                    cv2.polylines(canvas.light, [_ipoly(pts)], False,
                                  fx.scale_color(_mix(self.body, self.core, 0.4), 0.8 * conv * k), max(1, int(rb * 0.1)),
                                  cv2.LINE_AA)
                canvas.glow_line(self.blue_pos, self.red_pos, _deep(self.body), rb * 1.2, 0.35 * conv * k)
            self._blue_orb(ctx, self.blue_pos, r, k)
            self._red_orb(ctx, self.red_pos, r, k)
            canvas.add_warp(self.blue_pos, rb * 5.0, 0.3 * k * wk, "pinch")
            canvas.add_warp(self.red_pos, rb * 4.0, 0.2 * k * wk, "bulge")
        if merge > 0.0:
            c = self.core_pos
            pulse = 1.0 + 0.12 * math.sin(ctx.now * 22.0)
            R = 0.045 * diag * self.size * (0.4 + 0.6 * merge) * pulse
            k = merge * ik
            canvas.add_sprite(c, R * 4.0, _deep(self.body), 0.8 * k, layer="glow")
            canvas.add_sprite(c, R * 2.0, self.body, 1.0 * k, layer="light", falloff=1.5)
            canvas.add_sprite(c, R * 1.1, self.core, 1.3 * k, layer="light", falloff=1.0)
            canvas.light_circle(c, R * 0.45, (255, 255, 255), k)
            self._crackle(ctx, c, R, 6, 0.9 * k, (255, 200, 250))
            canvas.add_warp(c, R * 4.0, 0.9 * k * wk, "swirl")
            canvas.add_warp(c, R * 3.0, 0.2 * k * wk * pulse, "pinch")

    def _shell(self, n: int, c, r: float, phase: float, amp: float) -> np.ndarray:
        th = np.linspace(0, TAU, n, endpoint=False)
        rr = r * (1.0 + amp * (np.sin(th * 5 + phase) * 0.6 + np.sin(th * 9 - phase * 1.7) * 0.4))
        return c + np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32) * rr[:, None]

    def _wake(self, t: float, length: float, width: float, n: int = 10) -> Optional[np.ndarray]:
        t0 = max(0.0, t - length)
        if t - t0 < 1e-3:
            return None
        left, right = [], []
        side = fx.perp(self.dir)
        for i, tt in enumerate(np.linspace(t0, t, n)):
            pos, sc = self._flight(tt)
            s = i / (n - 1)
            w = self.R0 * sc * width * s ** 0.8
            left.append(pos + side * w)
            right.append(pos - side * w)
        return np.array(left + right[::-1], np.float32)

    def _render_orb(self, ctx: Context, pos, sc: float, k: float, launch: float) -> None:
        canvas = ctx.canvas
        d, side = self.dir, fx.perp(self.dir)
        r = self.R0 * sc * (0.75 + 0.25 * launch)
        body, core, deep = self.body, self.core, _deep(self.body)
        now = ctx.now
        canvas.add_sprite(pos, r * 3.2, deep, 0.9 * k, layer="glow")
        # Shell: a churning purple sphere, elongated slightly forward.
        shell = self._shell(40, pos, r, now * 9.0, 0.06)
        rel = shell - pos
        along = rel @ d
        shell = shell + d[None] * (np.clip(along, 0, None) * 0.15)[:, None]
        canvas.light_poly(shell, body, 0.42 * k)
        cv2.polylines(canvas.light, [_ipoly(shell)], True, fx.scale_color((255, 140, 235), 0.9 * k),
                      max(1, int(r * 0.07)), cv2.LINE_AA)
        inner = self._shell(28, pos + d * r * 0.08, r * 0.62, -now * 11.0, 0.08)
        canvas.light_poly(inner, _mix(body, core, 0.3), 0.4 * k)
        canvas.add_sprite(pos + d * r * 0.1, r * 0.65, core, 0.9 * k, layer="light", falloff=1.4)
        canvas.light_circle(pos + d * r * 0.1, r * 0.22, (255, 255, 255), k)
        # Bow shock ahead of it: the direction reads even when it's small.
        bow = fx.crescent(pos + d * r * 0.25, d, r * 1.25, r * 0.22, span_deg=130, n=20)
        canvas.light_poly(bow, (255, 170, 240), 0.45 * k)
        canvas.glow_poly(bow, deep, 0.5 * k)
        # Side fins of energy peeling backward.
        for s in (-1.0, 1.0):
            tip = pos - d * r * 1.6 + side * s * r * 0.9
            fin = np.array([pos + side * s * r * 0.95 + d * r * 0.1, pos + side * s * r * 0.7 - d * r * 0.5, tip],
                           np.float32)
            canvas.light_poly(fin, body, 0.5 * k)
        self._crackle(ctx, pos, r, 5, 0.85 * k, (255, 210, 250), reach=0.8)
        wk = self.warp_k
        canvas.add_warp(pos, r * 2.4, 0.45 * k * wk, "bulge")
        canvas.add_warp(pos - d * r * 0.8, r * 3.0, 1.1 * k * wk, "swirl")

    def _render_wake(self, ctx: Context, t: float, k: float) -> None:
        canvas = ctx.canvas
        outer = self._wake(t, self.WAKE, 1.05)
        if outer is None:
            return
        canvas.glow_poly(outer, _deep(self.body), 0.55 * k)
        canvas.light_poly(outer, self.body, 0.28 * k)
        mid = self._wake(t, self.WAKE * 0.65, 0.55)
        if mid is not None:
            canvas.light_poly(mid, _mix(self.body, self.core, 0.5), 0.35 * k)
        inner = self._wake(t, self.WAKE * 0.4, 0.2)
        if inner is not None:
            canvas.light_poly(inner, self.core, 0.45 * k)
        # Filaments spiralling along the wake.
        lines = []
        ts = np.linspace(max(0.0, t - self.WAKE), t, 9)
        side = fx.perp(self.dir)
        for j in range(3):
            pts = []
            for i, tt in enumerate(ts):
                pos, sc = self._flight(tt)
                s = i / (len(ts) - 1)
                pts.append(pos + side * self.R0 * sc * 0.8 * s * math.sin(ctx.now * 14.0 + i * 0.9 + j * 2.1))
            lines.append(_ipoly(pts))
        cv2.polylines(canvas.light, lines, False, fx.scale_color((255, 170, 240), 0.6 * k),
                      max(1, int(self.R0 * 0.04)), cv2.LINE_AA)

    def _render_flight(self, ctx: Context) -> None:
        canvas = ctx.canvas
        t, ik = self.progress, self.inten
        pos, sc = self._flight(t)
        launch = min(1.0, self.t / self.LAUNCH)
        self._render_wake(ctx, t, min(1.0, t / 0.1) * ik)
        dur = max(1e-3, self.duration(Phase.ACTIVE))
        for i, lag in enumerate((0.06, 0.12)):  # afterimages
            tt = t - lag / dur
            if tt > 0:
                p2, s2 = self._flight(tt)
                canvas.light_circle(p2, self.R0 * s2 * 0.9, self.body, 0.25 / (i + 1) * ik)
        self._render_orb(ctx, pos, sc, ik, launch)
        if launch < 1.0:  # launch beat: shock ring + streak at the origin
            f = 1.0 - fx.ease_in(launch)
            rr = self.R0 * (1.0 + 2.0 * fx.ease_out(launch))
            fx.ring(canvas, self.p0, rr, self.core, max(1, int(self.R0 * 0.1 * f + 1)), 0.8 * f * ik)
            canvas.add_warp(self.p0, rr, 0.8 * f * self.warp_k, "ripple", width=0.3)

    def _render_impact(self, ctx: Context) -> None:
        canvas = ctx.canvas
        t, ik = self.progress, self.inten
        r0 = self.R0 * max(0.4, self.end_scale)
        c = self.end
        fade = 1.0 - fx.ease_in(t)
        if t < 0.4:
            k = 1.0 - t / 0.4
            self._render_wake(ctx, 1.0, k * k * ik)
            canvas.add_sprite(c, r0 * (1.0 + 2.0 * t), self.core, 1.3 * k * ik, layer="light", falloff=1.3)
        canvas.add_sprite(c, r0 * (3.0 + 3.0 * t), _deep(self.body), 0.8 * fade * fade * ik, layer="glow")
        rad = r0 * (0.8 + 6.0 * fx.ease_out(t))
        fx.ring(canvas, c, rad, self.body, max(1, int(r0 * 0.18 * fade + 1)), 0.8 * fade * ik)
        fx.ring(canvas, c, rad * 0.75, self.core, max(1, int(r0 * 0.06 * fade + 1)), 0.5 * fade * fade * ik)
        t2 = max(0.0, t - 0.18) / 0.82
        if t2 > 0:
            fx.ring(canvas, c, r0 * (0.5 + 4.0 * fx.ease_out(t2)), (255, 140, 235),
                    max(1, int(r0 * 0.08 * (1 - t2) + 1)), 0.5 * (1 - t2) * ik)
        canvas.add_warp(c, rad, 1.0 * fade * self.warp_k, "ripple", width=0.3)
        if t < 0.3:
            canvas.add_warp(c, r0 * 3.0, 0.5 * (1 - t / 0.3) * self.warp_k, "bulge")


# --------------------------------------------------------------------------
# Black Flash
# --------------------------------------------------------------------------
class BlackFlash(_Technique):
    """Location: the fist (``ctx.sword.grip`` when the fist/"sword" tracker is
    confident), else a fist hand's knuckle (MCP) centroid, else any hand's
    knuckles, else the chest. Frozen at trigger time. The punch direction
    (sword motion when swinging, else the hand's wrist->knuckle direction,
    else up) biases the bolts and the afterimage offset.
    """

    name = "black_flash"
    label = "Black Flash"
    default_color = (30, 20, 230)
    default_core = (10, 5, 10)
    TICK = 1.0 / 20.0  # bolt pattern refresh interval (s)
    ECHOES = 2

    def _clear(self) -> None:
        self.point: Optional[np.ndarray] = None
        self.dir = UP.copy()
        self.radius = 0.0
        self._seed = 0
        self._last_tick = 0
        self._bolt_cache: dict = {}

    def _locate(self, ctx: Context) -> Tuple[np.ndarray, np.ndarray]:
        if self._sword_ok(ctx):
            sw = ctx.sword
            p = _finite(sw.grip)
            mv = _finite(sw.motion)
            spd = float(sw.speed) if math.isfinite(float(sw.speed)) else 0.0
            d = fx.unit(mv) if (mv is not None and spd > SWING_MIN * 0.6) else fx.unit(_finite(sw.direction))
            return p, d
        best = None
        for _, pts, g in self._hands(ctx):
            rank = 0 if g == "fist" else 1
            if best is None or rank < best[0]:
                best = (rank, pts[MCP].mean(axis=0), fx.unit(pts[9] - pts[0]))
        if best is not None:
            p = _finite(best[1])
            if p is not None:
                return p, best[2]
        return self._chest(ctx), UP.copy()

    def on_enter(self, phase: Phase, ctx: Context) -> None:
        ik = self.inten
        if phase == Phase.CHARGING:
            self._clear()
            self._seed = int(self._rng.integers(0, 1 << 30))
            self.point, self.dir = self._locate(ctx)
            self.radius = 0.02 * self._diag(ctx) * self.size
        elif phase == Phase.ACTIVE:
            if self.point is None:
                self.on_enter(Phase.CHARGING, ctx)
                self.phase = Phase.ACTIVE
            q, c, diag = self._q(ctx), self.point, self._diag(ctx)
            n = int(55 * q)
            ang = self._rng.uniform(0, TAU, n)
            v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
            v = fx.unit(self.dir) * 0.5 + v  # sprayed forward along the punch
            s = self._rng.uniform(0.3, 1.0, (n, 1)).astype(np.float32) * diag * 0.9
            h = n * 2 // 3
            ctx.particles.emit(h, c, v[:h] * s[:h], life=(0.12, 0.35), size=(1.0, 2.6), color=self.body, drag=5.0)
            ctx.particles.emit(n - h, c, v[h:] * s[h:] * 0.7, life=(0.08, 0.2), size=(0.8, 1.6),
                               color=(170, 160, 255), drag=6.0)
            # A dark-red flash: the frame drops toward black for an instant.
            self._flash(ctx, 0.45 * min(1.5, ik), (8, 0, 40))
            self._shake(ctx, 15.0 * ik)
        elif phase == Phase.COOLDOWN:
            self._clear()

    def on_update(self, ctx: Context) -> None:
        if self.point is None:
            return
        if self.phase == Phase.IMPACT:
            n = self._emit(ctx, 50.0 * (1.0 - self.progress))
            if n:
                ctx.particles.emit(n, self.point, (0, 0), spread_pos=self.radius * 3.0, spread_vel=self.radius * 4.0,
                                   life=(0.2, 0.45), size=(0.8, 1.6), color=(20, 20, 160), drag=3.0)

    # --- render ------------------------------------------------------------
    def _bolts(self, tick: int, diag: float) -> List[Tuple[np.ndarray, int]]:
        key = (tick, round(diag))
        hit = self._bolt_cache.get(key)
        if hit is not None:
            return hit
        if len(self._bolt_cache) > 8:
            self._bolt_cache.clear()
        hit = self._bolt_cache[key] = self._pack(self._make_bolts(tick, diag))
        return hit

    def _make_bolts(self, tick: int, diag: float) -> List[Tuple[np.ndarray, int]]:
        rng = random.Random((self._seed + tick * 7919) & 0x7FFFFFFF)
        out = []
        base = math.atan2(float(self.dir[1]), float(self.dir[0]))
        m = 6
        for i in range(m):
            a = base + TAU * i / m + rng.uniform(-0.35, 0.35)
            fwd = max(0.0, math.cos(a - base))
            ln = 0.15 * diag * self.size * rng.uniform(0.6, 1.0) * (0.75 + 0.6 * fwd)
            out.extend(branching_bolts(rng, (0.0, 0.0), a, ln, depth=2, segments=6, jag=0.55, branch_p=0.3))
        return out

    @staticmethod
    def _pack(bolts) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Per level 0..2: (all points concatenated, split indices) for batched transforms."""
        out = []
        for lvl in range(3):
            polys = [pts for pts, lv in bolts if min(lv, 2) == lvl]
            if polys:
                out.append((np.concatenate(polys, 0), np.cumsum([len(p) for p in polys])[:-1]))
            else:
                out.append((np.zeros((0, 2), np.float32), np.zeros(0, np.int64)))
        return out

    def _draw_bolts(self, ctx: Context, packed, c, scale: float, k: float, core_k: float) -> None:
        canvas = ctx.canvas
        if k <= 0.01:
            return
        rim = (50, 40, 245)
        g = canvas.gs
        widths = (max(4, int(self.radius * 0.32)), max(3, int(self.radius * 0.2)), max(1, int(self.radius * 0.08)))
        glow_col = _scaled(_deep(self.body), 0.45 * k)
        for lvl, (allp, cuts) in enumerate(packed):
            if not len(allp):
                continue
            P = np.clip(c + allp * scale, -COORD_LIMIT, COORD_LIMIT)
            ip = np.split(np.rint(P).astype(np.int32), cuts)
            gp = np.split(np.rint(P * g).astype(np.int32), cuts)
            cv2.polylines(canvas.glow, gp, False, glow_col, max(1, int(widths[lvl] * 1.5 * g + 1)), cv2.LINE_AA)
            cv2.polylines(canvas.light, ip, False, fx.scale_color(rim, k), widths[lvl], cv2.LINE_AA)
            if core_k > 0.05 and lvl < 2:
                # Black core line: zero light + black camera pixels.
                cw = max(1, int(round(widths[lvl] * 0.6)))
                cv2.polylines(canvas.light, ip, False, (0, 0, 0), cw, cv2.LINE_AA)
                cv2.polylines(canvas.frame, ip, False, tuple(int(x) for x in self.core), cw, cv2.LINE_AA)

    def render(self, ctx: Context) -> None:
        if self.point is None or self.phase not in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT):
            return
        canvas = ctx.canvas
        c, R, ik, diag = self.point, self.radius, self.inten, self._diag(ctx)
        body = self.body
        wk = self.warp_k
        if self.phase == Phase.CHARGING:
            p = fx.ease_out(self.progress)
            canvas.tint, canvas.tint_amount = (40, 32, 40), max(canvas.tint_amount, 0.45 * p)
            # Converging speed lines and a dark pinpoint.
            rng = np.random.default_rng(self._seed)
            n = 14
            ang = rng.uniform(0, TAU, n)
            v = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
            far = 0.22 * diag * (1.0 - 0.6 * p)
            a = c + v * far
            b = c + v * (far * 0.35 + R * 0.6)
            cv2.polylines(canvas.light, list(_ipoly(np.stack([a, b], 1))), False,
                          fx.scale_color(body, 0.6 * p * ik), 1, cv2.LINE_AA)
            r = max(2.0, R * (0.15 + 0.25 * p))
            canvas.add_sprite(c, R * 2.0, body, 0.7 * p * ik, layer="light", falloff=1.5)
            cv2.circle(canvas.light, _pt(c), int(r * 1.2), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas.frame, _pt(c), int(r), (0, 0, 0), -1, cv2.LINE_AA)
            canvas.add_warp(c, R * 5.0, 0.3 * p * wk, "pinch")
            return
        bt = self._burst_time()
        u = bt / self._burst_span()
        fade = 1.0 - fx.smoothstep(u)
        active = self.phase == Phase.ACTIVE
        # Time-slow grade recovers.
        canvas.tint, canvas.tint_amount = (40, 32, 40), max(canvas.tint_amount, 0.45 * (1.0 - fx.smoothstep(u * 1.4)))
        if bt < 0.12:
            canvas.aberration = max(canvas.aberration, 5.0 * (1.0 - bt / 0.12))
        tick = int(bt / self.TICK) if active else self._last_tick
        if active:
            self._last_tick = tick
        grow = fx.ease_out(min(1.0, bt / 0.08))
        # Echoes: the previous bolt patterns, larger and fainter (afterimage).
        for j in range(self.ECHOES, 0, -1):
            if tick - j < 0:
                continue
            ej = 0.35 ** j * fade
            off = -fx.unit(self.dir) * R * 0.6 * j
            self._draw_bolts(ctx, self._bolts(tick - j, diag), c + off, grow * (1.0 + 0.12 * j), 0.8 * ej * ik, 0.0)
        main_k = (1.0 if active else (1.0 - fx.smoothstep(self.progress * 1.6))) * ik
        self._draw_bolts(ctx, self._bolts(tick, diag), c, grow, main_k, main_k)
        # Dark impact core with a red rim.
        cr = R * (1.3 - 0.6 * u)
        canvas.add_sprite(c, cr * 5.0, _deep(body), 0.9 * fade * ik, layer="glow")
        canvas.add_sprite(c, cr * 2.4, body, 1.0 * fade * ik, layer="light", falloff=1.4)
        fx.ring(canvas, c, cr, (90, 80, 255), max(1, int(R * 0.15)), fade * ik)
        cv2.circle(canvas.light, _pt(c), max(1, int(cr * 0.85)), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas.frame, _pt(c), max(1, int(cr * 0.85 * fade + 1)), tuple(int(x) for x in self.core), -1,
                   cv2.LINE_AA)
        # Rapid shockwave: red rim with a black inner edge.
        sw_u = min(1.0, bt / 0.35)
        if sw_u < 1.0:
            rr = R * (1.5 + 12.0 * fx.ease_out(sw_u))
            sk = (1.0 - sw_u) ** 1.5
            fx.ring(canvas, c, rr, body, max(1, int(R * 0.35 * sk + 1)), 0.9 * sk * ik)
            cv2.circle(canvas.frame, _pt(c), int(rr * 0.94), (0, 0, 0), max(1, int(R * 0.12 * sk + 1)), cv2.LINE_AA)
            canvas.add_warp(c, rr * 1.15, 1.1 * sk * wk, "ripple", width=0.12)
