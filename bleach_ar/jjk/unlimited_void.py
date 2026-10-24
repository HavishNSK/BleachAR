"""Domain Expansion: Unlimited Void (Gojo Satoru).

The room is replaced by an infinite, moving cosmos while the user stays
visible (``canvas.backdrop()`` is composited behind the person mask).

Phases (durations from ``cfg["abilities"]["unlimited_void"]``):

* CHARGING  - stillness: the room dims and desaturates, a point of light
  gathers at the user. At ~30 % a thin bright boundary appears around the
  torso and accelerates outward; the void is visible *inside* the sphere
  (the backdrop is the camera image outside it). The title is shown then.
  Entering ACTIVE fires one controlled white-out flash.
* ACTIVE    - the whole room is the void: a rotating/zooming nebula with a
  white core behind the user, three parallax star layers falling outward
  from a vanishing point at the user, receding twisted polygon tunnel
  (lattice), counter-rotating dial rings and orbit arcs, and "information
  streams" (dash/glyph packets racing outward). Periodic ripple warps (the
  nebula/lattice rotate themselves, so no full-screen swirl is needed) and chromatic-aberration pulses. Speed and brightness build over the
  domain so it gets more overwhelming.
* IMPACT    - collapse: the boundary contracts back to the user as a broken,
  shedding ring (the room returns from the outside in), a soft pop flash
  when it closes, backdrop fades to 0.
* COOLDOWN  - nothing rendered.

Controls: ``trigger()`` from IDLE expands the domain. Calling it again while
CHARGING or ACTIVE collapses it early (jumps to IMPACT). During IMPACT /
COOLDOWN it does nothing.

Locking: other JJK techniques stay usable inside the domain. The domain sets
``ctx.locked_by = "unlimited_void"`` only during its own CHARGING (so another
domain cannot start mid-expansion) and refuses to start while ``locked_by``
belongs to anything else (e.g. ``"malevolent_shrine"``).

Performance: static pieces (nebula + baked dust + core, half resolution) are
built once per (resolution, quality); per frame is one warpAffine, vectorised
star/stream updates and a handful of batched ``cv2.polylines`` calls.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from ..abilities import Ability, Context, Phase
from ..fx import ease_in, ease_out, noise_texture, ring, smoothstep

WHITE = (255, 255, 255)
RING_COLOR = (255, 215, 170)  # blue-white (BGR)
INDIGO = (235, 110, 120)
AURA_WHITE = (255, 240, 230)
MAX_STARS = 2400
P_RING = 0.3  # CHARGING progress at which the boundary appears


def _quality(ctx: Context) -> str:
    cap = int(getattr(ctx.particles, "capacity", 900))
    if cap <= 500:
        return "low"
    if cap >= 1400:
        return "high"
    return "medium"


class _Cosmos:
    """The void environment for one resolution/quality. Paints into a buffer."""

    def __init__(self, w: int, h: int, quality: str, star_count: int, streams: int, seed: int = 7):
        self.w, self.h, self.quality = w, h, quality
        self.diag = float(math.hypot(w, h))
        self.D = 0.5 * self.diag  # projection scale
        self.rng = np.random.default_rng(seed)
        qmul = {"low": 0.5, "medium": 1.0, "high": 1.6}[quality]
        self.n = int(max(50, min(MAX_STARS, star_count * qmul)))
        self.m = int(max(6, min(80, streams * {"low": 0.6, "medium": 1.0, "high": 1.4}[quality])))
        self._build_base()
        self._init_stars()
        self._init_streams()
        arcs = {"low": 4, "medium": 7, "high": 10}[quality]
        r = self.rng
        self.arcs = np.stack([r.uniform(0.25, 1.3, arcs), r.uniform(0, 360, arcs), r.uniform(20, 90, arcs),
                              r.choice([-1.0, 1.0], arcs) * r.uniform(8, 30, arcs)], 1).astype(np.float32)
        self.t = 0.0  # cosmos time (speed-scaled)
        self.rot = 0.0  # global rotation (rad)

    # --- static base --------------------------------------------------------
    def _build_base(self) -> None:
        """Half-resolution square nebula covering radius ``diag`` around its centre.

        The smooth fields are evaluated at half that size again and upsampled
        (keeps the one-time build cheap); the dust is added at full base size."""
        S = int(self.diag) | 1
        c = S // 2
        s = S // 2
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        dx, dy = xx - s * 0.5, yy - s * 0.5
        r = np.sqrt(dx * dx + dy * dy) / (s * 0.5)  # 1.0 == diag in render px
        th = np.arctan2(dy, dx)
        n1 = noise_texture(s, s, cells=3, seed=11, octaves=4)
        n2 = noise_texture(s, s, cells=5, seed=23, octaves=3)
        n3 = noise_texture(s, s, cells=9, seed=5, octaves=2)
        arms = 0.5 + 0.5 * np.cos(3.0 * th + 9.0 * np.sqrt(r) + n3 * 2.5)  # faint spiral structure
        cloud = np.clip((n1 - 0.38) * 2.2, 0, 1) ** 2 * (0.45 + 0.55 * arms)
        fil = np.clip(1.0 - np.abs(n2 - 0.5) * 7.0, 0, 1) ** 3 * (0.4 + 0.6 * arms)  # filaments
        vign = np.clip(1.15 - r * 0.9, 0.25, 1.0)
        core = np.exp(-(r / 0.035) ** 2)
        halo = np.exp(-(r / 0.16) ** 2)
        far_halo = np.exp(-(r / 0.45) ** 2)
        small = np.empty((s, s, 3), np.float32)
        # BGR palette: black -> indigo -> deep blue -> cyan -> white
        for ch, (bg, ind, blu, cy, hal, wh) in enumerate(((10, 120, 150, 255, 255, 255),
                                                           (3, 22, 55, 190, 215, 255),
                                                           (4, 55, 12, 110, 175, 255))):
            small[:, :, ch] = (bg + ind * cloud * vign + blu * fil * vign * 0.9 + cy * fil * cloud * 0.6
                               + hal * (0.55 * halo + 0.22 * far_halo) + wh * core)
        img = cv2.resize(small, (S, S), interpolation=cv2.INTER_LINEAR)
        # Baked dust: thousands of tiny static stars (rotate with the nebula).
        nd = {"low": 2500, "medium": 4500, "high": 7000}[self.quality]
        rr = self.rng
        px = rr.integers(0, S, nd)
        py = rr.integers(0, S, nd)
        b = (rr.pareto(2.2, nd) * 40 + 25).clip(0, 255).astype(np.float32)
        tint = np.where(rr.random(nd)[:, None] < 0.8, np.array([[1.0, 0.9, 0.82]]), np.array([[1.0, 0.6, 0.75]]))
        img[py, px] += b[:, None] * tint
        self.base = np.clip(img, 0, 255).astype(np.uint8)
        self.base_c = float(c)
        self.bg = (10, 3, 4)

    # --- stars ---------------------------------------------------------------
    def _init_stars(self) -> None:
        r, n = self.rng, self.n
        self.layer = r.choice(3, n, p=[0.55, 0.3, 0.15]).astype(np.int8)
        self.theta = r.uniform(0, 2 * math.pi, n).astype(np.float32)
        self.rho = (np.sqrt(r.uniform(0.0004, 1.0, n)) * 1.35).astype(np.float32)
        self.z = r.uniform(0.1, 1.0, n).astype(np.float32)
        self.speed = (np.array([0.05, 0.14, 0.38], np.float32)[self.layer] * r.uniform(0.8, 1.25, n)).astype(np.float32)
        self.spin = np.array([0.025, -0.045, 0.08], np.float32)[self.layer]
        self.bright = (np.array([110, 175, 255], np.float32)[self.layer] * r.uniform(0.45, 1.0, n)).astype(np.float32)
        self.layer_col = np.array([[1.0, 0.82, 0.72], [1.0, 0.93, 0.86], [1.0, 0.98, 0.96]], np.float32)

    def _respawn(self, idx: np.ndarray) -> None:
        k = idx.size
        if k == 0:
            return
        r = self.rng
        self.theta[idx] = r.uniform(0, 2 * math.pi, k)
        self.rho[idx] = np.sqrt(r.uniform(0.0004, 1.0, k)) * 1.35
        self.z[idx] = r.uniform(0.88, 1.0, k)

    # --- streams -------------------------------------------------------------
    def _init_streams(self) -> None:
        r, m, K = self.rng, self.m, 7
        self.K = K
        self.s_theta = r.uniform(0, 2 * math.pi, m).astype(np.float32)
        self.s_speed = r.uniform(0.12, 0.38, m).astype(np.float32)
        self.s_phase = r.random(m).astype(np.float32)
        self.s_len = r.uniform(0.004, 0.03, (m, K)).astype(np.float32)
        gaps = r.uniform(0.006, 0.03, (m, K)).astype(np.float32)
        self.s_off = np.cumsum(self.s_len + gaps, axis=1) - (self.s_len + gaps)
        self.s_tick = r.random((m, K)) < 0.4
        self.s_group = r.choice(3, m, p=[0.55, 0.33, 0.12]).astype(np.int8)  # cyan / white / violet

    # --- simulation ------------------------------------------------------------
    def step(self, dt: float, mult: float, spin_mult: float) -> None:
        dt = float(min(dt, 0.1))
        self.t += dt * mult
        self.rot += dt * 0.05 * spin_mult
        self.z -= self.speed * (mult * dt)
        self.theta += self.spin * (spin_mult * dt)
        r = self.rho * self.D / np.maximum(self.z, 1e-3)
        self._respawn(np.nonzero((self.z < 0.06) | (r > self.diag * 1.15))[0])

    # --- painting ------------------------------------------------------------
    def paint(self, bd: np.ndarray, anchor, energy: float, mult: float, trail: float = 0.1) -> None:
        ax, ay = float(anchor[0]), float(anchor[1])
        zoom = 1.0 + 0.04 * math.sin(self.t * 0.35) + 0.0025 * self.t
        ang = math.degrees(self.rot)
        M = cv2.getRotationMatrix2D((self.base_c, self.base_c), ang, 2.0 * zoom)
        M[0, 2] += ax - self.base_c
        M[1, 2] += ay - self.base_c
        cv2.warpAffine(self.base, M, (self.w, self.h), dst=bd, flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=self.bg)
        self._paint_lattice(bd, ax, ay, energy)
        self._paint_streams(bd, ax, ay, energy)
        self._paint_stars(bd, ax, ay, energy, mult, trail)

    def _paint_stars(self, bd, ax, ay, energy, mult, trail) -> None:
        z = self.z
        cth, sth = np.cos(self.theta), np.sin(self.theta)
        r = self.rho * self.D / z
        zt = np.minimum(z + self.speed * (mult * trail), 1.0)
        rt = self.rho * self.D / zt
        x, y = ax + cth * r, ay + sth * r
        fade = np.clip((1.0 - z) / 0.12, 0.0, 1.0)
        b = np.minimum(255.0, self.bright * fade * np.minimum(1.6, 0.45 + 0.22 / z) * energy)
        self.heads = (x, y, b)
        w, h = self.w, self.h
        on = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1) & (b > 8)
        length = r - rt
        dots = on & (length < 1.6)
        if dots.any():
            xi, yi = x[dots].astype(np.int32), y[dots].astype(np.int32)
            col = (self.layer_col[self.layer[dots]] * b[dots, None]).astype(np.uint8)
            bd[yi, xi] = np.maximum(bd[yi, xi], col)
            big = self.layer[dots] >= 1
            if big.any():
                xb, yb, cb = xi[big], yi[big], (col[big] * 0.6).astype(np.uint8)
                bd[yb, xb + 1] = np.maximum(bd[yb, xb + 1], cb)
                bd[yb + 1, xb] = np.maximum(bd[yb + 1, xb], cb)
        streak = on & (length >= 1.6)
        if streak.any():
            idx = np.nonzero(streak)[0]
            seg = np.empty((idx.size, 2, 2), np.int32)
            seg[:, 0, 0] = x[idx]
            seg[:, 0, 1] = y[idx]
            seg[:, 1, 0] = ax + cth[idx] * rt[idx]
            seg[:, 1, 1] = ay + sth[idx] * rt[idx]
            lvl = np.minimum(3, (b[idx] / 64).astype(np.int32))
            lay = self.layer[idx]
            thick = np.where((lay == 2) & (lvl >= 3) & (length[idx] > 25), 2, 1)
            key = lay.astype(np.int32) * 8 + lvl * 2 + (thick - 1)
            for k in np.unique(key):
                sel = key == k
                L, lv, th = k // 8, (k % 8) // 2, k % 2 + 1
                col = tuple(float(c) for c in self.layer_col[L] * (lv + 0.75) * 64)
                cv2.polylines(bd, list(seg[sel]), False, col, int(th), cv2.LINE_AA)

    def _paint_streams(self, bd, ax, ay, energy) -> None:
        if energy < 0.02:
            return
        D = self.D
        rmin, rmax = 0.05 * D, self.diag * 1.05
        ratio = rmax / rmin
        u = np.mod(self.s_phase[:, None] + self.t * self.s_speed[:, None] + self.s_off, 1.0)
        ub = np.minimum(u + self.s_len, 1.0)
        ra = rmin * np.power(ratio, u)
        rb = rmin * np.power(ratio, ub)
        th = (self.s_theta + self.rot * 1.6)[:, None] + 0.15 * u  # slight curve as they fly out
        c, s = np.cos(th), np.sin(th)
        bri = np.sin(np.pi * u) ** 0.8 * energy
        segs = [np.stack([np.stack([ax + c * ra, ay + s * ra], -1), np.stack([ax + c * rb, ay + s * rb], -1)], -2)]
        # Glyph ticks: short perpendicular bars at the dash heads.
        tl = 2.0 + rb * 0.018
        hx, hy = ax + c * rb, ay + s * rb
        tick = np.stack([np.stack([hx - s * tl, hy + c * tl], -1), np.stack([hx + s * tl, hy - c * tl], -1)], -2)
        groups = np.broadcast_to(self.s_group[:, None], u.shape)
        all_seg = np.concatenate([segs[0].reshape(-1, 2, 2), tick[self.s_tick].reshape(-1, 2, 2)], 0)
        all_b = np.concatenate([bri.ravel(), bri[self.s_tick] * 0.8])
        all_g = np.concatenate([groups.ravel(), groups[self.s_tick]])
        ok = np.isfinite(all_seg).all(axis=(1, 2)) & (all_b > 0.05)
        all_seg = np.clip(all_seg[ok], -4 * self.diag, 4 * self.diag).astype(np.int32)
        lvl = np.minimum(2, (all_b[ok] * 3).astype(np.int32))
        key = all_g[ok].astype(np.int32) * 3 + lvl
        palette = ((255, 225, 120), (255, 250, 245), (255, 120, 190))  # cyan, white, violet (BGR)
        for k in np.unique(key):
            g, lv = k // 3, k % 3
            col = tuple(cc * (0.3 + 0.35 * lv) for cc in palette[g])
            cv2.polylines(bd, list(all_seg[key == k]), False, col, 1, cv2.LINE_AA)

    def _paint_lattice(self, bd, ax, ay, energy) -> None:
        D, t = self.D, self.t
        e = energy
        # Twisted polygon tunnel receding to the vanishing point.
        nr, sides = (6 if self.quality == "low" else 9), 6
        zz = 0.05 + 0.95 * np.mod(np.arange(nr) / nr - t * 0.05, 1.0)
        order = np.argsort(zz)
        zz = zz[order]
        rad = 0.16 * D / zz
        rot = self.rot * -2.0 + 1.4 * zz
        fade = (np.clip((1.0 - zz) / 0.25, 0, 1) * np.clip((rad - 0.06 * D) / (0.2 * D), 0, 1)
                * np.clip(1.6 - rad / self.diag, 0, 1))
        ang = rot[:, None] + np.arange(sides)[None, :] * (2 * math.pi / sides)
        vx = ax + np.cos(ang) * rad[:, None]
        vy = ay + np.sin(ang) * rad[:, None] * 0.92
        V = np.clip(np.stack([vx, vy], -1), -4 * self.diag, 4 * self.diag).astype(np.int32)
        for i in range(nr):
            f = fade[i] * e
            if f > 0.03:
                cv2.polylines(bd, [V[i]], True, (150 * f, 70 * f, 40 * f), 1, cv2.LINE_AA)
        spokes = [np.ascontiguousarray(V[:, j]) for j in range(sides)]
        cv2.polylines(bd, spokes, False, (110 * e, 45 * e, 25 * e), 1, cv2.LINE_AA)
        # Counter-rotating dial rings with tick marks.
        c = (int(ax), int(ay))
        for R, n, spd, colk in ((0.34 * D, 120, 1.0, 0.55), (0.62 * D, 72, -0.6, 0.4), (1.05 * D, 180, 0.35, 0.3)):
            a = np.arange(n) * (2 * math.pi / n) + self.rot * 3.0 * spd
            long = (np.arange(n) % 6 == 0)
            r1 = R + np.where(long, 0.05, 0.018) * D
            seg = np.stack([np.stack([ax + np.cos(a) * R, ay + np.sin(a) * R], -1),
                            np.stack([ax + np.cos(a) * r1, ay + np.sin(a) * r1], -1)], -2).astype(np.int32)
            k = colk * e
            cv2.polylines(bd, list(seg), False, (255 * k, 210 * k, 170 * k), 1, cv2.LINE_AA)
            cv2.circle(bd, c, int(R), (200 * k, 120 * k, 80 * k), 1, cv2.LINE_AA)
        # Orbit arcs and a slowly turning hexagram.
        for rr, a0, span, spd in self.arcs:
            R = int(rr * D)
            st = a0 + spd * t
            cv2.ellipse(bd, c, (R, R), 0, st, st + span, (255 * 0.45 * e, 200 * 0.45 * e, 150 * 0.45 * e), 1,
                        cv2.LINE_AA)
        Rh = 0.8 * D
        for off in (0.0, math.pi / 3):
            a = self.rot * -1.2 + off + np.arange(3) * (2 * math.pi / 3)
            tri = np.stack([ax + np.cos(a) * Rh, ay + np.sin(a) * Rh], -1).astype(np.int32)
            cv2.polylines(bd, [tri], True, (170 * 0.35 * e, 80 * 0.35 * e, 90 * 0.35 * e), 1, cv2.LINE_AA)


class UnlimitedVoid(Ability):
    name = "unlimited_void"
    label = "Unlimited Void"
    takes_over = True

    def __init__(self, params: dict):
        super().__init__(params)
        self._cosmos: Optional[_Cosmos] = None
        self._clear_state()

    def _clear_state(self) -> None:
        self._anchor: Optional[np.ndarray] = None
        self._r_now = 0.0  # current boundary radius (render px)
        self._r_from = 0.0  # radius at the start of the collapse
        self._titled = False
        self._popped = False
        self._next_pulse = 0.0
        self._pulse_t = -10.0
        self._ripple_t = 0.0

    # --- params --------------------------------------------------------------
    def _p(self, key, default):
        return self.params.get(key, default)

    # --- transitions ---------------------------------------------------------
    def can_trigger(self, ctx: Context) -> bool:
        if self.phase == Phase.IDLE:
            return ctx.locked_by is None or ctx.locked_by == self.name
        # Pressing again inside the domain collapses it early.
        return self.phase in (Phase.CHARGING, Phase.ACTIVE)

    def trigger(self, ctx: Context) -> bool:
        if not self.can_trigger(ctx):
            return False
        if self.phase == Phase.IDLE:
            self._advance_from(-1, ctx)
        else:
            self._enter(Phase.IMPACT, ctx)  # early collapse
        return True

    def _unlock(self, ctx: Context) -> None:
        if ctx.locked_by == self.name:
            ctx.locked_by = None

    def reset(self, ctx: Context) -> None:
        super().reset(ctx)
        self._unlock(ctx)
        ctx.energy_override = None
        self._clear_state()

    def on_enter(self, phase: Phase, ctx: Context) -> None:
        cv = ctx.canvas
        if phase == Phase.CHARGING:
            self._clear_state()
            ctx.locked_by = self.name
            self._update_anchor(ctx)
        elif phase == Phase.ACTIVE:
            self._unlock(ctx)
            cv.flash = max(cv.flash, float(self._p("flash", 0.85)))
            cv.flash_color = (255, 248, 240)
            ctx.shaker.kick(6)
            self._r_now = self._full_radius(ctx)
            self._next_pulse = 0.6
            self._ripple_t = 0.0
            if not self._titled:
                self._show_title(ctx)
        elif phase == Phase.IMPACT:
            self._unlock(ctx)
            # Cancelled before the boundary appeared -> nothing to contract, just the pop.
            self._r_from = self._r_now
            self._popped = False
            cv.flash = max(cv.flash, 0.25)
            cv.flash_color = (255, 240, 225)
        elif phase in (Phase.COOLDOWN, Phase.IDLE):
            self._unlock(ctx)

    def _show_title(self, ctx: Context) -> None:
        self._titled = True
        ctx.show_title("DOMAIN EXPANSION", sub="Unlimited Void", color=(255, 225, 190),
                       duration=float(self._p("title_duration", 3.0)))

    # --- geometry --------------------------------------------------------------
    def _update_anchor(self, ctx: Context) -> None:
        cv = ctx.canvas
        w, h = (cv.w, cv.h) if cv.w else (ctx.tracking.width or 960, ctx.tracking.height or 540)
        c = ctx.body_center()
        if c is not None and np.all(np.isfinite(c)):
            target = np.array([c[0], c[1] - 0.09 * h], np.float32)  # behind the neck/head
            target = np.clip(target, [0, 0], [w, h])
        elif self._anchor is not None:
            return  # hold last anchor on tracking loss
        else:
            target = np.array([w * 0.5, h * 0.42], np.float32)
        if self._anchor is None or not np.all(np.isfinite(self._anchor)):
            self._anchor = target
        else:
            k = 1.0 - math.exp(-6.0 * max(ctx.dt, 0.0))
            self._anchor = self._anchor + (target - self._anchor) * k

    def _anchor_px(self, ctx: Context) -> np.ndarray:
        if self._anchor is None:
            self._update_anchor(ctx)
        return self._anchor

    def _full_radius(self, ctx: Context) -> float:
        cv = ctx.canvas
        w, h = (cv.w, cv.h) if cv.w else (960, 540)
        a = self._anchor if self._anchor is not None else np.array([w / 2, h / 2], np.float32)
        return float(max(math.hypot(a[0] - x, a[1] - y) for x in (0, w) for y in (0, h))) + 8.0

    # --- update ------------------------------------------------------------------
    def on_update(self, ctx: Context) -> None:
        self._update_anchor(ctx)
        ph, p = self.phase, self.progress
        if ph in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT):
            pulse = 0.5 + 0.5 * math.sin(ctx.now * 3.0)
            k = 0.35 + 0.4 * pulse
            ctx.energy_override = tuple(int(INDIGO[i] + (AURA_WHITE[i] - INDIGO[i]) * k) for i in range(3))
        if ph == Phase.CHARGING:
            if p >= P_RING and not self._titled:
                self._show_title(ctx)
            ctx.aura_boost = max(ctx.aura_boost, 1.2 * smoothstep(p))
        elif ph == Phase.IMPACT:
            a = self._anchor_px(ctx)
            q = min(1.0, p / 0.75)
            if q < 1.0 and self._r_now > 4:
                n = int(10 * (1 - q)) + 2
                ang = np.random.uniform(0, 2 * math.pi, n)
                dirs = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                pos = a + dirs * self._r_now
                vel = -dirs * np.random.uniform(150, 500, (n, 1)) + np.random.normal(0, 60, (n, 2))
                ctx.particles.emit(n, pos, vel.astype(np.float32), life=(0.25, 0.6), size=(1.2, 2.6),
                                   color=RING_COLOR, drag=1.5)
            if q >= 1.0 and not self._popped:
                self._popped = True
                cv = ctx.canvas
                cv.flash = max(cv.flash, 0.3)
                ctx.shaker.kick(7)
                d = ctx.diag
                ang = np.random.uniform(0, 2 * math.pi, 90)
                dirs = np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
                ctx.particles.emit(90, a, dirs * np.random.uniform(0.15 * d, 0.6 * d, (90, 1)).astype(np.float32),
                                   life=(0.3, 0.8), size=(1.2, 3.0), color=AURA_WHITE, drag=2.5)

    # --- render -------------------------------------------------------------------
    def _ensure_cosmos(self, ctx: Context) -> _Cosmos:
        cv = ctx.canvas
        q = _quality(ctx)
        cz = self._cosmos
        if cz is None or (cz.w, cz.h, cz.quality) != (cv.w, cv.h, q):
            self._cosmos = cz = _Cosmos(cv.w, cv.h, q, int(self._p("star_count", 900)),
                                        int(self._p("stream_count", 28)), seed=int(self._p("seed", 7)))
        return cz

    def render(self, ctx: Context) -> None:
        ph = self.phase
        cv = ctx.canvas
        if ph in (Phase.IDLE, Phase.COOLDOWN) or cv.w == 0 or cv.frame is None:
            return
        p, dt = self.progress, ctx.dt
        a = self._anchor_px(ctx)
        if not np.all(np.isfinite(a)):
            self._anchor = a = np.array([cv.w * 0.5, cv.h * 0.42], np.float32)
        inten = float(self._p("intensity", 1.0))
        warps = bool(self._p("warps", True)) and _quality(ctx) != "low"
        full = self._full_radius(ctx)
        cz = self._ensure_cosmos(ctx)

        if ph == Phase.CHARGING:
            s = smoothstep(p / P_RING)
            cv.dim = max(cv.dim, 0.3 * s)
            cv.tint, cv.tint_amount = (95, 82, 80), max(cv.tint_amount, 0.28 * s)
            cv.add_sprite(a, 0.03 * ctx.diag + 0.05 * ctx.diag * p, AURA_WHITE, 0.5 * s * inten, layer="glow")
            cv.add_sprite(a, 6 + 10 * p, WHITE, 0.6 * s * inten)
            if p < P_RING:
                return
            q = (p - P_RING) / (1.0 - P_RING)
            r0 = 0.11 * ctx.diag
            self._r_now = R = r0 + (full - r0) * (q ** 2.2)
            cz.step(dt, 1.6, 1.0)
            self._paint(ctx, cz, a, R, energy=0.7 + 0.3 * q, mult=1.6)
            self._boundary(ctx, a, R, 1.0, broken=False, inten=inten)
            if warps:
                cv.add_warp(a, R, 0.6, kind="ripple", width=0.1)
        elif ph == Phase.ACTIVE:
            t = self.t
            dur = max(self.duration(Phase.ACTIVE), 1e-3)
            build = t / dur
            rush = 3.5 * math.exp(-t * 2.2)  # warp-speed burst on arrival
            mult = (1.0 + 0.9 * build + rush) * float(self._p("speed", 1.0))
            cz.step(dt, mult, 1.0 + 0.8 * build)
            energy = min(1.0, 0.6 + t / 0.5 * 0.4) * (0.9 + 0.25 * build)
            self._r_now = full
            self._paint(ctx, cz, a, None, energy=energy, mult=mult, trail=0.1 + 0.15 * min(rush, 1.0))
            core = 0.2 + 0.06 * math.sin(ctx.now * 2.3)
            cv.add_sprite(a, 0.22 * ctx.diag, (255, 225, 200), core * inten, layer="glow")
            self._star_glow(ctx, cz, inten)
            # Chromatic aberration pulses.
            if t >= self._next_pulse:
                self._pulse_t = t
                self._next_pulse = t + 1.4 + 0.8 * (1 - build)
            ab = float(self._p("aberration", 3.5)) * math.exp(-(t - self._pulse_t) * 4.0)
            if _quality(ctx) != "low":
                cv.aberration = max(cv.aberration, ab)
            if warps:
                self._ripple_t += dt
                rr = (self._ripple_t % 2.4) / 2.4
                cv.add_warp(a, 0.12 * ctx.diag + rr * 0.7 * ctx.diag, 0.35 * (1 - rr), kind="ripple", width=0.1)
        elif ph == Phase.IMPACT:
            q = min(1.0, p / 0.75)
            self._r_now = R = self._r_from * (1.0 - ease_in(q) * 0.98 - 0.02 * q)
            cz.step(dt, 0.6, 1.0)
            fade = 1.0 - smoothstep((p - 0.7) / 0.3)
            if R > 3:
                self._paint(ctx, cz, a, R, energy=1.0, mult=0.6, amount=fade)
                self._boundary(ctx, a, R, 1.0 - 0.4 * q, broken=True, inten=inten)
                if warps:
                    cv.add_warp(a, R * 1.05 + 10, 0.25 * (1 - q), kind="pinch")
            if self._popped:
                pp = min(1.0, (p - 0.75) / 0.25)
                ring(cv, a, 10 + 0.35 * ctx.diag * ease_out(pp), RING_COLOR, 2, (1 - pp) * 0.8 * inten)

    def _paint(self, ctx: Context, cz: _Cosmos, a, R: Optional[float], energy: float, mult: float,
               trail: float = 0.1, amount: float = 1.0) -> None:
        cv = ctx.canvas
        bd = cv.backdrop()
        cz.paint(bd, a, energy, mult, trail)
        if R is not None and R < self._full_radius(ctx) - 4:
            # Outside the boundary the room is still the room.
            bufs = getattr(self, "_rbufs", None)
            if bufs is None or bufs[0].shape != (cv.h, cv.w):  # reused while the size is unchanged
                bufs = self._rbufs = (np.zeros((cv.h, cv.w), np.uint8), np.zeros((cv.h, cv.w), np.float32),
                                      np.zeros((cv.h, cv.w), np.float32))
            mask, win, wout = bufs
            mask.fill(0)
            cv2.circle(mask, (int(a[0]), int(a[1])), int(max(1, R)), 255, -1, cv2.LINE_AA)
            np.multiply(mask, 1.0 / 255.0, out=win, casting="unsafe")
            np.subtract(1.0, win, out=wout)
            cv2.blendLinear(bd, cv.frame, win, wout, dst=bd)
        cv.backdrop_amount = max(cv.backdrop_amount, float(np.clip(amount, 0.0, 1.0)))

    def _boundary(self, ctx: Context, a, R: float, strength: float, broken: bool, inten: float) -> None:
        cv = ctx.canvas
        c = (int(a[0]), int(a[1]))
        if not broken:
            ring(cv, a, R, RING_COLOR, 2, 0.9 * strength * inten)
            cv2.circle(cv.light, c, int(R), (255, 255, 255), 1, cv2.LINE_AA)
            return
        # Shattering boundary: arcs with jittering gaps.
        n = 18
        rng = np.random.default_rng(int(ctx.now * 30))
        st = np.arange(n) * (360.0 / n) + rng.uniform(-6, 6, n)
        span = rng.uniform(8, 17, n)
        col = tuple(int(cc * strength * inten) for cc in (255, 245, 235))
        gcol = tuple(cc * strength * inten for cc in RING_COLOR)
        gc = (int(a[0] * cv.gs), int(a[1] * cv.gs))
        for s0, sp in zip(st, span):
            rr = int(R * rng.uniform(0.97, 1.03))
            cv2.ellipse(cv.light, c, (rr, rr), 0, s0, s0 + sp, col, 2, cv2.LINE_AA)
            g = max(1, int(rr * cv.gs))
            cv2.ellipse(cv.glow, gc, (g, g), 0, s0, s0 + sp, gcol, 2, cv2.LINE_AA)

    def _star_glow(self, ctx: Context, cz: _Cosmos, inten: float) -> None:
        """Bloom for the near, bright star heads (skipped where they are behind the user)."""
        x, y, b = cz.heads
        cv = ctx.canvas
        sel = (cz.layer == 2) & (b > 140) & (x >= 0) & (x < cv.w) & (y >= 0) & (y < cv.h)
        if not sel.any():
            return
        xi, yi, bb = x[sel].astype(np.int32), y[sel].astype(np.int32), b[sel]
        pm = cv.person_mask()
        if pm is not None:
            keep = pm[yi, xi] < 0.3
            xi, yi, bb = xi[keep], yi[keep], bb[keep]
        if xi.size == 0:
            return
        gx = np.minimum((xi * cv.gs).astype(np.int32), cv.gw - 1)
        gy = np.minimum((yi * cv.gs).astype(np.int32), cv.gh - 1)
        np.add.at(cv.glow, (gy, gx), (bb * 0.9 * inten)[:, None] * np.array([1.0, 0.92, 0.85], np.float32))
