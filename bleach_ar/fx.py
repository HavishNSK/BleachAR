"""Procedural rendering primitives shared by every effect.

Rendering model (all colors BGR):

* ``canvas.frame``  - the camera image (uint8, render resolution). Effects may
  darken or tint it directly.
* ``canvas.light``  - uint8 additive layer at render resolution, cleared every
  frame. Draw crisp energy shapes here (cv2 drawing with LINE_AA). It is
  added on top of the frame *and* fed into the bloom pass.
* ``canvas.glow``   - float32 additive layer at ``glow_scale`` resolution.
  Draw soft, wide light here using coordinates multiplied by ``canvas.gs``
  (or use ``canvas.glow_circle`` / ``canvas.glow_poly``). It is blurred and
  upsampled during ``composite``.

Additive light means "brighter = more energy", which is what makes the
effects read as glowing energy rather than pasted-on stickers.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[float, float, float]


def scale_color(color: Sequence[float], k: float) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, c * k))) for c in color)


def ipt(p) -> Tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * t


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([0.0, -1.0], np.float32)


def perp(v) -> np.ndarray:
    return np.array([-v[1], v[0]], np.float32)


class Canvas:
    def __init__(self, glow_scale: float = 0.33, intensity: float = 1.0):
        self.glow_scale = glow_scale
        self.intensity = intensity
        self.frame: Optional[np.ndarray] = None
        self.light: Optional[np.ndarray] = None
        self.glow: Optional[np.ndarray] = None
        self.w = self.h = self.gw = self.gh = 0
        self.gs = glow_scale  # multiply render coords by this for the glow layer
        self.flash = 0.0  # 0..1 white flash, decays in composite
        self.flash_color: Color = (255, 255, 255)
        self.dim = 0.0  # 0..1 darkening of the camera image this frame
        self._white = None
        self._glow_scale_seen = None

    def begin(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self.w, self.h) or self.glow is None or self._glow_scale_seen != self.glow_scale:
            self.w, self.h = w, h
            # Integer downscale factor: INTER_AREA is ~10x faster than with a
            # fractional one, and gs stays exact for coordinate mapping.
            k = max(1, int(round(1.0 / self.glow_scale)))
            self.gw, self.gh = max(8, w // k), max(8, h // k)
            self.gs = self.gw / w
            self._glow_scale_seen = self.glow_scale
            self.light = np.zeros((h, w, 3), np.uint8)
            self.glow = np.zeros((self.gh, self.gw, 3), np.float32)
            self._white = np.full((h, w, 3), 255, np.uint8)
        else:
            self.light.fill(0)
            self.glow.fill(0)
        self.frame = frame
        self.dim = 0.0

    # --- glow-layer helpers (take render-resolution coordinates) ---------
    def glow_circle(self, center, radius: float, color: Sequence[float], strength: float = 1.0) -> None:
        c = (int(center[0] * self.gs), int(center[1] * self.gs))
        r = max(1, int(radius * self.gs))
        k = strength * self.intensity
        cv2.circle(self.glow, c, r, (color[0] * k, color[1] * k, color[2] * k), -1, cv2.LINE_AA)

    def glow_poly(self, pts: np.ndarray, color: Sequence[float], strength: float = 1.0) -> None:
        p = (np.asarray(pts, np.float32) * self.gs).astype(np.int32)
        k = strength * self.intensity
        cv2.fillPoly(self.glow, [p], (color[0] * k, color[1] * k, color[2] * k), cv2.LINE_AA)

    def glow_line(self, a, b, color: Sequence[float], thickness: float, strength: float = 1.0) -> None:
        k = strength * self.intensity
        cv2.line(self.glow, (int(a[0] * self.gs), int(a[1] * self.gs)),
                 (int(b[0] * self.gs), int(b[1] * self.gs)),
                 (color[0] * k, color[1] * k, color[2] * k), max(1, int(thickness * self.gs)), cv2.LINE_AA)

    # --- light-layer helpers ---------------------------------------------
    def light_poly(self, pts: np.ndarray, color: Sequence[float], strength: float = 1.0) -> None:
        cv2.fillPoly(self.light, [np.asarray(pts, np.float32).astype(np.int32)],
                     scale_color(color, strength * self.intensity), cv2.LINE_AA)

    def light_line(self, a, b, color: Sequence[float], thickness: int, strength: float = 1.0) -> None:
        cv2.line(self.light, ipt(a), ipt(b), scale_color(color, strength * self.intensity),
                 max(1, int(thickness)), cv2.LINE_AA)

    def light_circle(self, center, radius: float, color: Sequence[float], strength: float = 1.0,
                     thickness: int = -1) -> None:
        cv2.circle(self.light, ipt(center), max(1, int(radius)),
                   scale_color(color, strength * self.intensity), thickness, cv2.LINE_AA)

    def composite(self, bloom: float = 1.0) -> np.ndarray:
        frame = self.frame
        if self.dim > 0.0:
            cv2.convertScaleAbs(frame, frame, alpha=max(0.0, 1.0 - self.dim))
        cv2.add(frame, self.light, dst=frame)
        # Bloom: the crisp layer bleeds into the soft glow layer.
        small = cv2.resize(self.light, (self.gw, self.gh), interpolation=cv2.INTER_AREA)
        glow = self.glow
        cv2.scaleAdd(small.astype(np.float32), 0.6 * bloom, glow, dst=glow)
        # Two box-blur passes approximate a wide Gaussian at a fraction of the
        # cost (box filters are O(1) per pixel regardless of kernel size).
        # Matches a Gaussian of sigma = 27 render px: two passes of a box of
        # width k give sigma^2 = (k^2 - 1) / 6.
        k = max(3, int(27 * self.gs * 2.449) | 1)
        glow = cv2.blur(cv2.blur(glow, (k, k)), (k, k))
        glow_u8 = cv2.convertScaleAbs(glow, alpha=1.4 * bloom)
        cv2.add(frame, cv2.resize(glow_u8, (self.w, self.h), interpolation=cv2.INTER_LINEAR), dst=frame)
        if self.flash > 0.01:
            cv2.addWeighted(frame, 1.0 - min(1.0, self.flash), self._white, min(1.0, self.flash), 0, dst=frame)
        return frame


class ParticleSystem:
    """Fixed-capacity, vectorised particles (no per-frame allocation of objects)."""

    def __init__(self, capacity: int = 900):
        self.resize(capacity)

    def resize(self, capacity: int) -> None:
        self.capacity = capacity
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)  # remaining seconds
        self.max_life = np.ones(capacity, np.float32)
        self.size = np.ones(capacity, np.float32)
        self.color = np.zeros((capacity, 3), np.float32)
        self.drag = np.zeros(capacity, np.float32)
        self.gravity = np.zeros(capacity, np.float32)
        self._cursor = 0

    @property
    def alive_count(self) -> int:
        return int(np.count_nonzero(self.life > 0))

    def emit(self, n: int, pos, vel, spread_pos: float = 0.0, spread_vel: float = 0.0,
             life: Tuple[float, float] = (0.4, 0.9), size: Tuple[float, float] = (1.5, 3.5),
             color: Sequence[float] = (255, 255, 255), drag: float = 1.5, gravity: float = 0.0) -> None:
        """Emit ``n`` particles. ``pos`` / ``vel`` may be a single point or an (n, 2) array."""
        n = int(n)
        if n <= 0:
            return
        n = min(n, self.capacity)
        idx = (self._cursor + np.arange(n)) % self.capacity  # ring buffer; oldest are overwritten
        self._cursor = int((self._cursor + n) % self.capacity)
        p = np.broadcast_to(np.asarray(pos, np.float32), (n, 2))
        v = np.broadcast_to(np.asarray(vel, np.float32), (n, 2))
        self.pos[idx] = p + np.random.normal(0, spread_pos, (n, 2)) if spread_pos else p
        self.vel[idx] = v + np.random.normal(0, spread_vel, (n, 2)) if spread_vel else v
        lf = np.random.uniform(life[0], life[1], n).astype(np.float32)
        self.life[idx] = lf
        self.max_life[idx] = lf
        self.size[idx] = np.random.uniform(size[0], size[1], n)
        self.color[idx] = np.asarray(color, np.float32)
        self.drag[idx] = drag
        self.gravity[idx] = gravity

    def attract(self, target, strength: float, dt: float) -> None:
        """Pull live particles toward ``target`` (used for charging effects)."""
        alive = self.life > 0
        if not alive.any():
            return
        d = np.asarray(target, np.float32) - self.pos[alive]
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1.0
        self.vel[alive] += d / dist * strength * dt

    def update(self, dt: float) -> None:
        alive = self.life > 0
        if not alive.any():
            return
        self.life[alive] -= dt
        self.vel[alive, 1] += self.gravity[alive] * dt
        self.vel[alive] *= np.exp(-self.drag[alive] * dt)[:, None]
        self.pos[alive] += self.vel[alive] * dt

    def clear(self) -> None:
        self.life.fill(0)

    def render(self, canvas: Canvas, max_cores: int = 400) -> None:
        alive = np.nonzero(self.life > 0)[0]
        if alive.size == 0:
            return
        fade = np.clip(self.life[alive] / self.max_life[alive], 0.0, 1.0)
        col = self.color[alive] * fade[:, None] * canvas.intensity
        # Soft halo: splat into the low-res glow layer (vectorised).
        gp = (self.pos[alive] * canvas.gs).astype(np.int32)
        ok = (gp[:, 0] >= 0) & (gp[:, 0] < canvas.gw) & (gp[:, 1] >= 0) & (gp[:, 1] < canvas.gh)
        if ok.any():
            np.add.at(canvas.glow, (gp[ok, 1], gp[ok, 0]), col[ok] * (self.size[alive][ok, None] * 0.9))
        # Crisp cores for the brightest/biggest particles.
        if alive.size > max_cores:
            order = np.argsort(-(fade * self.size[alive]))[:max_cores]
        else:
            order = np.arange(alive.size)
        light = canvas.light
        for j in order:
            i = alive[j]
            x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
            if 0 <= x < canvas.w and 0 <= y < canvas.h:
                r = max(1, int(self.size[i] * (0.5 + 0.5 * fade[j])))
                c = col[j]
                cv2.circle(light, (x, y), r, (float(c[0]), float(c[1]), float(c[2])), -1, cv2.LINE_AA)


class Trail:
    """Ribbon between two moving points (e.g. sword base and tip) over time."""

    def __init__(self, duration: float = 0.22, max_samples: int = 24):
        self.duration = duration
        self.samples: deque = deque(maxlen=max_samples)

    def add(self, t: float, a, b) -> None:
        self.samples.append((t, np.asarray(a, np.float32).copy(), np.asarray(b, np.float32).copy()))

    def clear(self) -> None:
        self.samples.clear()

    def prune(self, now: float) -> None:
        while self.samples and now - self.samples[0][0] > self.duration:
            self.samples.popleft()

    def render(self, canvas: Canvas, now: float, color: Sequence[float], strength: float = 1.0,
               min_speed_px: float = 6.0) -> None:
        self.prune(now)
        if len(self.samples) < 2:
            return
        prev = None
        for t, a, b in self.samples:
            if prev is not None:
                pa, pb = prev
                if np.linalg.norm(b - pb) >= min_speed_px or np.linalg.norm(a - pa) >= min_speed_px:
                    age = 1.0 - (now - t) / self.duration
                    quad = np.array([pa, pb, b, a], np.float32)
                    canvas.light_poly(quad, color, strength * 0.55 * age * age)
                    canvas.glow_poly(quad, color, strength * 0.9 * age)
            prev = (a, b)


class Shaker:
    """Decaying screen shake; ``offset`` is applied when blitting the frame."""

    def __init__(self):
        self.amount = 0.0
        self.enabled = True

    def kick(self, pixels: float) -> None:
        if self.enabled:
            self.amount = max(self.amount, pixels)

    def update(self, dt: float) -> Tuple[int, int]:
        self.amount *= math.exp(-7.0 * dt)
        if self.amount < 0.5:
            self.amount = 0.0
            return 0, 0
        return (int(np.random.uniform(-1, 1) * self.amount), int(np.random.uniform(-1, 1) * self.amount))


def crescent(center, direction, radius: float, thickness: float, span_deg: float = 150,
             n: int = 28) -> np.ndarray:
    """Polygon for a crescent (Getsuga-style) facing ``direction``.

    The outer arc bulges toward ``direction``; the inner arc is offset back so
    the shape tapers to points at both ends.
    """
    d = unit(direction)
    ang = math.atan2(d[1], d[0])
    half = math.radians(span_deg) / 2
    ts = np.linspace(-half, half, n)
    outer = np.stack([np.cos(ang + ts), np.sin(ang + ts)], 1) * radius
    inner_r = radius - thickness * np.cos(ts / half * (math.pi / 2)) ** 0.8
    inner = np.stack([np.cos(ang + ts), np.sin(ang + ts)], 1) * inner_r[:, None]
    inner = inner - d * thickness * 0.35
    pts = np.concatenate([outer, inner[::-1]], 0) + np.asarray(center, np.float32)
    return pts.astype(np.float32)


def jagged_line(a, b, segments: int = 8, jitter: float = 8.0) -> np.ndarray:
    """Lightning-like polyline from a to b."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    ts = np.linspace(0, 1, segments + 1)[:, None]
    pts = a + (b - a) * ts
    n = perp(unit(b - a))
    offs = np.random.uniform(-jitter, jitter, (segments + 1, 1))
    offs[0] = offs[-1] = 0
    return (pts + n * offs).astype(np.float32)


def ring(canvas: Canvas, center, radius: float, color: Sequence[float], thickness: int,
         strength: float = 1.0) -> None:
    canvas.light_circle(center, radius, color, strength, thickness=max(1, thickness))
    cv2.circle(canvas.glow, (int(center[0] * canvas.gs), int(center[1] * canvas.gs)),
               max(1, int(radius * canvas.gs)), scale_color(color, strength * canvas.intensity),
               max(1, int(thickness * canvas.gs * 2)), cv2.LINE_AA)
