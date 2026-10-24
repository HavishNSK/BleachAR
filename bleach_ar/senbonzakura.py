"""Senbonzakura Kageyoshi: Byakuya Kuchiki's Bankai as a swarm of cherry-blossom petals.

Three pieces:

* ``BladeSwarm`` - a fixed-capacity, vectorised (numpy structure-of-arrays)
  system of sakura petals (in the anime the thousand blades look like petals).
  Each petal has position, velocity, a facing direction, spin, a tumble
  phase, size, colour, glow and its own orbit parameters. The swarm has one behaviour mode at a time:

      SCATTER   free flight after the blade pillars shatter (-> RETURN)
      ORBIT     tilted rings around the body at different radii/heights/speeds
      RETURN    orbit targets with a soft spring that stiffens (-> ORBIT)
      FORMATION a comet-shaped fan gathered behind the commanding hand
      ATTACK    a staggered river of blades streaming to an impact point
      BURST     free flight scattering out from the impact point
      DISPERSE  Bankai released: blades slow, drift and fade, then the swarm is empty

  Steering is a critically damped spring toward a per-blade target (with a
  speed cap) integrated in fixed-size substeps, so motion is frame-rate
  independent. The orbit is an ellipse with a per-blade depth (sin of the
  orbit angle): front blades are bigger/brighter and drawn after the sword,
  back blades are smaller, dimmer, drawn before it, and fade where they pass
  behind the person's segmentation mask.

  Petals are real sakura shapes (narrow at the stem, round lobes either side
  of a notched tip), painted opaquely over the camera image with alpha
  (pastel pink, a deeper pink base and a pale vein) rather than as additive
  light, so they stay pink on bright backgrounds. Each petal tumbles: its
  width follows |cos| of a per-petal flip phase, and its back face is a
  touch darker. A faint pink glow splat and motion streaks for fast petals
  keep the spiritual-energy feel.

* ``PillarField`` - the forest of large blades that rise from the ground
  during the Bankai charge and shatter into the swarm on release.

* ``SenbonzakuraAttack`` - the commanded attack (Getsuga key / sword swing,
  routed by ``AbilityManager`` while in Senbonzakura Bankai):
  CHARGING = formation, ACTIVE = blade river, IMPACT = burst + pink
  shockwave, COOLDOWN = blades return to orbit.

The Bankai activation itself lives in ``transform.Bankai`` (same phases and
locking as the crimson "tensa" style); it drives the swarm and pillars
through ``ctx.swarm``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .abilities import BANKAI, Ability, Context, Phase
from .fx import ease_out, perp, ring, smoothstep, unit

STYLE = "senbonzakura"
CAP = 300  # hard cap on live blades

OFF, SCATTER, ORBIT, RETURN, FORMATION, ATTACK, BURST, DISPERSE = (
    "off", "scatter", "orbit", "return", "formation", "attack", "burst", "disperse")
FREE_MODES = (SCATTER, BURST, DISPERSE)

# Sakura petal outline in local units: x along the petal (-0.5 = stem,
# +0.5 = tip), y across (+-1 = half width at the widest point). Two rounded
# lobes either side of the notch at the tip; not convex, so it is drawn with
# fillPoly.
_HALF = np.array([[-0.50, 0.00], [-0.42, 0.20], [-0.28, 0.46], [-0.10, 0.72], [0.08, 0.90],
                  [0.24, 0.99], [0.36, 0.95], [0.45, 0.78], [0.50, 0.52], [0.48, 0.26]], np.float32)
_NOTCH = np.array([[0.37, 0.0]], np.float32)
_PETAL = np.concatenate([_HALF, _NOTCH, (_HALF * (1, -1))[::-1][:-1]], 0)
# Deeper-pink base of the petal (where it joins the flower).
_BASE = np.array([[-0.50, 0.0], [-0.40, 0.24], [-0.24, 0.40], [-0.12, 0.0], [-0.24, -0.40],
                  [-0.40, -0.24]], np.float32)


def senbon_active(ctx: Context) -> bool:
    return ctx.level == BANKAI and ctx.bankai_style == STYLE


def _finite(p) -> bool:
    return p is not None and bool(np.all(np.isfinite(p)))


def _colors(params: dict) -> np.ndarray:
    return np.array([params.get("color", (190, 120, 255)), params.get("accent_color", (235, 80, 255)),
                     params.get("core_color", (250, 235, 255))], np.float32)


def _petal_colors(params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(light petal, deep base, pale tip) colours, BGR."""
    return (np.array(params.get("petal_color", (196, 168, 255)), np.float32),
            np.array(params.get("petal_base_color", (150, 80, 225)), np.float32),
            np.array(params.get("petal_tip_color", (232, 218, 255)), np.float32))


def measure_anchor(ctx: Context) -> Optional[Tuple[np.ndarray, float]]:
    """Torso centre and body scale (shoulder width, px) from pose, else face, else None."""
    w, h = ctx.canvas.w, ctx.canvas.h
    pose = ctx.tracking.pose
    if pose.ok and pose.points is not None and len(pose.points) > 24:
        pts = pose.points[:, :2]
        ls, rs = pts[11], pts[12]
        if _finite(ls) and _finite(rs):
            sh = (ls + rs) * 0.5
            width = float(np.linalg.norm(ls - rs))
            c = sh
            lh, rh = pts[23], pts[24]
            if _finite(lh) and _finite(rh):
                hip = (lh + rh) * 0.5
                # Chest/stomach: a third of the way from shoulders to hips (hips are
                # often below the frame on a webcam, so don't go all the way).
                if float(np.linalg.norm(hip - sh)) < 3.0 * max(width, 1.0):
                    c = sh + (hip - sh) * 0.35
            if width > 4 and -0.5 * w < c[0] < 1.5 * w and -0.5 * h < c[1] < 1.5 * h:
                return c.astype(np.float32), width
    face = ctx.tracking.face
    if face.ok and face.points is not None and len(face.points) > 454:
        fp = face.points[:, :2]
        fw = float(np.linalg.norm(fp[454] - fp[234])) if _finite(fp[454]) and _finite(fp[234]) else 0.0
        fw = fw if fw > 4 else 0.18 * h
        chin = fp[152]
        if _finite(chin):
            return (chin + np.array([0.0, fw * 1.1], np.float32)).astype(np.float32), fw * 2.2
    return None


class PillarField:
    """Large blades rising from the ground behind/around the user (Bankai charge)."""

    M = 26

    def __init__(self):
        self.x = np.zeros(self.M, np.float32)
        self.height = np.zeros(self.M, np.float32)
        self.hw = np.zeros(self.M, np.float32)
        self.lean = np.zeros(self.M, np.float32)
        self.delay = np.zeros(self.M, np.float32)
        self.front = np.zeros(self.M, bool)
        self.count = 0
        self.ground = 0.0
        self.p = 0.0  # 0..1 rise progress (set by Bankai)
        self.alpha = 0.0
        self.on = False
        self.shattered = False
        self.cx = 0.0

    def clear(self) -> None:
        self.on = False
        self.shattered = False
        self.count = 0
        self.p = 0.0
        self.alpha = 0.0

    def start(self, center: np.ndarray, scale: float, w: int, h: int) -> None:
        cx = float(center[0])
        self.cx = cx
        body = max(0.42 * scale, 0.06 * w)
        xs, front = [], []
        for row, n, lo, hi in ((0, 16, 0.02, 0.98), (1, 10, 0.0, 1.0)):
            base = np.linspace(lo * w, hi * w, n) + np.random.uniform(-0.02, 0.02, n) * w
            keep = np.abs(base - cx) > body * (1.0 if row == 0 else 1.5)
            xs.extend(base[keep].tolist())
            front.extend([row == 1] * int(keep.sum()))
        n = min(self.M, len(xs))
        self.count = n
        self.x[:n] = xs[:n]
        self.front[:n] = front[:n]
        fr = self.front[:n]
        self.height[:n] = np.where(fr, np.random.uniform(0.28, 0.5, n), np.random.uniform(0.5, 0.9, n)) * h
        self.hw[:n] = np.where(fr, np.random.uniform(0.016, 0.022, n), np.random.uniform(0.010, 0.015, n)) * h
        self.lean[:n] = np.random.uniform(-0.03, 0.03, n) * h
        # Rises outward from the body like a wave.
        self.delay[:n] = np.clip(0.55 * np.abs(self.x[:n] - cx) / (0.5 * w) + np.random.uniform(0, 0.1, n), 0, 0.7)
        self.ground = h + 12.0
        self.p = 0.0
        self.alpha = 1.0
        self.on = True
        self.shattered = False

    def rise(self) -> np.ndarray:
        t = np.clip((self.p - self.delay[:self.count]) / 0.3, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def update(self, dt: float) -> None:
        if self.on and self.shattered:
            self.alpha -= dt / 0.14
            if self.alpha <= 0:
                self.clear()

    def sample(self, n: int, diag: float) -> Tuple[np.ndarray, np.ndarray]:
        """Points along the risen blades and outward velocities (for the shatter)."""
        r = self.rise()
        if self.count == 0 or float(r.sum()) <= 1e-3:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        weights = r * self.height[:self.count]
        idx = np.random.choice(self.count, n, p=weights / weights.sum())
        t = np.random.uniform(0.04, 1.0, n).astype(np.float32)
        top = self.ground - self.height[idx] * r[idx]
        y = self.ground + (top - self.ground) * t
        x = self.x[idx] + self.lean[idx] * t + np.random.uniform(-1, 1, n) * self.hw[idx]
        side = np.sign(self.x[idx] - self.cx + 1e-3)
        vx = side * np.random.uniform(0.05, 0.4, n) * diag + np.random.normal(0, 0.12 * diag, n)
        vy = -np.random.uniform(0.05, 0.45, n) * diag
        return np.stack([x, y], 1).astype(np.float32), np.stack([vx, vy], 1).astype(np.float32)

    def render(self, ctx: Context, colors: np.ndarray, inten: float) -> None:
        if not self.on or self.count == 0:
            return
        cv = ctx.canvas
        r = self.rise()
        flick = 0.85 + 0.15 * math.sin(ctx.now * 23.0)
        g = self.ground
        for i in range(self.count):
            if r[i] < 0.01:
                continue
            x, hw, ln = float(self.x[i]), float(self.hw[i]), float(self.lean[i])
            top = g - float(self.height[i]) * float(r[i])
            tip = (x + ln, top)
            poly = np.array([tip, (x + ln + hw, top + 5.5 * hw), (x + hw * 1.1, g), (x - hw * 1.1, g),
                             (x + ln - hw, top + 2.5 * hw)], np.float32)
            k = float(r[i]) * self.alpha * flick * inten * (0.75 if self.front[i] else 0.5)
            col = colors[1] if i % 3 == 0 else colors[0]
            cv2.fillConvexPoly(cv.light, np.round(poly * 16).astype(np.int32),
                               tuple(float(c) * 0.55 * k for c in col), cv2.LINE_AA, 4)
            cv2.line(cv.light, (int(x), int(g)), (int(x + ln * 0.97), int(top + 3 * hw)),
                     tuple(float(c) * 0.85 * k for c in colors[2]), 1 if hw < 9 else 2, cv2.LINE_AA)
            cv.glow_poly(poly, col, 0.55 * k)
        # Glowing ground line the blades rise out of.
        band = np.array([(0, g - 26), (cv.w, g - 26), (cv.w, g + 4), (0, g + 4)], np.float32)
        cv.glow_poly(band, colors[0], 0.6 * self.alpha * min(1.0, self.p * 3.0) * inten)


class BladeSwarm:
    """Fixed-capacity vectorised swarm of small glowing blades (see module docstring)."""

    def __init__(self, params: dict, seed: int = 7):
        self.params = params
        N = CAP
        rng = np.random.default_rng(seed)
        z2 = lambda: np.zeros((N, 2), np.float32)  # noqa: E731
        self.pos, self.vel, self.target = z2(), z2(), z2()
        self.dir = np.tile(np.array([1.0, 0.0], np.float32), (N, 1))
        self.spin = np.zeros(N, np.float32)
        self.life = np.zeros(N, np.float32)
        self.life0 = np.ones(N, np.float32)
        self.alpha = np.zeros(N, np.float32)
        self.depth = np.zeros(N, np.float32)
        self.occ = np.ones(N, np.float32)
        # Static per-blade looks.
        self.length_k = rng.uniform(0.75, 1.25, N).astype(np.float32)
        self.aspect = rng.uniform(0.32, 0.42, N).astype(np.float32)  # half width / length
        self.col_idx = rng.choice(3, N, p=[0.55, 0.27, 0.18]).astype(np.int32)
        self.tint = rng.uniform(0.0, 1.0, N).astype(np.float32)  # 0 pale .. 1 deeper pink
        self.glow_k = rng.uniform(0.6, 1.2, N).astype(np.float32)
        # Tumble: each petal flips around its long axis at its own rate.
        self.flip = rng.uniform(0, 2 * math.pi, N).astype(np.float32)
        self.flip_speed = (rng.uniform(2.5, 7.0, N) * rng.choice([-1.0, 1.0], N)).astype(np.float32)
        self._paint: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        # Orbit parameters: a few tilted ring bands at different heights/radii/
        # speeds, each carrying two ribbons of blades that flow together.
        nb = 5
        band = rng.integers(0, nb, N)
        b_h = np.array([-1.05, -0.62, -0.2, 0.18, 0.5]) + rng.uniform(-0.05, 0.05, nb)
        b_rad = np.array([0.78, 1.05, 1.25, 1.0, 1.15])
        b_tilt = rng.uniform(0.17, 0.3, nb)
        b_roll = np.array([0.22, -0.18, 0.1, -0.26, 0.16]) + rng.uniform(-0.05, 0.05, nb)
        b_speed = np.array([1.15, 0.9, 1.0, 1.2, 0.85])
        ribbon = rng.integers(0, 2, N) * math.pi + rng.uniform(0, 2 * math.pi, nb)[band]
        self.rad_k = (b_rad[band] * rng.uniform(0.94, 1.06, N)).astype(np.float32)
        self.h_k = (b_h[band] + rng.normal(0, 0.035, N)).astype(np.float32)
        self.speed_k = (b_speed[band] * rng.uniform(0.97, 1.03, N)).astype(np.float32)
        self.tilt = (b_tilt[band] * rng.uniform(0.9, 1.1, N)).astype(np.float32)
        self.roll = b_roll[band].astype(np.float32)
        self.phase = (ribbon + rng.uniform(-1.15, 1.15, N)).astype(np.float32)
        self.theta = self.phase.copy()
        # Attack lanes.
        self.lane = rng.uniform(-1, 1, N).astype(np.float32)
        self.rank = rng.random(N).astype(np.float32)
        self.n = 0
        self.mode = OFF
        self.mode_t = 0.0
        self.center = np.zeros(2, np.float32)
        self.scale = 1.0
        self.has_anchor = False
        self.origin = np.zeros(2, np.float32)
        self.aim = np.array([0.0, -1.0], np.float32)
        self.end = np.zeros(2, np.float32)
        self.travel = 0.6
        self.pillars = PillarField()

    # --- queries ----------------------------------------------------------
    @property
    def live_count(self) -> int:
        return int(np.count_nonzero(self.life[:self.n] > 0))

    def _p(self, key: str, default: float) -> float:
        return float(self.params.get(key, default))

    def desired_count(self, ctx: Context) -> int:
        q = float(np.clip(ctx.particles.capacity / 900.0, 0.65, 1.3))
        return int(np.clip(int(self._p("count", 160) * q), 16, CAP))

    # --- control ----------------------------------------------------------
    def clear(self) -> None:
        self.life.fill(0)
        self.alpha.fill(0)
        self.n = 0
        self.mode = OFF
        self.mode_t = 0.0
        self.pillars.clear()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.mode_t = 0.0

    def _spawn(self, n: int, pos: np.ndarray, vel: np.ndarray, alpha: float) -> None:
        n = int(min(CAP, n, len(pos)))
        self.n = n
        self.pos[:n] = pos[:n]
        self.vel[:n] = vel[:n]
        self.life[:n] = 1.0
        self.life[n:] = 0.0
        self.life0[:n] = 1.0
        self.alpha[:n] = alpha
        self.occ[:n] = 1.0
        self.spin[:n] = np.random.uniform(-12, 12, n)
        sp = np.linalg.norm(self.vel[:n], axis=1, keepdims=True)
        self.dir[:n] = np.where(sp > 1e-3, self.vel[:n] / np.maximum(sp, 1e-3), np.array([0.0, -1.0], np.float32))
        self.theta[:n] = self.phase[:n]

    def spawn_orbit(self, ctx: Context) -> None:
        """Appear directly on the orbit (e.g. Bankai level set without the activation)."""
        self._anchor(ctx, snap=not self.has_anchor)
        n = self.desired_count(ctx)
        self.n = n
        self.theta[:n] = self.phase[:n]
        tgt, _, _ = self._orbit_targets(ctx, n)
        self._spawn(n, tgt, np.zeros((n, 2), np.float32), 0.0)
        self._set_mode(ORBIT)

    def spawn_from(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Blades born from the shattered pillars: scatter, then return to orbit."""
        self._spawn(len(pos), pos, vel, 1.0)
        self._set_mode(SCATTER)

    def formation(self, origin, aim) -> None:
        self.set_aim(origin, aim)
        if self.mode != FORMATION:
            self._set_mode(FORMATION)

    def set_aim(self, origin, aim) -> None:
        if _finite(origin):
            self.origin = np.asarray(origin, np.float32)
        if _finite(aim) and float(np.linalg.norm(aim)) > 1e-3:
            self.aim = unit(aim)

    def attack(self, origin, aim, end, travel: float) -> None:
        self.set_aim(origin, aim)
        if _finite(end):
            self.end = np.asarray(end, np.float32)
        self.travel = max(0.2, float(travel))
        self._set_mode(ATTACK)

    def burst(self, ctx: Context) -> None:
        n = self.n
        if n == 0:
            return
        d = ctx.diag
        ang = np.random.uniform(0, 2 * math.pi, n)
        sp = np.random.uniform(0.25, 0.8, n) * d
        self.vel[:n] = self.vel[:n] * 0.15 + np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None]
        self.spin[:n] = np.random.uniform(-16, 16, n)
        self._set_mode(BURST)

    def return_to_orbit(self) -> None:
        if self.n and self.mode not in (OFF, DISPERSE, ORBIT, RETURN):
            self._set_mode(RETURN)

    def disperse(self, ctx: Context) -> None:
        n = self.n
        if n == 0 or self.mode in (OFF, DISPERSE):
            self.pillars.clear()
            return
        d = ctx.diag
        lf = np.random.uniform(0.45, 0.95, n).astype(np.float32)
        self.life[:n] = np.where(self.life[:n] > 0, lf, 0.0)
        self.life0[:n] = lf
        self.vel[:n] = self.vel[:n] * 0.3 + np.random.normal(0, 0.1 * d, (n, 2)) + np.array([0, -0.06 * d], np.float32)
        self.spin[:n] = np.random.uniform(-9, 9, n)
        self.pillars.clear()
        self._set_mode(DISPERSE)

    # --- anchoring --------------------------------------------------------
    def _anchor(self, ctx: Context, snap: bool = False) -> None:
        m = measure_anchor(ctx)
        cv = ctx.canvas
        if m is None:
            if not self.has_anchor:  # never seen anyone: screen centre
                self.center = np.array([cv.w * 0.5, cv.h * 0.55], np.float32)
                self.scale = 0.5 * cv.h
                self.has_anchor = True
            return  # hold the last anchor while tracking is lost
        c, s = m
        if snap or not self.has_anchor:
            self.center, self.scale = c, s
            self.has_anchor = True
            return
        a = 1.0 - math.exp(-8.0 * max(0.0, ctx.dt))
        self.center = (self.center + (c - self.center) * a).astype(np.float32)
        self.scale += (s - self.scale) * a

    # --- targets ----------------------------------------------------------
    def _orbit_targets(self, ctx: Context, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = ctx.diag
        r0 = float(np.clip(self.scale * 0.95, 0.10 * d, self._p("orbit_radius", 0.30) * d))
        th = self.theta[:n]
        R = r0 * self.rad_k[:n]
        ex = R * np.cos(th)
        ey = R * self.tilt[:n] * np.sin(th)
        cr, sr = np.cos(self.roll[:n]), np.sin(self.roll[:n])
        bob = 6.0 * np.sin(ctx.now * 1.3 + self.phase[:n])
        x = self.center[0] + ex * cr - ey * sr
        y = self.center[1] + self.h_k[:n] * self.scale * 0.85 + ex * sr + ey * cr + bob
        tgt = np.stack([x, y], 1).astype(np.float32)
        # Tangent (direction of travel) for the blade facing.
        tx, ty = -R * np.sin(th), R * self.tilt[:n] * np.cos(th)
        tan = np.stack([tx * cr - ty * sr, tx * sr + ty * cr], 1)
        return tgt, tan.astype(np.float32), np.sin(th).astype(np.float32)

    def _formation_targets(self, ctx: Context, n: int) -> np.ndarray:
        d = ctx.diag
        a, nrm = self.aim, perp(self.aim)
        s = self.rank[:n]
        swirl = np.sin(self.mode_t * 7.0 + self.phase[:n]) * 0.012 * d
        back = 0.04 * d + s * 0.24 * d
        lat = self.lane[:n] * (0.015 + 0.08 * s) * d + swirl
        return (self.origin - a * back[:, None] + nrm * lat[:, None]).astype(np.float32)

    def _attack_targets(self, ctx: Context, n: int) -> np.ndarray:
        d = ctx.diag
        a, nrm = self.aim, perp(self.aim)
        delay = self.rank[:n] * 0.32
        te = self.mode_t - delay
        s = np.clip(te / self.travel, 0.0, 1.0)
        es = 1.0 - (1.0 - s) ** 2
        dist = float(np.linalg.norm(self.end - self.origin))
        spread = (0.012 + 0.06 * np.sin(np.pi * s)) * d  # fan opens mid-flight, closes at the target
        wave = np.sin(s * 8.0 - self.mode_t * 10.0 + self.phase[:n]) * 0.014 * d * s
        lat = self.lane[:n] * spread + wave
        tgt = self.origin + a * (dist * es)[:, None] + nrm * lat[:, None]
        # Not yet launched: hold the formation.
        wait = te <= 0
        if wait.any():
            tgt[wait] = self._formation_targets(ctx, n)[wait]
        # Arrived: swirl tightly around the impact point.
        done = te > self.travel
        if done.any():
            ang = self.phase[:n][done] + self.mode_t * 9.0
            rr = (0.015 + 0.03 * self.rank[:n][done]) * d
            tgt[done] = self.end + np.stack([np.cos(ang), np.sin(ang)], 1) * rr[:, None]
        return tgt.astype(np.float32)

    # --- simulation -------------------------------------------------------
    def update(self, ctx: Context) -> None:
        cv = ctx.canvas
        dt = float(max(0.0, min(0.1, ctx.dt)))
        self.pillars.update(dt)
        if cv.w == 0:
            return
        self._anchor(ctx)
        if self.n == 0 or self.mode == OFF:
            return
        n = self.n
        self.mode_t += dt
        d = ctx.diag
        speed = self._p("orbit_speed", 1.0)
        self.theta[:n] += self.speed_k[:n] * 1.35 * speed * dt
        self.theta[:n] = np.mod(self.theta[:n], 2 * math.pi)
        self.flip[:n] = np.mod(self.flip[:n] + self.flip_speed[:n] * dt, 2 * math.pi)

        pref = None
        depth_tgt = 0.55
        if self.mode == SCATTER and self.mode_t > 0.35:
            self._set_mode(RETURN)
        if self.mode == RETURN and self.mode_t > 1.0:
            self._set_mode(ORBIT)

        if self.mode in (ORBIT, RETURN):
            tgt, tan, depth_tgt = self._orbit_targets(ctx, n)
            w = 6.0 if self.mode == ORBIT else 1.5 + 4.5 * smoothstep(self.mode_t / 1.0)
            self._steer(tgt, w, (1.2 if self.mode == ORBIT else 1.8) * d, dt)
            pref = tan
        elif self.mode == FORMATION:
            self._steer(self._formation_targets(ctx, n), 10.0, 2.2 * d, dt)
            pref = np.broadcast_to(self.aim * 900.0, (n, 2))
        elif self.mode == ATTACK:
            self._steer(self._attack_targets(ctx, n), 19.0, 3.2 * d, dt)
            pref = np.broadcast_to(self.aim * 150.0, (n, 2))
        else:  # free flight
            drag = {SCATTER: 3.0, BURST: 2.6, DISPERSE: 1.6}[self.mode]
            self.vel[:n] *= math.exp(-drag * dt)
            if self.mode == DISPERSE:
                self.vel[:n, 1] -= 0.03 * d * dt  # petals float up as they fade
            self.pos[:n] += self.vel[:n] * dt

        # Facing.
        if self.mode in FREE_MODES:
            ang = self.spin[:n] * dt
            c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
            dx, dy = self.dir[:n, :1], self.dir[:n, 1:]
            self.dir[:n] = np.hstack([dx * c - dy * s, dx * s + dy * c])
            self.spin[:n] *= math.exp(-1.2 * dt)
        else:
            want = self.vel[:n] + (pref if pref is not None else 0.0)
            wn = np.linalg.norm(want, axis=1, keepdims=True)
            ok = wn[:, 0] > 1e-3
            k = 1.0 - math.exp(-12.0 * dt)
            nd = self.dir[:n].copy()
            nd[ok] += (want[ok] / wn[ok] - nd[ok]) * k
            nn = np.linalg.norm(nd, axis=1, keepdims=True)
            self.dir[:n] = np.where(nn > 1e-4, nd / np.maximum(nn, 1e-4), self.dir[:n])

        # Depth, opacity, occlusion behind the person.
        self.depth[:n] += (depth_tgt - self.depth[:n]) * (1.0 - math.exp(-8.0 * dt))
        if self.mode == DISPERSE:
            self.life[:n] -= dt
            self.alpha[:n] = np.clip(self.life[:n] / self.life0[:n], 0, 1)
            if not (self.life[:n] > 0).any():
                self.clear()
                return
        else:
            self.alpha[:n] = np.minimum(1.0, self.alpha[:n] + 3.0 * dt)
        occ_t = np.ones(n, np.float32)
        mask = ctx.tracking.mask
        if mask is not None and mask.ndim == 2 and mask.size:
            mh, mw = mask.shape
            mx = np.clip((self.pos[:n, 0] * mw / cv.w).astype(np.int32), 0, mw - 1)
            my = np.clip((self.pos[:n, 1] * mh / cv.h).astype(np.int32), 0, mh - 1)
            behind = (self.depth[:n] < -0.1) & (mask[my, mx] > 0.5)
            occ_t[behind] = 0.12
        self.occ[:n] += (occ_t - self.occ[:n]) * (1.0 - math.exp(-10.0 * dt))
        self._sanitize(cv.w, cv.h)

    def _steer(self, tgt: np.ndarray, w: float, vmax: float, dt: float) -> None:
        n = self.n
        steps = max(1, int(math.ceil(dt / 0.034)))
        h = dt / steps
        x, v = self.pos[:n], self.vel[:n]
        for _ in range(steps):
            v += (w * w * (tgt - x) - 2.0 * w * v) * h
            sp = np.linalg.norm(v, axis=1)
            over = sp > vmax
            if over.any():
                v[over] *= (vmax / sp[over])[:, None]
            x += v * h

    def _sanitize(self, w: int, h: int) -> None:
        n = self.n
        for arr in (self.pos, self.vel):
            bad = ~np.isfinite(arr[:n]).all(axis=1)
            if bad.any():
                arr[:n][bad] = self.center if arr is self.pos else 0.0
        np.clip(self.pos[:n, 0], -w, 2 * w, out=self.pos[:n, 0])
        np.clip(self.pos[:n, 1], -h, 2 * h, out=self.pos[:n, 1])
        bad = ~np.isfinite(self.dir[:n]).all(axis=1)
        if bad.any():
            self.dir[:n][bad] = (0.0, -1.0)

    # --- rendering --------------------------------------------------------
    def render(self, ctx: Context, layer: str = "front") -> None:
        """``layer`` "back" draws the pillars and blades behind the body, "front" the rest."""
        colors = _colors(self.params)
        inten = self._p("intensity", 1.0) * ctx.canvas.intensity
        if layer == "back":
            self.pillars.render(ctx, colors, inten)
        n = self.n
        if n == 0 or self.mode == OFF:
            return
        cv = ctx.canvas
        depth = self.depth[:n]
        sel = (self.life[:n] > 0) & ((depth < 0) if layer == "back" else (depth >= 0))
        ids = np.nonzero(sel)[0]
        if ids.size == 0:
            return
        ids = ids[np.argsort(depth[ids])]
        dk = (depth[ids] + 1.0) * 0.5  # 0 = far side of the ring, 1 = near side
        bright = self.alpha[ids] * self.occ[ids] * (0.35 + 0.65 * dk) * inten
        size = (0.62 + 0.55 * dk) * self.length_k[ids] * (0.026 * ctx.diag * self._p("blade_size", 1.0))
        c = self.pos[ids]
        on = (bright > 0.02) & (c[:, 0] > -size) & (c[:, 0] < cv.w + size) & (c[:, 1] > -size) & (c[:, 1] < cv.h + size)
        if not on.any():
            return
        ids, bright, size, c = ids[on], bright[on], size[on], c[on]
        self._render_petals(ctx, ids, bright, size, c, inten)

    def _render_petals(self, ctx: Context, ids: np.ndarray, bright: np.ndarray, size: np.ndarray,
                       c: np.ndarray, inten: float) -> None:
        cv = ctx.canvas
        frame = cv.frame
        light_col, base_col, tip_col = _petal_colors(self.params)
        d = self.dir[ids]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        flip = np.cos(self.flip[ids])
        hw = np.maximum(size * self.aspect[ids] * (0.22 + 0.78 * np.abs(flip)), 0.8)
        P = (c[:, None, :] + (_PETAL[None, :, :1] * size[:, None, None]) * d[:, None, :]
             + (_PETAL[None, :, 1:] * hw[:, None, None]) * nrm[:, None, :])
        B = (c[:, None, :] + (_BASE[None, :, :1] * size[:, None, None]) * d[:, None, :]
             + (_BASE[None, :, 1:] * hw[:, None, None]) * nrm[:, None, :])
        vein_a = c - d * (0.40 * size)[:, None]
        vein_b = c + d * (0.02 * size)[:, None]
        # Colour: pale..deeper pink per petal, back face slightly darker.
        t = self.tint[ids][:, None]
        shade = np.where(flip < 0, 0.86, 1.0)[:, None]
        body = (light_col * (1.0 - 0.45 * t) + base_col * (0.45 * t)) * shade
        base = base_col * shade
        vein = tip_col
        opacity = np.clip(bright / max(inten, 1e-3), 0.0, 1.0) * 0.95

        if frame is not None:
            h, w = frame.shape[:2]
            if self._paint is None or self._paint.shape[:2] != (h, w):
                self._paint = np.zeros((h, w, 3), np.uint8)
                self._alpha = np.zeros((h, w), np.uint8)
            paint, amask = self._paint, self._alpha
            x0 = int(max(0, np.floor((c[:, 0] - size).min())))
            y0 = int(max(0, np.floor((c[:, 1] - size).min())))
            x1 = int(min(w, np.ceil((c[:, 0] + size).max()) + 1))
            y1 = int(min(h, np.ceil((c[:, 1] + size).max()) + 1))
            if x1 > x0 and y1 > y0:
                paint[y0:y1, x0:x1] = 0
                amask[y0:y1, x0:x1] = 0
                Pi = np.round(P * 16).astype(np.int32)
                Bi = np.round(B * 16).astype(np.int32)
                av = (opacity * 255).astype(np.int32).tolist()
                bl, bs, vc = body.tolist(), base.tolist(), vein.tolist()
                va = np.round(vein_a).astype(np.int32)
                vb = np.round(vein_b).astype(np.int32)
                for j in range(len(ids)):  # back to front (ids are depth-sorted)
                    cv2.fillPoly(paint, [Pi[j]], bl[j], cv2.LINE_AA, 4)
                    cv2.fillPoly(amask, [Pi[j]], av[j], cv2.LINE_AA, 4)
                    if size[j] > 7:
                        cv2.fillConvexPoly(paint, Bi[j], bs[j], cv2.LINE_AA, 4)
                        cv2.line(paint, tuple(va[j]), tuple(vb[j]), vc, 1, cv2.LINE_AA)
                # Blend only the petal pixels into the camera image.
                roi_a = amask[y0:y1, x0:x1]
                ys, xs = np.nonzero(roi_a)
                if ys.size:
                    a = roi_a[ys, xs].astype(np.float32)[:, None] * (1.0 / 255.0)
                    f = frame[y0:y1, x0:x1]
                    src = f[ys, xs].astype(np.float32)
                    f[ys, xs] = (src + (paint[y0:y1, x0:x1][ys, xs].astype(np.float32) - src) * a).astype(np.uint8)

        # Spiritual-energy hints: a faint pink halo per petal and streaks
        # behind fast petals (additive layers).
        colors = _colors(self.params)
        gp = (c * cv.gs).astype(np.int32)
        ok = (gp[:, 0] >= 0) & (gp[:, 0] < cv.gw) & (gp[:, 1] >= 0) & (gp[:, 1] < cv.gh)
        if ok.any():
            gcol = colors[self.col_idx[ids]] * (bright * self.glow_k[ids] * size * 0.07)[:, None]
            np.add.at(cv.glow, (gp[ok, 1], gp[ok, 0]), gcol[ok])
        vel = self.vel[ids]
        sp = np.linalg.norm(vel, axis=1) / ctx.diag
        fast = np.nonzero(sp > 0.45)[0]
        if fast.size:
            tails = c[fast] - vel[fast] * 0.018
            k = np.clip((sp[fast] - 0.45) * 1.5, 0.0, 1.0) * bright[fast]
            for j, a, b in zip(range(fast.size), tails, c[fast]):
                cv2.line(cv.light, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         tuple(float(x) * 0.45 * k[j] for x in colors[0]), 1, cv2.LINE_AA)


class SenbonzakuraAttack(Ability):
    """Commanded attack: formation -> blade river -> burst -> return to orbit."""

    name = "senbon_attack"
    label = "Senbonzakura"

    def __init__(self, params: dict):
        super().__init__(params)
        self.end = np.zeros(2, np.float32)
        self.origin = np.zeros(2, np.float32)
        self.aim = np.array([0.0, -1.0], np.float32)

    def can_trigger(self, ctx: Context) -> bool:
        sw = getattr(ctx, "swarm", None)
        return (super().can_trigger(ctx) and senbon_active(ctx) and sw is not None
                and sw.n > 0 and sw.mode not in (OFF, DISPERSE))

    def reset(self, ctx: Context) -> None:
        super().reset(ctx)
        sw = getattr(ctx, "swarm", None)
        if sw is not None and sw.mode in (FORMATION, ATTACK, BURST):
            sw.return_to_orbit()

    def _aim(self, ctx: Context) -> Tuple[np.ndarray, np.ndarray]:
        sw = ctx.swarm
        s = ctx.sword
        if s.confidence > 0.2 and _finite(s.direction) and _finite(s.grip) and float(np.linalg.norm(s.direction)) > 0.5:
            return np.asarray(s.grip, np.float32), unit(s.direction)
        for side in ("right", "left"):
            p = ctx.tracking.hands.get(side)
            if p is not None and p.ok and p.points is not None and len(p.points) >= 21:
                wr, mcp = p.points[0, :2], p.points[5, :2]
                v = mcp - wr
                if _finite(v) and float(np.linalg.norm(v)) > 4:
                    return np.asarray(mcp, np.float32), unit(v)
        c = sw.center
        return c.copy(), unit(np.array([ctx.canvas.w * 0.5, ctx.canvas.h * 0.12], np.float32) - c)

    def _end_point(self, ctx: Context) -> np.ndarray:
        w, h = ctx.canvas.w, ctx.canvas.h
        o, a = self.origin, self.aim
        ts = []
        for k, lim in ((0, w), (1, h)):
            if abs(a[k]) > 1e-4:
                ts.extend([(0 - o[k]) / a[k], (lim - o[k]) / a[k]])
        ts = [t for t in ts if t > 0]
        dist = min(ts) if ts else 0.3 * ctx.diag
        dist = max(0.25 * ctx.diag, 0.72 * dist)
        e = o + a * dist
        return np.array([np.clip(e[0], 0.1 * w, 0.9 * w), np.clip(e[1], 0.1 * h, 0.9 * h)], np.float32)

    def on_enter(self, phase: Phase, ctx: Context) -> None:
        sw = getattr(ctx, "swarm", None)
        if sw is None:
            return
        col = tuple(self.params.get("color", (190, 120, 255)))
        if phase == Phase.CHARGING:
            self.origin, self.aim = self._aim(ctx)
            sw.formation(self.origin, self.aim)
        elif phase == Phase.ACTIVE:
            self.origin, self.aim = self._aim(ctx)
            self.end = self._end_point(ctx)
            sw.attack(self.origin, self.aim, self.end, max(0.25, self.duration(Phase.ACTIVE) * 0.55))
            ctx.canvas.flash = max(ctx.canvas.flash, 0.08)
            ctx.shaker.kick(6)
        elif phase == Phase.IMPACT:
            sw.burst(ctx)
            d = ctx.diag
            ang = np.random.uniform(0, 2 * math.pi, 110)
            sp = np.random.uniform(0.15, 0.6, 110) * d
            vel = np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None]
            ctx.particles.emit(110, self.end, vel.astype(np.float32), life=(0.35, 0.8), size=(1.5, 3.5),
                               color=col, drag=2.6)
            ctx.particles.emit(40, self.end, vel[:40].astype(np.float32) * 0.6, life=(0.25, 0.5), size=(1.0, 2.2),
                               color=(255, 245, 255), drag=3.0)
            ctx.canvas.flash = max(ctx.canvas.flash, 0.16)
            ctx.shaker.kick(14)
        elif phase == Phase.COOLDOWN:
            sw.return_to_orbit()

    def on_update(self, ctx: Context) -> None:
        sw = getattr(ctx, "swarm", None)
        if sw is None:
            return
        if self.phase == Phase.CHARGING:
            o, a = self._aim(ctx)
            k = 1.0 - math.exp(-10.0 * ctx.dt)
            self.origin = self.origin + (o - self.origin) * k
            self.aim = unit(self.aim + (a - self.aim) * k)
            sw.set_aim(self.origin, self.aim)
            ctx.aura_boost = max(ctx.aura_boost, 0.8 * self.progress)
        elif self.phase == Phase.ACTIVE:
            ctx.aura_boost = max(ctx.aura_boost, 0.8)
            ctx.shaker.kick(3.0)

    def render(self, ctx: Context) -> None:
        if self.phase == Phase.CHARGING:
            col = tuple(self.params.get("color", (190, 120, 255)))
            ctx.canvas.glow_circle(self.origin, 30 + 50 * self.progress, col, 0.5 * self.progress)
        elif self.phase == Phase.IMPACT:
            cv = ctx.canvas
            col = tuple(self.params.get("color", (190, 120, 255)))
            p = self.progress
            e = ease_out(p)
            d = ctx.diag
            ring(cv, self.end, 12 + 0.22 * d * e, col, max(1, int(9 * (1 - p))), 1.1 * (1 - p))
            ring(cv, self.end, 6 + 0.13 * d * e, (255, 245, 255), max(1, int(4 * (1 - p))), 0.9 * (1 - p))
            cv.glow_circle(self.end, 0.06 * d + 0.08 * d * e, col, 0.9 * (1 - smoothstep(p)))
