"""The Zanpakuto: a 2D sword pose estimated from one hand, and its rendering.

Pose heuristic (an honest 2D approximation, no depth is inferred):

* Grip: the centroid of the four finger knuckles (MCP 5, 9, 13, 17), pulled
  30% of the way toward the wrist, i.e. the channel inside a closed fist.
* Direction for a fist: a held blade runs through that channel and exits on
  the thumb side, which is the pinky-MCP -> index-MCP axis (17 -> 5). Index
  is always the thumb side, so the axis never needs an "up" flip that would
  make the blade jump 180 degrees when crossing horizontal. A neutral wrist
  holds a sword tilted a little forward of the knuckle line, so 30% of the
  wrist -> middle-MCP vector is mixed in. When the fist is seen edge-on the
  knuckle span collapses in 2D and becomes noisy, so the direction blends
  toward wrist -> middle-MCP as the span shrinks.
* Direction for any other hand (and in "hand" mode): wrist -> middle MCP.
* The unit direction is smoothed with an EMA whose retention drops when the
  hand rotates fast, so slashes stay crisp and a still hand stays steady.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import fx
from .abilities import Context, SwordPose
from .config import PROJECT_ROOT
from .gestures import classify_hand, hand_size
from .tracking import Part

MCPS = (5, 9, 13, 17)
FINGERTIPS = (4, 8, 12, 16, 20)
# Share of the wrist -> middle-MCP vector mixed into the fist's knuckle axis.
FORWARD_TILT = 0.3


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
        self.image: Optional[np.ndarray] = self._load_image(cfg.get("assets", {}).get("sword_image", ""))
        self.reset()

    # --- setup --------------------------------------------------------------
    @staticmethod
    def _load_image(raw: str) -> Optional[np.ndarray]:
        if not raw:
            return None
        path = raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
        if img is None or img.ndim != 3 or img.shape[2] not in (3, 4):
            print(f"[sword] could not read sword image {path!r}; using the procedural blade.")
            return None
        if img.shape[2] == 3:
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        return img

    def reset(self) -> None:
        self.hand: Optional[str] = None
        self.direction = np.array([0.0, -1.0], np.float32)
        self.length = 0.0
        self.prev_tip: Optional[np.ndarray] = None
        self.vel = np.zeros(2, np.float32)
        self.armed = True
        self.trail.clear()

    # --- pose ---------------------------------------------------------------
    def _choose_hand(self, ctx: Context) -> Tuple[Optional[str], bool]:
        """Returns (hand key, is_fist). Sticky: keeps the current hand when it still qualifies."""
        hands = ctx.tracking.hands
        visible = [s for s in ("right", "left") if hands.get(s) is not None and hands[s].visible
                   and hands[s].points is not None]
        fists = [s for s in visible if classify_hand(hands[s].points, self.extend_ratio) == "fist"]
        if fists:
            side = self.hand if self.hand in fists else fists[0]
            return side, True
        if visible:
            side = self.hand if self.hand in visible else visible[0]
            return side, False
        # Nothing visible: keep using the held (fading) landmarks of the current hand.
        if self.hand and hands.get(self.hand) is not None and hands[self.hand].ok:
            return self.hand, classify_hand(hands[self.hand].points, self.extend_ratio) == "fist"
        held = [s for s in ("right", "left") if hands.get(s) is not None and hands[s].ok]
        if held:
            return held[0], classify_hand(hands[held[0]].points, self.extend_ratio) == "fist"
        return None, False

    def _raw_direction(self, pts: np.ndarray, is_fist: bool, hsize: float, hand_mode: bool) -> np.ndarray:
        fwd = fx.unit(pts[9, :2] - pts[0, :2])
        if hand_mode or not is_fist:
            return fwd
        axis = pts[5, :2] - pts[17, :2]
        span = float(np.linalg.norm(axis)) / max(hsize, 1e-3)
        # Synthetic/real fists have a knuckle span of ~0.6-0.7 hand sizes when seen face-on.
        w = float(np.clip((span - 0.2) / 0.35, 0.0, 1.0))
        return fx.unit(w * fx.unit(axis) + FORWARD_TILT * fwd + (1.0 - w) * fwd)

    def update(self, ctx: Context) -> None:
        pose: SwordPose = ctx.sword
        side, is_fist = self._choose_hand(ctx)
        part: Optional[Part] = ctx.tracking.hands.get(side) if side else None
        if part is None or not part.ok:
            pose.visible = False
            pose.confidence = max(0.0, pose.confidence - ctx.dt * 5.0)
            pose.speed = 0.0
            self.prev_tip = None
            self.vel[:] = 0
            self.hand = None
            return
        if side != self.hand:
            self.prev_tip = None
        self.hand = side
        pts = part.points
        hand_mode = ctx.sword_mode == "hand"
        hsize = max(1.0, hand_size(pts))
        diag = ctx.diag

        knuckles = pts[list(MCPS), :2].mean(axis=0)
        if hand_mode:
            grip = knuckles
        else:
            grip = knuckles + (pts[0, :2] - knuckles) * 0.3

        raw = self._raw_direction(pts, is_fist, hsize, hand_mode)
        if self.prev_tip is None:
            self.direction = raw
        else:
            cosang = float(np.clip(np.dot(raw, self.direction), -1.0, 1.0))
            ang = math.degrees(math.acos(cosang))
            # 0.6 retention per 1/30 s when still, down to ~0.1 when rotating >30 degrees/frame.
            keep = 0.6 * (1.0 - min(1.0, ang / 30.0)) + 0.1 * min(1.0, ang / 30.0)
            keep = keep ** max(0.25, ctx.dt * 30.0)
            self.direction = fx.unit(self.direction * keep + raw * (1.0 - keep))

        if hand_mode:
            target_len = float(np.clip(hsize * 2.6, 0.08 * diag, 0.3 * diag))
        else:
            target_len = float(np.clip(hsize * 5.5, 0.18 * diag, 0.55 * diag))
        self.length = target_len if self.length <= 0 or self.prev_tip is None else (
            self.length * 0.7 + target_len * 0.3)
        tip = grip + self.direction * self.length

        if self.prev_tip is not None and ctx.dt > 1e-4:
            v = (tip - self.prev_tip) / ctx.dt
            self.vel = self.vel * 0.35 + v.astype(np.float32) * 0.65
        else:
            self.vel[:] = 0
        self.prev_tip = tip.astype(np.float32)

        pose.visible = bool(part.visible)
        pose.hand = side
        pose.grip = grip.astype(np.float32)
        pose.tip = tip.astype(np.float32)
        pose.direction = self.direction.astype(np.float32)
        pose.length = self.length
        spd = float(np.linalg.norm(self.vel))
        pose.speed = spd / diag
        if spd > 1e-3:
            pose.motion = (self.vel / spd).astype(np.float32)
        pose.confidence = float(part.confidence)

        base = grip + (tip - grip) * (0.6 if hand_mode else 0.62)
        self.trail.add(ctx.now, base, tip)

        # Edge-triggered swing sparks: fire once on crossing, re-arm below 60%.
        if self.armed and pose.speed >= self.spark_speed and pose.confidence > 0.3:
            self.armed = False
            self._clash(ctx, pose)
        elif not self.armed and pose.speed < self.spark_speed * 0.6:
            self.armed = True

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
        if a < 0.02 or pose.length <= 1.0:
            return
        col = ctx.energy_color()
        boost = 1.0 + max(0.0, ctx.aura_boost)
        canvas = ctx.canvas
        swing = float(np.clip((pose.speed - 0.5) / 1.5, 0.0, 1.0))
        if swing > 0.0:
            self._render_trail(ctx, a * (0.2 + 0.5 * swing), col)
        else:
            self.trail.prune(ctx.now)

        if ctx.sword_mode == "hand":
            self._render_hand(ctx, a, col, boost)
            return
        style = ctx.current_weapon()
        if self.image is not None:
            self._render_image(ctx, pose, a, col, boost)
        elif style == "cleaver":
            self._render_cleaver(ctx, pose, a, col, boost)
        elif style == "dark":
            self._render_dark(ctx, pose, a, col, boost)
        else:
            self._render_katana(ctx, pose, a, col, boost)

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

    @staticmethod
    def _xf(pose: SwordPose, uv: np.ndarray) -> np.ndarray:
        """Blade-local (u along blade, v across) in units of length -> render pixels."""
        d = pose.direction
        n = fx.perp(d)
        L = pose.length
        uv = np.asarray(uv, np.float32)
        return (pose.grip[None, :] + uv[:, :1] * L * d[None, :] + uv[:, 1:2] * L * n[None, :]).astype(np.float32)

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

    @staticmethod
    def _wrap_handle(pose: SwordPose, u0: float, u1: float, half_w: float, n_diamonds: int) -> List[np.ndarray]:
        out = []
        step = (u1 - u0) / n_diamonds
        for k in range(n_diamonds):
            uc = u0 + step * (k + 0.5)
            out.append(Zanpakuto._xf(pose, np.array([[uc - step * 0.42, 0], [uc, half_w * 0.7],
                                                     [uc + step * 0.42, 0], [uc, -half_w * 0.7]])))
        return out

    # Katana --------------------------------------------------------------
    def _render_katana(self, ctx: Context, pose: SwordPose, a: float, col, boost: float) -> None:
        s = np.linspace(0.0, 1.0, 18)
        u = 0.11 + 0.89 * s
        curve = 0.035 * s ** 2
        half = 0.017 - 0.005 * s
        back = np.stack([u, curve + half], 1)
        edge = np.stack([u, curve - half], 1)
        # Kissaki: the edge sweeps up to meet the back at the tip.
        k = s > 0.9
        edge[k, 1] = (curve - half + 2 * half * np.clip((s - 0.9) / 0.1, 0.0, 1.0) ** 2.2)[k]
        tip = np.array([[1.0, curve[-1] + half[-1] * 0.6]])
        blade_uv = np.concatenate([back, tip, edge[::-1]], 0)
        shinogi_uv = np.concatenate([back[:-2], (np.stack([u, curve + half * 0.2], 1))[:-2][::-1]], 0)
        hamon_v = curve - half + 2 * half * (0.32 + 0.12 * np.sin(s * 37.0) * (1 - s))
        hamon_uv = np.stack([u, hamon_v], 1)[:-2]

        blade = self._xf(pose, blade_uv)
        shinogi = self._xf(pose, shinogi_uv)
        edge_px = self._xf(pose, edge[:-1])
        hamon = self._xf(pose, hamon_uv)
        handle = self._xf(pose, np.array([[-0.18, -0.02], [0.085, -0.019], [0.085, 0.019], [-0.18, 0.02]]))
        pommel = self._xf(pose, np.array([[-0.2, -0.021], [-0.175, -0.022], [-0.175, 0.022], [-0.2, 0.021]]))
        habaki = self._xf(pose, np.array([[0.1, -0.017], [0.125, -0.016], [0.125, 0.019], [0.1, 0.019]]))
        diamonds = self._wrap_handle(pose, -0.17, 0.075, 0.02, 7)
        L = pose.length
        n = fx.perp(pose.direction)
        tsuba_c = self._xf(pose, np.array([[0.093, 0.0]]))[0]
        ang = math.degrees(math.atan2(n[1], n[0]))
        axes = (max(2, int(L * 0.052)), max(1, int(L * 0.014)))

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
    def _render_cleaver(self, ctx: Context, pose: SwordPose, a: float, col, boost: float) -> None:
        back_v, edge_v = 0.06, -0.115
        blade_uv = np.array([[0.07, back_v], [0.985, back_v], [1.0, back_v - 0.02],
                             [0.9, edge_v * 0.55], [0.8, edge_v], [0.07, edge_v]])
        bevel_uv = np.array([[0.07, edge_v + 0.035], [0.8, edge_v + 0.035], [0.885, edge_v * 0.25],
                             [0.9, edge_v * 0.55], [0.8, edge_v], [0.07, edge_v]])
        fuller_uv = np.array([[0.1, back_v - 0.018], [0.9, back_v - 0.018]])
        edge_uv = np.array([[0.07, edge_v], [0.8, edge_v], [0.9, edge_v * 0.55], [1.0, back_v - 0.02]])
        blade = self._xf(pose, blade_uv)
        bevel = self._xf(pose, bevel_uv)
        fuller = self._xf(pose, fuller_uv)
        edge_px = self._xf(pose, edge_uv)
        edge_fine = np.concatenate([np.linspace(edge_px[i], edge_px[i + 1], 6)[:-1] for i in range(3)]
                                   + [edge_px[-1:]], 0)
        hilt = self._xf(pose, np.array([[-0.2, -0.022], [0.075, -0.024], [0.075, 0.024], [-0.2, 0.022]]))
        bands = []
        for k in range(8):
            u0 = -0.19 + k * 0.034
            bands.append(self._xf(pose, np.array([[u0, -0.024], [u0 + 0.02, 0.024]])))
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

        def draw(frame: np.ndarray) -> None:
            self._fill(frame, blade, (88, 86, 90))
            self._fill(frame, bevel, (190, 188, 192))
            self._poly(frame, fuller, (60, 58, 62), max(1, int(L * 0.006)))
            self._poly(frame, cloth, (215, 215, 210), max(2, int(L * 0.012)))
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
    def _render_dark(self, ctx: Context, pose: SwordPose, a: float, col, boost: float) -> None:
        s = np.linspace(0.0, 1.0, 10)
        u = 0.12 + 0.86 * s
        half = 0.02 - 0.006 * s
        top = np.stack([u, half], 1)
        bot = np.stack([u, -half], 1)
        blade_uv = np.concatenate([top, [[1.0, 0.004]], bot[::-1]], 0)
        blade = self._xf(pose, blade_uv)
        guard = self._xf(pose, np.array([[0.095, -0.05], [0.105, -0.058], [0.12, -0.05], [0.12, 0.05],
                                         [0.105, 0.058], [0.095, 0.05]]))
        handle = self._xf(pose, np.array([[-0.17, -0.018], [0.095, -0.018], [0.095, 0.018], [-0.17, 0.018]]))
        diamonds = self._wrap_handle(pose, -0.16, 0.085, 0.018, 7)
        L = pose.length
        links = self._xf(pose, np.array([[-0.19, 0.0], [-0.225, 0.008], [-0.26, 0.02]]))
        ring_r = max(2, int(L * 0.013))

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
    def _render_image(self, ctx: Context, pose: SwordPose, a: float, col, boost: float) -> None:
        img = self.image
        ih, iw = img.shape[:2]
        L = pose.length
        d = pose.direction
        ex = fx.perp(d)  # image +x
        ey = -d  # image +y (image "up" is the blade direction)
        sc = 1.2 * L / ih
        base = pose.grip - d * 0.2 * L
        t = base - sc * (ex * iw / 2.0 + ey * ih)
        corners = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], np.float32)
        world = t[None, :] + sc * (corners[:, :1] * ex[None, :] + corners[:, 1:2] * ey[None, :])
        box = self._roi(ctx.canvas, [world], pad=2)
        if box is None:
            return
        x0, y0, x1, y1 = box
        M = np.array([[sc * ex[0], sc * ey[0], t[0] - x0], [sc * ex[1], sc * ey[1], t[1] - y0]], np.float32)
        warped = cv2.warpAffine(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = warped[:, :, 3:4].astype(np.float32) * (a / 255.0)
        roi = ctx.canvas.frame[y0:y1, x0:x1]
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        g = a * boost
        tip = pose.grip + d * L
        ctx.canvas.glow_line(pose.grip + d * 0.15 * L, tip, col, L * 0.06, 0.45 * g)

    # Hand-only mode ---------------------------------------------------------------
    def _render_hand(self, ctx: Context, a: float, col, boost: float) -> None:
        part = ctx.tracking.hands.get(self.hand) if self.hand else None
        pose = ctx.sword
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
