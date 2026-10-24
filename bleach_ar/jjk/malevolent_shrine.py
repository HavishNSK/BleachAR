"""Domain Expansion: Malevolent Shrine (Sukuna).

An AR domain over the live camera. The user always stays in front: the
whole environment is painted into ``canvas.backdrop()``, which the canvas
composites *behind* the person mask.

Timeline (durations from ``cfg['abilities']['malevolent_shrine']``)::

    CHARGING  a red-black pulse spreads from the user (ripple warp, a
              darkening wave that reveals the domain world, red rim light,
              tint). Title card. Ground cracks start to run outward and the
              shrine begins to materialise (glowing outline, then the solid
              body rises from the ground up) in the last ~40%.
    ACTIVE    the shrine stands behind the user; slashes cut through the
              scene at three depths (far/mid behind the user in the
              backdrop, near ones crisp over everything), embers rise,
              debris drifts, fog rolls along the horizon.
    IMPACT    a final cross-slash, then the domain collapses: a contracting
              ring swallows the world back into the user while the shrine
              sinks and crumbles; backdrop -> 0 and the camera returns.
    COOLDOWN  nothing is drawn.

Cancel: ``trigger()`` while CHARGING or ACTIVE jumps straight to IMPACT (the
collapse plays from whatever state the domain had reached). The domain holds
``ctx.locked_by = "malevolent_shrine"`` from CHARGING through IMPACT so no
other domain (or locked technique) can start over it, and refuses to start
while another ability holds the lock (e.g. Unlimited Void). ``reset()``
returns to IDLE immediately and releases the lock / energy override.

Performance: the static environment (sky, mountains, bone spires, shrine,
skull piles, fog haze, vignette) is rendered once per (resolution, horizon
bucket) into cached images wider than the screen; per frame it is sliced
(horizontal parallax), and only cheap layers are animated (cracks, edge
pulse, slashes, debris, fog in the low-res glow layer, embers).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..abilities import Ability, Context, Phase
from ..fx import ease_out, noise_texture, smoothstep

# Palette (BGR).
RED = (40, 30, 235)
DEEP_RED = (18, 8, 120)
BLOOD = (30, 20, 170)
WHITE_HOT = (225, 225, 255)
EMBER = (40, 110, 255)
ENERGY = (30, 12, 175)  # aura override: red-black
FOG = (34, 18, 92)

FAR, MID, NEAR = 0, 1, 2

# Shared cache of pre-rendered environments: key -> _Scene (small LRU).
_SCENES: Dict[tuple, "_Scene"] = {}


def _lerp(a, b, t):
    return tuple(float(x) + (float(y) - float(x)) * t for x, y in zip(a, b))


def _soft(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Wide Gaussian blur done at 1/8 resolution (float32 0..1 out)."""
    h, w = mask.shape[:2]
    k = 8
    small = cv2.resize(mask.astype(np.float32), (max(2, w // k), max(2, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / k))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _detail(capacity: int) -> int:
    """0 = low (400 particles), 1 = medium (900), 2 = high (1600)."""
    return 0 if capacity < 650 else (1 if capacity < 1250 else 2)


# ---------------------------------------------------------------------------
# Static environment
# ---------------------------------------------------------------------------
class _Scene:
    """Pre-rendered domain world for one resolution / horizon.

    Images are ``W + 2*pad`` wide so the view can be sliced with an offset.
    ``base`` is the world without the shrine, ``full`` with it, ``edges`` the
    glowing contour lines of the shrine (additive), ``alpha`` the shrine's
    coverage (float32) used when it sinks/crumbles.
    """

    def __init__(self, w: int, h: int, yh: int, detail: int):
        self.w, self.h, self.yh, self.detail = w, h, yh, detail
        self.pad = pad = int(w * 0.2)
        W2 = w + 2 * pad
        self.W2 = W2
        self.cx = W2 / 2.0
        self.f = h * 0.9  # focal length for the ground-plane projection
        self.shrine_top = h
        base = self._sky_and_ground(W2, h, yh)
        self._far_layers(base)
        self._spires(base, behind=True)
        full = base.copy()
        alpha = np.zeros((h, W2), np.uint8)
        edges = np.zeros((h, W2, 3), np.uint8)
        self._shrine(full, alpha, edges)
        self._spires(full, behind=False)
        self._spires(base, behind=False)
        # Atmospheric haze over everything near the horizon (cached noise).
        self._haze(base)
        self._haze(full)
        vig = self._vignette(W2, h)
        for img in (base, full):
            cv2.multiply(img, vig, dst=img, dtype=cv2.CV_8U)
        self.base, self.full, self.edges = base, full, edges
        self.alpha = cv2.GaussianBlur(alpha, (3, 3), 0).astype(np.float32) / 255.0
        self.cracks = self._cracks()

    # --- projection helpers ---------------------------------------------
    def ground(self, X: float, Z: float) -> Tuple[float, float]:
        Z = max(Z, 0.6)
        return self.cx + self.f * X / Z, self.yh + self.f * 1.0 / Z

    # --- layers -------------------------------------------------------------
    def _sky_and_ground(self, W2: int, h: int, yh: int) -> np.ndarray:
        img = np.zeros((h, W2, 3), np.uint8)
        ys = np.arange(h, dtype=np.float32)
        col = np.zeros((h, 3), np.float32)
        top, hor, gnd, bot = (np.array(c, np.float32) for c in
                              ((4, 2, 8), (34, 18, 104), (14, 8, 40), (6, 3, 10)))
        up = ys < yh
        t = np.clip(ys[up] / max(1, yh), 0, 1) ** 2.2
        col[up] = top + (hor - top) * t[:, None]
        t = np.clip((ys[~up] - yh) / max(1, h - yh), 0, 1) ** 0.6
        col[~up] = gnd + (bot - gnd) * t[:, None]
        img[:] = col[:, None, :].astype(np.uint8)
        # A dim blood eclipse behind the roofs.
        c = (int(self.cx), int(yh - h * 0.36))
        R = int(h * 0.30)
        # Soft halo, computed only inside its bounding box.
        m = int(R * 1.25 + h * 0.25)
        x0, x1, y0, y1 = max(0, c[0] - m), min(W2, c[0] + m), max(0, c[1] - m), min(h, c[1] + m)
        halo = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.circle(halo, (c[0] - x0, c[1] - y0), int(R * 1.25), 1.0, -1, cv2.LINE_AA)
        halo = _soft(halo, h * 0.08)
        roi = img[y0:y1, x0:x1]
        cv2.add(roi, cv2.merge([halo * 10, halo * 10, halo * 70]).astype(np.uint8), dst=roi)
        cv2.circle(img, c, R, (16, 12, 58), -1, cv2.LINE_AA)
        cv2.circle(img, c, R, (30, 26, 120), 2, cv2.LINE_AA)
        cv2.circle(img, c, int(R * 0.93), (8, 6, 26), -1, cv2.LINE_AA)
        return img

    def _far_layers(self, img: np.ndarray) -> None:
        """Jagged ridges along the horizon: further = hazier/redder."""
        h, W2, yh = self.h, self.W2, self.yh
        rng = np.random.default_rng(11)
        for k, (amp, col) in enumerate((((0.10, (22, 12, 62))), ((0.065, (14, 7, 36))), ((0.035, (9, 5, 20))))):
            n = 18 + 10 * k
            xs = np.linspace(-20, W2 + 20, n)
            ys = yh - h * amp * (0.35 + 0.65 * rng.random(n)) * (1.0 - 0.5 * np.cos(np.linspace(0, 6.28, n)) ** 2)
            ys[::2] += h * amp * 0.25
            pts = np.concatenate([np.stack([xs, ys], 1), [[W2 + 20, yh + 6], [-20, yh + 6]]]).astype(np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA)

    def _spires(self, img: np.ndarray, behind: bool) -> None:
        """Rows of curved bone spires receding toward the vanishing point."""
        n = (5, 7, 9)[self.detail]
        zs = np.geomspace(3.4, 16.0, n)
        for side in (-1, 1):
            for i in range(n - 1, -1, -1):  # far to near
                Z = float(zs[i])
                if (Z > 7.0) != behind:
                    continue
                X = side * (4.4 + 0.25 * math.sin(i * 1.7))
                bx, by = self.ground(X, Z)
                Hs = (2.6 + 0.8 * math.sin(i * 2.3 + side)) * self.f / Z
                thick = 0.34 * self.f / Z
                haze = min(1.0, (Z - 3.0) / 13.0) ** 0.8
                col = _lerp((14, 10, 22), (30, 16, 78), haze)
                rim = _lerp((60, 50, 210), (34, 20, 100), haze)
                u = np.linspace(0, 1, 14)
                cx = bx - side * (u ** 2) * Hs * 0.55  # curve inward like a claw/rib
                cy = by - u * Hs
                wdt = thick * (1 - u) ** 1.3 + 0.6
                left = np.stack([cx - wdt, cy], 1)
                right = np.stack([cx + wdt, cy], 1)
                poly = np.concatenate([left, right[::-1]]).astype(np.int32)
                cv2.fillPoly(img, [poly], tuple(int(c) for c in col), cv2.LINE_AA)
                edge = (left if side > 0 else right).astype(np.int32)
                cv2.polylines(img, [edge], False, tuple(int(c) for c in rim), 1, cv2.LINE_AA)

    def _roof(self, img, alpha, edges, cx, y_eave, half_bottom, half_top, height, lift, col, rim):
        """One roof tier: concave profile with upturned eave tips + underside."""
        n = 24
        u = np.linspace(-1, 1, n)
        # Eave line: flat in the middle, curling up at the tips.
        ex = cx + u * half_bottom
        ey = y_eave - lift * np.abs(u) ** 6
        # Ridge profile from eave tips up to the top ridge (concave sag).
        s = np.linspace(0, 1, 10)
        right_x = cx + half_bottom - (half_bottom - half_top) * (1 - (1 - s) ** 2.2)
        right_y = (y_eave - lift) - (height - lift) * s ** 0.75
        left_x = 2 * cx - right_x
        bottom = np.stack([ex, ey], 1)
        right = np.stack([right_x, right_y], 1)
        poly = np.concatenate([bottom, right[1:], np.stack([left_x[::-1], right_y[::-1]], 1)[:-1]]).astype(np.int32)
        # Underside: a dark band below the eave with rafter ends.
        under = np.concatenate([np.stack([ex, ey], 1), np.stack([ex[::-1] * 0.985 + cx * 0.015,
                                                                 ey[::-1] + height * 0.16], 1)]).astype(np.int32)
        cv2.fillPoly(img, [under], (8, 5, 14), cv2.LINE_AA)
        cv2.fillPoly(alpha, [under], 255)
        for k in range(1, n - 1, 1):
            x0, y0 = ex[k], ey[k] + 2
            vx, vy = (cx - x0) * 0.05, height * 0.14
            cv2.line(img, (int(x0), int(y0)), (int(x0 + vx), int(y0 + vy)), (26, 14, 60), 1, cv2.LINE_AA)
        cv2.fillPoly(img, [poly], col, cv2.LINE_AA)
        cv2.fillPoly(alpha, [poly], 255)
        # Tiles: faint vertical striations converging slightly to the ridge.
        for k in range(-8, 9):
            a = k / 8.0
            p0 = (int(cx + a * half_bottom * 0.95), int(y_eave - lift * abs(a) ** 6 - 3))
            p1 = (int(cx + a * half_top * 0.9), int(y_eave - height + 3))
            cv2.line(img, p0, p1, tuple(int(c * 1.6) for c in col), 1, cv2.LINE_AA)
        cv2.polylines(img, [bottom.astype(np.int32)], False, rim, 2, cv2.LINE_AA)
        cv2.polylines(edges, [bottom.astype(np.int32)], False, (60, 50, 255), 2, cv2.LINE_AA)
        cv2.polylines(edges, [right.astype(np.int32)], False, (40, 30, 200), 1, cv2.LINE_AA)
        cv2.polylines(edges, [np.stack([left_x, right_y], 1).astype(np.int32)], False, (40, 30, 200), 1,
                      cv2.LINE_AA)
        # Horn ornaments on the eave tips.
        for sgn in (-1, 1):
            tx, ty = cx + sgn * half_bottom, y_eave - lift
            t = np.linspace(0, 1, 8)
            hx = tx + sgn * t * height * 0.35
            hy = ty - t ** 1.6 * height * 0.75
            wd = (1 - t) * height * 0.07 + 0.5
            horn = np.concatenate([np.stack([hx - wd, hy], 1), np.stack([hx + wd, hy], 1)[::-1]]).astype(np.int32)
            cv2.fillPoly(img, [horn], (40, 44, 62), cv2.LINE_AA)
            cv2.fillPoly(alpha, [horn], 255)
            cv2.polylines(edges, [np.stack([hx, hy], 1).astype(np.int32)], False, (60, 60, 230), 1, cv2.LINE_AA)
        return float(right_y[-1])

    def _skull(self, img, alpha, c, r, shade):
        bone = tuple(int(v * shade) for v in (70, 80, 104))
        x, y = int(c[0]), int(c[1])
        cv2.ellipse(img, (x, y), (int(r), int(r * 0.9)), 0, 0, 360, bone, -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(x - r * 0.55), y), (int(x + r * 0.55), int(y + r * 1.05)), bone, -1)
        cv2.ellipse(alpha, (x, y), (int(r), int(r * 0.9)), 0, 0, 360, 255, -1)
        hole = (6, 4, 18)
        for s in (-1, 1):
            cv2.ellipse(img, (int(x + s * r * 0.38), int(y + r * 0.1)), (max(1, int(r * 0.26)), max(1, int(r * 0.3))),
                        0, 0, 360, hole, -1, cv2.LINE_AA)
        cv2.line(img, (x, int(y + r * 0.45)), (x, int(y + r * 0.62)), hole, max(1, int(r * 0.12)))
        for k in range(-2, 3):
            xx = int(x + k * r * 0.2)
            cv2.line(img, (xx, int(y + r * 0.8)), (xx, int(y + r * 1.05)), hole, 1)

    def _shrine(self, img, alpha, edges) -> None:
        h, w, cx, yh = self.h, self.w, self.cx, self.yh
        rng = np.random.default_rng(3)
        body = (16, 10, 24)
        pillar = (24, 14, 36)
        rim = (40, 34, 170)
        y_b = yh + h * 0.035  # shrine base (on the ground, far)
        # Platform steps.
        for k, (hw, hh) in enumerate(((0.34, 0.025), (0.31, 0.022), (0.285, 0.02))):
            y1 = y_b - k * h * 0.022
            pts = np.array([[cx - hw * w, y1], [cx + hw * w, y1], [cx + (hw - 0.01) * w, y1 - hh * h],
                            [cx - (hw - 0.01) * w, y1 - hh * h]], np.int32)
            cv2.fillPoly(img, [pts], (12 + 3 * k, 8 + 2 * k, 20 + 5 * k), cv2.LINE_AA)
            cv2.fillPoly(alpha, [pts], 255)
            cv2.line(img, tuple(pts[3]), tuple(pts[2]), (30, 20, 90), 1, cv2.LINE_AA)
        # Stairs widening toward the viewer (perspective).
        st = np.array([[cx - 0.07 * w, y_b], [cx + 0.07 * w, y_b], [cx + 0.2 * w, h], [cx - 0.2 * w, h]], np.int32)
        cv2.fillPoly(img, [st], (14, 9, 26), cv2.LINE_AA)
        for k in range(1, 18):
            Z = 30.0 / (1 + k * 0.45)
            y = yh + self.f / Z
            if y >= h:
                break
            tt = (y - y_b) / max(1.0, h - y_b)
            hw = (0.07 + 0.13 * tt) * w
            cv2.line(img, (int(cx - hw), int(y)), (int(cx + hw), int(y)), (30, 18, 70), 1, cv2.LINE_AA)
        y_floor = y_b - 3 * h * 0.022
        # Main hall.
        hall_h = h * 0.22
        y_top = y_floor - hall_h
        hw = 0.25 * w
        cv2.rectangle(img, (int(cx - hw), int(y_top)), (int(cx + hw), int(y_floor)), body, -1)
        cv2.rectangle(alpha, (int(cx - hw), int(y_top)), (int(cx + hw), int(y_floor)), 255, -1)
        # Pillars.
        npil = 8
        for k in range(npil):
            x = cx - hw + (k + 0.5) * (2 * hw / npil)
            pw = w * 0.012
            cv2.rectangle(img, (int(x - pw), int(y_top)), (int(x + pw), int(y_floor)), pillar, -1)
            cv2.line(img, (int(x - pw), int(y_top)), (int(x - pw), int(y_floor)), (34, 22, 96), 1, cv2.LINE_AA)
            cv2.line(edges, (int(x - pw), int(y_top)), (int(x - pw), int(y_floor)), (30, 24, 150), 1, cv2.LINE_AA)
        # Talisman strips hanging between pillars (abstract, bone-coloured).
        for k in range(npil - 1):
            x = cx - hw + (k + 1) * (2 * hw / npil)
            ln = h * (0.05 + 0.04 * rng.random())
            cv2.line(img, (int(x), int(y_top + 4)), (int(x), int(y_top + ln)), (44, 50, 70), 2)
        # The maw: a dark fanged opening, lit deep red from inside.
        mw, mh = 0.11 * w, hall_h * 0.78
        my0, my1 = y_floor - mh, y_floor
        maw = np.array([[cx - mw, my1], [cx - mw, my0 + mh * 0.3], [cx - mw * 0.6, my0 + mh * 0.06], [cx, my0],
                        [cx + mw * 0.6, my0 + mh * 0.06], [cx + mw, my0 + mh * 0.3], [cx + mw, my1]], np.int32)
        cv2.fillPoly(img, [maw], (8, 4, 30), cv2.LINE_AA)
        glow = np.zeros_like(alpha)
        cv2.ellipse(glow, (int(cx), int(my1)), (int(mw * 0.9), int(mh * 0.75)), 0, 180, 360, 255, -1, cv2.LINE_AA)
        glow = _soft(glow, mw * 0.35) / 255.0
        m = np.zeros_like(alpha)
        cv2.fillPoly(m, [maw], 255)
        g = glow * (m.astype(np.float32) / 255.0)
        img[:] = np.clip(img + g[..., None] * np.array([30, 25, 190], np.float32), 0, 255).astype(np.uint8)
        cv2.polylines(img, [maw], True, rim, 2, cv2.LINE_AA)
        cv2.polylines(edges, [maw], True, (70, 60, 255), 2, cv2.LINE_AA)
        # Fangs: upper ones hang from the arch, lower ones rise from the floor.
        nf = 9
        for k in range(nf):
            u = (k + 0.5) / nf
            x = cx - mw * 0.92 + u * mw * 1.84
            # y of the arch at x
            ax = abs(x - cx) / mw
            ya = my0 + mh * (0.06 + 0.24 * ax ** 2) if ax > 0.6 else my0 + mh * 0.06 * (ax / 0.6) + 2
            ln = mh * (0.16 + 0.12 * (1 - abs(u - 0.5) * 2)) * (0.8 + 0.4 * rng.random())
            fw = mw * 0.085
            tooth = np.array([[x - fw, ya], [x + fw, ya], [x + fw * 0.15, ya + ln]], np.int32)
            cv2.fillPoly(img, [tooth], (66, 76, 100), cv2.LINE_AA)
            cv2.line(img, (int(x - fw), int(ya)), (int(x + fw * 0.15), int(ya + ln)), (40, 44, 70), 1, cv2.LINE_AA)
        for k in range(nf - 1):
            u = (k + 1) / nf
            x = cx - mw * 0.92 + u * mw * 1.84
            ln = mh * (0.10 + 0.08 * rng.random())
            fw = mw * 0.07
            tooth = np.array([[x - fw, my1], [x + fw, my1], [x - fw * 0.1, my1 - ln]], np.int32)
            cv2.fillPoly(img, [tooth], (58, 66, 90), cv2.LINE_AA)
        # Roof tiers (bottom to top).
        y = y_top
        tiers = ((0.43, 0.29, 0.11, 0.05), (0.31, 0.2, 0.085, 0.04), (0.2, 0.11, 0.075, 0.035))
        roof_col = (11, 7, 16)
        for i, (hb, ht, hh, lift) in enumerate(tiers):
            y_ridge = self._roof(img, alpha, edges, cx, y + h * 0.015, hb * w, ht * w, hh * h, lift * h,
                                 roof_col, rim)
            if i < len(tiers) - 1:
                # Upper storey body.
                bw = tiers[i + 1][1] * w * 0.85
                bh = h * 0.05
                cv2.rectangle(img, (int(cx - bw), int(y_ridge - bh)), (int(cx + bw), int(y_ridge + 2)), body, -1)
                cv2.rectangle(alpha, (int(cx - bw), int(y_ridge - bh)), (int(cx + bw), int(y_ridge + 2)), 255, -1)
                for k in range(7):
                    x = cx - bw + (k + 0.5) * (2 * bw / 7)
                    cv2.line(img, (int(x), int(y_ridge - bh)), (int(x), int(y_ridge)), (28, 16, 56), 2)
                # A row of tiny skull-like medallions under the eave.
                for k in range(5):
                    x = cx - bw * 0.8 + k * bw * 0.4
                    self._skull(img, alpha, (x, y_ridge - bh * 0.55), h * 0.009, 0.9)
                y = y_ridge - bh
            else:
                y = y_ridge
        # Finial: a tall spike with bone rings.
        sp = np.array([[cx - w * 0.008, y + 2], [cx + w * 0.008, y + 2], [cx, y - h * 0.11]], np.int32)
        cv2.fillPoly(img, [sp], (30, 30, 48), cv2.LINE_AA)
        cv2.fillPoly(alpha, [sp], 255)
        cv2.line(edges, (int(cx), int(y)), (int(cx), int(y - h * 0.11)), (80, 70, 255), 1, cv2.LINE_AA)
        for k in range(3):
            yy = int(y - h * (0.02 + 0.025 * k))
            cv2.ellipse(img, (int(cx), yy), (int(w * (0.016 - 0.003 * k)), 3), 0, 0, 360, (56, 62, 84), -1,
                        cv2.LINE_AA)
        self.shrine_top = int(y - h * 0.12)
        # Skull piles at the base on both sides of the stairs.
        n = (40, 70, 110)[self.detail]
        for side in (-1, 1):
            for k in range(n):
                u = rng.random()
                x = cx + side * (0.09 * w + u * 0.26 * w)
                mound = (1 - u) ** 0.7 * h * 0.07
                yy = y_b + 2 - rng.random() * mound
                r = h * (0.007 + 0.009 * rng.random()) * (1.2 - 0.5 * u)
                self._skull(img, alpha, (x, yy), r, 0.55 + 0.45 * rng.random())

    def _haze(self, img: np.ndarray) -> None:
        h, W2, yh = self.h, self.W2, self.yh
        nz = cv2.resize(noise_texture(256, 64, cells=6, seed=5, octaves=3), (W2, h), interpolation=cv2.INTER_LINEAR)
        y0, y1 = max(0, int(yh - h * 0.25)), min(h, int(yh + h * 0.25))
        if y1 <= y0:
            return
        ys = np.arange(y0, y1, dtype=np.float32)
        band = np.exp(-((ys - yh) / (h * 0.09)) ** 2)[:, None]
        a = np.clip(band * (0.25 + 0.55 * nz[y0:y1]), 0, 0.75).astype(np.float32)
        fog = np.empty((y1 - y0, W2, 3), np.uint8)
        fog[:] = FOG
        img[y0:y1] = cv2.blendLinear(fog, img[y0:y1], a, 1.0 - a)

    @staticmethod
    def _vignette(W2: int, h: int) -> np.ndarray:
        y, x = np.mgrid[0:h, 0:W2].astype(np.float32)
        r = np.sqrt(((x - W2 / 2) / (W2 * 0.55)) ** 2 + ((y - h * 0.45) / (h * 0.75)) ** 2)
        v = np.clip(1.15 - 0.55 * r ** 2, 0.35, 1.0)
        return cv2.merge([v, v, v])

    def _cracks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Jagged ground-plane cracks radiating from under the user.

        Returns (screen_pts (n,2) float32, cumulative world distance (n,))."""
        rng = np.random.default_rng(21)
        out = []
        n = (10, 16, 24)[self.detail]
        for i in range(n):
            ang = math.radians(-180 + (i + 0.5 + 0.6 * (rng.random() - 0.5)) * 180 / n)  # away from camera
            ang = -ang  # +Z is into the scene
            X, Z = 0.0, 2.6
            pts, ds, d = [], [], 0.0
            for k in range(26):
                pts.append(self.ground(X, Z))
                ds.append(d)
                ang += rng.normal(0, 0.28)
                step = 0.18 + 0.25 * rng.random()
                X += math.cos(ang) * step
                Z += abs(math.sin(ang)) * step + 0.04
                d += step
                if Z > 24 or abs(X) > 9:
                    break
            out.append((np.array(pts, np.float32), np.array(ds, np.float32)))
            if rng.random() < 0.5 and len(pts) > 6:  # a branch
                j = int(rng.integers(2, len(pts) - 3))
                bx, bz = (pts[j][0] - self.cx) * 1.0, 0.0
                X0 = (pts[j][0] - self.cx) / self.f * (self.f / max(1e-3, pts[j][1] - self.yh))
                Z0 = self.f / max(1e-3, pts[j][1] - self.yh)
                a2 = ang + rng.choice([-0.8, 0.8])
                bp, bd_ = [], []
                X, Z, d = X0, Z0, ds[j]
                for k in range(8):
                    bp.append(self.ground(X, Z))
                    bd_.append(d)
                    X += math.cos(a2) * 0.2
                    Z += abs(math.sin(a2)) * 0.2 + 0.03
                    d += 0.2
                    a2 += rng.normal(0, 0.3)
                out.append((np.array(bp, np.float32), np.array(bd_, np.float32)))
        return out


def _get_scene(w: int, h: int, yh: int, detail: int) -> _Scene:
    key = (w, h, yh, detail)
    sc = _SCENES.get(key)
    if sc is None:
        if len(_SCENES) >= 3:
            _SCENES.pop(next(iter(_SCENES)))
        sc = _SCENES[key] = _Scene(w, h, yh, detail)
    return sc


# ---------------------------------------------------------------------------
# Ability
# ---------------------------------------------------------------------------
class MalevolentShrine(Ability):
    name = "malevolent_shrine"
    label = "Malevolent Shrine"
    takes_over = True

    MAX_SLASHES = 48  # storage; the live cap per quality is lower

    def __init__(self, params: dict):
        super().__init__(params)
        self._rng = np.random.default_rng(int(params.get("seed", 7)))
        self._clear_state()

    def _p(self, key: str, default: float) -> float:
        return float(self.params.get(key, default))

    def _clear_state(self) -> None:
        self.T = 0.0  # seconds since trigger
        self._impact_from = 0.0  # materialisation level when IMPACT began
        self._anchor: Optional[np.ndarray] = None  # smoothed body centre (render px)
        self._origin: Optional[np.ndarray] = None  # anchor at domain start
        self._yh: Optional[int] = None
        self._scene: Optional[_Scene] = None
        self._next_slash = 0.0
        self.slashes: List[dict] = []
        self.slash_total = 0
        self.max_live_seen = 0
        self._recent: List[Tuple[float, float]] = []
        self._last_angles: List[float] = []
        self._debris = np.zeros((0, 6), np.float32)  # x, y, vx, vy, size, rot
        self._final_done = False
        self._cancelled = False
        self._cancel_wave = 1.0
        self._m_last = 0.0

    # --- state machine ----------------------------------------------------
    def can_trigger(self, ctx: Context) -> bool:
        return self.phase == Phase.IDLE and (ctx.locked_by is None or ctx.locked_by == self.name)

    def trigger(self, ctx: Context) -> bool:
        if self.phase in (Phase.CHARGING, Phase.ACTIVE):  # early collapse
            self._cancelled = True
            self._enter(Phase.IMPACT, ctx)
            return True
        if not self.can_trigger(ctx):
            return False
        self._clear_state()
        self._rng = np.random.default_rng(int(self.params.get("seed", 7)))
        self._advance_from(-1, ctx)
        return True

    def reset(self, ctx: Context) -> None:
        super().reset(ctx)
        if ctx.locked_by == self.name:
            ctx.locked_by = None
        ctx.energy_override = None
        self._clear_state()

    def on_enter(self, phase: Phase, ctx: Context) -> None:
        if phase == Phase.CHARGING:
            ctx.locked_by = self.name
            ctx.show_title("DOMAIN EXPANSION", sub="Malevolent Shrine", color=(40, 30, 230), duration=2.2)
            ctx.shaker.kick(6)
        elif phase == Phase.ACTIVE:
            ctx.locked_by = self.name
            self._next_slash = 0.5 + 0.2 * self._rng.random()
            ctx.shaker.kick(8)
        elif phase == Phase.IMPACT:
            ctx.locked_by = self.name
            self._impact_from = self._m_last
            self._final_done = False
            self.slashes.clear()
        elif phase in (Phase.COOLDOWN, Phase.IDLE):
            if ctx.locked_by == self.name:
                ctx.locked_by = None
            self.slashes.clear()
            self._debris = np.zeros((0, 6), np.float32)

    # --- timing helpers ---------------------------------------------------
    def _charge(self) -> float:
        return self.duration(Phase.CHARGING)

    def _materialize(self) -> float:
        """0..1 how much of the shrine exists (outline+body)."""
        C = self._charge()
        if self.phase == Phase.CHARGING:
            t = self.t - 0.55 * C
        elif self.phase == Phase.ACTIVE:
            t = 0.45 * C + self.t
        else:
            return self._impact_from
        return float(np.clip(t / (0.45 * C + 0.9), 0.0, 1.0))

    def _wave(self) -> float:
        """0..1 radius fraction of the domain wave (CHARGING expands, IMPACT contracts)."""
        if self.phase == Phase.CHARGING:
            return ease_out(self.t / max(1e-3, 0.85 * self._charge()))
        if self.phase == Phase.ACTIVE:
            return 1.0
        if self.phase == Phase.IMPACT:
            I = max(1e-3, self.duration(Phase.IMPACT))
            u = (self.t / I - 0.18) / 0.72
            start = 1.0 if not self._cancelled else max(0.0, min(1.0, self._cancel_wave))
            return float(start * (1.0 - smoothstep(u)) if u > 0 else start)
        return 0.0

    # --- per-frame update -------------------------------------------------
    def on_update(self, ctx: Context) -> None:
        self.T += ctx.dt
        if self.phase in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT):
            ctx.energy_override = ENERGY
            ctx.aura_boost = max(ctx.aura_boost, 0.6 if self.phase != Phase.IMPACT else 0.3)
        self._track(ctx)
        if self.phase == Phase.ACTIVE:
            self._schedule(ctx)
        self._update_debris(ctx)
        if self.phase in (Phase.CHARGING, Phase.ACTIVE):
            self._cancel_wave = self._wave()
            self._m_last = self._materialize()

    def _track(self, ctx: Context) -> None:
        w, h = ctx.canvas.w or 960, ctx.canvas.h or 540
        bc = ctx.body_center()
        if bc is not None and np.all(np.isfinite(bc)):
            bc = np.clip(np.asarray(bc, np.float32), [0, 0], [w, h])
            if self._anchor is None:
                self._anchor = bc.copy()
            else:
                k = 1.0 - math.exp(-6.0 * max(ctx.dt, 0.0))
                self._anchor += (bc - self._anchor) * k
        if self._anchor is None:
            self._anchor = np.array([w * 0.5, h * 0.45], np.float32)
        if self._origin is None:
            self._origin = self._anchor.copy()

    # --- slashes ----------------------------------------------------------
    def _density(self) -> float:
        return float(np.clip(self._p("slash_density", 1.0), 0.0, 4.0))

    def _live_cap(self, ctx: Context) -> int:
        return (8, 14, 22)[_detail(ctx.particles.capacity)]

    def max_total(self) -> int:
        """Hard bound on slashes per domain (tests rely on it)."""
        rate = self._p("slash_rate", 9.0) * 1.6 * self._density()
        return int(math.ceil(self.duration(Phase.ACTIVE) * rate)) + 6

    def _schedule(self, ctx: Context) -> None:
        dens = self._density()
        if dens <= 0:
            return
        A = max(1e-3, self.duration(Phase.ACTIVE))
        now = self.t
        self.slashes = [s for s in self.slashes if now - s["t0"] < s["dur"]]
        if now > A - 0.25:
            return
        while now >= self._next_slash and self.slash_total < self.max_total():
            if len(self.slashes) >= self._live_cap(ctx):
                self._next_slash = now + 0.03
                break
            burst = self._rng.random() < 0.18
            for k in range(3 if burst else 1):
                if self.slash_total >= self.max_total() or len(self.slashes) >= self._live_cap(ctx):
                    break
                self._spawn_slash(ctx, now + 0.045 * k)
            rate = self._p("slash_rate", 9.0) * dens * (0.55 + 0.9 * smoothstep(now / A))
            gap = self._rng.gamma(2.0, 1.0 / (2.0 * rate)) * (2.2 if burst else 1.0)
            self._next_slash += max(0.035, gap)
        self.max_live_seen = max(self.max_live_seen, len(self.slashes))

    def _spawn_slash(self, ctx: Context, t0: float, depth: Optional[int] = None, center=None,
                     angle: Optional[float] = None, length: Optional[float] = None, width: Optional[float] = None,
                     dur: Optional[float] = None, count: bool = True) -> None:
        rng = self._rng
        w, h = ctx.canvas.w or 960, ctx.canvas.h or 540
        diag = math.hypot(w, h)
        if depth is None:
            r = rng.random()
            depth = FAR if r < 0.42 else (MID if r < 0.72 else NEAR)
        if center is None:
            # Best-candidate sampling: far from recent slashes -> well spread.
            best, best_d = None, -1.0
            for _ in range(6):
                if depth == NEAR:
                    c = (w * (0.15 + 0.7 * rng.random()), h * (0.15 + 0.7 * rng.random()))
                elif depth == MID:
                    c = (w * (0.08 + 0.84 * rng.random()), h * (0.1 + 0.6 * rng.random()))
                else:
                    c = (w * (0.1 + 0.8 * rng.random()), h * (0.05 + 0.55 * rng.random()))
                d = min((math.hypot((c[0] - a) / w, (c[1] - b) / h) for a, b in self._recent), default=1.0)
                if d > best_d:
                    best, best_d = c, d
            center = best
        if angle is None:
            for _ in range(5):
                angle = float(rng.uniform(0, math.pi))
                if all(abs(math.remainder(angle - a, math.pi)) > math.radians(22) for a in self._last_angles):
                    break
        if length is None:
            length = diag * ((0.12, 0.3), (0.28, 0.5), (0.55, 1.1))[depth][0] + \
                diag * (((0.12, 0.3), (0.28, 0.5), (0.55, 1.1))[depth][1] - ((0.12, 0.3), (0.28, 0.5),
                                                                          (0.55, 1.1))[depth][0]) * rng.random()
        if width is None:
            width = h * (0.006, 0.011, 0.02)[depth] * (0.7 + 0.6 * rng.random())
        if dur is None:
            dur = (0.4, 0.45, 0.5)[depth] * (0.8 + 0.4 * rng.random())
        self.slashes.append({"t0": t0, "dur": dur, "c": np.array(center, np.float32), "ang": angle,
                             "len": float(length), "wid": float(width), "depth": depth,
                             "bow": float(rng.uniform(-0.035, 0.035)), "fx": False})
        if count:  # the final cross-slash is not part of the ACTIVE budget
            self.slash_total += 1
        self._recent = (self._recent + [tuple(center)])[-6:]
        self._last_angles = (self._last_angles + [angle])[-2:]

    @staticmethod
    def _slash_poly(s: dict, a0: float, a1: float, width_k: float, n: int = 10) -> Optional[np.ndarray]:
        """Tapered blade-cut polygon covering [a0, a1] of the slash (0..1 along)."""
        if a1 - a0 < 1e-3:
            return None
        d = np.array([math.cos(s["ang"]), math.sin(s["ang"])], np.float32)
        nrm = np.array([-d[1], d[0]], np.float32)
        u = np.linspace(a0, a1, n, dtype=np.float32)
        along = (u - 0.5) * s["len"]
        bow = s["bow"] * s["len"] * (1 - (2 * u - 1) ** 2)
        ctr = s["c"][None, :] + d[None, :] * along[:, None] + nrm[None, :] * bow[:, None]
        v = (u - a0) / (a1 - a0)
        half = s["wid"] * width_k * np.maximum(np.sin(np.pi * v), 0.0) ** 0.55 * (0.35 + 0.65 * v)
        left = ctr + nrm[None, :] * half[:, None]
        right = ctr - nrm[None, :] * half[:, None]
        return np.concatenate([left, right[::-1]]).astype(np.float32)

    def _render_slashes(self, ctx: Context, bd: Optional[np.ndarray], x_off: float, now: float) -> None:
        canvas = ctx.canvas
        light = canvas.light
        for s in self.slashes:
            age = now - s["t0"]
            if age < 0 or age >= s["dur"]:
                continue
            p = age / s["dur"]
            # The blade crosses in ~18% of its life, then the cut lingers,
            # thins and retracts toward its end (the trail).
            sweep = min(1.0, p / 0.18)
            head = ease_out(sweep)
            tail = max(0.0, head - 0.8) if sweep < 1 else smoothstep((p - 0.45) / 0.55)
            wk = 1.0 - 0.6 * smoothstep((p - 0.18) / 0.5)
            fade = 1.0 - smoothstep((p - 0.35) / 0.65)
            depth = s["depth"]
            # One-shot impact: sparks, micro-flash/aberration (sparing).
            if not s["fx"] and depth != FAR:
                s["fx"] = True
                self._slash_fx(ctx, s)
            if depth != FAR and age < 0.09 and self.params.get("slash_warp", True):
                # Tiny screen-space displacement along the cut.
                d = np.array([math.cos(s["ang"]), math.sin(s["ang"])], np.float32)
                for k in (-0.25, 0.25):
                    c = s["c"] + d * s["len"] * k * head
                    canvas.add_warp(c, s["len"] * 0.18, 0.06 * (1 - age / 0.09), "swirl")
            outer = self._slash_poly(s, tail, head, wk)
            core = self._slash_poly(s, tail, head, 0.3 * wk)
            if outer is None:
                continue
            trail = self._slash_poly(s, 0.0, head, 1.8)
            if depth == NEAR or bd is None:
                k = fade * canvas.intensity
                if trail is not None:
                    canvas.glow_poly(trail, (30, 15, 200), 0.5 * fade)
                cv2.fillPoly(light, [outer.astype(np.int32)], tuple(int(c * k) for c in (50, 35, 240)), cv2.LINE_AA)
                cv2.fillPoly(light, [core.astype(np.int32)], tuple(int(c * k) for c in WHITE_HOT), cv2.LINE_AA)
                canvas.glow_poly(outer, (50, 30, 255), 1.6 * fade)
            else:
                # Behind the user: painted into the environment, smaller and hazier.
                dim = 0.75 if depth == FAR else 1.0
                k = fade * dim
                o = outer.astype(np.int32)
                cv2.fillPoly(bd, [o + np.array([1, 2], np.int32)], (2, 0, 6), cv2.LINE_AA)  # the incision
                cv2.fillPoly(bd, [o], tuple(int(c * k) for c in (60, 50, 245)), cv2.LINE_AA)
                cv2.fillPoly(bd, [core.astype(np.int32)], tuple(int(c * k) for c in (215, 215, 255)), cv2.LINE_AA)
                canvas.glow_poly(outer, (30, 20, 220), (0.35 if depth == FAR else 0.7) * fade)

    def _slash_fx(self, ctx: Context, s: dict) -> None:
        canvas = ctx.canvas
        near = s["depth"] == NEAR
        if near:
            ctx.shaker.kick(4 + 4 * self._rng.random())
            canvas.aberration = max(canvas.aberration, 2.5)
            if self._rng.random() < 0.3:
                canvas.flash = max(canvas.flash, 0.12)
                canvas.flash_color = (200, 190, 255)
        # Sparks along the cut.
        cap = ctx.particles.capacity
        n = int((6 if near else 3) * cap / 900)
        if n > 0:
            d = np.array([math.cos(s["ang"]), math.sin(s["ang"])], np.float32)
            u = self._rng.uniform(-0.5, 0.5, n).astype(np.float32)[:, None]
            pos = s["c"][None, :] + d[None, :] * u * s["len"]
            nrm = np.array([-d[1], d[0]], np.float32)
            vel = nrm[None, :] * self._rng.choice([-1, 1], n)[:, None] * self._rng.uniform(40, 160, (n, 1)) \
                + d[None, :] * 120
            ctx.particles.emit(n, pos.astype(np.float32), vel.astype(np.float32), spread_vel=30,
                               life=(0.2, 0.5), size=(1.0, 2.2), color=(120, 110, 255), drag=3.0)

    # --- debris -------------------------------------------------------------
    def _update_debris(self, ctx: Context) -> None:
        w, h = ctx.canvas.w or 960, ctx.canvas.h or 540
        dt = ctx.dt
        cap = (14, 24, 40)[_detail(ctx.particles.capacity)]
        dbr = self._debris
        if self.phase in (Phase.CHARGING, Phase.ACTIVE) and len(dbr) < cap and self._rng.random() < 0.5:
            n = 1
            x = self._rng.uniform(0, w, n)
            y = self._rng.uniform(h * 0.55, h, n)
            new = np.stack([x, y, self._rng.uniform(-12, 12, n), self._rng.uniform(-40, -15, n),
                            self._rng.uniform(2, 6, n), self._rng.uniform(0, 6.28, n)], 1).astype(np.float32)
            dbr = np.concatenate([dbr, new])
        if len(dbr):
            dbr[:, 0] += dbr[:, 2] * dt
            dbr[:, 1] += dbr[:, 3] * dt
            dbr[:, 5] += dt * 1.5
            if self.phase == Phase.IMPACT:
                dbr[:, 3] += 260 * dt  # everything falls as the domain collapses
            keep = (dbr[:, 1] > -20) & (dbr[:, 1] < h + 20)
            dbr = dbr[keep]
        self._debris = dbr

    def _render_debris(self, bd: np.ndarray) -> None:
        for x, y, _, _, sz, rot in self._debris:
            a = rot + np.array([0, 2.3, 4.1], np.float32)
            pts = np.stack([x + np.cos(a) * sz, y + np.sin(a) * sz * 0.7], 1).astype(np.int32)
            cv2.fillConvexPoly(bd, pts, (6, 4, 10), cv2.LINE_AA)
            cv2.line(bd, tuple(pts[0]), tuple(pts[1]), (30, 20, 140), 1, cv2.LINE_AA)

    # --- rendering ----------------------------------------------------------
    def render(self, ctx: Context) -> None:
        if self.phase in (Phase.IDLE, Phase.COOLDOWN):
            return
        canvas = ctx.canvas
        w, h = canvas.w, canvas.h
        if w <= 0 or h <= 0 or canvas.frame is None:
            return
        if self._anchor is None:
            self._track(ctx)
        anchor = self._anchor
        origin = self._origin
        if self._yh is None or getattr(self, "_yh_size", None) != (w, h):
            self._yh_size = (w, h)  # quality/resolution changed mid-domain: re-derive the horizon
            self._yh = int(np.clip(origin[1] + h * 0.2, h * 0.55, h * 0.7) // max(1, h // 24) * max(1, h // 24))
        detail = _detail(ctx.particles.capacity)
        sc = self._scene
        if sc is None or (sc.w, sc.h, sc.yh, sc.detail) != (w, h, self._yh, detail):
            sc = self._scene = _get_scene(w, h, self._yh, detail)
        # Slice offset: the world is centred on where the user stood, with a
        # touch of parallax as they move.
        par = float(self.params.get("parallax", 0.06))
        cx_screen = float(origin[0]) - (float(anchor[0]) - float(origin[0])) * par
        x0 = int(np.clip(round(sc.cx - cx_screen), 0, 2 * sc.pad))
        full = sc.full[:, x0:x0 + w]
        base = sc.base[:, x0:x0 + w]
        edges = sc.edges[:, x0:x0 + w]
        ox = sc.cx - x0  # scene centre in screen px
        bd = canvas.backdrop()
        m = self._materialize()
        wave = self._wave()
        ph = self.phase
        now = self.t
        intensity = self._p("intensity", 1.0)

        # 1) The environment: base world, shrine rising in, edge glow.
        if ph == Phase.IMPACT:
            self._render_collapse(ctx, sc, bd, base, full, x0)
        elif m >= 1.0:
            np.copyto(bd, full)
        else:
            np.copyto(bd, base)
            body = smoothstep((m - 0.25) / 0.75)
            if body > 0:
                y_cut = int(h - (h - sc.shrine_top + 4) * body)
                y_cut = max(0, min(h, y_cut))
                cv2.addWeighted(full[y_cut:], 0.4 + 0.6 * body, base[y_cut:], 0.6 - 0.6 * body, 0,
                                dst=bd[y_cut:])
                # Glowing scan line where the shrine is forming.
                band = slice(max(0, y_cut - 3), min(h, y_cut + 3))
                cv2.add(bd[band], (edges[band].astype(np.float32) * 3.0).clip(0, 255).astype(np.uint8),
                        dst=bd[band])
        if ph != Phase.IMPACT:
            # Contours: bright while forming, then a slow ominous pulse.
            e_k = smoothstep(m / 0.4) * (1.0 - 0.55 * smoothstep((m - 0.6) / 0.4))
            e_k *= 0.75 + 0.25 * math.sin(self.T * 3.1)
            if e_k > 0.02:
                cv2.addWeighted(bd, 1.0, edges, e_k * 1.4, 0, dst=bd)
        # 2) Ground cracks spreading outward.
        self._render_cracks(ctx, sc, bd, x0)
        # 3) Debris + far/mid slashes behind the user.
        self._render_debris(bd)
        if ph == Phase.ACTIVE:
            self._render_slashes(ctx, bd, ox, now)
        elif ph == Phase.IMPACT:
            self._render_slashes(ctx, None, ox, self.t)

        # 4) Wave: during CHARGING (and IMPACT) the domain only exists inside it.
        center = np.array([anchor[0], anchor[1]], np.float32)
        R = math.hypot(max(center[0], w - center[0]), max(center[1], h - center[1])) * 1.12
        if ph == Phase.CHARGING or (ph == Phase.IMPACT and wave < 1.0):
            self._wave_mask(ctx, bd, center, wave * R, ph == Phase.CHARGING)
        canvas.backdrop_amount = 1.0 if (ph != Phase.IMPACT or wave > 0.0) else 0.0
        if ph in (Phase.CHARGING, Phase.IMPACT) and 0.0 < wave < 1.0:
            r = wave * R
            canvas.add_warp(center, r, 0.5 if ph == Phase.CHARGING else -0.6, "ripple", 0.12)
            ring_k = (1.0 - wave) ** 0.3 if ph == Phase.CHARGING else wave ** 0.5
            cv2.ellipse(canvas.light, (int(center[0]), int(center[1])), (int(r), int(r * 0.92)), 0, 0, 360,
                        tuple(int(c * ring_k) for c in (40, 30, 200)), 2, cv2.LINE_AA)
            cv2.ellipse(canvas.glow, (int(center[0] * canvas.gs), int(center[1] * canvas.gs)),
                        (max(1, int(r * canvas.gs)), max(1, int(r * 0.92 * canvas.gs))), 0, 0, 360,
                        tuple(c * ring_k * 1.3 for c in (30, 15, 220)), max(2, int(14 * canvas.gs)), cv2.LINE_AA)

        # 5) Grade: deep red / black.
        domain = wave if ph != Phase.ACTIVE else 1.0
        canvas.tint = (6, 2, 34)
        canvas.tint_amount = 0.14 * domain * intensity
        canvas.dim = max(canvas.dim, 0.12 * domain)
        # 6) Red backlight rim around the user, fog in the glow layer, embers.
        self._rim_and_fog(ctx, domain, m)
        if ph in (Phase.CHARGING, Phase.ACTIVE):
            self._embers(ctx, domain)
        if ph == Phase.CHARGING and self.t < 0.15:
            canvas.flash = max(canvas.flash, 0.18 * (1 - self.t / 0.15))
            canvas.flash_color = (20, 10, 120)

    def _wave_mask(self, ctx: Context, bd: np.ndarray, center, r: float, charging: bool) -> None:
        """Inside radius r: the domain; outside: the camera image (darkening rim)."""
        canvas = ctx.canvas
        gw, gh, gs = canvas.gw, canvas.gh, canvas.gs
        ys, xs = np.ogrid[0:gh, 0:gw]
        dist = np.sqrt((xs - center[0] * gs) ** 2 + ((ys - center[1] * gs) / 0.92) ** 2).astype(np.float32)
        band = max(2.0, 26.0 * gs)
        wgt = np.clip((r * gs - dist) / band, 0.0, 1.0)
        # A darkening front just ahead of the wave.
        dark = np.exp(-((dist - r * gs - band) / (band * 1.6)) ** 2) * 0.55 if charging else 0.0
        buf = self._wave_bufs(canvas.w, canvas.h)
        wfull, winv, dfull, dinv, black, tmp = buf
        cv2.resize(wgt.astype(np.float32), (canvas.w, canvas.h), dst=wfull, interpolation=cv2.INTER_LINEAR)
        np.subtract(1.0, wfull, out=winv)
        cam = canvas.frame
        if charging:
            keep = (1.0 - dark).astype(np.float32)
            cv2.resize(keep, (canvas.w, canvas.h), dst=dfull, interpolation=cv2.INTER_LINEAR)
            np.subtract(1.0, dfull, out=dinv)
            cv2.blendLinear(cam, black, dfull, dinv, dst=tmp)  # blending with black = darkening
            cam = tmp
        cv2.blendLinear(bd, cam, wfull, winv, dst=bd)

    def _wave_bufs(self, w: int, h: int):
        """Full-res scratch buffers for the wave, reused while the size is unchanged."""
        bufs = getattr(self, "_wbufs", None)
        if bufs is None or bufs[0].shape != (h, w):
            f = lambda: np.zeros((h, w), np.float32)  # noqa: E731
            bufs = self._wbufs = (f(), f(), f(), f(), np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.uint8))
        return bufs

    def _render_cracks(self, ctx: Context, sc: _Scene, bd: np.ndarray, x0: int) -> None:
        C = self._charge()
        if self.phase == Phase.CHARGING:
            spread = max(0.0, (self.t - 0.25 * C) / (0.75 * C + 1.6))
        elif self.phase == Phase.ACTIVE:
            spread = (0.75 * C + self.t) / (0.75 * C + 1.6)
        else:
            spread = 1.0
        spread = min(1.0, spread)
        if spread <= 0:
            return
        reach = 9.0 * ease_out(spread)
        fade = 1.0
        if self.phase == Phase.IMPACT:
            fade = 1.0 - smoothstep(self.t / max(1e-3, self.duration(Phase.IMPACT) * 0.7))
            if fade <= 0:
                return
        pulse = 0.75 + 0.25 * math.sin(self.T * 4.0)
        off = np.array([x0, 0], np.float32)
        outer, inner, gl = [], [], []
        for pts, ds in sc.cracks:
            n = int(np.searchsorted(ds, reach))
            if n < 2:
                continue
            p = (pts[:n] - off)
            outer.append(p.astype(np.int32))
            gl.append((p * ctx.canvas.gs).astype(np.int32))
        if not outer:
            return
        cv2.polylines(bd, outer, False, (4, 2, 10), 4, cv2.LINE_AA)
        cv2.polylines(bd, outer, False, tuple(int(c * pulse * fade) for c in (40, 30, 210)), 1, cv2.LINE_AA)
        cv2.polylines(ctx.canvas.glow, gl, False, tuple(c * 0.7 * pulse * fade for c in (30, 15, 230)), 1,
                      cv2.LINE_AA)

    def _render_collapse(self, ctx, sc: _Scene, bd, base, full, x0) -> None:
        """IMPACT: final cross-slash, then the shrine sinks and crumbles."""
        I = max(1e-3, self.duration(Phase.IMPACT))
        u = self.t / I
        h, w = sc.h, sc.w
        m = self._impact_from
        if not self._final_done:
            self._final_done = True
            if m > 0.5:
                diag = ctx.diag
                c = self._anchor.copy() if self._anchor is not None else np.array([w / 2, h / 2], np.float32)
                c[1] -= h * 0.08
                for ang in (math.radians(-38), math.radians(38)):
                    self._spawn_slash(ctx, self.t, NEAR, c, ang, diag * 1.25, h * 0.05, 0.6, count=False)
            ctx.canvas.flash = max(ctx.canvas.flash, 0.45)
            ctx.canvas.flash_color = (150, 140, 255)
            ctx.canvas.aberration = max(ctx.canvas.aberration, 6.0)
            ctx.shaker.kick(16)
            n = int(ctx.particles.capacity * 0.12)
            if n:
                xs = np.random.uniform(w * 0.2, w * 0.8, n)
                ys = np.random.uniform(sc.shrine_top, sc.yh, n)
                ctx.particles.emit(n, np.stack([xs, ys], 1).astype(np.float32), (0, 40), spread_vel=60,
                                   life=(0.5, 1.2), size=(1.2, 3.0), color=EMBER, drag=1.0, gravity=120)
        np.copyto(bd, base)
        vis = m * (1.0 - smoothstep((u - 0.2) / 0.7))
        if vis > 0.01:
            dy = int(h * 0.25 * ease_out(max(0.0, (u - 0.1) / 0.8)) ** 1.5)
            a = sc.alpha[:, x0:x0 + w]
            src = full
            if dy > 0:
                a = a[:h - dy]
                src = full[:h - dy]
            wgt = a * vis
            # Crumble: sample the shrine with a noise threshold that eats it away.
            if u > 0.15:
                nz = cv2.resize(noise_texture(160, 90, cells=10, seed=9, octaves=2), (w, h - dy),
                                interpolation=cv2.INTER_LINEAR)
                wgt = wgt * (nz > 0.15 + 0.85 * smoothstep((u - 0.12) / 0.7))
            region = bd[dy:]
            region[:] = cv2.blendLinear(src, region, wgt.astype(np.float32), (1.0 - wgt).astype(np.float32))

    def _rim_and_fog(self, ctx: Context, domain: float, m: float) -> None:
        canvas = ctx.canvas
        if domain <= 0.01:
            return
        gw, gh = canvas.gw, canvas.gh
        mask = ctx.tracking.mask
        small = None
        if mask is not None and mask.ndim == 2 and mask.size:
            small = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
            halo = cv2.GaussianBlur(small, (0, 0), 2.5) - small
            np.clip(halo, 0, 1, out=halo)
            k = (0.8 + 0.25 * math.sin(self.T * 5)) * domain * canvas.intensity
            canvas.glow += cv2.merge([halo * 30 * k, halo * 14 * k, halo * 230 * k])
        if self.phase == Phase.ACTIVE or m > 0.3:
            # Drifting fog along the horizon (low-res, kept off the person).
            nz = noise_texture(gw * 2, gh, cells=5, seed=13, octaves=2)
            sx = int((self.T * 12 * canvas.gs) % gw)
            f = nz[:, sx:sx + gw]
            yh = (self._yh or canvas.h * 0.6) * canvas.gs
            ys = np.arange(gh, dtype=np.float32)
            band = np.exp(-((ys - yh) / (gh * 0.12)) ** 2)[:, None]
            fog = f * band * min(1.0, m) * domain
            if small is not None:
                fog = fog * (1.0 - small)
            canvas.glow += cv2.merge([fog * 14, fog * 6, fog * 60])

    def _embers(self, ctx: Context, domain: float) -> None:
        cap = ctx.particles.capacity
        rate = cap * 0.07 * domain  # per second
        n = int(rate * ctx.dt + self._rng.random())
        if n <= 0:
            return
        w, h = ctx.canvas.w, ctx.canvas.h
        pos = np.stack([self._rng.uniform(0, w, n), self._rng.uniform(h * 0.6, h * 1.02, n)], 1).astype(np.float32)
        vel = np.stack([self._rng.uniform(-25, 25, n), self._rng.uniform(-90, -35, n)], 1).astype(np.float32)
        col = EMBER if self._rng.random() < 0.6 else (30, 40, 230)
        ctx.particles.emit(n, pos, vel, spread_vel=8, life=(1.2, 2.6), size=(0.8, 2.2), color=col, drag=0.3,
                           gravity=-10)
