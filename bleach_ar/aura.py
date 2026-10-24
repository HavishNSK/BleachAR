"""Reiatsu aura: spiritual pressure flaring off the body silhouette.

Everything is computed at the glow-layer resolution (``canvas.gw x
canvas.gh``, capped at ``_MAX_WORK_W`` wide) for speed. The silhouette comes from the segmentation mask when
available, otherwise it is drawn from the pose skeleton (or from the face
alone). A soft, upward-smeared copy of the silhouette is eroded by a
scrolling, tileable noise texture to get rising flame tongues; the result is
added to the soft glow layer and, upsampled inside the body's bounding box,
to the crisp light layer. In Bankai the flames turn into black wisps (the
camera image is darkened) with a violent colored rim.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .abilities import BANKAI, Context
from .fx import Canvas
from .tracking import FACE_OVAL

_NOISE_N = 128  # side of the square, tileable noise texture
_MAX_WORK_W = 320  # cap on the working width so the "high" preset stays cheap


def _tileable_noise(n: int, fx: float, fy: float, seed: int) -> np.ndarray:
    """Periodic band-limited noise in 0..1. ``fx``/``fy`` are cutoff frequencies (cycles per tile)."""
    rng = np.random.default_rng(seed)
    spec = np.fft.fft2(rng.standard_normal((n, n)))
    ky = np.fft.fftfreq(n)[:, None] * n
    kx = np.fft.fftfreq(n)[None, :] * n
    spec *= np.exp(-((kx / fx) ** 2 + (ky / fy) ** 2))
    out = np.real(np.fft.ifft2(spec))
    # Rank-equalise to a uniform 0..1 distribution so thresholds cut clean tongues.
    ranks = np.empty(out.size, np.float32)
    ranks[np.argsort(out, axis=None)] = np.linspace(0.0, 1.0, out.size, dtype=np.float32)
    return ranks.reshape(n, n)


def _valid(p: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(p)) and (abs(p[0]) > 1e-3 or abs(p[1]) > 1e-3))


class ReiatsuAura:
    """Persistent body aura. ``update`` builds the silhouette and emits embers; ``render`` draws."""

    def __init__(self, cfg: dict):
        self.params = cfg["abilities"]["aura"]
        # Two octaves, stretched vertically (lower frequency in y) so tongues look like flames.
        self.noise_a = _tileable_noise(_NOISE_N, fx=9.0, fy=4.0, seed=7)
        self.noise_b = _tileable_noise(_NOISE_N, fx=20.0, fy=10.0, seed=11)
        self.scroll_a = 0.0
        self.scroll_b = 0.0
        self.alpha = 0.0  # smoothed visibility (follows tracking confidence)
        self.sil: Optional[np.ndarray] = None  # float32 silhouette at glow resolution
        self._emit_acc = 0.0
        self._edge: Optional[np.ndarray] = None
        self.ws = 0.33  # render px -> working px

    @staticmethod
    def _work_size(cv: Canvas) -> Tuple[int, int, float]:
        f = min(1.0, _MAX_WORK_W / max(1, cv.gw))
        ww, wh = max(8, int(round(cv.gw * f))), max(8, int(round(cv.gh * f)))
        return ww, wh, ww / max(1, cv.w)

    # --- silhouette ---------------------------------------------------------
    def _silhouette(self, ctx: Context) -> Tuple[Optional[np.ndarray], float]:
        """Return (silhouette at glow res, confidence) or (None, 0)."""
        tr = ctx.tracking
        cv = ctx.canvas
        face, pose = tr.face, tr.pose
        conf = max(face.confidence if face.ok else 0.0, pose.confidence if pose.ok else 0.0)
        if conf <= 0.0 or cv.gw <= 0:
            return None, 0.0
        gw, gh, gs = self._work_size(cv)
        self.ws = gs
        if tr.mask is not None and tr.mask.size > 0:
            m = cv2.resize(tr.mask.astype(np.float32, copy=False), (gw, gh), interpolation=cv2.INTER_LINEAR)
            np.clip((m - 0.25) * 2.0, 0.0, 1.0, out=m)
            return m, conf
        img = np.zeros((gh, gw), np.uint8)
        drawn = False
        if pose.ok:
            p = pose.points[:, :2] * gs
            ls, rs = p[11], p[12]
            if _valid(ls) and _valid(rs):
                sw = float(np.linalg.norm(ls - rs))
                down = np.array([0.0, sw * 1.6], np.float32)
                lh = p[23] if _valid(p[23]) else ls + down
                rh = p[24] if _valid(p[24]) else rs + down
                cv2.fillPoly(img, [np.array([rs, ls, lh, rh], np.int32)], 255, cv2.LINE_AA)
                for chain, k in (((11, 13, 15), 0.2), ((12, 14, 16), 0.2),
                                 ((23, 25, 27), 0.3), ((24, 26, 28), 0.3)):
                    th = max(1, int(sw * k))
                    for a, b in zip(chain[:-1], chain[1:]):
                        if _valid(p[a]) and _valid(p[b]):
                            cv2.line(img, (int(p[a, 0]), int(p[a, 1])), (int(p[b, 0]), int(p[b, 1])),
                                     255, th, cv2.LINE_AA)
                mid = (ls + rs) / 2.0
                if face.ok:
                    head = face.points[152, :2] * gs
                else:
                    head = p[0] if _valid(p[0]) else mid - np.array([0, sw * 0.6], np.float32)
                    cv2.circle(img, (int(head[0]), int(head[1])), max(1, int(sw * 0.3)), 255, -1, cv2.LINE_AA)
                cv2.line(img, (int(mid[0]), int(mid[1])), (int(head[0]), int(head[1])), 255,
                         max(1, int(sw * 0.28)), cv2.LINE_AA)
                drawn = True
        if face.ok:
            fp = face.points
            oval = fp[FACE_OVAL, :2] * gs
            cv2.fillPoly(img, [oval.astype(np.int32)], 255, cv2.LINE_AA)
            if not drawn:
                # Face only (close-up): guess neck and shoulders below the chin.
                chin = fp[152, :2] * gs
                fwid = float(oval[:, 0].max() - oval[:, 0].min())
                fhgt = float(oval[:, 1].max() - oval[:, 1].min())
                cx = float(oval[:, 0].mean())
                sy = chin[1] + fhgt * 0.35
                body = np.array([[cx - fwid * 0.22, chin[1] - fhgt * 0.1], [cx + fwid * 0.22, chin[1] - fhgt * 0.1],
                                 [cx + fwid * 0.3, sy - fhgt * 0.12], [cx + fwid * 1.15, sy],
                                 [cx + fwid * 1.35, gh + 2], [cx - fwid * 1.35, gh + 2],
                                 [cx - fwid * 1.15, sy], [cx - fwid * 0.3, sy - fhgt * 0.12]], np.int32)
                cv2.fillPoly(img, [body], 255, cv2.LINE_AA)
        m = cv2.GaussianBlur(img, (0, 0), 1.2).astype(np.float32) * (1.0 / 255.0)
        return m, conf

    # --- per frame ----------------------------------------------------------
    def update(self, ctx: Context) -> None:
        dt = ctx.dt
        boost = max(0.0, ctx.aura_boost)
        self.scroll_a = (self.scroll_a + dt * (34.0 + 18.0 * boost)) % _NOISE_N
        self.scroll_b = (self.scroll_b + dt * (60.0 + 30.0 * boost)) % _NOISE_N
        if not ctx.show_aura:
            self.sil = None
            self.alpha = 0.0
            return
        sil, conf = self._silhouette(ctx)
        target = conf if sil is not None else 0.0
        rate = 1.0 - math.exp(-dt * 10.0) if dt > 0 else 1.0
        self.alpha += (target - self.alpha) * rate
        if sil is None:
            self.sil = None
            self._edge = None
            return
        self.sil = sil
        # Edge band around the 0.5 contour of the silhouette (used for embers and the rim).
        soft = cv2.GaussianBlur(sil, (0, 0), 1.3)
        edge = 1.0 - np.abs(soft - 0.5) * 2.4
        np.clip(edge, 0.0, 1.0, out=edge)
        self._edge = edge
        self._emit(ctx, edge)

    def _intensity(self, ctx: Context) -> float:
        return float(self.params.get("intensity", 1.0)) * (1.0 + max(0.0, ctx.aura_boost))

    def _emit(self, ctx: Context, edge: np.ndarray) -> None:
        k = self._intensity(ctx) * self.alpha
        if k <= 0.01 or ctx.dt <= 0:
            return
        self._emit_acc += ctx.dt * 45.0 * k
        n = min(int(self._emit_acc), 14)
        self._emit_acc -= int(self._emit_acc)
        if n <= 0:
            return
        gh = edge.shape[0]
        ys, xs = np.nonzero(edge[: max(1, gh - 3)] > 0.6)
        if ys.size == 0:
            return
        pick = np.random.randint(0, ys.size, n)
        gs = self.ws
        pos = np.stack([(xs[pick] + np.random.uniform(0, 1, n)) / gs,
                        (ys[pick] + np.random.uniform(0, 1, n)) / gs], 1).astype(np.float32)
        up = math.sqrt(k)
        vel = np.stack([np.random.normal(0, 22, n), -np.random.uniform(70, 190, n) * up], 1).astype(np.float32)
        col = ctx.energy_color()
        if ctx.level != BANKAI:
            col = tuple(min(255, c * 0.7 + 90) for c in col)
        ctx.particles.emit(n, pos, vel, life=(0.5, 1.2), size=(1.0, 2.6), color=col, drag=0.8)

    def render(self, ctx: Context) -> None:
        sil = self.sil
        cv = ctx.canvas
        if sil is None or self._edge is None or self.alpha <= 0.01:
            return
        gw, gh, gs = self._work_size(cv)
        if sil.shape != (gh, gw):
            return
        boost = max(0.0, ctx.aura_boost)
        # Brightness grows sub-linearly with the boost; the rest goes into flame height.
        k = float(self.params.get("intensity", 1.0)) * (1.0 + 0.3 * min(boost, 3.0)) * self.alpha
        bankai = ctx.level == BANKAI
        now = ctx.now
        color = np.asarray(ctx.energy_color(), np.float32)
        pulse = 0.85 + 0.15 * math.sin(now * 4.2) + 0.06 * math.sin(now * 11.3)

        # Upward-smeared soft silhouette: the "fuel" for the flames.
        base = max(1.5, gh * 0.022)
        soft = cv2.GaussianBlur(sil, (0, 0), base * 1.6)
        fuel = soft.copy()
        steps = 5 + min(4, int(boost * 1.4))
        shift = max(1, int(base * (1.2 + 0.2 * min(boost, 3.0))))
        for i in range(1, steps + 1):
            s = shift * i
            if s >= gh:
                break
            np.maximum(fuel[:-s], soft[s:] * (1.0 - i / (steps + 1.5)), out=fuel[:-s])

        # Scrolling noise, two octaves moving at different speeds so it evolves.
        ys = np.arange(gh)
        xs = np.arange(gw)
        # Per-row horizontal wobble (cheap domain warp) so tongues curl as they rise.
        sway = (3.0 * np.sin(ys * 0.13 + now * 3.1) + 2.0 * math.sin(now * 1.7)).astype(np.int32)
        na = self.noise_a[((ys + int(self.scroll_a)) % _NOISE_N)[:, None], (xs[None, :] + sway[:, None]) % _NOISE_N]
        nb = self.noise_b[np.ix_((ys + int(self.scroll_b)) % _NOISE_N, (xs * 1 + 37) % _NOISE_N)]
        noise = na * 0.65 + nb * 0.35

        flame = fuel * (1.7 + 0.2 * min(boost, 3.0)) - noise
        np.clip(flame, 0.0, 1.0, out=flame)
        outside = 1.0 - sil
        flame *= outside
        edge = self._edge * (0.75 + 0.5 * nb)

        # Soft glow layer: wide halo plus flames plus the rim.
        halo = soft * outside * 0.3
        if bankai:
            gfield = flame * (1.0 - flame) * 1.4 + edge * 0.6 + halo * 0.5
        else:
            gfield = flame * 1.1 + edge * 0.5 + halo
        strength = k * pulse * cv.intensity
        gimg = cv2.merge([gfield * (color[0] * strength), gfield * (color[1] * strength),
                          gfield * (color[2] * strength)])
        if (gw, gh) != (cv.gw, cv.gh):
            gimg = cv2.resize(gimg, (cv.gw, cv.gh), interpolation=cv2.INTER_LINEAR)
        cv2.add(cv.glow, gimg, dst=cv.glow)

        # Region of interest in render pixels (bounding box of everything drawn).
        x, y, bw, bh = cv2.boundingRect((fuel > 0.03).astype(np.uint8))
        if bw == 0 or bh == 0:
            return
        x0, y0 = max(0, int(x / gs) - 2), max(0, int(y / gs) - 2)
        x1, y1 = min(cv.w, int((x + bw) / gs) + 3), min(cv.h, int((y + bh) / gs) + 3)
        if x1 <= x0 or y1 <= y0:
            return
        gx0, gy0, gx1, gy1 = x, y, x + bw, y + bh
        size = (x1 - x0, y1 - y0)

        if bankai:
            # Black wisps: darken the camera image under the flame cores.
            dark = np.clip(flame * 1.6, 0.0, 1.0) * (0.75 * min(1.0, k))
            inv = cv2.resize((255.0 * (1.0 - dark[gy0:gy1, gx0:gx1])).astype(np.uint8), size,
                             interpolation=cv2.INTER_LINEAR)
            roi = cv.frame[y0:y1, x0:x1]
            cv2.multiply(roi, cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR), dst=roi, scale=1.0 / 255.0)
            # Violent colored rim on the outer boundary of each wisp.
            band = flame * (1.0 - flame) * 4.0
            lfield = band * (0.55 + 0.35 * min(boost, 2.0)) + edge * 0.85
        else:
            lfield = flame * 0.85 + edge * 0.6
        ls = min(1.4, k * pulse) * cv.intensity
        sub = lfield[gy0:gy1, gx0:gx1]
        hot = np.clip(color * 0.9 + 35.0 if bankai else color * 0.6 + 100.0, 0, 255)  # whiter core color for the crisp layer
        lc = cv2.merge([sub * (hot[0] * ls), sub * (hot[1] * ls), sub * (hot[2] * ls)])
        lc = cv2.resize(cv2.convertScaleAbs(lc), size, interpolation=cv2.INTER_LINEAR)
        lroi = cv.light[y0:y1, x0:x1]
        cv2.add(lroi, lc, dst=lroi)

        self._ground(ctx, sil, color, k * pulse)

    def _ground(self, ctx: Context, sil: np.ndarray, color: np.ndarray, k: float) -> None:
        """Flat glowing pool and ripples under the body when it reaches the bottom edge."""
        cv: Canvas = ctx.canvas
        row = sil[-2]
        xs = np.nonzero(row > 0.5)[0]
        if xs.size < 2:
            return
        cx = float(xs.mean()) / self.ws
        half = (float(xs.max() - xs.min()) / self.ws) * 0.5
        gs = cv.gs
        cy = cv.h - cv.h * 0.02
        ry = max(6.0, cv.h * 0.035)
        col = tuple(float(c) * k * cv.intensity for c in color)
        cv2.ellipse(cv.glow, (int(cx * gs), int(cy * gs)), (max(1, int(half * 1.5 * gs)), max(1, int(ry * gs))),
                    0, 0, 360, tuple(c * 0.55 for c in col), -1, cv2.LINE_AA)
        for i in range(3):
            ph = (ctx.now * 0.7 + i / 3.0) % 1.0
            ax = int(half * (0.7 + 1.3 * ph))
            ay = int(ry * (0.6 + 1.0 * ph))
            fade = (1.0 - ph) ** 1.5
            c = tuple(int(min(255, cc * 0.6 * fade)) for cc in col)
            cv2.ellipse(cv.light, (int(cx), int(cy)), (max(1, ax), max(1, ay)), 0, 0, 360, c, 2, cv2.LINE_AA)
