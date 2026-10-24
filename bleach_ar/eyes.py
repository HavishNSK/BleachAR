"""Glowing eyes and a procedural Hollow-style bone mask.

Eyes: the iris centre is face landmark 468 / 473 and its radius comes from
the refined iris ring (469-472 / 474-477). Each eye fades with an eyelid
open ratio (vertical / horizontal eyelid distance, open above ~0.20), so
blinking hides the glow. Fast head motion leaves short light streaks.

Mask: an original bone-mask design built in a face-local frame defined by
the two irises and the chin, so it follows head tilt and scale. An optional
RGBA image (``assets.mask_image``) replaces the procedural mask; it must be
front-facing with the eyes at (0.32, 0.42) and (0.68, 0.42) and the chin at
(0.5, 0.95) of the image size.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .abilities import BANKAI, Context
from .config import PROJECT_ROOT
from .fx import scale_color
from .tracking import EYE_A, EYE_B, FACE_OVAL

# Eye styles: (iris/halo color, hot core color), BGR.
EYE_STYLES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "hollow": ((30, 175, 255), (170, 240, 255)),
    "red": ((40, 40, 255), (190, 190, 255)),
    "blue": ((255, 160, 50), (255, 240, 200)),
    "white": ((255, 245, 235), (255, 255, 255)),
}

# (iris, iris ring, eyelid top, bottom, inner corner, outer corner, contour)
_EYES = ((468, (469, 470, 471, 472), 159, 145, 133, 33, EYE_A),
         (473, (474, 475, 476, 477), 386, 374, 362, 263, EYE_B))

BONE = np.array([200, 222, 232], np.float32)  # warm off-white (BGR)
CRIMSON = (40, 25, 175)
HOLE = (14, 10, 12)

# Reference points of an optional mask image, as fractions of its size.
_IMG_REF = np.array([[0.32, 0.42], [0.68, 0.42], [0.5, 0.95]], np.float32)

# Stripe markings in face-local units: x = -0.5 / +0.5 at the irises, y = 0
# on the iris line, y = 1 at the chin. Defined for the screen-left side and
# mirrored.
_STRIPES = (
    np.array([[-0.50, -0.78], [-0.36, -0.80], [-0.44, -0.30], [-0.52, -0.26]], np.float32),
    np.array([[-0.62, 0.20], [-0.44, 0.22], [-0.52, 0.62], [-0.58, 0.60]], np.float32),
)


def _eye_open_ratio(pts: np.ndarray, top: int, bot: int, inner: int, outer: int) -> float:
    horiz = float(np.linalg.norm(pts[inner, :2] - pts[outer, :2]))
    if horiz < 1e-3:
        return 0.0
    return float(np.linalg.norm(pts[top, :2] - pts[bot, :2])) / horiz


def _local_affine(pts: np.ndarray) -> Optional[np.ndarray]:
    """2x3 affine mapping face-local units to render pixels (irises and chin)."""
    src = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, 1.0]], np.float32)
    dst = np.array([pts[468, :2], pts[473, :2], pts[152, :2]], np.float32)
    if not np.all(np.isfinite(dst)):
        return None
    area = abs(float(np.cross(dst[1] - dst[0], dst[2] - dst[0])))
    if area < 25.0:
        return None
    return cv2.getAffineTransform(src, dst)


def _apply(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p @ m[:, :2].T + m[:, 2]).astype(np.float32)


class EyeEffects:
    """Glowing eyes (``ctx.show_eyes``) and the bone mask (``ctx.show_mask``)."""

    def __init__(self, cfg: dict):
        self.openness = [1.0, 1.0]
        self.history: deque = deque(maxlen=10)  # (t, iris_a, iris_b, radius)
        self.mask_img: Optional[np.ndarray] = None
        raw = (cfg.get("assets", {}) or {}).get("mask_image") or ""
        if raw:
            path = raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                print(f"[assets] mask_image '{path}' is missing or not an RGBA image; "
                      "using the procedural mask.")
            else:
                self.mask_img = img

    # --- per frame ----------------------------------------------------------
    def update(self, ctx: Context) -> None:
        face = ctx.tracking.face
        if not face.ok or face.points.shape[0] < 478:
            self.history.clear()
            return
        pts = face.points
        k = 1.0 - np.exp(-ctx.dt * 25.0) if ctx.dt > 0 else 1.0
        for i, (_, _, top, bot, inner, outer, _) in enumerate(_EYES):
            ratio = _eye_open_ratio(pts, top, bot, inner, outer)
            target = float(np.clip((ratio - 0.13) / (0.21 - 0.13), 0.0, 1.0))
            self.openness[i] += (target - self.openness[i]) * k
        r = self._iris_radius(pts)
        self.history.append((ctx.now, pts[468, :2].copy(), pts[473, :2].copy(), r))
        while self.history and ctx.now - self.history[0][0] > 0.22:
            self.history.popleft()

    @staticmethod
    def _iris_radius(pts: np.ndarray) -> float:
        d = (np.linalg.norm(pts[469, :2] - pts[471, :2]) + np.linalg.norm(pts[470, :2] - pts[472, :2])
             + np.linalg.norm(pts[474, :2] - pts[476, :2]) + np.linalg.norm(pts[475, :2] - pts[477, :2]))
        return max(2.0, float(d) / 8.0)

    def render(self, ctx: Context) -> None:
        if not (ctx.show_eyes or ctx.show_mask):
            return
        face = ctx.tracking.face
        if not face.ok or face.points.shape[0] < 478 or ctx.canvas.frame is None:
            return
        alpha = float(np.clip(face.confidence, 0.0, 1.0))
        if alpha <= 0.01:
            return
        if ctx.show_mask:
            self._render_mask(ctx, face.points, alpha)
        if ctx.show_eyes:
            self._render_eyes(ctx, face.points, alpha)

    # --- eyes -----------------------------------------------------------------
    def _render_eyes(self, ctx: Context, pts: np.ndarray, alpha: float) -> None:
        cv = ctx.canvas
        color, core = EYE_STYLES.get(ctx.eye_style, EYE_STYLES["hollow"])
        bankai = ctx.level == BANKAI
        power = alpha * min(1.8, (1.4 if bankai else 1.0) * (1.0 + 0.15 * max(0.0, ctx.aura_boost)))
        r = self._iris_radius(pts)
        for i, (iris, _, _, _, _, _, contour) in enumerate(_EYES):
            op = self.openness[i]
            if op <= 0.05:
                continue
            c = pts[iris, :2]
            s = power * op
            if ctx.eye_style == "hollow":
                self._darken_sclera(cv.frame, pts[contour, :2], min(1.0, 0.85 * alpha * op))
            # Soft colored halo (glow layer) and a horizontal anamorphic flare.
            halo = max(r * 4.0, 12.0) * (1.35 if bankai else 1.0)
            cv.glow_circle(c, halo, color, 0.45 * s)
            cv.glow_circle(c, max(r * 1.8, 6.0), color, 1.2 * s)
            flare = r * (9.0 if bankai else 5.5) * op
            cv.light_line(c - (flare, 0), c + (flare, 0), color, max(1, int(r * 0.25)), 0.35 * s)
            cv.glow_line(c - (flare * 1.4, 0), c + (flare * 1.4, 0), color, r * 0.9, 0.35 * s)
            # Crisp iris ring and white-hot core, squashed vertically as the lid closes.
            ax = (max(1, int(r * 1.05)), max(1, int(r * 1.05 * op)))
            cv2.ellipse(cv.light, (int(c[0]), int(c[1])), ax, 0, 0, 360,
                        scale_color(color, min(1.0, s) * cv.intensity), -1, cv2.LINE_AA)
            ax = (max(1, int(r * 0.55)), max(1, int(r * 0.55 * op)))
            cv2.ellipse(cv.light, (int(c[0]), int(c[1])), ax, 0, 0, 360,
                        scale_color(core, min(1.0, s * 1.1) * cv.intensity), -1, cv2.LINE_AA)
        self._render_trail(ctx, color, power)

    @staticmethod
    def _darken_sclera(frame: np.ndarray, contour: np.ndarray, strength: float) -> None:
        h, w = frame.shape[:2]
        x0, y0 = np.floor(contour.min(0)).astype(int) - 3
        x1, y1 = np.ceil(contour.max(0)).astype(int) + 4
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [(contour - (x0, y0)).astype(np.int32)], int(255 * strength), cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (3, 3), 0)
        roi = frame[y0:y1, x0:x1]
        cv2.multiply(roi, cv2.cvtColor(255 - m, cv2.COLOR_GRAY2BGR), dst=roi, scale=1.0 / 255.0)

    def _render_trail(self, ctx: Context, color: Sequence[int], power: float) -> None:
        hist = self.history
        if len(hist) < 2:
            return
        cv = ctx.canvas
        now = ctx.now
        items = list(hist)
        for (t0, a0, b0, r0), (t1, a1, b1, r1) in zip(items[:-1], items[1:]):
            age = 1.0 - (now - t1) / 0.22
            if age <= 0:
                continue
            for i, (p, q) in enumerate(((a0, a1), (b0, b1))):
                op = self.openness[i]
                if op <= 0.05 or np.linalg.norm(q - p) < max(4.0, r1 * 0.9):
                    continue
                k = power * op * age * age
                cv.light_line(p, q, color, max(1, int(r1 * 1.1 * age)), 0.8 * k)
                cv.glow_line(p, q, color, r1 * 2.5, 0.6 * k)

    # --- mask -----------------------------------------------------------------
    def _render_mask(self, ctx: Context, pts: np.ndarray, alpha: float) -> None:
        aff = _local_affine(pts)
        if aff is None:
            return
        frame = ctx.canvas.frame
        h, w = frame.shape[:2]
        oval = pts[FACE_OVAL, :2]
        center = oval.mean(0)
        oval = center + (oval - center) * 0.97
        bounds = oval
        if self.mask_img is not None:
            # The image may extend past the face oval (horns, hair): use its warped corners.
            ih, iw = self.mask_img.shape[:2]
            m = self._image_affine(aff)
            corners = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], np.float32)
            bounds = np.concatenate([oval, _apply(m, corners)], 0)
        x0, y0 = np.floor(bounds.min(0)).astype(int) - 6
        x1, y1 = np.ceil(bounds.max(0)).astype(int) + 7
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        off = np.array([x0, y0], np.float32)
        roi = frame[y0:y1, x0:x1]
        if self.mask_img is not None:
            self._blend_image(roi, aff, off, alpha)
            return
        rh, rw = roi.shape[:2]
        a = np.zeros((rh, rw), np.uint8)
        cv2.fillPoly(a, [(oval - off).astype(np.int32)], 255, cv2.LINE_AA)
        a = cv2.GaussianBlur(a, (5, 5), 0)

        # Shading: soft falloff toward the edges, lit from the top, modulated
        # by the real face's brightness so it sits in the scene's lighting.
        span = float(np.linalg.norm(pts[473, :2] - pts[468, :2]))
        # Wide blur done at quarter resolution (a large kernel at full res is slow).
        small = cv2.resize(a, (max(2, rw // 4), max(2, rh // 4)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small.astype(np.float32) * (1.0 / 255.0), (0, 0), max(1.0, span * 0.09))
        edge = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        up = _apply(aff, np.array([[0.0, -0.9]], np.float32))[0]
        dn = pts[152, :2]
        axis = dn - up
        L2 = float(axis @ axis) or 1.0
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        tpos = ((xx + x0 - up[0]) * axis[0] + (yy + y0 - up[1]) * axis[1]) / L2
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
        shade = (0.55 + 0.45 * np.clip(edge * 1.6 - 0.25, 0, 1)) * (1.08 - 0.28 * np.clip(tpos, 0, 1))
        shade *= 0.8 + 0.3 * gray
        col = cv2.merge([shade * BONE[0], shade * BONE[1], shade * BONE[2]])
        col = cv2.convertScaleAbs(col)

        def to_roi(local: np.ndarray) -> np.ndarray:
            return (_apply(aff, local) - off).astype(np.int32)

        # Crimson markings (mirrored pairs).
        for poly in _STRIPES:
            for sx in (1.0, -1.0):
                cv2.fillPoly(col, [to_roi(poly * (sx, 1.0))], CRIMSON, cv2.LINE_AA)

        # Eye holes: enlarged, outer corners swept upward; the real eye shows
        # through dimly (low alpha) so blinking still reads.
        inv = cv2.invertAffineTransform(aff)
        for (iris, _, _, _, _, _, contour) in _EYES:
            loc = _apply(inv, pts[contour, :2])
            ec = _apply(inv, pts[iris:iris + 1, :2])[0]
            d = loc - ec
            d[:, 0] *= 1.45
            d[:, 1] *= 2.1
            # Sweep the outer corner up: shift by a fraction of the hole height.
            outward = np.clip(np.sign(ec[0]) * d[:, 0], 0, None)
            hh = float(np.abs(d[:, 1]).max()) or 1e-3
            d[:, 1] -= (outward / (float(outward.max()) or 1e-3)) ** 2 * hh * 0.9
            hole = cv2.convexHull(to_roi(ec + d))
            cv2.fillPoly(col, [hole], HOLE, cv2.LINE_AA)
            cv2.fillPoly(a, [hole], 150, cv2.LINE_AA)
            cv2.polylines(col, [hole], True, (60, 70, 80), 2, cv2.LINE_AA)
            cv2.polylines(a, [hole], True, 255, 2, cv2.LINE_AA)

        # Toothy mouth: a dark slit across the lips with tooth marks.
        mouth = _apply(inv, ((pts[13, :2] + pts[14, :2]) / 2.0)[None])[0]
        gap = abs(float(_apply(inv, pts[14:15, :2])[0, 1] - _apply(inv, pts[13:14, :2])[0, 1]))
        half_w, tooth = 0.42, 0.06 + gap * 0.8
        xs = np.linspace(-half_w, half_w, 9)
        slit = np.stack([mouth[0] + xs, mouth[1] + 0.02 * (xs / half_w) ** 2], 1).astype(np.float32)
        thick = max(1, int(span * 0.03))
        cv2.polylines(col, [to_roi(slit)], False, HOLE, thick + 1, cv2.LINE_AA)
        for x in np.linspace(-half_w * 0.9, half_w * 0.9, 9):
            seg = np.array([[mouth[0] + x, mouth[1] - tooth], [mouth[0] + x, mouth[1] + tooth]], np.float32)
            cv2.polylines(col, [to_roi(seg)], False, (70, 80, 90), thick, cv2.LINE_AA)

        self._blend(roi, col, a, alpha * 0.94)

    def _blend_image(self, roi: np.ndarray, aff: np.ndarray, off: np.ndarray, alpha: float) -> None:
        img = self.mask_img
        m = self._image_affine(aff)
        m[:, 2] -= off
        rh, rw = roi.shape[:2]
        warped = cv2.warpAffine(img, m, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        self._blend(roi, warped[:, :, :3], warped[:, :, 3], alpha)

    def _image_affine(self, aff: np.ndarray) -> np.ndarray:
        """2x3 affine from mask-image pixels to render pixels."""
        ih, iw = self.mask_img.shape[:2]
        src = (_IMG_REF * (iw, ih)).astype(np.float32)
        dst = _apply(aff, np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, 1.0]], np.float32))
        return cv2.getAffineTransform(src, dst)

    @staticmethod
    def _blend(roi: np.ndarray, col: np.ndarray, a: np.ndarray, alpha: float) -> None:
        af = a.astype(np.float32) * (alpha / 255.0)
        af3 = cv2.merge([af, af, af])
        out = roi.astype(np.float32) * (1.0 - af3) + col.astype(np.float32) * af3
        np.copyto(roi, out.astype(np.uint8))
