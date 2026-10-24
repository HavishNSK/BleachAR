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

Reusable helpers (bottom of this module, see docs/RENDERING.md):

* ``energy_trail(canvas, pts, color, width, ...)`` - tapered ribbon with a
  white-hot core and glow along a polyline (``ribbon_polygon`` builds the
  outline).
* ``shockwave(canvas, center, radius, thickness, color, strength, warp)`` -
  ring + glow ring + optional ripple warp.
* ``lightning(a, b, branches, jitter, rng)`` -> list of polylines (main bolt
  first); ``draw_lightning(canvas, bolts, color)`` renders them.
* ``Afterimages`` - stores recent polygon sets, draws fading ghost copies.
* ``Starfield`` - vectorised parallax / zooming starfield.
* ``depth_scale(value, far, near)`` - size multiplier from a depth cue.

Performance notes: ``composite`` reuses its scratch buffers between frames
(no full-res allocations at a fixed size), only remaps the bounding box of
active warps, and skips bloom work for empty layers. ``canvas.quality``
(a config quality preset dict, see ``QUALITY_DEFAULTS``) can switch off
expensive features on slow machines; None (default) = everything on.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Sequence, Tuple

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


# Render-quality flags read by Canvas (``canvas.quality``); missing keys fall
# back to these, so ``quality = None`` (the default) means every feature on.
QUALITY_DEFAULTS = {
    "bloom": "dual",     # "dual" (wide + tight halo) or "single" (wide halo only)
    "warps": True,       # screen-space warps (add_warp) applied at all
    "max_warps": 0,      # keep only the N strongest warps per frame (0 = no limit)
    "aberration": True,  # chromatic aberration applied at all
    "warp_grid": 4,      # warp displacement evaluated every N render px (bigger = cheaper)
}


class Canvas:
    """Per-frame layers plus opt-in post effects (see module docstring).

    Opt-in extras, all free when unused and reset by ``begin()``:

    * ``backdrop()`` - a uint8 image the effect paints a replacement
      environment into; ``backdrop_amount`` (0..1) blends it over the camera
      image *behind the person* (person mask), so a Domain can fill the room
      while the user stays visible. Without a person mask it blends
      everywhere at ``backdrop_amount * backdrop_no_mask``.
    * ``add_warp(center, radius, strength, kind)`` - screen-space distortion
      of the camera/backdrop image, all warps applied in one remap.
      kinds: "pinch" (pull toward centre), "bulge" (push out), "swirl"
      (rotate by ``strength`` radians at the centre), "ripple" (a ring wave
      at ``radius``; ``width`` controls its thickness). The person is
      protected from warps when a mask is available (``warp_protect_person``).
    * ``aberration`` - radial chromatic aberration in pixels (0 = off).
    * ``flash`` / ``flash_color`` - full-screen flash (only raise ``flash``;
      the app decays it).
    * ``tint`` / ``tint_amount`` - colour grade of the camera image (BGR).
    * ``add_sprite(center, radius, color, strength, layer)`` - a soft radial
      gradient (cached) for bright energy cores, into "light" or "glow".
    * ``person_mask()`` - feathered, temporally smoothed float32 HxW mask of
      the user (1 = person) or None; set each frame with ``set_person_src``.
      The returned array is valid until the end of the *next* frame (two
      buffers are recycled); copy it if you need it for longer.

    ``quality`` (optional dict, default None = everything on) switches
    features off for slow machines; see ``QUALITY_DEFAULTS`` for the keys.
    The app sets it from the active config quality preset.
    """

    def __init__(self, glow_scale: float = 0.33, intensity: float = 1.0):
        self.glow_scale = glow_scale
        self.intensity = intensity
        self.frame: Optional[np.ndarray] = None
        self.light: Optional[np.ndarray] = None
        self.glow: Optional[np.ndarray] = None
        self.w = self.h = self.gw = self.gh = 0
        self.gs = glow_scale  # multiply render coords by this for the glow layer
        self.flash = 0.0  # 0..1 full-screen flash; the app decays it
        self.flash_color: Color = (255, 255, 255)
        self.dim = 0.0  # 0..1 darkening of the camera image this frame
        self.tint: Optional[Color] = None
        self.tint_amount = 0.0
        self.backdrop_amount = 0.0
        self.backdrop_no_mask = 0.75
        self.aberration = 0.0
        self.warp_protect_person = True
        self.bloom_wide = 1.4  # gains of the two bloom scales
        self.bloom_tight = 0.45
        self.quality: Optional[dict] = None  # render-quality flags, see QUALITY_DEFAULTS
        self._k = 1  # integer glow downscale factor
        self._white = None
        self._flash_buf_color = None
        self._glow_scale_seen = None
        self._backdrop: Optional[np.ndarray] = None
        self._backdrop_used = False
        self._warps: List[tuple] = []
        self._bufs: dict = {}  # reusable scratch buffers, keyed by name
        self._person_src: Optional[np.ndarray] = None
        self._person_ema: Optional[np.ndarray] = None
        self._person_full: Optional[np.ndarray] = None
        self._person_flip = 0
        self._person_ready = False
        self._person_inv_ready = False

    def q(self, key: str):
        """Value of a render-quality flag (``quality`` dict, else the default)."""
        qd = self.quality
        if qd is not None and key in qd:
            return qd[key]
        return QUALITY_DEFAULTS[key]

    def _buf(self, name: str, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Persistent scratch buffer; only reallocated when the shape changes."""
        b = self._bufs.get(name)
        if b is None or b.shape != shape or b.dtype != dtype:
            b = np.empty(shape, dtype)
            self._bufs[name] = b
        return b

    def _flat(self, name: str, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Contiguous view of a full-frame-sized scratch buffer, for ROI-sized
        work whose shape changes every frame (no reallocation)."""
        n = int(np.prod(shape))
        cap = self.h * self.w * (shape[2] if len(shape) > 2 else 1)
        b = self._bufs.get(name)
        if b is None or b.size < max(n, cap) or b.dtype != dtype:
            b = np.empty(max(n, cap), dtype)
            self._bufs[name] = b
        return b[:n].reshape(shape)

    def begin(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self.w, self.h) or self.glow is None or self._glow_scale_seen != self.glow_scale:
            self.w, self.h = w, h
            # Integer downscale factor: INTER_AREA is ~10x faster than with a
            # fractional one, and gs stays exact for coordinate mapping.
            k = max(1, int(round(1.0 / self.glow_scale)))
            self._k = k
            self.gw, self.gh = max(8, w // k), max(8, h // k)
            self.gs = self.gw / w
            self._glow_scale_seen = self.glow_scale
            self.light = np.zeros((h, w, 3), np.uint8)
            self.glow = np.zeros((self.gh, self.gw, 3), np.float32)
            self._white = None
            self._backdrop = None
            self._bufs = {}
            self._person_ema = None
        else:
            self.light.fill(0)
            self.glow.fill(0)
        self.frame = frame
        self.dim = 0.0
        self.tint = None
        self.tint_amount = 0.0
        self.backdrop_amount = 0.0
        self.aberration = 0.0
        self._backdrop_used = False
        self._warps.clear()
        self._person_ready = False
        self._person_inv_ready = False

    # --- person mask ------------------------------------------------------
    def set_person_src(self, mask: Optional[np.ndarray]) -> None:
        """Low-res float mask (1 = person) for this frame, any resolution."""
        self._person_src = mask if (mask is not None and mask.ndim == 2 and mask.size) else None

    def person_mask(self) -> Optional[np.ndarray]:
        if self._person_ready:
            return self._person_full
        self._person_ready = True
        src = self._person_src
        if src is None or self.w == 0:
            self._person_full = None
            self._person_ema = None
            return None
        gshape = (self.gh, self.gw)
        small = cv2.resize(src.astype(np.float32, copy=False), (self.gw, self.gh),
                           dst=self._buf("pm_small", gshape), interpolation=cv2.INTER_AREA)
        ema = self._person_ema
        if ema is not None and ema.shape == gshape:
            cv2.addWeighted(ema, 0.55, small, 0.45, 0, dst=ema)  # temporal smoothing
        else:
            ema = self._person_ema = self._buf("pm_ema", gshape)
            np.copyto(ema, small)
        feather = cv2.GaussianBlur(ema, (0, 0), 1.2, dst=self._buf("pm_feather", gshape))
        # Firm up the inside, keep a soft edge (avoids halos around hair/shoulders).
        np.subtract(feather, 0.25, out=feather)
        np.multiply(feather, 2.0, out=feather)
        np.clip(feather, 0.0, 1.0, out=feather)
        self._person_flip ^= 1  # two full-res buffers: last frame's mask stays valid
        full = self._buf("pm_full%d" % self._person_flip, (self.h, self.w))
        self._person_full = cv2.resize(feather, (self.w, self.h), dst=full, interpolation=cv2.INTER_LINEAR)
        return self._person_full

    def _person_inverse(self) -> Optional[np.ndarray]:
        """1 - person_mask(), computed at most once per frame."""
        pm = self.person_mask()
        if pm is None:
            return None
        inv = self._buf("pm_inv", pm.shape)
        if not self._person_inv_ready:
            np.subtract(1.0, pm, out=inv)
            self._person_inv_ready = True
        return inv

    # --- backdrop / warps -------------------------------------------------
    def backdrop(self) -> np.ndarray:
        """Replacement-environment buffer (uint8 BGR, render res) for this frame.

        Not cleared between frames (painters usually overwrite it fully)."""
        if self._backdrop is None or self._backdrop.shape[:2] != (self.h, self.w):
            self._backdrop = np.zeros((self.h, self.w, 3), np.uint8)
        self._backdrop_used = True
        return self._backdrop

    def add_warp(self, center, radius: float, strength: float, kind: str = "pinch", width: float = 0.25) -> None:
        if radius > 1.0 and abs(strength) > 1e-4 and np.all(np.isfinite(center)):
            self._warps.append((float(center[0]), float(center[1]), float(radius), float(strength), kind,
                                float(width)))

    @staticmethod
    def _warp_extent(rad: float, kind: str, width: float) -> float:
        """Radius beyond which a warp's displacement is negligible (< ~0.01 px)."""
        if kind == "ripple":
            return rad + 3.0 * max(4.0, rad * width)
        return rad

    def _active_warps(self) -> List[tuple]:
        warps = self._warps
        cap = int(self.q("max_warps") or 0)
        if 0 < cap < len(warps):
            warps = sorted(warps, key=lambda t: -abs(t[3]) * t[2])[:cap]
        return warps

    def _apply_warps(self, frame: np.ndarray) -> None:
        warps = self._active_warps()
        w, h = self.w, self.h
        step = max(2, int(self.q("warp_grid")))
        # Only the union of the warps' areas is remapped (outside it the
        # displacement is ~0, so the original would have copied pixels).
        ext = [(cx, cy, self._warp_extent(rad, kind, width)) for cx, cy, rad, st, kind, width in warps]
        x0 = max(0, int(math.floor(min(cx - e for cx, cy, e in ext))) - 2 * step)
        y0 = max(0, int(math.floor(min(cy - e for cx, cy, e in ext))) - 2 * step)
        x1 = min(w, int(math.ceil(max(cx + e for cx, cy, e in ext))) + 2 * step)
        y1 = min(h, int(math.ceil(max(cy + e for cx, cy, e in ext))) + 2 * step)
        rw, rh = x1 - x0, y1 - y0
        if rw < 2 or rh < 2:
            return
        # Displacement is computed on a coarse grid and upsampled: smooth fields
        # don't need per-pixel evaluation. Grid samples sit at the pixel
        # positions cv2.resize maps them to, so the upsampled field is aligned.
        cw, ch = max(2, -(-rw // step)), max(2, -(-rh // step))
        sx, sy = rw / cw, rh / ch
        X = (x0 + (np.arange(cw, dtype=np.float32) + 0.5) * sx - 0.5)
        Y = (y0 + (np.arange(ch, dtype=np.float32) + 0.5) * sy - 0.5)
        d = self._flat("warp_d", (ch, cw, 2))
        d.fill(0.0)
        for (cx, cy, rad, st, kind, width), (_, _, e) in zip(warps, ext):
            j0, j1 = np.searchsorted(X, cx - e), np.searchsorted(X, cx + e)
            i0, i1 = np.searchsorted(Y, cy - e), np.searchsorted(Y, cy + e)
            if j1 <= j0 or i1 <= i0:
                continue
            ox = (X[j0:j1] - cx)[None, :]
            oy = (Y[i0:i1] - cy)[:, None]
            r = np.sqrt(ox * ox + oy * oy) + 1e-3
            dx = d[i0:i1, j0:j1, 0]
            dy = d[i0:i1, j0:j1, 1]
            if kind == "ripple":
                band = max(4.0, rad * width)
                env = np.exp(-((r - rad) / band) ** 2)
                dr = st * band * env / r
                dx += ox * dr
                dy += oy * dr
                continue
            f = np.clip(1.0 - (r / rad) ** 2, 0.0, 1.0) ** 2
            if kind == "swirl":
                a = st * f
                ca, sa = np.cos(a), np.sin(a)
                dx += ox * ca - oy * sa - ox
                dy += ox * sa + oy * ca - oy
            else:  # pinch pulls the image toward the centre (sample from further out); bulge pushes
                k = st * f * (1.0 if kind == "pinch" else -1.0)
                dx += ox * k
                dy += oy * k
        # One 2-channel (x, y) map: half the resize/add work of separate maps.
        m = cv2.resize(d, (rw, rh), dst=self._flat("warp_map", (rh, rw, 2)), interpolation=cv2.INTER_LINEAR)
        base = self._bufs.get("warp_base")
        if base is None or base.shape[:2] != (h, w):
            base = np.dstack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
            self._bufs["warp_base"] = base
        cv2.add(m, base[y0:y1, x0:x1], dst=m)
        warped = cv2.remap(frame, m, None, cv2.INTER_LINEAR, dst=self._flat("warp_out", (rh, rw, 3), np.uint8),
                           borderMode=cv2.BORDER_REFLECT)
        roi = frame[y0:y1, x0:x1]
        pm = self.person_mask() if self.warp_protect_person else None
        if pm is not None:
            inv = self._person_inverse()
            cv2.blendLinear(warped, roi, inv[y0:y1, x0:x1], pm[y0:y1, x0:x1], dst=roi)
        else:
            np.copyto(roi, warped)

    def _apply_backdrop(self, frame: np.ndarray) -> None:
        a = float(np.clip(self.backdrop_amount, 0.0, 1.0))
        pm = self.person_mask()
        if pm is None:
            cv2.addWeighted(frame, 1.0 - a * self.backdrop_no_mask, self._backdrop, a * self.backdrop_no_mask, 0,
                            dst=frame)
            return
        inv = self._person_inverse()
        if a >= 1.0:
            wb, wf = inv, pm
        else:
            wb = np.multiply(inv, a, out=self._buf("bd_wb", inv.shape))
            wf = np.subtract(1.0, wb, out=self._buf("bd_wf", inv.shape))
        cv2.blendLinear(self._backdrop, frame, wb, wf, dst=frame)

    # --- sprites ------------------------------------------------------------
    def add_sprite(self, center, radius: float, color: Sequence[float], strength: float = 1.0,
                   layer: str = "light", falloff: float = 2.0) -> None:
        """Soft radial gradient (bright centre, smooth falloff) - energy cores."""
        if not np.all(np.isfinite(center)) or radius < 1:
            return
        k = strength * self.intensity
        if layer == "glow":
            cx, cy, r = center[0] * self.gs, center[1] * self.gs, max(2.0, radius * self.gs)
            dst = self.glow
        else:
            cx, cy, r = center[0], center[1], radius
            dst = self.light
        spr = radial_sprite(int(round(r)), falloff)
        R = spr.shape[0] // 2
        x0, y0 = int(round(cx)) - R, int(round(cy)) - R
        hh, ww = dst.shape[:2]
        ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(ww, x0 + 2 * R + 1), min(hh, y0 + 2 * R + 1)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        s = spr[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None] * np.asarray(color, np.float32)[None, None, :] * k
        if layer == "glow":
            dst[ay0:ay1, ax0:ax1] += s
        else:
            roi = dst[ay0:ay1, ax0:ax1]
            cv2.add(roi, s.astype(np.uint8), dst=roi)

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

    # --- compositing --------------------------------------------------------
    def composite(self, bloom: float = 1.0) -> np.ndarray:
        frame = self.frame
        if self._backdrop_used and self.backdrop_amount > 0.002:
            self._apply_backdrop(frame)
        if self._warps and self.q("warps"):
            self._apply_warps(frame)
        if self.tint is not None and self.tint_amount > 0.002:
            a = float(np.clip(self.tint_amount, 0.0, 1.0))
            # frame * (1 - a) + tint * a, without building a full-frame colour image.
            cv2.convertScaleAbs(frame, frame, alpha=1.0 - a)
            t = [int(c) * a for c in self.tint]
            cv2.add(frame, (t[0], t[1], t[2], 0.0), dst=frame)
        if self.dim > 0.0:
            cv2.convertScaleAbs(frame, frame, alpha=max(0.0, 1.0 - self.dim))
        cv2.add(frame, self.light, dst=frame)
        self._bloom(frame, bloom)
        if self.aberration > 0.3 and self.q("aberration"):
            self._apply_aberration(frame)
        if self.flash > 0.01:
            col = tuple(int(c) for c in self.flash_color)
            if self._white is None or self._flash_buf_color != col:
                self._white = np.empty((self.h, self.w, 3), np.uint8)
                self._white[:] = col
                self._flash_buf_color = col
            cv2.addWeighted(frame, 1.0 - min(1.0, self.flash), self._white, min(1.0, self.flash), 0, dst=frame)
        return frame

    def _bloom(self, frame: np.ndarray, bloom: float) -> None:
        # Bloom: the crisp layer bleeds into the soft glow layer, which is
        # blurred at two scales - a tight halo around bright cores and a wide,
        # low soft bloom - then highlight-compressed instead of hard clipped.
        gw, gh, k = self.gw, self.gh, self._k
        glow = self.glow
        gshape = (gh, gw, 3)
        light = self.light
        if cv2.hasNonZero(light.reshape(self.h, -1)):
            # Crop to an exact multiple of k so INTER_AREA takes its fast
            # integer path (a fractional ratio is ~5x slower at 1280x720).
            src = light[:gh * k, :gw * k] if (gh * k <= self.h and gw * k <= self.w) else light
            small = cv2.resize(src, (gw, gh), dst=self._buf("bl_small", gshape, np.uint8),
                               interpolation=cv2.INTER_AREA)
            small_f = self._buf("bl_small_f", gshape)
            np.copyto(small_f, small)
            cv2.scaleAdd(small_f, 0.6 * bloom, glow, dst=glow)
        flat = glow.reshape(gh, -1)
        if cv2.minMaxLoc(flat)[1] <= 0.0:
            return  # nothing glows this frame
        # Two box-blur passes of width k give sigma^2 = (k^2 - 1) / 6; the wide
        # pass matches a Gaussian of sigma = 27 render px, the tight one ~8 px.
        kw = max(3, int(27 * self.gs * 2.449) | 1)
        kt = max(3, int(8 * self.gs * 2.449) | 1)
        wide = self._buf("bl_wide", gshape)
        tmp = self._buf("bl_tmp", gshape)
        cv2.blur(glow, (kw, kw), dst=tmp)
        cv2.blur(tmp, (kw, kw), dst=wide)
        if self.q("bloom") == "single":
            np.multiply(wide, self.bloom_wide * bloom, out=wide)
        else:
            cv2.blur(glow, (kt, kt), dst=tmp)
            cv2.addWeighted(wide, self.bloom_wide * bloom, tmp, self.bloom_tight * bloom, 0, dst=wide)
        wflat = wide.reshape(gh, -1)
        if cv2.minMaxLoc(wflat)[1] > 170.0:
            # Soft knee: linear up to 170, then compressed toward 255:
            # min(x, 255 - 85 * exp(-(max(x, 170) - 170) / 85)).
            t = tmp.reshape(gh, -1)
            cv2.max(wflat, 170.0, dst=t)
            cv2.subtract(t, 170.0, dst=t)
            np.multiply(t, -1.0 / 85.0, out=t)
            cv2.exp(t, dst=t)
            np.multiply(t, -85.0, out=t)
            np.add(t, 255.0, out=t)
            cv2.min(wflat, t, dst=wflat)
        glow_u8 = self._buf("bl_u8", gshape, np.uint8)
        np.copyto(glow_u8, wide, casting="unsafe")  # truncates like astype(np.uint8)
        up = cv2.resize(glow_u8, (self.w, self.h), dst=self._buf("bl_up", (self.h, self.w, 3), np.uint8),
                        interpolation=cv2.INTER_LINEAR)
        cv2.add(frame, up, dst=frame)

    def _apply_aberration(self, frame: np.ndarray) -> None:
        a = float(min(self.aberration, 12.0))
        cx, cy = self.w * 0.5, self.h * 0.5
        plane = (self.h, self.w)
        # Planar copies are much faster to warp than strided channel views.
        chans = cv2.split(frame, [self._buf("ab_%d" % i, plane, np.uint8) for i in range(3)])
        out = list(chans)
        for ch, sgn in ((0, -1.0), (2, 1.0)):
            s = 1.0 + sgn * a / max(1.0, self.w * 0.5)
            m = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], np.float32)
            out[ch] = cv2.warpAffine(chans[ch], m, (self.w, self.h), dst=self._buf("ab_o%d" % ch, plane, np.uint8),
                                     borderMode=cv2.BORDER_REPLICATE)
        cv2.merge(out, dst=frame)


_SPRITES: dict = {}


def radial_sprite(radius: int, falloff: float = 2.0) -> np.ndarray:
    """Cached (2r+1)^2 float32 radial gradient, 1 at the centre, 0 at the edge."""
    radius = int(max(1, min(radius, 400)))
    key = (radius, round(falloff, 2))
    spr = _SPRITES.get(key)
    if spr is None:
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
        r = np.sqrt(x * x + y * y) / radius
        spr = np.clip(1.0 - r, 0.0, 1.0) ** falloff
        if len(_SPRITES) > 256:
            _SPRITES.clear()
        _SPRITES[key] = spr
    return spr


_NOISE: dict = {}


def noise_texture(w: int, h: int, cells: int = 8, seed: int = 0, octaves: int = 3) -> np.ndarray:
    """Cached tileable value noise in [0, 1], float32 (h, w)."""
    key = (w, h, cells, seed, octaves)
    tex = _NOISE.get(key)
    if tex is None:
        rng = np.random.default_rng(seed)
        tex = np.zeros((h, w), np.float32)
        amp, total = 1.0, 0.0
        for o in range(octaves):
            c = cells * (2 ** o)
            grid = rng.random((c, c)).astype(np.float32)
            grid = np.vstack([grid, grid[:1]])
            grid = np.hstack([grid, grid[:, :1]])
            layer = cv2.resize(grid, (w + w // c, h + h // c), interpolation=cv2.INTER_CUBIC)[:h, :w]
            tex += amp * layer
            total += amp
            amp *= 0.5
        tex = np.clip(tex / total, 0.0, 1.0)
        _NOISE[key] = tex
    return tex


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
        pos_arr = np.asarray(pos, np.float32)
        vel_arr = np.asarray(vel, np.float32)
        # Per-particle arrays may be sized for the unclamped n; truncate to match.
        if pos_arr.ndim == 2:
            n = min(n, int(pos_arr.shape[0]))
            pos_arr = pos_arr[:n]
        if vel_arr.ndim == 2:
            n = min(n, int(vel_arr.shape[0]))
            vel_arr = vel_arr[:n]
        if n <= 0:
            return
        idx = (self._cursor + np.arange(n)) % self.capacity  # ring buffer; oldest are overwritten
        self._cursor = int((self._cursor + n) % self.capacity)
        p = np.broadcast_to(pos_arr, (n, 2))
        v = np.broadcast_to(vel_arr, (n, 2))
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


# --- reusable graphics helpers ------------------------------------------------
# Small building blocks shared by effects (see docs/RENDERING.md). They only
# draw through the public Canvas API, so they follow its quality flags.


def depth_scale(value, far: float, near: float, s_far: float = 0.6, s_near: float = 1.4):
    """Size multiplier from a depth cue, linear between ``far`` and ``near``.

    ``value`` can be a screen y (lower on screen = closer: far=horizon_y,
    near=frame_h), a MediaPipe landmark z (more negative = closer: far=0.1*w,
    near=-0.3*w), a shoulder width in px, ... Clamped to [s_far, s_near]
    outside the range. Scalars return a float, arrays an array.
    """
    span = near - far
    t = (np.asarray(value, np.float32) - far) / (span if abs(span) > 1e-6 else 1e-6)
    out = s_far + (s_near - s_far) * np.clip(t, 0.0, 1.0)
    return float(out) if np.ndim(out) == 0 else out


def ribbon_polygon(pts, widths) -> np.ndarray:
    """Closed polygon of a ribbon of per-point ``widths`` (full width) along a polyline.

    Normals are averaged at interior points so joints don't kink. Returns a
    float32 (2N, 2) array (one side, then the other reversed); empty (0, 2)
    for fewer than two usable points.
    """
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    wd = np.broadcast_to(np.asarray(widths, np.float32), (len(p),))
    if len(p) >= 2:
        keep = np.ones(len(p), bool)
        keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-3  # drop duplicate points
        p, wd = p[keep], wd[keep]
    if len(p) < 2:
        return np.zeros((0, 2), np.float32)
    seg = np.diff(p, axis=0)
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-6)
    tang = np.empty_like(p)
    tang[0], tang[-1] = seg[0], seg[-1]
    tang[1:-1] = seg[:-1] + seg[1:]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1) * (wd[:, None] * 0.5)
    return np.concatenate([p + nrm, (p - nrm)[::-1]], 0).astype(np.float32)


def energy_trail(canvas: Canvas, pts, color: Sequence[float], width: float, strength: float = 1.0,
                 core_color: Sequence[float] = (255, 255, 255), taper: float = 1.0,
                 glow_width: float = 2.5) -> None:
    """Tapered energy ribbon along ``pts`` (tail first, head last).

    Draws a coloured body and a thin white-hot core on the light layer plus a
    wide soft halo on the glow layer. ``taper`` 0 = constant width, 1 = from a
    point at the tail to ``width`` at the head (sqrt profile, fuller near the
    head).
    """
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(p) < 2 or strength <= 0 or not np.isfinite(p).all():
        return
    t = np.linspace(0.0, 1.0, len(p), dtype=np.float32)
    prof = (1.0 - taper) + taper * np.sqrt(t)
    body = ribbon_polygon(p, width * prof)
    if len(body) == 0:
        return
    canvas.glow_poly(ribbon_polygon(p, width * glow_width * prof), color, strength * 0.8)
    canvas.light_poly(body, color, strength * 0.85)
    canvas.light_poly(ribbon_polygon(p, width * 0.35 * prof), core_color, strength)


def shockwave(canvas: Canvas, center, radius: float, thickness: float, color: Sequence[float],
              strength: float = 1.0, warp: float = 0.0) -> None:
    """Expanding-ring impact: crisp ring + soft glow ring, plus an optional
    ripple warp of the camera image (``warp`` ~0.3..1.5; 0 = no distortion)."""
    if radius < 1 or strength <= 0 or not np.all(np.isfinite(center)):
        return
    ring(canvas, center, radius, color, max(1, int(round(thickness))), strength)
    if warp:
        canvas.add_warp(center, radius, warp, "ripple", width=max(0.05, thickness * 1.5 / radius))


def lightning(a, b, branches: int = 2, jitter: float = 0.18, rng: Optional[np.random.Generator] = None,
              segments: int = 12, branch_length: float = 0.45, depth: int = 1) -> List[np.ndarray]:
    """Branching lightning bolt from ``a`` to ``b`` by midpoint displacement.

    Returns a list of float32 (N, 2) polylines: the main bolt first, then
    branches (each forking off the main bolt, ``branch_length`` times as long,
    recursing ``depth`` levels). ``jitter`` is relative to the bolt length.
    Pass a seeded ``rng`` for a stable shape (e.g. re-seed every few frames
    so the bolt flickers at a chosen rate rather than every frame).
    """
    rng = rng if rng is not None else np.random.default_rng()
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    length = float(np.linalg.norm(b - a))
    if length < 1e-3:
        return [np.stack([a, b]).astype(np.float32)]
    # Midpoint displacement: each level halves the offset (1/f-ish detail).
    pts = np.stack([a, b])
    amp = jitter * length
    n_side = perp(unit(b - a))
    while len(pts) - 1 < segments:
        mids = (pts[:-1] + pts[1:]) * 0.5 + n_side * rng.uniform(-amp, amp, (len(pts) - 1, 1)).astype(np.float32)
        out = np.empty((len(pts) * 2 - 1, 2), np.float32)
        out[0::2], out[1::2] = pts, mids
        pts = out
        amp *= 0.5
    bolts = [pts.astype(np.float32)]
    if depth > 0 and branches > 0:
        dirn = unit(b - a)
        for _ in range(int(branches)):
            i = int(rng.integers(1, max(2, len(pts) - 1)))
            start = pts[i]
            ang = rng.uniform(0.35, 0.9) * rng.choice([-1.0, 1.0])
            ca, sa = math.cos(ang), math.sin(ang)
            d = np.array([dirn[0] * ca - dirn[1] * sa, dirn[0] * sa + dirn[1] * ca], np.float32)
            end = start + d * length * branch_length * rng.uniform(0.6, 1.0)
            bolts += lightning(start, end, branches=max(0, branches - 1), jitter=jitter, rng=rng,
                               segments=max(4, segments // 2), branch_length=branch_length, depth=depth - 1)
    return bolts


def draw_lightning(canvas: Canvas, bolts: Sequence[np.ndarray], color: Sequence[float], thickness: int = 2,
                   strength: float = 1.0, core_color: Sequence[float] = (255, 255, 255)) -> None:
    """Draw ``lightning()`` polylines: white core + coloured edge + glow; branches thinner."""
    for i, pl in enumerate(bolts):
        p = np.asarray(pl, np.float32)
        if len(p) < 2 or not np.isfinite(p).all():
            continue
        k = strength * (1.0 if i == 0 else 0.6)
        th = max(1, int(round(thickness * (1.0 if i == 0 else 0.6))))
        cv2.polylines(canvas.light, [p.astype(np.int32)], False, scale_color(color, k * canvas.intensity),
                      th + 2, cv2.LINE_AA)
        cv2.polylines(canvas.light, [p.astype(np.int32)], False, scale_color(core_color, k * canvas.intensity),
                      th, cv2.LINE_AA)
        g = k * canvas.intensity
        cv2.polylines(canvas.glow, [(p * canvas.gs).astype(np.int32)], False,
                      (color[0] * g, color[1] * g, color[2] * g), max(1, int((th + 6) * canvas.gs * 2)), cv2.LINE_AA)


class Afterimages:
    """Fading ghost copies of a shape (speed afterimages, flash-step trails).

    Call ``add(now, polys)`` every frame with the current polygon(s); a
    snapshot is stored at most every ``interval`` seconds (up to ``count``).
    ``render`` draws each snapshot with brightness falling off with age
    (oldest dimmest, gone after ``duration``).
    """

    def __init__(self, count: int = 5, interval: float = 0.035, duration: float = 0.25):
        self.interval = interval
        self.duration = duration
        self.snaps: deque = deque(maxlen=max(1, int(count)))

    def add(self, now: float, polys) -> None:
        if self.snaps and now - self.snaps[-1][0] < self.interval:
            return
        if isinstance(polys, np.ndarray) and polys.ndim == 2:
            polys = [polys]
        snap = [np.asarray(p, np.float32).copy() for p in polys if len(p) >= 3]
        if snap:
            self.snaps.append((now, snap))

    def clear(self) -> None:
        self.snaps.clear()

    def render(self, canvas: Canvas, now: float, color: Sequence[float], strength: float = 1.0,
               skip_latest: bool = True, glow: bool = True) -> None:
        """Draw the ghosts. ``skip_latest`` leaves out the newest snapshot
        (usually drawn crisply by the effect itself)."""
        while self.snaps and now - self.snaps[0][0] > self.duration:
            self.snaps.popleft()
        snaps = list(self.snaps)[:-1] if (skip_latest and self.snaps) else list(self.snaps)
        for t, polys in snaps:
            age = 1.0 - (now - t) / self.duration
            if age <= 0:
                continue
            for p in polys:
                canvas.light_poly(p, color, strength * 0.5 * age * age)
                if glow:
                    canvas.glow_poly(p, color, strength * 0.6 * age)


class Starfield:
    """Vectorised parallax starfield for Domain backdrops / void effects.

    Stars live in normalised [0, 1) screen space with a depth in (0, 1]
    (1 = nearest). ``update(dt, drift=(vx, vy), zoom=z)`` moves them: drift is
    a screen velocity (fractions of the frame per second) scaled by depth for
    parallax; ``zoom`` > 0 streams stars outward from ``center`` (hyperspace),
    near stars faster. ``render(canvas, ...)`` draws onto any uint8 BGR image
    (e.g. ``canvas.backdrop()``) or the light layer, with twinkle.
    """

    def __init__(self, n: int = 300, layers: int = 3, seed: int = 0, twinkle: float = 0.35):
        self.rng = np.random.default_rng(seed)
        self.n = int(n)
        self.pos = self.rng.random((self.n, 2)).astype(np.float32)
        layer = self.rng.integers(0, max(1, layers), self.n)
        self.depth = ((layer + 1) / max(1, layers) * self.rng.uniform(0.8, 1.0, self.n)).astype(np.float32)
        self.phase = self.rng.uniform(0, 2 * math.pi, self.n).astype(np.float32)
        self.freq = self.rng.uniform(1.5, 5.0, self.n).astype(np.float32)
        self.twinkle = twinkle
        self.t = 0.0

    def update(self, dt: float, drift=(0.0, 0.0), zoom: float = 0.0, center=(0.5, 0.5)) -> None:
        self.t += dt
        d = self.depth[:, None]
        self.pos += np.asarray(drift, np.float32)[None, :] * d * dt
        if zoom:
            c = np.asarray(center, np.float32)
            off = self.pos - c
            self.pos += off * (zoom * dt) * (0.3 + d)
            far = np.abs(self.pos - c).max(axis=1) > 0.75
            if far.any():  # respawn near the centre
                k = int(far.sum())
                self.pos[far] = c + self.rng.uniform(-0.15, 0.15, (k, 2)).astype(np.float32)
        np.mod(self.pos, 1.0, out=self.pos)

    def brightness(self) -> np.ndarray:
        tw = 1.0 - self.twinkle * (0.5 + 0.5 * np.sin(self.t * self.freq + self.phase))
        return (0.25 + 0.75 * self.depth) * tw

    def render(self, canvas: Canvas, color: Sequence[float] = (255, 240, 230), strength: float = 1.0,
               target: Optional[np.ndarray] = None, glow: bool = True) -> None:
        """Stars as pixels (far) and small discs (near) on ``target``
        (default ``canvas.light``), plus a soft glow splat for near stars."""
        img = canvas.light if target is None else target
        h, w = img.shape[:2]
        b = self.brightness() * strength * canvas.intensity
        xs = (self.pos[:, 0] * w).astype(np.int32) % w
        ys = (self.pos[:, 1] * h).astype(np.int32) % h
        col = np.clip(np.asarray(color, np.float32)[None, :] * b[:, None], 0, 255).astype(np.uint8)
        np.maximum.at(img, (ys, xs), col)  # far stars: single pixels, vectorised
        near = np.nonzero(self.depth > 0.72)[0]
        for i in near:
            c = col[i]
            cv2.circle(img, (int(xs[i]), int(ys[i])), 1, (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
        if glow and near.size and canvas.glow is not None:
            gx = np.clip((xs[near] * canvas.gs).astype(np.int32), 0, canvas.gw - 1)
            gy = np.clip((ys[near] * canvas.gs).astype(np.int32), 0, canvas.gh - 1)
            np.add.at(canvas.glow, (gy, gx), col[near].astype(np.float32) * 0.8)
