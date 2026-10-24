"""Render the bundled sword and mask art (original designs) into ``assets/``.

    python tools/make_assets.py            # writes assets/*.png
    python tools/make_assets.py --out DIR

Everything is drawn offline with numpy/OpenCV: per-pixel shading at 4x
supersampling, then an area downscale for anti-aliasing. No external art.

Sword images follow the convention used by ``bleach_ar.sword``: RGBA, blade
pointing straight up, square pixels, the image spans blade-local u from
``IMG_U0`` (bottom, pommel) to ``IMG_U1`` (top, tip) and v = 0 is the
horizontal centre. The blade outlines come from the same profile functions
that position the energy-edge glow, so art and glow line up.

The mask matches ``bleach_ar.eyes._IMG_REF``: eyes at (0.32, 0.42) and
(0.68, 0.42) of the image size, chin at (0.5, 0.95).
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bleach_ar.sword import (CLEAVER_BACK_V, CLEAVER_BLADE, CLEAVER_EDGE_V, IMG_U0,  # noqa: E402
                             IMG_U1, dark_profile)

SS = 4  # supersampling factor
SWORD_H = 768  # final image height (px) for the 1.2-unit span
RNG = np.random.default_rng(7)


# --- helpers ------------------------------------------------------------------
def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fbm(shape, cell: float, octaves: int = 4, seed: int = 0, aniso=(1.0, 1.0)) -> np.ndarray:
    """Smooth value-noise fBm in [-1, 1]. ``aniso`` stretches cells (y, x)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cy = max(1.0, cell * aniso[0] / 2 ** o)
        cx = max(1.0, cell * aniso[1] / 2 ** o)
        gh, gw = int(h / cy) + 3, int(w / cx) + 3
        g = rng.random((gh, gw)).astype(np.float32) * 2 - 1
        big = cv2.resize(g, (int(gw * cx), int(gh * cy)), interpolation=cv2.INTER_CUBIC)
        oy, ox = rng.integers(0, max(1, int(cy))), rng.integers(0, max(1, int(cx)))
        out += amp * big[oy:oy + h, ox:ox + w]
        total += amp
        amp *= 0.5
    return out / total


class Img:
    """Premultiplied float canvas at supersampled resolution."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w * SS, h * SS
        self.P = np.zeros((self.h, self.w, 3), np.float32)
        self.A = np.zeros((self.h, self.w), np.float32)

    def over(self, color, alpha) -> None:
        a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        c = np.asarray(color, np.float32)
        if c.ndim == 1:
            c = np.broadcast_to(c, self.P.shape)
        self.P = self.P * (1 - a[..., None]) + c * a[..., None]
        self.A = self.A * (1 - a) + a

    def save(self, path: str) -> None:
        fw, fh = self.w // SS, self.h // SS
        P = cv2.resize(self.P, (fw, fh), interpolation=cv2.INTER_AREA)
        A = cv2.resize(self.A, (fw, fh), interpolation=cv2.INTER_AREA)
        col = P / np.maximum(A[..., None], 1e-4)
        out = np.dstack([np.clip(col * 255 + 0.5, 0, 255), np.clip(A * 255 + 0.5, 0, 255)]).astype(np.uint8)
        out[A < 1.0 / 255] = 0
        cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def outline(mask: np.ndarray, px: float = 1.0) -> np.ndarray:
    """Band of ~``px`` final pixels just inside a boolean mask."""
    k = max(1, int(round(px * SS)))
    er = cv2.erode(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    return (mask & (er == 0)).astype(np.float32)


def uv_grid(w: int, h: int):
    ppu = h / (IMG_U1 - IMG_U0)
    ys, xs = np.mgrid[0:h * SS, 0:w * SS].astype(np.float32)
    xs = (xs + 0.5) / SS
    ys = (ys + 0.5) / SS
    v = (xs - w / 2.0) / ppu
    u = IMG_U1 - ys / ppu
    return u, v, ppu


def poly_mask(img: Img, uv_poly: np.ndarray, ppu: float, w: int) -> np.ndarray:
    pts = np.stack([(w / 2.0 + uv_poly[:, 1] * ppu) * SS, (IMG_U1 - uv_poly[:, 0]) * ppu * SS], 1)
    m = np.zeros((img.h, img.w), np.uint8)
    cv2.fillPoly(m, [np.round(pts * 4).astype(np.int32)], 1, cv2.LINE_8, shift=2)
    return m.astype(bool)


def cyl(v, hw, light=-0.45):
    """Cylinder lighting across a handle of half width ``hw`` (light from screen-left)."""
    n = np.clip(v / hw, -1, 1)
    nz = np.sqrt(np.clip(1 - n * n, 0, 1))
    return np.clip(0.35 + 0.65 * (nz * 0.85 + (-n) * light * -1.0 * 0.5), 0, 1.3)


def wrap_handle(img: Img, u, v, u0, u1, hw_fn, n_win, ito, base_fn, seed, win_scale=(0.42, 0.66)):
    """Tsuka-style wrap: dark cord bands crossing over a light base with diamond windows."""
    hw = hw_fn(u)
    inside = (u >= u0) & (u <= u1) & (np.abs(v) <= hw)
    P = (u1 - u0) / n_win
    k = np.floor((u - u0) / P)
    uc = u0 + (k + 0.5) * P
    du = (u - uc) / (P * win_scale[0])
    dv = v / (hw * win_scale[1])
    win = inside & (np.abs(du) + np.abs(dv) < 1.0)
    shade = cyl(v, hw)
    # Base (samegawa / under-layer) visible through the windows.
    base = base_fn(u, v, hw)
    img.over(base * shade[..., None], win.astype(np.float32))
    # Hishigami: crisp white folded paper at the side corners of each window.
    hishi = win & (np.abs(dv) > 0.55) & (np.abs(du) < (1 - np.abs(dv)) * 0.9)
    img.over(np.array([0.93, 0.95, 0.96]) * shade[..., None], hishi.astype(np.float32) * 0.9)
    # Cord bands: diagonal weave whose direction flips at each crossing.
    cord = inside & ~win
    side = np.sign(v + 1e-6) * np.sign(u - (u0 + np.round((u - u0) / P) * P) + 1e-6)
    weave = np.sin(((u * 1.0 + side * v * 0.9) / (P * 0.09)) * np.pi)
    tex = 0.88 + 0.12 * weave + 0.08 * fbm(u.shape, 6 * SS, 3, seed)
    # Darken toward window borders (cord rolls under) and give each band a soft highlight.
    edge_d = np.clip((np.abs(du) + np.abs(dv) - 1.0) * 2.5, 0, 1)
    roll = 0.55 + 0.45 * sstep(0.0, 0.6, edge_d)
    # Crossings: the over-band's edge is a diagonal crease; the two bands differ in brightness.
    rel = (u - u0) / P - np.round((u - u0) / P)  # -0.5..0.5, 0 at a crossing
    crease = (np.abs(rel * P - 0.55 * v) < 0.0024) & (np.abs(v) < hw * 0.75)
    twist = np.where(crease, 0.35, 1.0) * (1.0 + 0.22 * side)
    # Soft highlight down the middle of each band (cords are round).
    band_c = np.exp(-((np.abs(du) + np.abs(dv) - 1.35) / 0.3) ** 2)
    col = np.asarray(ito, np.float32) * (tex * roll * twist * shade * (0.85 + 0.5 * band_c))[..., None]
    img.over(col, cord.astype(np.float32))
    return inside


# --- katana -------------------------------------------------------------------
def make_katana(path: str) -> None:
    W, H = 96, SWORD_H
    img = Img(W, H)
    u, v, ppu = uv_grid(W, H)
    shp = u.shape

    # Tsuka (handle): u -0.18 .. 0.077.
    def hw_tsuka(uu):
        return 0.0205 - 0.0016 * np.sin(np.clip((uu + 0.18) / 0.257, 0, 1) * np.pi)

    sme_noise = fbm(shp, 1.2 * SS, 2, 11)

    def samegawa(uu, vv, hw):
        nod = np.clip(sme_noise * 2.2, -1, 1)
        c = np.array([0.86, 0.90, 0.91], np.float32) * (0.9 + 0.1 * nod)[..., None]
        return c

    wrap_handle(img, u, v, -0.18, 0.077, hw_tsuka, 7, (0.16, 0.08, 0.07), samegawa, 3)
    hmask = (u >= -0.18) & (u <= 0.077) & (np.abs(v) <= hw_tsuka(u))
    img.over(np.array([0.05, 0.04, 0.05]), outline(hmask, 1.0) * 0.8)

    # Menuki: small gold ornament under the wrap on the centre line.
    men = ((u + 0.05) / 0.016) ** 2 + (v / 0.009) ** 2 < 1
    mshade = cyl(v, 0.012)
    img.over(np.array([0.22, 0.58, 0.80]) * mshade[..., None] * (0.85 + 0.15 * np.sin(u * 900))[..., None],
             men.astype(np.float32) * 0.92)

    # Kashira (pommel cap): u -0.199 .. -0.178, rounded bottom.
    hwk = 0.0215
    kas = (u <= -0.176) & (np.abs(v) <= hwk) & (u >= -0.197 + 0.004 * (v / hwk) ** 2)
    ks = cyl(v, hwk)
    kcol = np.array([0.20, 0.19, 0.21]) * ks[..., None]
    kcol = kcol + np.array([0.25, 0.25, 0.27]) * np.exp(-((v + 0.007) / 0.004) ** 2)[..., None]
    img.over(kcol, kas.astype(np.float32))
    band = kas & (u > -0.181)
    img.over(np.array([0.25, 0.62, 0.82]) * ks[..., None], band.astype(np.float32))
    img.over(np.array([0.04, 0.04, 0.05]), outline(kas, 1.0) * 0.9)

    # Fuchi (collar) u 0.076 .. 0.0855.
    fu = (u >= 0.0765) & (u <= 0.0858) & (np.abs(v) <= 0.0215)
    fs = cyl(v, 0.0215)
    img.over(np.array([0.17, 0.16, 0.18]) * fs[..., None], fu.astype(np.float32))
    img.over(np.array([0.30, 0.66, 0.86]) * fs[..., None], (fu & (u > 0.0835)).astype(np.float32))
    img.over(np.array([0.04, 0.04, 0.05]), outline(fu, 0.8) * 0.9)

    # Tsuba: dark iron oval seen slightly from above, with a visible rim thickness.
    tc, ta, tb = 0.0935, 0.0118, 0.054
    eu, ev = (u - tc) / ta, v / tb
    r = np.sqrt(eu ** 2 + ev ** 2)
    side = (((u - tc + 0.0045) / ta) ** 2 + ev ** 2 < 1) & ~(r < 1)
    img.over(np.array([0.08, 0.08, 0.09]) * (0.7 + 0.3 * np.abs(ev))[..., None], side.astype(np.float32))
    face = r < 1
    ang = np.arctan2(eu, ev)
    iron = 0.8 + 0.2 * fbm(shp, 3 * SS, 3, 5)
    light = 0.75 + 0.35 * np.clip(eu, -1, 1)  # brighter toward the top (blade side)
    tcol = np.array([0.15, 0.15, 0.17]) * (iron * light)[..., None]
    img.over(tcol, face.astype(np.float32))
    # Engraving: inner ring + eight petal spokes, slightly raised (light top, dark bottom).
    ring = face & (np.abs(r - 0.68) < 0.05)
    spokes = face & (r > 0.3) & (r < 0.68) & (np.abs(np.sin(ang * 4)) < 0.10)
    petals = face & (r > 0.72) & (r < 0.9) & (np.cos(ang * 8) > 0.6)
    img.over(np.array([0.32, 0.33, 0.36]), (ring | spokes).astype(np.float32) * 0.8)
    img.over(np.array([0.26, 0.55, 0.70]), petals.astype(np.float32) * 0.55)
    rim = face & (r > 0.92)
    img.over(np.array([0.36, 0.38, 0.42]) * light[..., None], rim.astype(np.float32))
    img.over(np.array([0.03, 0.03, 0.04]), outline(face | side, 0.9))

    # Seppa washers either side of the tsuba (only the strip beside the blade shows).
    for su in ((0.0855, 0.0885), (0.1025, 0.1052)):
        sp = (u >= su[0]) & (u <= su[1]) & (np.abs(v - 0.002) <= 0.0225)
        img.over(np.array([0.30, 0.64, 0.84]) * cyl(v, 0.0225)[..., None], sp.astype(np.float32))

    # Blade -------------------------------------------------------------------
    s = np.clip((u - 0.11) / 0.89, 0, 1)
    curve = 0.035 * s ** 2
    half = 0.017 - 0.005 * s
    back_v = curve + half
    edge_straight = curve - half
    kiss = np.clip((s - 0.9) / 0.1, 0, 1)
    edge_v = edge_straight + 2 * half * kiss ** 2.2
    # Extend the blade root under the habaki (u 0.10) with the base profile.
    blade = (u >= 0.10) & (u <= 1.0) & (v >= edge_v) & (v <= back_v)
    t0 = (v - edge_straight) / (2 * half)  # 0 at the edge, 1 at the back (ignoring kissaki)
    tk = (v - edge_v) / np.maximum(back_v - edge_v, 1e-5)  # follows the kissaki curve
    in_kissaki = s > 0.9
    shinogi_t = 0.64
    hira = t0 < shinogi_t

    # Base steel: polished, lit from screen-left/top, soft environment banding along u.
    env = 0.5 + 0.5 * np.sin(u * 7.0 + 1.3) * np.cos(u * 2.3)
    streak = fbm(shp, 40 * SS, 3, 21, aniso=(6.0, 0.15))  # long streaks along the blade
    grad_u = 0.82 + 0.22 * u
    steel_hira = 0.46 + 0.14 * env + 0.05 * streak
    # Specular streak in the flat (hira).
    spec = np.exp(-((t0 - 0.53) / 0.045) ** 2) * (0.55 + 0.45 * sstep(0.1, 0.5, u) * (1 - sstep(0.8, 0.95, u)))
    steel_ji = 0.30 + 0.10 * env + 0.04 * streak  # shinogi-ji angled away: darker
    lum = np.where(hira, steel_hira + 0.5 * spec, steel_ji) * grad_u
    tint = np.array([1.02, 1.0, 0.96], np.float32)  # cool steel (BGR)
    bcol = tint * lum[..., None]

    # Hamon: gunome waves with a soft, cloudy (nioi) boundary and sparkly nie.
    hn = fbm(shp, 10 * SS, 4, 31)
    wave = 0.27 + 0.07 * np.abs(np.sin(u * np.pi * 15.0)) ** 0.8 + 0.035 * np.sin(u * 41.0) + 0.03 * hn
    ht = np.where(in_kissaki, tk, t0)
    wave = np.where(in_kissaki, 0.38 + 0.02 * hn, wave)  # boshi follows the tip curve
    hard = 1.0 - sstep(wave - 0.035, wave + 0.035, ht)
    nioi = np.exp(-((ht - wave) / 0.03) ** 2)
    frost = np.array([0.95, 0.95, 0.93], np.float32) * (0.88 + 0.08 * fbm(shp, 3 * SS, 2, 41))[..., None]
    bcol = bcol * (1 - hard[..., None] * 0.92) + frost * (hard[..., None] * 0.92)
    bcol = bcol + 0.12 * nioi[..., None]
    nie = (RNG.random(shp) > 0.985) & (nioi > 0.3)
    bcol = np.where(nie[..., None], np.minimum(1.2, bcol + 0.35), bcol)
    # Hamachi: the hamon starts a little above the habaki.
    root = u < 0.13
    bcol = np.where(root[..., None], tint * (0.55 * grad_u)[..., None], bcol)

    # Shinogi ridge, mune (back) and the bright cutting edge.
    ridge = np.exp(-((t0 - shinogi_t) * 2 * half * ppu * SS / (0.9 * SS)) ** 2) * (~in_kissaki)
    bcol = bcol + 0.45 * ridge[..., None]
    mune = t0 > 0.94
    bcol = np.where(mune[..., None], tint * (0.42 * grad_u)[..., None], bcol)
    edge_line = (v - edge_v) * ppu < 0.9
    bcol = np.where(edge_line[..., None], np.array([1.0, 1.0, 1.0]), bcol)
    # Yokote: the line separating the kissaki.
    yok = np.abs(s - 0.9) * 0.89 * ppu < 0.5
    bcol = np.where((yok & (t0 < shinogi_t + 0.05))[..., None], bcol * 0.55, bcol)
    # Kissaki face: a touch brighter, the ko-shinogi continues to the tip.
    ko = in_kissaki & (np.abs(tk - 0.62) * np.maximum(back_v - edge_v, 1e-5) * ppu < 0.45)
    bcol = np.where(ko[..., None], bcol + 0.25, bcol)
    img.over(np.clip(bcol, 0, 1), blade.astype(np.float32))
    img.over(np.array([0.10, 0.10, 0.12]), outline(blade, 0.8) * 0.85)

    # Habaki: gold collar with diagonal file marks, over the blade root.
    hb0, hb1 = 0.1035, 0.128
    hbm = (u >= hb0) & (u <= hb1) & (v >= -0.0195 + 0.0010 * (u - hb0) / (hb1 - hb0) * 0)
    hbm &= (v <= 0.0205) & (v >= -0.0195)
    tt = (v + 0.0195) / 0.04
    gold = np.array([0.22, 0.60, 0.84], np.float32)
    gl = 0.55 + 0.6 * np.exp(-((tt - 0.35) / 0.18) ** 2) - 0.2 * tt
    files = 0.85 + 0.15 * np.sin((u * 1.0 - v * 0.8) * 2600.0)
    img.over(gold * (gl * files)[..., None], hbm.astype(np.float32))
    img.over(np.array([0.06, 0.18, 0.28]), outline(hbm, 0.8))
    img.save(path)


# --- cleaver ------------------------------------------------------------------
def make_cleaver(path: str) -> None:
    W, H = 176, SWORD_H
    img = Img(W, H)
    u, v, ppu = uv_grid(W, H)
    shp = u.shape
    BV, EV = CLEAVER_BACK_V, CLEAVER_EDGE_V

    # Hilt: cloth bandage wrapped diagonally, slightly uneven.
    hw = 0.0235
    hilt = (u >= -0.188) & (u <= 0.062) & (np.abs(v) <= hw * (1 + 0.04 * np.sin(u * 160)))
    pitch = 0.0175
    w_ = (u + v * 0.55) / pitch
    fr = w_ - np.floor(w_)
    shade = cyl(v, hw)
    cloth = 0.80 + 0.10 * fbm(shp, 2 * SS, 3, 51) + 0.06 * np.sin(v * 2400)  # weave
    overlap = 0.62 + 0.38 * sstep(0.0, 0.85, fr)  # each turn shadows the next
    hcol = np.array([0.80, 0.86, 0.88]) * (cloth * overlap * shade)[..., None]
    seam = np.abs(fr - 0.0) * pitch * ppu < 0.7
    hcol = np.where(seam[..., None], hcol * 0.5, hcol)
    img.over(hcol, hilt.astype(np.float32))
    img.over(np.array([0.20, 0.22, 0.24]), outline(hilt, 0.9) * 0.9)
    # Iron pommel ring.
    pm = (u >= -0.199) & (u < -0.186) & (np.abs(v) <= 0.026)
    img.over(np.array([0.22, 0.21, 0.22]) * cyl(v, 0.026)[..., None], pm.astype(np.float32))
    img.over(np.array([0.05, 0.05, 0.06]), outline(pm, 0.8))
    # Collar where blade meets hilt.
    co = (u >= 0.058) & (u <= 0.074) & (v >= EV - 0.006) & (v <= BV + 0.006)
    ct = (v - (EV - 0.006)) / (BV - EV + 0.012)
    img.over(np.array([0.24, 0.24, 0.26]) * (0.6 + 0.6 * np.exp(-((ct - 0.6) / 0.25) ** 2))[..., None],
             co.astype(np.float32))
    img.over(np.array([0.05, 0.05, 0.06]), outline(co, 0.8))

    # Blade.
    blade = poly_mask(img, CLEAVER_BLADE, ppu, W)
    blade &= u > 0.073
    t = (v - EV) / (BV - EV)  # 0 edge, 1 back
    streak = fbm(shp, 30 * SS, 3, 61, aniso=(5.0, 0.2))
    # Broad diagonal reflection bands give the flat face a polished look.
    refl = 0.5 + 0.5 * np.sin((u * 3.2 + v * 9.0) * np.pi)
    refl = sstep(0.25, 0.95, refl)
    face_l = (0.34 + 0.20 * refl + 0.04 * streak) * (0.88 + 0.18 * u)
    bcol = np.array([0.98, 0.98, 1.02]) * face_l[..., None]
    # Fuller near the back.
    fl = (u > 0.11) & (u < 0.86) & (v > BV - 0.030) & (v < BV - 0.014)
    ft = (v - (BV - 0.030)) / 0.016
    bcol = np.where(fl[..., None], bcol * (0.55 + 0.25 * ft)[..., None], bcol)
    fl_hi = (u > 0.11) & (u < 0.86) & (np.abs(v - (BV - 0.030)) * ppu < 0.6)
    bcol = np.where(fl_hi[..., None], bcol + 0.25, bcol)
    # Bevel: polished secondary grind along the edge, following the tip sweep.
    edge_dist = cv2.distanceTransform((blade.astype(np.uint8)), cv2.DIST_L2, 5) / (ppu * SS)
    bev_w = 0.035
    edge_side = (t < 0.45) | (u > 0.8)
    # Distance to the cutting edge only (not the back): mask the back with a synthetic boundary.
    edge_only = blade.copy()
    edge_only[(v > BV - 0.004) & (u < 0.985)] = True
    edge_only[(u > 0.03) & (u <= 0.0735) & (v > EV - 0.01) & (v < BV + 0.01)] = True
    ed = cv2.distanceTransform(np.pad(edge_only.astype(np.uint8), 1, constant_values=0), cv2.DIST_L2, 5)
    ed = ed[1:-1, 1:-1] / (ppu * SS)
    bevel = blade & (ed < bev_w) & edge_side
    bt = ed / bev_w  # 0 at edge
    bev_col = np.array([0.80, 0.80, 0.83]) * (1.15 - 0.35 * bt + 0.06 * streak)[..., None]
    bcol = np.where(bevel[..., None], bev_col, bcol)
    grind = blade & edge_side & (np.abs(ed - bev_w) * ppu < 0.55)
    bcol = np.where(grind[..., None], bcol * 0.55, bcol)
    cut = blade & (ed * ppu < 1.1) & edge_side
    bcol = np.where(cut[..., None], np.array([1.0, 1.0, 1.0]), bcol)
    # Spine.
    spine = blade & (v > BV - 0.0045)
    bcol = np.where(spine[..., None], bcol * 0.6, bcol)
    _ = edge_dist
    img.over(np.clip(bcol, 0, 1), blade.astype(np.float32))
    img.over(np.array([0.08, 0.08, 0.10]), outline(blade, 0.9) * 0.9)
    img.save(path)


# --- dark blade ---------------------------------------------------------------
def make_dark(path: str) -> None:
    W, H = 96, SWORD_H
    img = Img(W, H)
    u, v, ppu = uv_grid(W, H)
    shp = u.shape

    # Handle: black cord over dark grey, kashira with a ring for the chain.
    def hw_h(uu):
        return 0.0185 + 0 * uu

    def under(uu, vv, hw):
        return np.array([0.20, 0.19, 0.21], np.float32) * (0.9 + 0.1 * fbm(shp, 1.5 * SS, 2, 71))[..., None]

    wrap_handle(img, u, v, -0.17, 0.093, hw_h, 7, (0.07, 0.06, 0.07), under, 73)
    hm = (u >= -0.17) & (u <= 0.093) & (np.abs(v) <= 0.0185)
    img.over(np.array([0.02, 0.02, 0.02]), outline(hm, 0.9))
    ka = (u >= -0.196) & (u < -0.168) & (np.abs(v) <= 0.0175 + 0.06 * np.clip(u + 0.17, -0.02, 0))
    img.over(np.array([0.10, 0.09, 0.10]) * cyl(v, 0.0185)[..., None], ka.astype(np.float32))
    hole = ((u + 0.188) / 0.0055) ** 2 + (v / 0.0075) ** 2 < 1
    img.over(np.array([0.0, 0.0, 0.0]), (ka & hole).astype(np.float32) * 0.0)
    img.A[(ka & hole)] = 0.0
    img.P[(ka & hole)] = 0.0
    img.over(np.array([0.02, 0.02, 0.02]), outline(ka & ~hole, 0.8))

    # Guard: hexagonal plate with a bevelled rim and a crimson inlay line.
    gpoly = np.array([[0.095, -0.05], [0.104, -0.058], [0.12, -0.05], [0.12, 0.05], [0.104, 0.058], [0.095, 0.05]])
    gm = poly_mask(img, gpoly, ppu, W)
    gt = (u - 0.095) / 0.025
    gcol = np.array([0.08, 0.07, 0.08]) * (0.8 + 0.5 * gt + 0.15 * fbm(shp, 3 * SS, 2, 75))[..., None]
    sheen = np.exp(-((v + 0.02) / 0.012) ** 2) * 0.12
    gcol = gcol + np.array([0.45, 0.25, 0.15]) * sheen[..., None]
    img.over(gcol, gm.astype(np.float32))
    inner = poly_mask(img, np.array([[0.099, -0.043], [0.105, -0.049], [0.116, -0.043], [0.116, 0.043],
                                     [0.105, 0.049], [0.099, 0.043]]), ppu, W)
    img.over(np.array([0.10, 0.08, 0.45]), outline(inner, 0.6) * 0.9)
    img.over(np.array([0.30, 0.30, 0.34]), outline(gm, 0.6) * 0.6)

    # Blade: slim, black, with a blue-black sheen and a dark crimson edge line.
    blade = poly_mask(img, dark_profile(40), ppu, W) & (u > 0.118)
    s = np.clip((u - 0.12) / 0.86, 0, 1)
    half = 0.02 - 0.006 * s
    t = (v + half) / (2 * half)  # 0 at the edge side (-v), 1 at the back
    streak = fbm(shp, 30 * SS, 3, 81, aniso=(6.0, 0.15))
    base = np.array([0.055, 0.045, 0.05]) * (1.0 + 0.3 * streak)[..., None]
    band = np.exp(-((t - 0.58) / 0.10) ** 2) * (0.5 + 0.5 * np.sin(u * 6.0 + 0.5) ** 2)
    sheen = np.array([0.55, 0.30, 0.16]) * (band * 0.55)[..., None]
    hi = np.exp(-((t - 0.62) / 0.025) ** 2) * sstep(0.15, 0.5, u) * 0.35
    bcol = base + sheen + np.array([0.9, 0.7, 0.6]) * hi[..., None]
    ridge = np.abs(t - 0.5) * 2 * half * ppu < 0.5
    bcol = np.where(ridge[..., None], bcol + 0.05, bcol)
    ed = cv2.distanceTransform(np.pad(blade.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1] / (ppu * SS)
    edge_side = (t < 0.5) | (s > 0.93)
    crimson = np.array([0.10, 0.06, 0.62])
    cw = np.exp(-(ed * ppu / 1.3) ** 2) * edge_side
    glow_in = np.exp(-(ed * ppu / 4.0) ** 2) * edge_side * 0.35
    bcol = bcol * (1 - cw[..., None]) + crimson * cw[..., None] + crimson * 0.5 * glow_in[..., None]
    img.over(np.clip(bcol, 0, 1), blade.astype(np.float32))
    img.over(np.array([0.01, 0.01, 0.01]), outline(blade, 0.6) * (~edge_side) * 0.8)
    img.save(path)


# --- mask ---------------------------------------------------------------------
def make_mask(path: str) -> None:
    W, H = 384, 496
    img = Img(W, H)
    ys, xs = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
    # Units of image width; eyes at px = +-0.18 on the line py0, chin at py1.
    px = ((xs + 0.5) / SS - 0.5 * W) / W
    py = ((ys + 0.5) / SS) / W
    py0, py1 = 0.42 * H / W, 0.95 * H / W
    shp = px.shape
    ax = np.abs(px)

    # Silhouette: rounded brow, widest at the cheekbones, tapering jaw, rounded chin.
    top = 0.035
    wide = py0 + 0.05
    hw_up = 0.44 * np.sqrt(np.clip(1 - ((wide - py) / (wide - top)) ** 2, 0, 1))
    tj = np.clip((py - wide) / (py1 - wide), 0, 1)
    hw_dn = 0.44 - (0.44 - 0.10) * tj ** 1.35
    hw = np.where(py < wide, hw_up, hw_dn)
    chin_r = 0.10
    chin_w = 0.44 - (0.44 - 0.10) * ((py1 - chin_r - wide) / (py1 - wide)) ** 1.35
    chin = (py < py1 - chin_r) | ((px / chin_w) ** 2 + ((py - (py1 - chin_r)) / chin_r) ** 2 < 1)
    sil = (ax < hw) & (py > top) & chin
    dist = cv2.distanceTransform(np.pad(sil.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1] / (W * SS)

    # Height field -> normals -> lighting (key light from the upper left).
    def g2(cx, cy, sx, sy):
        return np.exp(-(((px - cx) / sx) ** 2 + ((py - cy) / sy) ** 2))

    dome = np.sqrt(np.clip(1 - (px / 0.5) ** 2 - ((py - 0.62) / 0.66) ** 2, 0, 1))
    hgt = 0.55 * dome
    hgt += 0.045 * (g2(-0.16, py0 - 0.09, 0.13, 0.04) + g2(0.16, py0 - 0.09, 0.13, 0.04))  # brow
    nose = np.exp(-(px / 0.045) ** 2) * sstep(py0 - 0.04, py0 + 0.16, py) * (1 - sstep(0.80, 0.87, py))
    hgt += 0.06 * nose
    hgt += 0.04 * (g2(-0.27, 0.70, 0.10, 0.07) + g2(0.27, 0.70, 0.10, 0.07))  # cheekbones
    hgt -= 0.06 * (g2(-0.18, py0, 0.11, 0.07) + g2(0.18, py0, 0.11, 0.07))  # eye sockets
    hgt -= 0.03 * (g2(-0.22, 0.88, 0.08, 0.06) + g2(0.22, 0.88, 0.08, 0.06))  # under the cheeks
    hgt += 0.0035 * fbm(shp, 10 * SS, 4, 91)
    hgt = cv2.GaussianBlur(hgt.astype(np.float32), (0, 0), 2.0 * SS)
    step = 1.0 / (W * SS)
    gy, gx = np.gradient(hgt, step)
    nrm = np.dstack([-gx, -gy, np.full(shp, 1.6, np.float32)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    Ld = np.array([-0.40, -0.55, 0.73], np.float32)
    Ld /= np.linalg.norm(Ld)
    diff = np.clip(nrm @ Ld, 0, 1)
    hv = Ld + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    spec = np.clip(nrm @ hv, 0, 1) ** 40
    ao = 0.80 + 0.20 * sstep(0.0, 0.45, hgt / 0.55)
    rim = sstep(0.0, 0.025, dist)  # bevelled, darker outer edge
    bone = np.array([0.80, 0.90, 0.95], np.float32)
    tex = 1.0 + 0.025 * fbm(shp, 6 * SS, 4, 93) + 0.015 * fbm(shp, 1.2 * SS, 2, 95)
    stain = sstep(0.15, 0.7, fbm(shp, 40 * SS, 3, 97)) * 0.07
    lum = (0.42 + 0.68 * diff) * ao * (0.6 + 0.4 * rim) * tex
    col = bone * lum[..., None] * (1 - stain[..., None] * np.array([1.0, 0.55, 0.25]))
    col += 0.28 * spec[..., None]

    # Crimson markings (original design): tapered brush strokes from the brow through each eye
    # down the cheek, a thin outer cheek stroke, and a diamond on the forehead.
    red = np.array([0.15, 0.10, 0.66], np.float32)
    marks = np.zeros(shp, np.float32)
    brush = fbm(shp, 3 * SS, 3, 99)
    for sgn in (-1.0, 1.0):
        y0, y1 = 0.12, 0.98
        tt = np.clip((py - y0) / (y1 - y0), 0, 1)
        cx = sgn * (0.20 + 0.06 * tt - 0.05 * tt ** 2)
        wdt = 0.030 * np.clip(np.sin(np.pi * tt ** 0.8), 0, 1) ** 0.7 + 0.003
        d = np.abs(px - cx) - wdt * (1 + 0.12 * brush)
        marks = np.maximum(marks, ((d < 0) & (py > y0) & (py < y1)).astype(np.float32))
        tt2 = np.clip((py - 0.62) / 0.32, 0, 1)
        cx2 = sgn * (0.335 - 0.07 * tt2)
        d2 = np.abs(px - cx2) - (0.011 * np.sin(np.pi * tt2) + 0.0015)
        marks = np.maximum(marks, ((d2 < 0) & (py > 0.62) & (py < 0.94)).astype(np.float32))
    marks = np.maximum(marks, ((ax / 0.032 + np.abs(py - 0.20) / 0.06) < 1).astype(np.float32))
    mcol = red * (lum * (0.97 + 0.06 * brush))[..., None] + 0.22 * spec[..., None]
    col = col * (1 - marks[..., None]) + mcol * marks[..., None]

    # Hairline cracks: branching walks, a dark groove with a lit lip below-right.
    cr = np.zeros((H * SS, W * SS), np.uint8)
    hi = np.zeros_like(cr)
    rng = np.random.default_rng(5)
    starts = [(0.33, 0.12, 2.3, 0.16), (-0.40, 0.36, 0.3, 0.12), (0.07, 1.10, -1.9, 0.10), (-0.12, 0.05, 1.6, 0.10)]
    for sx, sy, ang, ln in starts:
        stack = [(sx, sy, ang, ln, 3)]
        while stack:
            x, y, a, length, depth = stack.pop()
            pts = [(x, y)]
            n = 7
            for _ in range(n):
                a += rng.normal(0, 0.35)
                x += np.cos(a) * length / n
                y += np.sin(a) * length / n
                pts.append((x, y))
                if depth > 1 and rng.random() < 0.2:
                    stack.append((x, y, a + rng.choice([-1, 1]) * rng.uniform(0.5, 0.9), length * 0.45, depth - 1))
            p = np.array([[(qx + 0.5) * W * SS, qy * W * SS] for qx, qy in pts], np.float32)
            th = max(1, int(SS * (1.1 if depth == 3 else 0.7)))
            cv2.polylines(cr, [np.round(p).astype(np.int32)], False, 255, th, cv2.LINE_AA)
            cv2.polylines(hi, [np.round(p + (SS * 0.9, SS * 0.9)).astype(np.int32)], False, 255, th, cv2.LINE_AA)
    crf = cr.astype(np.float32) / 255 * sil
    hif = (hi.astype(np.float32) / 255) * (1 - crf) * sil
    col = col * (1 - 0.7 * crf[..., None]) + np.array([0.18, 0.22, 0.26]) * 0.7 * crf[..., None]
    col = col + 0.10 * hif[..., None]

    # Eye holes: almond shapes swept up at the outer corner; soft occlusion around them.
    holes = np.zeros(shp, np.float32)
    for sgn in (-1.0, 1.0):
        ex = (px - sgn * 0.18) * sgn  # + = outward
        ey = py - py0
        exn = np.where(ex > 0, ex / 0.13, ex / 0.10)
        eyl = ey + 0.075 * np.clip(ex / 0.13, 0, 1.2) ** 2
        hh = 0.052 * np.clip(1 - exn ** 2, 0, 1) ** 0.6
        holes = np.maximum(holes, ((np.abs(exn) < 1) & (np.abs(eyl) < hh)).astype(np.float32))
    near = cv2.GaussianBlur(holes, (0, 0), 3.0 * SS)
    col = col * (1 - 0.55 * np.clip(near * 1.6, 0, 1))[..., None]
    lip = np.clip(cv2.GaussianBlur(holes, (0, 0), 0.8 * SS) - holes, 0, 1)
    col = col * (1 - 0.6 * lip[..., None])

    # Toothy grin: a wide row of rounded teeth split by a dark slit.
    my = py0 + 0.62 * (py1 - py0)
    mw = 0.215
    xm = px / mw
    yc = my - 0.022 * xm ** 2 + 0.006
    inmouth = np.abs(xm) < 1
    th = 0.050 * np.clip(1 - xm ** 2, 0, 1) ** 0.45
    teeth = inmouth & (np.abs(py - yc) < th)
    ntooth = 12
    tx = (px + mw) / (2 * mw / ntooth)
    upper = py < yc
    tx = np.where(upper, tx, tx + 0.5)  # lower row interlocks
    tf = tx - np.floor(tx)
    tv = np.abs(py - yc) / np.maximum(th, 1e-4)  # 0 at the slit, 1 at the gum line
    tshade = (0.72 + 0.32 * np.sin(np.clip(tf, 0, 1) * np.pi)) * (1.05 - 0.25 * tv)
    tcol = bone * (tshade * (0.55 + 0.5 * diff))[..., None]
    col = np.where(teeth[..., None], tcol, col)
    # Rounded tooth tips: dark notches near the slit between teeth, thin gaps up to the gum.
    tipn = teeth & (np.abs(tf - 0.5) > 0.5 - 0.22 * np.clip(1 - tv * 2.2, 0, 1))
    gap = teeth & (np.abs(tf - 0.5) > 0.455)
    col = np.where((tipn | gap)[..., None], np.array([0.06, 0.06, 0.08]), col)
    slit = inmouth & (np.abs(py - yc) < 0.006 * np.clip(1 - xm ** 2, 0, 1) ** 0.5 + 0.0012)
    col = np.where(slit[..., None], np.array([0.02, 0.02, 0.03]), col)
    groove = outline(teeth, 1.2)
    col = col * (1 - 0.55 * groove[..., None])
    img.over(np.clip(col, 0, 1), sil.astype(np.float32))

    # Holes are dark and partly transparent so the real eye shows through dimly.
    hole = cv2.GaussianBlur(holes, (0, 0), 0.5 * SS) * sil
    img.P = img.P * (1 - hole[..., None]) + np.array([0.02, 0.02, 0.03]) * 0.55 * hole[..., None]
    img.A = img.A * (1 - hole) + 0.55 * hole
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--only", nargs="*", default=None, help="subset: katana cleaver dark mask")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = {"katana": (make_katana, "sword_katana.png"), "cleaver": (make_cleaver, "sword_cleaver.png"),
            "dark": (make_dark, "sword_dark.png"), "mask": (make_mask, "mask.png")}
    for name, (fn, fname) in jobs.items():
        if args.only and name not in args.only:
            continue
        p = os.path.join(args.out, fname)
        fn(p)
        print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
