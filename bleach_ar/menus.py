"""Full-screen menus: the universe (mode) selector, the per-mode ability
panel and the short transition overlay between menus and the live camera.

Everything is procedural and animated by ``t`` (seconds). Static layers
(gradients, motifs, titles, glows) are rendered once per size and cached;
per frame only cheap additive sprites, arcs and cached text are drawn.
Particles are stateless (positions are a function of ``t`` and a fixed
random table), so drawing has no hidden simulation to keep in sync.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from .modes import BLEACH, JJK, MODE_ORDER, MODES

WHITE = (240, 242, 250)
DIM = (150, 156, 175)
FONT_FAMILY = "helveticaneue,arial,dejavusans"

# Per-mode palettes (RGB).
PALETTE = {
    BLEACH: {"bg_top": (4, 8, 20), "bg_bot": (10, 22, 48), "glow": (70, 150, 255), "hi": (215, 235, 255),
             "mote": (120, 190, 255), "accent": MODES[BLEACH].accent, "accent2": MODES[BLEACH].accent2},
    JJK: {"bg_top": (8, 3, 16), "bg_bot": (26, 8, 40), "glow": (120, 60, 230), "hi": (240, 225, 255),
          "mote": (90, 160, 255), "red": (235, 50, 75), "accent": MODES[JJK].accent,
          "accent2": MODES[JJK].accent2},
}


def _lerp(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


def _mix(c1, c2, k: float) -> Tuple[int, int, int]:
    return tuple(int(_lerp(a, b, k)) for a, b in zip(c1, c2))


def _scale(c, k: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(v * k))) for v in c)


def _ease(k: float) -> float:
    k = max(0.0, min(1.0, k))
    return k * k * (3 - 2 * k)


def _glow_sprite(color, radius: int, power: float = 2.2) -> pygame.Surface:
    """Soft radial glow on black (blit with BLEND_ADD)."""
    d = radius * 2 + 1
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    r = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2) / max(1, radius)
    f = np.clip(1.0 - r, 0, 1) ** power
    arr = (f[..., None] * np.array(color, np.float32)[None, None]).astype(np.uint8)
    surf = pygame.Surface((d, d))
    pygame.surfarray.blit_array(surf, arr.transpose(1, 0, 2))
    return surf


def _blur(surf: pygame.Surface, factor: int = 8) -> pygame.Surface:
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
    return pygame.transform.smoothscale(small, (w, h))


def _vertical_gradient(size, top, bot) -> np.ndarray:
    w, h = size
    k = np.linspace(0, 1, h, dtype=np.float32)[None, :, None]
    return (np.array(top, np.float32) * (1 - k) + np.array(bot, np.float32) * k).repeat(w, 0)


def _rounded_alpha(surf: pygame.Surface, radius: int) -> pygame.Surface:
    out = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    out.blit(surf, (0, 0))
    mask = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    mask.fill((255, 255, 255, 0))
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return out


class Menus:
    """Mode selector, ability panel and transition overlay (see module doc)."""

    def __init__(self, fonts):
        self.fonts = fonts
        self._font_path = getattr(fonts, "path", None)
        self._fonts: Dict[Tuple[int, bool], pygame.font.Font] = {}
        self._text_cache: Dict[tuple, pygame.Surface] = {}
        self._cache: Dict[tuple, pygame.Surface] = {}
        self._glows: Dict[tuple, pygame.Surface] = {}
        self._scaled: Dict[tuple, pygame.Surface] = {}
        rng = np.random.default_rng(7)
        self._rand = {BLEACH: rng.random((64, 6)), JJK: rng.random((64, 6))}
        self._focus = [1.0, 0.0]
        self._last_t: Optional[float] = None

    # --- small cached primitives -----------------------------------------
    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        size = max(9, int(size))
        key = (size, bold)
        f = self._fonts.get(key)
        if f is None:
            f = None
            if self._font_path:
                try:
                    f = pygame.font.Font(self._font_path, size)
                except (OSError, pygame.error):
                    f = None
            if f is None:
                f = pygame.font.SysFont(FONT_FAMILY, size, bold=bold)
            self._fonts[key] = f
        return f

    def _txt(self, text: str, size: int, color, bold: bool = False, spacing: int = 0) -> pygame.Surface:
        key = (text, int(size), tuple(color), bold, spacing)
        img = self._text_cache.get(key)
        if img is None:
            font = self._font(size, bold)
            if spacing:
                glyphs = [font.render(ch, True, color) for ch in text]
                w = sum(g.get_width() for g in glyphs) + spacing * max(0, len(glyphs) - 1)
                img = pygame.Surface((max(1, w), font.get_height()), pygame.SRCALPHA)
                x = 0
                for g in glyphs:
                    img.blit(g, (x, 0))
                    x += g.get_width() + spacing
            else:
                img = font.render(text, True, color)
            if len(self._text_cache) > 600:
                self._text_cache.clear()
            self._text_cache[key] = img
        return img

    def _blit_txt(self, surf, text, size, color, pos, anchor="topleft", bold=False, spacing=0,
                  shadow=True) -> pygame.Rect:
        img = self._txt(text, size, color, bold, spacing)
        rect = img.get_rect(**{anchor: pos})
        if shadow:
            surf.blit(self._txt(text, size, (0, 0, 0), bold, spacing), rect.move(1, 2))
        surf.blit(img, rect)
        return rect

    def _glow(self, color, radius: int, power: float = 2.2) -> pygame.Surface:
        color = tuple(min(255, (int(c) + 6) // 12 * 12) for c in color)  # bounded sprite cache
        key = (color, int(radius), power)
        g = self._glows.get(key)
        if g is None:
            g = _glow_sprite(color, int(radius), power)
            self._glows[key] = g
        return g

    def _cached(self, key, build):
        s = self._cache.get(key)
        if s is None:
            if len(self._cache) > 80:
                self._cache.clear()
            s = build()
            self._cache[key] = s
        return s

    def _wrap(self, text: str, size: int, max_w: int, bold: bool = False) -> List[str]:
        font = self._font(size, bold)
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if font.size(trial)[0] <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _ellipsize(self, text: str, size: int, max_w: int, bold: bool = False) -> str:
        font = self._font(size, bold)
        if font.size(text)[0] <= max_w:
            return text
        while text and font.size(text + "...")[0] > max_w:
            text = text[:-1]
        return text.rstrip() + "..."

    # --- backgrounds -----------------------------------------------------
    def _backdrop(self, surf: pygame.Surface, backdrop: Optional[pygame.Surface], tint, darkness: int) -> None:
        """Blurred, dimmed camera (or a dark cinematic gradient) + vignette."""
        W, H = surf.get_size()
        if backdrop is not None:
            # Blur = shrink to a tiny image and smoothscale back; dim + tint the tiny one (free).
            if backdrop.get_bitsize() not in (24, 32):
                backdrop = backdrop.convert(24)
            bw, bh = backdrop.get_size()
            quarter = pygame.transform.scale(backdrop, (max(1, bw // 4), max(1, bh // 4)))  # nearest: cheap
            small = pygame.transform.smoothscale(quarter, (max(1, W // 24), max(1, H // 24)))
            small.fill(_scale((255, 255, 255), (255 - darkness) / 255), special_flags=pygame.BLEND_RGB_MULT)
            small.fill(_scale(tint, 0.12), special_flags=pygame.BLEND_RGB_ADD)
            half = pygame.transform.smoothscale(small, (max(1, W // 2), max(1, H // 2)))
            surf.blit(pygame.transform.scale(half, (W, H)), (0, 0))
            surf.blit(self._cached(("vig", W, H), lambda: self._vignette((W, H))), (0, 0))
        else:
            surf.blit(self._cached(("bgrad", W, H, tuple(tint)), lambda: self._bg_gradient((W, H), tint)), (0, 0))

    @staticmethod
    def _shade(size, darkness: int) -> pygame.Surface:
        s = pygame.Surface(size, pygame.SRCALPHA)
        s.fill((3, 4, 10, darkness))
        return s

    @staticmethod
    def _bg_gradient(size, tint) -> pygame.Surface:
        W, H = size
        arr = _vertical_gradient(size, (5, 6, 12), (12, 13, 24))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.55) / H) ** 2)
        arr += (np.clip(1 - r * 1.6, 0, 1) ** 2)[..., None].transpose(1, 0, 2) * np.array(tint, np.float32) * 0.18
        s = pygame.Surface(size)
        pygame.surfarray.blit_array(s, np.clip(arr, 0, 255).astype(np.uint8))
        s.blit(Menus._vignette(size), (0, 0))
        return s

    @staticmethod
    def _vignette(size) -> pygame.Surface:
        W, H = size
        sw, sh = max(2, W // 8), max(2, H // 8)
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        r = np.sqrt(((xx / sw) - 0.5) ** 2 * 1.2 + ((yy / sh) - 0.5) ** 2 * 1.6)
        a = (np.clip((r - 0.3) / 0.45, 0, 1) ** 1.6 * 200).astype(np.uint8)
        small = pygame.Surface((sw, sh), pygame.SRCALPHA)
        small.fill((0, 0, 0, 255))
        pygame.surfarray.pixels_alpha(small)[:] = a.T
        return pygame.transform.smoothscale(small, (W, H))

    # --- mode selector ---------------------------------------------------
    def mode_card_rects(self, size: Tuple[int, int]) -> List[pygame.Rect]:
        """Card hit boxes, index = MODE_ORDER index."""
        W, H = size
        cw = int(min(W * 0.38, H * 0.82))
        ch = int(H * 0.62)
        gap = int(W * 0.045)
        x0 = (W - (2 * cw + gap)) // 2
        y0 = int(H * 0.2)
        return [pygame.Rect(x0 + i * (cw + gap), y0, cw, ch) for i in range(len(MODE_ORDER))]

    def draw_mode_select(self, surf: pygame.Surface, t: float, selected: int,
                         backdrop: Optional[pygame.Surface], message: str = "", error: str = "") -> None:
        W, H = surf.get_size()
        s = H / 720.0
        selected = max(0, min(len(MODE_ORDER) - 1, int(selected)))
        dt = 0.0 if self._last_t is None else max(0.0, min(0.1, t - self._last_t))
        self._last_t = t
        for i in range(len(self._focus)):
            target = 1.0 if i == selected else 0.0
            self._focus[i] += (target - self._focus[i]) * min(1.0, dt * 12) if dt else (target - self._focus[i])
        k = max(0.0, min(1.0, self._focus[1]))
        if backdrop is not None:
            self._backdrop(surf, backdrop, _mix(PALETTE[BLEACH]["glow"], PALETTE[JJK]["glow"], k), 170)
        else:  # cross-fade the two cached mode gradients (no per-tint rebuilds)
            grads = [self._cached(("bgrad", W, H, tuple(PALETTE[m]["glow"])),
                                  lambda m=m: self._bg_gradient((W, H), PALETTE[m]["glow"])) for m in (BLEACH, JJK)]
            surf.blit(grads[0] if k < 0.5 else grads[1], (0, 0))
            if 0.02 < k < 0.98:
                top = grads[1] if k < 0.5 else grads[0]
                top.set_alpha(int(255 * (k if k < 0.5 else 1 - k)))
                surf.blit(top, (0, 0))
                top.set_alpha(None)

        # Header.
        accent = MODES[MODE_ORDER[selected]].accent
        self._blit_txt(surf, "BLEACH AR", int(15 * s + 4), _scale(accent, 0.9), (W // 2, int(H * 0.055)),
                       "center", bold=True, spacing=int(6 * s + 2))
        self._blit_txt(surf, "CHOOSE YOUR UNIVERSE", int(34 * s + 6), WHITE, (W // 2, int(H * 0.115)), "center",
                       bold=True, spacing=int(5 * s + 1))
        lw = int(W * 0.12)
        cy = int(H * 0.155)
        pygame.draw.line(surf, _scale(accent, 0.8), (W // 2 - lw, cy), (W // 2 + lw, cy), 1)
        pygame.draw.circle(surf, accent, (W // 2, cy), 3)

        rects = self.mode_card_rects((W, H))
        self.warm((W, H))
        for i, mid in enumerate(MODE_ORDER):
            self._draw_card(surf, rects[i], mid, t, self._focus[i], i == selected)

        # Footer / status.
        fy = int(H * 0.93)
        if error:
            y = int(H * 0.86) - 22 * (len(error.splitlines()) - 1)
            for line in error.splitlines():
                self._blit_txt(surf, line, int(17 * s + 4), (255, 140, 120), (W // 2, y), "center")
                y += int(22 * s + 4)
        elif message:
            self._blit_txt(surf, message, int(18 * s + 4), WHITE, (W // 2, int(H * 0.875)), "center", bold=True)
        self._hint_row(surf, [("LEFT / RIGHT", "select"), ("ENTER", "confirm"), ("ESC", "quit")],
                       (W // 2, fy), s, accent)

    def warm(self, size: Tuple[int, int]) -> None:
        """Build every static layer of the mode selector for ``size`` (cards and
        all halo levels) so later frames never hitch. Called automatically by
        the first draw_mode_select at a size; call earlier to prebuild. Also
        prebuilds the ability-panel backgrounds."""
        if ("warm", size) in self._cache:
            return
        for base, mid in zip(self.mode_card_rects(size), MODE_ORDER):
            ms = (base.w, base.h)
            radius = max(8, int(base.h * 0.035))
            self._cached(("card", mid, ms), lambda: self._build_card(mid, ms))
            for level in range(1, 9):
                self._cached(("halo", mid, ms, level), lambda: self._build_halo(mid, ms, radius, level / 8))
            # Ability-panel layers too, so opening the panel doesn't hitch.
            W, H = size
            self._cached(("pmotif", mid, W, H), lambda: self._panel_motif(mid, (W, H)))
            g = PALETTE[mid]["glow"]
            self._cached(("bgrad", W, H, tuple(g)), lambda: self._bg_gradient((W, H), g))
        self._cache[("warm", size)] = pygame.Surface((1, 1))

    def _hint_row(self, surf, items, center, s: float, accent) -> None:
        size = int(14 * s + 4)
        imgs = []
        for key, label in items:
            k = self._txt(key, size, (12, 14, 24), bold=True)
            lab = self._txt(label, size, DIM)
            imgs.append((k, lab))
        pad, gap = int(7 * s + 2), int(22 * s + 6)
        total = sum(k.get_width() + 2 * pad + 6 + lab.get_width() for k, lab in imgs) + gap * (len(imgs) - 1)
        x = center[0] - total // 2
        for k, lab in imgs:
            pill = pygame.Rect(x, 0, k.get_width() + 2 * pad, k.get_height() + 4)
            pill.centery = center[1]
            pygame.draw.rect(surf, _mix(accent, WHITE, 0.55), pill, border_radius=pill.h // 2)
            surf.blit(k, k.get_rect(center=pill.center))
            x = pill.right + 6
            surf.blit(lab, lab.get_rect(midleft=(x, center[1])))
            x += lab.get_width() + gap

    def _draw_card(self, surf, base: pygame.Rect, mid: str, t: float, focus: float, selected: bool) -> None:
        scale = 0.93 + 0.07 * _ease(focus)
        w = int(base.w * scale) // 2 * 2
        h = int(base.h * scale) // 2 * 2
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (base.centerx, base.centery - int(6 * focus))
        master_size = (base.w, base.h)
        master = self._cached(("card", mid, master_size), lambda: self._build_card(mid, master_size))
        if (w, h) != master_size:
            card = self._scaled.get((mid, w, h))
            if card is None:  # sizes seen while the selection animates (bounded, separate cache)
                if len(self._scaled) > 48:
                    self._scaled.clear()
                card = self._scaled[(mid, w, h)] = pygame.transform.smoothscale(master, (w, h))
        else:
            card = master
        pal = PALETTE[mid]
        radius = max(8, int(base.h * 0.035))

        # Outer glow behind the selected card.
        if focus > 0.02:
            pulse = 0.75 + 0.25 * math.sin(t * 3.0)
            level = max(1, min(8, round(focus * pulse * 8)))  # 8 pre-dimmed halos, no per-frame fill
            halo = self._cached(("halo", mid, master_size, level),
                                lambda: self._build_halo(mid, master_size, radius, level / 8))
            surf.blit(halo, halo.get_rect(center=rect.center), special_flags=pygame.BLEND_ADD)

        surf.blit(card, rect)
        clip = surf.get_clip()
        surf.set_clip(rect.inflate(-4, -4))
        if mid == BLEACH:
            self._animate_bleach(surf, rect, t, focus)
        else:
            self._animate_jjk(surf, rect, t, focus)
        surf.set_clip(clip)

        # Dim unselected cards.
        if focus < 0.98:
            shade = self._cached(("cshade", w, h), lambda: _rounded_alpha(self._shade((w, h), 255), radius))
            shade.set_alpha(int(150 * (1 - focus)))
            surf.blit(shade, rect)

        # Border: thin always, bright animated sheen when selected.
        border = _mix((60, 66, 90), pal["accent"], focus)
        pygame.draw.rect(surf, border, rect, max(1, int(1 + 1.5 * focus)), border_radius=radius)
        if focus > 0.05:
            per = 2 * (rect.w + rect.h)
            for off, size_k in ((0.0, 1.0), (0.5, 0.7)):
                d = ((t * 0.18 + off) % 1.0) * per
                if d < rect.w:
                    p = (rect.left + d, rect.top)
                elif d < rect.w + rect.h:
                    p = (rect.right, rect.top + d - rect.w)
                elif d < 2 * rect.w + rect.h:
                    p = (rect.right - (d - rect.w - rect.h), rect.bottom)
                else:
                    p = (rect.left, rect.bottom - (d - 2 * rect.w - rect.h))
                g = self._glow(_scale(pal["accent2"], 0.9 * focus), int(26 * size_k + 6))
                surf.blit(g, g.get_rect(center=(int(p[0]), int(p[1]))), special_flags=pygame.BLEND_ADD)
            # "Press ENTER" tag under the selected card.
            s = surf.get_height() / 720.0
            tag = self._txt("PRESS ENTER", int(12 * s + 4), (12, 14, 24), bold=True, spacing=3)
            pill = pygame.Rect(0, 0, tag.get_width() + 28, tag.get_height() + 8)
            pill.midtop = (rect.centerx, rect.bottom + int(10 * s))
            col = _mix(pal["accent"], WHITE, 0.35 + 0.15 * math.sin(t * 4))
            if focus > 0.6:
                pygame.draw.rect(surf, col, pill, border_radius=pill.h // 2)
                surf.blit(tag, tag.get_rect(center=pill.center))

    def _build_halo(self, mid: str, size, radius: int, k: float) -> pygame.Surface:
        w, h = size
        pad = 40
        s = pygame.Surface((w + 2 * pad, h + 2 * pad))
        pygame.draw.rect(s, _scale(PALETTE[mid]["accent"], k), (pad, pad, w, h), 10, border_radius=radius)
        return _blur(_blur(s, 6), 3)

    # Card static layers --------------------------------------------------
    def _build_card(self, mid: str, size) -> pygame.Surface:
        w, h = size
        pal = PALETTE[mid]
        arr = _vertical_gradient(size, pal["bg_top"], pal["bg_bot"])
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (0.6, 0.38) if mid == BLEACH else (0.5, 0.4)
        r = np.sqrt(((xx - w * cx) / w) ** 2 + ((yy - h * cy) / h) ** 2 * 0.8)
        glow = np.clip(1 - r * 2.0, 0, 1) ** 2.2
        arr += glow.T[..., None] * np.array(pal["glow"], np.float32) * 0.45
        if mid == JJK:  # a low crimson ember at the bottom
            r2 = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 1.05) / h) ** 2 * 2)
            arr += (np.clip(1 - r2 * 2.0, 0, 1) ** 2).T[..., None] * np.array(pal["red"], np.float32) * 0.35
        # Fine film grain / scanlines for a cinematic texture.
        grain = np.random.default_rng(3).normal(0, 3.0, (w, h, 1)).astype(np.float32)
        arr += grain
        arr[:, ::3] *= 0.94
        base = pygame.Surface(size)
        pygame.surfarray.blit_array(base, np.clip(arr, 0, 255).astype(np.uint8))

        self._draw_motif(base, mid)

        # Bottom fade for legible text.
        fade = pygame.Surface((w, h), pygame.SRCALPHA)
        fh = int(h * 0.48)
        a = (np.clip(np.linspace(0, 1, fh), 0, 1) ** 1.4 * 215).astype(np.uint8)
        fade_arr = pygame.surfarray.pixels_alpha(fade)
        fade_arr[:, h - fh:] = a[None, :]
        del fade_arr
        base.blit(fade, (0, 0))

        self._card_text(base, mid, w, h)
        return _rounded_alpha(base, max(8, int(h * 0.035)))

    def _card_text(self, base: pygame.Surface, mid: str, w: int, h: int) -> None:
        mode, pal = MODES[mid], PALETTE[mid]
        # Title sized to fit the card.
        size = int(h * 0.12)
        spacing = max(2, int(h * 0.008))
        while size > 14 and self._txt(mode.title, size, WHITE, True, spacing).get_width() > w * 0.86:
            size -= 2
        title = self._txt(mode.title, size, WHITE, True, spacing)
        glow_col = pal["accent"] if mid == BLEACH else pal["red"]
        gl = self._txt(mode.title, size, glow_col, True, spacing)
        ty = int(h * 0.70)
        pad = 30
        layer = pygame.Surface((gl.get_width() + 2 * pad, gl.get_height() + 2 * pad))
        layer.blit(gl, (pad, pad))
        layer = _blur(layer, 5)
        for _ in range(2):
            base.blit(layer, layer.get_rect(center=(w // 2, ty)), special_flags=pygame.BLEND_ADD)
        base.blit(self._txt(mode.title, size, (0, 0, 0), True, spacing), title.get_rect(center=(w // 2 + 2, ty + 3)))
        base.blit(title, title.get_rect(center=(w // 2, ty)))
        # Accent rule.
        ry = ty + title.get_height() // 2 + int(h * 0.02)
        rl = int(w * 0.18)
        pygame.draw.line(base, pal["accent"], (w // 2 - rl, ry), (w // 2 + rl, ry), 2)
        if mid == JJK:
            pygame.draw.line(base, pal["red"], (w // 2 - rl // 3, ry + 4), (w // 2 + rl // 3, ry + 4), 2)
        # Subtitle.
        fs = max(11, int(h * 0.038))
        y = ry + int(h * 0.035)
        for line in self._wrap(mode.subtitle, fs, int(w * 0.84))[:2]:
            img = self._txt(line, fs, (205, 210, 225))
            base.blit(img, img.get_rect(midtop=(w // 2, y)))
            y += img.get_height() + 2
        # Ability strip.
        labels = [a.label.split(":")[-1].strip() for a in mode.abilities]
        sep = "  \u00b7  "
        fs2 = max(10, int(h * 0.03))
        lines = [sep.join(labels)]
        if self._font(fs2).size(lines[0])[0] > w * 0.88:  # two balanced lines
            half = (len(labels) + 1) // 2
            lines = [sep.join(labels[:half]), sep.join(labels[half:])]
        y = h - int(h * 0.05) - (len(lines) - 1) * (fs2 + 4)
        for line in lines:
            img = self._txt(line, fs2, _mix(pal["accent"], WHITE, 0.3))
            base.blit(img, img.get_rect(center=(w // 2, y)))
            y += fs2 + 4

    def _draw_motif(self, base: pygame.Surface, mid: str) -> None:
        """Draw the mode motif at 2x and downsample (cheap anti-aliasing, built once)."""
        w, h = base.get_size()
        big = pygame.transform.smoothscale(base, (w * 2, h * 2))
        (self._motif_katana if mid == BLEACH else self._motif_domain)(big, w * 2, h * 2)
        base.blit(pygame.transform.smoothscale(big, (w, h)), (0, 0))

    def _motif_katana(self, base: pygame.Surface, w: int, h: int) -> None:
        """A drawn katana: curved blade, tsuba and wrapped grip, with a cold glow."""
        guard = np.array([w * 0.24, h * 0.60])
        tip = np.array([w * 0.90, h * 0.07])
        axis = tip - guard
        L = float(np.hypot(*axis))
        u = axis / L
        n = np.array([-u[1], u[0]])  # perpendicular (towards upper-left: the spine side)
        bw = L * 0.032
        edge, spine = [], []
        steps = 24
        for i in range(steps + 1):
            s = i / steps
            curve = math.sin(math.pi * s * 0.9) * L * 0.035  # sori
            c = guard + axis * s + n * curve
            width = bw * (1 - 0.25 * s)
            if s > 0.9:  # kissaki taper
                width *= max(0.0, (1 - s) / 0.1) ** 0.6
            spine.append(c + n * width * 0.5)
            edge.append(c - n * width * 0.5)
        blade = [tuple(p) for p in spine] + [tuple(p) for p in reversed(edge)]

        # Glow layer (blurred blade) added under the silhouette.
        glow = pygame.Surface((w, h))
        pygame.draw.polygon(glow, (60, 130, 255), blade)
        pygame.draw.lines(glow, (160, 210, 255), False, [tuple(p) for p in edge], 6)
        glow = _blur(glow, 10)
        base.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)
        base.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)

        pygame.draw.polygon(base, (20, 26, 40), blade)
        pygame.draw.lines(base, (205, 230, 255), False, [tuple(p) for p in edge], 2)
        pygame.draw.lines(base, (70, 90, 125), False, [tuple(p) for p in spine], 1)
        # Hamon (temper line) — a faint wavy line along the blade.
        ham = []
        for i, (a, b) in enumerate(zip(spine, edge)):
            k = 0.62 + 0.08 * math.sin(i * 1.7)
            ham.append(tuple(a * (1 - k) + b * k))
        pygame.draw.lines(base, (95, 125, 170), False, ham[:-3], 1)
        # Tsuba (guard).
        g = tuple(guard)
        tsuba = [tuple(guard + n * bw * 1.7 * math.cos(a) + u * bw * 0.45 * math.sin(a))
                 for a in np.linspace(0, 2 * math.pi, 20)]
        pygame.draw.polygon(base, (36, 34, 30), tsuba)
        pygame.draw.polygon(base, (150, 135, 95), tsuba, 2)
        # Grip with diamond wrap.
        grip_end = guard - u * L * 0.30
        gw = bw * 0.75
        gp = [tuple(guard - u * bw * 0.3 + n * gw), tuple(grip_end + n * gw * 0.9),
              tuple(grip_end - n * gw * 0.9), tuple(guard - u * bw * 0.3 - n * gw)]
        pygame.draw.polygon(base, (14, 14, 20), gp)
        for i in range(1, 9):
            a = guard - u * (bw * 0.3 + (L * 0.30 - bw * 0.3) * i / 9)
            b = guard - u * (bw * 0.3 + (L * 0.30 - bw * 0.3) * (i - 0.5) / 9)
            pygame.draw.line(base, (60, 70, 105), tuple(a + n * gw * 0.9), tuple(b - n * gw * 0.9), 2)
            pygame.draw.line(base, (60, 70, 105), tuple(a - n * gw * 0.9), tuple(b + n * gw * 0.9), 2)
        pygame.draw.circle(base, (150, 135, 95), tuple(int(v) for v in grip_end), int(gw * 0.9))
        # Edge in card-relative coordinates, for the animated shimmer.
        self._katana_edge = [np.array([q[0] / w, q[1] / h]) for q in edge]
        del g

    def _motif_domain(self, base: pygame.Surface, w: int, h: int) -> None:
        """Abstract domain: concentric rings, rune ticks, a void core."""
        c = (w // 2, int(h * 0.36))
        R = int(min(w, h) * 0.36)
        glow = pygame.Surface((w, h))
        pygame.draw.circle(glow, (110, 50, 220), c, R, max(3, R // 18))
        pygame.draw.circle(glow, (200, 40, 70), c, int(R * 0.62), 3)
        glow = _blur(glow, 10)
        base.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)
        base.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)
        # Void core: darker disc with a soft violet rim.
        core = pygame.Surface((R * 2, R * 2), pygame.SRCALPHA)
        for i in range(12):
            k = i / 12
            pygame.draw.circle(core, (2, 0, 6, int(30 + 14 * i)), (R, R), int(R * 0.6 * (1 - k * 0.6)))
        base.blit(core, (c[0] - R, c[1] - R))
        pygame.draw.circle(base, (170, 130, 255), c, R, 1)
        pygame.draw.circle(base, (90, 60, 160), c, int(R * 0.86), 1)
        pygame.draw.circle(base, (220, 70, 95), c, int(R * 0.62), 1)
        pygame.draw.circle(base, (60, 40, 110), c, int(R * 1.12), 1)
        # Rune ticks between the outer rings.
        for i in range(48):
            a = i / 48 * 2 * math.pi
            r1 = R * (0.88 if i % 4 else 0.80)
            p1 = (c[0] + math.cos(a) * r1, c[1] + math.sin(a) * r1)
            p2 = (c[0] + math.cos(a) * R * 0.98, c[1] + math.sin(a) * R * 0.98)
            pygame.draw.line(base, (140, 110, 230) if i % 4 else (235, 90, 110), p1, p2, 1)
        # Abstract seal: a dotted inner ring, four radial cuts and a small core ring.
        for i in range(36):
            a = i / 36 * 2 * math.pi
            pygame.draw.circle(base, (120, 90, 210), (int(c[0] + math.cos(a) * R * 0.48),
                                                      int(c[1] + math.sin(a) * R * 0.48)), max(1, R // 110))
        for k in range(4):
            a = k * math.pi / 2 + math.pi / 4
            p1 = (c[0] + math.cos(a) * R * 0.18, c[1] + math.sin(a) * R * 0.18)
            p2 = (c[0] + math.cos(a) * R * 0.40, c[1] + math.sin(a) * R * 0.40)
            pygame.draw.line(base, (200, 70, 100), p1, p2, max(1, R // 90))
        pygame.draw.circle(base, (150, 110, 255), c, int(R * 0.12), max(1, R // 90))

    # Card animation -------------------------------------------------------
    def _animate_bleach(self, surf, rect: pygame.Rect, t: float, focus: float) -> None:
        pal = PALETTE[BLEACH]
        rnd = self._rand[BLEACH]
        energy = 0.45 + 0.55 * focus
        n = 34
        for i in range(n):
            r0, r1, r2, r3, r4, r5 = rnd[i]
            y = (r1 - t * (0.05 + 0.12 * r2)) % 1.0
            x = r0 + 0.015 * math.sin(t * (1 + r3 * 2) + i)
            fade = min(1.0, y * 3) * min(1.0, (1 - y) * 4)
            k = fade * energy * (0.4 + 0.6 * r4)
            if k < 0.06:
                continue
            px, py = rect.x + int(x * rect.w), rect.y + int(y * rect.h)
            if r5 < 0.25:  # reiatsu streak
                ln = int(rect.h * (0.04 + 0.05 * r3))
                pygame.draw.line(surf, _scale(pal["mote"], k * 0.8), (px, py), (px, py + ln), 1)
            else:
                g = self._glow(_scale(pal["mote"], round(k * 4) / 4), int(4 + 6 * r3))
                surf.blit(g, g.get_rect(center=(px, py)), special_flags=pygame.BLEND_ADD)

        # Shimmer travelling along the blade edge.
        edge = getattr(self, "_katana_edge", None)
        if edge is not None:
            k = (t * 0.45) % 1.4
            if k < 1.0:
                idx = k * (len(edge) - 1)
                i0 = int(idx)
                i1 = min(len(edge) - 1, i0 + 1)
                q = edge[i0] * (1 - (idx - i0)) + edge[i1] * (idx - i0)
                cp = (rect.x + int(q[0] * rect.w), rect.y + int(q[1] * rect.h))
                g = self._glow(_scale(pal["hi"], round(0.9 * energy * 4) / 4), 22)
                surf.blit(g, g.get_rect(center=cp), special_flags=pygame.BLEND_ADD)

        # A Getsuga-like crescent sweeps across now and then.
        period = 3.2
        ph = (t % period) / period
        if ph < 0.35:
            k = ph / 0.35
            cx = rect.x + int(rect.w * (-0.15 + 1.3 * k))
            cy = rect.y + int(rect.h * 0.40)
            rad = int(rect.h * 0.26)
            alpha = math.sin(math.pi * k) * energy
            arc_rect = pygame.Rect(0, 0, rad * 2, int(rad * 2.4))
            arc_rect.center = (cx - rad, cy)
            for j, (col, wd) in enumerate(((pal["glow"], 9), (pal["mote"], 5), (pal["hi"], 2))):
                c = _scale(col, alpha * (0.5 + 0.25 * j))
                r = arc_rect.inflate(-j * 4, -j * 4)
                pygame.draw.arc(surf, c, r, -math.pi * 0.42, math.pi * 0.42, wd)

    def _animate_jjk(self, surf, rect: pygame.Rect, t: float, focus: float) -> None:
        pal = PALETTE[JJK]
        rnd = self._rand[JJK]
        energy = 0.45 + 0.55 * focus
        c = (rect.centerx, rect.y + int(rect.h * 0.36))
        R = int(min(rect.w, rect.h) * 0.36)
        # Rotating domain segments.
        for j, (rr, speed, col, segs) in enumerate(((1.05, 0.35, (150, 110, 255), 6),
                                                    (0.74, -0.55, (230, 70, 95), 4),
                                                    (1.2, 0.18, (80, 140, 255), 3))):
            r = int(R * rr)
            box = pygame.Rect(c[0] - r, c[1] - r, 2 * r, 2 * r)
            for k in range(segs):
                a0 = t * speed + k * 2 * math.pi / segs
                pygame.draw.arc(surf, _scale(col, 0.85 * energy), box, a0, a0 + 2 * math.pi / segs * 0.45, 2)
        # Blue and Red orbiting in opposite directions; brief violet flare when they meet.
        for col, sign, ph in ((pal["mote"], 1, 0.0), (pal["red"], -1, math.pi)):
            a = sign * t * 1.1 + ph
            p = (c[0] + int(math.cos(a) * R * 0.45), c[1] + int(math.sin(a) * R * 0.45 * 0.6))
            g = self._glow(_scale(col, energy), int(R * 0.22))
            surf.blit(g, g.get_rect(center=p), special_flags=pygame.BLEND_ADD)
            g2 = self._glow(_scale((255, 255, 255), energy * 0.8), int(R * 0.06))
            surf.blit(g2, g2.get_rect(center=p), special_flags=pygame.BLEND_ADD)
        meet = abs(math.cos(t * 1.1))  # 1 when the orbs overlap (a = 0 or pi)
        if meet > 0.92:
            k = (meet - 0.92) / 0.08
            g = self._glow(_scale((190, 120, 255), round(k * energy * 6) / 6), int(R * 0.5))
            surf.blit(g, g.get_rect(center=(c[0] + int(R * 0.45 * (1 if math.cos(t * 1.1) > 0 else -1)), c[1])),
                      special_flags=pygame.BLEND_ADD)
        # Cursed-energy wisps: wobbling smoke blobs rising and fading.
        cols = ((120, 60, 230), (70, 40, 160), (200, 40, 80), (60, 110, 240))
        for i in range(14):
            r0, r1, r2, r3, r4, r5 = rnd[i]
            y = (r1 - t * (0.04 + 0.06 * r2)) % 1.0
            x = r0 + 0.04 * math.sin(t * (0.8 + r3) + i * 1.3) * (1 - y)
            fade = min(1.0, y * 2.5) * min(1.0, (1 - y) * 3)
            k = round(fade * energy * (0.35 + 0.35 * r4) * 5) / 5
            if k <= 0:
                continue
            col = cols[i % len(cols)]
            rad = int(rect.w * (0.05 + 0.06 * r5) * (0.6 + 0.6 * (1 - y)))
            wob = int(rad * 0.25 * math.sin(t * 2 + i))
            g = self._glow(_scale(col, k * 0.55), max(4, rad // 4 * 4), 1.6)
            px, py = rect.x + int(x * rect.w), rect.y + int(y * rect.h)
            surf.blit(g, g.get_rect(center=(px + wob, py)), special_flags=pygame.BLEND_ADD)
            surf.blit(g, g.get_rect(center=(px - wob // 2, py - rad // 2)), special_flags=pygame.BLEND_ADD)
        # Electric sparks.
        for i in range(14, 40):
            r0, r1, r2, r3, r4, r5 = rnd[i]
            y = (r1 - t * (0.08 + 0.1 * r2)) % 1.0
            flick = 0.5 + 0.5 * math.sin(t * (9 + 6 * r3) + i)
            k = round(flick * energy * min(1.0, (1 - y) * 4) * 4) / 4
            if k <= 0.1:
                continue
            px, py = rect.x + int(r0 * rect.w), rect.y + int(y * rect.h)
            col = pal["mote"] if r5 < 0.6 else (230, 70, 110)
            g = self._glow(_scale(col, k), 5)
            surf.blit(g, g.get_rect(center=(px, py)), special_flags=pygame.BLEND_ADD)

    # --- ability panel ---------------------------------------------------
    def _panel_layout(self, size, mode_id: str):
        W, H = size
        n = len(MODES[mode_id].abilities)
        mx = int(W * 0.07)
        top = int(H * 0.265)
        bottom = int(H * 0.875)
        gutter = int((W - 2 * mx) * 0.17)
        row_h = min(int(H * 0.125), (bottom - top) // max(1, n))
        top += max(0, (bottom - top) - row_h * n) // 3  # fewer rows: sit a bit lower, not top-heavy
        x = mx + gutter
        return [pygame.Rect(x, top + i * row_h, W - mx - x, row_h - max(4, row_h // 10)) for i in range(n)], gutter

    def ability_row_rects(self, size, mode_id: str) -> List[pygame.Rect]:
        """Row hit boxes, index = MODES[mode_id].abilities index."""
        return self._panel_layout(size, mode_id)[0]

    def draw_ability_panel(self, surf: pygame.Surface, t: float, mode_id: str, key_labels: Dict[str, str],
                           selected: int, backdrop: Optional[pygame.Surface], footer: str,
                           gesture_hints: Dict[str, str]) -> None:
        W, H = surf.get_size()
        s = H / 720.0
        mode, pal = MODES[mode_id], PALETTE[mode_id]
        self._backdrop(surf, backdrop, pal["glow"], 185)
        # Faint mode motif on the right and drifting particles.
        motif = self._cached(("pmotif", mode_id, W, H), lambda: self._panel_motif(mode_id, (W, H)))
        surf.blit(motif, (0, 0), special_flags=pygame.BLEND_ADD)
        self._panel_particles(surf, mode_id, t, W, H)

        mx = int(W * 0.07)
        accent, accent2 = pal["accent"], pal["accent2"]
        self._blit_txt(surf, "TECHNIQUES" if mode_id == JJK else "ABILITIES", int(13 * s + 4),
                       _scale(accent, 0.95), (mx, int(H * 0.06)), bold=True, spacing=int(5 * s + 1))
        tr = self._blit_txt(surf, mode.title, int(52 * s + 6), WHITE, (mx - 2, int(H * 0.085)), bold=True,
                            spacing=int(3 * s + 1))
        pygame.draw.line(surf, accent, (mx, tr.bottom + 4), (mx + int(W * 0.08), tr.bottom + 4), 3)
        if mode_id == JJK:
            pygame.draw.line(surf, accent2, (mx + int(W * 0.08) + 6, tr.bottom + 4),
                             (mx + int(W * 0.11), tr.bottom + 4), 3)
        self._blit_txt(surf, mode.subtitle, int(17 * s + 3), (190, 196, 215), (mx, tr.bottom + int(14 * s)))

        rects, gutter = self._panel_layout((W, H), mode_id)
        selected = int(selected)
        prev_cat = None
        groups: List[Tuple[str, int, int]] = []
        for i, spec in enumerate(mode.abilities):
            if spec.category != prev_cat:
                groups.append((spec.category, i, i))
                prev_cat = spec.category
            else:
                groups[-1] = (groups[-1][0], groups[-1][1], i)
        # Category gutter: label + vertical rule spanning the group.
        for cat, i0, i1 in groups:
            r0, r1 = rects[i0], rects[i1]
            gx = mx
            col = accent2 if cat == "Domain Expansion" and mode_id == JJK else accent
            pygame.draw.line(surf, _scale(col, 0.75), (gx, r0.top + 4), (gx, r1.bottom - 4), 2)
            y = r0.top + 4
            for line in self._wrap(cat.upper(), int(12 * s + 3), gutter - 36, True):
                self._blit_txt(surf, line, int(12 * s + 3), col, (gx + 12, y), bold=True)
                y += int(16 * s + 4)

        for i, (spec, rect) in enumerate(zip(mode.abilities, rects)):
            self._draw_row(surf, rect, spec, key_labels.get(spec.id, "?"), gesture_hints.get(spec.id, ""),
                           i == selected, t, mode_id, s)

        if footer:
            self._footer(surf, footer, (W // 2, int(H * 0.94)), s, accent)

    def _footer(self, surf, footer: str, center, s: float, accent) -> None:
        """Footer like "ENTER: start  ESC: back" -> key pills; plain text otherwise."""
        parts = [p.strip() for p in footer.replace("·", "  ").split("  ") if p.strip()]
        items = []
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                items.append((k.strip(), v.strip()))
            else:
                items = []
                break
        if items:
            self._hint_row(surf, items, center, s, accent)
        else:
            self._blit_txt(surf, footer, int(15 * s + 4), DIM, center, "center")

    def _draw_row(self, surf, rect: pygame.Rect, spec, key: str, gesture: str, selected: bool, t: float,
                  mode_id: str, s: float) -> None:
        pal = PALETTE[mode_id]
        accent = pal["accent2"] if (mode_id == JJK and spec.category == "Domain Expansion") else pal["accent"]
        radius = max(6, rect.h // 6)
        bg_key = ("rowbg", rect.w, rect.h, selected, accent)

        def build():
            p = pygame.Surface(rect.size, pygame.SRCALPHA)
            if selected:
                arr = np.zeros((rect.w, rect.h, 4), np.uint8)
                k = np.linspace(1, 0.25, rect.w, dtype=np.float32)[:, None]
                for ch in range(3):
                    arr[..., ch] = np.clip(accent[ch] * 0.35 * k + 10, 0, 255).astype(np.uint8)
                arr[..., 3] = (np.clip(110 + 90 * k, 0, 255)).astype(np.uint8)
                pygame.surfarray.blit_array(p, arr[..., :3])
                pygame.surfarray.pixels_alpha(p)[:] = arr[..., 3]
                p = _rounded_alpha(p, radius)
            else:
                pygame.draw.rect(p, (12, 14, 24, 150), p.get_rect(), border_radius=radius)
            pygame.draw.rect(p, (*(accent if selected else (70, 76, 100)), 200 if selected else 90),
                             p.get_rect(), 1, border_radius=radius)
            return p

        surf.blit(self._cached(bg_key, build), rect)
        if selected:
            pygame.draw.rect(surf, accent, (rect.x, rect.y + 6, 4, rect.h - 12), border_radius=2)
            sx = rect.x + int(((t * 0.35) % 1.0) * rect.w)
            g = self._glow(_scale(accent, 0.35), max(10, rect.h // 2))
            clip = surf.get_clip()
            surf.set_clip(rect)
            surf.blit(g, g.get_rect(center=(sx, rect.centery)), special_flags=pygame.BLEND_ADD)
            surf.set_clip(clip)

        # Key badge.
        bs = int(rect.h * 0.58)
        badge = pygame.Rect(rect.x + int(rect.h * 0.24), 0, bs, bs)
        badge.centery = rect.centery
        if selected:
            g = self._glow(_scale(accent, 0.7), bs)
            surf.blit(g, g.get_rect(center=badge.center), special_flags=pygame.BLEND_ADD)
        pygame.draw.rect(surf, accent if selected else (26, 30, 46), badge, border_radius=max(4, bs // 5))
        pygame.draw.rect(surf, _mix(accent, WHITE, 0.4), badge, 1, border_radius=max(4, bs // 5))
        ks = int(bs * (0.5 if len(key) <= 2 else 0.34))
        self._blit_txt(surf, key, ks, (10, 12, 22) if selected else WHITE, badge.center, "center", bold=True,
                       shadow=False)

        # Gesture hint column on the right.
        gx = rect.right - int(rect.h * 0.3)
        gw = 0
        if gesture:
            gs = int(13 * s + 3)
            lab = self._txt("GESTURE", int(10 * s + 3), _scale(accent, 0.9), True, 2)
            val = self._txt(self._ellipsize(gesture, gs, int(rect.w * 0.24)), gs, WHITE if selected else DIM)
            gw = max(lab.get_width(), val.get_width())
            y = rect.centery - (lab.get_height() + val.get_height()) // 2
            surf.blit(lab, lab.get_rect(topright=(gx, y)))
            surf.blit(val, val.get_rect(topright=(gx, y + lab.get_height())))
            pygame.draw.line(surf, (70, 76, 100), (gx - gw - 14, rect.y + 10), (gx - gw - 14, rect.bottom - 10), 1)

        # Label + chip + description.
        x = badge.right + int(rect.h * 0.25)
        text_w = (gx - gw - 28) - x
        ls = int(rect.h * 0.27)
        ds = max(11, int(rect.h * 0.19))
        total = self._font(ls, True).get_height() + self._font(ds).get_height()
        y = rect.centery - total // 2
        lr = self._blit_txt(surf, spec.label, ls, WHITE, (x, y), bold=True)
        if spec.category == "Domain Expansion":
            chip_txt = self._txt("DOMAIN", int(ds * 0.8), (12, 10, 20), True, 2)
            chip = pygame.Rect(lr.right + 10, 0, chip_txt.get_width() + 14, chip_txt.get_height() + 4)
            chip.centery = lr.centery
            if chip.right < x + text_w:
                pygame.draw.rect(surf, accent, chip, border_radius=chip.h // 2)
                surf.blit(chip_txt, chip_txt.get_rect(center=chip.center))
        desc = self._ellipsize(spec.description, ds, max(40, text_w))
        self._blit_txt(surf, desc, ds, (200, 205, 222) if selected else DIM, (x, lr.bottom + 1))

    def _panel_motif(self, mode_id: str, size) -> pygame.Surface:
        """Large faint motif on the right side of the panel (RGB, BLEND_ADD)."""
        W, H = size
        cw, ch = int(H * 0.8), int(H * 0.9)
        card = pygame.Surface((cw, ch))
        self._draw_motif(card, mode_id)
        card.fill((60, 60, 60), special_flags=pygame.BLEND_RGB_MULT)  # ~25% intensity
        out = pygame.Surface((W, H))
        out.blit(card, (W - cw + int(cw * 0.08), int(H * 0.02)))
        return out

    def _panel_particles(self, surf, mode_id: str, t: float, W: int, H: int) -> None:
        pal = PALETTE[mode_id]
        rnd = self._rand[mode_id]
        for i in range(24):
            r0, r1, r2, r3, r4, r5 = rnd[40 + i % 24]
            y = (r1 - t * (0.03 + 0.05 * r2)) % 1.0
            x = (r0 + 0.02 * math.sin(t + i)) % 1.0
            fade = min(1.0, y * 3) * min(1.0, (1 - y) * 3)
            k = round(fade * (0.3 + 0.4 * r4) * 4) / 4
            if k <= 0:
                continue
            col = pal["mote"] if (mode_id == BLEACH or r5 < 0.6) else pal["accent2"]
            g = self._glow(_scale(col, k), int(4 + 5 * r3))
            surf.blit(g, g.get_rect(center=(int(x * W), int(y * H))), special_flags=pygame.BLEND_ADD)

    # --- transition -----------------------------------------------------
    def draw_transition(self, surf: pygame.Surface, p: float, mode_id: str) -> None:
        """Overlay for a menu <-> camera switch, ``p`` in 0..1.

        Coverage rises from 0 to fully opaque at ``p == 0.5`` and falls back to
        0 at ``p == 1``: draw the old screen underneath while ``p < 0.5`` and the
        new one afterwards. BLEACH: a diagonal slash wipe; JJK: a domain circle
        that closes over the screen and opens onto the new one.
        """
        p = max(0.0, min(1.0, float(p)))
        if p <= 0.0 or p >= 1.0:
            return
        W, H = surf.get_size()
        pal = PALETTE.get(mode_id, PALETTE[BLEACH])
        front = _ease(min(1.0, p * 2))
        back = _ease(max(0.0, p * 2 - 1))
        dark = (4, 6, 14) if mode_id != JJK else (5, 2, 10)
        if mode_id == JJK:
            c = (W // 2, H // 2)
            maxr = int(math.hypot(W, H) / 2) + 4
            ro, ri = int(maxr * front), int(maxr * back)
            if ro <= ri:
                return
            pygame.draw.circle(surf, dark, c, ro, ro - ri if ri > 0 else 0)
            for r, col, wd in ((ro, pal["accent"], 4), (ro - 8, pal["red"], 2), (ri, pal["accent"], 4),
                               (ri + 8, pal["red"], 2)):
                if 4 < r < maxr:
                    pygame.draw.circle(surf, col, c, r, wd)
            if 0 < ri < maxr * 0.9:
                for k in range(8):  # rotating rune arcs on the inner edge
                    rr = max(ri, 20) + 18
                    a0 = k * math.pi / 4 + p * 6
                    box = pygame.Rect(c[0] - rr, c[1] - rr, 2 * rr, 2 * rr)
                    if rr < ro:
                        pygame.draw.arc(surf, _scale(pal["accent"], 0.8), box, a0, a0 + 0.35, 2)
        else:
            slant = int(H * 0.45)
            span = W + slant + 40
            xf = int(-slant + span * front)
            xb = int(-slant + span * back)
            if xf <= xb:
                return
            poly = [(xb, H), (xf, H), (xf + slant, 0), (xb + slant, 0)]
            pygame.draw.polygon(surf, dark, poly)
            for x, wide in ((xf, True), (xb, False)):
                if back == 0 and not wide:
                    continue
                if front >= 1.0 and wide and back > 0:
                    continue
                pygame.draw.line(surf, pal["glow"], (x, H), (x + slant, 0), 10 if wide else 6)
                pygame.draw.line(surf, pal["mote"], (x, H), (x + slant, 0), 5 if wide else 3)
                pygame.draw.line(surf, (245, 250, 255), (x, H), (x + slant, 0), 2)
