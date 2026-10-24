"""Pygame HUD: ability cards, cooldowns, transformation and tracking status,
title cards, help overlay, calibration panel, startup and error screens."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pygame

from .abilities import BANKAI, SHIKAI, Context, Phase

WHITE = (240, 240, 245)
DIM = (150, 155, 170)
PANEL = (10, 12, 20, 170)
ACCENT = (110, 190, 255)  # RGB (pygame), soul-reaper blue
SENBON_PINK = (255, 140, 200)  # RGB
LEVEL_COLORS = {"sealed": (200, 205, 215), SHIKAI: (120, 200, 255), BANKAI: (235, 60, 70)}

BANKAI_STYLE_NAMES = {"senbonzakura": "Senbonzakura", "tensa": "Tensa"}
BANKAI_STYLE_SHORT = {"senbonzakura": "Senbon", "tensa": "Tensa"}
ABILITY_ROWS = [("getsuga", "Getsuga Tensho"), ("shikai", "Shikai"), ("bankai", "Bankai"), ("cero", "Cero")]


def bgr_to_rgb(c) -> Tuple[int, int, int]:
    return int(c[2]), int(c[1]), int(c[0])


class Fonts:
    def __init__(self, path: Optional[str] = None):
        def make(size: int, bold: bool = False) -> pygame.font.Font:
            if path:
                try:
                    return pygame.font.Font(path, size)
                except (OSError, pygame.error):
                    print(f"[assets] could not load font {path}; using system font.")
            return pygame.font.SysFont("helveticaneue,arial,dejavusans", size, bold=bold)

        self.small = make(15)
        self.body = make(18)
        self.bold = make(18, True)
        self.big = make(30, True)
        self.title = make(96, True)
        self.sub = make(30)


class HUD:
    def __init__(self, keys: Dict[str, str], font_path: Optional[str] = None):
        self.keys = keys
        self.fonts = Fonts(font_path)
        self.show_help = False
        self.show_calibration = False
        self._panel_cache: Dict[Tuple[int, int], pygame.Surface] = {}

    # --- helpers ----------------------------------------------------------
    def _panel(self, surf: pygame.Surface, rect: pygame.Rect, alpha: int = 170) -> None:
        key = (rect.w, rect.h)
        panel = self._panel_cache.get(key)
        if panel is None:
            panel = pygame.Surface(key, pygame.SRCALPHA)
            pygame.draw.rect(panel, (10, 12, 20, alpha), panel.get_rect(), border_radius=10)
            pygame.draw.rect(panel, (90, 120, 170, 120), panel.get_rect(), 1, border_radius=10)
            self._panel_cache[key] = panel
        surf.blit(panel, rect.topleft)

    def _text(self, surf, text, font, color, pos, anchor="topleft", shadow=True) -> pygame.Rect:
        img = font.render(text, True, color)
        rect = img.get_rect(**{anchor: pos})
        if shadow:
            surf.blit(font.render(text, True, (0, 0, 0)), rect.move(2, 2))
        surf.blit(img, rect)
        return rect

    def key_label(self, action: str) -> str:
        k = self.keys.get(action, "?")
        return "ESC" if k == "escape" else k.upper()

    # --- in-game HUD --------------------------------------------------------
    def draw(self, surf: pygame.Surface, ctx: Context, abilities: dict, stats: dict,
             gesture_frame, gestures_enabled: bool, quality: str, calib: dict) -> None:
        W, H = surf.get_size()
        self._draw_abilities(surf, ctx, abilities, gesture_frame)
        self._draw_status(surf, ctx, stats, gestures_enabled, quality, W)
        self._draw_titles(surf, ctx, W, H)
        if stats.get("camera_lost"):
            self._banner(surf, "Camera signal lost - reconnecting...", W, H)
        elif not any(ctx.tracking.status().values()):
            self._text(surf, "No person detected - step into view", self.fonts.body, DIM,
                       (W // 2, H - 24), anchor="midbottom")
        if self.show_calibration:
            self._draw_calibration(surf, gesture_frame, calib, ctx, W, H)
        if self.show_help:
            self._draw_help(surf, W, H)
        else:
            self._text(surf, f"{self.key_label('help')}: controls", self.fonts.small, DIM, (W - 12, H - 10),
                       anchor="bottomright")

    def _draw_abilities(self, surf, ctx: Context, abilities: dict, gesture_frame) -> None:
        x, y, w = 14, 14, 270
        rect = pygame.Rect(x, y, w, 46 + 52 * len(ABILITY_ROWS) + 30)
        self._panel(surf, rect)
        level = ctx.level
        self._text(surf, "BLEACH AR", self.fonts.bold, ACCENT, (x + 14, y + 10))
        level_color = SENBON_PINK if (level == BANKAI and ctx.bankai_style == "senbonzakura") \
            else LEVEL_COLORS.get(level, WHITE)
        self._text(surf, level.upper(), self.fonts.bold, level_color,
                   (x + w - 14, y + 10), anchor="topright")
        yy = y + 44
        held = gesture_frame.held if gesture_frame else {}
        senbon = ctx.level == BANKAI and ctx.bankai_style == "senbonzakura"
        for name, label in ABILITY_ROWS:
            ab = abilities[name]
            key = self.key_label(name)
            if name == "getsuga" and senbon and "senbon_attack" in abilities:
                # The Getsuga key / swing commands the blade swarm in this Bankai.
                ab, label = abilities["senbon_attack"], "Senbonzakura"
            elif name == "bankai":
                label = "Bankai: " + BANKAI_STYLE_SHORT.get(ctx.bankai_style, ctx.bankai_style)
            pygame.draw.rect(surf, (40, 50, 70), (x + 12, yy, 26, 26), border_radius=6)
            self._text(surf, key, self.fonts.bold, WHITE, (x + 25, yy + 13), anchor="center", shadow=False)
            self._text(surf, label, self.fonts.body, WHITE, (x + 48, yy + 2))
            state, color, frac = self._ability_state(ab, ctx, ab.name if name == "getsuga" else name)
            self._text(surf, state, self.fonts.small, color, (x + w - 14, yy + 4), anchor="topright")
            bar = pygame.Rect(x + 48, yy + 28, w - 62, 5)
            pygame.draw.rect(surf, (45, 50, 65), bar, border_radius=3)
            if frac > 0:
                pygame.draw.rect(surf, color, (bar.x, bar.y, int(bar.w * frac), bar.h), border_radius=3)
            g = ctx.cfg["gestures"]["bindings"].get(name)
            if g in held and held[g] > 0 and ab.phase == Phase.IDLE:
                pygame.draw.rect(surf, (255, 220, 120), (bar.x, bar.y + 7, int(bar.w * held[g]), 3))
            yy += 52
        toggles = [("toggle_sword", "Sword", ctx.show_sword), ("toggle_aura", "Aura", ctx.show_aura),
                   ("toggle_eyes", "Eyes", ctx.show_eyes), ("toggle_mask", "Mask", ctx.show_mask)]
        tx = x + 12
        for action, label, on in toggles:
            txt = f"{self.key_label(action)} {label}"
            img = self.fonts.small.render(txt, True, (20, 25, 35) if on else DIM)
            pill = pygame.Rect(tx, yy, img.get_width() + 12, 22)
            pygame.draw.rect(surf, ACCENT if on else (45, 50, 65), pill, border_radius=11)
            surf.blit(img, img.get_rect(center=pill.center))
            tx += pill.w + 5

    def _ability_state(self, ab, ctx: Context, name: str) -> Tuple[str, tuple, float]:
        if ctx.locked_by and ctx.locked_by != name:
            return "LOCKED", (120, 120, 130), 0.0
        if ab.phase == Phase.IDLE:
            if name == "shikai" and ctx.level == SHIKAI:
                return "ACTIVE - press to seal", LEVEL_COLORS[SHIKAI], 1.0
            if name == "bankai" and ctx.level == BANKAI:
                return "ACTIVE", LEVEL_COLORS[BANKAI], 1.0
            if name == "shikai" and ctx.level == BANKAI:
                return "in Bankai", DIM, 0.0
            return "READY", (120, 230, 150), 1.0
        if ab.phase == Phase.COOLDOWN:
            d = ab.duration(Phase.COOLDOWN)
            return f"{ab.cooldown_left():.1f}s", (240, 170, 80), 1.0 - (ab.t / d if d else 1.0)
        return ab.phase.value, (255, 230, 120), ab.progress

    def _draw_status(self, surf, ctx: Context, stats: dict, gestures_enabled: bool, quality: str, W: int) -> None:
        w, h = 280, 148
        rect = pygame.Rect(W - w - 14, 14, w, h)
        self._panel(surf, rect)
        x, y = rect.x + 14, rect.y + 10
        fps = stats.get("fps", 0.0)
        fps_color = (120, 230, 150) if fps >= 25 else (240, 200, 90) if fps >= 15 else (240, 90, 90)
        self._text(surf, f"{fps:5.1f} FPS", self.fonts.bold, fps_color, (x, y))
        self._text(surf, quality.upper(), self.fonts.small, DIM, (rect.right - 14, y + 2), anchor="topright")
        self._text(surf, f"cam {stats.get('cam_fps', 0):4.1f}  track {stats.get('track_ms', 0):4.1f}ms  "
                         f"fx {stats.get('fx_ms', 0):4.1f}ms", self.fonts.small, DIM, (x, y + 26))
        status = ctx.tracking.status()
        xx = x
        for label in ("face", "body", "hands"):
            on = status[label]
            pygame.draw.circle(surf, (110, 230, 140) if on else (200, 70, 70), (xx + 6, y + 60), 6)
            r = self._text(surf, label, self.fonts.small, WHITE if on else DIM, (xx + 16, y + 51))
            xx = r.right + 14
        g = "ON" if gestures_enabled else "OFF"
        self._text(surf, f"Gestures {g}  ({self.key_label('toggle_gestures')})", self.fonts.small,
                   WHITE if gestures_enabled else DIM, (x, y + 76))
        mode = "hand-only" if ctx.sword_mode == "hand" else ctx.current_weapon()
        if ctx.level == BANKAI and ctx.bankai_style == "senbonzakura":
            mode = "blade swarm"
        self._text(surf, f"Weapon: {mode}  Eyes: {ctx.eye_style}", self.fonts.small, DIM, (x, y + 96))
        style = BANKAI_STYLE_NAMES.get(ctx.bankai_style, ctx.bankai_style)
        self._text(surf, f"Bankai style: {style}  ({self.key_label('cycle_bankai')})", self.fonts.small, DIM,
                   (x, y + 116))

    def _draw_titles(self, surf, ctx: Context, W: int, H: int) -> None:
        for t in ctx.titles:
            age = ctx.now - t.start
            k = age / t.duration if t.duration else 1.0
            alpha = min(1.0, age / 0.15) * (1.0 - max(0.0, (k - 0.7) / 0.3))
            scale = 1.0 + 0.25 * math.exp(-age * 10)
            color = bgr_to_rgb(t.color)
            img = self.fonts.title.render(t.text, True, color)
            # Long names (e.g. "SENBONZAKURA KAGEYOSHI") shrink to fit the window.
            scale *= min(1.0, W * 0.9 / max(1, img.get_width() * 1.25))
            if scale != 1.0:
                img = pygame.transform.smoothscale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
            glow = self.fonts.title.render(t.text, True, (255, 255, 255))
            glow = pygame.transform.smoothscale(glow, img.get_size())
            cx, cy = W // 2, int(H * 0.36)
            for img_i, off, a in ((glow, (0, 0), 0.35), (img, (0, 0), 1.0)):
                img_i.set_alpha(int(255 * alpha * a))
                surf.blit(img_i, img_i.get_rect(center=(cx + off[0], cy + off[1])))
            # Slash line under the title.
            line_w = int(img.get_width() * min(1.0, age / 0.25))
            pygame.draw.line(surf, color, (cx - line_w // 2, cy + img.get_height() // 2),
                             (cx + line_w // 2, cy + img.get_height() // 2), 3)
            if t.sub:
                center = (cx, cy + img.get_height() // 2 + 30)
                outline = self.fonts.sub.render(t.sub, True, (0, 0, 0))
                outline.set_alpha(int(220 * alpha))
                for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (2, 2)):
                    surf.blit(outline, outline.get_rect(center=(center[0] + dx, center[1] + dy)))
                sub = self.fonts.sub.render(t.sub, True, WHITE)
                sub.set_alpha(int(255 * alpha))
                surf.blit(sub, sub.get_rect(center=center))

    def _banner(self, surf, text: str, W: int, H: int) -> None:
        img = self.fonts.big.render(text, True, (255, 210, 120))
        rect = img.get_rect(center=(W // 2, H // 2))
        self._panel(surf, rect.inflate(40, 30), 210)
        surf.blit(img, rect)

    def _draw_help(self, surf, W: int, H: int) -> None:
        rows = [
            ("getsuga", "Getsuga Tensho (or swing fast); in Senbonzakura: blade attack"),
            ("shikai", "Shikai on/off (or hold two fingers up)"),
            ("bankai", "Bankai on/off (or hold both palms up above shoulders)"),
            ("cero", "Cero (or hold an open palm)"),
            ("toggle_sword", "Toggle Zanpakuto (make a fist to grip it)"),
            ("toggle_aura", "Toggle reiatsu aura"),
            ("toggle_eyes", "Toggle glowing eyes"),
            ("toggle_mask", "Toggle mask"),
            ("cycle_eye_style", "Cycle eye style"),
            ("cycle_weapon", "Cycle sealed weapon style"),
            ("cycle_bankai", "Bankai style Senbonzakura / Tensa (not during Bankai)"),
            ("toggle_sword_mode", "Sword / hand-only mode"),
            ("toggle_gestures", "Gestures on/off"),
            ("toggle_calibration", "Gesture calibration panel"),
            ("cycle_quality", "Quality low / medium / high"),
            ("toggle_debug", "Show tracking landmarks"),
            ("fullscreen", "Fullscreen"),
            ("next_camera", "Switch camera (e.g. built-in vs iPhone)"),
            ("reset", "Reset abilities and transformations"),
            ("help", "Hide this help"),
            ("exit", "Exit"),
        ]
        w, h = 560, 40 + 25 * len(rows)
        rect = pygame.Rect((W - w) // 2, (H - h) // 2, w, h)
        self._panel(surf, rect, 225)
        self._text(surf, "Controls", self.fonts.big, ACCENT, (rect.x + 20, rect.y + 6))
        y = rect.y + 46
        for action, desc in rows:
            self._text(surf, self.key_label(action), self.fonts.bold, (255, 220, 120), (rect.x + 24, y))
            self._text(surf, desc, self.fonts.body, WHITE, (rect.x + 100, y))
            y += 25

    def _draw_calibration(self, surf, gf, calib: dict, ctx: Context, W: int, H: int) -> None:
        w, h = 420, 236
        rect = pygame.Rect(14, H - h - 14, w, h)
        self._panel(surf, rect, 215)
        x, y = rect.x + 14, rect.y + 10
        self._text(surf, "Gesture calibration", self.fonts.bold, ACCENT, (x, y))
        y += 28
        for side in ("left", "right"):
            g = gf.per_hand.get(side, "-") if gf else "-"
            fingers = gf.fingers.get(side, {}) if gf else {}
            fs = " ".join(f"{n[0].upper()}{'+' if v else '.'}" for n, v in fingers.items()) or "no hand"
            self._text(surf, f"{side:>5}: {g:<12} {fs}", self.fonts.small, WHITE, (x, y))
            y += 22
        held = ", ".join(f"{k} {v:.0%}" for k, v in (gf.held.items() if gf else [])) or "-"
        self._text(surf, f"holding: {held}", self.fonts.small, WHITE, (x, y))
        y += 22
        speed, thr = ctx.sword.speed, calib["swing_speed"]
        self._text(surf, f"swing speed {speed:4.2f} / threshold {thr:.2f}", self.fonts.small, WHITE, (x, y))
        bar = pygame.Rect(x, y + 22, w - 28, 8)
        pygame.draw.rect(surf, (45, 50, 65), bar)
        pygame.draw.rect(surf, (255, 200, 90), (bar.x, bar.y, int(bar.w * min(1.0, speed / max(0.1, thr * 1.5))), 8))
        tx = bar.x + int(bar.w / 1.5)
        pygame.draw.line(surf, (240, 80, 80), (tx, bar.y - 3), (tx, bar.bottom + 3), 2)
        y += 40
        lines: List[str] = [
            f"[ / ] swing threshold     - / = hold time ({calib['hold_seconds']:.1f}s)",
            f", / . finger extend ratio ({calib['finger_extend_ratio']:.2f})",
            "S save to config.json",
        ]
        for line in lines:
            self._text(surf, line, self.fonts.small, DIM, (x, y))
            y += 20

    # --- full screens -----------------------------------------------------
    def draw_startup(self, surf: pygame.Surface, t: float, message: str = "", error: str = "") -> None:
        W, H = surf.get_size()
        surf.fill((6, 8, 14))
        # Animated reiatsu streaks behind the title.
        for i in range(14):
            phase = t * 0.6 + i * 0.7
            x = int((i / 14) * W + 40 * math.sin(phase))
            h = int(H * (0.25 + 0.2 * (0.5 + 0.5 * math.sin(phase * 1.3))))
            col = (30 + 10 * (i % 3), 60 + 6 * i, 120 + 8 * i)
            pygame.draw.line(surf, col, (x, H), (x + 20, H - h), 2)
        title = self.fonts.title.render("BLEACH AR", True, (235, 240, 255))
        surf.blit(title, title.get_rect(center=(W // 2, int(H * 0.36))))
        pygame.draw.line(surf, ACCENT, (W // 2 - 260, int(H * 0.45)), (W // 2 + 260, int(H * 0.45)), 3)
        self._text(surf, "Real-time Soul Reaper combat in your webcam", self.fonts.sub, DIM,
                   (W // 2, int(H * 0.5)), anchor="center")
        if error:
            y = int(H * 0.6)
            for line in error.splitlines():
                self._text(surf, line, self.fonts.body, (255, 140, 120), (W // 2, y), anchor="center")
                y += 26
            self._text(surf, "ENTER: retry    ESC: quit", self.fonts.bold, WHITE, (W // 2, y + 20), anchor="center")
        elif message:
            self._text(surf, message, self.fonts.bold, WHITE, (W // 2, int(H * 0.66)), anchor="center")
        else:
            blink = 0.5 + 0.5 * math.sin(t * 4)
            col = tuple(int(140 + 115 * blink) for _ in range(3))
            self._text(surf, "Press SPACE or ENTER to start", self.fonts.big, col, (W // 2, int(H * 0.66)),
                       anchor="center")
            self._text(surf, "H: controls in game    ESC: quit", self.fonts.body, DIM, (W // 2, int(H * 0.74)),
                       anchor="center")
