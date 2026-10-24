"""Mode selector / ability panel / transition drawing and the mode-aware HUD."""

import os
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from bleach_ar.abilities import AbilityManager, Context  # noqa: E402
from bleach_ar.config import Config  # noqa: E402
from bleach_ar.fx import Canvas, ParticleSystem, Shaker  # noqa: E402
from bleach_ar.hud import HUD  # noqa: E402
from bleach_ar.menus import Menus  # noqa: E402
from bleach_ar.modes import MODE_ORDER, MODES  # noqa: E402
from bleach_ar.synthetic import make_tracking  # noqa: E402

SIZES = [(960, 540), (1280, 720)]
FOOTER = "ENTER: start  ESC: back"


def setUpModule():
    pygame.init()
    pygame.display.set_mode((1, 1))


def _backdrop(size):
    w, h = size
    arr = np.zeros((w, h, 3), np.uint8)
    arr[..., 0] = np.linspace(20, 200, w, dtype=np.uint8)[:, None]
    arr[..., 1] = np.linspace(30, 160, h, dtype=np.uint8)[None, :]
    surf = pygame.Surface(size)
    pygame.surfarray.blit_array(surf, arr)
    return surf


class TestMenus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config().data
        cls.hud = HUD(cls.cfg["keys"])
        cls.menus = Menus(cls.hud.fonts)
        cls.keys = {
            "bleach": {a.id: cls.cfg["keys"][a.id].upper() for a in MODES["bleach"].abilities},
            "jjk": {k: v.upper() for k, v in cls.cfg["modes"]["jjk"]["keys"].items()},
        }

    def test_draw_all_screens(self):
        for size in SIZES:
            surf = pygame.Surface(size)
            for bd in (None, _backdrop(size)):
                for sel in range(len(MODE_ORDER)):
                    for i in range(3):
                        self.menus.draw_mode_select(surf, 1.0 + i / 30, sel, bd)
                self.menus.draw_mode_select(surf, 2.0, 0, bd, message="Opening camera...")
                self.menus.draw_mode_select(surf, 2.1, 1, bd, error="Camera failed\nCheck permissions")
                for mid in MODE_ORDER:
                    n = len(MODES[mid].abilities)
                    for sel in (0, n - 1, -1):
                        self.menus.draw_ability_panel(surf, 0.5, mid, self.keys[mid], sel, bd, FOOTER,
                                                      {MODES[mid].abilities[0].id: "Open palm"})
                    self.menus.draw_ability_panel(surf, 0.5, mid, {}, 0, bd, "", {})
            for mid in MODE_ORDER:
                for p in (0.0, 0.1, 0.5, 0.75, 1.0):
                    self.menus.draw_transition(surf, p, mid)
            # Something visible was drawn.
            self.assertGreater(pygame.transform.average_color(surf)[:3], (0, 0, 0))

    def test_card_rects(self):
        for size in SIZES:
            rects = self.menus.mode_card_rects(size)
            self.assertEqual(len(rects), 2)
            self.assertFalse(rects[0].colliderect(rects[1]))
            bounds = pygame.Rect((0, 0), size)
            for r in rects:
                self.assertTrue(bounds.contains(r))
                self.assertGreater(r.w * r.h, 0)

    def test_row_rects(self):
        for size in SIZES:
            bounds = pygame.Rect((0, 0), size)
            for mid in MODE_ORDER:
                rects = self.menus.ability_row_rects(size, mid)
                self.assertEqual(len(rects), len(MODES[mid].abilities))
                for a, b in zip(rects, rects[1:]):
                    self.assertFalse(a.colliderect(b))
                self.assertTrue(all(bounds.contains(r) for r in rects))

    def test_frame_time(self):
        size = (1280, 720)
        surf = pygame.Surface(size)
        bd = _backdrop(size)
        jobs = {
            "mode_select": lambda t: self.menus.draw_mode_select(surf, t, int(t * 2) % 2, bd),
            "ability_panel": lambda t: self.menus.draw_ability_panel(surf, t, "jjk", self.keys["jjk"], 2, bd,
                                                                     FOOTER, {}),
            "transition": lambda t: self.menus.draw_transition(surf, t % 1.0, "jjk"),
        }
        for name, fn in jobs.items():
            for i in range(40):  # warm caches, incl. every selection-animation size
                fn(i / 30)
            times = []
            for i in range(60):
                t0 = time.perf_counter()
                fn(5 + i / 30)
                times.append((time.perf_counter() - t0) * 1000)
            mean = sum(times) / len(times)
            print(f"\n[menus] {name}: mean {mean:.2f} ms, max {max(times):.2f} ms")
            self.assertLess(mean, 12.0, name)


class TestModeHUD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config().data
        cls.hud = HUD(cls.cfg["keys"])

    def _setup(self, mode_id):
        ctx = Context(cfg=self.cfg, canvas=Canvas(0.33), particles=ParticleSystem(300), shaker=Shaker())
        mgr = AbilityManager(ctx)
        if mode_id:
            mgr.set_mode(mode_id)
        ctx.tracking = make_tracking(t=0.3)
        return ctx, mgr

    def _draw(self, ctx, abilities, **kw):
        surf = pygame.Surface((1280, 720))
        stats = {"fps": 30.0, "cam_fps": 30.0, "track_ms": 5.0, "fx_ms": 4.0}
        for help_on in (False, True):
            self.hud.show_help = help_on
            self.hud.draw(surf, ctx, abilities, stats, None, True, "medium", self.cfg["gestures"], **kw)
        self.hud.show_help = False

    def test_legacy(self):
        ctx, mgr = self._setup(None)
        self._draw(ctx, mgr.abilities)

    def test_bleach_mode(self):
        ctx, mgr = self._setup("bleach")
        keys = {a.id: self.cfg["keys"][a.id].upper() for a in MODES["bleach"].abilities}
        self._draw(ctx, mgr.active_abilities(), mode=MODES["bleach"], mode_keys=keys)

    def test_jjk_mode(self):
        ctx, mgr = self._setup("jjk")
        keys = {k: v.upper() for k, v in self.cfg["modes"]["jjk"]["keys"].items()}
        self._draw(ctx, mgr.active_abilities(), mode=MODES["jjk"], mode_keys=keys)
        # With a domain running and an attack cooling down.
        self.assertTrue(mgr.trigger("unlimited_void"))
        for _ in range(10):
            ctx.now += 1 / 30
            ctx.dt = 1 / 30
            mgr.update()
        self._draw(ctx, mgr.active_abilities(), mode=MODES["jjk"], mode_keys=keys)
        self._draw(ctx, mgr.active_abilities(), mode=MODES["jjk"], mode_keys=None)


if __name__ == "__main__":
    unittest.main()
