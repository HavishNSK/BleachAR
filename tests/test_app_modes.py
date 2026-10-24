"""The real app's menu flow and in-game mode switching, driven by scripted
key events with SDL's dummy video driver (no window, no camera)."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from bleach_ar.abilities import Phase  # noqa: E402
from bleach_ar.app import BleachAR  # noqa: E402
from bleach_ar.config import Config  # noqa: E402


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)


class ScriptedEvents:
    """Replaces pygame.event.get: one batch of events per call, then empty."""

    def __init__(self, batches):
        self.batches = list(batches)

    def __call__(self, *a, **kw):
        return self.batches.pop(0) if self.batches else []


_REAL_GET = pygame.event.get


class TestMenuFlow(unittest.TestCase):
    def make_app(self, batches, mode=None):
        app = BleachAR(Config(), demo=True, max_frames=0, mode=mode)
        app.pg.event.get = ScriptedEvents(batches)
        return app

    def tearDown(self):
        pygame.event.get = _REAL_GET

    def test_choose_jjk_from_menu(self):
        app = self.make_app([[key(pygame.K_RIGHT)], [key(pygame.K_RETURN)], [key(pygame.K_RETURN)]])
        try:
            self.assertTrue(app._menu_loop(in_game=False))
            self.assertEqual(app.ctx.mode, "jjk")
            self.assertIsNotNone(app.camera, "pipeline started once a mode was chosen")
            # JJK keys now trigger JJK abilities; the Bleach key "1" maps to Blue.
            self.assertEqual(app.ability_keymap[pygame.K_1], "blue")
            self.assertEqual(app.ability_keymap[pygame.K_z], "unlimited_void")
            self.assertEqual(app.gestures.cfg["bindings"]["unlimited_void"], "two_fingers")
        finally:
            app.close()

    def test_escape_at_mode_select_quits(self):
        app = self.make_app([[key(pygame.K_ESCAPE)]])
        try:
            self.assertFalse(app._menu_loop(in_game=False))
        finally:
            app.close()

    def test_back_from_ability_panel(self):
        app = self.make_app([[key(pygame.K_RETURN)], [key(pygame.K_ESCAPE)], [key(pygame.K_RIGHT)],
                             [key(pygame.K_RETURN)], [key(pygame.K_RETURN)]])
        try:
            self.assertTrue(app._menu_loop(in_game=False))
            self.assertEqual(app.ctx.mode, "jjk")
        finally:
            app.close()

    def test_in_game_switch_resets_and_keeps_camera(self):
        app = self.make_app([], mode="bleach")
        try:
            self.assertTrue(app.startup())
            cam = app.camera
            app._handle_key(key(pygame.K_1))  # Getsuga in Bleach
            self.assertNotEqual(app.manager.abilities["getsuga"].phase, Phase.IDLE)
            # TAB -> menu -> JJK -> play
            app.pg.event.get = ScriptedEvents([[key(pygame.K_RIGHT)], [key(pygame.K_RETURN)],
                                               [key(pygame.K_RETURN)]])
            app._handle_key(key(pygame.K_TAB))
            self.assertEqual(app.ctx.mode, "jjk")
            self.assertIs(app.camera, cam, "camera reused, not reopened")
            self.assertEqual(app.manager.abilities["getsuga"].phase, Phase.IDLE)
            app._handle_key(key(pygame.K_1))
            self.assertNotEqual(app.manager.abilities["blue"].phase, Phase.IDLE)
            app._handle_key(key(pygame.K_5))  # Bleach-only sword toggle is ignored in JJK
            self.assertFalse(app.ctx.show_sword)
        finally:
            app.close()

    def test_tab_then_escape_resumes_same_mode(self):
        app = self.make_app([], mode="jjk")
        try:
            self.assertTrue(app.startup())
            app.pg.event.get = ScriptedEvents([[key(pygame.K_ESCAPE)]])
            app._handle_key(key(pygame.K_TAB))
            self.assertTrue(app.running)
            self.assertEqual(app.ctx.mode, "jjk")
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
