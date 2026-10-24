"""Render every ability through every phase with synthetic tracking, plus
headless end-to-end runs of the real app."""

import os
import subprocess
import sys
import unittest

import numpy as np

from bleach_ar.abilities import BANKAI, SEALED, SHIKAI, AbilityManager, Context, Phase
from bleach_ar.config import Config
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.synthetic import make_frame, make_tracking
from bleach_ar.tracking import TrackingState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 1 / 30


class Harness:
    def __init__(self, **ctx_kw):
        self.cfg = Config().data
        self.ctx = Context(cfg=self.cfg, canvas=Canvas(0.33), particles=ParticleSystem(900), shaker=Shaker(),
                           **ctx_kw)
        self.mgr = AbilityManager(self.ctx)
        self.t = 0.0

    def step(self, tracking=None, n=1):
        out = None
        for _ in range(n):
            self.t += DT
            st = tracking if tracking is not None else make_tracking(t=self.t, swing=0.5 + 0.5 * np.sin(self.t * 3))
            self.ctx.tracking = st
            self.ctx.now, self.ctx.dt = self.t, DT
            self.mgr.update()
            frame = make_frame(st) if st.width else np.zeros((540, 960, 3), np.uint8)
            self.ctx.canvas.begin(frame)
            self.mgr.render()
            out = self.ctx.canvas.composite()
            assert out.shape == frame.shape and out.dtype == np.uint8
        return out

    def run_until_idle(self, name, tracking=None, limit=600):
        ab = self.mgr.abilities[name]
        for _ in range(limit):
            if ab.phase == Phase.IDLE:
                return
            self.step(tracking)
        raise AssertionError(f"{name} never returned to IDLE")


class TestAbilities(unittest.TestCase):
    def test_every_ability_full_cycle_each_level(self):
        for level in (SEALED, SHIKAI, BANKAI):
            for name in ("getsuga", "cero"):
                # Use tensa Bankai so Getsuga is not routed to Senbonzakura.
                h = Harness(show_eyes=True, show_mask=True)
                h.ctx.level = level
                h.ctx.bankai_style = "tensa"
                self.assertTrue(h.mgr.trigger(name), f"{name} at {level}")
                h.run_until_idle(name)
                self.assertEqual(h.ctx.level, level)

    def test_shikai_then_bankai_then_release(self):
        h = Harness()
        self.assertTrue(h.mgr.trigger("shikai"))
        h.run_until_idle("shikai")
        self.assertEqual(h.ctx.level, SHIKAI)
        self.assertTrue(h.mgr.trigger("bankai"))
        self.assertFalse(h.mgr.trigger("bankai"), "bankai must not retrigger mid-sequence")
        h.step(n=5)
        self.assertEqual(h.ctx.locked_by, "bankai")
        self.assertFalse(h.mgr.trigger("getsuga"), "other abilities are locked during bankai")
        self.assertFalse(h.mgr.trigger("cero"))
        h.run_until_idle("bankai")
        self.assertEqual(h.ctx.level, BANKAI)
        self.assertIsNone(h.ctx.locked_by)
        self.assertFalse(h.mgr.trigger("shikai"), "shikai key does nothing in bankai")
        # Default Bankai style is senbonzakura: the Getsuga key commands the blade swarm.
        self.assertTrue(h.mgr.trigger("getsuga"))
        self.assertEqual(h.mgr.abilities["senbon_attack"].phase, Phase.CHARGING)
        h.run_until_idle("senbon_attack")
        self.assertTrue(h.mgr.trigger("bankai"), "pressing bankai again releases it")
        h.run_until_idle("bankai")
        self.assertEqual(h.ctx.level, SEALED)

    def test_shikai_toggle_off(self):
        h = Harness()
        h.mgr.trigger("shikai")
        h.run_until_idle("shikai")
        self.assertTrue(h.mgr.trigger("shikai"))
        h.run_until_idle("shikai")
        self.assertEqual(h.ctx.level, SEALED)

    def test_reset_mid_bankai(self):
        h = Harness()
        h.mgr.trigger("bankai")
        h.step(n=10)
        h.ctx.sword_sink = 0.85
        h.mgr.reset()
        self.assertEqual(h.ctx.level, SEALED)
        self.assertIsNone(h.ctx.locked_by)
        self.assertEqual(h.ctx.sword_sink, 0.0)
        self.assertTrue(all(a.phase == Phase.IDLE for a in h.mgr.abilities.values()))
        h.step(n=3)

    def test_tracking_loss_during_abilities(self):
        h = Harness(show_eyes=True, show_mask=True)
        h.mgr.trigger("cero")
        h.step(n=10)
        empty = TrackingState(width=960, height=540)
        h.step(tracking=empty, n=40)  # person leaves mid-ability
        h.mgr.trigger("getsuga")
        h.step(tracking=empty, n=60)
        h.step(n=10)  # person comes back
        h.ctx.level = BANKAI
        h.step(tracking=TrackingState(), n=5)  # zero-size tracking state too

    def test_all_toggles_and_styles(self):
        for style in ("katana", "cleaver", "dark"):
            for mode in ("sword", "hand"):
                for eye in ("hollow", "red", "blue", "white"):
                    h = Harness(show_eyes=True, show_mask=True, weapon_style=style, sword_mode=mode, eye_style=eye)
                    h.step(n=3)
        h = Harness(show_sword=False, show_aura=False)
        h.step(n=3)

    def test_aura_without_segmentation(self):
        h = Harness()
        st = make_tracking()
        st.mask = None
        h.step(tracking=st, n=5)


class TestHeadlessApp(unittest.TestCase):
    def run_app(self, *args, timeout=120):
        env = dict(os.environ, SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
        return subprocess.run([sys.executable, "-m", "bleach_ar", "--headless", *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=timeout)

    def test_demo_mode_runs_and_exits(self):
        r = self.run_app("--demo", "--frames", "120")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("avg FPS", r.stdout)

    def test_real_tracker_on_synthetic_source(self):
        r = self.run_app("--source", "synthetic", "--frames", "40", "--quality", "low")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("avg FPS", r.stdout)

    def test_missing_camera_reports_error(self):
        r = self.run_app("--source", "/nonexistent/clip.mp4", "--frames", "5")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("Could not read frames", r.stdout)


if __name__ == "__main__":
    unittest.main()
