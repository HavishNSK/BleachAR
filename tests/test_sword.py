"""Zanpakuto: fist summoning with hysteresis, sinking, finger occlusion, assets."""

import contextlib
import copy
import io
import unittest

import numpy as np

from bleach_ar.abilities import Context
from bleach_ar.config import Config
from bleach_ar.eyes import EyeEffects
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.sword import Zanpakuto
from bleach_ar.synthetic import make_frame, make_tracking
from bleach_ar.tracking import TrackingState

DT = 1 / 30


class Rig:
    def __init__(self, cfg=None, **ctx_kw):
        self.cfg = cfg or copy.deepcopy(Config().data)
        self.ctx = Context(cfg=self.cfg, canvas=Canvas(0.33), particles=ParticleSystem(300), shaker=Shaker(),
                           **ctx_kw)
        self.sword = Zanpakuto(self.cfg)
        self.t = 0.0

    def step(self, gesture="fist", n=1, render=True, swing=0.5, tracking=None):
        for _ in range(n):
            self.t += DT
            st = tracking if tracking is not None else make_tracking(t=0.0, gesture=gesture, swing=swing)
            self.ctx.tracking = st
            self.ctx.now, self.ctx.dt = self.t, DT
            self.sword.update(self.ctx)
            if render:
                frame = make_frame(st) if st.width else np.zeros((540, 960, 3), np.uint8)
                self.ctx.canvas.begin(frame)
                self.before = frame.copy()
                self.sword.render(self.ctx)
        return self.ctx.sword


class TestSummon(unittest.TestCase):
    def test_sword_only_for_fist(self):
        r = Rig()
        pose = r.step("open_palm", n=10)
        self.assertFalse(pose.visible)
        self.assertEqual(pose.confidence, 0.0)
        pose = r.step("fist", n=6)
        self.assertTrue(pose.visible)
        self.assertGreater(pose.confidence, 0.95)
        self.assertGreater(pose.length, 1.0)

    def test_one_non_fist_frame_does_not_dismiss(self):
        r = Rig()
        r.step("fist", n=8)
        for g in ("open_palm", "fist", "point", "fist", "open_palm", "open_palm", "fist"):
            pose = r.step(g)
            self.assertTrue(pose.visible, g)
            self.assertGreater(pose.confidence, 0.95, g)

    def test_dismiss_after_sustained_open_hand(self):
        r = Rig()
        r.step("fist", n=8)
        pose = r.step("open_palm", n=12)
        self.assertFalse(pose.visible)
        self.assertEqual(pose.confidence, 0.0)
        self.assertEqual(pose.speed, 0.0)
        # Rendering a dismissed sword draws nothing.
        np.testing.assert_array_equal(r.ctx.canvas.frame, r.before)

    def test_fade_in_and_out(self):
        r = Rig()
        confs = [r.step("fist").confidence for _ in range(8)]
        self.assertEqual(confs[0], 0.0)  # needs two consecutive fist frames
        self.assertTrue(0.0 < confs[2] < 1.0)
        self.assertEqual(confs[-1], 1.0)
        outs = [r.step("open_palm").confidence for _ in range(10)]
        self.assertEqual(outs[0], 1.0)
        self.assertTrue(any(0.0 < c < 1.0 for c in outs))
        self.assertEqual(outs[-1], 0.0)

    def test_hand_mode_works_with_open_palm(self):
        r = Rig(sword_mode="hand")
        pose = r.step("open_palm", n=6)
        self.assertTrue(pose.visible)
        self.assertGreater(pose.confidence, 0.95)

    def test_summon_does_not_fake_a_swing(self):
        r = Rig()
        r.step("open_palm", n=5, swing=0.0)
        speeds = [r.step("fist", swing=1.0).speed for _ in range(4)]
        self.assertTrue(all(s < 0.5 for s in speeds), speeds)

    def test_empty_tracking_no_nan(self):
        r = Rig()
        r.step("fist", n=5)
        for st in (TrackingState(width=960, height=540), TrackingState()):
            pose = r.step(tracking=st, n=15, render=bool(st.width))
            for arr in (pose.grip, pose.tip, pose.direction, pose.motion):
                self.assertTrue(np.all(np.isfinite(arr)))
            self.assertTrue(np.isfinite(pose.speed) and np.isfinite(pose.confidence))
            self.assertFalse(pose.visible)
        pose = r.step("fist", n=6)
        self.assertTrue(pose.visible)

    def test_non_finite_landmarks_are_ignored(self):
        import warnings
        r = Rig()
        r.step("fist", n=8)
        st = make_tracking(gesture="fist")
        st.hands["right"].points[0, :2] = np.inf
        st.hands["right"].points[9, :2] = np.nan
        st.pose.points[15, :2] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pose = r.step(tracking=st, n=5, render=False)
            r.ctx.canvas.begin(np.zeros((540, 960, 3), np.uint8))
            r.sword.render(r.ctx)  # fading pose from the last good frame; must not raise
        for arr in (pose.grip, pose.tip, pose.direction, pose.motion):
            self.assertTrue(np.all(np.isfinite(arr)))
        self.assertFalse(pose.visible)
        self.assertTrue(r.step("fist", n=6).visible)

    def test_swing_still_fires_getsuga(self):
        r = Rig()
        r.step("fist", n=10, swing=0.0)
        peak = 0.0
        for i in range(12):
            sw = min(1.0, (i + 1) / 6)
            peak = max(peak, r.step("fist", swing=sw).speed)
        self.assertGreater(peak, float(r.cfg["gestures"]["swing_speed"]))


class TestRendering(unittest.TestCase):
    def test_sink_one_draws_nothing_but_updates_pose(self):
        r = Rig()
        r.step("fist", n=8)
        r.ctx.sword_sink = 1.0
        r.step("fist", n=3, swing=0.2)
        g0 = r.ctx.sword.grip.copy()
        np.testing.assert_array_equal(r.ctx.canvas.frame, r.before)
        self.assertEqual(int(r.ctx.canvas.light.max()), 0)
        r.step("fist", n=2, swing=0.8)
        self.assertGreater(float(np.linalg.norm(r.ctx.sword.grip - g0)), 5.0)
        self.assertTrue(r.ctx.sword.visible)

    def test_sink_lowers_and_fades_blade(self):
        r = Rig()
        r.step("fist", n=8)
        changed = (np.abs(r.ctx.canvas.frame.astype(int) - r.before).sum(2) > 30)
        r.ctx.sword_sink = 0.5
        r.step("fist")
        sunk = (np.abs(r.ctx.canvas.frame.astype(int) - r.before).sum(2) > 30)
        ys0, ys1 = np.nonzero(changed)[0], np.nonzero(sunk)[0]
        self.assertGreater(len(ys1), 0)
        self.assertGreater(ys1.mean(), ys0.mean() + 30)

    def _occlusion_case(self, cfg):
        r = Rig(cfg)
        r.step("fist", n=8)
        grip = r.ctx.sword.grip
        x, y = int(round(grip[0])), int(round(grip[1]))
        after = r.ctx.canvas.frame[y - 2:y + 3, x - 2:x + 3].astype(int)
        orig = r.before[y - 2:y + 3, x - 2:x + 3].astype(int)
        tip = r.ctx.sword.grip + (r.ctx.sword.tip - r.ctx.sword.grip) * 0.6
        tx, ty = int(round(tip[0])), int(round(tip[1]))
        blade_changed = np.abs(r.ctx.canvas.frame[ty, tx].astype(int) - r.before[ty, tx]).sum()
        return r, after, orig, blade_changed

    def test_fingers_cover_the_handle(self):
        for images in (True, False):
            cfg = copy.deepcopy(Config().data)
            if not images:
                cfg["assets"]["sword_images"] = {}
            r, after, orig, blade_changed = self._occlusion_case(cfg)
            self.assertLessEqual(int(np.abs(after - orig).max()), 3, f"images={images}")
            self.assertGreater(blade_changed, 30, "the blade beyond the fist stays visible")
            self.assertEqual(int(r.ctx.canvas.light[int(r.ctx.sword.grip[1]), int(r.ctx.sword.grip[0])].max()), 0)
            # Without occlusion the handle would be drawn there.
            r2 = Rig(cfg)
            r2.sword._occlusion_begin = lambda ctx: None
            r2.step("fist", n=8)
            g = r2.ctx.sword.grip
            x, y = int(round(g[0])), int(round(g[1]))
            self.assertGreater(int(np.abs(r2.ctx.canvas.frame[y, x].astype(int) - r2.before[y, x]).sum()), 30)

    def test_all_styles_render(self):
        for style in ("katana", "cleaver", "dark"):
            for images in (True, False):
                cfg = copy.deepcopy(Config().data)
                if not images:
                    cfg["assets"]["sword_images"] = {}
                r = Rig(cfg, weapon_style=style)
                r.ctx.aura_boost = 1.0
                r.step("fist", n=5)
                self.assertGreater(int(np.abs(r.ctx.canvas.frame.astype(int) - r.before).max()), 50)
                self.assertGreater(int(r.ctx.canvas.light.max()), 0, "energy edge glow")


class TestAssets(unittest.TestCase):
    def test_default_assets_load(self):
        z = Zanpakuto(Config().data)
        self.assertEqual(set(z.images), {"katana", "cleaver", "dark"})
        for mips in z.images.values():
            self.assertEqual(mips[0].shape[2], 4)
            self.assertGreater(len(mips), 1)
        self.assertIsNotNone(EyeEffects(Config().data).mask_img)

    def test_missing_assets_fall_back_with_one_warning(self):
        cfg = copy.deepcopy(Config().data)
        cfg["assets"]["sword_images"] = {"katana": "assets/__missing_katana__.png"}
        cfg["assets"]["mask_image"] = "assets/__missing_mask__.png"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(3):
                z = Zanpakuto(cfg)
                EyeEffects(cfg)
        out = buf.getvalue()
        self.assertEqual(out.count("__missing_katana__"), 1)
        self.assertEqual(out.count("__missing_mask__"), 1)
        self.assertEqual(z.images, {})
        r = Rig(cfg, show_mask=True)
        r.sword = z
        r.step("fist", n=5)
        self.assertGreater(int(np.abs(r.ctx.canvas.frame.astype(int) - r.before).max()), 50)


if __name__ == "__main__":
    unittest.main()
