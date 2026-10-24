"""Getsuga Tensho: lifecycle, aiming, anchoring and robustness (headless, deterministic)."""

import math
import unittest

import numpy as np

from bleach_ar.abilities import BANKAI, SEALED, SHIKAI, Context, Phase, SwordPose
from bleach_ar.config import Config
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.getsuga import GetsugaTensho
from bleach_ar.synthetic import make_tracking
from bleach_ar.tracking import TrackingState

DT = 1 / 60
W, H = 960, 540


class Rig:
    def __init__(self, w=W, h=H, level=SEALED, **params):
        np.random.seed(7)
        cfg = Config().data
        p = dict(cfg["abilities"]["getsuga"])
        p.update(params)
        self.ctx = Context(cfg=cfg, canvas=Canvas(0.33), particles=ParticleSystem(900), shaker=Shaker(),
                           tracking=TrackingState(width=w, height=h), level=level)
        self.w, self.h = w, h
        self.ctx.canvas.begin(np.zeros((h, w, 3), np.uint8))
        self.g = GetsugaTensho(p)
        self.now = 0.0

    def set_sword(self, grip, tip, speed=0.0, motion=(1.0, 0.0), conf=1.0, visible=True):
        sw = self.ctx.sword
        sw.grip = np.asarray(grip, np.float32)
        sw.tip = np.asarray(tip, np.float32)
        v = sw.tip - sw.grip
        sw.length = float(np.linalg.norm(v))
        sw.direction = (v / max(sw.length, 1e-6)).astype(np.float32)
        sw.speed = speed
        sw.motion = np.asarray(motion, np.float32)
        sw.confidence = conf
        sw.visible = visible

    def lose_sword(self, garbage=False):
        sw = self.ctx.sword
        sw.visible, sw.confidence, sw.speed = False, 0.0, 0.0
        if garbage:
            sw.grip = np.array([np.nan, np.inf], np.float32)
            sw.tip = np.array([np.nan, np.nan], np.float32)
            sw.length = float("nan")
            sw.motion = np.array([np.nan, np.nan], np.float32)

    def step(self, n=1, each=None):
        ctx = self.ctx
        for i in range(n):
            self.now += DT
            ctx.now, ctx.dt = self.now, DT
            if each is not None:
                each(i)
            self.g.update(ctx)
            ctx.particles.update(DT)
            ctx.canvas.begin(np.full((self.h, self.w, 3), 40, np.uint8))
            self.g.render(ctx)
            ctx.particles.render(ctx.canvas)
            ctx.canvas.composite()

    def run_to(self, phase, limit=400):
        for _ in range(limit):
            if self.g.phase == phase:
                return
            self.step()
        raise AssertionError(f"never reached {phase}")

    def assert_finite_state(self, tc):
        g = self.g
        for v in (g.p0, g.dir, g.vanish, g.end):
            tc.assertTrue(np.all(np.isfinite(v)), v)
        tc.assertTrue(math.isfinite(g.dist) and math.isfinite(g.radius))
        if g.anchor is not None:
            tc.assertTrue(np.all(np.isfinite(g.anchor)), g.anchor)


class TestLifecycle(unittest.TestCase):
    def test_phases_follow_param_timings(self):
        for level in (SEALED, SHIKAI, BANKAI):
            r = Rig(level=level)
            r.set_sword((400, 400), (400, 200))
            p = r.g.params
            self.assertTrue(r.g.trigger(r.ctx))
            self.assertEqual(r.g.phase, Phase.CHARGING)
            entered = {Phase.CHARGING: 0.0}
            t, prev = 0.0, Phase.CHARGING
            while r.g.phase != Phase.IDLE and t < 10:
                r.step()
                t += DT
                if r.g.phase != prev:
                    entered[r.g.phase] = t
                    prev = r.g.phase
            self.assertEqual(r.g.phase, Phase.IDLE)
            seq = [Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT, Phase.COOLDOWN, Phase.IDLE]
            for a, b, key in zip(seq, seq[1:], ("charge", "active", "impact", "cooldown")):
                # Each phase lasts its param (frame-quantised: up to one DT longer).
                self.assertAlmostEqual(entered[b] - entered[a], p[key], delta=1.5 * DT, msg=f"{level} {a}")

    def test_retrigger_blocked_while_running_and_in_cooldown(self):
        r = Rig()
        r.set_sword((400, 400), (400, 200))
        self.assertTrue(r.g.trigger(r.ctx))
        for ph in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT, Phase.COOLDOWN):
            r.run_to(ph)
            self.assertFalse(r.g.trigger(r.ctx), ph)
            self.assertEqual(r.g.phase, ph)
        r.run_to(Phase.IDLE)
        self.assertTrue(r.g.trigger(r.ctx))

    def test_reset_mid_flight_clears_state(self):
        r = Rig()
        r.set_sword((400, 400), (400, 200), speed=3.0, motion=(1, 0))
        r.g.trigger(r.ctx)
        r.run_to(Phase.ACTIVE)
        r.step(5)
        r.g.reset(r.ctx)
        g = r.g
        self.assertEqual(g.phase, Phase.IDLE)
        self.assertEqual(g.t, 0.0)
        self.assertIsNone(g.anchor)
        self.assertIsNone(g.swing_dir)
        self.assertEqual(len(g._swings), 0)
        self.assertEqual(g.radius, 0.0)
        self.assertEqual(g.dist, 0.0)
        # Nothing of ours is drawn any more.
        r.ctx.canvas.begin(np.zeros((H, W, 3), np.uint8))
        g.render(r.ctx)
        self.assertEqual(int(r.ctx.canvas.light.max()), 0)
        self.assertEqual(float(r.ctx.canvas.glow.max()), 0.0)
        self.assertTrue(g.trigger(r.ctx), "usable again right after reset")

    def test_shot_state_dropped_after_impact(self):
        r = Rig()
        r.set_sword((400, 400), (400, 200))
        r.g.trigger(r.ctx)
        r.run_to(Phase.COOLDOWN)
        self.assertIsNone(r.g.anchor)
        self.assertEqual(r.g.radius, 0.0)


class TestAiming(unittest.TestCase):
    def test_direction_follows_swing_during_charge(self):
        r = Rig()
        r.set_sword((300, 330), (300, 180))
        r.g.trigger(r.ctx)
        # Slash to the right while charging; the tip moves with the swing.
        def swing(i):
            x = 300 + 12 * i
            r.set_sword((x, 330), (x + 40, 180), speed=3.0, motion=(1.0, 0.05))
        r.step(int(r.g.params["charge"] / DT) + 2, each=swing)
        self.assertEqual(r.g.phase, Phase.ACTIVE)
        self.assertGreater(float(np.dot(r.g.dir, (1.0, 0.0))), 0.95, r.g.dir)

    def test_direction_uses_swing_just_before_trigger(self):
        r = Rig()
        # Gesture fires at the end of a fast downward-left slash...
        for _ in range(4):
            r.set_sword((700, 150), (760, 40), speed=3.0, motion=(-0.6, 0.8))
            r.step()
        # ...then the blade stops while the attack charges.
        r.set_sword((650, 230), (700, 110), speed=0.0)
        r.g.trigger(r.ctx)
        r.run_to(Phase.ACTIVE)
        d = r.g.dir
        self.assertGreater(float(np.dot(d, (-0.6, 0.8))), 0.9, d)

    def test_old_swing_is_ignored(self):
        r = Rig()
        r.set_sword((700, 150), (760, 40), speed=3.0, motion=(-0.6, 0.8))
        r.step()
        r.set_sword((300, 400), (300, 250), speed=0.0)
        r.step(60)  # one second of stillness
        r.g.trigger(r.ctx)
        r.run_to(Phase.ACTIVE)
        self.assertIsNone(r.g.swing_dir)

    def test_fallback_is_perpendicular_to_blade_toward_screen_centre(self):
        for x, want in ((250, 1.0), (710, -1.0)):
            r = Rig()
            r.set_sword((x, 420), (x, 220))  # upright blade, no motion
            r.g.trigger(r.ctx)
            r.run_to(Phase.ACTIVE)
            self.assertIsNone(r.g.swing_dir)
            self.assertGreater(float(r.g.dir[0]) * want, 0.95, (x, r.g.dir))

    def test_launch_origin_at_blade_tip_region(self):
        r = Rig()
        grip, tip = np.array([380, 420], np.float32), np.array([520, 200], np.float32)
        r.set_sword(grip, tip)
        r.g.trigger(r.ctx)
        r.run_to(Phase.ACTIVE)
        expect = grip + (tip - grip) * 0.8
        self.assertLess(float(np.linalg.norm(r.g.p0 - expect)), 2.0, r.g.p0)

    def test_anchor_follows_moving_blade_without_lag(self):
        r = Rig()
        r.set_sword((300, 420), (300, 220))
        r.g.trigger(r.ctx)
        r.step(5, each=lambda i: r.set_sword((300 + 4 * i, 420), (300 + 4 * i, 220)))
        expect = np.array([300 + 4 * 4, 260], np.float32)
        self.assertLess(float(np.linalg.norm(r.g.anchor - expect)), 4.0, r.g.anchor)

    def test_slash_stays_in_frame(self):
        r = Rig()
        # Slash straight down from low in the frame: impact must still land on screen.
        r.set_sword((400, 520), (420, 470), speed=3.0, motion=(0.0, 1.0))
        r.g.trigger(r.ctx)
        r.run_to(Phase.IMPACT)
        e = r.g.end
        self.assertTrue(0 <= e[0] <= W and 0 <= e[1] <= H, e)
        self.assertGreater(float(r.g.dir[1]), 0.0, "still heads the way of the slash")


class TestRobustness(unittest.TestCase):
    def test_tracking_lost_mid_charge_keeps_last_anchor(self):
        for garbage in (False, True):
            r = Rig()
            r.set_sword((300, 420), (300, 220))
            r.g.trigger(r.ctx)
            r.step(3)
            last = r.g.anchor.copy()
            r.lose_sword(garbage=garbage)
            r.step(3)
            np.testing.assert_allclose(r.g.anchor, last, atol=1e-4)
            r.assert_finite_state(self)
            r.run_to(Phase.ACTIVE)
            np.testing.assert_allclose(r.g.p0, last, atol=1e-4)
            r.run_to(Phase.IDLE)  # renders every phase with no sword at all
            r.assert_finite_state(self)

    def test_does_not_downgrade_to_body_centre(self):
        r = Rig()
        r.ctx.tracking = make_tracking(W, H)
        r.set_sword((250, 300), (200, 150))
        r.g.trigger(r.ctx)
        r.step(2)
        last = r.g.anchor.copy()
        r.lose_sword()
        r.step(4)
        np.testing.assert_allclose(r.g.anchor, last, atol=1e-4)

    def test_projectile_ignores_tracking_after_launch(self):
        r = Rig()
        r.set_sword((300, 420), (300, 220))
        r.g.trigger(r.ctx)
        r.run_to(Phase.ACTIVE)
        p0, d = r.g.p0.copy(), r.g.dir.copy()
        r.step(10, each=lambda i: r.set_sword((800, 100), (900, 50), speed=5.0, motion=(0, 1)))
        np.testing.assert_array_equal(r.g.p0, p0)
        np.testing.assert_array_equal(r.g.dir, d)
        # Head moves continuously (no teleport between frames).
        prev = None
        for i in range(30):
            pos, _ = r.g._flight(i / 29)
            if prev is not None:
                self.assertLess(float(np.linalg.norm(pos - prev)), 0.05 * r.ctx.diag)
            prev = pos

    def test_empty_tracking_and_no_sword(self):
        for tracking in (TrackingState(width=W, height=H), TrackingState()):
            r = Rig()
            r.ctx.tracking = tracking
            r.ctx.sword = SwordPose()
            self.assertTrue(r.g.trigger(r.ctx))
            r.run_to(Phase.ACTIVE)
            np.testing.assert_allclose(r.g.p0, (W * 0.5, H * 0.6), atol=1e-3)
            r.run_to(Phase.IDLE)
            r.assert_finite_state(self)

    def test_hand_only_anchor(self):
        r = Rig()
        st = make_tracking(W, H)
        r.ctx.tracking = st
        r.ctx.sword = SwordPose()
        r.g.trigger(r.ctx)
        r.run_to(Phase.ACTIVE)
        np.testing.assert_allclose(r.g.p0, st.hands["right"].points[9, :2], atol=1e-3)

    def test_nan_hand_landmarks_fall_back(self):
        r = Rig()
        st = make_tracking(W, H)
        st.hands["right"].points[:] = np.nan
        st.pose.points[:] = np.nan
        r.ctx.tracking = st
        r.ctx.sword = SwordPose()
        r.g.trigger(r.ctx)
        r.run_to(Phase.IDLE)
        r.assert_finite_state(self)

    def test_unbegun_canvas(self):
        cfg = Config().data
        ctx = Context(cfg=cfg, canvas=Canvas(0.33), particles=ParticleSystem(100), shaker=Shaker())
        g = GetsugaTensho(dict(cfg["abilities"]["getsuga"]))
        ctx.dt = DT
        self.assertTrue(g.trigger(ctx))
        for i in range(200):
            ctx.now += DT
            g.update(ctx)
        self.assertEqual(g.phase, Phase.IDLE)


class TestParams(unittest.TestCase):
    def launched(self, w=W, h=H, **params):
        r = Rig(w=w, h=h, **params)
        r.set_sword((w * 0.45, h * 0.6), (w * 0.45, h * 0.3))
        r.g.trigger(r.ctx)
        r.run_to(Phase.ACTIVE)
        return r

    def test_size_scales_radius(self):
        a, b = self.launched(size=1.0), self.launched(size=2.0)
        self.assertAlmostEqual(b.g.radius / a.g.radius, 2.0, places=4)

    def test_speed_scales_travel(self):
        a, b = self.launched(speed=0.5), self.launched(speed=1.0)
        self.assertAlmostEqual(b.g.dist / a.g.dist, 2.0, places=4)
        ea, eb = a.g._flight(1.0)[0], b.g._flight(1.0)[0]
        self.assertGreater(float(np.linalg.norm(eb - b.g.p0)), float(np.linalg.norm(ea - a.g.p0)))

    def test_scales_with_frame_size(self):
        a, b = self.launched(), self.launched(w=W * 2, h=H * 2)
        self.assertAlmostEqual(b.g.radius / a.g.radius, 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
