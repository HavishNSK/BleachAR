"""Domain Expansion: Unlimited Void - phases, cancel, reset, tracking loss, resolutions."""

import unittest

import numpy as np

from bleach_ar.abilities import AbilityManager, Context, Phase
from bleach_ar.config import Config
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.jjk.unlimited_void import MAX_STARS, UnlimitedVoid
from bleach_ar.synthetic import make_frame, make_tracking
from bleach_ar.tracking import TrackingState

DT = 1 / 30
NAME = "unlimited_void"


class Rig:
    def __init__(self, w=960, h=540, capacity=900, mode="jjk"):
        np.random.seed(0)
        self.w, self.h = w, h
        self.cfg = Config().data
        self.ctx = Context(cfg=self.cfg, canvas=Canvas(0.33), particles=ParticleSystem(capacity), shaker=Shaker())
        self.mgr = AbilityManager(self.ctx)
        if mode != "bleach":
            self.mgr.set_mode(mode)
        self.ab = self.mgr.abilities[NAME]
        self.t = 0.0
        self.backdrop_used = False

    def step(self, n=1, tracking=None):
        out = None
        for _ in range(n):
            self.t += DT
            st = tracking if tracking is not None else make_tracking(self.w, self.h, t=self.t)
            ctx = self.ctx
            ctx.tracking = st
            ctx.now, ctx.dt = self.t, DT
            self.mgr.update()
            frame = make_frame(st) if st.width else np.zeros((self.h, self.w, 3), np.uint8)
            ctx.canvas.begin(frame)
            ctx.canvas.set_person_src(st.mask)
            self.mgr.render()
            self.backdrop_used = ctx.canvas._backdrop_used and ctx.canvas.backdrop_amount > 0.002
            out = ctx.canvas.composite()
            ctx.canvas.flash *= np.exp(-6 * DT)
            assert out.shape == (self.h, self.w, 3) and out.dtype == np.uint8
        return out

    def steps_for(self, seconds):
        return int(round(seconds / DT))


class TestUnlimitedVoid(unittest.TestCase):
    def test_full_phase_sequence_timing(self):
        r = Rig(640, 360)
        p = r.cfg["abilities"][NAME]
        self.assertTrue(r.mgr.trigger(NAME))
        self.assertEqual(r.ab.phase, Phase.CHARGING)
        self.assertEqual(r.ctx.locked_by, NAME)
        seen = {}
        frames = 0
        while r.ab.phase != Phase.IDLE and frames < 1000:
            ph = r.ab.phase
            r.step()
            frames += 1
            seen[ph] = seen.get(ph, 0) + 1
            if r.ab.phase == Phase.ACTIVE:
                self.assertIsNone(r.ctx.locked_by, "JJK techniques stay usable inside the domain")
        for ph, key in ((Phase.CHARGING, "charge"), (Phase.ACTIVE, "active"), (Phase.IMPACT, "impact"),
                        (Phase.COOLDOWN, "cooldown")):
            self.assertAlmostEqual(seen[ph] * DT, p[key], delta=2 * DT, msg=ph)
        self.assertEqual(r.ab.phase, Phase.IDLE)
        self.assertIsNone(r.ctx.locked_by)

    def test_backdrop_during_active_not_after(self):
        r = Rig(640, 360)
        r.mgr.trigger(NAME)
        r.step(r.steps_for(r.ab.duration(Phase.CHARGING)) + 3)
        self.assertEqual(r.ab.phase, Phase.ACTIVE)
        r.step(5)
        self.assertTrue(r.backdrop_used)
        self.assertGreater(r.ctx.canvas.backdrop_amount, 0.9)
        self.assertIsNotNone(r.ctx.energy_override)
        while r.ab.phase != Phase.IDLE:
            r.step()
        r.step()
        self.assertFalse(r.backdrop_used)
        self.assertIsNone(r.ctx.energy_override)

    def test_cancel_during_active_collapses(self):
        r = Rig()
        r.mgr.trigger(NAME)
        r.step(r.steps_for(r.ab.duration(Phase.CHARGING)) + 30)
        self.assertEqual(r.ab.phase, Phase.ACTIVE)
        self.assertTrue(r.mgr.trigger(NAME))
        self.assertEqual(r.ab.phase, Phase.IMPACT)
        self.assertFalse(r.mgr.trigger(NAME), "pressing during IMPACT does nothing")
        self.assertEqual(r.ab.phase, Phase.IMPACT)
        r.step(r.steps_for(r.ab.duration(Phase.IMPACT)) + 2)
        self.assertEqual(r.ab.phase, Phase.COOLDOWN)
        self.assertFalse(r.mgr.trigger(NAME), "pressing during COOLDOWN does nothing")
        r.step()
        self.assertFalse(r.backdrop_used)
        while r.ab.phase != Phase.IDLE:
            r.step()
        self.assertTrue(r.mgr.trigger(NAME), "can expand again after cooldown")

    def test_cancel_during_charging(self):
        r = Rig()
        r.mgr.trigger(NAME)
        r.step(r.steps_for(r.ab.duration(Phase.CHARGING) * 0.6))
        self.assertEqual(r.ab.phase, Phase.CHARGING)
        self.assertTrue(r.mgr.trigger(NAME))
        self.assertEqual(r.ab.phase, Phase.IMPACT)
        self.assertIsNone(r.ctx.locked_by)
        r.step(r.steps_for(r.ab.duration(Phase.IMPACT)) + 2)
        self.assertEqual(r.ab.phase, Phase.COOLDOWN)

    def test_cancel_at_very_start_of_charging(self):
        r = Rig()
        r.mgr.trigger(NAME)
        r.step()
        self.assertTrue(r.mgr.trigger(NAME))
        r.step(r.steps_for(r.ab.duration(Phase.IMPACT)) + 2)
        self.assertEqual(r.ab.phase, Phase.COOLDOWN)

    def test_reset_mid_domain(self):
        r = Rig()
        r.mgr.trigger(NAME)
        r.step(5)
        self.assertEqual(r.ctx.locked_by, NAME)
        r.ab.reset(r.ctx)
        self.assertEqual(r.ab.phase, Phase.IDLE)
        self.assertIsNone(r.ctx.locked_by)
        self.assertIsNone(r.ctx.energy_override)
        r.mgr.trigger(NAME)
        r.step(r.steps_for(r.ab.duration(Phase.CHARGING)) + 10)
        self.assertEqual(r.ab.phase, Phase.ACTIVE)
        r.mgr.reset()
        self.assertEqual(r.ab.phase, Phase.IDLE)
        r.step()
        self.assertFalse(r.backdrop_used)
        self.assertTrue(r.mgr.trigger(NAME))

    def test_tracking_loss_mid_domain(self):
        r = Rig()
        r.mgr.trigger(NAME)
        r.step(10)
        lost = TrackingState(width=r.w, height=r.h)
        for _ in range(40):  # through the rest of charging into active
            r.step(tracking=lost)
            self.assertTrue(np.all(np.isfinite(r.ab._anchor)))
        self.assertEqual(r.ab.phase, Phase.ACTIVE)
        # Totally empty state (no size) too.
        r.step(5, tracking=TrackingState())
        r.mgr.trigger(NAME)
        r.step(r.steps_for(r.ab.duration(Phase.IMPACT)) + 2, tracking=lost)
        self.assertEqual(r.ab.phase, Phase.COOLDOWN)
        self.assertTrue(np.all(np.isfinite(r.ab._anchor)))
        self.assertTrue(np.all(np.isfinite(r.ctx.particles.pos)))

    def test_anchor_from_cold_start_without_tracking(self):
        r = Rig()
        r.mgr.trigger(NAME)
        r.step(60, tracking=TrackingState(width=r.w, height=r.h))
        a = r.ab._anchor
        self.assertTrue(0 <= a[0] <= r.w and 0 <= a[1] <= r.h)

    def test_resolutions(self):
        for w, h in ((640, 360), (960, 540), (1280, 720)):
            r = Rig(w, h)
            r.mgr.trigger(NAME)
            # charging, a bit of active, then collapse
            r.step(r.steps_for(r.ab.duration(Phase.CHARGING)) + 15)
            self.assertTrue(r.backdrop_used, (w, h))
            r.mgr.trigger(NAME)
            r.step(r.steps_for(r.ab.duration(Phase.IMPACT)) + 2)
            self.assertEqual(r.ab.phase, Phase.COOLDOWN)
            self.assertEqual((r.ab._cosmos.w, r.ab._cosmos.h), (w, h))

    def test_star_count_capped_and_scaled_by_quality(self):
        counts = {}
        for cap in (400, 900, 1600):
            r = Rig(640, 360, capacity=cap)
            r.cfg["abilities"][NAME]["star_count"] = 10 ** 6
            r.mgr.trigger(NAME)
            r.step(r.steps_for(r.ab.duration(Phase.CHARGING)) + 2)
            counts[cap] = r.ab._cosmos.n
            self.assertLessEqual(r.ab._cosmos.n, MAX_STARS)
        self.assertLessEqual(counts[400], counts[900])
        self.assertLessEqual(counts[900], counts[1600])
        r = Rig(640, 360, capacity=400)
        r.mgr.trigger(NAME)
        r.step(r.steps_for(r.ab.duration(Phase.CHARGING)) + 2)
        low = r.ab._cosmos.n
        r = Rig(640, 360, capacity=1600)
        r.mgr.trigger(NAME)
        r.step(r.steps_for(r.ab.duration(Phase.CHARGING)) + 2)
        self.assertLess(low, r.ab._cosmos.n, "quality scales the star count")

    def test_not_triggerable_in_bleach_mode(self):
        r = Rig(mode="bleach")
        self.assertEqual(r.ctx.mode, "bleach")
        self.assertFalse(r.mgr.trigger(NAME))
        self.assertEqual(r.ab.phase, Phase.IDLE)

    def test_refuses_while_other_domain_locks(self):
        r = Rig()
        r.ctx.locked_by = "malevolent_shrine"
        self.assertFalse(r.mgr.trigger(NAME))
        self.assertEqual(r.ab.phase, Phase.IDLE)
        self.assertEqual(r.ctx.locked_by, "malevolent_shrine")

    def test_standalone_without_manager(self):
        cfg = Config().data
        ctx = Context(cfg=cfg, canvas=Canvas(0.33), particles=ParticleSystem(900), shaker=Shaker(), mode="jjk")
        ab = UnlimitedVoid(cfg["abilities"][NAME])
        self.assertTrue(ab.trigger(ctx))
        t = 0.0
        for _ in range(int(14.5 / DT)):
            t += DT
            st = make_tracking(640, 360, t=t)
            ctx.tracking, ctx.now, ctx.dt = st, t, DT
            ab.update(ctx)
            ctx.particles.update(DT)
            ctx.canvas.begin(make_frame(st))
            ctx.canvas.set_person_src(st.mask)
            ab.render(ctx)
            ctx.particles.render(ctx.canvas)
            ctx.canvas.composite()
        self.assertEqual(ab.phase, Phase.IDLE)


if __name__ == "__main__":
    unittest.main()
