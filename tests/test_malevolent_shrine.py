"""Domain Expansion: Malevolent Shrine - headless, deterministic tests."""

import math
import unittest

import numpy as np

from bleach_ar.abilities import AbilityManager, Context, Phase
from bleach_ar.config import Config
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.jjk.malevolent_shrine import MalevolentShrine
from bleach_ar.synthetic import make_frame, make_tracking
from bleach_ar.tracking import TrackingState

DT = 1 / 30
NAME = "malevolent_shrine"


def make_ctx(w=960, h=540, capacity=900):
    cfg = Config().data
    ctx = Context(cfg=cfg, canvas=Canvas(0.33), particles=ParticleSystem(capacity), shaker=Shaker())
    ctx.mode = "jjk"
    return ctx


class Runner:
    """Drives one ability like the app does (update, render, composite, flash decay)."""

    def __init__(self, params=None, w=960, h=540, capacity=900, render=True):
        self.render = render
        self.ctx = make_ctx(w, h, capacity)
        p = dict(self.ctx.cfg["abilities"][NAME])
        p.update(params or {})
        self.ab = MalevolentShrine(p)
        self.w, self.h = w, h
        self.t = 0.0
        self.log = []  # (phase, backdrop_amount, backdrop_used)

    def step(self, n=1, tracking="auto"):
        ctx = self.ctx
        out = None
        for _ in range(n):
            self.t += DT
            st = make_tracking(self.w, self.h, t=self.t) if tracking == "auto" else tracking
            ctx.tracking, ctx.now, ctx.dt = st, self.t, DT
            ctx.energy_override = None
            self.ab.update(ctx)
            ctx.particles.update(DT)
            if not self.render:
                continue
            frame = make_frame(st) if st.width else np.zeros((self.h, self.w, 3), np.uint8)
            ctx.canvas.set_person_src(st.mask)
            ctx.canvas.begin(frame)
            self.ab.render(ctx)
            self.log.append((self.ab.phase, ctx.canvas.backdrop_amount, ctx.canvas._backdrop_used))
            ctx.particles.render(ctx.canvas)
            out = ctx.canvas.composite()
            ctx.canvas.flash *= math.exp(-6 * DT)
            assert out.shape == (frame.shape[0], frame.shape[1], 3) and out.dtype == np.uint8
        return out

    def run_until(self, phase, limit=2000):
        for _ in range(limit):
            if self.ab.phase == phase:
                return
            self.step()
        raise AssertionError(f"never reached {phase}")


class TestPhases(unittest.TestCase):
    def test_phase_sequence_follows_params(self):
        params = {"charge": 0.5, "active": 1.0, "impact": 0.4, "cooldown": 0.3}
        r = Runner(params)
        self.assertTrue(r.ab.trigger(r.ctx))
        self.assertEqual(r.ab.phase, Phase.CHARGING)
        seen, frames = [], {}
        for _ in range(200):
            r.step()
            ph = r.ab.phase
            if not seen or seen[-1] != ph:
                seen.append(ph)
            frames[ph] = frames.get(ph, 0) + 1
            if ph == Phase.IDLE:
                break
        self.assertEqual(seen, [Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT, Phase.COOLDOWN, Phase.IDLE])
        for ph, key in ((Phase.ACTIVE, "active"), (Phase.IMPACT, "impact"), (Phase.COOLDOWN, "cooldown")):
            self.assertAlmostEqual(frames[ph] * DT, params[key], delta=2.5 * DT)
        self.assertIsNone(r.ctx.locked_by)

    def test_title_lock_and_energy(self):
        r = Runner()
        r.ab.trigger(r.ctx)
        self.assertEqual(r.ctx.locked_by, NAME)
        self.assertTrue(any(t.text == "DOMAIN EXPANSION" for t in r.ctx.titles))
        r.step(5)
        self.assertIsNotNone(r.ctx.energy_override)
        self.assertGreater(r.ctx.energy_override[2], r.ctx.energy_override[0])  # red (BGR)

    def test_backdrop_active_on_and_off_after(self):
        r = Runner({"charge": 0.4, "active": 1.5, "impact": 0.5, "cooldown": 0.5})
        r.ab.trigger(r.ctx)
        r.run_until(Phase.IDLE)
        r.step(3)
        active = [(a, u) for ph, a, u in r.log if ph == Phase.ACTIVE]
        self.assertTrue(active)
        self.assertTrue(all(u and a >= 0.99 for a, u in active), "backdrop fully on during ACTIVE")
        after = [(a, u) for ph, a, u in r.log if ph in (Phase.COOLDOWN, Phase.IDLE)]
        self.assertTrue(after)
        self.assertTrue(all(a == 0.0 and not u for a, u in after), "clean return to camera")
        # The collapse ends with the backdrop gone.
        impact = [a for ph, a, _ in r.log if ph == Phase.IMPACT]
        self.assertEqual(impact[-1], 0.0)


class TestSlashes(unittest.TestCase):
    def count(self, density, capacity=900):
        r = Runner({"slash_density": density, "charge": 0.2, "active": 4.0, "impact": 0.3, "cooldown": 0.1},
                   w=640, h=360, capacity=capacity, render=False)
        r.ab.trigger(r.ctx)
        r.run_until(Phase.IMPACT)
        return r.ab.slash_total, r.ab.max_live_seen, r.ab.max_total(), r.ab._live_cap(r.ctx)

    def test_density_scales_and_bounded(self):
        low, _, _, _ = self.count(0.5)
        mid, _, _, _ = self.count(1.0)
        high, _, _, _ = self.count(2.0)
        self.assertGreater(low, 3)
        self.assertLess(low, mid)
        self.assertLess(mid, high)
        none, _, _, _ = self.count(0.0)
        self.assertEqual(none, 0)
        huge, live, bound, cap = self.count(1000.0)
        self.assertLessEqual(huge, bound)
        self.assertLessEqual(live, cap)
        self.assertLess(bound, 400)

    def test_live_cap_scales_with_quality(self):
        caps = [self.count(4.0, c)[1] for c in (400, 1600)]
        self.assertLessEqual(caps[0], 8)
        self.assertGreaterEqual(caps[1], caps[0])

    def test_deterministic(self):
        self.assertEqual(self.count(1.0)[0], self.count(1.0)[0])


class TestCancelReset(unittest.TestCase):
    def test_cancel_during_charging(self):
        r = Runner()
        r.ab.trigger(r.ctx)
        r.step(10)
        self.assertEqual(r.ab.phase, Phase.CHARGING)
        self.assertTrue(r.ab.trigger(r.ctx))
        self.assertEqual(r.ab.phase, Phase.IMPACT)
        r.run_until(Phase.IDLE)
        self.assertIsNone(r.ctx.locked_by)

    def test_cancel_during_active(self):
        r = Runner({"cooldown": 0.5})
        r.ab.trigger(r.ctx)
        r.run_until(Phase.ACTIVE)
        r.step(45)
        self.assertTrue(r.ab.trigger(r.ctx))
        self.assertEqual(r.ab.phase, Phase.IMPACT)
        self.assertFalse(r.ab.trigger(r.ctx), "no retrigger during the collapse")
        r.run_until(Phase.COOLDOWN)
        self.assertFalse(r.ab.trigger(r.ctx), "no retrigger during cooldown")
        r.run_until(Phase.IDLE)
        self.assertTrue(r.ab.trigger(r.ctx))

    def test_reset_mid_domain(self):
        r = Runner()
        r.ab.trigger(r.ctx)
        r.run_until(Phase.ACTIVE)
        r.step(20)
        r.ab.reset(r.ctx)
        self.assertEqual(r.ab.phase, Phase.IDLE)
        self.assertIsNone(r.ctx.locked_by)
        self.assertIsNone(r.ctx.energy_override)
        self.assertEqual(r.ab.slashes, [])
        r.step(3)
        self.assertEqual(r.log[-1][1], 0.0)
        self.assertTrue(r.ab.trigger(r.ctx))

    def test_tracking_loss_mid_domain(self):
        r = Runner({"charge": 0.6, "active": 2.5, "impact": 0.6, "cooldown": 0.2})
        r.ab.trigger(r.ctx)
        r.step(25)
        empty = TrackingState(width=960, height=540)
        r.step(40, tracking=empty)
        for s in r.ab.slashes:
            self.assertTrue(np.all(np.isfinite(s["c"])))
        self.assertTrue(np.all(np.isfinite(r.ab._anchor)))
        r.step(5, tracking=TrackingState())  # zero-size state too
        r.step(10)
        r.run_until(Phase.IDLE)

    def test_tracking_lost_from_start(self):
        r = Runner()
        r.ab.trigger(r.ctx)
        r.step(80, tracking=TrackingState(width=960, height=540))
        self.assertTrue(np.all(np.isfinite(r.ab._anchor)))


class TestResolutions(unittest.TestCase):
    def test_renders_at_common_sizes(self):
        for (w, h) in ((640, 360), (960, 540), (1280, 720)):
            for cap in (400, 1600):
                r = Runner({"charge": 0.5, "active": 1.2, "impact": 0.5, "cooldown": 0.1}, w=w, h=h, capacity=cap)
                r.ab.trigger(r.ctx)
                r.run_until(Phase.IDLE)
                out = r.step()
                self.assertEqual(out.shape, (h, w, 3))


class TestManager(unittest.TestCase):
    def harness(self):
        ctx = Context(cfg=Config().data, canvas=Canvas(0.33), particles=ParticleSystem(900), shaker=Shaker())
        return ctx, AbilityManager(ctx)

    def step(self, ctx, mgr, t, n=1):
        for _ in range(n):
            t += DT
            st = make_tracking(t=t)
            ctx.tracking, ctx.now, ctx.dt = st, t, DT
            mgr.update()
            ctx.canvas.begin(make_frame(st))
            mgr.render()
            ctx.canvas.composite()
        return t

    def test_mode_gating(self):
        ctx, mgr = self.harness()
        self.assertEqual(ctx.mode, "bleach")
        self.assertFalse(mgr.trigger(NAME), "not available in Bleach mode")
        mgr.set_mode("jjk")
        self.assertTrue(mgr.trigger(NAME))
        t = self.step(ctx, mgr, 0.0, 20)
        self.assertEqual(ctx.locked_by, NAME)
        self.assertIsNotNone(ctx.energy_override)
        self.assertTrue(mgr.trigger(NAME), "second press collapses the domain")
        self.assertEqual(mgr.abilities[NAME].phase, Phase.IMPACT)
        self.step(ctx, mgr, t, 60)
        mgr.set_mode("bleach")
        self.assertEqual(mgr.abilities[NAME].phase, Phase.IDLE)
        self.assertIsNone(ctx.locked_by)

    def test_refuses_while_unlimited_void_holds_lock(self):
        ctx, mgr = self.harness()
        mgr.set_mode("jjk")
        ctx.locked_by = "unlimited_void"
        self.assertFalse(mgr.trigger(NAME))
        self.assertEqual(mgr.abilities[NAME].phase, Phase.IDLE)
        self.assertEqual(ctx.locked_by, "unlimited_void")
        ctx.locked_by = None
        self.assertTrue(mgr.trigger(NAME))

    def test_reset_via_manager(self):
        ctx, mgr = self.harness()
        mgr.set_mode("jjk")
        mgr.trigger(NAME)
        self.step(ctx, mgr, 0.0, 70)
        mgr.reset()
        self.assertEqual(mgr.abilities[NAME].phase, Phase.IDLE)
        self.assertIsNone(ctx.locked_by)
        self.step(ctx, mgr, 3.0, 3)
        self.assertEqual(ctx.canvas.backdrop_amount, 0.0)


if __name__ == "__main__":
    unittest.main()
