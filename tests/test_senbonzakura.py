"""Senbonzakura Kageyoshi: Bankai activation, blade swarm, commanded attack,
release, reset, tracking loss and style switching (deterministic, headless)."""

import unittest

import numpy as np

from bleach_ar.abilities import BANKAI, SEALED, SHIKAI, AbilityManager, Context, Phase
from bleach_ar.app import cycle_bankai_style
from bleach_ar.config import Config
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.senbonzakura import ATTACK, BURST, CAP, DISPERSE, FORMATION, OFF, ORBIT, RETURN
from bleach_ar.synthetic import make_frame, make_tracking
from bleach_ar.tracking import TrackingState

DT = 1 / 30
W, H = 960, 540


class Harness:
    def __init__(self, style="senbonzakura", capacity=900, **cfg_over):
        np.random.seed(0)
        self.cfg = Config(cfg_over).data
        self.ctx = Context(cfg=self.cfg, canvas=Canvas(0.33), particles=ParticleSystem(capacity), shaker=Shaker(),
                           bankai_style=style)
        self.mgr = AbilityManager(self.ctx)
        self.t = 0.0
        self.max_live = 0

    @property
    def sw(self):
        return self.mgr.swarm

    def step(self, n=1, tracking=None, frame=None):
        for _ in range(n):
            self.t += DT
            st = tracking if tracking is not None else make_tracking(W, H, t=self.t)
            self.ctx.tracking = st
            self.ctx.now, self.ctx.dt = self.t, DT
            self.mgr.update()
            img = frame.copy() if frame is not None else (
                make_frame(st) if st.width else np.zeros((H, W, 3), np.uint8))
            self.ctx.canvas.begin(img)
            self.mgr.render()
            out = self.ctx.canvas.composite()
            assert out.shape == (H, W, 3)
            self.max_live = max(self.max_live, self.sw.live_count)
            n_ = self.sw.n
            assert np.isfinite(self.sw.pos[:n_]).all() and np.isfinite(self.sw.dir[:n_]).all()
            assert 0.0 <= self.ctx.sword_sink <= 1.0

    def until_idle(self, name, limit=600, tracking=None):
        ab = self.mgr.abilities[name]
        for _ in range(limit):
            if ab.phase == Phase.IDLE:
                return
            self.step(tracking=tracking)
        raise AssertionError(f"{name} never returned to IDLE")

    def enter_bankai(self):
        assert self.mgr.trigger("bankai")
        self.until_idle("bankai")


class TestActivation(unittest.TestCase):
    def test_full_activation_phases(self):
        h = Harness()
        bk = h.mgr.abilities["bankai"]
        self.assertTrue(h.mgr.trigger("bankai"))
        self.assertEqual(bk.style, "senbonzakura")
        seen, sinks, pillars = set(), [], False
        for _ in range(400):
            if bk.phase == Phase.IDLE:
                break
            seen.add(bk.phase)
            h.step()
            sinks.append(h.ctx.sword_sink)
            pillars |= h.sw.pillars.on and h.sw.pillars.rise().max(initial=0) > 0.5
        self.assertEqual(seen, {Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT, Phase.COOLDOWN})
        self.assertTrue(pillars, "blade pillars rose during the charge")
        self.assertEqual(sinks[0], 0.0, "sword starts normal")
        self.assertTrue(all(b >= a - 1e-6 for a, b in zip(sinks, sinks[1:])), "sword sinks monotonically")
        self.assertEqual(h.ctx.level, BANKAI)
        self.assertIsNone(h.ctx.locked_by)
        self.assertEqual(h.ctx.sword_sink, 1.0)
        self.assertEqual(h.sw.mode, ORBIT)
        self.assertGreater(h.sw.live_count, 50)
        # Persistent: stays orbiting with the sword hidden.
        h.step(60)
        self.assertEqual(h.sw.mode, ORBIT)
        self.assertEqual(h.ctx.sword_sink, 1.0)
        self.assertEqual(h.ctx.energy_color(), tuple(h.cfg["abilities"]["senbonzakura"]["color"]))

    def test_title_card(self):
        h = Harness()
        h.mgr.trigger("bankai")
        bk = h.mgr.abilities["bankai"]
        while bk.phase != Phase.IMPACT:
            h.step()
        self.assertTrue(any(t.text == "SENBONZAKURA KAGEYOSHI" and t.sub == "Bankai" for t in h.ctx.titles))

    def test_single_modest_flash(self):
        h = Harness()
        h.mgr.trigger("bankai")
        peak = 0.0
        while h.mgr.abilities["bankai"].phase != Phase.IDLE:
            h.step()
            peak = max(peak, h.ctx.canvas.flash)
            h.ctx.canvas.flash *= np.exp(-6 * DT)  # app.py decays it
        self.assertLess(peak, 0.6)

    def test_lock_blocks_retrigger_and_others(self):
        h = Harness()
        h.mgr.trigger("bankai")
        h.step(5)
        self.assertEqual(h.ctx.locked_by, "bankai")
        self.assertFalse(h.mgr.trigger("bankai"))
        for name in ("getsuga", "cero", "shikai", "senbon_attack"):
            self.assertFalse(h.mgr.trigger(name), name)
        self.assertFalse(cycle_bankai_style(h.ctx, h.mgr))
        self.assertEqual(h.ctx.bankai_style, "senbonzakura")
        h.step(40)  # into the release / title
        self.assertFalse(h.mgr.trigger("getsuga"))
        self.assertFalse(cycle_bankai_style(h.ctx, h.mgr))

    def test_style_stored_at_trigger(self):
        h = Harness()
        h.mgr.trigger("bankai")
        h.step(3)
        h.ctx.bankai_style = "tensa"  # programmatic change mid-sequence
        h.step(3)
        self.assertEqual(h.mgr.abilities["bankai"].style, "senbonzakura")
        h.ctx.bankai_style = "senbonzakura"
        h.until_idle("bankai")
        self.assertEqual(h.ctx.sword_sink, 1.0)


class TestRouting(unittest.TestCase):
    def test_getsuga_key_routes_in_senbonzakura_bankai(self):
        h = Harness()
        h.enter_bankai()
        self.assertEqual(h.mgr.routed("getsuga"), "senbon_attack")
        self.assertTrue(h.mgr.trigger("getsuga"))
        self.assertEqual(h.mgr.abilities["senbon_attack"].phase, Phase.CHARGING)
        self.assertEqual(h.mgr.abilities["getsuga"].phase, Phase.IDLE)

    def test_getsuga_unchanged_elsewhere(self):
        for level, style in ((SEALED, "senbonzakura"), (SHIKAI, "senbonzakura"), (BANKAI, "tensa"),
                             (SEALED, "tensa")):
            h = Harness(style=style)
            h.ctx.level = level
            self.assertTrue(h.mgr.trigger("getsuga"), f"{level}/{style}")
            self.assertEqual(h.mgr.abilities["getsuga"].phase, Phase.CHARGING)
            self.assertEqual(h.mgr.abilities["senbon_attack"].phase, Phase.IDLE)

    def test_attack_only_in_senbonzakura_bankai(self):
        for level, style in ((SEALED, "senbonzakura"), (BANKAI, "tensa")):
            h = Harness(style=style)
            h.ctx.level = level
            self.assertFalse(h.mgr.trigger("senbon_attack"))


class TestAttack(unittest.TestCase):
    def test_attack_lifecycle_returns_to_orbit(self):
        h = Harness()
        h.enter_bankai()
        atk = h.mgr.abilities["senbon_attack"]
        self.assertTrue(h.mgr.trigger("getsuga"))
        self.assertFalse(h.mgr.trigger("getsuga"), "no retrigger mid-attack")
        modes = {}
        for _ in range(300):
            if atk.phase == Phase.IDLE:
                break
            modes.setdefault(atk.phase, set()).add(h.sw.mode)
            h.step()
        self.assertEqual(atk.phase, Phase.IDLE)
        self.assertIn(FORMATION, modes[Phase.CHARGING])
        self.assertIn(ATTACK, modes[Phase.ACTIVE])
        self.assertIn(BURST, modes[Phase.IMPACT])
        self.assertTrue(modes[Phase.COOLDOWN] & {RETURN, ORBIT})
        h.step(40)
        self.assertEqual(h.sw.mode, ORBIT)
        self.assertEqual(h.ctx.level, BANKAI)
        self.assertGreater(h.sw.live_count, 50)

    def test_blades_reach_impact_point(self):
        h = Harness()
        h.enter_bankai()
        atk = h.mgr.abilities["senbon_attack"]
        h.mgr.trigger("getsuga")
        while atk.phase != Phase.IMPACT:
            h.step()
        n = h.sw.n
        dist = np.linalg.norm(h.sw.pos[:n] - atk.end, axis=1)
        self.assertGreater(np.mean(dist < 0.15 * h.ctx.diag), 0.5, "most blades converge on the target")

    def test_release_during_attack(self):
        h = Harness()
        h.enter_bankai()
        h.mgr.trigger("getsuga")
        h.step(12)
        self.assertTrue(h.mgr.trigger("bankai"), "Bankai can be released mid-attack")
        h.until_idle("bankai")
        h.step(30)
        self.assertEqual(h.ctx.level, SEALED)
        self.assertEqual(h.mgr.abilities["senbon_attack"].phase, Phase.IDLE)
        self.assertEqual(h.sw.live_count, 0)


class TestRelease(unittest.TestCase):
    def test_seal_disperses_and_resurfaces_sword(self):
        h = Harness()
        h.enter_bankai()
        self.assertTrue(h.mgr.trigger("bankai"))
        self.assertEqual(h.mgr.abilities["bankai"].mode, "seal")
        h.step(2)
        self.assertEqual(h.sw.mode, DISPERSE)
        frames = 0
        while h.sw.live_count > 0:
            h.step()
            frames += 1
            self.assertLess(frames, 45, "swarm must fade within ~1.5 s")
        h.until_idle("bankai")
        self.assertEqual(h.ctx.level, SEALED)
        self.assertEqual(h.ctx.sword_sink, 0.0)
        self.assertEqual(h.sw.mode, OFF)
        self.assertIsNone(h.ctx.locked_by)
        # Style may be switched again once sealed and idle.
        self.assertTrue(cycle_bankai_style(h.ctx, h.mgr))
        self.assertEqual(h.ctx.bankai_style, "tensa")


class TestReset(unittest.TestCase):
    def _assert_clear(self, h):
        self.assertEqual(h.ctx.level, SEALED)
        self.assertIsNone(h.ctx.locked_by)
        self.assertEqual(h.ctx.sword_sink, 0.0)
        self.assertEqual(h.sw.live_count, 0)
        self.assertEqual(h.sw.mode, OFF)
        self.assertFalse(h.sw.pillars.on)
        self.assertTrue(all(a.phase == Phase.IDLE for a in h.mgr.abilities.values()))
        h.step(5)
        self.assertEqual(h.sw.live_count, 0)
        self.assertEqual(h.ctx.sword_sink, 0.0)

    def test_reset_mid_activation(self):
        for frames in (10, 40, 70, 90):  # dim, sink + pillars, release, title
            h = Harness()
            h.mgr.trigger("bankai")
            h.step(frames)
            h.mgr.reset()
            self._assert_clear(h)

    def test_reset_mid_attack(self):
        h = Harness()
        h.enter_bankai()
        h.mgr.trigger("getsuga")
        h.step(20)
        h.mgr.reset()
        self._assert_clear(h)


class TestRobustness(unittest.TestCase):
    def test_tracking_lost_holds_anchor(self):
        h = Harness()
        h.enter_bankai()
        c0 = h.sw.center.copy()
        empty = TrackingState(width=W, height=H)
        h.step(30, tracking=empty)
        np.testing.assert_allclose(h.sw.center, c0, atol=1e-3)
        self.assertTrue(h.mgr.trigger("getsuga"))
        h.step(80, tracking=empty)
        h.step(5, tracking=TrackingState())  # zero-size state too
        np.testing.assert_allclose(h.sw.center, c0, atol=1e-3)
        h.step(20)  # person comes back
        self.assertTrue(np.isfinite(h.sw.center).all())

    def test_nan_landmarks_are_ignored(self):
        h = Harness()
        h.enter_bankai()
        st = make_tracking(W, H)
        st.pose.points = st.pose.points.copy()
        st.pose.points[11, :2] = np.nan
        st.hands["right"].points = st.hands["right"].points.copy()
        st.hands["right"].points[0, :2] = np.inf
        h.mgr.trigger("getsuga")
        h.step(60, tracking=st, frame=np.zeros((H, W, 3), np.uint8))

    def test_blade_count_never_exceeds_cap(self):
        h = Harness(capacity=1600, abilities={"senbonzakura": {"count": 5000}})
        h.enter_bankai()
        h.mgr.trigger("getsuga")
        h.until_idle("senbon_attack")
        self.assertLessEqual(h.max_live, CAP)
        self.assertLessEqual(h.sw.n, CAP)

    def test_level_set_directly_spawns_orbit(self):
        h = Harness()
        h.ctx.level = BANKAI
        h.step(5)
        self.assertEqual(h.sw.mode, ORBIT)
        self.assertEqual(h.ctx.sword_sink, 1.0)
        h.ctx.level = SEALED
        h.step(45)
        self.assertEqual(h.sw.live_count, 0)
        self.assertEqual(h.ctx.sword_sink, 0.0)


class TestTensaUnchanged(unittest.TestCase):
    def test_tensa_has_no_swarm_and_keeps_sword(self):
        h = Harness(style="tensa")
        h.mgr.trigger("bankai")
        while h.mgr.abilities["bankai"].phase != Phase.IDLE:
            h.step()
            self.assertEqual(h.sw.live_count, 0)
            self.assertFalse(h.sw.pillars.on)
            self.assertEqual(h.ctx.sword_sink, 0.0)
        self.assertEqual(h.ctx.level, BANKAI)
        self.assertEqual(h.ctx.energy_color(), tuple(h.cfg["abilities"]["bankai"]["color"]))
        self.assertTrue(h.mgr.trigger("getsuga"))
        self.assertEqual(h.mgr.abilities["getsuga"].phase, Phase.CHARGING)
        h.until_idle("getsuga")
        h.mgr.trigger("bankai")
        h.until_idle("bankai")
        self.assertEqual(h.ctx.level, SEALED)

    def test_cycle_style_toggles_when_idle(self):
        h = Harness()
        self.assertTrue(cycle_bankai_style(h.ctx, h.mgr))
        self.assertEqual(h.ctx.bankai_style, "tensa")
        self.assertTrue(cycle_bankai_style(h.ctx, h.mgr))
        self.assertEqual(h.ctx.bankai_style, "senbonzakura")
        h.enter_bankai()
        self.assertFalse(cycle_bankai_style(h.ctx, h.mgr), "no switching while in Bankai")
        self.assertEqual(h.ctx.bankai_style, "senbonzakura")


if __name__ == "__main__":
    unittest.main()
