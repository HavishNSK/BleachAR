"""Regression tests for issues found in code review."""

import unittest

from bleach_ar.abilities import BANKAI, Phase
from bleach_ar.synthetic import make_tracking
from bleach_ar.sword import Zanpakuto
from tests.test_integration import Harness


class TestTransformCancelsAttacks(unittest.TestCase):
    def _senbon_bankai(self):
        h = Harness()
        h.ctx.bankai_style = "senbonzakura"
        self.assertTrue(h.mgr.trigger("bankai"))
        h.run_until_idle("bankai")
        self.assertEqual(h.ctx.level, BANKAI)
        return h

    def test_seal_during_attack_charge_keeps_swarm_dispersing(self):
        from bleach_ar.senbonzakura import ATTACK, BURST
        h = self._senbon_bankai()
        self.assertTrue(h.mgr.trigger("getsuga"))  # routed to the swarm attack
        atk = h.mgr.abilities["senbon_attack"]
        h.step(n=3)
        self.assertEqual(atk.phase, Phase.CHARGING)
        self.assertTrue(h.mgr.trigger("bankai"))  # release Bankai mid-charge
        self.assertEqual(atk.phase, Phase.IDLE)
        for _ in range(40):
            h.step()
            self.assertNotIn(h.ctx.swarm.mode, (ATTACK, BURST), "attack must not re-command a dispersing swarm")

    def test_bankai_cancels_getsuga_in_flight(self):
        h = Harness()
        h.ctx.bankai_style = "tensa"
        self.assertTrue(h.mgr.trigger("getsuga"))
        h.step(n=2)
        self.assertTrue(h.mgr.trigger("bankai"))
        self.assertEqual(h.mgr.abilities["getsuga"].phase, Phase.IDLE)
        self.assertEqual(h.mgr.abilities["bankai"].phase, Phase.CHARGING)


class TestSummonIsTimeBased(unittest.TestCase):
    def _frames_to_summon(self, dt):
        h = Harness()
        z = Zanpakuto(h.cfg)
        st = make_tracking(gesture="fist")
        h.ctx.tracking = st
        for i in range(1, 60):
            h.ctx.now, h.ctx.dt = i * dt, dt
            z.update(h.ctx)
            if z.summoned:
                return i
        return None

    def test_summon_delay_similar_across_frame_rates(self):
        n60 = self._frames_to_summon(1 / 60)
        n30 = self._frames_to_summon(1 / 30)
        self.assertIsNotNone(n60)
        self.assertIsNotNone(n30)
        self.assertGreaterEqual(n60 / 60, 2 / 30 - 1e-6, "at 60 FPS the summon must still wait ~2/30 s")
        self.assertLessEqual(abs(n60 / 60 - n30 / 30), 1 / 30 + 1e-6)



class TestMultiModeRegressions(unittest.TestCase):
    def test_resolution_change_mid_domain(self):
        from tests.test_integration import DT
        from bleach_ar.synthetic import make_frame
        for dom in ("malevolent_shrine", "unlimited_void"):
            h = Harness()
            h.mgr.set_mode("jjk")
            self.assertTrue(h.mgr.trigger(dom))
            for i in range(80):
                w, hh = (960, 540) if i < 40 else (640, 360)
                h.t += DT
                st = make_tracking(w, hh, t=h.t)
                h.ctx.tracking = st
                h.ctx.now, h.ctx.dt = h.t, DT
                h.mgr.update()
                h.ctx.canvas.begin(make_frame(st))
                h.mgr.render()
                h.ctx.canvas.composite()

    def test_reset_clears_screen_shake(self):
        h = Harness()
        h.ctx.shaker.kick(30)
        h.mgr.set_mode("jjk")
        self.assertEqual(h.ctx.shaker.amount, 0.0)


if __name__ == "__main__":
    unittest.main()
