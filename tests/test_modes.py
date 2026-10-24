"""Mode registry, mode switching and cross-mode isolation."""

import unittest

from bleach_ar.abilities import BANKAI, SEALED, Phase
from bleach_ar.modes import ABILITY_CLASSES, BLEACH, JJK, MODE_ORDER, MODES, load_class, mode_for_ability
from tests.test_integration import Harness


class TestRegistry(unittest.TestCase):
    def test_every_ability_registered_to_exactly_one_mode(self):
        for aid in ABILITY_CLASSES:
            owners = [m.id for m in MODES.values() if aid in m.ability_ids]
            self.assertEqual(len(owners), 1, f"{aid} owned by {owners}")
            self.assertEqual(mode_for_ability(aid), owners[0])

    def test_classes_load_and_names_match(self):
        for aid in ABILITY_CLASSES:
            self.assertEqual(load_class(aid).name, aid)

    def test_panels_do_not_leak(self):
        bleach_ids = {a.id for a in MODES[BLEACH].abilities}
        jjk_ids = {a.id for a in MODES[JJK].abilities}
        self.assertFalse(bleach_ids & jjk_ids)
        self.assertIn("unlimited_void", jjk_ids)
        self.assertIn("getsuga", bleach_ids)
        self.assertEqual(MODE_ORDER, [BLEACH, JJK])

    def test_jjk_categories(self):
        cats = {a.id: a.category for a in MODES[JJK].abilities}
        self.assertEqual(cats["unlimited_void"], "Domain Expansion")
        self.assertEqual(cats["malevolent_shrine"], "Domain Expansion")
        self.assertEqual(cats["hollow_purple"], "Cursed Technique")


class TestModeSwitching(unittest.TestCase):
    def test_default_mode_is_bleach_and_jjk_abilities_blocked(self):
        h = Harness()
        self.assertEqual(h.ctx.mode, BLEACH)
        for aid in ("blue", "red", "hollow_purple", "black_flash", "unlimited_void", "malevolent_shrine"):
            self.assertFalse(h.mgr.trigger(aid), aid)
        self.assertTrue(h.mgr.trigger("getsuga"))

    def test_bleach_abilities_blocked_in_jjk(self):
        h = Harness()
        h.mgr.set_mode(JJK)
        for aid in ("getsuga", "shikai", "bankai", "cero", "senbon_attack"):
            self.assertFalse(h.mgr.trigger(aid), aid)
        self.assertEqual(set(h.mgr.active_abilities()), set(MODES[JJK].ability_ids))

    def test_switch_cleans_up_previous_mode(self):
        h = Harness()
        h.ctx.bankai_style = "senbonzakura"
        h.mgr.trigger("bankai")
        h.run_until_idle("bankai")
        self.assertEqual(h.ctx.level, BANKAI)
        h.mgr.trigger("getsuga")  # swarm attack in flight
        h.step(n=3)
        h.mgr.set_mode(JJK)
        self.assertEqual(h.ctx.level, SEALED)
        self.assertIsNone(h.ctx.locked_by)
        self.assertEqual(h.ctx.sword_sink, 0.0)
        self.assertEqual(h.ctx.swarm.live_count, 0)
        self.assertTrue(all(a.phase == Phase.IDLE for a in h.mgr.abilities.values()))
        self.assertEqual(h.ctx.particles.alive_count, 0)
        self.assertFalse(h.ctx.show_sword)
        h.step(n=5)  # nothing from Bleach keeps running or drawing
        self.assertEqual(h.ctx.swarm.live_count, 0)

    def test_switch_mid_jjk_ability_and_back(self):
        h = Harness()
        h.mgr.set_mode(JJK)
        for aid in ("unlimited_void", "hollow_purple"):
            self.assertTrue(h.mgr.trigger(aid), aid)
            h.step(n=4)
            h.mgr.set_mode(BLEACH)
            self.assertTrue(all(a.phase == Phase.IDLE for a in h.mgr.abilities.values()))
            h.step(n=3)
            h.mgr.set_mode(JJK)
        self.assertTrue(h.mgr.trigger("unlimited_void"), "not stuck in cooldown after a switch")

    def test_unknown_mode_rejected(self):
        h = Harness()
        with self.assertRaises(KeyError):
            h.mgr.set_mode("naruto")



class TestDomainRules(unittest.TestCase):
    def test_collapse_press_does_not_cancel_other_techniques(self):
        h = Harness()
        h.mgr.set_mode(JJK)
        self.assertTrue(h.mgr.trigger("unlimited_void"))
        uv = h.mgr.abilities["unlimited_void"]
        for _ in range(200):
            if uv.phase == Phase.ACTIVE:
                break
            h.step()
        self.assertTrue(h.mgr.trigger("red"))
        h.step(n=2)
        h.mgr.trigger("unlimited_void")  # early collapse
        self.assertNotEqual(h.mgr.abilities["red"].phase, Phase.IDLE)

    def test_only_one_domain_at_a_time(self):
        h = Harness()
        h.mgr.set_mode(JJK)
        self.assertTrue(h.mgr.trigger("unlimited_void"))
        uv = h.mgr.abilities["unlimited_void"]
        for _ in range(200):
            if uv.phase == Phase.ACTIVE:
                break
            h.step()
        self.assertFalse(h.mgr.trigger("malevolent_shrine"))


if __name__ == "__main__":
    unittest.main()
