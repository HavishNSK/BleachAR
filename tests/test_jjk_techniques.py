"""Jujutsu Kaisen cursed techniques: Blue, Red, Hollow Purple, Black Flash."""

import math
import unittest

import numpy as np

from bleach_ar.abilities import AbilityManager, Context, Phase, SwordPose
from bleach_ar.config import Config
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.jjk.techniques import BlackFlash, Blue, HollowPurple, Red, branching_bolts
from bleach_ar.synthetic import make_frame, make_tracking
from bleach_ar.tracking import TrackingState

DT = 1 / 30
CLASSES = {"blue": Blue, "red": Red, "hollow_purple": HollowPurple, "black_flash": BlackFlash}
NAMES = list(CLASSES)


class Runner:
    """Drives one technique directly (no manager, so ``ctx.sword`` is ours)."""

    def __init__(self, name, w=960, h=540, cap=900, params=None):
        self.cfg = Config().data
        p = dict(self.cfg["abilities"][name])
        p["cooldown"] = 0.3  # keep the suite fast; cooldown blocking is still exercised
        p.update(params or {})
        self.ctx = Context(cfg=self.cfg, canvas=Canvas(0.33), particles=ParticleSystem(cap), shaker=Shaker(),
                           mode="jjk")
        self.ab = CLASSES[name](p)
        self.w, self.h = w, h
        self.t = 0.0
        self.ctx.canvas.begin(np.zeros((h, w, 3), np.uint8))
        self.default = make_tracking(w, h, 0.0, gesture="open_palm", second_hand="open_palm")
        self.overrides = []
        self.render = True
        self._frames = {}

    def _frame(self, st):
        f = self._frames.get(id(st))
        if f is None:
            ok = st.width and all(not p.ok or np.all(np.isfinite(p.points)) for p in st.hands.values())
            f = (make_frame(st) if ok else np.full((self.h, self.w, 3), 40, np.uint8), st)
            self._frames[id(st)] = f  # keeps st alive so its id() stays unique
        return f[0].copy()

    def step(self, tracking=None, n=1, sword=None):
        ctx, out = self.ctx, None
        for _ in range(n):
            self.t += DT
            st = tracking if tracking is not None else self.default
            ctx.now, ctx.dt, ctx.tracking = self.t, DT, st
            ctx.energy_override = None
            if sword is not None:
                ctx.sword = sword
            self.ab.update(ctx)
            self.overrides.append((self.ab.phase, ctx.energy_override))
            ctx.particles.update(DT)
            assert np.all(np.isfinite(ctx.particles.pos[ctx.particles.life > 0]))
            if not self.render:
                continue
            frame = self._frame(st)
            ctx.canvas.begin(frame)
            ctx.canvas.set_person_src(st.mask)
            self.ab.render(ctx)
            ctx.particles.render(ctx.canvas)
            out = ctx.canvas.composite()
            ctx.canvas.flash *= math.exp(-6.0 * DT)
            assert out.shape == frame.shape and out.dtype == np.uint8
        return out

    def run_until(self, phase, tracking=None, limit=400, sword=None):
        for _ in range(limit):
            if self.ab.phase == phase:
                return
            self.step(tracking, sword=sword)
        raise AssertionError(f"never reached {phase}")


def _finite(*vs):
    return all(v is not None and np.all(np.isfinite(np.asarray(v, np.float64))) for v in vs)


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_full_lifecycle_each(self):
        for name in NAMES:
            r = Runner(name)
            self.assertTrue(r.ab.trigger(r.ctx), name)
            seen = [r.ab.phase]
            for _ in range(600):
                r.step()
                if r.ab.phase != seen[-1]:
                    seen.append(r.ab.phase)
                if r.ab.phase == Phase.IDLE:
                    break
            self.assertEqual(seen, [Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT, Phase.COOLDOWN, Phase.IDLE], name)
            # Aura recoloured while the technique is busy, never in cooldown.
            busy = [o for ph, o in r.overrides if ph in (Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT)]
            self.assertTrue(busy and all(o == tuple(int(c) for c in r.ab.body) for o in busy), name)
            self.assertTrue(all(o is None for ph, o in r.overrides if ph in (Phase.COOLDOWN, Phase.IDLE)), name)

    def test_retrigger_blocked_while_busy_and_cooling_down(self):
        for name in NAMES:
            r = Runner(name)
            self.assertTrue(r.ab.trigger(r.ctx))
            r.step(n=2)
            self.assertFalse(r.ab.trigger(r.ctx), f"{name} retriggered mid-charge")
            r.run_until(Phase.ACTIVE)
            self.assertFalse(r.ab.trigger(r.ctx))
            r.run_until(Phase.COOLDOWN)
            self.assertFalse(r.ab.trigger(r.ctx), f"{name} retriggered in cooldown")
            r.run_until(Phase.IDLE)
            self.assertTrue(r.ab.trigger(r.ctx))

    def test_keyboard_only_empty_tracking(self):
        for name in NAMES:
            for st in (TrackingState(width=960, height=540), TrackingState()):
                r = Runner(name)
                self.assertTrue(r.ab.trigger(r.ctx))
                for _ in range(400):
                    r.step(st)
                    if r.ab.phase == Phase.IDLE:
                        break
                self.assertEqual(r.ab.phase, Phase.IDLE, name)

    def test_state_clean_after_impact_and_reset(self):
        for name in NAMES:
            r = Runner(name)
            r.ab.trigger(r.ctx)
            r.run_until(Phase.COOLDOWN)
            a = r.ab
            self.assertTrue(all(getattr(a, k, None) is None for k in ("anchor", "point", "core_pos")), name)
            r.step(n=3)  # cooldown renders nothing and must not crash

    def test_renders_at_all_resolutions(self):
        for w, h in ((640, 360), (960, 540), (1280, 720)):
            for name in NAMES:
                r = Runner(name, w, h, cap={640: 400, 960: 900, 1280: 1600}[w])
                r.default = make_tracking(w, h, 0.0, gesture="open_palm", second_hand="open_palm")
                r.ab.trigger(r.ctx)
                r.run_until(Phase.COOLDOWN)

    def test_particle_counts_bounded(self):
        for cap in (400, 900, 1600):
            for name in NAMES:
                r = Runner(name, cap=cap, params={"particles": 3.0, "intensity": 3.0})
                r.render = cap == 900  # rendering is covered elsewhere; keep this fast
                for _ in range(2):
                    r.ab.trigger(r.ctx)
                    for _ in range(200):
                        r.step()
                        self.assertLessEqual(r.ctx.particles.alive_count, cap)
                        if r.ab.phase == Phase.COOLDOWN:
                            break
                    r.ab.reset(r.ctx)
        # Higher quality presets get more particles.
        counts = []
        for cap in (400, 1600):
            np.random.seed(0)
            r = Runner("red", cap=cap)
            r.ab.trigger(r.ctx)
            r.run_until(Phase.ACTIVE)
            counts.append(r.ctx.particles.alive_count)
        self.assertLess(counts[0], counts[1])


class TestTracking(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_tracking_lost_mid_charge_keeps_anchor(self):
        empty = TrackingState(width=960, height=540)
        bad = make_tracking(960, 540, gesture="open_palm", second_hand="open_palm")
        for part in bad.hands.values():
            part.points = part.points.copy()
            part.points[:, :2] = np.nan
        bad.pose.points = np.full_like(bad.pose.points, np.inf)
        for name, attr in (("blue", "anchor"), ("red", "anchor"), ("hollow_purple", "src_a"),
                           ("black_flash", "point")):
            for lost in (empty, bad):
                r = Runner(name, params={"charge": 1.0} if name != "hollow_purple" else None)
                r.render = lost is bad
                r.ab.trigger(r.ctx)
                r.step(n=3)
                before = np.array(getattr(r.ab, attr), np.float32)
                r.step(lost, n=5)
                self.assertEqual(r.ab.phase, Phase.CHARGING)
                after = getattr(r.ab, attr)
                self.assertTrue(_finite(after), name)
                np.testing.assert_allclose(after, before, atol=1e-3, err_msg=name)
                for _ in range(300):
                    r.step(lost)
                    if r.ab.phase == Phase.IDLE:
                        break
                self.assertEqual(r.ab.phase, Phase.IDLE)

    def test_blue_follows_hand_source(self):
        r = Runner("blue")
        st = make_tracking(960, 540, gesture="point")
        r.ab.trigger(r.ctx)
        r.step(st, n=4)
        tip = st.hands["right"].points[8, :2]
        self.assertLess(float(np.linalg.norm(r.ab.anchor - tip)), 60.0)

    def test_black_flash_at_confident_fist(self):
        r = Runner("black_flash")
        sw = SwordPose(visible=True, hand="right", grip=np.array([700.0, 300.0], np.float32),
                       tip=np.array([700.0, 150.0], np.float32), confidence=0.9, speed=1.5,
                       motion=np.array([1.0, 0.0], np.float32))
        r.ctx.sword = sw
        r.ab.trigger(r.ctx)
        np.testing.assert_allclose(r.ab.point, [700, 300])
        np.testing.assert_allclose(r.ab.dir, [1, 0], atol=1e-5)
        # Without a confident fist it falls back to the fist hand's knuckles.
        r2 = Runner("black_flash")
        st = make_tracking(960, 540, gesture="fist")
        r2.ctx.tracking = st
        r2.ab.trigger(r2.ctx)
        knuckles = st.hands["right"].points[[5, 9, 13, 17], :2].mean(0)
        np.testing.assert_allclose(r2.ab.point, knuckles, atol=1e-3)


class TestHollowPurple(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_sources_separate_then_converge(self):
        r = Runner("hollow_purple")
        ab = r.ab
        ab.trigger(r.ctx)
        diag = math.hypot(960, 540)
        dists, last = [], None
        while ab.phase == Phase.CHARGING:
            d = float(np.linalg.norm(ab.blue_pos - ab.red_pos))
            dists.append((ab.progress, d))
            last = d
            # Blue on the screen-left source, Red on the screen-right one, until they merge.
            if ab.progress < ab.SPLIT:
                self.assertGreater(d, 0.1 * diag)
                self.assertLess(ab.blue_pos[0], ab.red_pos[0])
            r.step()
        self.assertLess(last, 1.0, "sources must have merged by the end of CHARGING")
        early = [d for p, d in dists if p < ab.SPLIT]
        late = [d for p, d in dists if p > (ab.SPLIT + ab.MERGE) / 2]
        self.assertGreater(min(early), max(late))
        # Blue started near the left hand, Red near the right one.
        left = make_tracking(960, 540, gesture="open_palm", second_hand="open_palm")
        self.assertLess(left.hands["left"].points[0, 0], left.hands["right"].points[0, 0])

    def _flight(self, sword=None, tracking=None):
        r = Runner("hollow_purple")
        r.ab.trigger(r.ctx)
        r.run_until(Phase.ACTIVE, tracking=tracking, sword=sword)
        ab = r.ab
        p0, d = ab.p0.copy(), ab.dir.copy()
        pos = []
        while ab.phase == Phase.ACTIVE:
            pos.append(ab.pos.copy())
            r.step(tracking, sword=sword)
        self.assertEqual(ab.phase, Phase.IMPACT)
        return ab, p0, d, np.array(pos)

    def test_projectile_moves_along_launch_direction_and_impacts_at_end(self):
        ab, p0, d, pos = self._flight()
        rel = pos - p0
        along = rel @ d
        across = rel @ np.array([-d[1], d[0]])
        self.assertTrue(np.all(np.diff(along) >= -1e-3), "must keep moving forward")
        self.assertGreater(along[-1] - along[0], 50.0)
        self.assertLess(float(np.abs(across).max()), 1.0)
        end, _ = ab._flight(1.0)
        np.testing.assert_allclose(ab.end, end, atol=1e-3)
        self.assertLess(float(np.linalg.norm(ab.end - pos[-1])), 0.15 * math.hypot(960, 540))
        self.assertGreater(float(np.dot(ab.end - p0, d)), 0.0)
        # Impact rendering is centred on the end point and it then settles.
        self.assertTrue(ab.forward, "no sword / pointing hand -> screen-forward with shrink")
        self.assertLess(ab.end_scale, 0.5)

    def test_direction_follows_sword(self):
        for want in ((1.0, 0.0), (-0.6, -0.8), (0.0, 1.0)):
            sw = SwordPose(visible=True, hand="right", grip=np.array([480.0, 300.0], np.float32),
                           tip=np.array([480.0, 200.0], np.float32),
                           direction=np.array(want, np.float32), confidence=0.9, speed=0.0)
            ab, p0, d, pos = self._flight(sword=sw)
            np.testing.assert_allclose(d, want, atol=1e-4)
            self.assertFalse(ab.forward)
            self.assertGreater(float(np.dot(pos[-1] - p0, d)), 20.0)
            # Impact lands inside the frame.
            self.assertTrue(0 <= ab.end[0] <= 960 and 0 <= ab.end[1] <= 540, ab.end)
        # A fast swing uses the swing motion instead of the blade direction.
        sw = SwordPose(visible=True, grip=np.array([480.0, 300.0], np.float32), tip=np.array([480.0, 200.0]),
                       direction=np.array([0.0, -1.0], np.float32), confidence=0.9, speed=2.0,
                       motion=np.array([-1.0, 0.0], np.float32))
        _, _, d, _ = self._flight(sword=sw)
        np.testing.assert_allclose(d, [-1, 0], atol=1e-4)

    def test_pointing_hand_direction(self):
        st = make_tracking(960, 540, gesture="point", hand_angle=-180.0)
        _, _, d, _ = self._flight(tracking=st)
        pts = st.hands["right"].points
        want = (pts[8, :2] - pts[5, :2]) / np.linalg.norm(pts[8, :2] - pts[5, :2])
        np.testing.assert_allclose(d, want, atol=1e-4)

    def test_keyboard_only_hollow_purple(self):
        st = TrackingState(width=960, height=540)
        r = Runner("hollow_purple")
        r.ab.trigger(r.ctx)
        r.step(st, n=3)
        self.assertLess(r.ab.blue_pos[0], r.ab.red_pos[0])
        ab, p0, d, pos = self._flight(tracking=st)
        self.assertTrue(_finite(p0, d, pos, ab.end))

    def test_reset_mid_flight(self):
        r = Runner("hollow_purple")
        r.ab.trigger(r.ctx)
        r.run_until(Phase.ACTIVE)
        r.step(n=5)
        r.ab.reset(r.ctx)
        self.assertEqual(r.ab.phase, Phase.IDLE)
        self.assertIsNone(r.ab.core_pos)
        self.assertEqual(r.ab.dist, 0.0)
        r.step(n=3)
        self.assertTrue(r.ab.trigger(r.ctx))
        r.run_until(Phase.IDLE)


class TestFields(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def _probe(self, name):
        r = Runner(name)
        r.ab.trigger(r.ctx)
        r.run_until(Phase.ACTIVE)
        r.step()
        c = (r.ab.anchor).copy()
        ps = r.ctx.particles
        ps.clear()
        ps.emit(1, c + np.array([60.0, 0.0], np.float32), (0.0, 0.0), life=(5, 5), drag=0.0)
        r.step(n=2)
        i = int(np.argmax(ps.life))
        return float(ps.vel[i, 0])

    def test_blue_attracts_red_repels(self):
        self.assertLess(self._probe("blue"), 0.0)  # pulled back toward the core (-x)
        self.assertGreater(self._probe("red"), 0.0)  # pushed away (+x)

    def test_branching_bolts_bounded(self):
        import random
        for seed in range(20):
            bolts = branching_bolts(random.Random(seed), (100.0, 100.0), 0.3, 200.0, depth=2, branch_p=1.0)
            self.assertLessEqual(len(bolts), 1 + 4 + 16)
            self.assertTrue(all(np.all(np.isfinite(p)) and p.shape[1] == 2 for p, _ in bolts))
            np.testing.assert_allclose(bolts[-1][0][0], (100, 100))  # trunk starts at the origin
            self.assertEqual(bolts[-1][1], 0)


class TestManager(unittest.TestCase):
    def _mgr(self):
        cfg = Config().data
        ctx = Context(cfg=cfg, canvas=Canvas(0.33), particles=ParticleSystem(900), shaker=Shaker())
        mgr = AbilityManager(ctx)
        return ctx, mgr

    def _frame(self, ctx, mgr, t, st):
        ctx.tracking, ctx.now, ctx.dt = st, t, DT
        mgr.update()
        ctx.canvas.begin(make_frame(st) if st.width else np.zeros((540, 960, 3), np.uint8))
        mgr.render()
        return ctx.canvas.composite()

    def test_mode_gating(self):
        ctx, mgr = self._mgr()
        mgr.set_mode("bleach")
        for name in NAMES:
            self.assertFalse(mgr.trigger(name), f"{name} must not fire in Bleach mode")
        mgr.set_mode("jjk")
        st = make_tracking(gesture="open_palm", second_hand="open_palm")
        t = 0.0
        for name in NAMES:
            self.assertTrue(mgr.trigger(name), name)
            for _ in range(400):
                t += DT
                self._frame(ctx, mgr, t, st)
                if mgr.abilities[name].phase in (Phase.COOLDOWN, Phase.IDLE):
                    break
            self.assertIn(mgr.abilities[name].phase, (Phase.COOLDOWN, Phase.IDLE))
        self.assertFalse(mgr.trigger("getsuga"))
        # Switching mode resets everything mid-technique.
        mgr.abilities["blue"].reset(ctx)
        self.assertTrue(mgr.trigger("blue"))
        self._frame(ctx, mgr, t + DT, st)
        mgr.set_mode("bleach")
        self.assertEqual(mgr.abilities["blue"].phase, Phase.IDLE)
        self.assertIsNone(mgr.abilities["blue"].anchor)

    def test_all_four_at_once_keyboard_only(self):
        ctx, mgr = self._mgr()
        mgr.set_mode("jjk")
        st = TrackingState(width=960, height=540)
        for name in NAMES:
            self.assertTrue(mgr.trigger(name))
        t = 0.0
        for _ in range(150):
            t += DT
            self._frame(ctx, mgr, t, st)
        self.assertTrue(all(mgr.abilities[n].phase in (Phase.COOLDOWN, Phase.IDLE) for n in NAMES))


if __name__ == "__main__":
    unittest.main()
