"""Unit tests for config, state machine, gestures and camera error handling."""

import json
import os
import tempfile
import unittest

import numpy as np

from bleach_ar.abilities import Ability, Context, Phase
from bleach_ar.camera import Camera, CameraError
from bleach_ar.config import Config, deep_merge
from bleach_ar.fx import Canvas, ParticleSystem, Shaker
from bleach_ar.gestures import GestureRecognizer, classify_hand
from bleach_ar.synthetic import make_tracking
from bleach_ar.tracking import TrackingState


def make_ctx(cfg=None) -> Context:
    cfg = cfg or Config().data
    return Context(cfg=cfg, canvas=Canvas(), particles=ParticleSystem(100), shaker=Shaker())


class Recorder(Ability):
    name = "rec"

    def __init__(self, params):
        super().__init__(params)
        self.entered = []

    def on_enter(self, phase, ctx):
        self.entered.append(phase)


class TestConfig(unittest.TestCase):
    def test_deep_merge_keeps_unspecified_defaults(self):
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}})
        self.assertEqual(merged, {"a": {"b": 5, "c": 2}})

    def test_partial_override(self):
        cfg = Config({"abilities": {"cero": {"cooldown": 9}}})
        self.assertEqual(cfg["abilities"]["cero"]["cooldown"], 9)
        self.assertIn("charge", cfg["abilities"]["cero"])

    def test_invalid_quality(self):
        with self.assertRaises(ValueError):
            Config({"quality": "ultra"})

    def test_bad_json_exits_with_message(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{not json")
        try:
            with self.assertRaises(SystemExit):
                Config.load(fh.name)
        finally:
            os.unlink(fh.name)

    def test_roundtrip_and_cycle_quality(self):
        cfg = Config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            cfg.save(path)
            with open(path) as fh:
                self.assertEqual(json.load(fh)["quality"], "medium")
        self.assertEqual(cfg.cycle_quality(), "high")
        self.assertEqual(cfg.cycle_quality(), "low")

    def test_missing_asset_falls_back(self):
        cfg = Config({"assets": {"sword_image": "does/not/exist.png", "mask_image": "does/not/exist_mask.png"}})
        self.assertIsNone(cfg.asset_path("sword_image"))
        self.assertIsNone(cfg.asset_path("mask_image"))


class TestStateMachine(unittest.TestCase):
    def test_full_cycle_and_no_retrigger(self):
        ctx = make_ctx()
        ab = Recorder({"charge": 0.2, "active": 0.3, "impact": 0.1, "cooldown": 0.5})
        self.assertTrue(ab.trigger(ctx))
        self.assertEqual(ab.phase, Phase.CHARGING)
        self.assertFalse(ab.trigger(ctx), "must not restart while running")
        seen = []
        ctx.dt = 0.05
        for _ in range(40):
            ab.update(ctx)
            seen.append(ab.phase)
            if ab.phase == Phase.COOLDOWN:
                self.assertFalse(ab.trigger(ctx), "must not trigger during cooldown")
        self.assertEqual(ab.phase, Phase.IDLE)
        self.assertEqual(ab.entered, [Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT, Phase.COOLDOWN, Phase.IDLE])
        self.assertTrue(ab.trigger(ctx))

    def test_zero_duration_phases_skipped_but_hooked(self):
        ctx = make_ctx()
        ab = Recorder({"charge": 0, "active": 0.1, "impact": 0, "cooldown": 0.2})
        ab.trigger(ctx)
        self.assertEqual(ab.phase, Phase.ACTIVE)
        ctx.dt = 0.15
        ab.update(ctx)
        self.assertEqual(ab.phase, Phase.COOLDOWN)
        self.assertIn(Phase.IMPACT, ab.entered)
        self.assertIn(Phase.CHARGING, ab.entered)

    def test_lock_blocks_other_abilities(self):
        ctx = make_ctx()
        ctx.locked_by = "bankai"
        self.assertFalse(Recorder({"active": 1}).trigger(ctx))

    def test_cooldown_left(self):
        ctx = make_ctx()
        ab = Recorder({"cooldown": 2.0})
        ab.trigger(ctx)
        self.assertEqual(ab.phase, Phase.COOLDOWN)
        ctx.dt = 0.5
        ab.update(ctx)
        self.assertAlmostEqual(ab.cooldown_left(), 1.5, places=5)


class TestGestures(unittest.TestCase):
    def test_classify_synthetic_hands(self):
        for g in ("fist", "open_palm", "two_fingers", "point"):
            for angle in (-150, -100, -60, 0, 90):
                st = make_tracking(gesture=g, hand_angle=angle)
                self.assertEqual(classify_hand(st.hands["right"].points), g, f"{g} at {angle} deg")

    def _cfg(self, **kw):
        g = dict(Config().data["gestures"])
        g.update(kw)
        return g

    def test_hold_debounce_fires_once(self):
        rec = GestureRecognizer(self._cfg(stable_frames=3, hold_seconds=0.5, global_cooldown=0.0))
        st = make_tracking(gesture="two_fingers")
        fired = []
        for i in range(60):  # 2 seconds at 30 fps, pose held the whole time
            fired += rec.update(st, 0.0, now=i / 30).triggered
        self.assertEqual(fired, ["two_fingers"])
        self.assertEqual(rec.ability_for("two_fingers"), "shikai")

    def test_short_hold_does_not_fire(self):
        rec = GestureRecognizer(self._cfg(stable_frames=3, hold_seconds=1.0))
        st = make_tracking(gesture="open_palm")
        fired = []
        for i in range(20):  # 0.66 s only
            fired += rec.update(st, 0.0, now=i / 30).triggered
        self.assertEqual(fired, [])

    def test_release_and_rehold_fires_again(self):
        rec = GestureRecognizer(self._cfg(stable_frames=2, hold_seconds=0.2, global_cooldown=0.0))
        palm, fist = make_tracking(gesture="open_palm"), make_tracking(gesture="fist")
        fired, t = [], 0.0
        for st, n in ((palm, 15), (fist, 5), (palm, 15)):
            for _ in range(n):
                fired += rec.update(st, 0.0, now=t).triggered
                t += 1 / 30
        self.assertEqual(fired, ["open_palm", "open_palm"])

    def test_swing_respects_global_cooldown(self):
        rec = GestureRecognizer(self._cfg(swing_speed=2.0, global_cooldown=0.8))
        st = TrackingState(width=960, height=540)
        fired = []
        for i in range(30):  # fast swinging for 1 s
            got = rec.update(st, 5.0, now=i / 30).triggered
            if got:
                rec.confirm(i / 30)  # the app confirms when the ability accepted it
            fired += got
        self.assertEqual(fired.count("swing"), 2)

    def test_rejected_trigger_does_not_start_cooldown(self):
        rec = GestureRecognizer(self._cfg(swing_speed=2.0, global_cooldown=0.8))
        st = TrackingState(width=960, height=540)
        fired = [rec.update(st, 5.0, now=i / 30).triggered for i in range(3)]  # never confirmed
        self.assertTrue(all(f == ["swing"] for f in fired))

    def test_hold_survives_one_dropped_frame(self):
        rec = GestureRecognizer(self._cfg(stable_frames=2, hold_seconds=0.3, global_cooldown=0.0))
        st = make_tracking(gesture="two_fingers")
        fired = []
        for i in range(20):
            part = st.hands["right"]
            part.visible = i != 5  # MediaPipe misses frame 5; points are held
            fired += rec.update(st, 0.0, now=i / 30).triggered
        self.assertEqual(fired, ["two_fingers"])

    def test_both_palms_up(self):
        rec = GestureRecognizer(self._cfg(stable_frames=2, hold_seconds=0.2, global_cooldown=0.0))
        st = make_tracking(gesture="open_palm", second_hand="open_palm")
        # Raise both wrists above the shoulders.
        for part in st.hands.values():
            part.points[:, 1] -= 400
        fired = []
        for i in range(20):
            fired += rec.update(st, 0.0, now=i / 30).triggered
        self.assertIn("both_palms_up", fired)
        self.assertNotIn("open_palm", fired)

    def test_disabled_never_triggers(self):
        rec = GestureRecognizer(self._cfg(enabled=False))
        self.assertEqual(rec.update(TrackingState(), 10.0, now=1.0).triggered, [])

    def test_no_hands_no_crash(self):
        rec = GestureRecognizer(self._cfg())
        gf = rec.update(TrackingState(width=640, height=480), 0.0, now=0.0)
        self.assertEqual(gf.per_hand, {})


class TestTracking(unittest.TestCase):
    def test_non_finite_landmarks_are_dropped(self):
        from bleach_ar.tracking import Part, Tracker
        t = Tracker.__new__(Tracker)  # no MediaPipe graph needed for _update_part
        t.smoothing, t.hold_seconds = 0.45, 0.35
        part = Part()
        good = np.ones((21, 3), np.float32)
        t._update_part(part, good, now=0.0)
        bad = good.copy()
        bad[0, 0] = np.inf
        t._update_part(part, bad, now=0.01)
        self.assertFalse(part.visible)
        self.assertTrue(np.isfinite(part.points).all())


class TestCamera(unittest.TestCase):
    def test_synthetic_source(self):
        cam = Camera("synthetic", 320, 180).start()
        try:
            fid, frame = cam.read()
            self.assertEqual(frame.shape, (180, 320, 3))
        finally:
            cam.release()

    def test_missing_video_file(self):
        with self.assertRaises(CameraError):
            Camera("/nonexistent/video.mp4")


class TestFx(unittest.TestCase):
    def test_particles_ring_buffer_and_render(self):
        ps = ParticleSystem(50)
        ps.emit(80, (100, 100), (0, -50), spread_vel=20)
        self.assertEqual(ps.alive_count, 50)
        c = Canvas(0.33)
        frame = np.zeros((270, 480, 3), np.uint8)
        c.begin(frame)
        ps.update(1 / 30)
        ps.render(c)
        out = c.composite()
        self.assertEqual(out.shape, (270, 480, 3))
        self.assertGreater(int(out.max()), 0)

    def test_emit_clamps_per_particle_arrays(self):
        """n > capacity with (n, 2) pos/vel must truncate, not ValueError on broadcast."""
        cap = 20
        n = 50
        ps = ParticleSystem(cap)
        pos = np.linspace(0, 100, n * 2, dtype=np.float32).reshape(n, 2)
        vel = np.ones((n, 2), np.float32) * (10.0, -5.0)
        ps.emit(n, pos, vel)
        self.assertEqual(ps.alive_count, cap)
        np.testing.assert_allclose(ps.pos[:cap], pos[:cap])
        np.testing.assert_allclose(ps.vel[:cap], vel[:cap])


if __name__ == "__main__":
    unittest.main()
