"""Tests for the rendering core (bleach_ar/fx.py): composite, post effects,
quality flags, buffer reuse and the reusable graphics helpers."""

import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bleach_ar import fx  # noqa: E402
from bleach_ar.config import DEFAULTS  # noqa: E402
from bleach_ar.fx import Canvas  # noqa: E402

RESOLUTIONS = [(640, 360), (960, 540), (1280, 720)]


def smooth_frame(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 255, (h // 24 + 2, w // 24 + 2, 3), dtype=np.uint8), (w, h),
                      interpolation=cv2.INTER_CUBIC)


# --- reference implementations of the pre-optimisation code (for equivalence) --
def ref_warp(frame, warps):
    """Original full-frame warp: numpy on a w/4 grid, two resizes, remap."""
    h, w = frame.shape[:2]
    gw, gh = max(16, w // 4), max(16, h // 4)
    ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
    sx, sy = w / gw, h / gh
    X, Y = xs * sx + sx * 0.5, ys * sy + sy * 0.5
    FX, FY = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = np.zeros_like(X), np.zeros_like(Y)
    for cx, cy, rad, st, kind, width in warps:
        ox, oy = X - cx, Y - cy
        r = np.sqrt(ox * ox + oy * oy) + 1e-3
        if kind == "ripple":
            band = max(4.0, rad * width)
            dr = st * band * np.exp(-((r - rad) / band) ** 2)
            dx += ox / r * dr
            dy += oy / r * dr
            continue
        f = np.clip(1.0 - (r / rad) ** 2, 0.0, 1.0) ** 2
        if kind == "swirl":
            a = st * f
            dx += ox * np.cos(a) - oy * np.sin(a) - ox
            dy += ox * np.sin(a) + oy * np.cos(a) - oy
        else:
            k = st * f * (1.0 if kind == "pinch" else -1.0)
            dx += ox * k
            dy += oy * k
    mx = FX + cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR)
    my = FY + cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def ref_bloom(frame, light, glow, gs, w, h, gw, gh, bloom=1.0, bw=1.4, bt=0.45):
    out = cv2.add(frame, light)
    small = cv2.resize(light, (gw, gh), interpolation=cv2.INTER_AREA)
    g = glow + 0.6 * bloom * small.astype(np.float32)
    kw = max(3, int(27 * gs * 2.449) | 1)
    kt = max(3, int(8 * gs * 2.449) | 1)
    wide = cv2.blur(cv2.blur(g, (kw, kw)), (kw, kw))
    tight = cv2.blur(g, (kt, kt))
    wide = cv2.addWeighted(wide, bw * bloom, tight, bt * bloom, 0)
    wide = np.minimum(wide, 170.0 + 85.0 * (1.0 - np.exp(-(np.maximum(wide, 170.0) - 170.0) / 85.0)))
    return cv2.add(out, cv2.resize(wide.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR))


def draw_content(c):
    w, h = c.w, c.h
    col = (255, 200, 90)
    pts = fx.crescent((w * 0.5, h * 0.5), (1, -0.3), h * 0.3, h * 0.08)
    c.light_poly(pts, col, 0.9)
    c.glow_poly(pts, col, 2.0)  # bright enough to hit the bloom knee
    c.light_line((w * 0.1, h * 0.2), (w * 0.8, h * 0.7), (255, 255, 255), 3)
    c.add_sprite((w * 0.3, h * 0.4), h * 0.1, col, 1.0)
    c.glow_circle((w * 0.7, h * 0.3), h * 0.1, col, 1.5)


def person_left_half(w, h):
    m = np.zeros((144, 256), np.float32)
    m[:, :128] = 1.0
    return m


class TestComposite(unittest.TestCase):
    def test_composite_three_resolutions(self):
        for w, h in RESOLUTIONS:
            with self.subTest(res=(w, h)):
                c = Canvas(0.33)
                for i in range(3):
                    frame = smooth_frame(w, h, i)
                    c.begin(frame)
                    draw_content(c)
                    c.set_person_src(person_left_half(w, h))
                    c.backdrop()[:] = (60, 0, 80)
                    c.backdrop_amount = 0.7
                    c.add_warp((w * 0.6, h * 0.5), h * 0.3, 0.5, "pinch")
                    c.add_warp((w * 0.3, h * 0.5), h * 0.2, 0.8, "ripple")
                    c.aberration = 5.0
                    c.tint, c.tint_amount = (100, 30, 140), 0.3
                    c.flash = 0.2
                    out = c.composite()
                    self.assertIs(out, frame)
                    self.assertEqual(out.shape, (h, w, 3))
                    self.assertEqual(out.dtype, np.uint8)
                    self.assertTrue(np.isfinite(c.glow).all())
                    pm = c.person_mask()
                    self.assertTrue(np.isfinite(pm).all())
                    self.assertGreaterEqual(float(pm.min()), 0.0)
                    self.assertLessEqual(float(pm.max()), 1.0)

    def test_bloom_matches_reference(self):
        for w, h in RESOLUTIONS:
            with self.subTest(res=(w, h)):
                c = Canvas(0.33)
                frame = smooth_frame(w, h)
                c.begin(frame.copy())
                draw_content(c)
                light, glow = c.light.copy(), c.glow.copy()
                ref = ref_bloom(frame, light, glow, c.gs, w, h, c.gw, c.gh)
                out = c.composite()
                d = np.abs(out.astype(int) - ref.astype(int))
                # 960x540 is bit-exact; other sizes crop <= 2 px of the light
                # layer to keep INTER_AREA on its integer fast path.
                self.assertLessEqual(int(d.max()), 6)
                self.assertLess(float(d.mean()), 0.2)
                if (w, h) == (960, 540):
                    self.assertEqual(int(d.max()), 0)

    def test_empty_layers_leave_frame_untouched(self):
        c = Canvas(0.33)
        frame = smooth_frame(640, 360)
        ref = frame.copy()
        c.begin(frame)
        np.testing.assert_array_equal(c.composite(), ref)

    def test_backdrop_only_outside_person(self):
        w, h = 960, 540
        c = Canvas(0.33)
        for _ in range(4):  # let the mask's temporal smoothing settle
            frame = np.full((h, w, 3), 100, np.uint8)
            c.begin(frame)
            c.set_person_src(person_left_half(w, h))
            c.backdrop()[:] = (0, 0, 250)
            c.backdrop_amount = 1.0
            out = c.composite()
        left = out[:, : w // 2 - 40]
        right = out[:, w // 2 + 40:]
        self.assertTrue(np.all(left == 100), "person area must keep the camera image")
        self.assertTrue(np.all(right == (0, 0, 250)), "background must be replaced by the backdrop")

    def test_backdrop_without_mask_blends_everywhere(self):
        c = Canvas(0.33)
        frame = np.full((360, 640, 3), 100, np.uint8)
        c.begin(frame)
        c.backdrop()[:] = 200
        c.backdrop_amount = 1.0
        out = c.composite()
        expect = round(100 * (1 - c.backdrop_no_mask) + 200 * c.backdrop_no_mask)
        self.assertTrue(np.all(np.abs(out.astype(int) - expect) <= 1))

    def test_tint_matches_reference(self):
        frame = smooth_frame(640, 360)
        ref = cv2.addWeighted(frame, 0.65, np.full_like(frame, (100, 30, 140)), 0.35, 0)
        c = Canvas(0.33)
        c.begin(frame.copy())
        c.tint, c.tint_amount = (100, 30, 140), 0.35
        out = c.composite()
        self.assertLessEqual(int(np.abs(out.astype(int) - ref).max()), 1)


def blob_centroid_dist(img, center):
    g = img[:, :, 1].astype(np.float32)
    ys, xs = np.nonzero(g > 128)
    return float(np.hypot(xs.mean() - center[0], ys.mean() - center[1]))


class TestWarps(unittest.TestCase):
    W, H = 640, 360
    C = (320.0, 180.0)

    def frame_with_dot(self, offset=60):
        f = np.zeros((self.H, self.W, 3), np.uint8)
        cv2.circle(f, (int(self.C[0] + offset), int(self.C[1])), 8, (0, 255, 0), -1)
        return f

    def warped(self, kind, strength, radius=150.0, frame=None, **kw):
        c = Canvas(0.33)
        frame = self.frame_with_dot() if frame is None else frame
        c.begin(frame)
        c.add_warp(self.C, radius, strength, kind, **kw)
        return c.composite()

    def test_pinch_pulls_toward_centre(self):
        before = blob_centroid_dist(self.frame_with_dot(), self.C)
        after = blob_centroid_dist(self.warped("pinch", 0.6), self.C)
        self.assertLess(after, before - 5)

    def test_bulge_pushes_out(self):
        before = blob_centroid_dist(self.frame_with_dot(), self.C)
        after = blob_centroid_dist(self.warped("bulge", 0.4), self.C)
        self.assertGreater(after, before + 5)

    def test_swirl_rotates(self):
        out = self.warped("swirl", 0.8)
        ys, xs = np.nonzero(out[:, :, 1] > 128)
        self.assertGreater(abs(ys.mean() - self.C[1]), 5)  # dot moved off the horizontal axis

    def test_ripple_is_localised_to_its_ring(self):
        frame = smooth_frame(self.W, self.H)
        out = self.warped("ripple", 1.0, radius=100.0, frame=frame.copy(), width=0.15)
        d = np.abs(out.astype(int) - frame.astype(int)).sum(axis=2)
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        r = np.hypot(xx - self.C[0], yy - self.C[1])
        self.assertEqual(int(d[r < 40].max()), 0)    # centre untouched
        self.assertEqual(int(d[r > 180].max()), 0)   # outside untouched
        self.assertGreater(float(d[(r > 85) & (r < 115)].mean()), 1.0)  # the ring moves

    def test_matches_reference_warp(self):
        w, h = 960, 540
        frame = smooth_frame(w, h, 3)
        warps = [(300.0, 270.0, 160.0, 0.5, "pinch", 0.25), (600.0, 250.0, 120.0, 0.4, "bulge", 0.25),
                 (480.0, 300.0, 140.0, 0.9, "swirl", 0.25), (700.0, 300.0, 100.0, 0.8, "ripple", 0.2)]
        ref = ref_warp(frame, warps)
        c = Canvas(0.33)
        c.begin(frame.copy())
        for cx, cy, rad, st, kind, width in warps:
            c.add_warp((cx, cy), rad, st, kind, width)
        out = c.composite()
        d = np.abs(out.astype(int) - ref.astype(int))
        self.assertLess(float(d.mean()), 0.3)
        self.assertLess(float(np.percentile(d, 99.9)), 8)

    def test_person_protected_from_warp(self):
        w, h = 640, 360
        c = Canvas(0.33)
        for _ in range(4):
            frame = smooth_frame(w, h)
            ref = frame.copy()
            c.begin(frame)
            c.set_person_src(person_left_half(w, h))
            c.add_warp((w * 0.5, h * 0.5), 200, 0.7, "swirl")
            out = c.composite()
        d = np.abs(out.astype(int) - ref.astype(int)).sum(axis=2)
        self.assertEqual(int(d[:, : w // 2 - 30].max()), 0)
        self.assertGreater(int(d[:, w // 2 + 30:].max()), 0)

    def test_degenerate_warps_are_ignored(self):
        c = Canvas(0.33)
        frame = smooth_frame(320, 180)
        ref = frame.copy()
        c.begin(frame)
        c.add_warp((np.nan, 5), 50, 0.5)
        c.add_warp((10, 10), 0.5, 0.5)
        c.add_warp((10, 10), 50, 0.0)
        c.add_warp((-5000, -5000), 50, 0.5)  # entirely off screen
        np.testing.assert_array_equal(c.composite(), ref)


class TestAberrationAndQuality(unittest.TestCase):
    def base(self, w=640, h=360, setup=None, quality=None):
        c = Canvas(0.33)
        c.quality = quality
        frame = smooth_frame(w, h)
        c.begin(frame)
        draw_content(c)
        if setup:
            setup(c)
        return c.composite()

    def test_aberration_off_at_zero(self):
        a = self.base()
        b = self.base(setup=lambda c: setattr(c, "aberration", 0.0))
        np.testing.assert_array_equal(a, b)
        d = self.base(setup=lambda c: setattr(c, "aberration", 6.0))
        self.assertGreater(int(np.abs(d.astype(int) - a).max()), 0)

    def test_aberration_matches_reference(self):
        w, h = 640, 360
        frame = smooth_frame(w, h)
        ref = frame.copy()
        for ch, sgn in ((0, -1.0), (2, 1.0)):
            s = 1.0 + sgn * 6.0 / (w * 0.5)
            m = np.array([[s, 0, w * 0.5 * (1 - s)], [0, s, h * 0.5 * (1 - s)]], np.float32)
            ref[:, :, ch] = cv2.warpAffine(ref[:, :, ch], m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        c = Canvas(0.33)
        c.begin(frame)
        c.aberration = 6.0
        np.testing.assert_array_equal(c.composite(), ref)

    def test_quality_flags_disable_features(self):
        plain = self.base()
        warp = lambda c: c.add_warp((320, 180), 120, 0.6, "pinch")  # noqa: E731
        ab = lambda c: setattr(c, "aberration", 6.0)  # noqa: E731
        np.testing.assert_array_equal(self.base(setup=warp, quality={"warps": False}), plain)
        np.testing.assert_array_equal(self.base(setup=ab, quality={"aberration": False}), plain)
        self.assertFalse(np.array_equal(self.base(setup=warp), plain))
        single = self.base(quality={"bloom": "single"})
        self.assertFalse(np.array_equal(single, plain))
        self.assertGreater(int(single.max()), 0)

    def test_max_warps_keeps_strongest(self):
        c = Canvas(0.33)
        c.quality = {"max_warps": 2}
        c.begin(smooth_frame(320, 180))
        c.add_warp((50, 50), 30, 0.1)
        c.add_warp((100, 50), 30, 0.9)
        c.add_warp((150, 50), 30, 0.5)
        kept = c._active_warps()
        self.assertEqual([round(wp[3], 2) for wp in kept], [0.9, 0.5])

    def test_config_presets_have_render_flags(self):
        for name, preset in DEFAULTS["quality_presets"].items():
            for key in fx.QUALITY_DEFAULTS:
                self.assertIn(key, preset, f"{name} preset missing {key}")
            c = Canvas(preset["glow_scale"])
            c.quality = preset
            frame = smooth_frame(preset["render_width"], preset["render_width"] * 9 // 16)
            c.begin(frame)
            draw_content(c)
            c.add_warp((100, 100), 80, 0.5)
            c.aberration = 4
            self.assertEqual(c.composite().shape, frame.shape)

    def test_default_quality_is_full(self):
        self.assertIsNone(Canvas().quality)
        c = Canvas()
        for k, v in fx.QUALITY_DEFAULTS.items():
            self.assertEqual(c.q(k), v)


class TestBufferReuse(unittest.TestCase):
    def test_no_reallocation_across_frames(self):
        w, h = 960, 540
        c = Canvas(0.33)
        frame = smooth_frame(w, h)

        def run():
            f = frame.copy()
            c.begin(f)
            draw_content(c)
            c.set_person_src(person_left_half(w, h))
            c.backdrop()[:] = 30
            c.backdrop_amount = 0.6
            c.add_warp((300, 200), 150, 0.5, "pinch")
            c.add_warp((600, 300), 100, 0.8, "ripple")
            c.aberration = 4.0
            c.tint, c.tint_amount = (10, 20, 30), 0.2
            c.composite()
            return {k: v.ctypes.data for k, v in c._bufs.items()} | {
                "light": c.light.ctypes.data, "glow": c.glow.ctypes.data, "backdrop": c._backdrop.ctypes.data}

        run()
        run()
        a = run()
        b = run()
        self.assertEqual(a, b)

    def test_person_mask_previous_frame_stays_valid(self):
        c = Canvas(0.33)
        c.begin(smooth_frame(320, 180))
        c.set_person_src(person_left_half(320, 180))
        m1 = c.person_mask()
        snap = m1.copy()
        c.begin(smooth_frame(320, 180))
        c.set_person_src(np.zeros((144, 256), np.float32))
        m2 = c.person_mask()
        self.assertIsNot(m1, m2)
        np.testing.assert_array_equal(m1, snap)

    def test_resolution_change_reallocates(self):
        c = Canvas(0.33)
        for w, h in RESOLUTIONS + [(320, 180)]:
            f = smooth_frame(w, h)
            c.begin(f)
            c.add_warp((w / 2, h / 2), h / 3, 0.5)
            c.aberration = 3
            self.assertEqual(c.composite().shape, (h, w, 3))
            self.assertEqual(c.light.shape, (h, w, 3))


class TestHelpers(unittest.TestCase):
    def canvas(self, w=640, h=360):
        c = Canvas(0.33)
        c.begin(np.zeros((h, w, 3), np.uint8))
        return c

    def test_depth_scale(self):
        self.assertAlmostEqual(fx.depth_scale(0, 0, 100), 0.6, places=5)
        self.assertAlmostEqual(fx.depth_scale(100, 0, 100), 1.4, places=5)
        self.assertAlmostEqual(fx.depth_scale(50, 0, 100, 1.0, 2.0), 1.5, places=5)
        self.assertAlmostEqual(fx.depth_scale(-50, 0, 100), 0.6, places=5)  # clamped
        # z cue: more negative = nearer
        self.assertGreater(fx.depth_scale(-0.2, 0.1, -0.3), fx.depth_scale(0.0, 0.1, -0.3))
        arr = fx.depth_scale(np.array([0, 50, 100]), 0, 100)
        self.assertEqual(arr.shape, (3,))
        self.assertIsInstance(fx.depth_scale(10, 5, 5), float)  # degenerate range, no crash

    def test_ribbon_polygon(self):
        poly = fx.ribbon_polygon([(0, 0), (100, 0), (200, 0)], 20)
        self.assertEqual(poly.shape, (6, 2))
        self.assertAlmostEqual(float(np.abs(poly[:, 1]).max()), 10.0, places=4)
        self.assertEqual(fx.ribbon_polygon([(5, 5)], 3).shape, (0, 2))
        self.assertEqual(fx.ribbon_polygon([(5, 5), (5, 5)], 3).shape, (0, 2))

    def test_energy_trail_tapers_and_draws(self):
        c = self.canvas()
        pts = np.stack([np.linspace(50, 590, 20), np.full(20, 180)], 1)
        fx.energy_trail(c, pts, (255, 150, 50), 30)
        self.assertGreater(int(c.light.max()), 200)
        self.assertGreater(float(c.glow.max()), 0)
        tail = c.light[150:210, 50:80].astype(int).sum()
        head = c.light[150:210, 560:590].astype(int).sum()
        self.assertGreater(head, tail * 2)
        fx.energy_trail(c, [(np.nan, 0), (1, 1)], (255, 0, 0), 10)  # ignored, no crash

    def test_shockwave_ring_and_warp(self):
        c = self.canvas()
        fx.shockwave(c, (320, 180), 100, 6, (255, 255, 255), 1.0, warp=0.8)
        self.assertGreater(int(c.light[180, 420].max()), 100)  # on the ring
        self.assertEqual(int(c.light[180, 320].max()), 0)  # centre empty
        self.assertEqual(len(c._warps), 1)
        self.assertEqual(c._warps[0][4], "ripple")
        fx.shockwave(c, (320, 180), 100, 6, (255, 255, 255))
        self.assertEqual(len(c._warps), 1)  # warp=0 adds none
        out = c.composite()
        self.assertEqual(out.shape, (360, 640, 3))

    def test_lightning(self):
        rng = np.random.default_rng(3)
        bolts = fx.lightning((10, 10), (300, 200), branches=3, rng=rng, segments=12)
        self.assertGreaterEqual(len(bolts), 4)
        main = bolts[0]
        self.assertGreaterEqual(len(main), 13)
        np.testing.assert_allclose(main[0], (10, 10), atol=1e-4)
        np.testing.assert_allclose(main[-1], (300, 200), atol=1e-4)
        for b in bolts:
            self.assertTrue(np.isfinite(b).all())
            self.assertEqual(b.dtype, np.float32)
        same = fx.lightning((10, 10), (300, 200), branches=3, rng=np.random.default_rng(3), segments=12)
        np.testing.assert_array_equal(same[0], main)  # seeded = reproducible
        self.assertEqual(len(fx.lightning((5, 5), (5, 5))), 1)
        c = self.canvas()
        fx.draw_lightning(c, bolts, (255, 200, 120))
        self.assertGreater(int(c.light.max()), 200)
        self.assertGreater(float(c.glow.max()), 0)

    def test_afterimages_fade(self):
        ai = fx.Afterimages(count=4, interval=0.03, duration=0.2)
        sq = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], np.float32)
        for i in range(10):
            ai.add(i * 0.04, sq + (i * 50, 100))
        self.assertEqual(len(ai.snaps), 4)
        ai.add(0.36 + 0.001, sq)  # within interval: ignored
        self.assertEqual(len(ai.snaps), 4)
        c = self.canvas()
        ai.render(c, 0.37, (255, 255, 255))
        older = int(c.light[120, 320, 0])  # snapshot at t=0.24
        newer = int(c.light[120, 370, 0])  # snapshot at t=0.28
        self.assertGreater(newer, older)
        self.assertEqual(int(c.light[120, 470, 0]), 0)  # newest skipped by default
        ai.render(c, 10.0, (255, 255, 255))
        self.assertEqual(len(ai.snaps), 0)  # all expired

    def test_starfield(self):
        sf = fx.Starfield(n=200, layers=3, seed=1)
        before = sf.pos.copy()
        sf.update(0.1, drift=(0.2, 0.0))
        moved = sf.pos[:, 0] - before[:, 0]
        moved = (moved + 0.5) % 1.0 - 0.5
        near, far = sf.depth > 0.8, sf.depth < 0.4
        self.assertGreater(moved[near].mean(), moved[far].mean())  # parallax
        for _ in range(50):
            sf.update(0.05, zoom=2.0)
        self.assertTrue(((sf.pos >= 0) & (sf.pos < 1)).all())
        c = self.canvas()
        sf.render(c)
        self.assertGreater(int(np.count_nonzero(c.light.max(axis=2))), 100)
        bd = c.backdrop()
        bd[:] = 0
        sf.render(c, target=bd, glow=False)
        self.assertGreater(int(bd.max()), 0)


if __name__ == "__main__":
    unittest.main()
