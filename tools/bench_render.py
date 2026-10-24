"""Micro-benchmark for ``Canvas.composite`` (the per-frame post-processing).

Usage::

    .venv/bin/python tools/bench_render.py                 # current bleach_ar/fx.py
    .venv/bin/python tools/bench_render.py --fx old_fx.py  # compare another copy
    .venv/bin/python tools/bench_render.py --res 960x540 --iters 300 --quality low
    .venv/bin/python tools/bench_render.py --ab old_fx.py  # interleaved A/B: old vs current

The A/B mode alternates the two implementations per scenario for
``--rounds`` rounds and keeps each one's best median, which keeps the
comparison fair on a loaded / thermally throttling (fanless) machine.

Each scenario draws typical content into fresh layers (not timed), then times
``Canvas.composite()`` only. Reports median / p95 in milliseconds. Results
are recorded in docs/RENDERING.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_fx(path):
    if path is None:
        sys.path.insert(0, ROOT)
        from bleach_ar import fx
        return fx
    spec = importlib.util.spec_from_file_location("fx_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_frame(w, h, seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8), (w, h),
                      interpolation=cv2.INTER_CUBIC)
    noise = rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
    return cv2.add(base, noise)


def make_mask(w, h):
    """Holistic-like segmentation mask at 256x144 (person ellipse, soft edge)."""
    m = np.zeros((144, 256), np.float32)
    cv2.ellipse(m, (128, 100), (45, 80), 0, 0, 360, 1.0, -1)
    cv2.circle(m, (128, 30), 22, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 2.0)


def draw_light(fx, c, rng):
    w, h = c.w, c.h
    col = (255, 220, 120)
    crescent = fx.crescent((w * 0.5, h * 0.5), (1, -0.2), h * 0.3, h * 0.08)
    c.light_poly(crescent, col, 0.9)
    c.glow_poly(crescent, col, 1.4)
    for i in range(6):
        a = rng.random(2) * (w, h)
        b = rng.random(2) * (w, h)
        c.light_line(a, b, (255, 255, 255), 3, 0.8)
        c.glow_line(a, b, col, 14, 0.8)
    for i in range(4):
        p = rng.random(2) * (w, h)
        c.add_sprite(p, h * 0.08, col, 1.0)
        c.glow_circle(p, h * 0.12, col, 0.6)


SCENARIOS = ["nothing", "light_glow", "backdrop_mask", "warp1", "warp3", "warp6", "aberration", "everything"]


def setup(fx, c, name, mask, rng):
    w, h = c.w, c.h
    if name in ("light_glow", "everything"):
        draw_light(fx, c, rng)
    if name in ("backdrop_mask", "warp1", "warp3", "warp6", "everything"):
        c.set_person_src(mask)
    if name in ("backdrop_mask", "everything"):
        bd = c.backdrop()
        bd[:] = (40, 10, 60)
        c.backdrop_amount = 0.85
    nw = {"warp1": 1, "warp3": 3, "warp6": 6, "everything": 6}.get(name, 0)
    kinds = ["pinch", "bulge", "swirl", "ripple", "pinch", "ripple"]
    for i in range(nw):
        c.add_warp((w * (0.2 + 0.12 * i), h * 0.5), h * 0.3, 0.5, kinds[i % len(kinds)])
    if name in ("aberration", "everything"):
        c.aberration = 6.0
    if name == "everything":
        c.flash = 0.3
        c.tint, c.tint_amount = (80, 20, 120), 0.3


def bench(fx, w, h, name, iters, quality=None, warmup=15):
    c = fx.Canvas(0.33)
    if quality is not None:
        c.quality = quality
    src = make_frame(w, h)
    frame = src.copy()
    mask = make_mask(w, h)
    rng = np.random.default_rng(1)
    times = []
    for i in range(iters + warmup):
        np.copyto(frame, src)
        c.begin(frame)
        setup(fx, c, name, mask, rng)
        t0 = time.perf_counter()
        c.composite()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    t = np.array(times)
    return float(np.median(t)), float(np.percentile(t, 95))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fx", help="path to an alternative fx.py to benchmark")
    ap.add_argument("--res", action="append", help="WxH (repeatable); default 640x360, 960x540, 1280x720")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scenario", action="append", choices=SCENARIOS)
    ap.add_argument("--quality", choices=["low", "medium", "high"],
                    help="apply a config quality preset's render flags (canvas.quality)")
    ap.add_argument("--ab", metavar="OLD_FX", help="interleaved comparison: OLD_FX vs current fx.py")
    ap.add_argument("--rounds", type=int, default=3, help="A/B rounds (best median kept)")
    ap.add_argument("--threads", type=int, default=None, help="cv2.setNumThreads (default: OpenCV's)")
    args = ap.parse_args(argv)
    if args.threads is not None:
        cv2.setNumThreads(args.threads)
    fx = load_fx(args.fx)
    quality = None
    if args.quality:
        sys.path.insert(0, ROOT)
        from bleach_ar.config import DEFAULTS
        quality = DEFAULTS["quality_presets"][args.quality]
    resolutions = [tuple(int(v) for v in r.split("x")) for r in (args.res or ["640x360", "960x540", "1280x720"])]
    scenarios = args.scenario or SCENARIOS
    print(f"fx: {fx.__file__}  cv2 {cv2.__version__} threads={cv2.getNumThreads()}  iters={args.iters}"
          + (f"  quality={args.quality}" if args.quality else ""))
    header = "scenario".ljust(15) + "".join(f"{w}x{h} med/p95".rjust(20) for w, h in resolutions)
    if args.ab:
        old = load_fx(args.ab)
        print(f"A/B: old={args.ab}")
        print("scenario".ljust(15) + "".join(f"{w}x{h} old->new".rjust(24) for w, h in resolutions))
        for name in scenarios:
            row = name.ljust(15)
            for w, h in resolutions:
                best = {0: 1e9, 1: 1e9}
                for _ in range(args.rounds):
                    for i, mod in ((0, old), (1, fx)):
                        best[i] = min(best[i], bench(mod, w, h, name, args.iters, quality)[0])
                row += f"{best[0]:6.2f} ->{best[1]:6.2f} ({best[0] / max(best[1], 1e-3):4.1f}x)".rjust(24)
            print(row, flush=True)
        return
    print(header)
    for name in scenarios:
        row = name.ljust(15)
        for w, h in resolutions:
            med, p95 = bench(fx, w, h, name, args.iters, quality)
            row += f"{med:8.2f} /{p95:7.2f}".rjust(20)
        print(row, flush=True)


if __name__ == "__main__":
    main()
