"""Entry point: ``python -m bleach_ar [options]``."""

from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m bleach_ar", description="Bleach-inspired webcam AR combat.")
    p.add_argument("--config", help="JSON config file (default: ./config.json if present)")
    p.add_argument("--camera", type=int, help="camera index (default from config: 0)")
    p.add_argument("--source", help="video file path, or 'synthetic' for a test pattern")
    p.add_argument("--quality", choices=["low", "medium", "high"], help="quality preset")
    p.add_argument("--width", type=int, help="camera capture width")
    p.add_argument("--height", type=int, help="camera capture height")
    p.add_argument("--fps", type=int, help="camera frame rate")
    p.add_argument("--demo", action="store_true",
                   help="no camera: synthetic person with a scripted ability showcase")
    p.add_argument("--frames", type=int, default=0, help="exit after N frames and print timings")
    p.add_argument("--headless", action="store_true", help="no window (for automated tests)")
    p.add_argument("--write-config", metavar="PATH", help="write the full default config to PATH and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from .config import Config

    if args.write_config:
        Config().save(args.write_config)
        print(f"Wrote default config to {args.write_config}")
        return 0

    cfg = Config.load(args.config)
    if cfg.source:
        print(f"Loaded config: {cfg.source}")
    if args.quality:
        cfg.data["quality"] = args.quality
    for key in ("width", "height", "fps"):
        if getattr(args, key):
            cfg.data["camera"][key] = getattr(args, key)
    source = args.source if args.source else args.camera

    from .app import BleachAR

    app = BleachAR(cfg, source=source, demo=args.demo, max_frames=args.frames)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
