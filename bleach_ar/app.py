"""Main loop: camera -> tracking -> gestures -> abilities -> compositing -> HUD."""

from __future__ import annotations

import math
import os
import statistics
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .abilities import AbilityManager, Context
from .camera import Camera, CameraError, resolve_camera
from .config import Config
from .fx import Canvas, ParticleSystem, Shaker
from .gestures import GestureRecognizer
from .synthetic import make_frame
from .tracking import AsyncTracker

EYE_STYLES = ["hollow", "red", "blue", "white"]
WEAPON_STYLES = ["katana", "cleaver", "dark"]
BANKAI_STYLES = ["senbonzakura", "tensa"]
BANKAI_STYLE_LABELS = {"senbonzakura": "BANKAI: SENBONZAKURA", "tensa": "BANKAI: TENSA ZANGETSU"}


def cycle_bankai_style(ctx: Context, manager: AbilityManager) -> bool:
    """Switch the Bankai style; refused (returns False) during or while in Bankai."""
    from .abilities import BANKAI, Phase

    if ctx.level == BANKAI or manager.abilities["bankai"].phase != Phase.IDLE:
        ctx.show_title("RELEASE BANKAI FIRST", "", (200, 200, 210), 0.9)
        return False
    cur = ctx.bankai_style if ctx.bankai_style in BANKAI_STYLES else BANKAI_STYLES[0]
    ctx.bankai_style = BANKAI_STYLES[(BANKAI_STYLES.index(cur) + 1) % len(BANKAI_STYLES)]
    ctx.cfg["abilities"]["bankai"]["style"] = ctx.bankai_style  # "S" in calibration saves it
    color = ctx.cfg["abilities"]["senbonzakura" if ctx.bankai_style == "senbonzakura" else "bankai"]["color"]
    ctx.show_title(BANKAI_STYLE_LABELS[ctx.bankai_style], "", tuple(color), 1.2)
    return True


class Stats:
    def __init__(self, window: int = 90):
        self.frame_times = deque(maxlen=window)
        self.track_ms = deque(maxlen=window)
        self.fx_ms = deque(maxlen=window)
        self.all_frame_ms = []
        self.all_track_ms = []
        self.all_fx_ms = []
        self.cam_times = deque(maxlen=window)  # arrival times of new camera frames
        self.cam_count = 0
        self.t_start = None
        self.t_last = None

    def camera_frame(self, now: float) -> None:
        self.cam_times.append(now)
        self.cam_count += 1
        if self.t_start is None:
            self.t_start = now
        self.t_last = now

    def add(self, frame_s: float, track_ms: Optional[float], fx_ms: float) -> None:
        self.frame_times.append(frame_s)
        self.all_frame_ms.append(frame_s * 1000)
        self.fx_ms.append(fx_ms)
        self.all_fx_ms.append(fx_ms)
        if track_ms is not None:
            self.track_ms.append(track_ms)
            self.all_track_ms.append(track_ms)

    def snapshot(self) -> dict:
        ft = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        return {
            "fps": 1.0 / ft if ft > 0 else 0.0,
            "track_ms": sum(self.track_ms) / len(self.track_ms) if self.track_ms else 0.0,
            "fx_ms": sum(self.fx_ms) / len(self.fx_ms) if self.fx_ms else 0.0,
            "cam_fps": ((len(self.cam_times) - 1) / (self.cam_times[-1] - self.cam_times[0])
                        if len(self.cam_times) > 1 and self.cam_times[-1] > self.cam_times[0] else 0.0),
        }

    def summary(self, skip: int = 15) -> str:
        def describe(xs):
            xs = xs[skip:] if len(xs) > skip * 2 else xs
            if not xs:
                return "n/a"
            s = sorted(xs)
            p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
            return f"mean {statistics.mean(xs):6.1f} ms  median {statistics.median(xs):6.1f}  p95 {p95:6.1f}"

        frames = self.all_frame_ms[skip:] if len(self.all_frame_ms) > skip * 2 else self.all_frame_ms
        fps = 1000.0 / statistics.mean(frames) if frames else 0.0
        span = (self.t_last - self.t_start) if self.t_start is not None and self.t_last else 0.0
        cam_fps = (self.cam_count - 1) / span if span > 0 else 0.0
        return (f"frames: {len(self.all_frame_ms)}  avg FPS: {fps:.1f}  "
                f"camera frames shown/s: {cam_fps:.1f}\n"
                f"  frame    {describe(self.all_frame_ms)}\n"
                f"  tracking {describe(self.all_track_ms)}\n"
                f"  effects  {describe(self.all_fx_ms)}")


class DemoDirector:
    """Scripted synthetic person + ability timeline (``--demo``)."""

    TIMELINE = [(1.0, "getsuga"), (3.0, "shikai"), (5.5, "getsuga"), (7.5, "cero"),
                (10.5, "bankai"), (15.5, "getsuga"), (17.5, "bankai"), (19.0, "shikai")]
    LOOP = 21.0

    def __init__(self):
        self.t0 = time.perf_counter()
        self.fired = set()
        self.loop = 0

    def step(self, w: int, h: int):
        from .synthetic import make_tracking

        t = time.perf_counter() - self.t0
        loop, lt = divmod(t, self.LOOP)
        if loop != self.loop:
            self.loop, self.fired = loop, set()
        gesture = "open_palm" if 7.0 <= lt < 9.5 else "fist"
        swing = 0.5 + 0.5 * math.sin(lt * 2.2)
        st = make_tracking(w, h, t=lt, gesture=gesture, swing=swing)
        due = [name for at, name in self.TIMELINE if lt >= at and (at, name) not in self.fired]
        for at, name in self.TIMELINE:
            if lt >= at:
                self.fired.add((at, name))
        return st, due


class BleachAR:
    def __init__(self, cfg: Config, source=None, demo: bool = False, max_frames: int = 0):
        import pygame

        self.pg = pygame
        self.cfg = cfg
        self.source = resolve_camera(source if source is not None else cfg["camera"]["index"])
        self.demo = demo
        self.max_frames = max_frames
        self.camera: Optional[Camera] = None
        self.tracker = None
        self.debug = False
        self.running = True
        self.stats = Stats()

        pygame.init()
        pygame.display.set_caption("Bleach AR")
        win = cfg["window"]
        flags = pygame.RESIZABLE | (pygame.FULLSCREEN if win["fullscreen"] else 0)
        self.screen = pygame.display.set_mode((win["width"], win["height"]), flags)
        self.clock = pygame.time.Clock()

        from .hud import HUD
        self.hud = HUD(cfg["keys"], cfg.asset_path("font"))
        self.keymap = {}
        for action, key in cfg["keys"].items():
            try:
                self.keymap[pygame.key.key_code(key)] = action
            except ValueError:
                print(f"[config] unknown key '{key}' for '{action}', ignoring.")

        preset = cfg.preset
        eff = cfg["effects"]
        self.canvas = Canvas(preset["glow_scale"], eff["intensity"])
        self.shaker = Shaker()
        self.shaker.enabled = eff["screen_shake"]
        self.ctx = Context(cfg=cfg.data, canvas=self.canvas, particles=ParticleSystem(preset["max_particles"]),
                           shaker=self.shaker)
        self.ctx.weapon_style = eff["weapon_style"]
        self.ctx.eye_style = eff["eye_style"]
        self.ctx.sword_mode = eff["sword_mode"]
        self.ctx.show_mask = eff["show_mask"]
        style = cfg["abilities"]["bankai"].get("style", "senbonzakura")
        self.ctx.bankai_style = style if style in BANKAI_STYLES else BANKAI_STYLES[0]
        self.manager = AbilityManager(self.ctx)
        self.gestures = GestureRecognizer(cfg["gestures"])
        self.director = DemoDirector() if demo else None

    # --- setup -------------------------------------------------------------
    def _start_pipeline(self) -> Optional[str]:
        """Open camera + tracker. Returns an error message instead of raising."""
        cam_cfg = self.cfg["camera"]
        try:
            if self.camera is None:
                src = "synthetic" if self.demo else self.source
                try:
                    cam = Camera(src, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"], mirror=cam_cfg["mirror"])
                except CameraError:
                    # macOS renumbers cameras when an iPhone (Continuity Camera)
                    # connects or disconnects, so fall back to any working index.
                    if not isinstance(src, int):
                        raise
                    cam = self._open_other_camera(src)
                    if cam is None:
                        raise
                    print(f"Camera {src} unavailable; using camera {self.source} instead (press V to switch).")
                self.camera = cam.start()
            if self.tracker is None and not self.demo:
                self._make_tracker()
        except CameraError as exc:
            self.camera = None
            return str(exc)
        except Exception as exc:  # model-loading errors etc.
            return f"Failed to start tracking: {exc}"
        return None

    def _make_tracker(self) -> None:
        from .tracking import Tracker

        preset, tr = self.cfg.preset, self.cfg["tracking"]
        if self.tracker is not None:
            self.tracker.close()
        tracker = Tracker(
            model_complexity=preset["model_complexity"], segmentation=preset["segmentation"],
            tracking_width=preset["tracking_width"],
            min_detection_confidence=tr["min_detection_confidence"],
            min_tracking_confidence=tr["min_tracking_confidence"],
            smoothing=tr["smoothing"], hold_seconds=tr["hold_seconds"])
        self.tracker = AsyncTracker(tracker) if tr["async"] else tracker

    def startup_screen(self) -> bool:
        pg = self.pg
        error, t0 = "", time.perf_counter()
        auto = self.max_frames > 0  # headless test runs skip the prompt
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return False
                    if event.key in (pg.K_SPACE, pg.K_RETURN, pg.K_KP_ENTER):
                        auto = True
            if auto:
                self.hud.draw_startup(self.screen, time.perf_counter() - t0, "Opening camera and loading models...")
                pg.display.flip()
                error = self._start_pipeline() or ""
                if not error:
                    return True
                print(error)
                auto = False
                if self.max_frames:
                    return False
            self.hud.draw_startup(self.screen, time.perf_counter() - t0, error=error)
            pg.display.flip()
            self.clock.tick(30)

    # --- input -------------------------------------------------------------
    def _handle_key(self, event) -> None:
        pg, ctx = self.pg, self.ctx
        if self.hud.show_calibration and self._handle_calibration_key(event):
            return
        action = self.keymap.get(event.key)
        if action is None:
            return
        if action == "exit":
            self.running = False
        elif action in ("getsuga", "shikai", "bankai", "cero"):
            self.manager.trigger(action)
        elif action == "toggle_sword":
            ctx.show_sword = not ctx.show_sword
        elif action == "toggle_aura":
            ctx.show_aura = not ctx.show_aura
        elif action == "toggle_eyes":
            ctx.show_eyes = not ctx.show_eyes
        elif action == "toggle_mask":
            ctx.show_mask = not ctx.show_mask
        elif action == "cycle_eye_style":
            ctx.eye_style = EYE_STYLES[(EYE_STYLES.index(ctx.eye_style) + 1) % len(EYE_STYLES)] \
                if ctx.eye_style in EYE_STYLES else EYE_STYLES[0]
            ctx.show_eyes = True
        elif action == "cycle_weapon":
            ctx.weapon_style = WEAPON_STYLES[(WEAPON_STYLES.index(ctx.weapon_style) + 1) % len(WEAPON_STYLES)] \
                if ctx.weapon_style in WEAPON_STYLES else WEAPON_STYLES[0]
        elif action == "cycle_bankai":
            self.cycle_bankai_style()
        elif action == "toggle_sword_mode":
            ctx.sword_mode = "hand" if ctx.sword_mode == "sword" else "sword"
        elif action == "toggle_gestures":
            self.gestures.enabled = not self.gestures.enabled
        elif action == "toggle_debug":
            self.debug = not self.debug
        elif action == "toggle_calibration":
            self.hud.show_calibration = not self.hud.show_calibration
        elif action == "cycle_quality":
            self._apply_quality(self.cfg.cycle_quality())
        elif action == "reset":
            self.manager.reset()
            self.gestures.reset()
        elif action == "help":
            self.hud.show_help = not self.hud.show_help
        elif action == "next_camera":
            self._next_camera()
        elif action == "fullscreen":
            pg.display.toggle_fullscreen()

    def cycle_bankai_style(self) -> bool:
        """Toggle Senbonzakura / Tensa. Only while sealed/shikai and Bankai is idle."""
        return cycle_bankai_style(self.ctx, self.manager)

    def _open_other_camera(self, current: int) -> Optional[Camera]:
        """Open the next camera index after ``current`` that delivers frames."""
        cam_cfg = self.cfg["camera"]
        for step in range(1, 5):
            idx = (current + step) % 5
            if idx == current:
                continue
            try:
                cam = Camera(idx, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"], mirror=cam_cfg["mirror"])
            except CameraError:
                continue
            self.source = idx
            return cam
        return None

    def _next_camera(self) -> None:
        """Switch to the next working camera (e.g. built-in webcam vs iPhone)."""
        if self.demo or not isinstance(self.source, int):
            return
        if self.camera is not None:
            self.camera.release()  # macOS may refuse a second open of the same device
            self.camera = None
        previous = self.source
        cam = self._open_other_camera(previous)
        if cam is None:
            self._start_pipeline()  # reopen the original camera
            self.ctx.show_title("NO OTHER CAMERA", "", (180, 180, 255), 1.2)
            return
        self.camera = cam.start()
        self.ctx.show_title(f"CAMERA {self.source}", "press V again to switch", (255, 255, 255), 1.2)

    def _handle_calibration_key(self, event) -> bool:
        pg, g = self.pg, self.cfg["gestures"]
        steps = {pg.K_LEFTBRACKET: ("swing_speed", -0.2), pg.K_RIGHTBRACKET: ("swing_speed", 0.2),
                 pg.K_MINUS: ("hold_seconds", -0.1), pg.K_EQUALS: ("hold_seconds", 0.1),
                 pg.K_COMMA: ("finger_extend_ratio", -0.02), pg.K_PERIOD: ("finger_extend_ratio", 0.02)}
        if event.key in steps:
            key, step = steps[event.key]
            g[key] = round(max(0.05, g[key] + step), 2)
            return True
        if event.key == pg.K_s:
            from .config import DEFAULT_CONFIG_PATH
            path = self.cfg.source or DEFAULT_CONFIG_PATH
            self.cfg.save(path)
            self.ctx.show_title("SAVED", os.path.basename(path), (180, 255, 180), 1.0)
            return True
        return False

    def _apply_quality(self, name: str) -> None:
        preset = self.cfg.preset
        self.canvas.glow_scale = preset["glow_scale"]
        self.ctx.particles.resize(preset["max_particles"])
        if self.tracker is not None:
            self._make_tracker()
        self.ctx.show_title(name.upper(), "quality", (255, 255, 255), 0.8)

    # --- frame -------------------------------------------------------------
    def _render_frame(self, raw: np.ndarray) -> np.ndarray:
        rw = self.cfg.preset["render_width"]
        h, w = raw.shape[:2]
        if w != rw:
            return cv2.resize(raw, (rw, int(round(h * rw / w))), interpolation=cv2.INTER_AREA)
        return raw.copy()

    def _draw_debug(self, frame: np.ndarray) -> None:
        tr = self.ctx.tracking
        if tr.face.ok:
            for p in tr.face.points[::6, :2].astype(int):
                cv2.circle(frame, tuple(p), 1, (0, 255, 0), -1)
        if tr.pose.ok:
            for p in tr.pose.points[:25, :2].astype(int):
                cv2.circle(frame, tuple(p), 4, (0, 200, 255), -1)
        for part in tr.hands.values():
            if part.ok:
                for p in part.points[:, :2].astype(int):
                    cv2.circle(frame, tuple(p), 3, (255, 120, 0), -1)

    def run(self) -> None:
        pg = self.pg
        try:
            if not self.startup_screen():
                return
            last = time.perf_counter()
            last_id = -1
            frame = None
            frames = 0
            reconnect_at = 0.0
            while self.running:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.running = False
                    elif event.type == pg.KEYDOWN:
                        self._handle_key(event)

                now = time.perf_counter()
                dt = min(0.1, now - last)
                last = now

                camera_lost = self.camera is None or self.camera.lost
                if camera_lost and now >= reconnect_at and not self.demo:
                    reconnect_at = now + 2.0
                    if self.camera is not None:
                        self.camera.release()
                        self.camera = None
                    self._start_pipeline()

                track_ms = None
                if self.camera is not None:
                    frame_id, raw = self.camera.read()
                    if frame_id != last_id and raw is not None:
                        last_id = frame_id
                        self.stats.camera_frame(now)
                        frame = self._render_frame(raw)
                        if self.director is not None:
                            st, due = self.director.step(frame.shape[1], frame.shape[0])
                            self.ctx.tracking = st
                            frame = self._demo_frame = make_frame(st)  # draw the synthetic figure
                            for name in due:
                                self.manager.trigger(name)
                        elif isinstance(self.tracker, AsyncTracker):
                            self.tracker.submit(frame.copy())  # the canvas draws into `frame`
                        elif self.tracker is not None:
                            self.ctx.tracking = self.tracker.process(frame)
                            track_ms = self.ctx.tracking.inference_ms
                    elif frame is not None:
                        frame = self._demo_frame.copy() if self.director is not None else self._render_frame(raw)
                if frame is None:
                    frame = np.zeros((540, 960, 3), np.uint8)
                if isinstance(self.tracker, AsyncTracker):
                    self.ctx.tracking = self.tracker.state
                    track_ms = self.tracker.take_timing()

                t_fx = time.perf_counter()
                ctx = self.ctx
                ctx.now, ctx.dt = now, dt
                self.manager.update()
                gf = self.gestures.update(ctx.tracking, ctx.sword.speed if ctx.sword.visible else 0.0, now)
                for g in gf.triggered:
                    ability = self.gestures.ability_for(g)
                    if ability and self.manager.trigger(ability):
                        self.gestures.confirm(now)
                        break

                self.canvas.begin(frame)
                if self.debug:
                    self._draw_debug(frame)
                self.manager.render()
                out = self.canvas.composite()
                self.canvas.flash *= math.exp(-6.0 * dt)
                fx_ms = (time.perf_counter() - t_fx) * 1000

                self._present(out, gf, camera_lost)
                self.stats.add(dt, track_ms, fx_ms)
                frames += 1
                if self.max_frames and frames >= self.max_frames:
                    break
                # Effects animate between camera frames, but rendering faster
                # than 60 Hz only burns CPU.
                self.clock.tick(60)
        finally:
            self.close()

    def _present(self, frame: np.ndarray, gf, camera_lost: bool) -> None:
        pg = self.pg
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pg.image.frombuffer(rgb.data, (w, h), "RGB")
        W, H = self.screen.get_size()
        s = min(W / w, H / h)
        size = (int(w * s), int(h * s))
        if size != (w, h):
            surf = pg.transform.scale(surf, size)
        ox, oy = self.shaker.update(self.ctx.dt)
        self.screen.fill((0, 0, 0))
        self.screen.blit(surf, ((W - size[0]) // 2 + ox, (H - size[1]) // 2 + oy))
        stats = self.stats.snapshot()
        stats["camera_lost"] = camera_lost and not self.demo
        g = self.cfg["gestures"]
        self.hud.draw(self.screen, self.ctx, self.manager.abilities, stats, gf, self.gestures.enabled,
                      self.cfg["quality"], g)
        pg.display.flip()

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        self.pg.quit()
        if self.stats.all_frame_ms:
            print("\nPerformance this session (" + self.cfg["quality"] + " quality):")
            print(self.stats.summary())
