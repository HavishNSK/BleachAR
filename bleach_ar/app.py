"""Main loop: camera -> tracking -> gestures -> abilities -> compositing -> HUD."""

from __future__ import annotations

import math
import os
import statistics
import time
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np

from .abilities import AbilityManager, Context
from .camera import Camera, CameraError, resolve_camera
from .config import Config
from .fx import Canvas, ParticleSystem, Shaker
from .gestures import GestureRecognizer
from .modes import BLEACH, MODE_ORDER, MODES
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
    """Scripted synthetic person + ability timeline (``--demo``), per mode."""

    TIMELINES = {
        "bleach": ([(1.0, "getsuga"), (3.0, "shikai"), (5.5, "getsuga"), (7.5, "cero"),
                    (10.5, "bankai"), (15.5, "getsuga"), (17.5, "bankai"), (19.0, "shikai")], 21.0,
                   ((7.0, 9.5, "open_palm"),)),
        "jjk": ([(1.0, "blue"), (3.2, "red"), (5.4, "hollow_purple"), (9.0, "black_flash"),
                 (10.5, "unlimited_void"), (22.0, "malevolent_shrine")], 34.0,
                ((0.5, 9.0, "open_palm"),)),
    }

    def __init__(self, mode: str = "bleach"):
        self.t0 = time.perf_counter()
        self.fired = set()
        self.loop = 0
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        self.timeline, self.loop_len, self.palm = self.TIMELINES.get(mode, self.TIMELINES["bleach"])
        self.t0 = time.perf_counter()
        self.fired, self.loop = set(), 0

    def step(self, w: int, h: int):
        from .synthetic import make_tracking

        t = time.perf_counter() - self.t0
        loop, lt = divmod(t, self.loop_len)
        if loop != self.loop:
            self.loop, self.fired = loop, set()
        gesture = next((g for a, b, g in self.palm if a <= lt < b), "fist")
        swing = 0.5 + 0.5 * math.sin(lt * 2.2)
        st = make_tracking(w, h, t=lt, gesture=gesture, swing=swing,
                           second_hand="open_palm" if gesture == "open_palm" else None)
        due = [name for at, name in self.timeline if lt >= at and (at, name) not in self.fired]
        for at, name in self.timeline:
            if lt >= at:
                self.fired.add((at, name))
        return st, due


class BleachAR:
    # Bleach-only common actions (ignored in other modes).
    BLEACH_ONLY = {"getsuga", "shikai", "bankai", "cero", "toggle_sword", "toggle_mask", "cycle_weapon",
                   "cycle_bankai", "toggle_sword_mode"}
    TOGGLE_ACTIONS = {"toggle_sword": "sword", "toggle_aura": "aura", "toggle_eyes": "eyes", "toggle_mask": "mask"}

    def __init__(self, cfg: Config, source=None, demo: bool = False, max_frames: int = 0,
                 mode: Optional[str] = None):
        import pygame

        self.pg = pygame
        self.cfg = cfg
        # "auto" is resolved when the camera is opened (see _start_pipeline):
        # macOS only lists an iPhone Continuity Camera once the app has a window.
        self.camera_pref = source if source is not None else cfg["camera"]["index"]
        self.source = self.camera_pref if self.camera_pref != "auto" else 0
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
        self.canvas.quality = preset  # bloom / warps / aberration limits per quality preset
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
        from .menus import Menus
        self.menus = Menus(self.hud.fonts)
        start = mode or cfg.data.get("modes", {}).get("start") or ""
        self.start_mode = start if start in MODES else None
        self.director = DemoDirector(self.start_mode or BLEACH) if demo else None
        self._last_surface = None  # last camera frame as a pygame Surface (menu backdrop)
        self._trans_p = 1.0  # menus.draw_transition progress; 0.5 = fully covered, 1 = done
        self.set_mode(self.start_mode or BLEACH)

    # --- setup -------------------------------------------------------------
    def _start_pipeline(self) -> Optional[str]:
        """Open camera + tracker. Returns an error message instead of raising."""
        cam_cfg = self.cfg["camera"]
        try:
            if self.camera is None:
                if self.camera_pref == "auto" and not self.demo:
                    self.source = resolve_camera("auto")
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
                if self.camera_pref == "auto" and not self.demo:
                    cam = self._recheck_builtin(cam)
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

    # --- modes ---------------------------------------------------------------
    def mode_keys(self, mode_id: str) -> Dict[str, str]:
        """ability id -> key name for a mode."""
        if mode_id == BLEACH:
            return {a.id: self.cfg["keys"].get(a.key_action, "") for a in MODES[BLEACH].abilities}
        keys = self.cfg.data.get("modes", {}).get(mode_id, {}).get("keys", {})
        return {a.id: keys.get(a.key_action, "") for a in MODES[mode_id].abilities}

    def mode_gestures(self, mode_id: str) -> Dict[str, str]:
        if mode_id == BLEACH:
            return dict(self.cfg["gestures"]["bindings"])
        return dict(self.cfg.data.get("modes", {}).get(mode_id, {}).get("gestures", {}))

    def set_mode(self, mode_id: str) -> None:
        """Switch universe: abilities, keys and gestures; camera/tracking are kept."""
        pg = self.pg
        self.manager.set_mode(mode_id)
        self.ability_keymap = {}
        for aid, key in self.mode_keys(mode_id).items():
            try:
                self.ability_keymap[pg.key.key_code(key)] = aid
            except ValueError:
                print(f"[config] unknown key '{key}' for '{aid}', ignoring.")
        self.gestures.cfg = dict(self.cfg["gestures"], bindings=self.mode_gestures(mode_id))
        self.gestures.reset()
        if self.director is not None:
            self.director.set_mode(mode_id)

    def _menu_loop(self, in_game: bool) -> bool:
        """Mode selector, then that mode's ability panel. Returns False to quit."""
        pg = self.pg
        t0 = time.perf_counter()
        screen = "modes"
        sel_mode = MODE_ORDER.index(self.ctx.mode) if self.ctx.mode in MODE_ORDER else 0
        sel_ab = 0
        error, message = "", ""
        while True:
            t = time.perf_counter() - t0
            size = self.screen.get_size()
            mode_id = MODE_ORDER[sel_mode]
            go = None
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return False
                if event.type == pg.MOUSEMOTION or event.type == pg.MOUSEBUTTONDOWN:
                    rects = self.menus.mode_card_rects(size) if screen == "modes" else \
                        self.menus.ability_row_rects(size, mode_id)
                    for i, r in enumerate(rects):
                        if r.collidepoint(event.pos):
                            if screen == "modes":
                                sel_mode = i
                                if event.type == pg.MOUSEBUTTONDOWN:
                                    go = "enter"
                            else:
                                sel_ab = i
                if event.type == pg.KEYDOWN:
                    k = event.key
                    if k == pg.K_ESCAPE:
                        go = "back"
                    elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
                        go = "enter"
                    elif k in (pg.K_LEFT, pg.K_a) and screen == "modes":
                        sel_mode = (sel_mode - 1) % len(MODE_ORDER)
                    elif k in (pg.K_RIGHT, pg.K_d) and screen == "modes":
                        sel_mode = (sel_mode + 1) % len(MODE_ORDER)
                    elif k in (pg.K_UP, pg.K_w) and screen == "abilities":
                        sel_ab = max(0, sel_ab - 1)
                    elif k in (pg.K_DOWN, pg.K_s) and screen == "abilities":
                        sel_ab = min(len(MODES[mode_id].abilities) - 1, sel_ab + 1)
                    elif k == pg.K_TAB and in_game and screen == "modes":
                        return True  # TAB again resumes
            mode_id = MODE_ORDER[sel_mode]
            if go == "back":
                if screen == "abilities":
                    screen = "modes"
                elif in_game:
                    return True
                else:
                    return False
            elif go == "enter":
                if screen == "modes":
                    screen, sel_ab, error = "abilities", 0, ""
                else:
                    if mode_id != self.ctx.mode or not in_game:
                        self.set_mode(mode_id)
                    if self.camera is None:
                        self.menus.warm(self.screen.get_size())
                        message = "Opening camera and loading models..."
                        self.menus.draw_ability_panel(self.screen, t, mode_id, self._key_labels(mode_id),
                                                      sel_ab, self._last_surface, message, self._gesture_hints(mode_id))
                        pg.display.flip()
                        error = self._start_pipeline() or ""
                        message = ""
                        if error:
                            print(error)
                            screen = "modes"
                            continue
                    self._close_transition(mode_id, screen, sel_ab, t)
                    return True
            if screen == "modes":
                self.menus.draw_mode_select(self.screen, t, sel_mode, self._last_surface, message, error)
            else:
                footer = "ENTER: resume  ESC: back to modes" if in_game else "ENTER: play  ESC: back"
                self.menus.draw_ability_panel(self.screen, t, mode_id, self._key_labels(mode_id), sel_ab,
                                              self._last_surface, footer, self._gesture_hints(mode_id))
            pg.display.flip()
            self.clock.tick(60)

    def _close_transition(self, mode_id: str, screen: str, sel_ab: int, t: float) -> None:
        """First half of the menu -> camera wipe, drawn over the ability panel."""
        if self.max_frames:
            return
        start = time.perf_counter()
        while True:
            p = min(0.5, (time.perf_counter() - start) / 0.5)
            self.menus.draw_ability_panel(self.screen, t, mode_id, self._key_labels(mode_id), sel_ab,
                                          self._last_surface, "", self._gesture_hints(mode_id))
            self.menus.draw_transition(self.screen, p, mode_id)
            self.pg.display.flip()
            for event in self.pg.event.get():  # swallow keys pressed during the wipe
                if event.type == self.pg.QUIT:
                    self.running = False
            if p >= 0.5:
                break
            self.clock.tick(60)
        self._trans_p = 0.5

    def _key_labels(self, mode_id: str) -> Dict[str, str]:
        return {aid: ("ESC" if k == "escape" else k.upper()) for aid, k in self.mode_keys(mode_id).items()}

    def _gesture_hints(self, mode_id: str) -> Dict[str, str]:
        names = {"swing": "fast fist swing", "open_palm": "hold open palm", "two_fingers": "hold two fingers up",
                 "both_palms_up": "both palms above shoulders", "point": "hold index finger up", "fist": "hold fist"}
        return {aid: names.get(g, g) for aid, g in self.mode_gestures(mode_id).items()}

    def startup(self) -> bool:
        """Menus at launch (skipped for headless runs or when a start mode is set)."""
        if self.max_frames > 0 or self.start_mode:
            error = self._start_pipeline()
            if error:
                print(error)
                return False
            return True
        return self._menu_loop(in_game=False)

    # --- input -------------------------------------------------------------
    def _handle_key(self, event) -> None:
        pg, ctx = self.pg, self.ctx
        if self.hud.show_calibration and self._handle_calibration_key(event):
            return
        aid = self.ability_keymap.get(event.key)
        if aid is not None:
            self.manager.trigger(aid)
            return
        action = self.keymap.get(event.key)
        if action is None:
            return
        if ctx.mode != BLEACH and action in self.BLEACH_ONLY:
            return
        toggle = self.TOGGLE_ACTIONS.get(action)
        if toggle and toggle not in self.manager.mode.toggles:
            return
        if action == "menu":
            if not self._menu_loop(in_game=True):
                self.running = False
            # Holds measured with the wall clock would complete across the paused menu.
            self.gestures.reset()
            return
        if action == "exit":
            self.running = False
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

    def _recheck_builtin(self, cam: Camera) -> Camera:
        """Opening a camera can make an iPhone appear and shift the numbering;
        if the built-in webcam moved, reopen it at its new index."""
        idx = resolve_camera("auto", quiet=True)
        if idx == self.source:
            return cam
        cam.release()
        cam_cfg = self.cfg["camera"]
        try:
            new = Camera(idx, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"], mirror=cam_cfg["mirror"])
        except CameraError:
            # Couldn't reopen; go back to what we had.
            return Camera(self.source, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"], mirror=cam_cfg["mirror"])
        print(f"Camera list changed after opening; switched to the built-in camera at index {idx}.")
        self.source = idx
        return new

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
        self.canvas.quality = preset
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
            if not self.startup():
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
        self._last_surface = surf  # menu backdrop when TAB opens the menu
        stats = self.stats.snapshot()
        stats["camera_lost"] = camera_lost and not self.demo
        g = self.cfg["gestures"]
        self.hud.draw(self.screen, self.ctx, self.manager.active_abilities(), stats, gf, self.gestures.enabled,
                      self.cfg["quality"], g, mode=self.manager.mode, mode_keys=self._key_labels(self.ctx.mode))
        if self._trans_p < 1.0:  # second half of the menu -> camera wipe
            self.menus.draw_transition(self.screen, self._trans_p, self.ctx.mode)
            self._trans_p = min(1.0, self._trans_p + self.ctx.dt / 0.5)
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
