"""Ability state machine and the per-frame context shared by all effects.

Every timed ability runs through the same explicit states:

    IDLE -> CHARGING -> ACTIVE -> IMPACT -> COOLDOWN -> IDLE

A phase with a duration of 0 is skipped. ``trigger()`` only works from IDLE,
so a held key or repeated gesture can never restart an ability mid-flight.
Subclasses override ``on_enter`` (one-shot work such as spawning a projectile
or kicking screen shake), ``on_update`` and ``render``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .fx import Canvas, ParticleSystem, Shaker
from .tracking import TrackingState


class Phase(Enum):
    IDLE = "IDLE"
    CHARGING = "CHARGING"
    ACTIVE = "ACTIVE"
    IMPACT = "IMPACT"
    COOLDOWN = "COOLDOWN"


ORDER = [Phase.CHARGING, Phase.ACTIVE, Phase.IMPACT, Phase.COOLDOWN]

# Transformation levels.
SEALED, SHIKAI, BANKAI = "sealed", "shikai", "bankai"


@dataclass
class SwordPose:
    """Where the Zanpakuto is this frame (render pixels). Set by ``sword.Zanpakuto``."""

    visible: bool = False
    hand: Optional[str] = None  # "left" / "right" key into tracking.hands
    grip: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    tip: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0], np.float32))
    length: float = 0.0
    # Tip speed as a fraction of the frame diagonal per second.
    speed: float = 0.0
    # Unit vector of the tip's motion (valid when speed > 0).
    motion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], np.float32))
    confidence: float = 0.0


@dataclass
class Title:
    text: str
    sub: str
    color: Tuple[int, int, int]
    start: float
    duration: float


@dataclass
class Context:
    """Everything an effect needs for one frame. Built once, mutated per frame."""

    cfg: dict  # the full config dict (Config.data)
    canvas: Canvas
    particles: ParticleSystem
    shaker: Shaker
    tracking: TrackingState = field(default_factory=TrackingState)
    sword: SwordPose = field(default_factory=SwordPose)
    now: float = 0.0
    dt: float = 0.0
    level: str = SEALED
    # Persistent toggles (keys 5-8).
    show_sword: bool = True
    show_aura: bool = True
    show_eyes: bool = False
    show_mask: bool = False
    weapon_style: str = "katana"
    eye_style: str = "hollow"
    sword_mode: str = "sword"  # "sword" | "hand"
    # Effects may raise this for a frame (Bankai/charging boosts the aura).
    aura_boost: float = 0.0
    # Set while a transformation sequence plays; blocks other triggers.
    locked_by: Optional[str] = None
    # 0 = blade drawn normally, 1 = blade fully sunk/dissolved (Senbonzakura
    # lowers the sword into the ground and it stays hidden during that Bankai).
    sword_sink: float = 0.0
    # Which Bankai plays: "tensa" (crimson black-blade release) or "senbonzakura".
    # app.py initialises it from cfg['abilities']['bankai']['style']; the B key
    # cycles it, but only while not in (or transforming into) Bankai.
    bankai_style: str = "senbonzakura"
    # The persistent Senbonzakura blade swarm (senbonzakura.BladeSwarm), owned
    # by AbilityManager; None when an ability runs without a manager.
    swarm: Optional[Any] = None
    titles: List[Title] = field(default_factory=list)

    @property
    def diag(self) -> float:
        return float(np.hypot(self.canvas.w, self.canvas.h)) or 1.0

    def energy_color(self) -> Tuple[int, int, int]:
        """Current spiritual-energy color for the transformation level."""
        ab = self.cfg["abilities"]
        senbon = self.bankai_style == "senbonzakura"
        if senbon and self.locked_by == "bankai":  # Senbonzakura activation / release
            return tuple(ab["senbonzakura"]["color"])
        if self.level == BANKAI:
            if senbon:
                return tuple(ab["senbonzakura"]["color"])
            return tuple(ab["bankai"]["color"])
        if self.level == SHIKAI:
            return tuple(ab["shikai"]["color"])
        return tuple(ab["aura"]["color"])

    def current_weapon(self) -> str:
        ab = self.cfg["abilities"]
        if self.level == BANKAI:
            return ab["bankai"]["weapon_style"]
        if self.level == SHIKAI:
            return ab["shikai"]["weapon_style"]
        return self.weapon_style

    def show_title(self, text: str, sub: str = "", color=(255, 255, 255), duration: float = 1.5) -> None:
        self.titles = [t for t in self.titles if t.text != text]
        self.titles.append(Title(text, sub, tuple(color), self.now, duration))

    def body_center(self) -> Optional[np.ndarray]:
        pose = self.tracking.pose
        if pose.ok and len(pose.points) > 12:
            pts = pose.points
            return ((pts[11, :2] + pts[12, :2]) / 2.0).astype(np.float32)
        if self.tracking.face.ok and len(self.tracking.face.points) > 152:
            return self.tracking.face.points[152, :2] + np.array([0, 60], np.float32)
        return None


class Ability:
    """Base timed ability. Durations come from ``cfg['abilities'][name]``."""

    name = "ability"
    label = "Ability"

    def __init__(self, params: dict):
        self.params = params
        self.phase = Phase.IDLE
        self.t = 0.0  # seconds spent in the current phase

    # --- durations --------------------------------------------------------
    def duration(self, phase: Phase) -> float:
        key = {Phase.CHARGING: "charge", Phase.ACTIVE: "active",
               Phase.IMPACT: "impact", Phase.COOLDOWN: "cooldown"}[phase]
        return float(self.params.get(key, 0.0))

    @property
    def progress(self) -> float:
        d = self.duration(self.phase) if self.phase != Phase.IDLE else 0.0
        return min(1.0, self.t / d) if d > 0 else 1.0

    @property
    def busy(self) -> bool:
        return self.phase not in (Phase.IDLE, Phase.COOLDOWN)

    def cooldown_left(self) -> float:
        if self.phase == Phase.COOLDOWN:
            return max(0.0, self.duration(Phase.COOLDOWN) - self.t)
        return 0.0

    # --- transitions ------------------------------------------------------
    def can_trigger(self, ctx: Context) -> bool:
        return self.phase == Phase.IDLE and (ctx.locked_by is None or ctx.locked_by == self.name)

    def trigger(self, ctx: Context) -> bool:
        if not self.can_trigger(ctx):
            return False
        self._advance_from(-1, ctx)
        return True

    def _enter(self, phase: Phase, ctx: Context) -> None:
        self.phase = phase
        self.t = 0.0
        self.on_enter(phase, ctx)

    def _advance_from(self, idx: int, ctx: Context) -> None:
        for nxt in ORDER[idx + 1:]:
            if self.duration(nxt) > 0:
                self._enter(nxt, ctx)
                return
            # Zero-length phases still get their one-shot hook.
            self.on_enter(nxt, ctx)
        self._enter(Phase.IDLE, ctx)

    def update(self, ctx: Context) -> None:
        if self.phase == Phase.IDLE:
            return
        self.t += ctx.dt
        self.on_update(ctx)
        if self.phase != Phase.IDLE and self.t >= self.duration(self.phase):
            self._advance_from(ORDER.index(self.phase), ctx)

    def reset(self, ctx: Context) -> None:
        self.phase = Phase.IDLE
        self.t = 0.0

    # --- hooks for subclasses ---------------------------------------------
    def on_enter(self, phase: Phase, ctx: Context) -> None:
        pass

    def on_update(self, ctx: Context) -> None:
        pass

    def render(self, ctx: Context) -> None:
        pass


class AbilityManager:
    """Owns the timed abilities and the persistent effects, in draw order."""

    def __init__(self, ctx: Context):
        # Imported here so each effect module only depends on this one.
        from .aura import ReiatsuAura
        from .cero import Cero
        from .eyes import EyeEffects
        from .getsuga import GetsugaTensho
        from .senbonzakura import BladeSwarm, SenbonzakuraAttack
        from .sword import Zanpakuto
        from .transform import Bankai, Shikai

        ab = ctx.cfg["abilities"]
        self.ctx = ctx
        ctx.sword_sink = 0.0
        self.swarm = BladeSwarm(ab["senbonzakura"])
        ctx.swarm = self.swarm
        self.sword = Zanpakuto(ctx.cfg)
        self.aura = ReiatsuAura(ctx.cfg)
        self.eyes = EyeEffects(ctx.cfg)
        self.abilities: Dict[str, Ability] = {
            "getsuga": GetsugaTensho(ab["getsuga"]),
            "shikai": Shikai(ab["shikai"]),
            "bankai": Bankai(ab["bankai"]),
            "cero": Cero(ab["cero"]),
            "senbon_attack": SenbonzakuraAttack(ab["senbonzakura"]),
        }

    def routed(self, name: str) -> str:
        """In Senbonzakura Bankai the Getsuga key / sword swing commands the blades."""
        if name == "getsuga" and self.ctx.level == BANKAI and self.ctx.bankai_style == "senbonzakura":
            return "senbon_attack"
        return name

    def trigger(self, name: str) -> bool:
        ab = self.abilities.get(self.routed(name))
        ok = bool(ab and ab.trigger(self.ctx))
        if ok and ab is self.abilities["bankai"]:
            # A transformation (or its release) takes over the scene: cancel any
            # attack still in flight so it can't play through the sequence or
            # re-command a swarm that is dispersing.
            for other in self.abilities.values():
                if other is not ab and other.busy:
                    other.reset(self.ctx)
        return ok

    def update(self) -> None:
        ctx = self.ctx
        ctx.aura_boost = 0.0
        self.sword.update(ctx)
        for ab in self.abilities.values():
            ab.update(ctx)
        self._update_swarm()
        self.aura.update(ctx)
        self.eyes.update(ctx)
        ctx.particles.update(ctx.dt)
        ctx.titles = [t for t in ctx.titles if ctx.now - t.start < t.duration]

    def _update_swarm(self) -> None:
        """Keep the blade swarm consistent with the transformation level."""
        ctx, sw = self.ctx, self.swarm
        bankai = self.abilities["bankai"]
        from .senbonzakura import DISPERSE, OFF, senbon_active
        if senbon_active(ctx):
            # Level set without the activation (or swarm faded): appear on the orbit.
            if not bankai.busy and sw.mode in (OFF, DISPERSE):
                sw.spawn_orbit(ctx)
        else:
            atk = self.abilities["senbon_attack"]
            if atk.phase != Phase.IDLE:
                atk.reset(ctx)
            releasing = bankai.busy and bankai.mode == "release"
            if not releasing and sw.mode not in (OFF, DISPERSE):
                sw.disperse(ctx)
        sw.update(ctx)

    def render(self) -> None:
        """Back-to-front: aura, far-side blades, eyes/mask, sword, abilities,
        near-side blades, particles."""
        ctx = self.ctx
        if ctx.show_aura:
            self.aura.render(ctx)
        self.swarm.render(ctx, "back")
        self.eyes.render(ctx)
        if ctx.show_sword:
            self.sword.render(ctx)
        for ab in self.abilities.values():
            ab.render(ctx)
        self.swarm.render(ctx, "front")
        ctx.particles.render(ctx.canvas)

    def reset(self) -> None:
        ctx = self.ctx
        for ab in self.abilities.values():
            ab.reset(ctx)
        self.sword.reset()
        ctx.sword = SwordPose()
        ctx.level = SEALED
        ctx.locked_by = None
        ctx.sword_sink = 0.0
        self.swarm.clear()
        # ctx.bankai_style is kept (the user's last choice).
        ctx.titles.clear()
        ctx.particles.clear()
        ctx.canvas.flash = 0.0
