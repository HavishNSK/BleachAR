"""Anime modes and the abilities registered to each.

One app, one camera/tracking/rendering pipeline; a *mode* decides which
abilities exist, their keys and gestures, the persistent effects (sword,
aura, eyes, mask) that make sense, and the UI colours. ``AbilityManager``
only updates, renders and triggers the abilities of the active mode, and
switching modes resets every ability and screen effect so nothing leaks
from one universe into the other.

Ability ids map to keys in ``cfg["abilities"]`` (their parameters) and to
classes in ``ABILITY_CLASSES`` (resolved lazily so importing this module is
cheap and each effect module only needs ``abilities``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

BLEACH, JJK = "bleach", "jjk"


@dataclass(frozen=True)
class AbilitySpec:
    id: str  # key into the manager's ability dict and cfg["abilities"]
    label: str
    category: str
    description: str
    key_action: str  # action name looked up in the mode's key bindings


@dataclass(frozen=True)
class Mode:
    id: str
    title: str
    subtitle: str
    accent: Tuple[int, int, int]  # RGB (pygame)
    accent2: Tuple[int, int, int]
    abilities: List[AbilitySpec]
    # Persistent effects shown in this mode (toggle keys 5-8 only act on these).
    toggles: Tuple[str, ...] = ()
    # Hidden abilities that exist in the mode but are reached via routing
    # (e.g. the Senbonzakura attack replaces Getsuga during that Bankai).
    hidden: Tuple[str, ...] = ()
    defaults: Dict[str, object] = field(default_factory=dict)

    @property
    def ability_ids(self) -> List[str]:
        return [a.id for a in self.abilities] + list(self.hidden)


MODES: Dict[str, Mode] = {
    BLEACH: Mode(
        id=BLEACH,
        title="BLEACH",
        subtitle="Soul Reaper combat: Zanpakuto, Getsuga Tensho, Shikai and Bankai",
        accent=(110, 190, 255),
        accent2=(235, 245, 255),
        abilities=[
            AbilitySpec("getsuga", "Getsuga Tensho", "Sword technique",
                        "A crescent of blue-white energy launched from your blade. Swing your fist fast or press the key.",
                        "getsuga"),
            AbilitySpec("shikai", "Shikai", "Release",
                        "Release your Zanpakuto. Hold two fingers up or press the key; press again to seal.",
                        "shikai"),
            AbilitySpec("bankai", "Bankai", "Release",
                        "Senbonzakura Kageyoshi or Tensa (B switches). Hold both palms above your shoulders.",
                        "bankai"),
            AbilitySpec("cero", "Cero", "Spiritual blast",
                        "A charged crimson orb and beam. Hold an open palm or press the key.",
                        "cero"),
        ],
        toggles=("sword", "aura", "eyes", "mask"),
        hidden=("senbon_attack",),
        defaults={"eye_style": "hollow", "show_sword": True, "show_aura": True},
    ),
    JJK: Mode(
        id=JJK,
        title="JUJUTSU KAISEN",
        subtitle="Cursed techniques and Domain Expansions",
        accent=(150, 110, 255),
        accent2=(235, 60, 80),
        abilities=[
            AbilitySpec("blue", "Lapse: Blue", "Cursed Technique",
                        "A compact point of attraction that pulls particles and the room inward.", "blue"),
            AbilitySpec("red", "Reversal: Red", "Cursed Technique",
                        "A red-white repulsion burst with expanding shockwaves.", "red"),
            AbilitySpec("hollow_purple", "Hollow Purple", "Cursed Technique",
                        "Blue and Red converge into a violet-white projectile that tears forward.", "hollow_purple"),
            AbilitySpec("black_flash", "Black Flash", "Cursed Technique",
                        "A precise black-red impact. Punch fast (fist swing) or press the key.", "black_flash"),
            AbilitySpec("unlimited_void", "Unlimited Void", "Domain Expansion",
                        "An overwhelming cosmic domain. Press again to collapse it early.", "unlimited_void"),
            AbilitySpec("malevolent_shrine", "Malevolent Shrine", "Domain Expansion",
                        "A shrine rises behind you and slashes cut through the space. Press again to end it.",
                        "malevolent_shrine"),
        ],
        toggles=("aura", "eyes"),
        defaults={"eye_style": "blue", "show_sword": False, "show_aura": True},
    ),
}

MODE_ORDER = [BLEACH, JJK]

# id -> "module:Class" (relative to the bleach_ar package)
ABILITY_CLASSES: Dict[str, str] = {
    "getsuga": "getsuga:GetsugaTensho",
    "shikai": "transform:Shikai",
    "bankai": "transform:Bankai",
    "cero": "cero:Cero",
    "senbon_attack": "senbonzakura:SenbonzakuraAttack",
    "blue": "jjk.techniques:Blue",
    "red": "jjk.techniques:Red",
    "hollow_purple": "jjk.techniques:HollowPurple",
    "black_flash": "jjk.techniques:BlackFlash",
    "unlimited_void": "jjk.unlimited_void:UnlimitedVoid",
    "malevolent_shrine": "jjk.malevolent_shrine:MalevolentShrine",
}


def mode_for_ability(ability_id: str) -> str:
    for m in MODES.values():
        if ability_id in m.ability_ids:
            return m.id
    raise KeyError(ability_id)


def load_class(ability_id: str):
    import importlib

    mod, cls = ABILITY_CLASSES[ability_id].split(":")
    return getattr(importlib.import_module(f"bleach_ar.{mod}"), cls)
