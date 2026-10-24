"""Settings, quality presets, key bindings and gesture thresholds.

Everything here can be overridden by a JSON file (``config.json`` in the
project root is loaded automatically, or pass ``--config path``). Only the
keys you want to change need to appear in the file; they are deep-merged
over the defaults. Run ``python -m bleach_ar --write-config config.json`` to
get a complete file to edit.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

# Colors are BGR (OpenCV order) everywhere in the effect code.
DEFAULTS: Dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "mirror": True,
    },
    "window": {
        "width": 1280,
        "height": 720,
        "fullscreen": False,
    },
    # "low" is meant for integrated graphics / older laptops.
    "quality": "medium",
    "quality_presets": {
        "low": {
            "render_width": 640,
            "tracking_width": 480,
            "model_complexity": 0,
            "segmentation": False,
            "max_particles": 400,
            "glow_scale": 0.25,
        },
        "medium": {
            "render_width": 960,
            "tracking_width": 640,
            "model_complexity": 0,
            "segmentation": True,
            "max_particles": 900,
            "glow_scale": 0.33,
        },
        "high": {
            "render_width": 1280,
            "tracking_width": 640,
            "model_complexity": 1,
            "segmentation": True,
            "max_particles": 1600,
            "glow_scale": 0.5,
        },
    },
    "effects": {
        "intensity": 1.0,
        "screen_shake": True,
        "sword_mode": "sword",  # "sword" or "hand" (hand-only energy, no blade)
        "weapon_style": "katana",  # katana | cleaver | dark
        "eye_style": "hollow",  # hollow | red | blue | white
        "show_mask": False,
    },
    "tracking": {
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        # Landmarks are held (and faded) for this long after tracking is lost.
        "hold_seconds": 0.35,
        # 0 = no smoothing, closer to 1 = smoother but laggier.
        "smoothing": 0.45,
        # Run inference on a worker thread in parallel with rendering. Higher
        # FPS; overlays may trail fast motion by ~1 inference (~25 ms).
        "async": True,
    },
    "gestures": {
        "enabled": True,
        # Frames a static gesture must be held before it counts.
        "stable_frames": 6,
        # Seconds a pose must be held to trigger shikai / bankai / cero.
        "hold_seconds": 0.8,
        # Sword-tip speed (fraction of frame diagonal per second) that counts
        # as a swing and fires Getsuga Tensho.
        "swing_speed": 2.4,
        # Minimum time between any two gesture-triggered abilities.
        "global_cooldown": 0.8,
        # finger is "extended" when tip is this much farther from the wrist
        # than its PIP joint (ratio).
        "finger_extend_ratio": 1.12,
        "bindings": {
            "getsuga": "swing",
            "shikai": "two_fingers",
            "bankai": "both_palms_up",
            "cero": "open_palm",
        },
    },
    "abilities": {
        "getsuga": {
            "charge": 0.25,
            "active": 0.9,
            "impact": 0.35,
            "cooldown": 1.5,
            "speed": 1.0,
            "size": 1.0,
            "intensity": 1.0,
            "color": [255, 230, 170],
            "core_color": [255, 255, 255],
        },
        "shikai": {
            "charge": 1.0,
            "active": 1.6,
            "cooldown": 1.0,
            "release_name": "SHIKAI",
            "command": "Rise up",
            "weapon_style": "cleaver",
            "color": [255, 200, 90],
        },
        "bankai": {
            "dim": 0.6,
            "charge": 1.6,
            "release": 0.5,
            "title": 1.6,
            "cooldown": 2.0,
            "release_name": "BANKAI",
            "weapon_style": "dark",
            "color": [60, 40, 230],
            "rim_color": [30, 20, 20],
        },
        "cero": {
            "charge": 1.1,
            "active": 0.9,
            "impact": 0.45,
            "cooldown": 3.0,
            "size": 1.0,
            "intensity": 1.0,
            "color": [60, 40, 255],
            "core_color": [235, 235, 255],
        },
        "aura": {
            "color": [255, 190, 110],
            "intensity": 1.0,
        },
    },
    # Optional image assets. Leave empty to use the procedural versions.
    # Sword PNG: RGBA, blade pointing straight up, grip at the bottom centre.
    # Mask PNG: RGBA, front-facing, eyes on a horizontal line.
    "assets": {
        "sword_image": "",
        "mask_image": "",
        "font": "",
    },
    "keys": {
        "getsuga": "1",
        "shikai": "2",
        "bankai": "3",
        "cero": "4",
        "toggle_sword": "5",
        "toggle_aura": "6",
        "toggle_eyes": "7",
        "toggle_mask": "8",
        "cycle_eye_style": "9",
        "cycle_weapon": "0",
        "toggle_gestures": "g",
        "toggle_debug": "d",
        "toggle_calibration": "c",
        "cycle_quality": "q",
        "toggle_sword_mode": "m",
        "reset": "r",
        "help": "h",
        "fullscreen": "f",
        "next_camera": "v",
        "exit": "escape",
    },
}

QUALITY_ORDER = ["low", "medium", "high"]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Dict-backed settings with attribute-free, explicit access helpers."""

    def __init__(self, data: Dict[str, Any] | None = None, source: str | None = None):
        self.data = deep_merge(DEFAULTS, data or {})
        self.source = source
        if self.data["quality"] not in self.data["quality_presets"]:
            raise ValueError(
                f"Unknown quality '{self.data['quality']}'. "
                f"Choose one of: {', '.join(self.data['quality_presets'])}"
            )

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = path or (DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None)
        if not path:
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls(json.load(fh), source=path)
        except FileNotFoundError:
            raise SystemExit(f"Config file not found: {path}")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Config file {path} is not valid JSON: {exc}")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def preset(self) -> Dict[str, Any]:
        return self.data["quality_presets"][self.data["quality"]]

    def cycle_quality(self) -> str:
        idx = QUALITY_ORDER.index(self.data["quality"]) if self.data["quality"] in QUALITY_ORDER else 0
        self.data["quality"] = QUALITY_ORDER[(idx + 1) % len(QUALITY_ORDER)]
        return self.data["quality"]

    def asset_path(self, name: str) -> str | None:
        """Resolve an optional asset path; returns None (with a warning) if missing."""
        raw = self.data["assets"].get(name) or ""
        if not raw:
            return None
        path = raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)
        if not os.path.exists(path):
            print(f"[assets] '{name}' not found at {path}; using the procedural fallback.")
            return None
        return path
