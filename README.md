# Bleach AR: Real-Time Anime Combat in Your Webcam

A desktop app that tracks your face, eyes, hands and body through your webcam and layers anime abilities over the live feed. It has two universes, chosen from a menu at startup:

- **BLEACH:** a Zanpakutō that follows your fist, Getsuga Tenshō slashes, a reiatsu aura, Shikai and Bankai transformations (Senbonzakura Kageyoshi or Tensa), Cero blasts, glowing eyes and a mask.
- **JUJUTSU KAISEN:** Lapse: Blue, Reversal: Red, Hollow Purple and Black Flash, plus two Domain Expansions, Unlimited Void and Malevolent Shrine, which replace the room behind you while you stay visible.

Everything runs locally on the CPU with Python, OpenCV, MediaPipe and pygame. There are no API keys, no cloud services and no GPU requirement, and every effect is generated procedurally. The app ships no copyrighted artwork.

> Fan-made, non-commercial project for learning computer vision. *Bleach* and related names are trademarks of Tite Kubo, Shueisha and VIZ Media; *Jujutsu Kaisen* and related names are trademarks of Gege Akutami and Shueisha. All visuals are original procedural designs. This repo started as a fork of [ioscbasotcstw/NarutoAR](https://github.com/ioscbasotcstw/NarutoAR); see [Credits](#credits).

---

## Quick start

Python **3.9–3.12** is required. MediaPipe 0.10.21 has no wheels for 3.13 or newer. Development and testing used 3.11.

### macOS

```bash
git clone https://github.com/HavishNSK/BleachAR.git
cd BleachAR
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m bleach_ar
```

The first time, macOS asks whether your terminal app (Terminal, iTerm, VS Code…) may use the camera. Allow it. If you denied it earlier, enable it in **System Settings → Privacy & Security → Camera**, then restart the terminal.

### Windows

```powershell
git clone https://github.com/HavishNSK/BleachAR.git
cd BleachAR
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m bleach_ar
```

The app opens on a **mode selector**:
1. Pick **BLEACH** or **JUJUTSU KAISEN** with the arrow keys or the mouse, and press **Enter**.
2. The next screen lists that mode's abilities with their keys and gestures. Press **Enter** to start the camera.
3. In game, **Tab** reopens the menu. You can switch universes there without restarting the camera.

To skip the menu, run `python -m bleach_ar --mode jjk` (or `--mode bleach`). The first launch downloads a small MediaPipe pose model (a few MB, one time only).

### Try it without a webcam

```bash
python -m bleach_ar --demo
```

`--demo` replaces the camera and tracking with a synthetic figure and plays a scripted showcase of every ability in the chosen mode (`python -m bleach_ar --demo --mode jjk` for Jujutsu Kaisen). The keyboard controls still work.

---

## Controls

### Bleach mode

| Key | Action |
|---|---|
| `1` | **Getsuga Tenshō** (in Senbonzakura Bankai: command the blade swarm to attack) |
| `2` | **Shikai**: release, or press again to seal |
| `3` | **Bankai**: transformation, or press again to release |
| `B` | Bankai style: **Senbonzakura** (default) / **Tensa** (crimson). Only while not in Bankai |
| `4` | **Cero** |
| `5` | Toggle Zanpakutō |
| `6` | Toggle reiatsu aura |
| `7` | Toggle glowing eyes |
| `8` | Toggle mask |
| `9` | Cycle eye style (hollow / red / blue / white) |
| `0` | Cycle sealed weapon style (katana / cleaver / dark) |
| `M` | Switch between sword mode and hand-only energy mode |
| `G` | Gestures on/off |
| `C` | Gesture calibration panel |
| `Q` | Quality: low / medium / high |
| `D` | Show tracking landmarks |
| `F` | Fullscreen |
| `V` | Switch to the next camera (e.g. built-in webcam vs iPhone Continuity Camera) |
| `R` | Reset all abilities and transformations |
| `H` | Show/hide controls |
| `ESC` | Exit |

### Jujutsu Kaisen mode

| Key | Action |
|---|---|
| `1` | **Lapse: Blue** |
| `2` | **Reversal: Red** |
| `3` | **Hollow Purple** |
| `4` | **Black Flash** |
| `Z` | **Domain Expansion: Unlimited Void**; press again to collapse it early |
| `X` | **Domain Expansion: Malevolent Shrine**; press again to end it early |
| `6` / `7` | Toggle aura / glowing eyes (blue by default) |

The common keys (`9`, `G`, `C`, `Q`, `D`, `F`, `V`, `R`, `H`, `ESC`) work in both modes. Bleach-only keys (`5`, `8`, `0`, `B`, `M`) do nothing in Jujutsu Kaisen mode.

### Both modes

| Key | Action |
|---|---|
| `Tab` | Open the menu: switch universe, or `Esc` to resume |

Bleach bindings live under `keys` in the config, and Jujutsu Kaisen bindings under `modes.jjk.keys` (see [Configuration](#configuration)).

### Gestures

Gestures add to the keyboard; they never replace it. They are rule-based, built on MediaPipe hand landmarks, with no trained model. A pose must be stable for a few frames and then held, so it can't fire every frame. A global cooldown also stops one motion from triggering two abilities.

| Gesture | Triggers |
|---|---|
| Make a **fist**: the sword appears in it | Zanpakutō grip |
| **Swing** the sword quickly | Getsuga Tenshō (Senbonzakura attack while in that Bankai) |
| Hold an **open palm** (~0.8 s) | Cero (raise your palm to fire at the camera, or point your forearm to fire a beam) |
| Hold **index + middle finger** up (~0.8 s) | Shikai |
| Hold **both palms open above your shoulders** (~0.8 s) | Bankai |

In Jujutsu Kaisen mode the same gestures map to:

| Gesture | Triggers |
|---|---|
| Fast **fist swing** (a punch) | Black Flash |
| Hold an **open palm** | Hollow Purple |
| Hold **index + middle finger** up | Unlimited Void |
| Hold **both palms above your shoulders** | Malevolent Shrine |

Blue and Red are keyboard-only (`1` / `2`). Gesture bindings live in `gestures.bindings` (Bleach) and `modes.jjk.gestures`.

MediaPipe Holistic finds hands from your body pose, so **keep your head and shoulders in frame**. Close-ups of only your hands are not detected.

### Calibrating gestures

Press `C` to open the calibration panel. It shows, live:
- which fingers are read as extended on each hand;
- the recognized gesture and its hold progress;
- your sword-swing speed against the trigger threshold.

While the panel is open, these keys adjust the thresholds:

| Keys | Adjusts |
|---|---|
| `[` / `]` | Swing speed threshold |
| `-` / `=` | Hold time |
| `,` / `.` | Finger-extension ratio |
| `S` | Saves your settings to `config.json` |

---

## Abilities

Every timed ability runs through the same state machine: `IDLE → CHARGING → ACTIVE → IMPACT → COOLDOWN → IDLE`. It can only be triggered from `IDLE`, so holding a key or repeating a gesture never restarts it mid-flight. Every timing, size, speed, intensity, cooldown and color is configurable.

| Ability | What you see |
|---|---|
| **Zanpakutō** | A blade held in your fist. Make a fist to summon it; it stays while the hand remains a fist (a few non-fist frames in a row are needed to dismiss it, so a misread frame doesn't flicker it) and fades in and out. Its direction follows the diagonal of a real grip: your knuckle line tilted toward your forearm (pose elbow to wrist), leaning on the forearm when the fist is seen edge-on. It is a 2D approximation; no 3D tracking is claimed. An adaptive (One-Euro) filter keeps it steady when still without dulling fast swings. Your fingers are drawn over the handle, taken from the camera image. Fast swings leave a slash trail with sparks and an impact flash. Styles: katana, cleaver (Shikai) and dark (Bankai), drawn from the bundled art in `assets/` with an energy edge glow. Hand-only mode replaces the blade with an energy blade around any visible hand. |
| **Getsuga Tenshō** | Energy gathers on the blade, then a layered crescent (glow, body, white core, afterimages, shed particles) launches in your swing direction. It shrinks while drifting toward a vanishing point so it seems to fly away from you, and ends in a shockwave impact with screen shake. In Bankai it is black with a crimson rim. |
| **Reiatsu aura** | Flame-like spiritual pressure rising off your silhouette, plus embers, a pulsing rim and a glowing pool where your body meets the bottom of the frame. It uses MediaPipe person segmentation; low quality, or a missing mask, falls back to a silhouette built from pose landmarks. It follows you and fades out if tracking is lost. |
| **Shikai** | Particles spiral into the sword, the aura flares, then a flash releases the transformed blade with a "SHIKAI — Rise up" title. Press `2` again to seal. |
| **Bankai (Tensa style)** | The screen dims, energy floods into your body as a rising column of light with building shake, then it releases with a white flash, a radial particle burst, shockwave rings and the "BANKAI" title. The sword, aura and energy colors all change. All other abilities are locked while the sequence plays and it cannot retrigger. Press `3` again to release Bankai. |
| **Senbonzakura Kageyoshi** (Bankai style, `B`) | Byakuya's Bankai. The screen dims, your sword sinks away, and a forest of glowing pink-white blades rises from the ground around you. One pink-white flash shatters it into a swarm of around 160 cherry-blossom petals (real sakura shapes with notched tips, tumbling as they fly, the way the thousand blades look in the anime), and the "SENBONZAKURA KAGEYOSHI" title appears. The swarm then orbits you in tilted rings for as long as Bankai lasts. Blades on the far side are smaller and dimmer and fade behind your silhouette. Blades on the near side are drawn in front of you. The swarm follows your torso and holds its place if tracking drops out. While in this Bankai, `1` or a sword swing commands the swarm: it gathers into a fan behind your hand, streams as a river of petals in the direction you point, bursts with a pink shockwave, then flies back into orbit. Press `3` to release: the petals scatter and fade, and your sword returns. |
| **Cero** | A crimson orb charges at your open palm, with crackling arcs and converging particles. It then fires a layered beam along your forearm, or, if your palm faces the camera, a screen-filling blast toward the viewer, followed by an impact. |
| **Eyes and mask** | Glowing irises anchored to MediaPipe's iris landmarks, hidden when you blink, leaving light streaks when you turn quickly. The "hollow" style darkens the whites of the eyes. The optional procedural bone mask follows head position, scale and tilt. Both fade out gracefully when face tracking is lost. |

### Jujutsu Kaisen abilities

| Ability | What you see |
|---|---|
| **Lapse: Blue** (cursed technique) | A compact blue-black core with a bright rim at your pointing finger or palm. Spiral streaks and particles are pulled in and swallowed, and the room bends inward around it with a pinch-and-swirl distortion. It then collapses and releases with a small pop. |
| **Reversal: Red** (cursed technique) | A white-hot red core that bursts outward: three staggered shockwave rings, flying debris, a short bulge and ripple distortion, and nearby particles pushed away. It fades smoothly, with no lingering flash. |
| **Hollow Purple** (cursed technique) | A Blue orb at one hand and a Red orb at the other, clearly separate, then orbiting together and merging into a violet-white core. The core launches with a churning purple shell, a white centre, a bow shock, a tapered wake and a moving distortion. It ends in an expanding distortion ring. It aims along your fist or swing, or your pointing finger; with no hands it flies away into the scene in perspective. |
| **Black Flash** (cursed technique) | A short anticipation (the room darkens, speed lines converge on your fist), then a black-and-red impact: angular branching lightning with black cores and red rims, a fast shockwave, chromatic aberration, and fading afterimages of the bolts. It does not try to detect the anime's timing rule; it is a responsive visual. |
| **Unlimited Void** (Domain Expansion) | The room dims and a bright boundary ring expands from you. Inside it, the room is replaced (you stay visible) by an infinite cosmic space: a rotating nebula, three parallax star layers rushing outward, a twisted hexagonal tunnel, rotating dial rings and streams of glyph-like light. It builds in speed, then the boundary collapses and the room returns. `Z` again collapses it early. Other cursed techniques still work inside it. |
| **Malevolent Shrine** (Domain Expansion) | A dark red pulse spreads from you, and a shrine materialises behind you: tiered roofs with upturned eaves, a fanged maw, bone spires, skull piles and red fog. Glowing cracks spread across the ground. Slashes cut through the scene at three depths, irregular and spread out, with bright cores and short trails. It ends with a cross-slash as the shrine crumbles. `X` again ends it early. Other techniques are locked while it runs. |

Only one Domain Expansion can run at a time. Switching modes (Tab) stops everything from the previous mode: abilities, transformations, particles, flashes, shake, and domain backdrops. The camera and tracking keep running.

---

## Configuration

Defaults live in `bleach_ar/config.py`. To override them, create `config.json` in the project root; it is loaded automatically. You can also pass `--config path.json`. List only the keys you want to change. For example:

```json
{
  "quality": "low",
  "camera": { "index": 1 },
  "abilities": { "getsuga": { "cooldown": 0.8, "size": 1.4 } },
  "keys": { "cero": "x" }
}
```

Bankai style and Senbonzakura tuning live under `abilities.bankai.style` (`"senbonzakura"` or `"tensa"`) and `abilities.senbonzakura`. The tunable keys are `count` (blade count at medium quality, scaled by the quality preset and capped at 300), `orbit_radius` (maximum orbit radius as a fraction of the frame diagonal), `orbit_speed`, `blade_size`, `intensity`, the three energy colours, the petal paint colours (`petal_color`, `petal_base_color`, `petal_tip_color`), and the attack timings `charge` / `active` / `impact` / `cooldown`.

Jujutsu Kaisen tuning lives under `abilities.blue`, `abilities.red`, `abilities.hollow_purple`, `abilities.black_flash`, `abilities.unlimited_void` and `abilities.malevolent_shrine`. The keys are the phase timings (`charge` / `active` / `impact` / `cooldown`, where `active` is the domain's duration), plus `intensity`, `size` and colours. The techniques also take `warp`, `particles`, `shake` and `flash` multipliers. Unlimited Void adds `star_count`, `stream_count`, `speed`, `aberration` and `warps`. Malevolent Shrine adds `slash_density`, `slash_rate`, `slash_warp` and `parallax`. `modes.start` (`""`, `"bleach"` or `"jjk"`) skips the mode selector.

To get a complete file to edit, run `python -m bleach_ar --write-config config.json`.

### Command-line options

| Option | Description |
|---|---|
| `--camera N` | Camera index (try `1` if the default is wrong) |
| `--quality low\|medium\|high` | Quality preset |
| `--width / --height / --fps` | Camera capture settings |
| `--source FILE` | Play a video file instead of the webcam |
| `--source synthetic` | Animated test pattern |
| `--mode bleach\|jjk` | Start in this mode and skip the mode selector |
| `--demo` | Synthetic person and scripted showcase |
| `--frames N` | Exit after N frames and print a timing report |
| `--headless` | No window (for tests) |

### Quality presets

| Preset | Render width | Tracking input | Pose model | Segmentation | Particles |
|---|---|---|---|---|---|
| `low` (integrated graphics / older laptops) | 640 px | 480 px | lite | off (pose silhouette) | 400 |
| `medium` (default) | 960 px | 640 px | lite | on | 900 |
| `high` | 1280 px | 640 px | full | on | 1600 |

Each preset also sets rendering limits: `bloom` (single or dual scale), `warps`, `max_warps` and `warp_grid` (distortion budget), and `aberration`. Low quality uses single-scale bloom, at most 2 warps, and no aberration.

Tracking runs on a worker thread by default (`"tracking": {"async": true}`), so inference overlaps rendering. The trade-off is that overlays can trail very fast motion by about one inference (~25 ms). Set `async` to `false` for perfectly frame-locked overlays at a lower frame rate.

### Optional assets

The repo ships original sword and mask art in `assets/` (about 210 KB in total), rendered offline by `tools/make_assets.py` with numpy/OpenCV at 4x supersampling. Run `python tools/make_assets.py` to regenerate it. Every effect also has a procedural fallback, so the art is optional. Point these keys under `assets` at your own files to replace it, or set them to `""` to use the procedural versions:

| Key | What to supply |
|---|---|
| `sword_images` | One RGBA PNG per style (`katana`, `cleaver`, `dark`). Blade pointing straight up, square pixels. The image height spans the blade from the pommel end (bottom, 0.2 blade lengths behind the grip) to the tip (top), and the blade axis is the horizontal centre. The grip sits 1/6 of the way up from the bottom. Default: `assets/sword_<style>.png`. |
| `sword_image` | Legacy single image (same layout) used for any style without its own entry |
| `mask_image` | RGBA PNG, front-facing, eyes at (32%, 42%) and (68%, 42%) of the image, chin at (50%, 95%). Default: `assets/mask.png` |
| `font` | A `.ttf` / `.otf` file for the HUD and titles |

A missing or unreadable asset prints one warning and falls back to the procedural version. Image swords are warped into a small region around the blade each frame (a pre-scaled copy is picked so it is never shrunk much), and they keep the energy edge glow. At 960×540 on an M1 Air, the sword's update and render together take about 0.7 ms procedural and 1.4–1.6 ms with image art, finger occlusion included.

---

## Performance (measured)

Measured on an **Apple M1 MacBook Air (8 GB)** running Python 3.11. The input was the full app with real MediaPipe tracking of a real person, from a 30 fps 1280×720 clip built from MediaPipe's test photo. Command: `python -m bleach_ar --headless --source clip.mp4 --frames 300 --quality …`.

| Quality | Render loop FPS | Camera frames shown/s | Tracking (median) | Effects (median) |
|---|---|---|---|---|
| low, async | 53 | 26.1 | 25.2 ms | 3.8 ms |
| medium, async (default) | 58 | 27.1 | 25.5 ms | 6.8 ms |
| high, async | 58 | 27.0 | 31.4 ms | 9.5 ms |
| medium, sync tracking | 29 | 26.3 | 24.0 ms | 5.4 ms |

- The render loop is capped at 60 FPS.
- The clip played back at about 27 fps from a file. A webcam is paced by the driver instead.
- Peak effect load (the Bankai release burst) measured about 10–11 ms per frame at 960×540.
- Live webcam sessions on the same Mac have measured **24–30 camera fps**, with tracking at **31–88 ms median**. The slow end came with heavy memory pressure, low disk space and thermal throttling over long sessions. The HUD shows live FPS, camera fps, tracking time and effects time.

**Effect and compositing cost at 960×540** (synthetic tests; see [docs/RENDERING.md](docs/RENDERING.md) for the full benchmark):

| Effect | Effect update + render | Composite incl. backdrop/warps |
|---|---|---|
| Plain frame (bloom only) | — | ~1.7 ms |
| Unlimited Void (active) | ~2.5 ms | ~5.5 ms |
| Malevolent Shrine (active) | ~3.1 ms | ~4.3 ms (7.4 ms total) |
| Blue / Red / Hollow Purple / Black Flash | 1–3 ms | 3.5–6 ms (warps add ~2–4 ms) |
| Everything on at once (backdrop + 6 warps + aberration) | — | ~6 ms (was 11 ms before optimisation) |

**Rendering approach:** a CPU pipeline built with OpenCV and numpy, drawn into pygame. I evaluated ModernGL (OpenGL 4.1 over Metal on the M1). A shader pass benchmarked at 7.5 ms per frame, of which about 4 ms was just uploading the frame and reading it back, so it is not used. [docs/RENDERING.md](docs/RENDERING.md) has the numbers, the techniques used and the path to a GPU renderer if ever needed.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

The 195 tests cover:
- the mode registry, so every ability belongs to exactly one mode; no cross-mode triggering; full cleanup when switching mid-Bankai or mid-domain; the one-domain-at-a-time rule;
- the real menu flow, driven by scripted key presses: picking a mode, going back, Tab mid-game to switch modes while the camera is reused, and Bleach-only keys ignored in Jujutsu Kaisen mode;
- each Jujutsu Kaisen ability's lifecycle, early cancel, reset, tracking loss, resolution changes, Blue and Red converging into Hollow Purple, projectile direction and impact, and particle limits;
- the compositor: backdrop only behind the person, warp directions, quality flags, and no per-frame reallocation;
- config merging and errors;
- every state-machine transition, including cooldowns, retrigger blocking and the Bankai lock;
- gesture classification, debounce, hold, release and re-hold, the global cooldown, and dropped frames;
- camera and missing-video error handling;
- every ability rendered through every phase at every transformation level, with all weapon, eye and toggle combinations;
- tracking loss in the middle of an ability, and reset in the middle of Bankai;
- headless end-to-end runs: demo mode, real MediaPipe on a synthetic source, and a missing video source.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| It shows your iPhone instead of the Mac's webcam (Continuity Camera) | The default `"index": "auto"` picks the built-in camera by name (needs `pyobjc-framework-AVFoundation`, included in requirements). You can also press `V` to switch cameras, or force one with `--camera N`. |
| "Could not read frames from camera 0" | Grant camera permission (see Quick start), close other apps using the camera, or try `--camera 1`. |
| Pose-model download fails (SSL error on macOS) | `pip install certifi` (already in requirements). If it still fails, the app falls back to the bundled full model automatically. |
| `mediapipe has no attribute 'solutions'` | Install the pinned version: `pip install mediapipe==0.10.21`. |
| `pip` can't find mediapipe | You're on Python 3.13 or newer. Create the venv with Python 3.11 or 3.12. |
| Hands or gestures aren't detected | Keep your head and shoulders in view, use good front lighting, and open `C` (calibration) to see what's recognized. |
| Low FPS | Press `Q` for low quality, turn off the aura (`6`), or lower the camera resolution with `--width 960 --height 540`. |
| Getsuga fires too easily, or not at all | Press `C`, then use `[` / `]` to adjust the swing threshold, and `S` to save. |

---

## Known limitations

- **The Jujutsu Kaisen mode hasn't been played on a live webcam yet.** Its effects were checked on synthetic frames and in headless runs. The Bleach mode has been played live. Gesture thresholds and effect brightness may need tuning for your room and lighting.
- **Domains replace the room using the segmentation mask.** On low quality there is no segmentation, so the person outline is approximated from pose landmarks and is rougher. Hands raised away from the body can be partly covered by the backdrop.
- **First-frame hitch for domains.** The first time each domain runs at a given resolution, it spends about 40–90 ms building its environment. Later uses hit the cache.
- **Blue and Red are keyboard-only.** There is no gesture for them.
- **The sword is a 2D approximation** from hand landmarks. With knuckles facing the camera and your forearm upright, the blade leans diagonally toward the thumb side, as a real grip would, not straight up. When the fist points straight at the camera, the direction is less certain. Finger occlusion uses a mask built from the landmarks, so the handle edge can be off by a few pixels around your fingers.
- **Hands need the body in view.** Holistic finds hands from the body pose, so close-up hands without a body aren't detected.
- **Async tracking can lag slightly.** Overlays can trail very fast motion by about one inference (~25 ms). Set `"async": false` to avoid this, at a lower frame rate.
- **Segmentation is soft.** MediaPipe's segmentation edge is soft, so the aura's mask threshold may need tuning for some backgrounds.
- **Charging pulls in all particles.** During charging, effects attract every live particle, including aura embers. This is deliberate, but not strictly per-effect.
- **Bloom bleeds onto the face.** At peak Bankai, the additive bloom spills over the face and mask.

---

## Project structure

```
bleach_ar/
  __main__.py     CLI entry point (python -m bleach_ar)
  app.py          main loop: camera -> tracking -> gestures -> abilities -> compositing -> HUD
  camera.py       threaded webcam/video capture, helpful errors
  tracking.py     MediaPipe Holistic wrapper: smoothing, loss-hold, async worker, model download
  gestures.py     rule-based hand gestures, debounce, hold, cooldown
  modes.py        anime modes (Bleach / Jujutsu Kaisen) and the abilities registered to each
  menus.py        mode selector, ability panel, transitions
  abilities.py    ability state machine, shared per-frame Context, mode-aware AbilityManager
  fx.py           compositing (additive light, two-scale bloom, backdrop behind the person, warps,
                  chromatic aberration, quality flags), particles, trails, lightning, starfield, shake
  jjk/            Jujutsu Kaisen: techniques.py (Blue, Red, Hollow Purple, Black Flash),
                  unlimited_void.py, malevolent_shrine.py
  senbonzakura.py Senbonzakura petal swarm and attack
  sword.py        Zanpakuto summon/dismiss, pose + One-Euro filter, blade/image rendering, finger occlusion, sinking
  getsuga.py      Getsuga Tensho
  transform.py    Shikai and Bankai
  cero.py         Cero
  aura.py         reiatsu aura
  eyes.py         glowing eyes and mask
  hud.py          HUD, titles, help, calibration, startup screen
  config.py       defaults, presets, key bindings, JSON overrides
  synthetic.py    synthetic person for --demo and tests
tools/
  make_assets.py  renders the bundled sword and mask art into assets/
  bench_render.py compositor benchmark (see docs/RENDERING.md)
docs/RENDERING.md rendering pipeline, benchmarks, techniques and the ModernGL evaluation
assets/           original sword (katana, cleaver, dark) and mask PNGs
tests/            unittest suite
```

## Credits

- **Original project:** [NarutoAR](https://github.com/ioscbasotcstw/NarutoAR) by ioscbasotcstw. The pygame/OpenCV loop structure, the MediaPipe Holistic setup, the alpha-blend overlay approach and the eye-openness ratio are adapted from it. The Naruto-specific code, the YOLO hand-sign model and its assets, and the Gemini integration were removed: they don't apply to Bleach, and they needed PyTorch and a paid API key. The original repo has no license file, so check with its author before publishing derived work.
- **Tracking:** [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0).
