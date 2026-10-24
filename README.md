# Bleach AR: Real-Time Soul Reaper Combat in Your Webcam

A desktop app that tracks your face, eyes, hands and body through your webcam and layers Bleach-inspired abilities over the live feed. You get a Zanpakutō that follows your fist, Getsuga Tenshō slashes, a reiatsu aura, Shikai and Bankai transformations, Cero blasts, glowing eyes and a mask.

Everything runs locally on the CPU with Python, OpenCV, MediaPipe and pygame. There are no API keys, no cloud services and no GPU requirement, and every effect is generated procedurally. The app ships no copyrighted artwork.

> Fan-made, non-commercial project for learning computer vision. *Bleach* and related names are trademarks of Tite Kubo, Shueisha and VIZ Media. This repo started as a fork of [ioscbasotcstw/NarutoAR](https://github.com/ioscbasotcstw/NarutoAR); see [Credits](#credits).

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

On the startup screen, press **Space** or **Enter**. The first launch downloads a small MediaPipe pose model (a few MB, one time only).

### Try it without a webcam

```bash
python -m bleach_ar --demo
```

`--demo` replaces the camera and tracking with a synthetic figure and plays a scripted showcase of every ability. The keyboard controls still work.

---

## Controls

| Key | Action |
|---|---|
| `1` | **Getsuga Tenshō** |
| `2` | **Shikai**: release, or press again to seal |
| `3` | **Bankai**: transformation, or press again to release |
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

All bindings can be changed under `keys` in the config (see [Configuration](#configuration)).

### Gestures

Gestures add to the keyboard; they never replace it. They are rule-based, built on MediaPipe hand landmarks, with no trained model. A pose must be stable for a few frames and then held, so it can't fire every frame. A global cooldown also stops one motion from triggering two abilities.

| Gesture | Triggers |
|---|---|
| Make a **fist**: the sword appears in it | Zanpakutō grip |
| **Swing** the sword quickly | Getsuga Tenshō |
| Hold an **open palm** (~0.8 s) | Cero (raise your palm to fire at the camera, or point your forearm to fire a beam) |
| Hold **index + middle finger** up (~0.8 s) | Shikai |
| Hold **both palms open above your shoulders** (~0.8 s) | Bankai |

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
| **Zanpakutō** | A procedural blade aligned with your fist. Its direction comes from your knuckle line (a 2D approximation; no 3D tracking is claimed) and is smoothed and length-scaled to your hand. Fast swings leave a slash trail with sparks and an impact flash. Styles: katana, cleaver (Shikai) and dark (Bankai). Hand-only mode replaces the blade with an energy blade around your hand. |
| **Getsuga Tenshō** | Energy gathers on the blade, then a layered crescent (glow, body, white core, afterimages, shed particles) launches in your swing direction. It shrinks while drifting toward a vanishing point so it seems to fly away from you, and ends in a shockwave impact with screen shake. In Bankai it is black with a crimson rim. |
| **Reiatsu aura** | Flame-like spiritual pressure rising off your silhouette, plus embers, a pulsing rim and a glowing pool where your body meets the bottom of the frame. It uses MediaPipe person segmentation; low quality, or a missing mask, falls back to a silhouette built from pose landmarks. It follows you and fades out if tracking is lost. |
| **Shikai** | Particles spiral into the sword, the aura flares, then a flash releases the transformed blade with a "SHIKAI — Rise up" title. Press `2` again to seal. |
| **Bankai** | The screen dims, energy floods into your body as a rising column of light with building shake, then it releases with a white flash, a radial particle burst, shockwave rings and the "BANKAI" title. The sword, aura and energy colors all change. All other abilities are locked while the sequence plays and it cannot retrigger. Press `3` again to release Bankai. |
| **Cero** | A crimson orb charges at your open palm, with crackling arcs and converging particles. It then fires a layered beam along your forearm, or, if your palm faces the camera, a screen-filling blast toward the viewer, followed by an impact. |
| **Eyes and mask** | Glowing irises anchored to MediaPipe's iris landmarks, hidden when you blink, leaving light streaks when you turn quickly. The "hollow" style darkens the whites of the eyes. The optional procedural bone mask follows head position, scale and tilt. Both fade out gracefully when face tracking is lost. |

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

To get a complete file to edit, run `python -m bleach_ar --write-config config.json`.

### Command-line options

| Option | Description |
|---|---|
| `--camera N` | Camera index (try `1` if the default is wrong) |
| `--quality low\|medium\|high` | Quality preset |
| `--width / --height / --fps` | Camera capture settings |
| `--source FILE` | Play a video file instead of the webcam |
| `--source synthetic` | Animated test pattern |
| `--demo` | Synthetic person and scripted showcase |
| `--frames N` | Exit after N frames and print a timing report |
| `--headless` | No window (for tests) |

### Quality presets

| Preset | Render width | Tracking input | Pose model | Segmentation | Particles |
|---|---|---|---|---|---|
| `low` (integrated graphics / older laptops) | 640 px | 480 px | lite | off (pose silhouette) | 400 |
| `medium` (default) | 960 px | 640 px | lite | on | 900 |
| `high` | 1280 px | 640 px | full | on | 1600 |

Tracking runs on a worker thread by default (`"tracking": {"async": true}`), so inference overlaps rendering. The trade-off is that overlays can trail very fast motion by about one inference (~25 ms). Set `async` to `false` for perfectly frame-locked overlays at a lower frame rate.

### Optional assets

All effects are procedural, so nothing is required. To use your own art, add these under `assets` in the config:

| Key | What to supply |
|---|---|
| `sword_image` | RGBA PNG, blade pointing straight up, grip at the bottom centre |
| `mask_image` | RGBA PNG, front-facing, eyes at (32%, 42%) and (68%, 42%) of the image, chin at (50%, 95%) |
| `font` | A `.ttf` / `.otf` file for the HUD and titles |

A missing or unreadable asset prints one warning and falls back to the procedural version. Hand-painted sword and mask art, plus an anime-style display font, are the assets that would improve the look the most.

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
- These figures have **not** been measured with a live webcam. The HUD shows live FPS, camera fps, tracking time and effects time, so you can check your own machine.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

The 33 tests cover:
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
| It shows your iPhone instead of the Mac's webcam (Continuity Camera) | Press `V` to switch cameras, or put `{"camera": {"index": 1}}` in `config.json` (or run with `--camera 1`). |
| "Could not read frames from camera 0" | Grant camera permission (see Quick start), close other apps using the camera, or try `--camera 1`. |
| Pose-model download fails (SSL error on macOS) | `pip install certifi` (already in requirements). If it still fails, the app falls back to the bundled full model automatically. |
| `mediapipe has no attribute 'solutions'` | Install the pinned version: `pip install mediapipe==0.10.21`. |
| `pip` can't find mediapipe | You're on Python 3.13 or newer. Create the venv with Python 3.11 or 3.12. |
| Hands or gestures aren't detected | Keep your head and shoulders in view, use good front lighting, and open `C` (calibration) to see what's recognized. |
| Low FPS | Press `Q` for low quality, turn off the aura (`6`), or lower the camera resolution with `--width 960 --height 540`. |
| Getsuga fires too easily, or not at all | Press `C`, then use `[` / `]` to adjust the swing threshold, and `S` to save. |

---

## Known limitations

- **No live-webcam testing yet.** Tracking was checked on real photos and video, but nobody has run the app against a live camera. Gesture thresholds and effect brightness may need tuning for your room and lighting.
- **The sword is a 2D approximation** from hand landmarks. With knuckles facing the camera, the blade points sideways, toward the thumb side. The blade's handle is drawn over your fingers, not behind them.
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
  abilities.py    ability state machine, shared per-frame Context, AbilityManager
  fx.py           compositing (additive light + bloom), particles, trails, shake
  sword.py        Zanpakuto pose estimation, blade rendering, slash trails
  getsuga.py      Getsuga Tensho
  transform.py    Shikai and Bankai
  cero.py         Cero
  aura.py         reiatsu aura
  eyes.py         glowing eyes and mask
  hud.py          HUD, titles, help, calibration, startup screen
  config.py       defaults, presets, key bindings, JSON overrides
  synthetic.py    synthetic person for --demo and tests
tests/            unittest suite
```

## Credits

- **Original project:** [NarutoAR](https://github.com/ioscbasotcstw/NarutoAR) by ioscbasotcstw. The pygame/OpenCV loop structure, the MediaPipe Holistic setup, the alpha-blend overlay approach and the eye-openness ratio are adapted from it. The Naruto-specific code, the YOLO hand-sign model and its assets, and the Gemini integration were removed: they don't apply to Bleach, and they needed PyTorch and a paid API key. The original repo has no license file, so check with its author before publishing derived work.
- **Tracking:** [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0).
