# Rendering pipeline: performance, helpers, quality flags

The target machine is a fanless M1 MacBook Air running Python 3.11, OpenCV 4.11,
numpy 1.26 and MediaPipe Holistic 0.10.21, with pygame 2.6 for display. Every
number below was measured on that machine.

## 1. Frame pipeline (recap)

```
camera -> resize to render_width -> [tracking thread: Holistic @ tracking_width]
       -> canvas.begin(frame) -> effects draw into light / glow / backdrop, add warps
       -> canvas.composite()  -> cv2.cvtColor RGB -> pygame.image.frombuffer -> blit -> flip
```

`Canvas.composite()` runs these steps: backdrop blend (behind the person mask),
warps (one remap), tint and dim, `+ light`, bloom (light downsampled into glow,
two box-blur scales, soft knee, upsample, add), chromatic aberration, and flash.

## 2. Benchmarks

Run the tool with:

```
.venv/bin/python tools/bench_render.py                    # current fx.py, 3 resolutions
.venv/bin/python tools/bench_render.py --ab path/old_fx.py  # interleaved old-vs-new
.venv/bin/python tools/bench_render.py --quality low --res 640x360
```

Each scenario draws typical content into fresh layers, which is not timed, and
then times only `composite()`. Glow scale is 0.33, OpenCV uses 8 threads, and
values are medians in ms.

| scenario | contents |
|---|---|
| nothing | empty layers |
| light_glow | crescent + 6 lines + 4 sprites + glow (bloom knee active) |
| backdrop_mask | backdrop at 0.85 behind a 256x144 Holistic-like person mask |
| warp1/3/6 | 1/3/6 warps (pinch, bulge, swirl, ripple...), radius 0.3h, person-protected |
| aberration | 6 px |
| everything | all of the above + tint + flash |

### Quiet machine (before any other agents' load), median ms

| scenario | 640x360 before | 640x360 after | 960x540 before | 960x540 after | 1280x720 before | 1280x720 after |
|---|---|---|---|---|---|---|
| nothing | 1.99 | 0.12 | 2.04 | 0.12 | 5.48 | 0.21 |
| light_glow | 2.05 | 0.82 | 2.09 | 1.74 | 5.60 | 2.62 |
| backdrop_mask | 3.00 | 0.78 | 3.36 | 0.81 | 7.74 | 1.46 |
| warp1 | 3.78 | 0.99 | 4.74 | 1.04 | 10.02 | 1.67 |
| warp3 | 4.04 | 1.28 | 5.25 | 1.27 | 10.64 | 2.26 |
| warp6 | 5.07 | 1.61 | 6.40 | 2.59 | 12.65 | 4.18 |
| aberration | 3.20 | 0.50 | 4.01 | 0.95 | 8.63 | 1.58 |
| everything | 7.05 | 3.52 | 11.22 | 6.02 | 20.48 | 11.38 |

### Interleaved A/B under heavy load (load average 10 to 12 from parallel jobs), best of 3 medians

| scenario | 640x360 | 960x540 | 1280x720 |
|---|---|---|---|
| nothing | 2.80 -> 0.07 | 3.01 -> 0.16 | 7.71 -> 0.26 |
| light_glow | 3.22 -> 1.41 | 3.64 -> 3.11 | 8.87 -> 5.03 |
| backdrop_mask | 4.93 -> 1.46 | 5.76 -> 1.64 | 10.90 -> 2.18 |
| warp1 | 5.59 -> 1.46 | 7.35 -> 1.55 | 14.82 -> 2.88 |
| warp3 | 6.26 -> 1.93 | 9.97 -> 3.00 | 18.90 -> 4.82 |
| warp6 | 7.72 -> 3.38 | 10.40 -> 5.18 | 19.90 -> 7.77 |
| aberration | 5.56 -> 1.32 | 7.53 -> 2.67 | 14.33 -> 3.96 |
| everything | 11.65 -> 6.94 | 16.41 -> 9.57 | 31.44 -> 19.07 |

### Quality presets, medians under load

| scenario | low @640 | medium @960 | high @1280 |
|---|---|---|---|
| light_glow | 1.12 | 2.57 | 4.76 |
| warp6 | 1.73 | 3.16 | 7.71 |
| aberration | 0.07 (off) | 1.89 | 3.85 |
| everything | 3.48 | 10.98 | 19.55 |

### Whole demo (`python -m bleach_ar --demo --headless --frames 600`)

- **Before, quiet machine:** 54.5 FPS. Frame time averaged 18.3 ms (p95 20.6) and effects 8.7 ms. The loop is capped at 60 Hz.
- **During this work:** three other agents were running test suites, so the load average stayed between 10 and 12. Full-demo FPS moved between 25 and 41 from run to run in both versions, which makes it useless for an A/B comparison. To isolate the composite, I wrapped `Canvas.composite` with a timer inside the real demo (a monkeypatch around `bleach_ar.__main__.main`) and alternated the old and new fx.py three times:

| run | old composite mean / median / p95 | new composite mean / median / p95 |
|---|---|---|
| 1 | 5.03 / 4.55 / 7.22 | 3.40 / 3.27 / 4.54 |
| 2 | 3.61 / 3.52 / 4.76 | 2.77 / 2.64 / 3.47 |
| 3 | 3.51 / 3.27 / 4.36 | 3.32 / 2.91 / 4.57 |

The demo script mostly draws light and glow (bloom path) and rarely uses
warps, the backdrop or aberration, so its gain is the bloom gain: about 15 to
30%. Domain-style effects that use the backdrop, warps and aberration gain
2 to 5x. Most of the remaining "effects" time in the demo goes to the effect
modules' own drawing code, not the composite.

## 3. What was optimised (fx.py internals only; the public API is unchanged)

| stage | before | after | why |
|---|---|---|---|
| light -> glow downsample | `INTER_AREA` with a fractional ratio at 1280 (1280/3), **2.8 ms** | crop to an exact multiple of k, then the integer `INTER_AREA` path, 0.54 ms | OpenCV's area resize has a fast path only for integer ratios. The crop drops at most k-1 px of light from the bloom. |
| bloom on empty layers | full blur chain always | `cv2.hasNonZero(light)` + `minMaxLoc(glow)` skip it entirely | about 2 ms saved on frames with nothing glowing |
| soft knee | numpy `exp/maximum/minimum` with temporaries, 0.7 to 0.8 ms | skipped unless `max > 170`; otherwise in-place cv2/numpy on a 1-channel view, about 0.3 ms | same formula: `min(x, 255 - 85 exp(-(max(x,170)-170)/85))` |
| tint | `np.full_like(frame, tint)` each frame, **1.8 ms** at 960 | `convertScaleAbs(alpha=1-a)` + `cv2.add(scalar)`, 0.36 ms | off by at most 1 LSB (double rounding) |
| warps | numpy displacement on a full-frame 1/4 grid, two full-res `resize` + two `+FX/FY` adds, remap of the whole frame, person blend of the whole frame | **only the union bounding box of the warps** is evaluated, remapped and blended; each warp is evaluated only on its own grid sub-block; one 2-channel (x,y) map (1 resize + 1 add); the remap writes into a reused buffer | outside a warp's radius (`radius + 3*band` for ripple) the displacement is below 0.01 px, so the old code was copying pixels there anyway |
| backdrop / protect-person blends | `1 - pm` and `(1 - pm) * a` allocated full-res every frame; `frame[:] = blendLinear(...)` (alloc + copy) | `1 - pm` cached once per frame; weights in reused buffers; `blendLinear(dst=frame)` in place | |
| person_mask | 4 allocations per call | `resize/GaussianBlur/clip` into reused buffers; two full-res buffers are ping-ponged, so last frame's mask stays valid | |
| aberration | `warpAffine` on strided channel views (`frame[:,:,ch]`) | `cv2.split` into planar buffers, warp contiguous planes, `cv2.merge(dst=frame)`; **bit-exact** | strided views force hidden copies |
| allocations | about 15 full-res temporaries per frame | none at a fixed resolution (`Canvas._bufs`; verified by `tests/test_fx.py`) | lower GC pressure and fewer page faults |

### Equivalence (checked in tests/test_fx.py against copies of the old code)

- Bloom is bit-exact at 960x540. At 640 and 1280 it differs by at most 4 LSB, mean under 0.1, because of the crop.
- Aberration and backdrop are bit-exact. Tint is within 1 LSB.
- On smooth images, warps have a mean difference under 0.3 LSB and a p99.9 under 8. On white noise they reach a max of 20 LSB at single pixels, mean 0.6. The new coarse grid samples the field at the pixel positions `cv2.resize` maps it to; the old grid was off by half a pixel. Grid density changes slightly as well. Displacement directions and magnitudes are unchanged.

## 4. Reusable graphics helpers (bleach_ar/fx.py)

These helpers draw only through the public Canvas API, so they follow the quality flags too.

```python
depth_scale(value, far, near, s_far=0.6, s_near=1.4) -> float | ndarray
ribbon_polygon(pts, widths) -> (2N, 2) float32
energy_trail(canvas, pts, color, width, strength=1.0, core_color=(255,255,255),
             taper=1.0, glow_width=2.5) -> None
shockwave(canvas, center, radius, thickness, color, strength=1.0, warp=0.0) -> None
lightning(a, b, branches=2, jitter=0.18, rng=None, segments=12,
          branch_length=0.45, depth=1) -> list[(N, 2) float32]   # main bolt first
draw_lightning(canvas, bolts, color, thickness=2, strength=1.0, core_color=(255,255,255))
class Afterimages(count=5, interval=0.035, duration=0.25):
    add(now, polys); render(canvas, now, color, strength=1.0, skip_latest=True, glow=True); clear()
class Starfield(n=300, layers=3, seed=0, twinkle=0.35):
    update(dt, drift=(vx, vy), zoom=0.0, center=(0.5, 0.5)); brightness();
    render(canvas, color=(255,240,230), strength=1.0, target=None, glow=True)
```

Usage notes:

- **`energy_trail`:** pass `pts` tail first. It draws a coloured body, a white-hot core at 35% of the width, and a glow at 2.5x the width. Taper uses a sqrt profile, so the ribbon stays full near the head.
- **`shockwave`:** draws a ring and a glow ring, and adds a `ripple` warp when `warp` is set. Animate `radius` with `ease_out(t)` and `strength` with `1 - t`.
- **`lightning`:** uses midpoint displacement and halves the amplitude at each level, like 1/f fBm octaves. Branches recurse. Re-seed the rng every few frames, for example `rng=np.random.default_rng(int(now*20))`, to get controlled flicker.
- **`Afterimages`:** for flash-step ghosts. Call `add()` each frame with the current outline polygons and `render()` to draw them fading with age squared.
- **`Starfield`:** vectorised: `np.maximum.at` for far-star pixels, AA discs only for near stars, and `np.add.at` glow splats. `drift` gives parallax (near stars move more) and `zoom` gives a hyperspace stream. Draw into `canvas.backdrop()` for Domain environments with `target=bd, glow=False`, or into the light layer.
- **`depth_scale`:** maps any depth cue linearly between `far` and `near`, clamped. The cue can be screen y, MediaPipe z (more negative means nearer, so `far > near` is fine), or shoulder width in px.

## 5. Quality flags (`canvas.quality`)

`Canvas.quality` is a new optional dict that defaults to `None`, which turns every feature on and leaves existing behaviour and tests unchanged. Canvas reads these keys through `canvas.q(key)`, and `fx.QUALITY_DEFAULTS` supplies any missing key:

| key | low | medium | high | effect |
|---|---|---|---|---|
| `bloom` | `"single"` | `"dual"` | `"dual"` | `"single"` drops the tight halo blur |
| `warps` | True | True | True | False skips all `add_warp` distortion |
| `max_warps` | 2 | 4 | 8 | keeps the N warps with the largest `abs(strength) * radius` (0 = unlimited) |
| `aberration` | False | True | True | False ignores `canvas.aberration` |
| `warp_grid` | 8 | 6 | 4 | warp field evaluated every N px; bigger is cheaper and blurrier |

These keys live in `config.py` under `quality_presets`, next to `render_width` and the other preset keys.

**App wiring (for the lead, in app.py):** set it wherever the preset is applied:

```python
# __init__, right after: self.canvas = Canvas(preset["glow_scale"], eff["intensity"])
self.canvas.quality = preset
# _apply_quality(), next to: self.canvas.glow_scale = preset["glow_scale"]
self.canvas.quality = preset
```

The whole preset dict can be passed because Canvas ignores keys it doesn't know. Effects can read the same flags, for example `ctx.canvas.q("max_warps")`, to skip work they know will be dropped.

## 6. Segmentation mask quality (tracking.py): investigated, not changed

Test input: the OpenCV sample `messi5.jpg`, a full-body athlete against a busy
crowd, letterboxed to 960x540. It went through `Tracker.process` 40 times with
±1 px jitter and sensor noise. Holistic complexity 0 at tracking width 640
took about 36 ms per inference under load.

| variant | cost | 0.1<m<0.9 band | edge alignment\* | flicker mean / p99 |
|---|---|---|---|---|
| Holistic raw (current) | 0 | 3.6% | 156.5 | 0.0060 / 0.210 |
| fast guided filter (1/2 res, r=4, eps=2e-3) | 1.74 ms | 9.1% | 158.7 | 0.0049 / 0.099 |
| fast guided filter (1/4 res, r=2, eps=5e-3) | 0.78 ms | 10.4% | 154.3 | 0.0047 / 0.092 |
| morphological open 5x5 | 0.26 ms | 3.6% | 158.0 | 0.0060 / 0.207 |

\*Edge alignment is the image-gradient magnitude averaged along the mask's
gradient. Higher means the mask edge sits on real image edges.

Findings:

- **No change made.** Guided or joint-bilateral refinement roughly halves p99 edge flicker but **widens** the soft band about 2.5x. In a backdrop composite, which I checked visually, this gives *more* visible halo: lighter fringes around the hair and arms. The half-res variant also exceeds the 1.5 ms budget. Morphological cleanup changes nothing measurable.
- **What already handles halo and flicker.** `Canvas.person_mask()` already applies 0.55/0.45 EMA temporal smoothing, a 1.2 px feather, and a `(m - 0.25) * 2` firm-up curve. Together these give tight, stable edges on Holistic's mask. Holistic also runs with `smooth_segmentation=True`.
- **Tasks `ImageSegmenter`** (selfie_segmenter.tflite, about 250 KB, VIDEO mode) measured **7.2 ms median, 16 ms p95 per frame** on this M1 at 640x360. That is a *second* inference on top of Holistic on the same tracking thread. Its mask was visibly worse on a full-body shot: blocky 256x256 staircase edges, background blobs at the feet, and the arm lost. The model is designed for near-field selfies. It is not worth the cost, so I did not switch.
- **Where to look if halos appear on real webcam footage.** Raise the firm-up threshold in `person_mask()` (0.25 to 0.3), or set the backdrop over a slightly eroded mask. Both cost nothing.

## 7. ModernGL / GPU compositing evaluation: not integrated

I installed `moderngl` (with `glcontext` 3.1.0) temporarily, measured it, then
uninstalled it again; `requirements.txt` is unchanged. Setup: a standalone
context reporting **OpenGL 4.1 Metal - 90.5, Apple M1**, 960x540. Each frame:
upload the BGR frame, the uint8 light layer and the float glow layer; run 5
passes (glow+light downsample, two separable 25-tap blur passes, then one
composite pass with 6 warps, aberration, soft knee and add); finish; read the
pixels back. Median over 200 frames, under load:

| step | median | p95 |
|---|---|---|
| upload (frame + light + glow) | 2.13 ms | 3.89 |
| draw (5 passes, incl. `finish`) | 3.17 ms | 3.60 |
| readback (`read_into`) | 1.96 ms | 2.93 |
| **total** | **7.47 ms** | 9.45 |
| OpenCV composite "everything" (after this work, same machine, same load) | 6.0 to 9.6 ms | |

Decision: **not worth it now.**

1. **Transfers alone cost about 4 ms**, roughly as much as the whole optimised CPU composite of a typical frame (1.5 to 3 ms in the demo). The effects draw on the CPU with cv2 (fillPoly, AA lines, particles), so the light and glow layers have to be uploaded every frame. The display path is `pygame.image.frombuffer` + blit (SDL software), so the result also has to come back.
2. **macOS caps OpenGL at 4.1 and has deprecated it.** It runs as a translation layer over Metal and could disappear in a future macOS. It also adds a native dependency and failure mode for users.
3. **The CPU path is still cheap at the target resolution.** After the optimisations, the composite at 960x540 costs under 3 ms in typical frames, which is not the bottleneck. Holistic inference (25 to 36 ms on its thread) and the effect modules' drawing are.

**Path if it is ever needed.** Make the display a GL window (`pygame.display.set_mode(..., OPENGL | DOUBLEBUF)`) so there is no readback. Upload the camera frame once and move effect drawing to GL primitives: instanced quads for particles, triangle strips for ribbons. Port the HUD/menus from pygame surfaces to textures. That is a renderer rewrite, not a backend switch, and it only pays off at 1280x720+ or with many full-screen warps. Any GL backend must fall back to the OpenCV composite automatically when context creation fails.

## 8. Sources and techniques used

- **The Book of Shaders** (thebookofshaders.com): *Shaping functions*, for the exponential soft knee (`255 - 85 e^{-(x-170)/85}`) and the sqrt taper profile in `energy_trail`; *Noise / fBm*, where halving the amplitude per octave became the per-level halving in `lightning`'s midpoint displacement; *Shapes / gradients*, for the Gaussian ring envelope of the `ripple` warp and `shockwave`.
- **ModernGL docs** (moderngl.readthedocs.io): `create_standalone_context`, `Texture.write`, `Framebuffer.read_into` and the alignment=1 note for 3-channel textures, used in the benchmark in section 7.
- **pygame docs** (pygame.org/docs): `image.frombuffer` (zero-copy wrap) and the blend flags. These confirmed that the display path needs a CPU buffer, which drove the GPU decision.
- **Shadertoy:** inspiration only, no code copied. I took the "hyperspace starfield" look (radial streaming, near layers faster) for `Starfield(zoom=...)`.
- **He et al., "Guided Image Filtering"** (fast variant: He & Sun 2015): the refinement I evaluated and rejected in section 6.
