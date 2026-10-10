# pointimg — Technical Documentation

A pointillist filter that decomposes an image into solid-colored dots of varying sizes.
Each dot takes the average color of its zone and its radius is modulated by local luminance
and variance.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Launch](#2-launch)
3. [Parameters (`FilterParams`)](#3-parameters-filterparams)
4. [Processing Pipeline](#4-processing-pipeline)
5. [Density Map](#5-density-map)
6. [Radius Calculation (`radius_for_dot`)](#6-radius-calculation-radius_for_dot)
7. [Placement Algorithms](#7-placement-algorithms)
8. [Spatial Acceleration (`SeedGrid`)](#8-spatial-acceleration-seedgrid)
9. [Rendering (draw order)](#9-rendering-draw-order)
10. [GUI Architecture](#10-gui-architecture)
    - [10.1 UI decisions](#101-ui-decisions-shared-by-both-front-ends)
    - [10.1.1 What is *not* shared](#1011-what-is-not-shared-and-why)
    - [10.2 egui](#102-egui-srcgui)
    - [10.3 GTK4 / libadwaita](#103-gtk4--libadwaita-srcgtk)
    - [10.4 Dithering](#104-dithering-srcfilterditherrs)
11. [Export Size](#11-export-size)
12. [SVG Export](#12-svg-export)
13. [Quality Tools](#13-quality-tools)

---

## 1. Project Structure

```
pointimg/
├── Cargo.toml          — dependencies, three binaries + a lib
├── src/
│   ├── lib.rs          — crate root; exposes `pub mod filter`, `color`, `frontend`
│   ├── color.rs        — ICC-aware decoding (embedded/`--input-profile`) and
│   │                     output-profile encoding (moxcms)
│   ├── frontend.rs     — helpers shared by the egui and GTK4 front-ends
│   │                     (formatting, output paths, atomic writes, presets)
│   ├── filter/         — all logic, split by concern
│   │   ├── mod.rs      — module root: public API (apply, apply_with_progress,
│   │   │               compute_dots), re-exports, unit tests
│   │   ├── params.rs   — Algorithm, DotShape, FilterParams, Dot, validate_params,
│   │   │               to_toml_string / from_toml_str (presets)
│   │   ├── density.rs  — compute_density_map / compute_density_image (SAT)
│   │   ├── render.rs   — render, draw_dot, anti-aliasing (4×4 supersampling),
│   │   │               point_in_regular_polygon, quantize_dots/quantize_palette_centers,
│   │   │               radius_for_dot, render_halftone (ink multiply compositing)
│   │   ├── dither.rs   — floyd_steinberg (post-render palette quantization)
│   │   ├── halftone.rs — HalftoneMode (Cmyk/Dominant), Screening (Am/Fm),
│   │   │               separate_cmyk / separate_dominant / screen_am / screen_fm
│   │   ├── gamma.rs    — sRGB ⇄ linéaire LUTs, image wrappers (gamma_correct)
│   │   ├── svg.rs      — render_svg_from_dots / render_svg / render_svg_dynamic
│   │   ├── seedgrid.rs — SeedGrid spatial acceleration
│   │   ├── sampling.rs — importance_sample, make_rng_seed, lcg_next,
│   │   │               nearest_neighbor_radii, build_dots_from_seeds
│   │   ├── util.rs     — luminance, pixel_sum, pixel_variance, flatten_to_rgb,
│   │   │               estimate_memory_bytes(_for), resize_to_limits
│   │   ├── gpu.rs      — optional WGSL compute density pass (`gpu` feature)
│   │   └── algorithms/ — placement algorithms, one file each
│   │       ├── grid.rs     — dots_grid
│   │       ├── kmeans.rs   — dots_kmeans_progressive, compute_dots_kmeans,
│   │       │               dots_from_kmeans_centers
│   │       ├── quadtree.rs — dots_quadtree, subdivide
│   │       └── voronoi.rs  — dots_voronoi_progressive, compute_dots_voronoi
│   ├── main.rs         — CLI binary (`pointimg`)
│   ├── gui/            — egui/wgpu front-end (`pointimg-gui`, feature `gui`)
│   │   ├── main.rs     — eframe entry point, `argv`, window icon
│   │   ├── app.rs      — application state and worker generations
│   │   ├── compute.rs  — background compute/preview orchestration
│   │   ├── ui.rs       — panels, controls, history
│   │   ├── palette.rs  — command palette (`Ctrl+K`)
│   │   ├── notify.rs   — toast stack (errors, confirmations)
│   │   ├── ranges.rs   — test: UI bounds vs `validate_params`
│   │   ├── sliders.rs  — slider widgets and display precision
│   │   ├── prefs.rs    — interface scale and theme
│   │   ├── io.rs       — open/load helpers
│   │   ├── save.rs     — multi-format export
│   │   ├── convert.rs  — texture conversion
│   │   └── format.rs   — display formatting
│   └── gtk/            — GTK4/libadwaita front-end (`pointimg-gtk`, feature `gtk`)
│       ├── main.rs     — application bootstrap, `HANDLES_OPEN` / `argv`
│       ├── actions.rs  — GActions and keyboard shortcuts
│       ├── controls.rs — parameter widgets
│       ├── sections.rs — panel sections
│       ├── preview.rs  — GDK texture preview + zoom
│       ├── recent.rs   — `GtkRecentManager` integration
│       ├── ranges.rs   — test: UI bounds vs `validate_params`
│       ├── shortcuts.rs — keyboard shortcuts window
│       ├── compare.rs  — before/after wipe
│       ├── files.rs    — file dialogs and drag & drop
│       ├── wire.rs     — actions, menu models, signal wiring
│       └── ui.rs       — window composition, breakpoint, recent images
├── assets/
│   ├── logo/           — icon master (`pointimg.svg`) + the 256 px PNG
│   │                     embedded in the egui binary
│   ├── presets/        — one thumbnail per built-in preset, generated
│   │                     (`cargo run --example preset_thumbnails`)
│   └── examples/       — sample images referenced by the empty state
├── examples/
│   └── preset_thumbnails.rs — renders the preset thumbnails
├── packaging/
│   ├── pointimg-gui.desktop, pointimg-gtk.desktop — entries, `%F` so the
│   │                     file manager can open an image in either front-end
│   └── icons/          — `hicolor` icon tree, generated from the SVG by
│                         `icons/regenerate.sh`
├── tests/              — integration tests (public API end-to-end)
│   ├── pipeline.rs     — algorithm dimensions, progress, dots, flatten, density
│   ├── svg.rs          — SVG header, shapes, bg color, empty-image error
│   ├── reproducibility.rs — same seed → identical, different seed → diverge
│   ├── palette.rs      — palette quantization reduces SVG fill colors
│   ├── halftone.rs     — CMYK rosette, dominant colors, FM screening, TOML presets
│   └── gpu_density.rs  — CPU/GPU density parity (`gpu` feature)
├── benches/
│   └── filter.rs       — criterion benches (apply per algo, density, svg)
└── fuzz/               — cargo-fuzz targets (image, preset) + versioned corpus
```

| Crate             | Role                                                             |
| ----------------- | ---------------------------------------------------------------- |
| `image 0.25`      | Loading / saving / manipulating `RgbImage`                       |
| `moxcms 0.9`      | Pure-Rust ICC transforms (sRGB working space)                    |
| `clap 4`          | CLI argument parsing                                             |
| `rayon 1`         | Parallel iteration (density map, Lloyd, k-means)                 |
| `anyhow 1`        | Error handling (`Result`, `anyhow!`)                             |
| `log` / `env_logger` | Logging (`-v`/`-vv`/`-q`, `RUST_LOG`)                         |
| `eframe 0.36`     | egui framework (wgpu backend) — *feature-gated* `gui`            |
| `egui 0.36`       | Immediate-mode GUI widgets — *feature-gated* `gui`               |
| `wgpu 30`         | GPU backend (Vulkan + GL fallback) — *feature-gated* `gui`/`gpu` |
| `rfd 0.17`        | Native file dialogs — *feature-gated* `gui`                      |
| `gtk4 0.11`       | GTK4 bindings — *feature-gated* `gtk`                            |
| `libadwaita 0.9`  | libadwaita widgets — *feature-gated* `gtk`                       |
| `async-channel 2` | Worker → UI messaging (GTK) — *feature-gated* `gtk`              |
| `serde 1`         | Serialization framework (`FilterParams` derive)                  |
| `toml 1`          | TOML preset files (`--preset` / `--save-preset`)                 |
| `glob 0.3`        | Glob expansion for batch `--input` (`*`, `?`, `[`)               |
| `criterion 0.8`   | Benchmarks — *dev-dependency*                                    |


> **Optional features:** `avif` (pulls `image/avif`; AVIF encoder via `ravif`).
> `gpu` (pulls `wgpu`, `pollster`, `bytemuck`; opt-in at runtime with
> `POINTIMG_GPU=1`). WebP is included in `image`'s default formats.

### Optional GPU density pass

When compiled with `--features gpu` and `POINTIMG_GPU=1`,
`compute_density_map` dispatches a WGSL compute shader with one invocation per
pixel. It computes the 9×9 local RGB variance directly in parallel, then the
CPU performs the same global normalization curve as the SAT path. This is a
real GPU compute path, not a CPU thread wrapper.

The shader uses centered deltas (`x - reference_pixel`) rather than
`E[x²] - E[x]²` on raw 8-bit values to avoid f32 cancellation on uniform
mid-gray images. A global mutex serializes device creation/readback because
some native drivers are not safe when several test threads create devices at
once. The default remains the exact CPU summed-area-table implementation:
without `POINTIMG_GPU=1`, or when no adapter/device/readback is available, the
function returns to the CPU path automatically.

> **Feature-gating:** GUI dependencies are behind the `gui` feature
> (enabled by default). To compile CLI-only: `cargo build --no-default-features`.
> **MSRV:** Rust 1.95 (declared via `rust-version` in `Cargo.toml`,
> enforced by the `msrv` CI job).

---

## 2. Launch

### CLI

```bash
# Voronoi, 1500 points, white background
pointimg -i photo.jpg -o result.png --algorithm voronoi --num-points 1500

# Grid 60 columns, black background
pointimg -i photo.jpg -o result.png --algorithm grid --cols 60 --bg black

# Custom hex background
pointimg -i photo.jpg -o result.png --bg "#1a1a2e"

# Quadtree, high variance sensitivity
pointimg -i photo.jpg -o result.png --algorithm quadtree --variance-sensitivity 0.9

# Square shape
pointimg -i photo.jpg -o result.png --shape square

# Hexagonal polygon
pointimg -i photo.jpg -o result.png --shape polygon --polygon-sides 6

# Ellipse with rotation
pointimg -i photo.jpg -o result.png --shape ellipse --ellipse-aspect 2.0 --ellipse-angle 45

# Transparent background (RGBA output, alpha 0 between dots)
pointimg -i photo.jpg -o result.png --bg transparent

# Gamma correction (averages in linear color space)
pointimg -i photo.jpg -o result.png --gamma

# Palette 8 colors + Floyd-Steinberg dithering
pointimg -i photo.jpg -o result.png --palette 8 --dithering

# Save/recall preset TOML
pointimg --save-preset out.toml --algorithm voronoi --num-points 1500 --gamma
pointimg --preset out.toml -i photo.jpg -o result.png

# Batch processing (glob or directory) + per-file pattern
pointimg -i "photos/*.jpg" -o "out/{stem}_{n}.png" --algorithm grid --cols 50

# Fast preview (downscale source before pipeline)
pointimg -i big.jpg -o preview.png --preview 300x300 --algorithm voronoi

# Halftone angled screen (Grid only)
pointimg -i photo.jpg -o result.png --algorithm grid --grid-angle 30

# Output formats (extension-driven)
pointimg -i photo.jpg -o result.webp   # webp (default feature)
cargo build --release --features avif   # opt-in AVIF
pointimg -i photo.jpg -o result.avif

# Halftone rosette (CMYK 4 canaux, angles 15°/75°/0°/45°)
pointimg -i photo.jpg -o rosette.png --halftone cmyk --halftone-freq 60

# Couleurs dominantes (k-means) — 6 canaux, angles répartis par section d'or
pointimg -i photo.jpg -o dominants.png --halftone dominant-6 --halftone-freq 80

# FM screening stochastique (blue noise) au lieu de AM (grille rotationée)
pointimg -i photo.jpg -o stochastic.png --halftone cmyk --screening fm
```

**All flags:**

| Flag                     | Short | Default      | Description                                                                                         |
| ------------------------ | ----- | ------------ | --------------------------------------------------------------------------------------------------- |
| `--input`                | `-i`  | required     | Source image, glob, or directory. Optional if `--save-preset` alone.                                |
| `--output`               | `-o`  | `output.png` | Output file. Patterns: `{n}` (index), `{stem}` (source name sans ext), `{name}` (source full name). |
| `--algorithm`            | `-a`  | `voronoi`    | `grid` \| `kmeans` \| `voronoi` \| `quadtree`                                                       |
| `--num-points`           | `-n`  | `800`        | Number of points (kmeans/voronoi/quadtree)                                                          |
| `--cols`                 | `-c`  | `80`         | Grid columns (grid only)                                                                            |
| `--min-radius`           |       | `0.003`      | Min radius = fraction of `min(W,H)`                                                                 |
| `--max-radius`           |       | `0.06`       | Max radius = fraction of `min(W,H)`                                                                 |
| `--bg`                   | `-b`  | `white`      | `white`, `black`, `transparent`/`none`, or `#rrggbb`                                                |
| `--shape`                |       | `circle`     | `circle` \| `square` \| `ellipse` \| `polygon`                                                      |
| `--ellipse-aspect`       |       | `1.5`        | Width/height ratio (ellipse)                                                                        |
| `--ellipse-angle`        |       | `0.0`        | Rotation angle in degrees (ellipse)                                                                 |
| `--polygon-sides`        |       | `6`          | Number of sides (polygon, 3-12)                                                                     |
| `--iterations`           |       | `10`         | Lloyd / k-means iterations                                                                          |
| `--variance-sensitivity` |       | `0.7`        | Variance redistribution strength `[0,1]`                                                            |
| `--max-boost`            |       | `2.5`        | Max multiplier in uniform zones                                                                     |
| `--seed`                 |       | *(random)*   | RNG seed for exact reproduction                                                                     |
| `--palette`              |       | *(disabled)* | Number of colors in reduced palette                                                                 |
| `--svg`                  |       | *(disabled)* | Also export SVG (same path, `.svg` extension)                                                       |
| `--gamma`                |       | *(disabled)* | Gamma-correct perceptual averages (linear space)                                                    |
| `--dithering`            |       | *(disabled)* | Floyd-Steinberg dithering on rendered image                                                         |
| `--preset`               |       | *(disabled)* | Load `FilterParams` from a TOML preset file                                                         |
| `--save-preset`          |       | *(disabled)* | Save current params to a TOML preset file and exit (if `--input` absent)                            |
| `--verbose`              | `-v`  | *(0)*        | `-v` = info, `-vv` = debug. Overridable via `RUST_LOG`.                                             |
| `--quiet`                | `-q`  | *(disabled)* | Suppress info messages (warnings only)                                                              |
| `--preview`              |       | *(disabled)* | Downscale source to `WxH` (max, preserves aspect) before pipeline                                   |
| `--grid-angle`           |       | `0.0`        | Rotation of grid placement in degrees (Grid only, halftone screen effect)                           |
| `--halftone`             |       | `off`        | `off` \| `cmyk` (4-channel rosette) \| `dominant-N` (N colors via k-means)                          |
| `--screening`            |       | `am`         | `am` (rotated-grid rosette) \| `fm` (stochastic blue noise)                                         |
| `--halftone-freq`        |       | `60.0`       | AM cells per `min(W,H)` (≈ dot frequency in lpp at 1 dpi)                                           |
| `--halftone-min-radius`  |       | `0.002`      | Min radius per ink dot (fraction of `min(W,H)`)                                                     |
| `--halftone-max-dot`     |       | `0.85`       | Max radius per ink dot (fraction of trame step)                                                     |

**Output formats (driven by file extension):** PNG, JPEG, BMP, TIFF, WebP by default.
AVIF requires `cargo build --features avif` (pulls in `ravif`). Transparent PNG
(`--bg transparent`) produces RGBA; SVG omits the background `<rect>` when
`transparent` is set.

### Halftone multi-canal (rosette)

L'algorithme `--halftone` active un pipeline alternatif (bypass `algorithm`):

1. **Séparation en N canaux d'encre** — `Cmyk` produit Cyan=`[0,255,255]`,
   Magenta=`[255,0,255]`, Yellow=`[255,255,0]`, Black=`[0,0,0]` à angles
   `15° / 75° / 0° / 45°`. `Dominant { n, .. }` lance un k-means sur les pixels
   source pour trouver N couleurs et distribue les angles à `base + i × 180°/n`.
2. **Coverage par canal** — pour chaque pixel, l'encre i reçoit un coverage ∈ [0,1]
   (formule CMYK standard + UCR pour CMYK ; similarité au centre le plus proche
   pour dominant).
3. **Screening** — `Am` place une grille rotationnée à l'angle du canal et émet
   un dot par intersection, rayon proportionnel au coverage moyenné dans un disque
   de rayon `step/2`. `Fm` accepte stochastiquement chaque dot de la grille avec
   probabilité = coverage (densité variable, rayon quasi constant).
4. **Compositeur multiply** — `render_halftone` part d'un papier blanc (ou
   `bg_color` si non-transparent) et applique chaque dot en mode multiply
   (transmission mask) — soustractif, commutatif: Magenta + Jaune = Rouge.
   `blend_multiply_coverage` rasterise un dot anti-aliasé (AA 4×4) et multiplie
   son mask par la couverture alpha.

Les canaux K/C/M/Y sont totalement indépendantes de `algorithm` (Grid, Voronoi,
etc.) — `--halftone` override complètement le pipeline historique.

### GUI

```bash
pointimg-gui
```

Left panel: all parameters with sliders + algorithm selector.
Right panel: side-by-side source / result preview.
Auto-recalculate on every parameter change.

---

## 3. Parameters (`FilterParams`)

```rust
pub struct FilterParams {
    pub algorithm: Algorithm,          // Voronoi by default
    pub num_points: usize,             // 800
    pub cols: u32,                     // 80 (Grid only)
    pub min_radius_ratio: f32,         // 0.003
    pub max_radius_ratio: f32,         // 0.06
    pub bg_color: [u8; 3],             // [255,255,255]
    pub iterations: usize,             // 10
    pub variance_sensitivity: f32,     // 0.7
    pub max_boost: f32,                // 2.5
    pub rng_seed: Option<u64>,         // None = system clock
    pub palette_size: Option<usize>,   // None = all colors
    pub dot_shape: DotShape,           // Circle by default
    pub transparent: bool,             // RGBA transparent background output
    pub gamma_correct: bool,           // averages in linear space
    pub dithering: bool,               // Floyd-Steinberg on rendered image
    pub grid_angle_deg: f32,           // Rotation angle of Grid placement (halftone screen)
    pub halftone: HalftoneMode,        // Off | Cmyk{angles} | Dominant{n, base_angle_deg}
    pub screening: Screening,          // Am (grid rotation) | Fm (stochastic blue noise)
    pub halftone_frequency: f32,       // AM cells per min(W,H)
    pub halftone_min_radius_ratio: f32,
    pub halftone_max_dot_ratio: f32,
}
```

`is_bg_light()` — derived method from `bg_color` (perceptual luminance > 127.5). Faux si `transparent`.

**`min_radius_ratio` / `max_radius_ratio`:** fractions of `min(width, height)` of the image.
On an 800×600 image, `0.003` → 1.8 px and `0.06` → 36 px.

**`variance_sensitivity`** affects three places:
- The density map (blend between uniform and pure variance)
- Lloyd weights exponent (`power = 1 + sensitivity × 3`)
- Radius boost in flat zones

**`max_boost`** caps the radius multiplier in uniform zones.
`1.0` = no boost. `2.5` = a point in a totally flat zone can be 2.5× `r_max`.

**Validation (`validate_params`):** all parameters are checked before any
work starts — `min_radius_ratio > 0`, `max >= min`, `max <= 1.0`,
`num_points > 0` and `≤ 100000`, `cols > 0` and `≤ 8192`,
`iterations ≤ 100`, `palette_size ∈ [2, 256]` (if set),
`variance_sensitivity ∈ [0, 1]`, `max_boost >= 1.0`, polygon `sides ∈ 3..=12`,
ellipse `aspect ∈ (0, 10]`, halftone frequency `> 0`, and image dimensions
`≤ 65535` and `≤ 8M` pixels. An out-of-range value returns `Err` instead of
panicking or silently misbehaving (e.g. `cols = 0` previously panicked on
division).

### Preset format

Presets written by `to_toml_string()` use an explicit version envelope:

```toml
preset_version = 1

[params]
algorithm = "Voronoi"
num_points = 800
```

The reader still accepts the pre-versioned flat format for migration. Unknown
versions are rejected so that a future schema cannot silently change the meaning
of a parameter.

---

## 4. Processing Pipeline

```
src: DynamicImage (JPEG, PNG, RGBA…)
      │
      ▼ flatten_to_rgb()  (alpha composition on bg_color if RGBA)
      │
      ▼
compute_density_map()   →   density: Vec<f32>   (one float per pixel, rayon parallel)
      │
      ├─ Grid      → dots_grid()
      ├─ Kmeans    → dots_kmeans_progressive()   (apply_with_progress → callback per iteration)
      ├─ Voronoi   → dots_voronoi_progressive()  (apply_with_progress → callback per iteration)
      └─ Quadtree  → dots_quadtree()
                              │
                              ▼ quantize_dots() (optional, palette_size Some(k))
                              │
                              ▼
                          render()   →   dst: RgbImage
                              │
                              ▼ render_svg_from_dots() (optional, SVG export)
```

**`apply_with_progress`** returns `(RgbImage, Vec<Dot>)` — the dots match
exactly the rendered image, eliminating double-calculation and ensuring PNG/SVG
consistency. For Voronoi and K-means, the last iteration of the progressive
loop returns its dots directly without re-rendering the image (eliminating double rendering).

**SVG-only path (`compute_dots`):** when only dots are needed (SVG export
without PNG), the `compute_dots()` function calculates dots without rendering the PNG image.
Internally, it dispatches to `compute_dots_voronoi()` or `compute_dots_kmeans()`
which run the Lloyd/K-means iterations and return dots directly.

```
compute_dots()
  ├─ Grid / Quadtree  → apply_with_progress() (classic rendering, dots extracted)
  ├─ Voronoi          → compute_dots_voronoi() (Lloyd iterations without PNG rendering)
  └─ Kmeans           → compute_dots_kmeans()  (K-means iterations without PNG rendering)
```

---

## 5. Density Map

**File:** `filter.rs` — `compute_density_map()`

**Purpose:** assign to each pixel a `[0,1]` value indicating its detail level.
Value close to `1` = textured/contrasted zone. Close to `0` = flat/uniform zone.

**Algorithm (Summed-Area Table):**

The calculation uses a **summed-area table** (SAT) to get neighborhood
statistics in O(1) per pixel, instead of O(81) for the naive 9×9 loop approach.

**Phase 1 — SAT Construction:**

Six prefix-sum tables of dimensions `(w+1)×(h+1)` are built (zero-padded):
- `sum_r, sum_g, sum_b` — sum of channel values
- `sq_r, sq_g, sq_b` — sum of squares (for variance calculation)

Each table is filled in one pass with the classic SAT formula:
```
SAT[y][x] = val + SAT[y-1][x] + SAT[y][x-1] − SAT[y-1][x-1]
```

**Phase 2 — Per-Pixel Query:**

For each pixel `(px, py)`, the 9×9 window (radius 4, clamped edges) is queried
in O(1) via 4 reads in each SAT:
```
sum = SAT[y2][x2] − SAT[y1][x2] − SAT[y2][x1] + SAT[y1][x1]
```

1. Calculate variance per R, G, B channel:
   ```
   var_c = sum_sq_c/n  −  (sum_c/n)²
   ```
2. Average the three variances:
   ```
   raw = (var_R + var_G + var_B) / 3
   ```
3. Normalize across entire image (`max_var = max(all raw, 1e-6)`):
   ```
   norm = sqrt(raw / max_var)       ∈ [0, 1]
   ```
   The square root softens the curve (avoids only extreme edge pixels
   having high density).

   > **Note:** `max_var` is clamped to `1e-6` (not `1.0`) to preserve
   > contrast on nearly-uniform images. On a solid image, `max_var=0`
   > → epsilon → `norm=0` → `density=0.3` (with `sensitivity=0.7`), which is correct.

4. Blend with `variance_sensitivity`:
   ```
   density[px] = 1 - sensitivity × (1 - norm)
   ```
   - `sensitivity=0` → everything is `1.0` (uniform distribution, variance ignored)
   - `sensitivity=1` → `density = norm` (maximum redistribution)

**Complexity:** O(W×H) for SAT construction + O(W×H) for queries.
**Parallelism:** row-major queries via `rayon::par_iter` (SATs are read-only).

---

## 6. Radius Calculation (`radius_for_dot`)

```text
fn radius_for_dot(lum: f32, local_density: f32, img_min_side: f32, params: &FilterParams) -> f32
```

**Step 1 — Convert to pixels:**
```
r_min = min_radius_ratio × img_min_side
r_max = max_radius_ratio × img_min_side
```

**Step 2 — Halftone modulation (luminance → size):**
```
r_lum = r_max − (r_max − r_min) × lum
```
- Dark pixel (`lum → 0`) → radius close to `r_max` (large dot)
- Light pixel (`lum → 1`) → radius close to `r_min` (small dot)

Reproduces the classic halftone screen effect where dark zones have larger dots.

**Step 3 — Uniformity boost:**
```
uniformity = 1 − local_density          ∈ [0, 1]
boost = 1 + uniformity × variance_sensitivity × (max_boost − 1)
result = min(r_lum × boost,  max_boost × r_max)
```
- Detailed zone (`local_density → 1`): `boost ≈ 1`, no enlargement
- Flat zone (`local_density → 0`): `boost` can reach `max_boost`
- The `min(…, max_boost × r_max)` prevents dots from becoming unlimited

**NN Cap (Voronoi / K-means):**  
In `build_dots_from_seeds`, the radius is additionally capped to
`0.8 × (half-distance to nearest neighbor)` to guarantee visible spacing
between dots even in sparse zones.

---

## 7. Placement Algorithms

### 7.1 Grid (`Grid`)

**Complexity:** O(W×H)

1. Divide the image into square cells of size `cell = W / cols`.
2. For each cell: calculate average color and average density.
3. Emit a point at the cell center.
4. Radius capped to `cell/2 × 0.8` (80% of half-cell).

**Characteristic:** fixed and regular spatial distribution. Only radius varies.
Fast, useful as a reference.

---

### 7.2 Spatial K-means (`Kmeans`)

**Complexity:** O(iterations × W×H×candidates) — the spatial `SeedGrid`
variant `nearest_by` makes the per-pixel candidate count ≪ k, and the
fixed-point early stop skips redundant iterations.

1. **Initialization:** `k` seeds via `importance_sample` (biased towards detailed zones).
   Seeds are placed with a random **sub-pixel jitter** of ±0.5 pixel
   to avoid clustering on pixel centers.
2. Represent each pixel as a normalized 5D vector `[x/W, y/H, r/255, g/255, b/255]`.
   The normalized positions/colors are precomputed once (LUTs) instead of
   being re-divided per pixel per iteration.
3. **Iterations:**
   - Assign each pixel to the nearest center (5D Euclidean distance) via
     `SeedGrid::nearest_by` — the spatial squared distance is a lower bound
     of the total squared distance, so rings beyond `best²` are skipped.
     Ties resolve to the smallest index, exactly like `Iterator::min_by`.
   - Partial sums accumulate in **fixed row chunks** combined in index
     order, so the f64 reduction is deterministic (no work-stealing
     dependence, unlike the previous `fold/reduce`).
   - Recalculate each center as the mean of its assigned pixels.
4. Emit one point per surviving center.

**Early stopping (convergence):** the loop stops when all centers are
bitwise unchanged (exact fixed point): any further iteration would be an
exact no-op, so the final result is unchanged.

**Double rendering elimination:** on the last iteration, dots are built
and returned directly without re-rendering the complete image.

**Characteristic:** groups by both spatial proximity and color similarity.
Can produce non-convex clusters. No NN cap applied.

---

### 7.3 Voronoi / Weighted Lloyd (`Voronoi`) ← default algorithm

**Complexity:** O(iterations × W×H) thanks to `SeedGrid` — see §8

1. **Initialization:** `k` seeds via `importance_sample`.
   Seeds are placed with a random **sub-pixel jitter** of ±0.5 pixel
   to reduce clustering on pixel centers.
2. **Lloyd weights:** `w[pixel] = density[pixel] ^ power`
   with `power = 1 + variance_sensitivity × 3` ∈ [1, 4].
   Detailed pixels attract seeds more strongly.
3. **Weighted Lloyd iterations:**
   - Build a `SeedGrid` from current positions.
   - For each pixel, find the nearest seed via the grid.
   - Accumulate: `sum_x[best] += fx × w`, `sum_y[best] += fy × w`, `sum_w[best] += w`.
   - Update each seed: `seed = (sum_x/sum_w, sum_y/sum_w)`.
4. **Point construction:** `build_dots_from_seeds` — average color of Voronoi cell
   + NN cap on radius.

**Characteristic:** seeds converge towards weighted barycenters of their cells.
Detailed zones → many small dots. Flat zones → few large dots.

**Double rendering elimination:** on the last iteration, dots are built
and returned directly without re-rendering the complete image.

---

### 7.4 Adaptive Quadtree (`Quadtree`)

**Complexity:** O(W×H) to build the integral images + O(1) per node query —
integer u64 **summed-area tables** (sums and sum-of-squares per RGB channel,
same technique as the density map) replace the old O(area) per-node rescans.

**Internal parameters:**
```
min_cell  = max(sqrt(W×H / num_points) / 2, 2)   pixels
threshold = 800 × (1 − variance_sensitivity × 0.8)
```

**Recursive subdivision (`subdivide`):**

```
subdivide(cell [x,y,w,h]):
  if w < 2 or h < 2 → stop
  calculate average_color and variance of cell via the SATs
  (exact integer expansion: Σ(p−a)² = Σp² − 2a·Σp + n·a²)
  if variance < threshold  OR  size ≤ min_cell:
      emit a point at center
      local_density = min(min(w,h) / img_min, 1.0)
      radius = radius_for_dot(lum, local_density, img_min, params)
  else:
      subdivide into 4 quadrants and recurse
```

`local_density` is inferred from the cell's relative size:
a large cell (flat zone) → low `local_density` → large dot.

**Characteristic:** no explicit density map. Adapts directly to local contrast.
Flat zones → one cell, one large dot. Detailed zones → tiny cells,
dense small dots.

---

## 8. Spatial Acceleration (`SeedGrid`)

**Problem solved:** naive nearest seed search is O(k) per pixel.
With 1500 seeds and an 800×600 image, that's 720 000 × 1500 = **1.08 billion**
comparisons per Lloyd iteration.

**Principle:** 2D hash grid. Each cell contains the list of seeds that fall in it.
The search is done only in neighboring cells.

**Construction:**
```
cell_area = (W × H / k) × 4         ≈ 4 seeds per cell
cell_size = sqrt(cell_area)
cols = ceil(W / cell_size)
rows = ceil(H / cell_size)
```
Each seed is inserted into the cell corresponding to its coordinates.

**Query nearest(`fx, fy`):**

Concentric ring expansion around the cell of `(fx, fy)`:
```
for radius = 0, 1, 2, ... :
    if min_possible_distance(radius)² > best_dist_so_far → stop
    examine only the ring border (not the already-processed interior)
    for each seed in border cells:
        calculate Euclidean distance, update best
```

The minimum possible distance accounts for the point's position
within its cell: `min_possible = (radius - 1) × cell_size`, clamped to 0
for `radius ≤ 1` (adjacent cells can always contain a seed closer
than the current best).

Early stopping (`min_possible² > best`) means that on average,
only cells in the 3×3 neighborhood are examined (~4–8 seeds instead of k).

**Effective complexity:** O(pixels × seeds_per_cell) ≈ O(W×H) per iteration.

**`nearest_by` (K-means variant):** a generalized ring search that takes a
distance closure over the seed indices, plus two scale factors that convert
grid pixels into the distance's spatial units (e.g. normalized K-means
coordinates). Ties resolve to the smallest seed index, matching
`Iterator::min_by` semantics — bit-identical to the naive full scan.

---

## 9. Rendering (draw order)

```text
fn render(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbImage
```

The `render()` and `render_svg_from_dots()` functions take dots by
**shared reference** (`&[Dot]`) and don't clone them — the sorting for draw
order (largest to smallest) is done on a local copy of the slice.

1. Create an empty canvas with the background color.
2. **Sort dots by decreasing radius** (`sort_unstable_by`).
3. Draw from largest to smallest (painter's algorithm):
   - Large dots occupy the background (uniform zones).
   - Small detail dots overlap in the foreground.
4. **Parallel bands (rayon):** `render`/`render_rgba` split the canvas into
   horizontal bands. Each band draws all dots in the global sort order but
   clips pixels to its own rows, so the result is identical to the
   sequential draw — determined via `draw_dot_clipped` + a per-band write
   closure (`blend_rgb_band`/`blend_rgba_band`).
5. Each dot is drawn anti-aliased according to `params.dot_shape`
   (custom implementation, not `imageproc`):

**Anti-aliasing (`coverage_aa` + `blend_coverage`):** each dot is rasterized
with 4×4 supersampling (16 coverage levels per pixel). A corner pretest gives
two fast paths — if all 4 pixel corners are inside the (convex) shape, the
pixel is painted opaquely; if all 4 are outside *and* the centroid is
outside, the pixel is skipped. Only edge pixels fall through to the 4×4
subsample loop, keeping the cost low while eliminating the staircase
aliasing of the previous all-or-nothing test. The dot center is kept as
`f32` (sub-pixel position), so small dots and slight offsets never snap to
integer coordinates.

**Dithering (`floyd_steinberg`):** when `palette_size` is set and `dithering`
is true, the rendered RGB buffer is post-quantified: each pixel is replaced
by its nearest palette entry and the error (original − quantized) is
distributed to the next-right, next-row-left, next-row-center, and
next-row-right pixels with weights 7/16, 3/16, 5/16, 1/16. The palette
centers come from the same k-means as the `quantize_dots` path
(`compute_palette_centers`), but the dithering writes pixels directly instead
of snapping individual dots. The result is a classic "offset print" look
where the strategy averages out to the original intent. When `dithering` is
false, the previous path (`quantize_dots`) is used unchanged (each dot takes
the nearest palette center).

**Gamma correction (`gamma.rs`):** when `gamma_correct` is true, the source
image is linearized via a 256-entry sRGB→linéaire LUT, the entire pipeline
runs on the linearized buffer (so averages become perceptually correct), and
the result is re-encoded to sRGB via the inverse LUT before being returned.
This avoids the "mud" midtones of mixed black/white averages (sRGB average
127 vs the perceptually-correct ~188 in linear space). Previews emitted to
the GUI callback are also re-encoded so they remain displayable.

```rust
pub enum DotShape {
    Circle,                                 // Solid disk (default)
    Square,                                 // Square, side = 2×radius
    Ellipse { aspect: f32, angle_deg: f32 },// Ellipse with ratio + rotation
    RegularPolygon { sides: u8 },           // Regular polygon (3-12 sides, debug_assert sides≥3)
}
```

```
struct Dot {
    x, y   : f32    // center in pixels
    color  : [u8;3] // average RGB color of the zone
    radius : f32    // radius in pixels
}
```

---

## 10. GUI Architecture

Two front-ends share the library (`src/frontend.rs` for formatting, atomic
writes, export and presets):

| Binary          | Toolkit                | Role                                     |
| --------------- | ---------------------- | ---------------------------------------- |
| `pointimg-gui`  | `eframe`/`egui`        | immediate mode, default egui look         |
| `pointimg-gtk`  | `gtk4`/`libadwaita`    | native widgets, `adw::PreferencesPage`    |

### 10.1 UI decisions (shared by both front-ends)

- **Actions live in the header bar**: open, folder, presets, recalculate,
  undo/redo, export, copy. The side panel only holds *settings*; nothing
  important is left below the fold.
- **Before/after comparison** (`ViewMode::Compare`, the default): source and
  result are drawn in the same area, on either side of a draggable handle.
  GTK: `gtk::DrawingArea` with cairo surfaces clipped at the handle
  (`src/gtk/compare.rs`). egui: two `Painter::image` calls with distinct
  `clip_rect`s. Both draw the handle as a dark halo under a light stroke: a
  light-on-light handle vanished on a Voronoi render with a white background.
- **Fast preview + finalization**: `State.quality` ∈ {`Half`, `Quarter`,
  `Full`}. Below `Full`, the worker receives a downscaled image
  (`compute_source()`); after ~900 ms without a change, a `final_pass` at full
  resolution runs. The result is only `result_is_final` once that pass lands:
  image and SVG export stay disabled while it is stale.
- **Density map is indexed by scale**: `State.density_scale` records the scale
  the cache was computed at; a half-resolution density map is never reused by a
  full-resolution pass (the sizes do not match).
- **Built-in presets** live in the library (`filter::builtin_presets()`, tested:
  each one passes `validate_params` and a TOML round trip). Both front-ends
  offer the list on top of TOML files.
- **Presets are picked by their result**: each one ships a thumbnail rendered
  *with its own settings* on an example image, so the picker shows what a preset
  does instead of naming it. `cargo run --example preset_thumbnails`
  regenerates them from `builtin_presets()` — not from copied command-line
  flags, so a modified preset cannot leave a stale thumbnail behind.
  `frontend::preset_thumbnail` only *locates* the file; egui lays it out as a
  three-column grid, GTK as a `GtkFlowBox`. `tests/preset_thumbs.rs` fails if a
  preset has no thumbnail. The images are optional: an installation without
  `assets/presets/` falls back to a list of names.
- **Typed values**: GTK uses `adw::SpinRow` (slider *and* text entry) bounded by
  `validate_params`; egui uses `DragValue` with the same ranges. Sliders now
  stop exactly where the engine refuses to go.
- **Conditional settings are hidden, not greyed out.** GTK once kept "Palette
  colors", "Dithering" and "Seed" visible and insensitive; they read as
  available settings and the two front-ends diverged for nothing (egui already
  hid them).
- **The settings panel is resizable and bounded.** GTK uses
  `adw::OverlaySplitView` with a 360 px sidebar and an `adw::Breakpoint` at
  900 px: below that the panel folds over the preview and a header button
  brings it back. A `gtk::Paned` cannot do this, and it imposed the natural
  width of its widest row — 540 px, leaving 280 px of preview in an 820 px
  window.
- **Errors and confirmations are notifications**, not status-bar text. The
  status bar describes the current state; during a compute it shows progress,
  which erased any message a few frames after it was set. GTK uses
  `adw::Toast` (`src/gtk/preview.rs`); egui has an equivalent of its own
  (`src/gui/notify.rs`, 3 s for a confirmation, 8 s for an error).
- **Wheel inside the settings panel** (GTK): an `EventControllerScroll` in the
  *capture* phase forwards the wheel to the parent `ScrolledWindow` (Ctrl/Shift
  still adjusts the setting). Without it, scrolling the panel silently changed
  values and launched a compute per notch.
- **Synchronized views**: scrolling one preview pane is mirrored to the others
  (relative position, not absolute value).
- **Zoom**: "Fit" derives the factor from the viewport and displays it as such
  (the zoom widget is insensitive in that mode); any zoom change keeps the
  viewport center instead of jumping to the top-left corner.
- **Empty state**: `adw::StatusPage` / centered egui panel with "Open an
  image", "Open a folder" and "Use a sample" (when `assets/examples/` sits next
  to the binary).
- **A path on the command line opens it**: `pointimg-gtk photo.jpg` and
  `pointimg-gui photo.jpg`, and a directory activates folder navigation. GTK
  sets `HANDLES_OPEN` and reads `run_with_args_os`, so the file manager reaches
  the application through the same path; egui parses `argv` and defers the load
  to the first frame (`App::pending_open`), since the window does not exist yet
  when `main` runs.

### 10.1.1 What is *not* shared, and why

The two front-ends share the library, not a UI description. There is
deliberately **no `ParamSpec`**, no shared action table, no generated widget
tree: a single declarative description of the settings would force both fronts
to pick the same widgets, and each would lose what its toolkit does well
(`adw::SpinRow` with an embedded adjustment, `adw::ShortcutsWindow`,
`AdwOverlaySplitView` on one side; a `Slider` with the value in the rail,
`CollapsingHeader`, an immediate command palette on the other).

The duplication that remains is deliberate, and each copy declares its own
range, curve and scale. What *is* enforced is that no front-end offers a value
the engine refuses: `src/gui/ranges.rs` and `src/gtk/ranges.rs` each declare
their bounds and check both ends of every control against
`filter::validate_params`, plus the value just past it. A bound that drifts out
of contract then fails in the front-end that declared it, not at export time.

`src/frontend.rs` stays shared because it holds logic rather than presentation:
formatting, `Curve`, `ViewMode`, `Quality`, atomic writes, presets, export.

### 10.2 egui (`src/gui/`)

```
Main thread (egui)                    Worker thread
─────────────────────────             ──────────────────────────────────────
eframe::App::ui() [every frame]       filter::apply_with_progress(…, cb)
  ├─ poll worker results                   │  cb(iter, total, &img) per iteration
  ├─ shortcuts / drag & drop               │  → progress Arc<Mutex<(usize,usize)>>
  ├─ cancel + debounce                     │  → result Arc<Mutex<Option<RgbaImage>>>
  ├─ Lloyd progress bar                     │  → result_is_final Arc<AtomicBool>
  ├─ upload egui textures                  │  → returns (RgbImage, Vec<Dot>)
  ├─ finalize after inactivity       ◄─── computing=false, result=Some(img)
  └─ on params change:
       on_params_changed() → 300 ms debounce
       → trigger_compute() → cancel then start_compute()
```

Panel order matters: `Panel::top` (actions), `Panel::bottom` (status),
`Panel::left` (settings, inside a `ScrollArea`), then `CentralPanel`. The
bottom panel must be allocated *before* the left one, otherwise the settings
panel overflows on top of it.

Interface preferences (scale, theme) live in `src/gui/prefs.rs`: a small
`key = value` file next to eframe's own state, written as soon as a preference
changes. eframe's `App::save` is not used because it only runs on its 30-second
autosave — a preference changed just before quitting would be lost. "Auto" maps
to egui's `ThemePreference::System`, which is reapplied on the first frame.

- **A single recompute guard**: `show_control_panel` compares `params` before and
  after drawing *all* the settings sections, and only calls `on_params_changed`
  if something actually moved. Without it the comparison has to be written field
  by field — which is exactly what the integer fields were missing, so changing
  the point count or the iteration count recomputed nothing.
- **Display precision**: `display_decimals` (in `src/gui/sliders.rs`) pins each
  `DragValue` to the number of decimals of its value and its range. egui
  otherwise picks a precision from the pointer's distance to the screen edge,
  and rewrote the text on focus loss whenever that was coarser.
- **Export size** (`src/gui/exports.rs`): the measurement encodes the real
  image (see `export`), so it runs in a thread and comes back through a shared
  slot (`export_sizes_slot`), invalidated by a generation counter. It only runs
  on the *final* result — the size of a preview would mean nothing.
- **Sliders** (`src/gui/sliders.rs`): `slider_row` delegates to `egui::Slider`
  (logarithmic on wide ranges), `range_slider` draws a two-handle slider for
  bound pairs, with two disjoint interaction zones split halfway between the
  handles — so no state has to be kept across frames. The mapping comes from
  `frontend::Curve`, shared with GTK.
- **Collapsible sections** (`section` in `ui.rs`): the settings panel is four
  `CollapsingHeader`s. A thirty-control single column had to be scrolled in
  full to find one option. The open state is a `CollapsingState`, stored in
  egui's memory, which eframe persists — the sections open at launch are the
  ones left open last time. Folding is presentation only; nothing is hidden
  from the engine, and GTK keeps its own `PreferencesGroup`s.
- **Command palette** (`src/gui/palette.rs`, `Ctrl+K`): twenty commands with
  their shortcut printed next to them. egui had no menu and no shortcuts
  window — its seventeen shortcuts were only discoverable on hover, and the
  density map and full screen were unreachable except by cycling `Tab`. It is
  an `egui::Window`, not a port of GTK's `GMenu`: same intent, own
  implementation. Unavailable commands are shown greyed rather than hidden, so
  they are discoverable.
- **Notifications** (`src/gui/notify.rs`): a stack of toasts over the preview.
  Expiry needs an explicit `request_repaint_after` — egui has no clock, so
  without it a message would stay until the user touched something.
- **Window icon**: egui has no icon theme, so the 256 px PNG is embedded with
  `include_bytes!` and decoded at startup (`window_icon()` in `main.rs`).
  `assets/logo/pointimg.svg` is the master; `packaging/icons/regenerate.sh`
  derives every size from it.

### 10.3 GTK4 / libadwaita (`src/gtk/`)

```
GTK main thread                        Worker threads
─────────────────────────             ──────────────────────────────────────
State (Rc<RefCell<..>>) + Widgets     filter::apply_with_progress(…, publish)
  ├─ win.* actions (SimpleActionGroup)      │  publish → Msg::Progress / Preview
  ├─ widget signals (wire.rs)               │  → async-channel
  ├─ cancel compute (AtomicBool)       ◄─── Msg::Done { image, dots, elapsed_ms,
  ├─ zoom / views (preview.rs)                  full_resolution }
  └─ spawn_local: channel consumer      Msg::Failed / Msg::Density
```

- `State` is built around a **generation counter**: an obsolete worker never
  publishes (`is_current`), be it a result, a progress update or an error.
- Any state write from a GTK callback releases its `borrow` **before** touching
  a widget: GTK setters emit synchronous signals whose handlers call
  `state.borrow()` (this caused "already mutably borrowed" aborts).
- An `updating` flag neutralizes handlers during programmatic updates (undo,
  preset, preset file).
- `win.*` actions live in a `SimpleActionGroup` inserted on the window;
  accelerators go through a window-level `ShortcutController`
  (`gtk_shortcut_action_parse_string` expects `action(win.name)`), and
  non-`Ctrl` shortcuts through an `EventControllerKey`.
- **Sliders** (`controls.rs`): a `GtkScale` doubling every wide-range
  `SpinRow`. GTK has no logarithmic scale, so the slider works in relative
  position (0 → 1) and `SliderSpec` (copied by value into the callbacks, which
  cannot capture the widget) converts through `frontend::Curve`, rounding to the
  `SpinRow`'s precision. `adw::PreferencesGroup` only accepts
  `AdwPreferencesRow` and an `AdwActionRow` suffix does not expand, so a slider
  row is a title-less `ActionRow` with the slider aligned to its right — and
  that row is what must be hidden when the doubled setting disappears, not the
  slider alone.
- **`commit_value_change`** (`wire.rs`) is the common exit point for spin rows
  and sliders: it compares `params` before/after before scheduling a recompute
  (GTK re-emits `value-changed` on focus loss, with the re-parsed entry value)
  and re-syncs the widgets when the value came from a slider.
- **Render parity**: neither front-end modulates the result opacity while
  computing. The veil GTK used to apply (55% on the result, 70% on the comparison)
  dulled the image for no reason — the provisional nature of the render is
  already carried by the status bar text. Any visual divergence between the two
  must be justifiable by a line of code present on both sides.
- **Export size** (`exports.rs`): the worker encodes the render and publishes
  `Msg::ExportSizes` on the existing channel — only the GTK loop ever touches a
  widget, and the label is refreshed from the handler. GTK has no place to poll a
  mutex from.
- **Resizable panel** (`controls.rs`, `ui.rs`): `adw::OverlaySplitView`, sidebar
  pinned at 360 px, plus `set_propagate_natural_width(false)` on the
  `ScrolledWindow`. That last call is the one that matters: without it the
  sidebar reports the natural width of its widest row as its minimum, which is
  what made the panel 540 px wide regardless of the window. An
  `adw::Breakpoint` (`max-width: 900px`) collapses the sidebar over the
  preview and reveals a header button; the condition is evaluated by GTK on
  every resize, so there is no width handler to maintain.
- **Main menu** (`wire.rs`): the application menu — open, folder, previous /
  next, presets, view, full screen, shortcuts — sits under a hamburger button.
  It used to hang off the "Export" split button, which had nothing to do with
  opening a folder or reading the shortcuts. Labels carry `_` mnemonics. The
  export button keeps an export-only menu (`export_menu_model`).
- **Recent images** (`recent.rs`): `GtkRecentManager`, the desktop's own
  mechanism — the same list that shows up in every GNOME application's menu.
  Entries are filtered to supported images that still exist. egui has no such
  mechanism and keeps its own list in `prefs.rs`; the intent is shared, the
  storage is each toolkit's.
- **`if let` and `RefCell`**: in `load_image`, `state.borrow()` inside the
  scrutinee lives until the end of the block, so the `borrow_mut()` that
  followed it panicked. Folder navigation with `←`/`→` aborted the application.
  The read borrow is now scoped in a block of its own.

### 10.4 Dithering (`src/filter/dither.rs`)

Four palette-quantization algorithms, selected by `FilterParams::dithering`:

- `Off`: nearest color, no pattern.
- `FloydSteinberg`: error diffusion to four neighbours — the historical method,
  and the only one that propagated error before this section.
- `Riemersma`: a `taps`-pixel queue along the scanline with geometric weights
  summing to 1. Each share is paid out **once** (a pure shift, no re-injection of
  the error still queued), so the mean is preserved exactly. `RIEMERSMA_DECAY` is
  calibrated on a 160 px wide gradient — at 0.7, the mean jump between adjacent
  pixels drops from 127 to 66, the usual Riemersma trade-off.

  **The queue crosses row boundaries**, and the scan is **serpentine**: the queue
  is `taps` long (not as wide as the image, which lost shares on narrow images),
  it is not reset between rows, and every other row is walked backwards — so the
  share of error crossing the end of a row lands on the pixel just below. Measured:
  resetting discarded the residual and left a drift of 1.0 luma unit on a
  horizontal ramp (0.83 on a 2D gradient); it is now 0.00.

  **What mean preservation does not prove** — established by mutation, not assumed.
  A "mean luminance survives quantization" test fails if the weight signs are
  flipped (positive feedback, 37.85 drift), but **passes** if the weight
  normalization is dropped, and **passes** too if all error propagation is removed.
  Two structural reasons: diffusion is a closed feedback loop — the injected error
  changes the quantizer's choice, hence the next error, and the steady state
  conserves the mean whatever the gain — and per-pixel independent quantization
  already conserves the mean of a monotone ramp, the staircase steps cancelling.

  Verified split of the checks: loop **stability** is covered by the mean test,
  **smoothing** (hence the presence and the gain of a diffusion) by
  `riemersma_tail_smooths_the_pattern`, and the **calibration invariant** by
  `les_poids_de_la_queue_soment_a_un`, on the weight construction
  (`riemersma_weights`) — the only test that de-normalization fails. General rule:
  any end-to-end property is confronted to a mutation before being trusted.
- `Bayer4` / `Bayer8`: ordered dithering, no diffusion. Each pixel picks between
  its two nearest palette colors from the matrix threshold at its position,
  projected on the segment joining them. The matrix is **generated** with
  `M(2n) = 4·M(n) + [[0, 2], [3, 1]]` (Bayer, 1973), never copied.

Serialization: `dithering = "bayer4"`. 0.1 presets wrote `dithering = true`, so
`deserialize_dithering` accepts both a boolean and a name — no existing file
becomes unreadable. On the CLI, `--dithering` with no value means
Floyd-Steinberg (`Option<Dithering>` + `default_missing_value`: with
`num_args = 0..=1`, a plain type would make the argument mandatory).

## 11. Export Size (`src/export.rs`)

No formula gives the size of a PNG or WebP reliably: it depends on the entropy
of the render. `measure_export_sizes` therefore encodes the real image, per
format and in parallel (rayon), through the **same** path as the export —
`frontend::encode_rgba_extension`, used by both `save_rgba_image` and the
measurement. A test checks that the announced size is the one of the file
actually written.

Two safeguards: the measurement only ever runs on the final result (invalidated
by generation as soon as the user touches a setting again), and WebP — *lossless*,
hence slow to encode — is not measured above `export::WEBP_MEASURE_LIMIT`
(4 Mpx); the interface then shows “—” instead of an invented number.

## 12. SVG Export

`render_svg_from_dots(w, h, dots, params) -> Result<String>`

Generates a standalone SVG document with:
- `<rect>` background (color `bg_color`)
- One SVG element per dot according to `dot_shape`:
  - `Circle` → `<circle cx cy r fill>`
  - `Square` → `<rect x y width height fill>`
  - `Ellipse` → `<ellipse cx cy rx ry transform fill>`
  - `RegularPolygon` → `<polygon points fill>`

SVG size is identical to the source image in pixels.
The SVG can be opened in a browser, Inkscape, or further vectorized.

**Public API:**
```text
pub fn render_svg_from_dots(w: u32, h: u32, dots: &[Dot], params: &FilterParams) -> Result<String>
pub fn render_svg(src: &RgbImage, params: &FilterParams) -> Result<String>
pub fn render_svg_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<String>
```

---

## 13. Quality Tools

### Available immediately

```bash
# Type checking without compiling (gtk excluded: needs GTK4/libadwaita)
cargo check --features gui,gpu,avif
cargo check --no-default-features          # CLI-only

# Automatic code formatting
cargo fmt

# Official Rust linter
cargo clippy --features gui,gpu,avif --all-targets -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings

# Unit tests (src/filter/* + src/main.rs + src/color.rs)
cargo test --lib

# Integration tests (tests/*.rs — pipeline, svg, repro, palette, halftone)
cargo test --features gui,gpu,avif
cargo test --no-default-features            # CLI-only build

# Benchmarks (criterion)
cargo bench

# Generate Rust documentation
cargo doc --no-deps --features gui,gpu,avif
```

### Installed

```bash
# cargo-audit — vulnerability detection in dependencies
cargo install cargo-audit
cargo audit
# Note: the `audit` CI job runs this on each push (continue-on-error).
```

### CI (GitHub Actions)

The full pipeline lives in `.github/workflows/ci.yml` and runs on every
push/PR to `main`:

- **fmt** — `cargo fmt --check`
- **clippy** — matrix `ubuntu/macos/windows`,
  `cargo clippy --features gui,gpu,avif --all-targets -- -D warnings` plus
  `--no-default-features --all-targets` (CLI-only coverage)
- **test** — matrix `ubuntu/macos/windows`, `cargo test --features gui,gpu,avif --locked`
  plus `cargo test --no-default-features --locked`
- **gpu** — lavapipe software adapter, CPU/GPU density parity
- **msrv** — `cargo +1.95 check --features gui,gpu,avif --locked` (and `stable`)
- **build** — `cargo build --release --features gui,gpu,avif`
- **gtk** — Linux-only, installs `libgtk-4-dev`/`libadwaita-1-dev`, clippy +
  release build of `pointimg-gtk`
- **docs** — `cargo doc --no-deps --features gui,gpu,avif` with `--cfg docsrs`
- **audit** — `cargo audit` (scheduled workflow too)
- **fuzz-build** — nightly `cargo fuzz` smoke runs (100 runs per target)

`gtk` is deliberately excluded from `--all-features` everywhere: GTK4 and
libadwaita are not available on all runners nor on docs.rs.

All Rust jobs use `Swatinem/rust-cache@v2` for dependency caching.
Release builds (cross-platform binaries) are in `.github/workflows/release.yml`.
