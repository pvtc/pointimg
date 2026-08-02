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
11. [SVG Export](#11-svg-export)
12. [Quality Tools](#12-quality-tools)

---

## 1. Project Structure

```
pointimg/
├── Cargo.toml          — dependencies, two binaries + a lib
├── src/
│   ├── lib.rs          — crate root; exposes `pub mod filter`
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
│   │   ├── util.rs     — luminance, pixel_sum, pixel_variance, flatten_to_rgb
│   │   └── algorithms/ — placement algorithms, one file each
│   │       ├── grid.rs     — dots_grid
│   │       ├── kmeans.rs   — dots_kmeans_progressive, compute_dots_kmeans,
│   │       │               dots_from_kmeans_centers
│   │       ├── quadtree.rs — dots_quadtree, subdivide
│   │       └── voronoi.rs  — dots_voronoi_progressive, compute_dots_voronoi
│   ├── main.rs         — CLI binary (`pointimg`)
│   └── gui/
│       └── main.rs     — GUI binary (`pointimg-gui`)
├── tests/              — integration tests (public API end-to-end)
│   ├── pipeline.rs     — algorithm dimensions, progress, dots, flatten, density
│   ├── svg.rs          — SVG header, shapes, bg color, empty-image error
│   ├── reproducibility.rs — same seed → identical, different seed → diverge
│   ├── palette.rs      — palette quantization reduces SVG fill colors
│   └── halftone.rs     — CMYK rosette, dominant colors, FM screening, TOML presets
└── benches/
    └── filter.rs       — criterion benches (apply per algo, density, svg)
```

| Crate | Role |
|---|---|
| `image 0.25` | Loading / saving / manipulating `RgbImage` |
| `clap 4` | CLI argument parsing |
| `rayon 1` | Parallel iteration (density map, Lloyd, k-means) |
| `anyhow 1` | Error handling (`Result`, `anyhow!`) |
| `eframe 0.31` | egui framework (wgpu backend) — *feature-gated* `gui` |
| `egui 0.31` | Immediate-mode GUI widgets — *feature-gated* `gui` |
| `wgpu 24` | GPU backend (Vulkan + GL fallback) — *feature-gated* `gui` |
| `rfd 0.15` | Native file dialogs — *feature-gated* `gui` |
| `serde 1` | Serialization framework (`FilterParams` derive) |
| `toml 0.8` | TOML preset files (`--preset` / `--save-preset`) |
| `glob 0.3` | Glob expansion for batch `--input` (`*`, `?`, `[`) |
| `serde 1` | Serialization framework (`FilterParams` derive, presets TOML) |
| `criterion 0.5` | Benchmarks — *dev-dependency* |

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
> **MSRV:** Rust 1.88 (declared via `rust-version` in `Cargo.toml`,
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

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | required | Source image, glob, or directory. Optional if `--save-preset` alone. |
| `--output` | `-o` | `output.png` | Output file. Patterns: `{n}` (index), `{stem}` (source name sans ext), `{name}` (source full name). |
| `--algorithm` | `-a` | `voronoi` | `grid` \| `kmeans` \| `voronoi` \| `quadtree` |
| `--num-points` | `-n` | `800` | Number of points (kmeans/voronoi/quadtree) |
| `--cols` | `-c` | `80` | Grid columns (grid only) |
| `--min-radius` | | `0.003` | Min radius = fraction of `min(W,H)` |
| `--max-radius` | | `0.06` | Max radius = fraction of `min(W,H)` |
| `--bg` | `-b` | `white` | `white`, `black`, `transparent`/`none`, or `#rrggbb` |
| `--shape` | | `circle` | `circle` \| `square` \| `ellipse` \| `polygon` |
| `--ellipse-aspect` | | `1.5` | Width/height ratio (ellipse) |
| `--ellipse-angle` | | `0.0` | Rotation angle in degrees (ellipse) |
| `--polygon-sides` | | `6` | Number of sides (polygon, 3-12) |
| `--iterations` | | `10` | Lloyd / k-means iterations |
| `--variance-sensitivity` | | `0.7` | Variance redistribution strength `[0,1]` |
| `--max-boost` | | `2.5` | Max multiplier in uniform zones |
| `--seed` | | *(random)* | RNG seed for exact reproduction |
| `--palette` | | *(disabled)* | Number of colors in reduced palette |
| `--svg` | | *(disabled)* | Also export SVG (same path, `.svg` extension) |
| `--gamma` | | *(disabled)* | Gamma-correct perceptual averages (linear space) |
| `--dithering` | | *(disabled)* | Floyd-Steinberg dithering on rendered image |
| `--preset` | | *(disabled)* | Load `FilterParams` from a TOML preset file |
| `--save-preset` | | *(disabled)* | Save current params to a TOML preset file and exit (if `--input` absent) |
| `--verbose` | `-v` | *(0)* | `-v` = info, `-vv` = debug. Overridable via `RUST_LOG`. |
| `--quiet` | `-q` | *(disabled)* | Suppress info messages (warnings only) |
| `--preview` | | *(disabled)* | Downscale source to `WxH` (max, preserves aspect) before pipeline |
| `--grid-angle` | | `0.0` | Rotation of grid placement in degrees (Grid only, halftone screen effect) |
| `--halftone` | | `off` | `off` \| `cmyk` (4-channel rosette) \| `dominant-N` (N colors via k-means) |
| `--screening` | | `am` | `am` (rotated-grid rosette) \| `fm` (stochastic blue noise) |
| `--halftone-freq` | | `60.0` | AM cells per `min(W,H)` (≈ dot frequency in lpp at 1 dpi) |
| `--halftone-min-radius` | | `0.002` | Min radius per ink dot (fraction of `min(W,H)`) |
| `--halftone-max-dot` | | `0.85` | Max radius per ink dot (fraction of trame step) |

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
`num_points > 0`, `cols > 0`, `palette_size >= 2` (if set),
`variance_sensitivity ∈ [0, 1]`, `max_boost >= 1.0`, polygon `sides ∈ 3..=12`,
ellipse `aspect ∈ (0, 10]`, image dimensions `≤ 65535` and
`≤ 256M` pixels. An out-of-range value returns `Err` instead of panicking
or silently misbehaving (e.g. `cols = 0` previously panicked on division).

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

**Complexity:** O(iterations × W×H×k) — slow for k > 500

1. **Initialization:** `k` seeds via `importance_sample` (biased towards detailed zones).
   Seeds are placed with a random **sub-pixel jitter** of ±0.5 pixel
   to avoid clustering on pixel centers.
2. Represent each pixel as a normalized 5D vector `[x/W, y/H, r/255, g/255, b/255]`.
3. **Iterations:**
   - Assign each pixel to the nearest center (5D Euclidean distance).
   - Recalculate each center as the mean of its assigned pixels.
4. Emit one point per surviving center.

**Early stopping (convergence):** if the maximum movement of all centers
is less than 0.5 pixel between two iterations, the loop stops early.
This avoids unnecessary iterations when convergence is already reached.

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

**Complexity:** O(W×H×log(max_depth)) amortized

**Internal parameters:**
```
min_cell  = max(sqrt(W×H / num_points) / 2, 2)   pixels
threshold = 800 × (1 − variance_sensitivity × 0.8)
```

**Recursive subdivision (`subdivide`):**

```
subdivide(cell [x,y,w,h]):
  if w < 2 or h < 2 → stop
  calculate average_color and variance of cell
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
4. Each dot is drawn anti-aliased according to `params.dot_shape`
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

**Threading model:**

```
Main thread (egui)                    Worker thread
─────────────────────────             ──────────────────────────────────────
App::update() [each frame]            filter::apply_with_progress(…, cb)
  ├─ drag & drop → load_image()           │  cb(iter, total, &img) per iteration
  ├─ draw widgets                         │  → progress Arc<Mutex<(usize,usize)>>
  ├─ Lloyd progress bar                   │  → result Arc<Mutex<Option<RgbImage>>>
  ├─ compute time (Instant)               │  → ctx.request_repaint() (immediate repaint)
  ├─ read computing (AtomicBool)          │  → returns (RgbImage, Vec<Dot>)
  ├─ if result ready:                ◄─── computing=false, result=Some(img), last_dots=Some(dots)
  │    upload egui texture
  └─ if params changed:
       trigger_compute()
         → cancel=true (cancel previous compute)
         → wait computing=false
         → start_compute(ctx) (spawn new thread)
```

**Inter-thread sharing:**

| Arc | Type | Role |
|---|---|---|
| `result` | `Arc<Mutex<Option<RgbImage>>>` | Image produced by worker |
| `last_dots` | `Arc<Mutex<Option<Vec<Dot>>>>` | Dots from last calculation (SVG export) |
| `computing` | `Arc<AtomicBool>` | "calculation in progress" flag |
| `cancel` | `Arc<AtomicBool>` | Cancellation request (checked between iterations) |
| `progress` | `Arc<Mutex<(usize, usize)>>` | (current_iter, total_iter) for progress bar |
| `compute_error` | `Arc<Mutex<Option<String>>>` | Error from compute thread |

All `Mutex::lock()` use `unwrap_or_else(|e| e.into_inner())` to recover
content even if the worker panicked (poison-safe).

**ViewMode (before/after comparison):**

```rust
enum ViewMode { Side, ResultOnly, SourceOnly, DensityMap }
```

- `Side` : source and result side by side (splits available space in two)
- `ResultOnly` : result fullscreen
- `SourceOnly` : source fullscreen
- `DensityMap` : density map preview

**Density map cache:** `src_rgb: Option<RgbImage>` is calculated once
at load time (including alpha composition), and reused for each recalculation.
`refresh_src_rgb()` is only called when `bg_color` changes.
The density map preview is **automatically recalculated** when
`variance_sensitivity` changes (in addition to load and `bg_color` change).

**Throttled progressive preview:** for Voronoi and K-means, each Lloyd
iteration can publish an intermediate result via the callback. Image cloning
for the preview is **limited to 100 ms intervals** to avoid saturating the main
thread with unnecessary copies. The last iteration is always published.

**Adaptive zoom:** "Fit" mode uses a boolean flag `zoom_fit` instead of
the magic value `zoom=0.0`. Zoom is calculated dynamically each frame
based on available space. Any manual zoom interaction
(slider, `+`, `-`) disables `zoom_fit`; the "Fit" button re-enables it.

**Keyboard shortcuts:**

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open an image (file dialog) |
| `Ctrl+S` | Save the result (file dialog, PNG/JPEG/WebP/BMP/TIFF) |
| `Ctrl+Z` | Undo (last params change, 50-entry stack) |
| `Ctrl+Y` or `Ctrl+Shift+Z` | Redo |
| `Space` | Manually recalculate |

**Undo / Redo architecture:** `App.history: Vec<FilterParams>` + `future:
Vec<FilterParams>` + `last_committed: Option<FilterParams>`. Un changement de
paramètre (slider, checkbox…) positionne `pending_commit=true`. Quand le calcul
associé se termine (compute_start.take()), `commit_history()` pousse
l'ancienne tête dans `history` et clears `future`. `FilterParams: PartialEq`
évite de pousser un état identique (ex: drag sans variante). Undo repousse
l'état courant vers `future`, restore depuis `history`, debounce relance le
compute. 50 entrées max (FIFO).

**GPU Backend:** forced to Vulkan (+ GL fallback) to avoid the EGL/Wayland
crash with NVIDIA drivers on Wayland:
```text
WgpuSetup::CreateNew { backends: VULKAN | GL, .. }
```

---

## 11. SVG Export

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

## 12. Quality Tools

### Available immediately

```bash
# Type checking without compiling
cargo check --all-features

# Automatic code formatting
cargo fmt

# Official Rust linter
cargo clippy --all-features -- -D warnings

# Unit tests (41 tests in src/filter/mod.rs + dither + gamma)
cargo test --lib

# Integration tests (22 tests across tests/*.rs — pipeline, svg, repro,
# palette, halftone)
cargo test --all-features

# Benchmarks (criterion)
cargo bench

# Generate Rust documentation
cargo doc --no-deps --all-features
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
- **clippy** — matrix `ubuntu/macos/windows`, `cargo clippy --all-features -D warnings`
- **test** — matrix `ubuntu/macos/windows`, `cargo test --lib --all-features`
- **msrv** — `cargo +1.88 check --all-features --locked` (and `stable`)
- **build** — `cargo build --release --all-features`
- **docs** — `cargo doc --no-deps --all-features` with `--cfg docsrs`
- **audit** — `cargo audit` (continue-on-error)

All Rust jobs use `Swatinem/rust-cache@v2` for dependency caching.
Release builds (cross-platform binaries) are in `.github/workflows/release.yml`.
