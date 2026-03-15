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
│   ├── filter.rs       — all logic: algorithms, rendering, helpers
│   ├── main.rs         — CLI binary (`pointimg`)
│   └── gui/
│       └── main.rs     — GUI binary (`pointimg-gui`)
└── assets/             — test images and outputs
```

| Crate | Role |
|---|---|
| `image 0.25` | Loading / saving / manipulating `RgbImage` |
| `clap 4` | CLI argument parsing |
| `rayon 1` | Parallel iteration (density map) |
| `eframe 0.31` | egui framework (wgpu backend) — *feature-gated* `gui` |
| `egui 0.31` | Immediate-mode GUI widgets — *feature-gated* `gui` |
| `wgpu 24` | GPU backend (Vulkan + GL fallback) — *feature-gated* `gui` |
| `rfd 0.15` | Native file dialogs — *feature-gated* `gui` |

> **Feature-gating:** GUI dependencies are behind the `gui` feature
> (enabled by default). To compile CLI-only: `cargo build --no-default-features`.

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
```

**All flags:**

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | required | Source image (JPEG, PNG, RGBA supported) |
| `--output` | `-o` | `output.png` | Output PNG image |
| `--algorithm` | `-a` | `voronoi` | `grid` \| `kmeans` \| `voronoi` \| `quadtree` |
| `--num-points` | `-n` | `800` | Number of points (kmeans/voronoi/quadtree) |
| `--cols` | `-c` | `80` | Grid columns (grid only) |
| `--min-radius` | | `0.003` | Min radius = fraction of `min(W,H)` |
| `--max-radius` | | `0.06` | Max radius = fraction of `min(W,H)` |
| `--bg` | `-b` | `white` | `white`, `black`, or `#rrggbb` |
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
}
```

`is_bg_light()` — derived method from `bg_color` (perceptual luminance > 127.5).

**`min_radius_ratio` / `max_radius_ratio`:** fractions of `min(width, height)` of the image.
On an 800×600 image, `0.003` → 1.8 px and `0.06` → 36 px.

**`variance_sensitivity`** affects three places:
- The density map (blend between uniform and pure variance)
- Lloyd weights exponent (`power = 1 + sensitivity × 3`)
- Radius boost in flat zones

**`max_boost`** caps the radius multiplier in uniform zones.
`1.0` = no boost. `2.5` = a point in a totally flat zone can be 2.5× `r_max`.

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

```rust
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

```rust
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
4. Each dot is drawn according to `params.dot_shape` (custom implementation,
   not `imageproc`):

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
| `Ctrl+S` | Save the result (file dialog, PNG or JPEG) |
| `Space` | Manually recalculate |

**GPU Backend:** forced to Vulkan (+ GL fallback) to avoid the EGL/Wayland
crash with NVIDIA drivers on Wayland:
```rust
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
```rust
pub fn render_svg_from_dots(w: u32, h: u32, dots: &[Dot], params: &FilterParams) -> Result<String>
pub fn render_svg(src: &RgbImage, params: &FilterParams) -> Result<String>
pub fn render_svg_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<String>
```

---

## 12. Quality Tools

### Available immediately

```bash
# Type checking without compiling
cargo check

# Automatic code formatting
cargo fmt

# Official Rust linter
cargo clippy -- -D warnings

# Unit tests (26 tests in filter.rs)
cargo test --lib

# Generate Rust documentation
cargo doc --open
```

### Installed

```bash
# Clippy 1.92.0 (from rc-buggy)
sudo apt-get install -t rc-buggy rust-clippy=1.92.0+dfsg1-1~exp1

# cargo-audit — vulnerability detection in dependencies
sudo apt install cargo-audit
cargo audit
# Note: cargo audit may fail if advisory-db contains CVSS 4.0 entries
# (upstream bug RUSTSEC-2026-0026) — non-blocking
```

### Suggested CI (GitHub Actions)

```yaml
- run: cargo fmt --check
- run: cargo clippy -- -D warnings
- run: cargo test --lib
- run: cargo build --release
```