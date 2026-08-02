# pointimg

[![Release](https://github.com/pvtc/pointimg/actions/workflows/release.yml/badge.svg)](https://github.com/pvtc/pointimg/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)
[![MSRV 1.88](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://www.rust-lang.org)

A pointillist filter that transforms images into compositions of colored dots of varying sizes. Each dot takes the average color of its zone and its radius is modulated by local luminance and variance.

## Features

- **4 algorithms**: Grid, K-means, Voronoi (weighted Lloyd), Adaptive Quadtree
- **4 dot shapes**: Circle, Square, Ellipse (aspect + rotation), Regular Polygon (3-12 sides)
- **Density map**: automatic redistribution of dots based on local detail
- **Export PNG and SVG** vector
- **Interactive GUI** (egui/wgpu) with progressive preview and drag & drop
- **CLI** with all parameters accessible
- **Reduced palette** option (color quantization)
- **Customizable background**: white, black, any color `#rrggbb`, or `transparent` (RGBA output)
- **Gamma correction** option: averages are computed in linear color space to avoid muddy midtones
- **Anti-aliased rendering**: 4×4 supersampled dots (no staircase edges)
- **Batch processing**: glob/directory input, per-file output pattern (`{n}`, `{stem}`, `{name}`)
- **Fast preview**: downscale source before pipeline via `--preview WxH`
- **Presets**: load/save `FilterParams` as TOML
- **Floyd-Steinberg dithering** on quantized palette (offset-print look)
- **Halftone screen angle**: rotate the Grid lattice via `--grid-angle`
- **CMYK rosette** & **N-colors dominant** halftone (`--halftone cmyk|dominant-N`) — true ink-over-ink multiply compositing, AM rotating-screen or FM stochastic-blue-noise screening
- **Output formats**: PNG/JPEG/BMP/TIFF/WebP by default; AVIF via `--features avif`
- **Optional GPU density pass**: build with `--features gpu`, enable with
  `POINTIMG_GPU=1`; CPU/SAT fallback remains automatic

## Examples

| Source | Result (Voronoi, 1200 points) |
|---|---|
| sample photo | [`assets/examples/pexels-kofishelbyfotos-38152015.jpg`](assets/examples/pexels-kofishelbyfotos-38152015.jpg) |

Sample test images: [`kilauea25_0.jpeg`](assets/examples/kilauea25_0.jpeg),
[`pexels-kofishelbyfotos-38152015.jpg`](assets/examples/pexels-kofishelbyfotos-38152015.jpg),
[`pexels-ruyan-ayten-153760-4190725.jpg`](assets/examples/pexels-ruyan-ayten-153760-4190725.jpg).

## Installation

### Pre-built binaries

Download binaries from the [Releases](https://github.com/pvtc/pointimg/releases) page for your platform:
- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon, Universal)
- Windows (x86_64)

### Build from source

```bash
cargo build --release
```

Binaries will be in `target/release/`:
- `pointimg` -- CLI tool
- `pointimg-gui` -- GUI application (requires Vulkan or OpenGL)

To build CLI only (without GUI dependencies):

```bash
cargo build --release --no-default-features
```

## CLI Usage

```bash
# Voronoi, 1500 points, white background
pointimg -i photo.jpg -o result.png --algorithm voronoi --num-points 1500

# Grid, black background
pointimg -i photo.jpg -o result.png --algorithm grid --cols 60 --bg black

# Custom hex background
pointimg -i photo.jpg -o result.png --bg "#1a1a2e"

# Square shape
pointimg -i photo.jpg -o result.png --shape square

# Hexagons
pointimg -i photo.jpg -o result.png --shape polygon --polygon-sides 6

# SVG export
pointimg -i photo.jpg --svg --algorithm voronoi --num-points 2000

# Transparent background (RGBA output, alpha 0 between dots)
pointimg -i photo.jpg -o result.png --bg transparent

# Gamma correction (perceptual averages)
pointimg -i photo.jpg -o result.png --gamma

# Batch processing (glob or directory) + per-file pattern
pointimg -i "photos/*.jpg" -o "out/{stem}_{n}.png" --algorithm grid --cols 50

# Fast preview (downscale source first)
pointimg -i big.jpg -o preview.png --preview 300x300

# Palette + Floyd-Steinberg dithering
pointimg -i photo.jpg -o result.png --palette 8 --dithering

# Halftone screen angle (Grid)
pointimg -i photo.jpg -o result.png --algorithm grid --cols 60 --grid-angle 30

# Save / recall presets as TOML
pointimg --save-preset out.toml --algorithm voronoi --num-points 1500 --gamma
pointimg --preset out.toml -i photo.jpg -o result.png

# WebP out-of-the-box ; AVIF requires `cargo build --features avif`
pointimg -i photo.jpg -o result.webp

# CMYK rosette halftone (true ink multiply, 4 channels at 15°/75°/0°/45°)
pointimg -i photo.jpg -o rosette.png --halftone cmyk --halftone-freq 60

# Halftone using the N most dominant colors (instead of CMYK)
pointimg -i photo.jpg -o dominants.png --halftone dominant-6 --halftone-freq 80

# Stochastic FM screening (blue-noise density) instead of AM rotated grid
pointimg -i photo.jpg -o stochastic.png --halftone cmyk --screening fm
```

Run `pointimg --help` for the complete list of options.

## GUI Usage

```bash
pointimg-gui
```

- Drag & drop an image or click "Open"
- Adjust parameters in the left panel (automatic recalculation with debounce)
- Export to PNG or SVG

**Keyboard shortcuts:**

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open an image |
| `Ctrl+S` | Save the result |
| `Ctrl+Z` | Undo (last params change) |
| `Ctrl+Y` or `Ctrl+Shift+Z` | Redo |
| `Space` | Recalculate |

## Tests

```bash
cargo test --all-features    # 63 tests (41 lib + 22 integration)
cargo clippy --all-features -- -D warnings
cargo bench                   # criterion benchmarks (benches/filter.rs)
```

## Debugging

Enable debug logging with the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug cargo run --release -- -i photo.jpg -o result.png
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
