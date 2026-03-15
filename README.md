# pointimg

[![Release](https://github.com/pvtc/pointimg/actions/workflows/release.yml/badge.svg)](https://github.com/pvtc/pointimg/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)

A pointillist filter that transforms images into compositions of colored dots of varying sizes. Each dot takes the average color of its zone and its radius is modulated by local luminance and variance.

## Features

- **4 algorithms**: Grid, K-means, Voronoi (weighted Lloyd), Adaptive Quadtree
- **4 dot shapes**: Circle, Square, Ellipse (aspect + rotation), Regular Polygon (3-12 sides)
- **Density map**: automatic redistribution of dots based on local detail
- **Export PNG and SVG** vector
- **Interactive GUI** (egui/wgpu) with progressive preview and drag & drop
- **CLI** with all parameters accessible
- **Reduced palette** option (color quantization)
- **Customizable background**: white, black, or any color `#rrggbb`

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
| `Space` | Recalculate |

## Tests

```bash
cargo test --lib
cargo clippy -- -D warnings
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