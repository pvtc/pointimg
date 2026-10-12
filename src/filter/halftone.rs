//! Multi-channel halftone mode (classic CMYK rosette or dominant colors).
//!
//! The partitioning algorithms produce a cloud of dots whose radii are
//! modulated by the local density, with colors close to the zonal average. The
//! halftone mode, on the contrary, splits the image into **N ink channels**
//! (Cyan, Magenta, Yellow, Black — or N dominant colors extracted by k-means),
//! then places an **independent rotated screen per channel**. The dots of a
//! channel have the full color of their ink (e.g. Cyan = [0, 255, 255]) and a
//! radius proportional to the ink coverage at their position.
//!
//! The final compositor stacks the channels in **multiply** mode (a
//! transmission mask), not an additive "over", so as to reproduce the
//! subtractive mixing of inks on white paper correctly.

use crate::filter::params::Dot;
use image::RgbImage;
use image::{Rgb, Rgb32FImage};
use serde::{Deserialize, Serialize};

// ─── Public API types (they live here, referenced by params.rs) ───────────────

/// Ink channel separation mode. `Off` keeps the historical pipeline.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum HalftoneMode {
    /// Historical pipeline: 1 dot per zone, mean color. Default.
    #[default]
    Off,
    /// 4 channels Cyan, Magenta, Yellow, Black with the classic angles 15°/75°/0°/45°.
    Cmyk {
        /// Angles [cyan, magenta, yellow, black] in degrees.
        angles: [f32; 4],
    },
    /// N channels extracted via k-means over the pixels of the source image.
    Dominant {
        /// Number of dominant channels (≥ 2).
        n: usize,
        /// Base angle in degrees. The following channels are placed at
        /// `base + i × 180° / n` (uniform spacing avoiding moiré).
        base_angle_deg: f32,
    },
}

/// Screening mode (placement of the dots per channel).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum Screening {
    /// AM: regular grid rotated per channel (rosette).
    #[default]
    Am,
    /// FM: stochastic blue noise, variable density.
    Fm,
}

// ─── Internal types (channels, config) ────────────────────────────────────────

/// Ink channel ready to be screened.
pub(crate) struct InkChannel {
    /// Full color of the ink (will be the `Dot::color`).
    pub color: [u8; 3],
    /// AM screen angle in degrees (ignored in FM).
    pub angle_deg: f32,
    /// Per-pixel coverage quantized on 8 bits (0..=255), in the order
    /// `(y * width + x)`. `u8` storage divides the memory of the coverage maps
    /// by 4 compared to `f32`, which is critical for dominant mode (up to
    /// `n` full-resolution maps).
    pub coverage: Vec<u8>,
}

/// Quantifies a floating coverage ∈ [0, 1] onto 8 bits.
#[inline]
fn cov_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ─── Centralized halftone parameters ──────────────────────────────────────────

/// View over the halftone fields of `FilterParams` to avoid CLI plumbing.
pub struct HalftoneConfig<'a> {
    pub mode: &'a HalftoneMode,
    pub screening: Screening,
    /// AM screen frequency in "cells per min(W,H)" (× ratio).
    /// 60 → about 60 dots along the smallest dimension (60 lpp screen).
    pub screen_frequency: f32,
    #[allow(dead_code)]
    pub dot_shape: crate::filter::params::DotShape,
    /// Minimum halftone dot radius (fraction of `min(W,H)`).
    pub min_radius_ratio: f32,
    /// Maximum halftone dot radius (fraction of the screen step). > 1.0 → the dots touch.
    pub max_dot_ratio: f32,
    /// RNG seed for FM screening (determinism).
    pub rng_seed: Option<u64>,
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Produces the whole set of halftone dots (all channels together) to paint in
/// the order in which they were generated. For the multiply render, the order
/// does not matter (commutative blending).
///
/// Every dot carries the color of its channel: when rendering, multiply will
/// simulate the subtractive absorption.
pub(crate) fn dots_halftone(src: &RgbImage, cfg: &HalftoneConfig) -> Vec<Dot> {
    let channels = match cfg.mode {
        HalftoneMode::Off => return Vec::new(),
        HalftoneMode::Cmyk { angles } => separate_cmyk(src, *angles),
        HalftoneMode::Dominant { n, base_angle_deg } => {
            separate_dominant(src, *n, *base_angle_deg, cfg.rng_seed)
        }
    };
    let mut out = Vec::new();
    for ch in channels {
        let dots = match cfg.screening {
            Screening::Am => screen_am(src, ch, cfg),
            Screening::Fm => screen_fm(src, ch, cfg),
        };
        out.extend(dots);
    }
    out
}

/// High-precision CMYK separation for gamma-corrected halftone rendering.
/// Dominant-color screening keeps the historical RGB8 path for now because its
/// palette extraction is intentionally defined in display RGB space.
pub(crate) fn dots_halftone_linear(src: &Rgb32FImage, cfg: &HalftoneConfig) -> Vec<Dot> {
    let HalftoneMode::Cmyk { angles } = cfg.mode else {
        let rgb = RgbImage::from_fn(src.width(), src.height(), |x, y| {
            let p = src.get_pixel(x, y);
            Rgb([
                (p[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (p[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (p[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ])
        });
        return dots_halftone(&rgb, cfg);
    };
    let channels = separate_cmyk_linear(src, *angles);
    let mut out = Vec::new();
    for channel in channels {
        let dots = match cfg.screening {
            Screening::Am => screen_am_dimensions(src.width(), src.height(), channel, cfg),
            Screening::Fm => screen_fm_dimensions(src.width(), src.height(), channel, cfg),
        };
        out.extend(dots);
    }
    out
}

// ─── CMYK separation ─────────────────────────────────────────────────────────

/// RGB → CMYK conversion with simple Under Color Removal (UCR):
/// when C+M+Y ≥ threshold, the neutral component (min(C,M,Y)) is replaced with Black.
fn separate_cmyk(src: &RgbImage, angles: [f32; 4]) -> Vec<InkChannel> {
    let (w, h) = src.dimensions();
    let n = (w as usize) * (h as usize);

    let mut c_cov = vec![0u8; n];
    let mut m_cov = vec![0u8; n];
    let mut y_cov = vec![0u8; n];
    let mut k_cov = vec![0u8; n];

    // UCR threshold: when the C+M+Y sum exceeds this threshold, the common part
    // is replaced with Black. 0.3 = moderate start.
    const UCR_THRESHOLD: f32 = 0.30;

    for idx in 0..n {
        let x = idx as u32 % w;
        let y = idx as u32 / w;
        let p = src.get_pixel(x, y);
        let (r, g, b) = (
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
        );
        let k_inv = r.max(g).max(b);
        let k = 1.0 - k_inv; // "natural" black coverage (max RGB)
        // `scale` normalizes the CMY components by `1 - k = max(r,g,b)`.
        // - On pure black (k = 1.0), CMY is not needed → 0.
        // - On a saturated color (cyan = (0,1,1)), max=1 → scale=1.0, and c=1.
        //   The previous formula tested `k_inv < 1.0`, which skipped pure cyan.
        let scale = if k < 1.0 { 1.0 / k_inv } else { 0.0 };
        let mut c = (1.0 - r - k) * scale;
        let mut m = (1.0 - g - k) * scale;
        let mut y = (1.0 - b - k) * scale;
        // Clamp for safety.
        c = c.clamp(0.0, 1.0);
        m = m.clamp(0.0, 1.0);
        y = y.clamp(0.0, 1.0);
        // UCR: when C+M+Y exceeds the threshold, remove the common part.
        let sum = c + m + y;
        if sum > UCR_THRESHOLD {
            let grey = c.min(m).min(y); // neutral component = gray
            c -= grey * 0.5; // removes 50% of the neutral component (partial UCR)
            m -= grey * 0.5;
            y -= grey * 0.5;
            // Black takes over the component that replaces the removed neutral
            // component. The total ink coverage stays the same on screen: that
            // is the UCR goal. (Approximation: the exact gain is not computed.)
        }
        c_cov[idx] = cov_u8(c);
        m_cov[idx] = cov_u8(m);
        y_cov[idx] = cov_u8(y);
        k_cov[idx] = cov_u8(k);
    }

    vec![
        InkChannel {
            color: [0, 255, 255],
            angle_deg: angles[0],
            coverage: c_cov,
        },
        InkChannel {
            color: [255, 0, 255],
            angle_deg: angles[1],
            coverage: m_cov,
        },
        InkChannel {
            color: [255, 255, 0],
            angle_deg: angles[2],
            coverage: y_cov,
        },
        InkChannel {
            color: [0, 0, 0],
            angle_deg: angles[3],
            coverage: k_cov,
        },
    ]
}

fn separate_cmyk_linear(src: &Rgb32FImage, angles: [f32; 4]) -> Vec<InkChannel> {
    let (w, h) = src.dimensions();
    let n = (w as usize) * (h as usize);
    let mut c_cov = vec![0u8; n];
    let mut m_cov = vec![0u8; n];
    let mut y_cov = vec![0u8; n];
    let mut k_cov = vec![0u8; n];
    const UCR_THRESHOLD: f32 = 0.30;

    for idx in 0..n {
        let p = src.get_pixel(idx as u32 % w, idx as u32 / w);
        let (r, g, b) = (
            p[0].clamp(0.0, 1.0),
            p[1].clamp(0.0, 1.0),
            p[2].clamp(0.0, 1.0),
        );
        let k_inv = r.max(g).max(b);
        let k = 1.0 - k_inv;
        let scale = if k < 1.0 { 1.0 / k_inv.max(1e-6) } else { 0.0 };
        let mut c = ((1.0 - r - k) * scale).clamp(0.0, 1.0);
        let mut m = ((1.0 - g - k) * scale).clamp(0.0, 1.0);
        let mut y = ((1.0 - b - k) * scale).clamp(0.0, 1.0);
        let sum = c + m + y;
        if sum > UCR_THRESHOLD {
            let grey = c.min(m).min(y);
            c -= grey * 0.5;
            m -= grey * 0.5;
            y -= grey * 0.5;
        }
        c_cov[idx] = cov_u8(c);
        m_cov[idx] = cov_u8(m);
        y_cov[idx] = cov_u8(y);
        k_cov[idx] = cov_u8(k);
    }
    vec![
        InkChannel {
            color: [0, 255, 255],
            angle_deg: angles[0],
            coverage: c_cov,
        },
        InkChannel {
            color: [255, 0, 255],
            angle_deg: angles[1],
            coverage: m_cov,
        },
        InkChannel {
            color: [255, 255, 0],
            angle_deg: angles[2],
            coverage: y_cov,
        },
        InkChannel {
            color: [0, 0, 0],
            angle_deg: angles[3],
            coverage: k_cov,
        },
    ]
}

// ─── Dominant color separation (k-means) ──────────────────────────────────────

/// Deduplicates the pixel colors. Returns the unique colors (in order of
/// appearance) and, for every pixel, the index of its unique color.
///
/// The closest center only depends on the color: deduplicating avoids
/// repeating `n` multiplications per pixel (a large gain on flat graphic
/// images), without changing the result.
///
/// The order of appearance and the index of each color are preserved.
///
/// The library `HashMap` is enough: a hand-rolled open-addressing table
/// measured no gain (1.23 s on both sides on a 900×700 photo), the cost of
/// `Dominant` mode lying in the screening and not in the hashing.
fn unique_pixel_colors(src: &RgbImage) -> (Vec<[u8; 3]>, Vec<u32>) {
    use std::collections::HashMap;
    let n_pixels = (src.width() as usize) * (src.height() as usize);
    let mut unique: Vec<[u8; 3]> = Vec::new();
    let mut mapping: Vec<u32> = Vec::with_capacity(n_pixels);
    // Capacity ceiling: avoids reserving millions of entries at once.
    let mut index_of = HashMap::with_capacity(n_pixels.min(1 << 16));
    for p in src.pixels() {
        let color = [p[0], p[1], p[2]];
        let idx = *index_of.entry(color).or_insert_with(|| {
            unique.push(color);
            (unique.len() - 1) as u32
        });
        mapping.push(idx);
    }
    (unique, mapping)
}

/// K-means over a list of unique colors, with per-pixel accumulation in the
/// original order (stable f64 sums). `mapping[pixel]` gives the index of the
/// unique color of the pixel.
fn compute_dominant_centers_dedup(
    unique: &[[u8; 3]],
    mapping: &[u32],
    n: usize,
    rng_seed: Option<u64>,
) -> Vec<[u8; 3]> {
    use rayon::prelude::*;
    let n_unique = unique.len();
    let n_pixels = mapping.len();
    if n_pixels == 0 || n == 0 || n_unique == 0 {
        return Vec::new();
    }

    // Initialization: uniform sampling of the unique colors as centers
    let step = (n_unique / n.max(1)).max(1);
    let mut centers: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let c = unique[(i * step).min(n_unique - 1)];
            [c[0] as f32, c[1] as f32, c[2] as f32]
        })
        .collect();

    // K-means (max 10 iterations, early stop). Reuses the pattern of
    // `render::compute_centers` but applied to the source pixels.
    for _ in 0..10 {
        // Best center per unique color: a single search per color.
        let best_per_unique: Vec<u32> = unique
            .par_iter()
            .map(|c| nearest_center(&centers, &[c[0] as f32, c[1] as f32, c[2] as f32]) as u32)
            .collect();

        let (sums, counts) = (0..n_pixels)
            .into_par_iter()
            .fold(
                || (vec![[0f64; 3]; n], vec![0u64; n]),
                |(mut sums, mut counts), idx| {
                    let u = mapping[idx] as usize;
                    let best = best_per_unique[u] as usize;
                    let c = unique[u];
                    sums[best][0] += c[0] as f64;
                    sums[best][1] += c[1] as f64;
                    sums[best][2] += c[2] as f64;
                    counts[best] += 1;
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![[0f64; 3]; n], vec![0u64; n]),
                |(mut s, mut c), (bs, bc)| {
                    for i in 0..n {
                        for j in 0..3 {
                            s[i][j] += bs[i][j];
                        }
                        c[i] += bc[i];
                    }
                    (s, c)
                },
            );

        let mut converged = true;
        for i in 0..n {
            let cnt = counts[i] as f64;
            if cnt > 0.0 {
                let new_c = [
                    (sums[i][0] / cnt) as f32,
                    (sums[i][1] / cnt) as f32,
                    (sums[i][2] / cnt) as f32,
                ];
                let shift = (new_c[0] - centers[i][0]).powi(2)
                    + (new_c[1] - centers[i][1]).powi(2)
                    + (new_c[2] - centers[i][2]).powi(2);
                if shift > 1.0 {
                    converged = false;
                }
                centers[i] = new_c;
            }
        }
        if converged {
            break;
        }
    }
    let _ = rng_seed; // (reserved for a k-means++ init if one wants the variance)
    centers
        .iter()
        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
        .collect()
}

#[inline]
fn nearest_center(centers: &[[f32; 3]], p: &[f32; 3]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, c) in centers.iter().enumerate() {
        let dr = c[0] - p[0];
        let dg = c[1] - p[1];
        let db = c[2] - p[2];
        let d = dr * dr + dg * dg + db * db;
        if d < best_dist {
            best_dist = d;
            best = i;
            if d == 0.0 {
                break;
            }
        }
    }
    best
}

fn separate_dominant(
    src: &RgbImage,
    n: usize,
    base_angle_deg: f32,
    _rng_seed: Option<u64>,
) -> Vec<InkChannel> {
    let (w, h) = src.dimensions();
    let n_pixels = (w as usize) * (h as usize);
    let n = n.max(2);

    let (unique, mapping) = unique_pixel_colors(src);
    let colors = compute_dominant_centers_dedup(&unique, &mapping, n, _rng_seed);
    if colors.is_empty() {
        return Vec::new();
    }
    let centers_f: Vec<[f32; 3]> = colors
        .iter()
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect();

    // Best center per unique color (one search per color).
    let best_per_unique: Vec<u32> = unique
        .iter()
        .map(|c| nearest_center(&centers_f, &[c[0] as f32, c[1] as f32, c[2] as f32]) as u32)
        .collect();

    // Per-channel coverage: "hard assign" to the closest center, weighted by
    // similarity (1 - normalized_distance). u8 maps (÷4 memory).
    let mut coverages: Vec<Vec<u8>> = (0..n).map(|_| vec![0u8; n_pixels]).collect();
    for (idx, &u) in mapping.iter().enumerate() {
        let ui = u as usize;
        let best = best_per_unique[ui] as usize;
        let c = centers_f[best];
        let p = unique[ui];
        let prgb = [p[0] as f32, p[1] as f32, p[2] as f32];
        let dist_sq =
            (c[0] - prgb[0]).powi(2) + (c[1] - prgb[1]).powi(2) + (c[2] - prgb[2]).powi(2);
        // Similarity ∈ [0, 1]: 1 when identical, ~0 at the antipodes.
        let max_color_dist = 255.0 * 255.0 * 1.5; // maximum expected distance
        let sim = 1.0 - (dist_sq / max_color_dist).clamp(0.0, 1.0);
        // Boost the similarity so that an assignment close to the real pixel
        // gives a reasonable coverage (otherwise everything is too weak).
        let cov = (sim * 1.3).clamp(0.0, 1.0);
        coverages[best][idx] = cov_u8(cov);
    }

    // Angles spread uniformly per channel to avoid moiré.
    let angles = (0..n)
        .map(|i| base_angle_deg + (i as f32) * 180.0 / n as f32)
        .collect::<Vec<_>>();

    (0..n)
        .map(|i| InkChannel {
            color: colors[i],
            angle_deg: angles[i],
            coverage: std::mem::take(&mut coverages[i]),
        })
        .collect()
}

// ─── AM screening (rosette) ──────────────────────────────────────────────────
//
// For every channel, a rotated grid centered on the image is generated at the
// `screen_frequency` step, then one dot is emitted per grid intersection.
// The radius is proportional to the mean coverage inside the cell.

#[allow(clippy::too_many_lines)]
fn screen_am(src: &RgbImage, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let (w, h) = src.dimensions();
    screen_am_dimensions(w, h, channel, cfg)
}

fn screen_am_dimensions(w: u32, h: u32, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let img_min = w.min(h) as f32;
    // Screen step in pixels: the higher the frequency → the smaller the cell.
    let step = (img_min / cfg.screen_frequency).max(2.0);
    // Grid bounds in rotated space (wide enough to cover the whole image).
    let half_diag = ((w as f32).hypot(h as f32)) / 2.0 + step;
    let extent = (half_diag / step).ceil() as i32 + 1;

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let ang = channel.angle_deg * std::f32::consts::PI / 180.0;
    let (ux, uy) = (ang.cos(), ang.sin());
    let (vx, vy) = (-uy, ux);

    // Min/max radius in pixels.
    let r_min = cfg.min_radius_ratio * img_min;
    let r_max = cfg.max_dot_ratio * step; // ≤ step → the dots touch

    // Sampling cell = 1×1 pixel in image space (not step×step), because the
    // coverage map indexed by (x_pix, y_pix) is read directly. In practice the
    // average is taken over a disk of radius ≈ step/2 around the center.

    let mut dots = Vec::new();
    for j in -extent..=extent {
        for i in -extent..=extent {
            // Coordinates in the rotated frame (centered on the image).
            let sx = i as f32 * step;
            let sy = j as f32 * step;
            // Position in image space.
            let px = cx + sx * ux + sy * vx;
            let py = cy + sx * uy + sy * vy;
            if px < 0.0 || py < 0.0 || px >= w as f32 || py >= h as f32 {
                continue;
            }
            // Mean coverage in a disk of radius step/2 around (px, py).
            let cov = sample_coverage_disk(&channel.coverage, w, h, px, py, step * 0.5);
            if cov <= 0.005 {
                continue;
            }
            let radius = r_min + (r_max - r_min) * cov;
            // Anti-suppression of the too small ones.
            if radius < 0.5 {
                continue;
            }
            dots.push(Dot {
                x: px,
                y: py,
                color: channel.color,
                radius,
            });
        }
    }
    dots
}

/// Mean of `coverage` (quantized to u8) inside a disk centered on `(px, py)`
/// of radius `r`. Returns a value ∈ [0, 1].
fn sample_coverage_disk(coverage: &[u8], w: u32, h: u32, px: f32, py: f32, r: f32) -> f32 {
    let r_i = r.ceil() as i32;
    if r_i <= 0 {
        let ix = px as u32 % w;
        let iy = py as u32 % h;
        return coverage[(iy * w + ix) as usize] as f32 / 255.0;
    }
    let r2 = r * r;
    let mut sum = 0u32;
    let mut n = 0u32;
    let cx_i = px as i32;
    let cy_i = py as i32;
    for dy in -r_i..=r_i {
        let py_i = cy_i + dy;
        if py_i < 0 || py_i >= h as i32 {
            continue;
        }
        for dx in -r_i..=r_i {
            if (dx * dx + dy * dy) as f32 > r2 {
                continue;
            }
            let px_i = cx_i + dx;
            if px_i < 0 || px_i >= w as i32 {
                continue;
            }
            sum += coverage[(py_i as u32 * w + px_i as u32) as usize] as u32;
            n += 1;
        }
    }
    if n > 0 {
        sum as f32 / (n as f32 * 255.0)
    } else {
        // Fallback at the exact center.
        let ix = (px as u32).min(w - 1);
        let iy = (py as u32).min(h - 1);
        coverage[(iy * w + ix) as usize] as f32 / 255.0
    }
}

// ─── FM screening (stochastic) ───────────────────────────────────────────────
//
// Pragmatic variant of "frequency-modulated" screening: the rotated grid is kept
// (as in AM), but the coverage is examined at every point and the dot is emitted
// only with probability ≈ coverage. The radius is nearly constant (= r_max), it
// is the **density** of the dots that encodes the ink coverage (unlike in AM,
// where it is the **size** that encodes). The sub-pixel jitter breaks the
// regular look of the grid → direct effect → a more organic, less moiréd render.

fn screen_fm(src: &RgbImage, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let (w, h) = src.dimensions();
    screen_fm_dimensions(w, h, channel, cfg)
}

fn screen_fm_dimensions(w: u32, h: u32, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let img_min = w.min(h) as f32;
    let step = (img_min / cfg.screen_frequency).max(2.0);
    let half_diag = ((w as f32).hypot(h as f32)) / 2.0 + step;
    let extent = (half_diag / step).ceil() as i32 + 1;

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let ang = channel.angle_deg * std::f32::consts::PI / 180.0;
    let (ux, uy) = (ang.cos(), ang.sin());
    let (vx, vy) = (-uy, ux);

    let r_max = cfg.max_dot_ratio * step * 0.5;
    let r_min = cfg.min_radius_ratio * img_min;

    // Deterministic RNG per (channel, seed). A simple derivation from the seed
    // and from the channel color to get independent placements.
    let seed = cfg
        .rng_seed
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xdeadbeef)
        })
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(
            (channel.color[0] as u64) << 40
                | (channel.color[1] as u64) << 24
                | channel.color[2] as u64,
        );
    let mut rng: u64 = seed;

    let mut dots = Vec::new();
    for j in -extent..=extent {
        for i in -extent..=extent {
            // Coordinates in the rotated frame (centered on the image).
            let sx = i as f32 * step;
            let sy = j as f32 * step;
            // Position in image space.
            let px = cx + sx * ux + sy * vx;
            let py = cy + sx * uy + sy * vy;
            if px < 0.0 || py < 0.0 || px >= w as f32 || py >= h as f32 {
                continue;
            }
            let cov = sample_coverage_disk(&channel.coverage, w, h, px, py, step * 0.5);
            if cov <= 0.005 {
                continue;
            }
            // Stochastic acceptance: the dot is emitted with probability = cov.
            let thr = lcg_next(&mut rng);
            if thr > cov {
                continue;
            }
            // Light sub-pixel jitter to break the regularity of the grid.
            let jx = (lcg_next(&mut rng) - 0.5) * step * 0.4;
            let jy = (lcg_next(&mut rng) - 0.5) * step * 0.4;
            // Radius nearly constant (slight variation for naturalness).
            let r = r_min + (r_max - r_min) * (0.7 + 0.3 * lcg_next(&mut rng));
            if r < 0.5 {
                continue;
            }
            dots.push(Dot {
                x: px + jx,
                y: py + jy,
                color: channel.color,
                radius: r,
            });
        }
    }
    dots
}

/// LCG dedicated to FM screening (independent of the crate's
/// sampling::lcg_next to avoid breaking encapsulation). Same algorithm.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 32) as f32) / (u32::MAX as f32)
}
