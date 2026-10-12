use crate::filter::params::{Dot, DotShape, FilterParams};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use rayon::prelude::*;
use std::f32::consts::PI;

// ─── Common render ────────────────────────────────────────────────────────────

// ─── Anti-aliasing ───────────────────────────────────────────────────────────

/// Supersampling factor: 4×4 = 16 sub-pixels per pixel.
/// Quality/perf compromise: covers 16 opacity levels per edge pixel.
const AA_SS: u32 = 4;
const AA_TOTAL: u32 = AA_SS * AA_SS;
const AA_STEP: f32 = 1.0 / AA_SS as f32;

/// Coverage (in sixteenths of a pixel) of a pixel by a convex shape,
/// in local coords `(lx, ly)` centered on the dot.
///
/// Pretest of the 4 corners of the pixel: for a convex shape,
///   - 4 corners inside ⇒ fully covered pixel (fast path).
///   - 4 corners outside + centroid outside ⇒ fully uncovered pixel (skip).
///   - otherwise ⇒ 4×4 supersampling (slow path ≈ edge pixels).
///
/// `cx`, `cy`: float position of the dot center in image coords.
#[inline]
fn coverage_aa<F>(px: i32, py: i32, cx: f32, cy: f32, inside: F) -> u32
where
    F: Fn(f32, f32) -> bool,
{
    let lx_lo = px as f32 - 0.5 - cx;
    let lx_hi = lx_lo + 1.0;
    let ly_lo = py as f32 - 0.5 - cy;
    let ly_hi = ly_lo + 1.0;

    let c00 = inside(lx_lo, ly_lo);
    let c10 = inside(lx_hi, ly_lo);
    let c01 = inside(lx_lo, ly_hi);
    let c11 = inside(lx_hi, ly_hi);

    if c00 && c10 && c01 && c11 {
        return AA_TOTAL;
    }
    if !c00 && !c10 && !c01 && !c11 && !inside(lx_lo + 0.5, ly_lo + 0.5) {
        return 0;
    }

    let mut count = 0u32;
    for sy in 0..AA_SS {
        let ly = ly_lo + (sy as f32 + 0.5) * AA_STEP;
        for sx in 0..AA_SS {
            let lx = lx_lo + (sx as f32 + 0.5) * AA_STEP;
            if inside(lx, ly) {
                count += 1;
                if count == AA_TOTAL {
                    return AA_TOTAL;
                }
            }
        }
    }
    count
}

pub(crate) fn render(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbImage {
    let (w, h) = src.dimensions();
    let bg_rgb = params.bg_color;
    let mut dst = RgbImage::from_pixel(w, h, Rgb(bg_rgb));

    // With a palette + dithering, the original average colors of the dots are
    // kept and the rendered image is post-quantized with the chosen algorithm.
    let quantized: Vec<Dot>;
    let palette_centers: Vec<[u8; 3]>;
    let (dots_to_draw, dither_palette): (&[Dot], &[[u8; 3]]) = match params.palette_size {
        Some(n_colors) if params.dithering.is_on() => {
            let n = n_colors.max(2);
            palette_centers = quantize_palette_centers(dots, n);
            (dots, &palette_centers[..])
        }
        Some(n_colors) => {
            quantized = quantize_dots(dots, n_colors.max(2));
            (&quantized, &[][..])
        }
        None => (dots, &[][..]),
    };

    // Sort: the biggest dots first (background), the small ones last (foreground)
    let mut sorted: Vec<&Dot> = dots_to_draw.iter().collect();
    sorted.sort_unstable_by(|a, b| b.radius.total_cmp(&a.radius));

    draw_sorted_bands_rgb(&mut dst, &sorted, params.dot_shape);

    if !dither_palette.is_empty() {
        crate::filter::dither::apply(&mut dst, dither_palette, params.dithering);
    }
    dst
}

/// Draws the dots in parallel horizontal bands (by radius). Each band applies
/// the dots in the global order (decreasing radius) while bounding the pixels
/// to its own lines — a result identical to sequential drawing.
fn draw_sorted_bands_rgb(dst: &mut RgbImage, sorted: &[&Dot], shape: DotShape) {
    let (w, h) = dst.dimensions();
    let stride = w as usize * 3;
    let n_bands = (rayon::current_num_threads().max(1) * 4)
        .min(h as usize)
        .max(1);
    let band_rows = (h as usize).div_ceil(n_bands.max(1));
    let raw: &mut [u8] = dst.as_mut();
    raw.par_chunks_mut(band_rows * stride)
        .enumerate()
        .for_each(|(b, band)| {
            let y0 = b * band_rows;
            let y1 = y0.saturating_add(band_rows).min(h as usize);
            for dot in sorted {
                if dot.radius <= 0.0 {
                    continue;
                }
                let color = dot.color;
                draw_dot_clipped(
                    w,
                    h,
                    dot.x,
                    dot.y,
                    dot.radius,
                    shape,
                    (y0 as u32, y1 as u32),
                    // `impl FnMut` rather than `&mut dyn FnMut`: the closure
                    // is monomorphized with the call, so `(py - y0) * stride`
                    // is hoisted out of the loop over `px` and the call is no
                    // longer indirect. Identical output, one call fewer per
                    // blended pixel.
                    &mut |px: u32, py: u32, cov: u32| {
                        let off = (py as usize - y0) * stride + px as usize * 3;
                        blend_rgb_band(band, off, color, cov);
                    },
                );
            }
        });
}

/// RGB blend of a pixel (same formula as the sequential path).
#[inline]
fn blend_rgb_band(band: &mut [u8], off: usize, color: [u8; 3], coverage: u32) {
    debug_assert!(coverage > 0 && coverage <= AA_TOTAL);
    if coverage == AA_TOTAL {
        band[off..off + 3].copy_from_slice(&color);
        return;
    }
    let a = coverage as f32 / AA_TOTAL as f32;
    let inv = 1.0 - a;
    let p = &mut band[off..off + 3];
    for c in 0..3 {
        let v = p[c] as f32 * inv + color[c] as f32 * a + 0.5;
        p[c] = if v > 255.0 { 255 } else { v as u8 };
    }
}

/// Draws an anti-aliased dot according to the chosen shape, bounded to the
/// lines `y_clip = (y0, y1)` (out-of-clip lines ignored → band drawing).
///
/// `(cx_f, cy_f)`: center in pixels (float, preserves the sub-pixel position).
/// `r_f`: radius in pixels (float).
/// `blend(px, py, coverage)` paints the touched pixel (coverage ∈ (0, AA_TOTAL]).
#[allow(clippy::too_many_arguments)]
fn draw_dot_clipped(
    img_w: u32,
    img_h: u32,
    cx_f: f32,
    cy_f: f32,
    r_f: f32,
    shape: DotShape,
    y_clip: (u32, u32),
    blend: &mut impl FnMut(u32, u32, u32),
) {
    let bbox = match shape {
        DotShape::Ellipse { aspect, .. } => {
            let a = r_f;
            let b = (r_f / aspect.max(0.01)).max(1.0);
            (a.max(b).ceil() as i32) + 1
        }
        _ => (r_f.ceil() as i32) + 1,
    };
    let b = bbox as f32;
    let (iw, ih) = (img_w as i32, img_h as i32);

    let px_min = (cx_f - b).ceil() as i32;
    let px_max = (cx_f + b).floor() as i32;
    let py_min = (cy_f - b).ceil() as i32;
    let py_max = (cy_f + b).floor() as i32;
    // Clip to the y_clip band then to the image bounds.
    let px0 = px_min.max(0);
    let px1 = px_max.min(iw - 1);
    let py0 = py_min.max(y_clip.0 as i32).max(0);
    let py1 = py_max.min(y_clip.1 as i32 - 1).min(ih - 1);
    if px0 > px1 || py0 > py1 {
        return;
    }

    match shape {
        DotShape::Circle => {
            let r2 = r_f * r_f;
            for py in py0..=py1 {
                for px in px0..=px1 {
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| lx * lx + ly * ly <= r2);
                    if cov > 0 {
                        blend(px as u32, py as u32, cov);
                    }
                }
            }
        }
        DotShape::Square => {
            for py in py0..=py1 {
                for px in px0..=px1 {
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        lx.abs() <= r_f && ly.abs() <= r_f
                    });
                    if cov > 0 {
                        blend(px as u32, py as u32, cov);
                    }
                }
            }
        }
        DotShape::Ellipse { aspect, angle_deg } => {
            let a = r_f;
            let b_axis = (r_f / aspect.max(0.01)).max(1.0);
            let ang = angle_deg * PI / 180.0;
            let (cos_a, sin_a) = (ang.cos(), ang.sin());
            for py in py0..=py1 {
                for px in px0..=px1 {
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        // Inverse rotation: frame of the ellipse.
                        let rx = lx * cos_a + ly * sin_a;
                        let ry = -lx * sin_a + ly * cos_a;
                        (rx / a).powi(2) + (ry / b_axis).powi(2) <= 1.0
                    });
                    if cov > 0 {
                        blend(px as u32, py as u32, cov);
                    }
                }
            }
        }
        DotShape::RegularPolygon { sides } => {
            let n = sides.max(3) as usize;
            for py in py0..=py1 {
                for px in px0..=px1 {
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        point_in_regular_polygon(lx, ly, r_f, n)
                    });
                    if cov > 0 {
                        blend(px as u32, py as u32, cov);
                    }
                }
            }
        }
    }
}

/// Tests whether (px, py) is inside a regular polygon with `n` sides of radius `r`.
pub(crate) fn point_in_regular_polygon(px: f32, py: f32, r: f32, n: usize) -> bool {
    let d = (px * px + py * py).sqrt();
    if d > r {
        return false;
    }
    if d == 0.0 {
        return true;
    }
    let angle = py.atan2(px);
    let sector_angle = 2.0 * PI / n as f32;
    let sector = ((angle % sector_angle) + sector_angle) % sector_angle;
    let apothem_angle = sector - sector_angle / 2.0;
    let apothem = r * (PI / n as f32).cos();
    let dist_to_edge = apothem / apothem_angle.cos().abs();
    d <= dist_to_edge
}

/// K-means over the colors of the `dots`, returning `n` centers in f32.
/// Common factor between `quantize_dots` (nearest-color mapping) and
/// `quantize_palette_centers` (centers alone, for dithering).
fn compute_centers(dots: &[Dot], n: usize) -> Vec<[f32; 3]> {
    if dots.is_empty() || n >= dots.len() {
        return Vec::new();
    }
    // Initialization: uniform subsampling of the dots as centers
    let step = dots.len() / n;
    let mut centers: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let c = dots[i * step].color;
            [c[0] as f32, c[1] as f32, c[2] as f32]
        })
        .collect();

    // The colors of the dots do not change between iterations: they are
    // deduplicated once and the closest center is searched only once per
    // unique color. The distance only depends on the color, so the result
    // stays bit-for-bit identical to a per-dot search, but repeating `n`
    // multiplications for identical colors is avoided.
    let (unique_colors, dot_to_unique) = unique_dot_colors(dots);

    // Stop early when the color centers have converged.
    for _ in 0..10 {
        // Best center per unique color: increasing scan, the first minimum
        // wins (the behavior of `min_by` on the center index), with an early
        // exit on an exact match.
        let best_per_unique: Vec<usize> = unique_colors
            .iter()
            .map(|&color| {
                let cr = color[0] as f32;
                let cg = color[1] as f32;
                let cb = color[2] as f32;
                let mut best = 0usize;
                let mut best_dist = f32::INFINITY;
                for (i, c) in centers.iter().enumerate() {
                    let dr = c[0] - cr;
                    let dg = c[1] - cg;
                    let db = c[2] - cb;
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
            })
            .collect();

        let mut sums = vec![[0f64; 3]; n];
        let mut counts = vec![0u64; n];
        // Accumulation in the original order of the dots to preserve the f64
        // sum exactly (hence the bit-for-bit result).
        for (dot, &unique_idx) in dots.iter().zip(&dot_to_unique) {
            let best = best_per_unique[unique_idx];
            sums[best][0] += dot.color[0] as f64;
            sums[best][1] += dot.color[1] as f64;
            sums[best][2] += dot.color[2] as f64;
            counts[best] += 1;
        }

        let mut converged = true;
        for i in 0..n {
            let cnt = counts[i] as f64;
            if cnt > 0.0 {
                let new_center = [
                    (sums[i][0] / cnt) as f32,
                    (sums[i][1] / cnt) as f32,
                    (sums[i][2] / cnt) as f32,
                ];
                let shift = (new_center[0] - centers[i][0]).powi(2)
                    + (new_center[1] - centers[i][1]).powi(2)
                    + (new_center[2] - centers[i][2]).powi(2);
                if shift > 0.5 {
                    converged = false;
                }
                centers[i] = new_center;
            }
        }
        if converged {
            break;
        }
    }
    centers
}

/// Deduplicates the colors of the dots. Returns the list of unique colors (in
/// order of appearance) and, for every dot, the index of its unique color.
fn unique_dot_colors(dots: &[Dot]) -> (Vec<[u8; 3]>, Vec<usize>) {
    use std::collections::HashMap;
    let mut index_of: HashMap<[u8; 3], usize> = HashMap::with_capacity(dots.len());
    let mut unique = Vec::new();
    let mut mapping = Vec::with_capacity(dots.len());
    for dot in dots {
        let color = dot.color;
        let idx = *index_of.entry(color).or_insert_with(|| {
            unique.push(color);
            unique.len() - 1
        });
        mapping.push(idx);
    }
    (unique, mapping)
}

/// Palette centers in u8 (useful for Floyd-Steinberg dithering).
pub(crate) fn quantize_palette_centers(dots: &[Dot], n: usize) -> Vec<[u8; 3]> {
    compute_centers(dots, n)
        .iter()
        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
        .collect()
}

/// Quantizes the colors of the dots to `n` colors via simple k-means over RGB,
/// then remaps every dot to its closest center (nearest-color).
pub(crate) fn quantize_dots(dots: &[Dot], n: usize) -> Vec<Dot> {
    if dots.is_empty() || n >= dots.len() {
        return dots.to_vec();
    }
    let centers = compute_centers(dots, n);

    // Remap every dot to the color of its closest center
    dots.iter()
        .map(|dot| {
            let best_color = centers
                .iter()
                .min_by(|a, b| {
                    let da = (a[0] - dot.color[0] as f32).powi(2)
                        + (a[1] - dot.color[1] as f32).powi(2)
                        + (a[2] - dot.color[2] as f32).powi(2);
                    let db = (b[0] - dot.color[0] as f32).powi(2)
                        + (b[1] - dot.color[1] as f32).powi(2)
                        + (b[2] - dot.color[2] as f32).powi(2);
                    da.total_cmp(&db)
                })
                .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
                .unwrap_or(dot.color);
            Dot {
                color: best_color,
                ..*dot
            }
        })
        .collect()
}

// ─── Radius computation ──────────────────────────────────────────────────────

pub(crate) fn radius_for_dot(
    lum: f32,
    local_density: f32, // 0=uniform 1=detailed
    img_min_side: f32,
    params: &FilterParams,
) -> f32 {
    let r_min = params.min_radius_ratio * img_min_side;
    let r_max = params.max_radius_ratio * img_min_side;

    // Halftone: dark → large (lum=0 → r_max, lum=1 → r_min)
    let r_lum = r_max - (r_max - r_min) * lum;

    // Uniformity: flat zone → pushed even larger
    let uniformity = 1.0 - local_density;
    let boost = 1.0 + uniformity * params.variance_sensitivity * (params.max_boost - 1.0);
    (r_lum * boost).min(params.max_boost * r_max)
}

// ─── RGBA rendering (transparent background) ─────────────────────────────────

/// RGBA variant of `render`. When `params.transparent`, the initial background
/// is transparent (alpha 0); the pixels not covered by a dot stay transparent.
/// Otherwise the initial background is `bg_color` opaque (identical to `render`
/// but in RGBA).
pub fn render_rgba(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbaImage {
    let (w, h) = src.dimensions();
    let bg_pixel = if params.transparent {
        Rgba([
            params.bg_color[0],
            params.bg_color[1],
            params.bg_color[2],
            0,
        ])
    } else {
        Rgba([
            params.bg_color[0],
            params.bg_color[1],
            params.bg_color[2],
            255,
        ])
    };
    let mut dst = RgbaImage::from_pixel(w, h, bg_pixel);

    let quantized: Vec<Dot>;
    let palette_centers: Vec<[u8; 3]>;
    let (dots_to_draw, dither_palette): (&[Dot], &[[u8; 3]]) = match params.palette_size {
        Some(n_colors) if params.dithering.is_on() => {
            palette_centers = quantize_palette_centers(dots, n_colors.max(2));
            (dots, &palette_centers)
        }
        Some(n_colors) => {
            quantized = quantize_dots(dots, n_colors.max(2));
            (&quantized, &[])
        }
        None => (dots, &[]),
    };

    let mut sorted: Vec<&Dot> = dots_to_draw.iter().collect();
    sorted.sort_unstable_by(|a, b| b.radius.total_cmp(&a.radius));

    draw_sorted_bands_rgba(&mut dst, &sorted, params.dot_shape);

    if !dither_palette.is_empty() {
        // The dithering works on RGB; the already computed alpha is preserved.
        let mut rgb = RgbImage::from_fn(w, h, |x, y| {
            let p = dst.get_pixel(x, y);
            Rgb([p[0], p[1], p[2]])
        });
        crate::filter::dither::apply(&mut rgb, dither_palette, params.dithering);
        for (x, y, pixel) in rgb.enumerate_pixels() {
            if dst.get_pixel(x, y)[3] != 0 {
                let out = dst.get_pixel_mut(x, y);
                out[0] = pixel[0];
                out[1] = pixel[1];
                out[2] = pixel[2];
            }
        }
    }
    dst
}

/// Draws the RGBA dots in parallel horizontal bands (by radius).
/// Each band applies the dots in the global order (decreasing radius) while
/// bounding the pixels to its own lines — a result identical to sequential
/// drawing.
fn draw_sorted_bands_rgba(dst: &mut RgbaImage, sorted: &[&Dot], shape: DotShape) {
    let (w, h) = dst.dimensions();
    let stride = w as usize * 4;
    let n_bands = (rayon::current_num_threads().max(1) * 4)
        .min(h as usize)
        .max(1);
    let band_rows = (h as usize).div_ceil(n_bands.max(1));
    let raw: &mut [u8] = dst.as_mut();
    raw.par_chunks_mut(band_rows * stride)
        .enumerate()
        .for_each(|(b, band)| {
            let y0 = b * band_rows;
            let y1 = y0.saturating_add(band_rows).min(h as usize);
            for dot in sorted {
                if dot.radius <= 0.0 {
                    continue;
                }
                let color = dot.color;
                draw_dot_clipped(
                    w,
                    h,
                    dot.x,
                    dot.y,
                    dot.radius,
                    shape,
                    (y0 as u32, y1 as u32),
                    &mut |px: u32, py: u32, cov: u32| {
                        let off = (py as usize - y0) * stride + px as usize * 4;
                        blend_rgba_band(band, off, color, cov);
                    },
                );
            }
        });
}

/// RGBA blend ("over" operator on alpha) of a pixel (same formula as the
/// sequential path).
#[inline]
fn blend_rgba_band(band: &mut [u8], off: usize, color: [u8; 3], coverage: u32) {
    debug_assert!(coverage > 0 && coverage <= AA_TOTAL);
    if coverage == AA_TOTAL {
        band[off..off + 3].copy_from_slice(&color);
        band[off + 3] = 255;
        return;
    }
    let src_a = coverage as f32 / AA_TOTAL as f32;
    let p = &mut band[off..off + 4];
    let dst_a = p[3] as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a <= 0.0 {
        return;
    }
    let inv_out = 1.0 / out_a;
    let blend_w = dst_a * (1.0 - src_a) * inv_out;
    let src_w = src_a * inv_out;
    for c in 0..3 {
        let v = color[c] as f32 * src_w + p[c] as f32 * blend_w + 0.5;
        p[c] = if v > 255.0 { 255 } else { v as u8 };
    }
    p[3] = (out_a * 255.0 + 0.5) as u8;
}

// ─── Halftone rendering (multiply composite, ink-over-ink) ────────────────────

/// Paints a dot on an RGBA canvas in **multiply** mode: every already painted
/// pixel is multiplied by the transmission mask of the ink (modulated by the
/// alpha coverage).
///
/// This simulates the subtractive mixing of inks: starting from white paper,
/// two successive inks make each of them absorb a complementary part. Unlike
/// "over" mode, the result of Magenta + Yellow is truly Red and not a
/// yellow/magenta stacking.
fn blend_multiply_coverage(dst: &mut RgbaImage, px: u32, py: u32, ink: [u8; 3], coverage: u32) {
    debug_assert!(coverage > 0 && coverage <= AA_TOTAL);
    // alpha in [0, 1] = fraction of the pixel covered by the dot.
    let a = coverage as f32 / AA_TOTAL as f32;
    if a >= 1.0 {
        // Multiply pleine : dst[i] = dst[i] * mask[i] / 255.
        let p = dst.get_pixel_mut(px, py);
        p[0] = ((p[0] as u32 * ink[0] as u32) / 255) as u8;
        p[1] = ((p[1] as u32 * ink[1] as u32) / 255) as u8;
        p[2] = ((p[2] as u32 * ink[2] as u32) / 255) as u8;
        // In multiply mode, the paper stays opaque: alpha is forced to 255.
        p[3] = 255;
        return;
    }
    let p = dst.get_pixel_mut(px, py);
    for c in 0..3 {
        let dst_v = p[c] as f32;
        let mask = ink[c] as f32 / 255.0;
        // dst_after_full_ink = dst_v * mask
        // blended = dst_v * (1 - a) + dst_after_full_ink * a
        //         = dst_v * (1 - a + a * mask)
        let factor = 1.0 - a + a * mask;
        let v = dst_v * factor + 0.5;
        p[c] = if v > 255.0 { 255 } else { v as u8 };
    }
    p[3] = 255;
}

fn draw_dot_halftone(
    dst: &mut RgbaImage,
    cx_f: f32,
    cy_f: f32,
    r_f: f32,
    ink: [u8; 3],
    shape: DotShape,
) {
    let (iw, ih) = dst.dimensions();

    let bbox = match shape {
        DotShape::Ellipse { aspect, .. } => {
            let a_axis = r_f;
            let b_axis = (r_f / aspect.max(0.01)).max(1.0);
            (a_axis.max(b_axis).ceil() as i32) + 1
        }
        _ => (r_f.ceil() as i32) + 1,
    };
    let b = bbox as f32;

    let px_min = (cx_f - b).ceil() as i32;
    let px_max = (cx_f + b).floor() as i32;
    let py_min = (cy_f - b).ceil() as i32;
    let py_max = (cy_f + b).floor() as i32;

    match shape {
        DotShape::Circle => {
            let r2 = r_f * r_f;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| lx * lx + ly * ly <= r2);
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
        DotShape::Square => {
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        lx.abs() <= r_f && ly.abs() <= r_f
                    });
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
        DotShape::Ellipse { aspect, angle_deg } => {
            let a_axis = r_f;
            let b_axis = (r_f / aspect.max(0.01)).max(1.0);
            let ang = angle_deg * PI / 180.0;
            let (cos_a, sin_a) = (ang.cos(), ang.sin());
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        let rx = lx * cos_a + ly * sin_a;
                        let ry = -lx * sin_a + ly * cos_a;
                        (rx / a_axis).powi(2) + (ry / b_axis).powi(2) <= 1.0
                    });
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
        DotShape::RegularPolygon { sides } => {
            let n = sides.max(3) as usize;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        point_in_regular_polygon(lx, ly, r_f, n)
                    });
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
    }
}

/// Halftone rendering: starts from the paper (bg_color, or white if
/// multiplicative), and applies every dot in multiply mode. The order does not
/// matter (commutative). The uncovered pixels stay paper.
pub fn render_halftone(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbaImage {
    let (w, h) = src.dimensions();
    // Paper: in halftone white is the default starting point (multiply does not
    // work on black). If the user has explicitly requested a color, it is used.
    let paper = if params.transparent {
        Rgba([255, 255, 255, 0])
    } else {
        Rgba([
            params.bg_color[0],
            params.bg_color[1],
            params.bg_color[2],
            255,
        ])
    };
    let mut dst = RgbaImage::from_pixel(w, h, paper);

    // No sorting needed: multiply is commutative. But the generation order
    // (per channel) is kept for consistency with the Jest-like tests.
    for dot in dots {
        if dot.radius <= 0.0 {
            continue;
        }
        draw_dot_halftone(
            &mut dst,
            dot.x,
            dot.y,
            dot.radius,
            dot.color,
            params.dot_shape,
        );
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Old implementation (iterator + `min_by`) kept as a reference to verify
    /// that the deduplicated optimization stays bit-for-bit identical.
    fn compute_centers_reference(dots: &[Dot], n: usize) -> Vec<[f32; 3]> {
        if dots.is_empty() || n >= dots.len() {
            return Vec::new();
        }
        let step = dots.len() / n;
        let mut centers: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let c = dots[i * step].color;
                [c[0] as f32, c[1] as f32, c[2] as f32]
            })
            .collect();

        for _ in 0..10 {
            let mut sums = vec![[0f64; 3]; n];
            let mut counts = vec![0u64; n];
            for dot in dots {
                let best = centers
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let dr = c[0] - dot.color[0] as f32;
                        let dg = c[1] - dot.color[1] as f32;
                        let db = c[2] - dot.color[2] as f32;
                        (i, dr * dr + dg * dg + db * db)
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                sums[best][0] += dot.color[0] as f64;
                sums[best][1] += dot.color[1] as f64;
                sums[best][2] += dot.color[2] as f64;
                counts[best] += 1;
            }

            let mut converged = true;
            for i in 0..n {
                let cnt = counts[i] as f64;
                if cnt > 0.0 {
                    let new_center = [
                        (sums[i][0] / cnt) as f32,
                        (sums[i][1] / cnt) as f32,
                        (sums[i][2] / cnt) as f32,
                    ];
                    let shift = (new_center[0] - centers[i][0]).powi(2)
                        + (new_center[1] - centers[i][1]).powi(2)
                        + (new_center[2] - centers[i][2]).powi(2);
                    if shift > 0.5 {
                        converged = false;
                    }
                    centers[i] = new_center;
                }
            }
            if converged {
                break;
            }
        }
        centers
    }

    fn dot(x: f32, y: f32, color: [u8; 3]) -> Dot {
        Dot {
            x,
            y,
            color,
            radius: 1.0,
        }
    }

    #[test]
    fn deduplicated_centers_match_reference_bit_for_bit() {
        // Colors drawn from a small palette → many duplicates, which exercises
        // the deduplication.
        let palette: [[u8; 3]; 8] = [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
            [200, 30, 90],
            [10, 220, 30],
        ];
        let mut state = 0x1234_5678u64;
        let dots: Vec<Dot> = (0..500)
            .map(|i| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let color = palette[(state >> 33) as usize % palette.len()];
                dot(i as f32, i as f32, color)
            })
            .collect();

        for n in [2usize, 3, 7, 16] {
            let optimized = compute_centers(&dots, n);
            let reference = compute_centers_reference(&dots, n);
            assert_eq!(optimized.len(), reference.len(), "taille pour n={n}");
            for (a, b) in optimized.iter().zip(&reference) {
                assert_eq!(a, b, "centres divergents pour n={n}");
            }
        }
    }

    #[test]
    fn unique_dot_colors_groups_identical_colors() {
        let dots = vec![
            dot(0.0, 0.0, [1, 2, 3]),
            dot(1.0, 1.0, [4, 5, 6]),
            dot(2.0, 2.0, [1, 2, 3]),
        ];
        let (unique, mapping) = unique_dot_colors(&dots);
        assert_eq!(unique, vec![[1, 2, 3], [4, 5, 6]]);
        assert_eq!(mapping, vec![0, 1, 0]);
    }
}
