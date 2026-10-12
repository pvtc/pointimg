use image::{GrayImage, Luma, RgbImage};
use rayon::prelude::*;

/// Average density of a rectangular zone of the density map.
pub(crate) fn zone_density(
    density: &[f32],
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    img_w: u32,
    img_h: u32,
) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for py in y0..(y0 + h).min(img_h) {
        for px in x0..(x0 + w).min(img_w) {
            sum += density[(py * img_w + px) as usize];
            n += 1;
        }
    }
    if n > 0 { sum / n as f32 } else { 1.0 }
}

// ─── Density map ─────────────────────────────────────────────────────────────
//
// Local variance in a 9×9 neighborhood, normalized 0→1.
// sensitivity=0 → everything to 1 (uniform), sensitivity=1 → pure variance.
//
// A luminance SAT is sufficient for detail redistribution and uses two tables
// instead of six RGB tables. Keep f64 because global prefix sums can be large
// on the maximum supported image and are later subtracted to get local sums.

/// Local detail map (0 = uniform, 1 = detailed), normalized.
///
/// Local 9×9 variant in sum of squares, computed with integral tables.
/// Employs the WGSL shader when the `gpu` feature is enabled, `POINTIMG_GPU=1`
/// is set and a usable adapter exists; otherwise the CPU/SAT path.
pub fn compute_density_map(src: &RgbImage, sensitivity: f32) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    if let Some(density) = crate::filter::gpu::compute_density_map(src, sensitivity) {
        return density;
    }
    compute_density_map_cpu(src, sensitivity)
}

/// Normalizes a raw local variance into a density map (curve shared between
/// the CPU and GPU paths to guarantee their parity).
///
/// The normalization happens **in place**: on an image at the ceiling this
/// avoids a second full-resolution `Vec<f32>` allocation (~33 MB) coexisting
/// with the first one.
pub(crate) fn normalize_variance(mut raw: Vec<f32>, sensitivity: f32) -> Vec<f32> {
    // Keep a small floor to preserve contrast on nearly uniform images.
    let max_var = raw.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    for value in &mut raw {
        let norm = (*value / max_var).sqrt();
        // sensitivity=0 -> 1.0 everywhere; sensitivity=1 -> pure norm
        *value = 1.0 - sensitivity * (1.0 - norm);
    }
    raw
}

/// Historical CPU path (Summed-Area Tables), always available.
pub fn compute_density_map_cpu(src: &RgbImage, sensitivity: f32) -> Vec<f32> {
    let (width, height) = src.dimensions();
    let radius: usize = 4;
    let w = width as usize;
    let h = height as usize;
    let n_pixels = w * h;

    // Summed-area tables make each neighborhood query constant-time.
    // 2 tables: luminance sum and luminance sum-of-squares.
    // SAT uses (w+1)x(h+1) layout with zero padding row/col for branchless queries.
    // Replaces the O(81) per-pixel loop with O(1) per-pixel after O(n) precomputation.
    let sat_len = (w + 1) * (h + 1);
    let mut sat_l = vec![0f64; sat_len];
    let mut sat_l2 = vec![0f64; sat_len];

    let stride = w + 1;
    for y in 0..h {
        for x in 0..w {
            let p = src.get_pixel(x as u32, y as u32);
            let luminance = 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
            let idx = (y + 1) * stride + (x + 1);
            let left = idx - 1;
            let up = idx - stride;
            let diag = up - 1;
            sat_l[idx] = luminance + sat_l[left] + sat_l[up] - sat_l[diag];
            sat_l2[idx] = luminance * luminance + sat_l2[left] + sat_l2[up] - sat_l2[diag];
        }
    }

    // Query helper: sum over rect [x0,y0]..[x1,y1] inclusive (0-indexed pixel coords)
    let sat_query = |sat: &[f64], x0: usize, y0: usize, x1: usize, y1: usize| -> f64 {
        sat[(y1 + 1) * stride + (x1 + 1)]
            - sat[y0 * stride + (x1 + 1)]
            - sat[(y1 + 1) * stride + x0]
            + sat[y0 * stride + x0]
    };

    // ── Compute variance per pixel using SAT (O(1) per pixel) ───────────
    let raw: Vec<f32> = (0..n_pixels)
        .into_par_iter()
        .map(|idx| {
            let px = idx % w;
            let py = idx / w;
            let x0 = px.saturating_sub(radius);
            let y0 = py.saturating_sub(radius);
            let x1 = (px + radius).min(w - 1);
            let y1 = (py + radius).min(h - 1);
            let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
            if n == 0.0 {
                return 0.0;
            }
            let sum = sat_query(&sat_l, x0, y0, x1, y1);
            let sum_squared = sat_query(&sat_l2, x0, y0, x1, y1);
            (sum_squared / n - (sum / n).powi(2)).max(0.0) as f32
        })
        .collect();

    normalize_variance(raw, sensitivity)
}

/// Returns the normalized density map as a grayscale image.
/// Useful for previewing in the GUI.
pub fn compute_density_image(src: &RgbImage, sensitivity: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let density = compute_density_map(src, sensitivity);
    density_to_image(&density, w, h)
}

/// Renders the density map as a grayscale image (GUI preview).
pub fn density_to_image(density: &[f32], w: u32, h: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        let v = density[(y * w + x) as usize];
        // Rounding consistent with the other u8 conversions of the project.
        Luma([(v.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}
