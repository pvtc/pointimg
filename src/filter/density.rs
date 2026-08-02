use image::{GrayImage, Luma, RgbImage};
use rayon::prelude::*;

/// Densité moyenne d'une zone rectangulaire de la density map.
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
// Variance locale dans un voisinage 9×9, normalisée 0→1.
// sensitivity=0 → tout à 1 (uniforme), sensitivity=1 → variance pure.
//
// Bug 3 corrigé : calcul en f64 pour éviter la troncature entière.

pub(crate) fn compute_density_map(src: &RgbImage, sensitivity: f32) -> Vec<f32> {
    let (width, height) = src.dimensions();
    let radius: usize = 4;
    let w = width as usize;
    let h = height as usize;
    let n_pixels = w * h;

    // ── Summed-area tables (SAT) — P1 ────────────────────────────────────
    // 6 tables: sum and sum-of-squares for R, G, B.
    // SAT uses (w+1)x(h+1) layout with zero padding row/col for branchless queries.
    // Replaces the O(81) per-pixel loop with O(1) per-pixel after O(n) precomputation.
    let sat_len = (w + 1) * (h + 1);
    let mut sat_r = vec![0f64; sat_len];
    let mut sat_g = vec![0f64; sat_len];
    let mut sat_b = vec![0f64; sat_len];
    let mut sat_r2 = vec![0f64; sat_len];
    let mut sat_g2 = vec![0f64; sat_len];
    let mut sat_b2 = vec![0f64; sat_len];

    let stride = w + 1;
    for y in 0..h {
        for x in 0..w {
            let p = src.get_pixel(x as u32, y as u32);
            let (pr, pg, pb) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let idx = (y + 1) * stride + (x + 1);
            let left = idx - 1;
            let up = idx - stride;
            let diag = up - 1;
            sat_r[idx] = pr + sat_r[left] + sat_r[up] - sat_r[diag];
            sat_g[idx] = pg + sat_g[left] + sat_g[up] - sat_g[diag];
            sat_b[idx] = pb + sat_b[left] + sat_b[up] - sat_b[diag];
            sat_r2[idx] = pr * pr + sat_r2[left] + sat_r2[up] - sat_r2[diag];
            sat_g2[idx] = pg * pg + sat_g2[left] + sat_g2[up] - sat_g2[diag];
            sat_b2[idx] = pb * pb + sat_b2[left] + sat_b2[up] - sat_b2[diag];
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
            let sr = sat_query(&sat_r, x0, y0, x1, y1);
            let sg = sat_query(&sat_g, x0, y0, x1, y1);
            let sb = sat_query(&sat_b, x0, y0, x1, y1);
            let sr2 = sat_query(&sat_r2, x0, y0, x1, y1);
            let sg2 = sat_query(&sat_g2, x0, y0, x1, y1);
            let sb2 = sat_query(&sat_b2, x0, y0, x1, y1);
            // Var = E[X^2] - E[X]^2  — en f64 pour eviter troncature (bug 3 corrige)
            let var_r = (sr2 / n - (sr / n).powi(2)).max(0.0);
            let var_g = (sg2 / n - (sg / n).powi(2)).max(0.0);
            let var_b = (sb2 / n - (sb / n).powi(2)).max(0.0);
            ((var_r + var_g + var_b) / 3.0) as f32
        })
        .collect();

    // C3: use 1e-6 instead of 1.0 to preserve contrast on near-solid images
    let max_var = raw.iter().cloned().fold(0.0f32, f32::max).max(1e-6);

    raw.iter()
        .map(|&v| {
            let norm = (v / max_var).sqrt();
            // sensitivity=0 -> 1.0 partout ; sensitivity=1 -> norm pur
            1.0 - sensitivity * (1.0 - norm)
        })
        .collect()
}

/// Retourne la density map normalisée comme image en niveaux de gris.
/// Utile pour la prévisualisation dans la GUI.
pub fn compute_density_image(src: &RgbImage, sensitivity: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let density = compute_density_map(src, sensitivity);
    GrayImage::from_fn(w, h, |x, y| {
        let v = density[(y * w + x) as usize];
        Luma([(v * 255.0) as u8])
    })
}
