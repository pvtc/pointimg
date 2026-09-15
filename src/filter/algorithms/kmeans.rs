use crate::filter::params::{Dot, FilterParams};
use crate::filter::render::radius_for_dot;
use crate::filter::render::render;
use crate::filter::sampling::{importance_sample, make_rng_seed};
use crate::filter::seedgrid::SeedGrid;
use crate::filter::util::luminance;
use anyhow::Result;

use crate::filter::cancelled;
use image::RgbImage;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

// ─── Algorithme 2 : K-means spatial ──────────────────────────────────────────

/// Itère le Lloyd k-means et retourne les centres finaux (x, y, r, g, b).
///
/// Optimisations par rapport à la boucle naïve :
/// - positions/couleurs normalisées précalculées (LUT) hors de la boucle ;
/// - distance 5D déroulée, scalarisée (pas d'itérateur `zip`) ;
/// - recherche du centre le plus proche via `SeedGrid` avec borne inférieure
///   sur la distance spatiale² (exact : départage par plus petit index comme
///   `Iterator::min_by`) ;
/// - accumulation par chunks de lignes fixes combinés dans l'ordre des
///   indices → sommes f64 déterministes (l'ancien `fold/reduce` dépendait du
///   work-stealing et pouvait dériver dans le dernier ulp) ;
/// - arrêt précoce quand les centres atteignent un point fixe bit-à-bit
///   (les itérations restantes seraient des no-ops).
///
/// `on_iter(iter, centers)` est appelé après chaque itération effective.
fn run_kmeans<F>(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &AtomicBool,
    mut on_iter: F,
) -> Result<Vec<[f32; 5]>>
where
    F: FnMut(usize, &[[f32; 5]]),
{
    let (width, height) = src.dimensions();
    let k = params.num_points;
    let iters = params.iterations;

    let seeds_pos = importance_sample(density, width, height, k, make_rng_seed(params));
    let mut centers: Vec<[f32; 5]> = seeds_pos
        .iter()
        .map(|&(x, y)| {
            let p = src.get_pixel(x.min(width - 1), y.min(height - 1));
            [x as f32, y as f32, p[0] as f32, p[1] as f32, p[2] as f32]
        })
        .collect();

    let w_f = width as f32;
    let h_f = height as f32;
    let norm_x = 1.0 / w_f;
    let norm_y = 1.0 / h_f;

    // LUTs : positions et couleurs normalisées (recalculées à chaque itération
    // auparavant, alors qu'elles sont constantes). Divisions identiques à
    // l'implémentation d'origine — parité des valeurs garantie.
    let xn: Vec<f32> = (0..width).map(|px| px as f32 / w_f).collect();
    let yn: Vec<f32> = (0..height).map(|py| py as f32 / h_f).collect();
    let mut norm_lut = [0f32; 256];
    for (i, v) in norm_lut.iter_mut().enumerate() {
        *v = i as f32 / 255.0;
    }
    let raw = src.as_raw();
    let stride = width as usize * 3;

    // Chunks de lignes fixes (un par thread) → réduction déterministe.
    let n_chunks = rayon::current_num_threads().max(1);
    let chunk_rows = (height as usize).div_ceil(n_chunks);
    let n_chunks = (height as usize).div_ceil(chunk_rows.max(1));

    for iter in 0..iters {
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled());
        }

        // Grille sur les positions des centres (rebuilt chaque itération, O(k)).
        let positions: Vec<(f32, f32)> = centers.iter().map(|c| (c[0], c[1])).collect();
        let grid = SeedGrid::new(&positions, width, height);

        let centers_norm: Vec<[f32; 5]> = centers
            .iter()
            .map(|c| {
                [
                    c[0] / w_f,
                    c[1] / h_f,
                    c[2] / 255.0,
                    c[3] / 255.0,
                    c[4] / 255.0,
                ]
            })
            .collect();

        let partials: Vec<(Vec<[f64; 5]>, Vec<u64>)> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk| {
                let y0 = chunk * chunk_rows;
                let y1 = y0.saturating_add(chunk_rows).min(height as usize);
                let mut sums = vec![[0f64; 5]; k];
                let mut counts = vec![0u64; k];
                for (rel_py, &fy) in yn[y0..y1].iter().enumerate() {
                    let py = y0 + rel_py;
                    let row_start = py * stride;
                    let row = &raw[row_start..row_start + stride];
                    for (px, p) in row.as_chunks::<3>().0.iter().enumerate() {
                        let fx = xn[px];
                        let fr = norm_lut[p[0] as usize];
                        let fg = norm_lut[p[1] as usize];
                        let fb = norm_lut[p[2] as usize];
                        let best = grid.nearest_by(px as f32, py as f32, norm_x, norm_y, |i| {
                            let cn = &centers_norm[i];
                            let dx = fx - cn[0];
                            let dy = fy - cn[1];
                            let dr = fr - cn[2];
                            let dg = fg - cn[3];
                            let db = fb - cn[4];
                            dx * dx + dy * dy + dr * dr + dg * dg + db * db
                        });
                        sums[best][0] += px as f64;
                        sums[best][1] += py as f64;
                        sums[best][2] += p[0] as f64;
                        sums[best][3] += p[1] as f64;
                        sums[best][4] += p[2] as f64;
                        counts[best] += 1;
                    }
                }
                (sums, counts)
            })
            .collect();

        // Combinaison des chunks dans l'ordre des indices (déterministe).
        let mut sums = vec![[0f64; 5]; k];
        let mut counts = vec![0u64; k];
        for (ps, pc) in partials {
            for i in 0..k {
                for j in 0..5 {
                    sums[i][j] += ps[i][j];
                }
                counts[i] += pc[i];
            }
        }

        // Mise à jour des centres + détection du point fixe (bit-à-bit).
        let mut converged = true;
        for (i, c) in centers.iter_mut().enumerate() {
            let n = counts[i] as f64;
            if n > 0.0 {
                let new = [
                    (sums[i][0] / n) as f32,
                    (sums[i][1] / n) as f32,
                    (sums[i][2] / n) as f32,
                    (sums[i][3] / n) as f32,
                    (sums[i][4] / n) as f32,
                ];
                for j in 0..5 {
                    if new[j].to_bits() != c[j].to_bits() {
                        converged = false;
                        break;
                    }
                }
                *c = new;
            }
        }

        on_iter(iter, &centers);

        // Point fixe atteint : les itérations restantes seraient identiques.
        if converged {
            break;
        }
    }
    Ok(centers)
}

// (la version canonique est dots_kmeans_progressive, utilisée via apply_with_progress)

pub(crate) fn dots_kmeans_progressive<F>(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &AtomicBool,
    on_progress: &mut F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, &RgbImage),
{
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
    let iters = params.iterations;

    let mut last: Option<(RgbImage, Vec<Dot>)> = None;
    let centers = run_kmeans(src, density, params, cancel, |iter, centers| {
        // Preview + dots après chaque itération effective.
        let dots = dots_from_kmeans_centers(centers, density, width, height, img_min, params);
        let preview = render(src, &dots, params);
        on_progress(iter + 1, iters, &preview);
        last = Some((preview, dots));
    })?;

    if let Some((img, dots)) = last {
        return Ok((img, dots));
    }

    // Fallback (iters == 0)
    let dots = dots_from_kmeans_centers(&centers, density, width, height, img_min, params);
    let img = render(src, &dots, params);
    Ok((img, dots))
}

/// K-means dot computation without rendering.
pub(crate) fn compute_dots_kmeans(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &AtomicBool,
) -> Result<Vec<Dot>> {
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;

    let centers = run_kmeans(src, density, params, cancel, |_, _| {})?;
    Ok(dots_from_kmeans_centers(
        &centers, density, width, height, img_min, params,
    ))
}

/// Construit les Dots depuis les centres K-means.
/// Extrait pour dédupliquer le code entre preview et résultat final.
pub(crate) fn dots_from_kmeans_centers(
    centers: &[[f32; 5]],
    density: &[f32],
    width: u32,
    height: u32,
    img_min: f32,
    params: &FilterParams,
) -> Vec<Dot> {
    centers
        .iter()
        .filter(|c| c[0] >= 0.0 && c[0] < width as f32 && c[1] >= 0.0 && c[1] < height as f32)
        .map(|c| {
            let avg = [c[2] as u8, c[3] as u8, c[4] as u8];
            let lum = luminance(avg[0], avg[1], avg[2]);
            let cx = (c[0] as u32).min(width - 1);
            let cy = (c[1] as u32).min(height - 1);
            let d = density[(cy * width + cx) as usize];
            Dot {
                x: c[0],
                y: c[1],
                color: avg,
                radius: radius_for_dot(lum, d, img_min, params),
            }
        })
        .collect()
}
