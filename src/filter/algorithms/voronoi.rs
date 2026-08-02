use crate::filter::params::{Dot, FilterParams};
use crate::filter::render::render;
use crate::filter::sampling::{build_dots_from_seeds, importance_sample, make_rng_seed};
use crate::filter::seedgrid::SeedGrid;
use anyhow::{Result, anyhow};
use image::RgbImage;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

// ─── Algorithme 3 : Voronoï / Lloyd ──────────────────────────────────────────
// (la version canonique est dots_voronoi_progressive, utilisée via apply_with_progress)

pub(crate) fn dots_voronoi_progressive<F>(
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
    let k = params.num_points;
    let iters = params.iterations;

    let mut seeds: Vec<(f32, f32)> =
        importance_sample(density, width, height, k, make_rng_seed(params))
            .into_iter()
            .map(|(x, y)| (x as f32 + 0.5, y as f32 + 0.5))
            .collect();

    let power = 1.0 + params.variance_sensitivity * 3.0;
    let lloyd_weights: Vec<f64> = density
        .iter()
        .map(|&d| (d as f64).powf(power as f64))
        .collect();

    // Pré-allouer les accumulateurs (réutilisés à chaque itération)
    for iter in 0..iters {
        // ── Vérifier le cancel AVANT la lourde par_iter ─────────────────────
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }

        let grid = SeedGrid::new(&seeds, width, height);

        let (sum_x, sum_y, sum_w) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut sx, mut sy, mut sw), py| {
                    for px in 0..width {
                        let fx = px as f32 + 0.5;
                        let fy = py as f32 + 0.5;
                        let best = grid.nearest(fx, fy, &seeds);
                        let w = lloyd_weights[(py * width + px) as usize];
                        sx[best] += fx as f64 * w;
                        sy[best] += fy as f64 * w;
                        sw[best] += w;
                    }
                    (sx, sy, sw)
                },
            )
            .reduce(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut ax, mut ay, mut aw), (bx, by, bw)| {
                    for i in 0..k {
                        ax[i] += bx[i];
                        ay[i] += by[i];
                        aw[i] += bw[i];
                    }
                    (ax, ay, aw)
                },
            );

        for i in 0..k {
            if sum_w[i] > 0.0 {
                seeds[i] = ((sum_x[i] / sum_w[i]) as f32, (sum_y[i] / sum_w[i]) as f32);
            }
        }

        // Preview après chaque itération
        let dots = build_dots_from_seeds(src, density, &seeds, params);
        let preview = render(src, &dots, params);
        on_progress(iter + 1, iters, &preview);

        // P2: keep last iteration's result to avoid recomputing
        if iter + 1 == iters {
            return Ok((preview, dots));
        }
    }

    // Fallback (iters == 0)
    let dots = build_dots_from_seeds(src, density, &seeds, params);
    let img = render(src, &dots, params);
    Ok((img, dots))
}

/// Voronoi dot computation without rendering (Q2).
pub(crate) fn compute_dots_voronoi(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &AtomicBool,
) -> Result<Vec<Dot>> {
    let (width, height) = src.dimensions();
    let k = params.num_points;
    let iters = params.iterations;

    let mut seeds: Vec<(f32, f32)> =
        importance_sample(density, width, height, k, make_rng_seed(params))
            .into_iter()
            .map(|(x, y)| (x as f32 + 0.5, y as f32 + 0.5))
            .collect();

    let power = 1.0 + params.variance_sensitivity * 3.0;
    let lloyd_weights: Vec<f64> = density
        .iter()
        .map(|&d| (d as f64).powf(power as f64))
        .collect();

    for _iter in 0..iters {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let grid = SeedGrid::new(&seeds, width, height);
        let (sum_x, sum_y, sum_w) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut sx, mut sy, mut sw), py| {
                    for px in 0..width {
                        let fx = px as f32 + 0.5;
                        let fy = py as f32 + 0.5;
                        let best = grid.nearest(fx, fy, &seeds);
                        let w = lloyd_weights[(py * width + px) as usize];
                        sx[best] += fx as f64 * w;
                        sy[best] += fy as f64 * w;
                        sw[best] += w;
                    }
                    (sx, sy, sw)
                },
            )
            .reduce(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut ax, mut ay, mut aw), (bx, by, bw)| {
                    for i in 0..k {
                        ax[i] += bx[i];
                        ay[i] += by[i];
                        aw[i] += bw[i];
                    }
                    (ax, ay, aw)
                },
            );
        for i in 0..k {
            if sum_w[i] > 0.0 {
                seeds[i] = ((sum_x[i] / sum_w[i]) as f32, (sum_y[i] / sum_w[i]) as f32);
            }
        }
    }
    Ok(build_dots_from_seeds(src, density, &seeds, params))
}
