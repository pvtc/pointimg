use crate::filter::params::{Dot, FilterParams};
use crate::filter::render::radius_for_dot;
use crate::filter::render::render;
use crate::filter::sampling::{importance_sample, make_rng_seed};
use crate::filter::util::luminance;
use anyhow::{Result, anyhow};
use image::RgbImage;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

// ─── Algorithme 2 : K-means spatial ──────────────────────────────────────────
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

    for iter in 0..iters {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }

        // Pré-normaliser les centres une seule fois par itération (perf 9)
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

        let (sums, counts) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut sums, mut counts), py| {
                    for px in 0..width {
                        let p = src.get_pixel(px, py);
                        let feat = [
                            px as f32 / w_f,
                            py as f32 / h_f,
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                        ];
                        let best = centers_norm
                            .iter()
                            .enumerate()
                            .map(|(i, cn)| {
                                let d: f32 = feat
                                    .iter()
                                    .zip(cn.iter())
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum();
                                (i, d)
                            })
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        sums[best][0] += px as f64;
                        sums[best][1] += py as f64;
                        sums[best][2] += p[0] as f64;
                        sums[best][3] += p[1] as f64;
                        sums[best][4] += p[2] as f64;
                        counts[best] += 1;
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut as_, mut ac), (bs, bc)| {
                    for i in 0..centers.len() {
                        for j in 0..5 {
                            as_[i][j] += bs[i][j];
                        }
                        ac[i] += bc[i];
                    }
                    (as_, ac)
                },
            );

        for (i, c) in centers.iter_mut().enumerate() {
            let n = counts[i] as f64;
            if n > 0.0 {
                c[0] = (sums[i][0] / n) as f32;
                c[1] = (sums[i][1] / n) as f32;
                c[2] = (sums[i][2] / n) as f32;
                c[3] = (sums[i][3] / n) as f32;
                c[4] = (sums[i][4] / n) as f32;
            }
        }

        // Preview après chaque itération K-means (UX 12)
        let dots = dots_from_kmeans_centers(&centers, density, width, height, img_min, params);
        let preview = render(src, &dots, params);
        on_progress(iter + 1, iters, &preview);

        // P2: keep last iteration's result to avoid recomputing
        if iter + 1 == iters {
            return Ok((preview, dots));
        }
    }

    // Fallback (iters == 0)
    let dots = dots_from_kmeans_centers(&centers, density, width, height, img_min, params);
    let img = render(src, &dots, params);
    Ok((img, dots))
}

/// K-means dot computation without rendering (Q2).
pub(crate) fn compute_dots_kmeans(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &AtomicBool,
) -> Result<Vec<Dot>> {
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
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

    for _iter in 0..iters {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
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

        let (sums, counts) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut sums, mut counts), py| {
                    for px in 0..width {
                        let p = src.get_pixel(px, py);
                        let feat = [
                            px as f32 / w_f,
                            py as f32 / h_f,
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                        ];
                        let best = centers_norm
                            .iter()
                            .enumerate()
                            .map(|(i, cn)| {
                                let d: f32 = feat
                                    .iter()
                                    .zip(cn.iter())
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum();
                                (i, d)
                            })
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        sums[best][0] += px as f64;
                        sums[best][1] += py as f64;
                        sums[best][2] += p[0] as f64;
                        sums[best][3] += p[1] as f64;
                        sums[best][4] += p[2] as f64;
                        counts[best] += 1;
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut as_, mut ac), (bs, bc)| {
                    for i in 0..centers.len() {
                        for j in 0..5 {
                            as_[i][j] += bs[i][j];
                        }
                        ac[i] += bc[i];
                    }
                    (as_, ac)
                },
            );

        for (i, c) in centers.iter_mut().enumerate() {
            let n = counts[i] as f64;
            if n > 0.0 {
                c[0] = (sums[i][0] / n) as f32;
                c[1] = (sums[i][1] / n) as f32;
                c[2] = (sums[i][2] / n) as f32;
                c[3] = (sums[i][3] / n) as f32;
                c[4] = (sums[i][4] / n) as f32;
            }
        }
    }
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
