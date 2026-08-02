use crate::filter::params::Dot;
use crate::filter::params::FilterParams;
use crate::filter::render::radius_for_dot;
use crate::filter::seedgrid::SeedGrid;
use crate::filter::util::luminance;
use image::RgbImage;
use rayon::prelude::*;

pub(crate) fn make_rng_seed(params: &FilterParams) -> u64 {
    params.rng_seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64) // Bug 8 corrigé : nanos complets > subsec_nanos
            .unwrap_or(0xdeadbeef_cafebabe)
    })
}

pub(crate) fn importance_sample(
    weights: &[f32],
    width: u32,
    height: u32,
    k: usize,
    seed: u64,
) -> Vec<(u32, u32)> {
    let total: f64 = weights.iter().map(|&w| w as f64).sum();
    if total <= 0.0 {
        let mut rng: u64 = seed;
        return (0..k)
            .map(|_| {
                // Bug 9 corrigé : utiliser un clamp strict pour éviter coord == width/height
                let x = ((lcg_next(&mut rng) * width as f32) as u32).min(width - 1);
                let y = ((lcg_next(&mut rng) * height as f32) as u32).min(height - 1);
                (x, y)
            })
            .collect();
    }

    let step = total / k as f64;
    let mut rng: u64 = seed;
    let mut result = Vec::with_capacity(k);

    let mut cdf = 0.0f64;
    let mut pixel_idx: usize = 0;
    let n_pixels = (width * height) as usize;

    for i in 0..k {
        let jitter = lcg_next(&mut rng) as f64 - 0.5;
        let target = (i as f64 + 0.5 + jitter * 0.9) * step;
        let target = target.clamp(0.0, total - 1e-9);

        while pixel_idx < n_pixels && cdf + weights[pixel_idx] as f64 <= target {
            cdf += weights[pixel_idx] as f64;
            pixel_idx += 1;
        }
        let idx = pixel_idx.min(n_pixels - 1);
        let px = (idx as u32) % width;
        let py = (idx as u32) / width;
        // C2: sub-pixel jitter to reduce clustering/duplicates when k is large
        let jx = lcg_next(&mut rng);
        let jy = lcg_next(&mut rng);
        let fx = (px as f32 + jx).min(width as f32 - 0.01);
        let fy = (py as f32 + jy).min(height as f32 - 0.01);
        result.push((fx as u32, fy as u32));
    }

    result
}

/// Construit les Dots depuis les graines finales.
///
/// Perf 7 corrigé : parallélisé avec rayon.
pub(crate) fn build_dots_from_seeds(
    src: &RgbImage,
    density: &[f32],
    seeds: &[(f32, f32)],
    params: &FilterParams,
) -> Vec<Dot> {
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
    let k = seeds.len();

    let grid = SeedGrid::new(seeds, width, height);

    // Accumulation parallèle : chaque thread accumule ses propres vecteurs partiels
    let (sum_r, sum_g, sum_b, sum_d, counts) = (0..height)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0f64; k],
                    vec![0u64; k],
                )
            },
            |(mut sr, mut sg, mut sb, mut sd, mut cnt), py| {
                for px in 0..width {
                    let fx = px as f32 + 0.5;
                    let fy = py as f32 + 0.5;
                    let best = grid.nearest(fx, fy, seeds);
                    let p = src.get_pixel(px, py);
                    sr[best] += p[0] as u64;
                    sg[best] += p[1] as u64;
                    sb[best] += p[2] as u64;
                    sd[best] += density[(py * width + px) as usize] as f64;
                    cnt[best] += 1;
                }
                (sr, sg, sb, sd, cnt)
            },
        )
        .reduce(
            || {
                (
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0f64; k],
                    vec![0u64; k],
                )
            },
            |(mut ar, mut ag, mut ab, mut ad, mut ac), (br, bg_vals, bb, bd, bc)| {
                for i in 0..k {
                    ar[i] += br[i];
                    ag[i] += bg_vals[i];
                    ab[i] += bb[i];
                    ad[i] += bd[i];
                    ac[i] += bc[i];
                }
                (ar, ag, ab, ad, ac)
            },
        );

    let nn_radii = nearest_neighbor_radii(seeds);

    seeds
        .iter()
        .enumerate()
        .filter(|(i, _)| counts[*i] > 0)
        .map(|(i, &(x, y))| {
            let n = counts[i];
            let avg = [
                (sum_r[i] / n) as u8,
                (sum_g[i] / n) as u8,
                (sum_b[i] / n) as u8,
            ];
            let lum = luminance(avg[0], avg[1], avg[2]);
            let avg_density = (sum_d[i] / n as f64) as f32;
            let r_uncapped = radius_for_dot(lum, avg_density, img_min, params);
            let r_nn_cap = nn_radii[i] * 0.8;
            Dot {
                x,
                y,
                color: avg,
                radius: r_uncapped.min(r_nn_cap),
            }
        })
        .collect()
}

pub(crate) fn nearest_neighbor_radii(seeds: &[(f32, f32)]) -> Vec<f32> {
    let k = seeds.len();
    if k <= 1 {
        return vec![f32::MAX; k];
    }

    let mut by_x: Vec<usize> = (0..k).collect();
    by_x.sort_unstable_by(|&a, &b| seeds[a].0.total_cmp(&seeds[b].0));

    let mut min_dist_sq = vec![f32::MAX; k];

    for (rank, &i) in by_x.iter().enumerate() {
        let (ax, ay) = seeds[i];
        for &j in by_x.iter().skip(rank + 1) {
            let dx = seeds[j].0 - ax;
            let dx2 = dx * dx;
            if dx2 >= min_dist_sq[i] && dx2 >= min_dist_sq[j] {
                break;
            }
            let dy = seeds[j].1 - ay;
            let d2 = dx2 + dy * dy;
            if d2 < min_dist_sq[i] {
                min_dist_sq[i] = d2;
            }
            if d2 < min_dist_sq[j] {
                min_dist_sq[j] = d2;
            }
        }
    }

    min_dist_sq.iter().map(|&d| d.sqrt() * 0.5).collect()
}

pub(crate) fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Fix #2 : >> 32 donne un u32 complet, divisé par u32::MAX → [0.0, 1.0]
    ((*state >> 32) as f32) / (u32::MAX as f32)
}
