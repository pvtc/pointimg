//! Logique du filtre pointilliste.
//!
//! Quatre algorithmes de partitionnement :
//!
//! - `Grid`     : grille uniforme
//! - `Kmeans`   : k-means spatial (x, y, r, g, b)
//! - `Voronoi`  : stippling de Lloyd pondéré par la density map
//! - `Quadtree` : subdivision récursive par contraste
//!
//! La `density_map` contrôle la distribution des points :
//!   densité élevée (zone détaillée) → plus de points, petits
//!   densité faible  (zone uniforme)  → moins de points, gros
//!
//! `variance_sensitivity` règle à quel point cette redistribution est forte.

pub use anyhow::{Result, anyhow};

mod algorithms;
mod density;
mod dither;
mod gamma;
mod halftone;
mod params;
mod render;
mod sampling;
mod seedgrid;
mod svg;
mod util;

pub use density::compute_density_image;
pub use halftone::{HalftoneMode, Screening};
pub use params::{Algorithm, Dot, DotShape, FilterParams};
pub use render::{render_halftone, render_rgba};
pub use svg::{render_svg, render_svg_dynamic, render_svg_from_dots};
pub use util::flatten_to_rgb;

use algorithms::{
    compute_dots_kmeans, compute_dots_voronoi, dots_grid, dots_kmeans_progressive, dots_quadtree,
    dots_voronoi_progressive,
};
use density::compute_density_map;
use halftone::dots_halftone;
use params::validate_params;
use render::render;

use image::{DynamicImage, RgbImage};

// ─── Entrée principale ────────────────────────────────────────────────────────

// Helpers gamma pour les wrappers ci-dessous.
fn linearized_clone(params: &FilterParams) -> FilterParams {
    let mut p = params.clone();
    p.bg_color = gamma::srgb_to_linear_rgb(params.bg_color);
    p.gamma_correct = false; // évite le double-wrap quand un *_inner rappelle un wrapper.
    p
}

fn dots_to_srgb(dots: &[Dot]) -> Vec<Dot> {
    dots.iter()
        .map(|d| Dot {
            color: gamma::linear_to_srgb_rgb(d.color),
            ..*d
        })
        .collect()
}

// ── Versions « inner » : le pipeline historique, sans correction gamma.

fn apply_dynamic_inner(src: &DynamicImage, params: &FilterParams) -> Result<RgbImage> {
    let rgb = flatten_to_rgb(src, params.bg_color);
    apply_inner(&rgb, params)
}

fn apply_with_progress_inner<F>(
    src: &RgbImage,
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    mut on_progress: F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, &RgbImage),
{
    use std::sync::atomic::Ordering;
    let (w, h) = src.dimensions();
    validate_params(w, h, params)?;

    let density = compute_density_map(src, params.variance_sensitivity);

    match params.algorithm {
        Algorithm::Voronoi => {
            dots_voronoi_progressive(src, &density, params, cancel, &mut on_progress)
        }
        Algorithm::Kmeans => {
            dots_kmeans_progressive(src, &density, params, cancel, &mut on_progress)
        }
        _ => {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!("cancelled"));
            }
            let dots = match params.algorithm {
                Algorithm::Grid => dots_grid(src, &density, params),
                Algorithm::Quadtree => dots_quadtree(src, params),
                _ => unreachable!(),
            };
            let result = render(src, &dots, params);
            on_progress(1, 1, &result);
            Ok((result, dots))
        }
    }
}

fn apply_inner(src: &RgbImage, params: &FilterParams) -> Result<RgbImage> {
    match params.algorithm {
        Algorithm::Halftone if params.halftone != HalftoneMode::Off => {
            let (rgba, _) = apply_rgba_inner(src, params)?;
            Ok(image::DynamicImage::ImageRgba8(rgba).to_rgb8())
        }
        Algorithm::Grid => {
            let density = compute_density_map(src, params.variance_sensitivity);
            let dots = dots_grid(src, &density, params);
            Ok(render(src, &dots, params))
        }
        Algorithm::Quadtree => {
            let dots = dots_quadtree(src, params);
            Ok(render(src, &dots, params))
        }
        Algorithm::Voronoi | Algorithm::Kmeans => {
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            let (img, _) = apply_with_progress_inner(src, params, &never_cancel, |_, _, _| {})?;
            Ok(img)
        }
        Algorithm::Halftone if params.halftone == HalftoneMode::Off => {
            // Halftone sélectionné mais mode Off → image vide (cas dégénéré).
            Ok(RgbImage::new(src.width(), src.height()))
        }
        Algorithm::Halftone => {
            // Déjà géré par la guard Halftone+!Off ci-dessus.
            unreachable!()
        }
    }
}

fn compute_dots_inner(src: &RgbImage, params: &FilterParams) -> Result<Vec<Dot>> {
    let (w, h) = src.dimensions();
    validate_params(w, h, params)?;
    if params.algorithm == Algorithm::Halftone && params.halftone != HalftoneMode::Off {
        let cfg = params.halftone_config();
        return Ok(dots_halftone(src, &cfg));
    }
    let density = compute_density_map(src, params.variance_sensitivity);

    let dots = match params.algorithm {
        Algorithm::Grid => dots_grid(src, &density, params),
        Algorithm::Quadtree => dots_quadtree(src, params),
        Algorithm::Voronoi => {
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            compute_dots_voronoi(src, &density, params, &never_cancel)?
        }
        Algorithm::Kmeans => {
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            compute_dots_kmeans(src, &density, params, &never_cancel)?
        }
        Algorithm::Halftone => {
            // Halftone est sélectionné mais mode = Off → pas de dots.
            // (Cas dégénéré : on retombe sur un rendu vide.)
            Vec::new()
        }
    };
    Ok(dots)
}

fn apply_rgba_inner(src: &RgbImage, params: &FilterParams) -> Result<(image::RgbaImage, Vec<Dot>)> {
    // Halftone : pipeline dédié (composite multiply, ink-over-ink).
    if params.algorithm == Algorithm::Halftone && params.halftone != HalftoneMode::Off {
        let (w, h) = src.dimensions();
        validate_params(w, h, params)?;
        let cfg = params.halftone_config();
        let dots = dots_halftone(src, &cfg);
        let img = render_halftone(src, &dots, params);
        return Ok((img, dots));
    }
    let dots = compute_dots_inner(src, params)?;
    let img = render_rgba(src, &dots, params);
    Ok((img, dots))
}

// ── API publique : wrappers appliquant la correction gamma si `params.gamma_correct`.

/// Applique le filtre sur une `DynamicImage` (supporte RGBA, niveaux de gris, etc.)
/// Le fond transparent est composé sur `params.bg_color`. Quand `gamma_correct`
/// est vrai, le pipeline s'opère en espace linéaire puis ré-encode le résultat.
pub fn apply_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<RgbImage> {
    if params.gamma_correct {
        let rgb = flatten_to_rgb(src, params.bg_color);
        let lin = gamma::srgb_to_linear_image(&rgb);
        let params_lin = linearized_clone(params);
        let (dst_lin, _) = apply_with_progress_inner(
            &lin,
            &params_lin,
            &std::sync::atomic::AtomicBool::new(false),
            |_, _, _| {},
        )?;
        Ok(gamma::linear_to_srgb_image(&dst_lin))
    } else {
        apply_dynamic_inner(src, params)
    }
}

/// Applique le filtre itération par itération (Voronoï/K-means).
/// Appelle `on_progress(iter, total, image_intermediaire)` après chaque itération Lloyd.
/// Pratique pour la preview progressive dans la GUI.
/// Pour les autres algorithmes, appelle `on_progress` une seule fois à la fin.
///
/// Retourne `(image_finale, dots)` — les dots correspondent exactement à l'image
/// rendue, ce qui évite le double-calcul qu'on avait avant.
///
/// Le token `cancel` est vérifié entre chaque itération. Si `cancel` est vrai,
/// la fonction retourne `Err(anyhow!("cancelled"))`.
///
/// Quand `params.gamma_correct` est vrai, les previews intermédiaires publiées au
/// callback sont déjà ré-encodées en sRGB (donc affichables telles quelles).
pub fn apply_with_progress<F>(
    src: &RgbImage,
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    mut on_progress: F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, &RgbImage),
{
    use std::sync::atomic::Ordering;
    // Halftone : pas d'itérations Lloyd → un seul pass, pas de preview progressive.
    if params.algorithm == Algorithm::Halftone && params.halftone != HalftoneMode::Off {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let (rgba, dots) = apply_rgba_inner(src, params)?;
        let rgb = image::DynamicImage::ImageRgba8(rgba).to_rgb8();
        on_progress(1, 1, &rgb);
        return Ok((rgb, dots));
    }
    if params.gamma_correct {
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        // Wrapper de callback : réencode chaque preview linéaire vers sRGB pour la GUI.
        let mut wrapped = |iter, total, preview_lin: &RgbImage| {
            let preview_srgb = gamma::linear_to_srgb_image(preview_lin);
            on_progress(iter, total, &preview_srgb);
        };
        let (dst_lin, dots_lin) =
            apply_with_progress_inner(&lin, &params_lin, cancel, &mut wrapped)?;
        Ok((
            gamma::linear_to_srgb_image(&dst_lin),
            dots_to_srgb(&dots_lin),
        ))
    } else {
        apply_with_progress_inner(src, params, cancel, on_progress)
    }
}

pub fn apply(src: &RgbImage, params: &FilterParams) -> Result<RgbImage> {
    if params.gamma_correct {
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        let (dst_lin, _) = apply_with_progress_inner(
            &lin,
            &params_lin,
            &std::sync::atomic::AtomicBool::new(false),
            |_, _, _| {},
        )?;
        Ok(gamma::linear_to_srgb_image(&dst_lin))
    } else {
        apply_inner(src, params)
    }
}

/// Variante RGBA de `apply` : rend sur un canvas RGBA (pratique quand
/// `params.transparent` est vrai pour produire un PNG avec alpha).
///
/// Résultat : `(image RGBA, dots)`. Les dots correspondent à l'image rendue.
/// Contrairement à `apply_with_progress`, ce chemin ne publie pas de preview
/// intermédiaire — il calcule les dots puis rend l'image finale.
/// Quand `params.gamma_correct` est vrai, le pipeline s'opère en espace linéaire
/// puis ré-encode les canaux RGB du buffer RGBA (alpha inchangé) en sRGB.
pub fn apply_rgba(src: &RgbImage, params: &FilterParams) -> Result<(image::RgbaImage, Vec<Dot>)> {
    if params.gamma_correct {
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        let (dst_lin, dots_lin) = apply_rgba_inner(&lin, &params_lin)?;
        Ok((
            gamma::linear_to_srgba_image_rgb(&dst_lin),
            dots_to_srgb(&dots_lin),
        ))
    } else {
        apply_rgba_inner(src, params)
    }
}

/// Calcule les dots (sans rendu PNG) — permet de mettre en cache pour la GUI.
/// Q2: avoids rendering a full PNG image that would be discarded.
pub fn compute_dots(src: &RgbImage, params: &FilterParams) -> Result<Vec<Dot>> {
    if params.gamma_correct {
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        let dots_lin = compute_dots_inner(&lin, &params_lin)?;
        Ok(dots_to_srgb(&dots_lin))
    } else {
        compute_dots_inner(src, params)
    }
}

// Imports uniquement nécessaires aux tests.
#[cfg(test)]
use crate::filter::render::point_in_regular_polygon;
#[cfg(test)]
use crate::filter::render::radius_for_dot;
#[cfg(test)]
use crate::filter::sampling::lcg_next;
#[cfg(test)]
use crate::filter::sampling::nearest_neighbor_radii;
#[cfg(test)]
use crate::filter::seedgrid::SeedGrid;
#[cfg(test)]
use crate::filter::util::luminance;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use std::sync::atomic::AtomicBool;

    fn solid_image(w: u32, h: u32, r: u8, g: u8, b: u8) -> RgbImage {
        RgbImage::from_fn(w, h, |_, _| image::Rgb([r, g, b]))
    }

    fn checkerboard(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgb([255, 255, 255])
            } else {
                image::Rgb([0, 0, 0])
            }
        })
    }

    // ── luminance ────────────────────────────────────────────────────────────

    #[test]
    fn luminance_black_is_zero() {
        assert_eq!(luminance(0, 0, 0), 0.0);
    }

    #[test]
    fn luminance_white_is_one() {
        let l = luminance(255, 255, 255);
        assert!((l - 1.0).abs() < 0.001, "luminance blanc ≈ 1, got {l}");
    }

    #[test]
    fn luminance_pure_green_dominant() {
        let l_green = luminance(0, 255, 0);
        let l_red = luminance(255, 0, 0);
        let l_blue = luminance(0, 0, 255);
        assert!(l_green > l_red, "vert > rouge");
        assert!(l_green > l_blue, "vert > bleu");
    }

    // ── radius_for_dot ───────────────────────────────────────────────────────

    fn default_params() -> FilterParams {
        FilterParams::default()
    }

    #[test]
    fn radius_respects_minimum() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r = radius_for_dot(1.0, 1.0, img_min, &p);
        assert!(
            r >= p.min_radius_ratio * img_min * 0.9,
            "rayon trop petit : {r}"
        );
    }

    #[test]
    fn radius_dark_bigger_than_light() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r_dark = radius_for_dot(0.0, 0.5, img_min, &p);
        let r_light = radius_for_dot(1.0, 0.5, img_min, &p);
        assert!(
            r_dark > r_light,
            "sombre doit être plus grand : {r_dark} vs {r_light}"
        );
    }

    #[test]
    fn radius_uniform_zone_bigger_than_detailed() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r_uniform = radius_for_dot(0.5, 0.2, img_min, &p);
        let r_detailed = radius_for_dot(0.5, 1.0, img_min, &p);
        assert!(
            r_uniform > r_detailed,
            "uniforme doit être plus grand : {r_uniform} vs {r_detailed}"
        );
    }

    #[test]
    fn radius_never_exceeds_hard_cap() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r_max_abs = p.max_boost * p.max_radius_ratio * img_min;
        for lum in [0.0f32, 0.5, 1.0] {
            for density in [0.0f32, 0.5, 1.0] {
                let r = radius_for_dot(lum, density, img_min, &p);
                assert!(
                    r <= r_max_abs + 0.01,
                    "rayon dépasse le cap : {r} > {r_max_abs}"
                );
            }
        }
    }

    // ── compute_density_map ──────────────────────────────────────────────────

    #[test]
    fn density_map_uniform_image_is_low_variance() {
        let img = solid_image(32, 32, 128, 128, 128);
        let sensitivity = 0.7;
        let density = compute_density_map(&img, sensitivity);
        let first = density[0];
        for &v in &density {
            assert!(
                (v - first).abs() < 1e-4,
                "densité non uniforme sur image unie"
            );
        }
        assert!(
            (first - 0.3).abs() < 0.05,
            "density sur image unie ≈ 0.3, got {first}"
        );
    }

    #[test]
    fn density_map_checkerboard_is_higher_than_solid() {
        let solid = solid_image(32, 32, 128, 128, 128);
        let checker = checkerboard(32, 32);
        let sensitivity = 1.0;
        let d_solid: f32 =
            compute_density_map(&solid, sensitivity).iter().sum::<f32>() / (32 * 32) as f32;
        let d_checker: f32 = compute_density_map(&checker, sensitivity)
            .iter()
            .sum::<f32>()
            / (32 * 32) as f32;
        assert!(
            d_checker > d_solid,
            "damier doit avoir density > image unie : {d_checker} vs {d_solid}"
        );
    }

    #[test]
    fn density_map_sensitivity_zero_is_all_ones() {
        let img = checkerboard(16, 16);
        let density = compute_density_map(&img, 0.0);
        for &v in &density {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "sensitivity=0 → density=1 partout, got {v}"
            );
        }
    }

    // ── nearest_neighbor_radii ───────────────────────────────────────────────

    #[test]
    fn nn_radii_single_seed_is_max() {
        let seeds = vec![(10.0f32, 20.0f32)];
        let radii = nearest_neighbor_radii(&seeds);
        assert_eq!(radii.len(), 1);
        assert_eq!(radii[0], f32::MAX);
    }

    #[test]
    fn nn_radii_empty_is_empty() {
        let seeds: Vec<(f32, f32)> = vec![];
        let radii = nearest_neighbor_radii(&seeds);
        assert!(radii.is_empty());
    }

    #[test]
    fn nn_radii_two_seeds_correct_distance() {
        let seeds = vec![(0.0f32, 0.0f32), (10.0f32, 0.0f32)];
        let radii = nearest_neighbor_radii(&seeds);
        assert_eq!(radii.len(), 2);
        assert!(
            (radii[0] - 5.0).abs() < 0.01,
            "rayon attendu 5.0, got {}",
            radii[0]
        );
        assert!(
            (radii[1] - 5.0).abs() < 0.01,
            "rayon attendu 5.0, got {}",
            radii[1]
        );
    }

    #[test]
    fn nn_radii_three_seeds_picks_closest() {
        let seeds = vec![(0.0f32, 0.0f32), (4.0f32, 0.0f32), (100.0f32, 0.0f32)];
        let radii = nearest_neighbor_radii(&seeds);
        assert!(
            (radii[0] - 2.0).abs() < 0.01,
            "A: demi-dist 2, got {}",
            radii[0]
        );
        assert!(
            (radii[1] - 2.0).abs() < 0.01,
            "B: demi-dist 2, got {}",
            radii[1]
        );
        assert!(
            (radii[2] - 48.0).abs() < 0.01,
            "C: demi-dist 48, got {}",
            radii[2]
        );
    }

    // ── SeedGrid ─────────────────────────────────────────────────────────────

    #[test]
    fn seedgrid_nearest_brute_force_match() {
        let mut rng: u64 = 0xdeadbeef;
        let seeds: Vec<(f32, f32)> = (0..50)
            .map(|_| (lcg_next(&mut rng) * 200.0, lcg_next(&mut rng) * 200.0))
            .collect();

        let grid = SeedGrid::new(&seeds, 200, 200);

        for qx in [5.0f32, 50.0, 100.0, 150.0, 195.0] {
            for qy in [5.0f32, 50.0, 100.0, 150.0, 195.0] {
                let grid_result = grid.nearest(qx, qy, &seeds);
                let bf_result = seeds
                    .iter()
                    .enumerate()
                    .map(|(i, &(sx, sy))| (i, (qx - sx).powi(2) + (qy - sy).powi(2)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i)
                    .unwrap();
                assert_eq!(
                    grid_result, bf_result,
                    "SeedGrid vs brute-force mismatch à ({qx},{qy}): grid={grid_result} bf={bf_result}"
                );
            }
        }
    }

    #[test]
    fn seedgrid_single_seed_always_returns_zero() {
        let seeds = vec![(50.0f32, 50.0f32)];
        let grid = SeedGrid::new(&seeds, 100, 100);
        for x in [0.0f32, 25.0, 50.0, 99.0] {
            for y in [0.0f32, 25.0, 50.0, 99.0] {
                assert_eq!(grid.nearest(x, y, &seeds), 0);
            }
        }
    }

    // ── apply (intégration) ──────────────────────────────────────────────────

    #[test]
    fn apply_produces_correct_dimensions() {
        let img = solid_image(64, 48, 200, 100, 50);
        let params = FilterParams {
            num_points: 20,
            iterations: 2,
            ..FilterParams::default()
        };
        let result = apply(&img, &params).unwrap();
        assert_eq!(result.dimensions(), (64, 48));
    }

    #[test]
    fn apply_bg_white_and_black() {
        let img = solid_image(32, 32, 0, 0, 0);
        let params_white = FilterParams {
            bg_color: [255, 255, 255],
            num_points: 10,
            iterations: 1,
            ..FilterParams::default()
        };
        let params_black = FilterParams {
            bg_color: [0, 0, 0],
            num_points: 10,
            iterations: 1,
            ..FilterParams::default()
        };
        let res_white = apply(&img, &params_white).unwrap();
        let res_black = apply(&img, &params_black).unwrap();
        let corner_w = res_white.get_pixel(0, 0);
        let corner_b = res_black.get_pixel(0, 0);
        let lum_w = luminance(corner_w[0], corner_w[1], corner_w[2]);
        let lum_b = luminance(corner_b[0], corner_b[1], corner_b[2]);
        assert!(lum_w > lum_b, "fond blanc plus clair que fond noir");
    }

    #[test]
    fn apply_all_algorithms_run() {
        let img = checkerboard(32, 32);
        for algo in [
            Algorithm::Grid,
            Algorithm::Kmeans,
            Algorithm::Voronoi,
            Algorithm::Quadtree,
        ] {
            let params = FilterParams {
                algorithm: algo,
                num_points: 20,
                cols: 8,
                iterations: 2,
                ..FilterParams::default()
            };
            let result = apply(&img, &params).unwrap();
            assert_eq!(
                result.dimensions(),
                (32, 32),
                "algo {algo:?} dimensions incorrectes"
            );
        }
    }

    #[test]
    fn apply_invalid_params_returns_error() {
        let img = solid_image(32, 32, 128, 128, 128);
        let params = FilterParams {
            min_radius_ratio: 0.5,
            max_radius_ratio: 0.1,
            ..FilterParams::default()
        };
        assert!(
            apply(&img, &params).is_err(),
            "min > max doit retourner une erreur"
        );

        let params2 = FilterParams {
            num_points: 0,
            ..FilterParams::default()
        };
        assert!(
            apply(&img, &params2).is_err(),
            "num_points=0 doit retourner une erreur"
        );
    }

    // ── validate_params (nouvelles branches) ─────────────────────────────────

    #[test]
    fn validate_cols_zero_rejected() {
        let p = FilterParams {
            cols: 0,
            ..FilterParams::default()
        };
        assert!(validate_params(64, 64, &p).is_err(), "cols=0 doit echouer");
    }

    #[test]
    fn validate_palette_size_one_rejected() {
        let p = FilterParams {
            palette_size: Some(1),
            ..FilterParams::default()
        };
        assert!(
            validate_params(64, 64, &p).is_err(),
            "palette_size=Some(1) doit echouer"
        );
    }

    #[test]
    fn validate_palette_size_two_accepted() {
        let p = FilterParams {
            palette_size: Some(2),
            ..FilterParams::default()
        };
        assert!(
            validate_params(64, 64, &p).is_ok(),
            "palette_size=Some(2) doit passer"
        );
    }

    #[test]
    fn validate_variance_sensitivity_out_of_range_rejected() {
        for bad in [-0.1f32, 1.5] {
            let p = FilterParams {
                variance_sensitivity: bad,
                ..FilterParams::default()
            };
            assert!(
                validate_params(64, 64, &p).is_err(),
                "variance_sensitivity={bad} doit echouer"
            );
        }
    }

    #[test]
    fn validate_variance_sensitivity_bounds_accepted() {
        for ok in [0.0f32, 1.0] {
            let p = FilterParams {
                variance_sensitivity: ok,
                ..FilterParams::default()
            };
            assert!(
                validate_params(64, 64, &p).is_ok(),
                "variance_sensitivity={ok} doit passer"
            );
        }
    }

    #[test]
    fn validate_max_boost_below_one_rejected() {
        let p = FilterParams {
            max_boost: 0.5,
            ..FilterParams::default()
        };
        assert!(
            validate_params(64, 64, &p).is_err(),
            "max_boost=0.5 doit echouer"
        );
    }

    #[test]
    fn validate_iterations_zero_accepted() {
        // 0 est valide : pas de relaxation Lloyd, dots bruts depuis seeds
        let p = FilterParams {
            iterations: 0,
            ..FilterParams::default()
        };
        assert!(
            validate_params(64, 64, &p).is_ok(),
            "iterations=0 doit passer"
        );
    }

    // ── Formes ───────────────────────────────────────────────────────────────

    #[test]
    fn all_dot_shapes_produce_correct_dimensions() {
        let img = checkerboard(32, 32);
        for shape in [
            DotShape::Circle,
            DotShape::Square,
            DotShape::Ellipse {
                aspect: 2.0,
                angle_deg: 30.0,
            },
            DotShape::RegularPolygon { sides: 6 },
        ] {
            let params = FilterParams {
                num_points: 10,
                iterations: 1,
                dot_shape: shape,
                ..FilterParams::default()
            };
            let result = apply(&img, &params).unwrap();
            assert_eq!(result.dimensions(), (32, 32), "shape {shape:?}");
        }
    }

    #[test]
    fn point_in_regular_polygon_center_is_inside() {
        for sides in [3u8, 4, 5, 6, 8] {
            assert!(
                point_in_regular_polygon(0.0, 0.0, 10.0, sides as usize),
                "centre doit être dans polygone à {sides} côtés"
            );
        }
    }

    #[test]
    fn point_in_regular_polygon_far_outside() {
        for sides in [3u8, 4, 6] {
            assert!(
                !point_in_regular_polygon(20.0, 0.0, 10.0, sides as usize),
                "point loin doit être hors polygone à {sides} côtés"
            );
        }
    }

    // ── apply_with_progress + cancel ─────────────────────────────────────────

    #[test]
    fn apply_with_progress_calls_callback() {
        let img = solid_image(32, 32, 100, 150, 200);
        let params = FilterParams {
            algorithm: Algorithm::Voronoi,
            num_points: 10,
            iterations: 3,
            ..FilterParams::default()
        };
        let cancel = AtomicBool::new(false);
        let mut call_count = 0usize;
        let result = apply_with_progress(&img, &params, &cancel, |iter, total, _preview| {
            assert!(iter <= total);
            call_count += 1;
        });
        assert!(result.is_ok());
        assert_eq!(call_count, 3, "callback appelé 3 fois pour 3 itérations");
    }

    #[test]
    fn apply_with_progress_cancel_returns_err() {
        let img = solid_image(64, 64, 100, 150, 200);
        let params = FilterParams {
            algorithm: Algorithm::Voronoi,
            num_points: 50,
            iterations: 10,
            ..FilterParams::default()
        };
        let cancel = AtomicBool::new(true); // déjà annulé
        let result = apply_with_progress(&img, &params, &cancel, |_, _, _| {});
        assert!(result.is_err(), "doit retourner Err quand cancel=true");
        assert!(
            result.unwrap_err().to_string().contains("cancelled"),
            "message d'erreur doit contenir 'cancelled'"
        );
    }

    // ── density image ────────────────────────────────────────────────────────

    #[test]
    fn compute_density_image_correct_size() {
        let img = checkerboard(40, 30);
        let di = compute_density_image(&img, 0.7);
        assert_eq!(di.dimensions(), (40, 30));
    }
}
