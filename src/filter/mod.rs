//! Pointillist filter logic.
//!
//! Four partitioning algorithms:
//!
//! - `Grid`     : uniform grid
//! - `Kmeans`   : spatial k-means (x, y, r, g, b)
//! - `Voronoi`  : Lloyd stippling weighted by the density map
//! - `Quadtree` : recursive subdivision by contrast
//!
//! The `density_map` controls the distribution of the dots:
//!   high density (detailed zone) → more dots, small
//!   low density  (uniform zone)  → fewer dots, big
//!
//! `variance_sensitivity` tunes how strong this redistribution is.

pub use anyhow::{Result, anyhow};

mod algorithms;
mod density;
mod dither;
mod gamma;
#[cfg(feature = "gpu")]
pub mod gpu;
mod halftone;
mod params;
mod presets;
mod render;
mod sampling;
mod seedgrid;
mod svg;
mod util;

pub use density::{
    compute_density_image, compute_density_map, compute_density_map_cpu, density_to_image,
};
pub use dither::Dithering;
pub use halftone::{HalftoneMode, Screening};

/// Error returned when the computation was interrupted via the `cancel` token.
///
/// Detect it with [`is_cancelled`] rather than by comparing the message:
///
/// ```
/// use std::sync::atomic::AtomicBool;
/// use image::RgbImage;
/// use pointimg::filter::{self, FilterParams, is_cancelled};
///
/// let cancel = AtomicBool::new(true);
/// let img = RgbImage::from_pixel(8, 8, image::Rgb([0, 0, 0]));
/// let err = filter::apply_with_progress(&img, &FilterParams::default(), &cancel, |_, _, _| {})
///     .unwrap_err();
/// assert!(is_cancelled(&err));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Returns `true` when the error comes from cancelling the computation
/// (see [`Cancelled`]).
pub fn is_cancelled(err: &anyhow::Error) -> bool {
    err.downcast_ref::<Cancelled>().is_some()
}

fn cancelled() -> anyhow::Error {
    anyhow::Error::new(Cancelled)
}
pub use params::{
    Algorithm, Dot, DotShape, FilterParams, MAX_IMAGE_DIMENSION, MAX_IMAGE_PIXELS,
    validate_image_dimensions, validate_params,
};
pub use presets::{BuiltinPreset, builtin_presets};
pub use render::{render_halftone, render_rgba};
pub use svg::{render_svg, render_svg_dynamic, render_svg_from_dots};
pub use util::{
    estimate_memory_bytes, estimate_memory_bytes_for, flatten_to_rgb, resize_to_limits,
};

use algorithms::{
    compute_dots_kmeans, compute_dots_voronoi, dots_grid, dots_kmeans_progressive, dots_quadtree,
    dots_voronoi_progressive,
};
use halftone::{dots_halftone, dots_halftone_linear};
use render::render;

use image::{DynamicImage, RgbImage};

// ─── Main entry point ─────────────────────────────────────────────────────────

// Gamma helpers for the wrappers below.
fn linearized_clone(params: &FilterParams) -> FilterParams {
    let mut p = params.clone();
    p.bg_color = gamma::srgb_to_linear_rgb(params.bg_color);
    p.gamma_correct = false; // avoids double-wrapping when an *_inner calls back a wrapper.
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

// ── "inner" versions: the historical pipeline, without gamma correction.

/// Should a preview be rendered at every Lloyd iteration?
///
/// [`Previews::Each`] feeds the progressive display of an interface;
/// [`Previews::None`] only computes the dots, and lets the final render happen
/// only once. The difference is not cosmetic: a render on 8.4 Mpx costs a
/// full-pixel pass plus ~25 MB of image — in [`Previews::Each`] mode, one per
/// iteration.
///
/// *Progress* is still announced in both cases: only the image disappears.
/// The internal contract is therefore `Option<&RgbImage>`, with `None` meaning
/// "no preview", and the public entry point that promises an image rebuilds it
/// from that `Option` (with [`Previews::Each`] it is always present).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Previews {
    Each,
    None,
}

fn apply_with_progress_inner<F>(
    src: &RgbImage,
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    cached_density: Option<&[f32]>,
    previews: Previews,
    mut on_progress: F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, Option<&RgbImage>),
{
    use std::sync::atomic::Ordering;
    let (w, h) = src.dimensions();
    validate_params(w, h, params)?;

    // The quadtree segments on contrast through its own integral tables and
    // never uses the density map. Computing it would cost an O(pixels) pass
    // and two full-resolution `f64` tables for nothing.
    let owned_density;
    let density: &[f32] = match cached_density {
        Some(cached) => cached,
        None if params.algorithm == Algorithm::Quadtree => &[],
        None => {
            owned_density = compute_density_map(src, params.variance_sensitivity);
            &owned_density
        }
    };

    match params.algorithm {
        Algorithm::Voronoi => {
            dots_voronoi_progressive(src, density, params, cancel, previews, &mut on_progress)
        }
        Algorithm::Kmeans => {
            dots_kmeans_progressive(src, density, params, cancel, previews, &mut on_progress)
        }
        _ => {
            if cancel.load(Ordering::Relaxed) {
                return Err(cancelled());
            }
            let dots = match params.algorithm {
                Algorithm::Grid => dots_grid(src, density, params),
                Algorithm::Quadtree => dots_quadtree(src, params),
                _ => unreachable!(),
            };
            let result = render(src, &dots, params);
            on_progress(1, 1, Some(&result));
            Ok((result, dots))
        }
    }
}

fn apply_inner(src: &RgbImage, params: &FilterParams) -> Result<RgbImage> {
    let (w, h) = src.dimensions();
    validate_params(w, h, params)?;
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
            // Direct path: the intermediate previews would be computed then
            // thrown away at every iteration, so render only once.
            let dots = compute_dots_inner(src, params)?;
            Ok(render(src, &dots, params))
        }
        Algorithm::Halftone if params.halftone == HalftoneMode::Off => {
            // Halftone selected but mode Off → empty image (degenerate case).
            Ok(RgbImage::new(src.width(), src.height()))
        }
        Algorithm::Halftone => {
            // Already handled by the Halftone+!Off guard above.
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
    // The quadtree does not use the density map (see `apply_with_progress_inner`).
    let owned_density;
    let density: &[f32] = if params.algorithm == Algorithm::Quadtree {
        &[]
    } else {
        owned_density = compute_density_map(src, params.variance_sensitivity);
        &owned_density
    };

    let dots = match params.algorithm {
        Algorithm::Grid => dots_grid(src, density, params),
        Algorithm::Quadtree => dots_quadtree(src, params),
        Algorithm::Voronoi => {
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            compute_dots_voronoi(src, density, params, &never_cancel)?
        }
        Algorithm::Kmeans => {
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            compute_dots_kmeans(src, density, params, &never_cancel)?
        }
        Algorithm::Halftone => {
            // Halftone is selected but mode = Off → no dots.
            // (Degenerate case: falls back to an empty render.)
            Vec::new()
        }
    };
    Ok(dots)
}

fn apply_rgba_inner(src: &RgbImage, params: &FilterParams) -> Result<(image::RgbaImage, Vec<Dot>)> {
    // Halftone: dedicated pipeline (multiply composite, ink-over-ink).
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

fn is_cmyk_halftone(params: &FilterParams) -> bool {
    params.algorithm == Algorithm::Halftone && matches!(params.halftone, HalftoneMode::Cmyk { .. })
}

// ── Public API: wrappers applying the gamma correction if `params.gamma_correct`.

/// Applies the filter to a `DynamicImage` (supports RGBA, grayscale, etc.).
/// The transparent background is composited over `params.bg_color`. When
/// `gamma_correct` is true, the pipeline works in linear space then re-encodes
/// the result.
///
/// Delegates to [`apply`] after flattening: it therefore handles gamma, the
/// CMYK halftone (`f32` path) and all the algorithms uniformly, through the
/// same path as the rest of the pipeline.
pub fn apply_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<RgbImage> {
    let rgb = flatten_to_rgb(src, params.bg_color);
    apply(&rgb, params)
}

/// Applies the filter iteration by iteration (Voronoi/K-means).
/// Calls `on_progress(iter, total, intermediate_image)` after each Lloyd
/// iteration. Handy for the progressive preview in the GUI.
/// For the other algorithms, calls `on_progress` once at the end.
///
/// Returns `(final_image, dots)` — the dots match the rendered image exactly,
/// which avoids a second render.
///
/// The `cancel` token is checked between every iteration. When `cancel` is
/// true, the function returns a [`Cancelled`] error, detectable with
/// [`is_cancelled`].
///
/// When `params.gamma_correct` is true, the intermediate previews published to
/// the callback are already re-encoded to sRGB (so displayable as they are).
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
    // Halftone: no Lloyd iterations → a single pass, no progressive preview.
    if params.algorithm == Algorithm::Halftone && params.halftone != HalftoneMode::Off {
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled());
        }
        if params.gamma_correct {
            let (rgba, dots) = apply_rgba(src, params)?;
            let rgb = image::DynamicImage::ImageRgba8(rgba).to_rgb8();
            on_progress(1, 1, &rgb);
            return Ok((rgb, dots));
        }
        let (rgba, dots) = apply_rgba_inner(src, params)?;
        let rgb = image::DynamicImage::ImageRgba8(rgba).to_rgb8();
        on_progress(1, 1, &rgb);
        return Ok((rgb, dots));
    }
    if params.gamma_correct {
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        // Callback wrapper: re-encodes every linear preview into sRGB for the
        // GUI. The image is guaranteed by `Previews::Each`, passed one line
        // below.
        let mut wrapped = |iter, total, preview_lin: Option<&RgbImage>| {
            let preview_lin = preview_lin.expect("Previews::Each produit une image");
            let preview_srgb = gamma::linear_to_srgb_image(preview_lin);
            on_progress(iter, total, &preview_srgb);
        };
        let (dst_lin, dots_lin) = apply_with_progress_inner(
            &lin,
            &params_lin,
            cancel,
            None,
            Previews::Each,
            &mut wrapped,
        )?;
        Ok((
            gamma::linear_to_srgb_image(&dst_lin),
            dots_to_srgb(&dots_lin),
        ))
    } else {
        // `Previews::Each`: the image is always there, the adapter cannot lose
        // the preview on the way.
        apply_with_progress_inner(
            src,
            params,
            cancel,
            None,
            Previews::Each,
            &mut |iter, total, preview: Option<&RgbImage>| {
                on_progress(
                    iter,
                    total,
                    preview.expect("Previews::Each produit une image"),
                )
            },
        )
    }
}

/// Variant of [`apply_with_progress`] that renders no intermediate preview.
///
/// The callback only receives `(iteration, total)`. This is the CLI path: its
/// callback announced progress on stderr while throwing the image away, when
/// every Lloyd iteration cost it a `build_dots_from_seeds` (a full-pixel pass)
/// plus a `render` (25 MB of image at 8.4 Mpx) — the dominant cost of a
/// Voronoi run. The dots are computed once then rendered only once, like the
/// direct `apply` path.
///
/// An interface that *displays* the previews keeps [`apply_with_progress`].
pub fn apply_with_iteration_counts<F>(
    src: &RgbImage,
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    mut on_iter: F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize),
{
    // An active halftone has no Lloyd loop: a single pass, as in
    // `apply_with_progress`. The guard sits outside the gamma branch, otherwise
    // a halftone without gamma would reach the pointillist path's
    // `unreachable!()`.
    if params.algorithm == Algorithm::Halftone && params.halftone != HalftoneMode::Off {
        let (rgba, dots) = apply_rgba(src, params)?;
        let rgb = image::DynamicImage::ImageRgba8(rgba).to_rgb8();
        on_iter(1, 1);
        return Ok((rgb, dots));
    }
    if params.gamma_correct {
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        let total = params.iterations.max(1);
        // The iteration callback is the only one left: the image is no longer
        // produced, so there is nothing to re-encode for it.
        let mut wrapped = |iter, _total, _preview: Option<&RgbImage>| {
            on_iter(iter, total);
        };
        let (dst_lin, dots_lin) = apply_with_progress_inner(
            &lin,
            &params_lin,
            cancel,
            None,
            Previews::None,
            &mut wrapped,
        )?;
        return Ok((
            gamma::linear_to_srgb_image(&dst_lin),
            dots_to_srgb(&dots_lin),
        ));
    }
    let total = params.iterations.max(1);
    apply_with_progress_inner(
        src,
        params,
        cancel,
        None,
        Previews::None,
        &mut |iter, _total, _preview: Option<&RgbImage>| on_iter(iter, total),
    )
}

/// Variant of `apply_with_progress` using an already computed density map.
pub fn apply_with_progress_cached<F>(
    src: &RgbImage,
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    density: &[f32],
    mut on_progress: F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, &RgbImage),
{
    let expected = (src.width() as usize)
        .checked_mul(src.height() as usize)
        .ok_or_else(|| anyhow!("dimensions d'image trop grandes"))?;
    if density.len() != expected {
        return Err(anyhow!(
            "density map invalide : {} valeurs, {} attendues",
            density.len(),
            expected
        ));
    }
    if params.gamma_correct
        || (params.algorithm == Algorithm::Halftone && params.halftone != HalftoneMode::Off)
    {
        // The cached path handles neither gamma nor halftone (dedicated
        // pipeline): delegate to `apply_with_progress` rather than reaching
        // `unreachable!()` in `apply_with_progress_inner`.
        return apply_with_progress(src, params, cancel, on_progress);
    }
    apply_with_progress_inner(
        src,
        params,
        cancel,
        Some(density),
        Previews::Each,
        // `on_progress` receives an image; the `Each` path always guarantees one.
        &mut |iter, total, preview: Option<&RgbImage>| {
            (on_progress)(
                iter,
                total,
                preview.expect("Previews::Each produit une image"),
            )
        },
    )
}

/// Applies the filter to an `RgbImage` and renders the result.
///
/// This is the direct path, without intermediate preview: the dots are
/// computed once then rendered only once. When `gamma_correct` is true, the
/// pipeline works in linear space then re-encodes the result; an active
/// halftone goes through the `apply_rgba` compositor (hence `render_halftone`).
pub fn apply(src: &RgbImage, params: &FilterParams) -> Result<RgbImage> {
    if params.gamma_correct {
        // Every active halftone (CMYK or dominant) goes through `apply_rgba`,
        // which handles the gamma path. Without this general guard, a dominant
        // halftone under gamma would reach `unreachable!()` in
        // `apply_with_progress_inner`.
        if params.algorithm == Algorithm::Halftone && params.halftone != HalftoneMode::Off {
            let (rgba, _) = apply_rgba(src, params)?;
            return Ok(image::DynamicImage::ImageRgba8(rgba).to_rgb8());
        }
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        let (dst_lin, _) = apply_with_progress_inner(
            &lin,
            &params_lin,
            &std::sync::atomic::AtomicBool::new(false),
            None,
            Previews::None,
            |_, _, _| {},
        )?;
        Ok(gamma::linear_to_srgb_image(&dst_lin))
    } else {
        apply_inner(src, params)
    }
}

/// RGBA variant of `apply`: renders onto an RGBA canvas (handy when
/// `params.transparent` is true to produce a PNG with alpha).
///
/// Result: `(RGBA image, dots)`. The dots match the rendered image.
/// Unlike `apply_with_progress`, this path publishes no intermediate preview —
/// it computes the dots then renders the final image.
/// When `params.gamma_correct` is true, the pipeline works in linear space
/// then re-encodes the RGB channels of the RGBA buffer (alpha unchanged) to
/// sRGB.
pub fn apply_rgba(src: &RgbImage, params: &FilterParams) -> Result<(image::RgbaImage, Vec<Dot>)> {
    if params.gamma_correct {
        if is_cmyk_halftone(params) {
            let lin = gamma::srgb_to_linear_f32_image(src);
            let params_lin = linearized_clone(params);
            let dots = dots_halftone_linear(&lin, &params_lin.halftone_config());
            let image = render::render_halftone(src, &dots, params);
            return Ok((image, dots));
        }
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

/// Computes the dots (without PNG rendering) — lets the GUI cache them.
/// Computes dots directly without rendering an intermediate image.
pub fn compute_dots(src: &RgbImage, params: &FilterParams) -> Result<Vec<Dot>> {
    if params.gamma_correct {
        if is_cmyk_halftone(params) {
            let lin = gamma::srgb_to_linear_f32_image(src);
            let params_lin = linearized_clone(params);
            return Ok(dots_halftone_linear(&lin, &params_lin.halftone_config()));
        }
        let lin = gamma::srgb_to_linear_image(src);
        let params_lin = linearized_clone(params);
        let dots_lin = compute_dots_inner(&lin, &params_lin)?;
        Ok(dots_to_srgb(&dots_lin))
    } else {
        compute_dots_inner(src, params)
    }
}

// Imports only needed by the tests.
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

    // ── apply (integration) ───────────────────────────────────────────────────

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
        // 0 is valid: no Lloyd relaxation, raw dots from the seeds
        let p = FilterParams {
            iterations: 0,
            ..FilterParams::default()
        };
        assert!(
            validate_params(64, 64, &p).is_ok(),
            "iterations=0 doit passer"
        );
    }

    #[test]
    fn validate_halftone_dot_ratio_upper_bounds() {
        let base = FilterParams {
            algorithm: Algorithm::Halftone,
            halftone: HalftoneMode::Cmyk { angles: [0.0; 4] },
            ..FilterParams::default()
        };
        assert!(validate_params(64, 64, &base).is_ok());

        let huge_min = FilterParams {
            halftone_min_radius_ratio: 5.0,
            ..base.clone()
        };
        assert!(
            validate_params(64, 64, &huge_min).is_err(),
            "halftone_min_radius_ratio énorme doit échouer"
        );

        let huge_max = FilterParams {
            halftone_max_dot_ratio: 1.0e9,
            ..base.clone()
        };
        assert!(
            validate_params(64, 64, &huge_max).is_err(),
            "halftone_max_dot_ratio énorme doit échouer"
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
        let cancel = AtomicBool::new(true); // already cancelled
        let result = apply_with_progress(&img, &params, &cancel, |_, _, _| {});
        assert!(result.is_err(), "doit retourner Err quand cancel=true");
        assert!(
            is_cancelled(&result.unwrap_err()),
            "l'erreur doit être de type Cancelled"
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
