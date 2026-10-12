//! Integration tests for the complete pipeline through the public API.
//!
//! These tests only touch the public API of `pointimg::filter`, unlike
//! the unit tests in `src/filter/mod.rs` which inspect the internals.

use pointimg::filter::{self, Algorithm, DotShape, FilterParams, HalftoneMode};
use std::sync::atomic::AtomicBool;

fn checkerboard(w: u32, h: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        if (x + y) % 2 == 0 {
            image::Rgb([255, 255, 255])
        } else {
            image::Rgb([0, 0, 0])
        }
    })
}

/// Each check in `validate_params` must mention the field it rejects.
///
/// `validate_params` has been split into seven functions; these tests check
/// that the split neither merged two checks nor lost a message. The boundary
/// that is *just accepted* is tested too, so that a check which has become too
/// loose shows up.
#[test]
fn chaque_controle_nomme_le_champ_quil_refuse() {
    fn refuse(nom: &str, corps: impl FnOnce(&mut FilterParams)) {
        let mut params = FilterParams::default();
        corps(&mut params);
        let err = filter::validate_params(64, 64, &params)
            .expect_err(&format!("{nom} aurait dû être refusé"));
        assert!(
            err.to_string().contains(nom),
            "{nom} : le message doit nommer le champ, trouvé « {err} »"
        );
    }
    fn accepte(corps: impl FnOnce(&mut FilterParams)) {
        let mut params = FilterParams::default();
        corps(&mut params);
        filter::validate_params(64, 64, &params).expect("réglage valide");
    }

    refuse("min_radius_ratio", |p| p.min_radius_ratio = 0.0);
    refuse("max_radius_ratio", |p| p.max_radius_ratio = 2.0);
    refuse("num_points", |p| p.num_points = 0);
    refuse("cols", |p| p.cols = 0);
    refuse("iterations", |p| p.iterations = 200);
    refuse("palette_size", |p| p.palette_size = Some(1));
    refuse("variance_sensitivity", |p| p.variance_sensitivity = 2.0);
    refuse("max_boost", |p| p.max_boost = 0.5);
    refuse("halftone_frequency", |p| p.halftone_frequency = 0.0);
    refuse("halftone_min_radius_ratio", |p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone = HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        };
        p.halftone_min_radius_ratio = 2.0;
    });
    refuse("halftone_max_dot_ratio", |p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone = HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        };
        p.halftone_max_dot_ratio = 20.0;
    });
    refuse("polygon_sides", |p| {
        p.dot_shape = DotShape::RegularPolygon { sides: 2 }
    });
    refuse("ellipse_aspect", |p| {
        p.dot_shape = DotShape::Ellipse {
            aspect: 20.0,
            angle_deg: 0.0,
        }
    });
    // The missing ink mode is a special case: the message talks about
    // the algorithm, not about a field.
    let params = FilterParams {
        algorithm: Algorithm::Halftone,
        ..FilterParams::default()
    };
    assert!(
        filter::validate_params(64, 64, &params)
            .unwrap_err()
            .to_string()
            .contains("Halftone"),
        "un halftone sans mode d'encre doit être refusé par son nom"
    );

    // And the boundary that is just accepted, so that a loosened check shows up.
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone = HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        };
        p.halftone_min_radius_ratio = 1.0;
    });
}

#[test]
fn input_dimension_limits_are_checked_before_processing() {
    assert!(filter::validate_image_dimensions(4096, 2048).is_ok());
    assert!(filter::validate_image_dimensions(4096, 4096).is_err());
    assert!(filter::validate_image_dimensions(0, 100).is_err());
}

#[test]
fn oversized_images_are_resized_to_pipeline_limits() {
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::new(8_000, 2_000));
    let (resized, changed) = filter::resize_to_limits(image);
    assert!(changed);
    assert!(filter::validate_image_dimensions(resized.width(), resized.height()).is_ok());
    assert!(filter::estimate_memory_bytes(resized.width(), resized.height()) > 0);
}

#[test]
fn non_finite_parameters_are_rejected() {
    let img = checkerboard(8, 8);
    let params = FilterParams {
        max_boost: f32::NAN,
        ..FilterParams::default()
    };
    assert!(filter::apply(&img, &params).is_err());
}

#[test]
fn invalid_parameters_are_rejected_for_grid_too() {
    let img = checkerboard(8, 8);
    let params = FilterParams {
        algorithm: Algorithm::Grid,
        min_radius_ratio: 0.0,
        ..FilterParams::default()
    };
    assert!(filter::apply(&img, &params).is_err());
}

#[test]
fn invalid_halftone_frequency_is_rejected() {
    let img = checkerboard(8, 8);
    let params = FilterParams {
        algorithm: Algorithm::Halftone,
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        halftone_frequency: 0.0,
        ..FilterParams::default()
    };
    assert!(filter::apply(&img, &params).is_err());
}

#[test]
fn halftone_without_mode_is_rejected() {
    let img = checkerboard(16, 16);
    let params = FilterParams {
        algorithm: Algorithm::Halftone,
        ..FilterParams::default()
    };
    assert!(filter::apply(&img, &params).is_err());
}

#[test]
fn iterative_algorithms_reject_excessive_point_counts() {
    let params = FilterParams {
        algorithm: Algorithm::Voronoi,
        num_points: 50_001,
        ..FilterParams::default()
    };
    assert!(filter::validate_params(64, 64, &params).is_err());
}

#[test]
fn cached_density_matches_regular_pipeline() {
    let img = checkerboard(32, 24);
    let params = FilterParams {
        algorithm: Algorithm::Voronoi,
        num_points: 20,
        iterations: 2,
        rng_seed: Some(5),
        ..FilterParams::default()
    };
    let density = filter::compute_density_map(&img, params.variance_sensitivity);
    let cancel = AtomicBool::new(false);
    let (_, normal) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    let (_, cached) =
        filter::apply_with_progress_cached(&img, &params, &cancel, &density, |_, _, _| {}).unwrap();
    assert_eq!(normal.len(), cached.len());
    for (a, b) in normal.iter().zip(cached.iter()) {
        assert_eq!(a.x, b.x);
        assert_eq!(a.y, b.y);
    }
}

#[test]
fn cached_density_rejects_wrong_length() {
    let img = checkerboard(16, 12);
    let params = FilterParams::default();
    let cancel = AtomicBool::new(false);
    let error = filter::apply_with_progress_cached(&img, &params, &cancel, &[], |_, _, _| {})
        .expect_err("une density map de taille incorrecte doit être rejetée");
    assert!(error.to_string().contains("density map invalide"));
}

#[test]
fn each_algorithm_preserves_dimensions() {
    let img = checkerboard(64, 48);
    for algo in [
        Algorithm::Grid,
        Algorithm::Kmeans,
        Algorithm::Voronoi,
        Algorithm::Quadtree,
    ] {
        let params = FilterParams {
            algorithm: algo,
            num_points: 30,
            cols: 10,
            iterations: 3,
            ..FilterParams::default()
        };
        let out = filter::apply(&img, &params).expect("apply doit réussir");
        assert_eq!(out.dimensions(), (64, 48), "algo {algo:?}");
    }
}

#[test]
fn apply_with_progress_returns_dots_matching_image() {
    let img = checkerboard(80, 60);
    let params = FilterParams {
        algorithm: Algorithm::Voronoi,
        num_points: 40,
        iterations: 4,
        rng_seed: Some(7),
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    let (out, dots) =
        filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).expect("ok");
    assert_eq!(out.dimensions(), (80, 60));
    assert!(!dots.is_empty(), "doit produire des dots");
    for d in &dots {
        assert!(d.radius > 0.0, "rayon positif requis");
    }
}

#[test]
fn compute_dots_matches_apply_in_count() {
    let img = checkerboard(50, 50);
    let params = FilterParams {
        algorithm: Algorithm::Quadtree,
        num_points: 60,
        rng_seed: Some(11),
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    let (_, dots_apply) =
        filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).expect("ok");
    let dots_compute = filter::compute_dots(&img, &params).expect("ok");
    // Quadtree is deterministic (no RNG), so both paths must produce the same
    // number of dots.
    assert_eq!(
        dots_apply.len(),
        dots_compute.len(),
        "compute_dots vs apply_with_progress: {} vs {}",
        dots_apply.len(),
        dots_compute.len()
    );
}

#[test]
fn flatten_to_rgb_composites_alpha_on_bg() {
    use image::{DynamicImage, Rgba, RgbaImage};
    let rgba = RgbaImage::from_pixel(8, 8, Rgba([255, 0, 0, 128]));
    let dyn_img = DynamicImage::ImageRgba8(rgba);
    let rgb_white = filter::flatten_to_rgb(&dyn_img, [255, 255, 255]);
    let p = rgb_white.get_pixel(0, 0);
    // alpha 128/255 on a white background → red faded toward white
    assert!(p[0] > 128, "R composite doit etre > 128, got {}", p[0]);
    assert!(
        p[1] > 0 && p[1] < 200,
        "G composite intermediaire, got {}",
        p[1]
    );
}

#[test]
fn compute_density_image_dimensions_match_source() {
    let img = checkerboard(40, 30);
    let di = filter::compute_density_image(&img, 0.7);
    assert_eq!(di.dimensions(), (40, 30));
}

#[test]
fn zero_iterations_voronoi_produces_dots() {
    // iterations=0 is valid: raw seeds, no Lloyd relaxation
    let img = checkerboard(48, 48);
    let params = FilterParams {
        algorithm: Algorithm::Voronoi,
        num_points: 20,
        iterations: 0,
        rng_seed: Some(3),
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    let (out, dots) =
        filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).expect("ok");
    assert_eq!(out.dimensions(), (48, 48));
    assert!(!dots.is_empty(), "même sans itérations, des dots sont émis");
}

#[test]
fn apply_rgba_transparent_produces_alpha_channel() {
    // Solid white 80x80 image → white dots, but a transparent background.
    let img = image::RgbImage::from_pixel(80, 80, image::Rgb([255, 255, 255]));
    let params = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 10,
        min_radius_ratio: 0.01,
        max_radius_ratio: 0.02,
        transparent: true,
        ..FilterParams::default()
    };
    let (dst, _dots) = filter::apply_rgba(&img, &params).expect("ok");
    assert_eq!(dst.dimensions(), (80, 80), "dimensions conservées");
    // Count the transparent pixels versus the opaque/partial ones.
    let mut transparent = 0u32;
    let mut covered = 0u32;
    for px in dst.pixels() {
        if px[3] == 0 {
            transparent += 1;
        } else {
            covered += 1;
        }
    }
    // cell = 8px, r <= 0.02 * 80 = 1.6px, and capped at cell/2*0.8 = 3.2px.
    // All the dots have a very small radius, so there should be many pixels
    // between the dots that stay transparent.
    assert!(
        transparent > 0,
        "doit avoir au moins un pixel transparent (entre les dots), got {transparent}"
    );
    assert!(
        covered > 0,
        "doit avoir au moins un pixel couvert par un dot, got {covered}"
    );
}

#[test]
fn rgba_palette_dithering_matches_rgb_when_background_is_opaque() {
    let img = checkerboard(32, 24);
    let params = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 8,
        palette_size: Some(4),
        dithering: pointimg::filter::Dithering::FloydSteinberg,
        transparent: false,
        ..FilterParams::default()
    };
    let rgb = filter::apply(&img, &params).expect("rendu RGB");
    let (rgba, _) = filter::apply_rgba(&img, &params).expect("rendu RGBA");
    for (rgb_pixel, rgba_pixel) in rgb.pixels().zip(rgba.pixels()) {
        assert_eq!(rgb_pixel.0, [rgba_pixel[0], rgba_pixel[1], rgba_pixel[2]]);
        assert_eq!(rgba_pixel[3], 255);
    }
}

#[test]
fn render_svg_from_dots_transparent_omits_bg_rect() {
    let img = checkerboard(20, 16);
    let cancel = AtomicBool::new(false);
    let params = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 8,
        rng_seed: Some(1),
        transparent: true,
        ..FilterParams::default()
    };
    let (_, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    let svg = filter::render_svg_from_dots(20, 16, &dots, &params).unwrap();
    assert!(svg.starts_with("<svg"), "doit commencer par <svg");
    assert!(
        !svg.contains("<rect"),
        "mode transparent ne doit pas émettre de <rect> de fond"
    );
    assert!(svg.contains("</svg>"), "doit finir par </svg>");
}

#[test]
fn filter_params_toml_round_trip_preserves_fields() {
    let params = FilterParams {
        algorithm: Algorithm::Voronoi,
        num_points: 1234,
        cols: 42,
        min_radius_ratio: 0.005,
        max_radius_ratio: 0.07,
        bg_color: [26, 58, 92],
        iterations: 7,
        variance_sensitivity: 0.55,
        max_boost: 3.0,
        rng_seed: Some(99),
        palette_size: Some(16),
        dot_shape: pointimg::filter::DotShape::RegularPolygon { sides: 5 },
        transparent: true,
        gamma_correct: true,
        dithering: pointimg::filter::Dithering::Off,
        grid_angle_deg: 0.0,
        halftone: pointimg::filter::HalftoneMode::Off,
        screening: pointimg::filter::Screening::Am,
        halftone_frequency: 60.0,
        halftone_min_radius_ratio: 0.002,
        halftone_max_dot_ratio: 0.85,
    };
    let toml = params.to_toml_string().expect("serialize OK");
    let back = FilterParams::from_toml_str(&toml).expect("deserialize OK");
    assert_eq!(back.algorithm, params.algorithm);
    assert_eq!(back.num_points, params.num_points);
    assert_eq!(back.cols, params.cols);
    assert_eq!(back.bg_color, params.bg_color);
    assert_eq!(back.iterations, params.iterations);
    assert!((back.variance_sensitivity - params.variance_sensitivity).abs() < 1e-6);
    assert_eq!(back.rng_seed, params.rng_seed);
    assert_eq!(back.palette_size, params.palette_size);
    assert_eq!(back.dot_shape, params.dot_shape);
    assert_eq!(back.transparent, params.transparent);
    assert_eq!(back.gamma_correct, params.gamma_correct);
}

#[test]
fn invalid_preset_parameters_are_rejected_before_serialization() {
    let params = FilterParams {
        min_radius_ratio: 0.0,
        ..FilterParams::default()
    };
    assert!(params.to_toml_string().is_err());
}

#[test]
fn presets_keep_legacy_flat_format_readable() {
    let params = FilterParams::default();
    let legacy = toml::to_string(&params).expect("sérialisation legacy");
    let loaded = FilterParams::from_toml_str(&legacy).expect("preset legacy lisible");
    assert_eq!(loaded, params);
}

#[test]
fn presets_reject_unknown_format_versions() {
    let params = FilterParams::default();
    let current = params.to_toml_string().expect("sérialisation versionnée");
    let future = current.replacen("preset_version = 1", "preset_version = 999", 1);
    let error = FilterParams::from_toml_str(&future).expect_err("version future rejetée");
    assert!(error.to_string().contains("non supportée"));
}

#[test]
fn gamma_correction_makes_mixed_midtones_brighter() {
    // Image half black, half white: the perceptual (linear) mean
    // ~= 188 (sRGB), whereas the naive sRGB mean = 128. The gamma correction
    // must therefore produce a significantly lighter colored dot.
    let img = image::RgbImage::from_fn(16, 16, |x, _y| {
        if x < 8 {
            image::Rgb([0, 0, 0])
        } else {
            image::Rgb([255, 255, 255])
        }
    });
    let params_gamma = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 1, // a single dot covering the whole image
        gamma_correct: true,
        rng_seed: Some(1),
        ..FilterParams::default()
    };
    let params_no_gamma = FilterParams {
        gamma_correct: false,
        ..params_gamma.clone()
    };
    let cancel = AtomicBool::new(false);
    let (_, dots_g) =
        filter::apply_with_progress(&img, &params_gamma, &cancel, |_, _, _| {}).unwrap();
    let (_, dots_ng) =
        filter::apply_with_progress(&img, &params_no_gamma, &cancel, |_, _, _| {}).unwrap();
    let avg_gamma = dots_g[0].color[0] as i32;
    let avg_no_gamma = dots_ng[0].color[0] as i32;
    // Expected: 127-128 without gamma (sRGB mean), ~188 with gamma (linear).
    assert!(
        (120..=128).contains(&avg_no_gamma),
        "sans gamma, moyenne sRGB de (0,255) ≈ 127-128, got {avg_no_gamma}"
    );
    assert!(
        avg_gamma >= 160,
        "avec gamma, moyenne linéaire doit être bien plus claire que 128, got {avg_gamma}"
    );
    assert!(
        avg_gamma > avg_no_gamma + 20,
        "gamma doit significativement éclaircir la moyenne mixte"
    );
}
