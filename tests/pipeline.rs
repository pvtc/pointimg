//! Tests d'intégration du pipeline complet via l'API publique.
//!
//! Ces tests ne touchent que l'API publique de `pointimg::filter`, contrairement
//! aux tests unitaires dans `src/filter/mod.rs` qui inspectent les internals.

use pointimg::filter::{self, Algorithm, FilterParams};
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
    // Quadtree est déterministe (pas de RNG), donc les deux chemins doivent
    // produire le même nombre de dots.
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
    // alpha 128/255 sur fond blanc → rouge atténué vers blanc
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
    // iterations=0 est valide : seeds bruts, pas de relaxation Lloyd
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
    // Image 80x80 blanche unie → dots blancs, mais le fond transparent.
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
    // Compter les pixels transparents vs opaques/partiels.
    let mut transparent = 0u32;
    let mut covered = 0u32;
    for px in dst.pixels() {
        if px[3] == 0 {
            transparent += 1;
        } else {
            covered += 1;
        }
    }
    // cell = 8px, r ≤ 0.02 * 80 = 1.6px, et capped à cell/2*0.8 = 3.2px.
    // Tous les dots ont un rayon très petit, donc il doit y avoir beaucoup de
    // pixels entre les dots qui restent transparents.
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
        dithering: false,
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
fn gamma_correction_makes_mixed_midtones_brighter() {
    // Image moitié noire, moitié blanche : la moyenne perceptuelle (linéaire)
    // ≈ 188 (sRGB), alors que la moyenne naïve sRGB = 128. La correction gamma
    // doit donc produire une dot color significativement plus claire.
    let img = image::RgbImage::from_fn(16, 16, |x, _y| {
        if x < 8 {
            image::Rgb([0, 0, 0])
        } else {
            image::Rgb([255, 255, 255])
        }
    });
    let params_gamma = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 1, // un seul dot couvrant toute l'image
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
    // Attente : 127-128 sans gamma (moyenne sRGB), ~188 avec gamma (linéaire).
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
