//! Integration tests for the halftone mode (CMYK rosette + dominant colors).

use image::RgbImage;
use pointimg::filter::{self, FilterParams, HalftoneMode, Screening};
use std::sync::atomic::AtomicBool;

fn checkerboard(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        if (x + y) % 2 == 0 {
            image::Rgb([255, 255, 255])
        } else {
            image::Rgb([0, 0, 0])
        }
    })
}

#[test]
fn halftone_cmyk_preserves_dimensions() {
    // Image with saturated Cyan/Magenta/Yellow/Black zones to activate all 4 channels.
    let img = RgbImage::from_fn(64, 64, |x, y| match (x / 16, y / 16) {
        (0, 0) | (3, 3) => image::Rgb([0, 0, 0]), // black → K channel
        (1, 0) | (2, 3) => image::Rgb([0, 255, 255]), // pure cyan
        (2, 0) | (1, 3) => image::Rgb([255, 0, 255]), // pure magenta
        (3, 0) | (0, 3) => image::Rgb([255, 255, 0]), // pure yellow
        (0, 1) | (3, 2) => image::Rgb([255, 0, 0]), // red (magenta+yellow) → M+Y
        (1, 1) | (2, 2) => image::Rgb([0, 255, 0]), // green (cyan+yellow) → C+Y
        (2, 1) | (1, 2) => image::Rgb([0, 0, 255]), // blue (cyan+magenta) → C+M
        _ => image::Rgb([128, 128, 128]),         // neutral → activates K
    });
    let params = FilterParams {
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        rng_seed: Some(7),
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    let (out, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    assert_eq!(out.dimensions(), (64, 64), "dimensions conservées");
    assert!(!dots.is_empty(), "halftone CMYK doit émettre des dots");
    // 4 channels → at least 4 distinct colors (C, M, Y, K).
    use std::collections::HashSet;
    let colors: HashSet<_> = dots.iter().map(|d| d.color).collect();
    assert!(
        colors.len() >= 3,
        "CMYK doit contenir ≥ 3 couleurs, got {}",
        colors.len()
    );
    assert!(colors.contains(&[0, 255, 255]), "Cyan manquant");
    assert!(colors.contains(&[255, 0, 255]), "Magenta manquant");
    assert!(colors.contains(&[255, 255, 0]), "Yellow manquant");
    assert!(colors.contains(&[0, 0, 0]), "Black manquant");
}

#[test]
fn halftone_dominant_emits_n_channels() {
    let img = RgbImage::from_fn(60, 60, |x, y| match ((x / 30) + (y / 30)) % 3 {
        0 => image::Rgb([255, 0, 0]),
        1 => image::Rgb([0, 255, 0]),
        _ => image::Rgb([0, 0, 255]),
    });
    let n_channels = 3;
    let params = FilterParams {
        halftone: HalftoneMode::Dominant {
            n: n_channels,
            base_angle_deg: 0.0,
        },
        rng_seed: Some(42),
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    let (out, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    assert_eq!(out.dimensions(), (60, 60));
    use std::collections::HashSet;
    let colors: HashSet<_> = dots.iter().map(|d| d.color).collect();
    // The k-means centers come close to the 3 source colors; we accept ≥ 2.
    assert!(
        colors.len() >= 2,
        "Dominant-3 doit produire ≥ 2 canaux distincts, got {}",
        colors.len()
    );
}

#[test]
fn halftone_fm_screening_produces_content() {
    let img = checkerboard(50, 50);
    let params = FilterParams {
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        screening: Screening::Fm,
        rng_seed: Some(99),
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    let (out, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    assert_eq!(out.dimensions(), (50, 50));
    assert!(!dots.is_empty(), "FM doit émettre des dots");
    // FM must contain at least 1 ink color (true on a contrasted image).
    use std::collections::HashSet;
    let colors: HashSet<_> = dots.iter().map(|d| d.color).collect();
    assert!(!colors.is_empty(), "FM doit avoir au moins 1 couleur");
}

#[test]
fn gamma_corrected_cmyk_halftone_produces_content() {
    let img = RgbImage::from_fn(64, 64, |x, y| {
        if (x / 8 + y / 8) % 2 == 0 {
            image::Rgb([32, 96, 180])
        } else {
            image::Rgb([220, 180, 64])
        }
    });
    let params = FilterParams {
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        gamma_correct: true,
        rng_seed: Some(4),
        ..FilterParams::default()
    };
    let (out, dots) = filter::apply_rgba(&img, &params).unwrap();
    assert_eq!(out.dimensions(), (64, 64));
    assert!(!dots.is_empty());
}

#[test]
fn halftone_off_path_unchanged() {
    // Make sure HalftoneMode::Off falls back to the historical pipeline.
    let img = checkerboard(32, 32);
    let params = FilterParams {
        algorithm: pointimg::filter::Algorithm::Grid,
        cols: 8,
        rng_seed: Some(1),
        halftone: HalftoneMode::Off,
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    let (out, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    assert_eq!(out.dimensions(), (32, 32));
    // Grid 8 cols → at least 50 dots for 32×32.
    assert!(
        dots.len() >= 50,
        "Mode Off Grid 8 cols devrait produire ~64 dots, got {}",
        dots.len()
    );
}

#[test]
fn apply_dynamic_gamma_cmyk_halftone_does_not_panic() {
    // Regression: the gamma+halftone path of `apply_dynamic` went through an
    // internal route that reached `unreachable!()`.
    use pointimg::filter::Algorithm;
    let img = RgbImage::from_fn(32, 32, |x, y| {
        if (x + y) % 2 == 0 {
            image::Rgb([0, 255, 255])
        } else {
            image::Rgb([255, 0, 0])
        }
    });
    let dynamic = image::DynamicImage::ImageRgb8(img);
    let params = FilterParams {
        algorithm: Algorithm::Halftone,
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        gamma_correct: true,
        rng_seed: Some(1),
        ..FilterParams::default()
    };
    let out = filter::apply_dynamic(&dynamic, &params).unwrap();
    assert_eq!(out.dimensions(), (32, 32));
}

#[test]
fn apply_gamma_dominant_halftone_does_not_panic() {
    // Regression: `apply` in gamma would route CMYK alone to `apply_rgba`, and
    // the dominant mode would reach `unreachable!()`.
    use pointimg::filter::Algorithm;
    let img = RgbImage::from_fn(32, 32, |x, y| match (x / 16 + y / 16) % 3 {
        0 => image::Rgb([255, 0, 0]),
        1 => image::Rgb([0, 255, 0]),
        _ => image::Rgb([0, 0, 255]),
    });
    let params = FilterParams {
        algorithm: Algorithm::Halftone,
        halftone: HalftoneMode::Dominant {
            n: 3,
            base_angle_deg: 0.0,
        },
        gamma_correct: true,
        rng_seed: Some(1),
        ..FilterParams::default()
    };
    let out = filter::apply(&img, &params).unwrap();
    assert_eq!(out.dimensions(), (32, 32));
}

#[test]
fn apply_with_progress_cached_halftone_does_not_panic() {
    // Regression: the "cached" path sent halftone to
    // `apply_with_progress_inner`, which does not handle it (`unreachable!()`).
    use pointimg::filter::Algorithm;
    let img = RgbImage::from_fn(16, 16, |_, _| image::Rgb([0, 255, 255]));
    let density = vec![1.0f32; 16 * 16];
    let cancel = AtomicBool::new(false);
    let params = FilterParams {
        algorithm: Algorithm::Halftone,
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        ..FilterParams::default()
    };
    let (out, dots) =
        filter::apply_with_progress_cached(&img, &params, &cancel, &density, |_, _, _| {}).unwrap();
    assert_eq!(out.dimensions(), (16, 16));
    assert!(!dots.is_empty());
}

#[test]
fn halftone_params_toml_round_trip() {
    let params = FilterParams {
        halftone: HalftoneMode::Cmyk {
            angles: [10.0, 70.0, 5.0, 40.0],
        },
        screening: Screening::Fm,
        halftone_frequency: 75.0,
        ..FilterParams::default()
    };
    let s = params.to_toml_string().unwrap();
    let back = FilterParams::from_toml_str(&s).unwrap();
    assert_eq!(back.halftone, params.halftone);
    assert_eq!(back.screening, params.screening);
    assert_eq!(back.halftone_frequency, params.halftone_frequency);
}
