//! The preview-less path must render exactly what the direct path renders.
//!
//! `apply_with_iteration_counts` was introduced for the CLI, which never
//! displayed the images that `apply_with_progress` returned to it at each
//! Lloyd iteration. Dropping that work must not change a single pixel.
//!
//! The test therefore compares the two functions on the iterative algorithms,
//! sweeping the settings that drive progression.

use pointimg::filter::{self, Algorithm, FilterParams, HalftoneMode};
use std::sync::atomic::AtomicBool;

fn source(w: u32, h: u32) -> image::RgbImage {
    // A noisy gradient rather than a checkerboard: Lloyd needs variance
    // to move the points, otherwise every iteration is a no-op and the test
    // would not prove anything.
    image::RgbImage::from_fn(w, h, |x, y| {
        let t = (x * 7 + y * 13) % 97;
        image::Rgb([
            (t * 2 % 255) as u8,
            (t * 5 % 255) as u8,
            (t * 3 % 255) as u8,
        ])
    })
}

fn base(algorithm: Algorithm) -> FilterParams {
    FilterParams {
        algorithm,
        ..FilterParams::default()
    }
}

/// Compares the two paths, pixel by pixel and dot by dot.
fn identique(algorithm: Algorithm, iterations: usize, variance: f32) {
    let src = source(96, 72);
    let mut params = base(algorithm);
    params.iterations = iterations;
    params.variance_sensitivity = variance;
    // The seed is fixed: without it (`rng_seed: None`) the initial sampling
    // pulls from the clock, and two calls *must* differ - this is documented.
    // Comparing therefore requires an explicit seed.
    params.rng_seed = Some(42);
    // A small number of points keeps the test short without changing the nature
    // of the path: the Lloyd loop still runs.
    params.num_points = 120;

    let cancel = AtomicBool::new(false);
    let attendu = filter::apply(&src, &params).expect("chemin direct");
    let (image, dots_compte) =
        filter::apply_with_iteration_counts(&src, &params, &cancel, |iter, total| {
            assert!(iter <= total, "la progression dépasse le total")
        })
        .expect("chemin sans aperçus");

    // The image itself: `apply` is the historical direct path, the one the
    // CLI already used elsewhere.
    assert_eq!(
        image, attendu,
        "{algorithm:?} it={iterations} var={variance} : image différente"
    );

    // The dots separately, because an image equality does not say *where* the
    // computation diverged if it ever diverges.
    let (_img_direct, dots_direct) =
        filter::apply_with_progress(&src, &params, &cancel, |_iter, _total, _preview| {})
            .expect("chemin avec aperçus");
    assert_eq!(
        dots_direct.len(),
        dots_compte.len(),
        "{algorithm:?} it={iterations} var={variance} : nombre de points différent"
    );
    for (a, b) in dots_direct.iter().zip(dots_compte.iter()) {
        assert_eq!(
            a.x, b.x,
            "{algorithm:?} it={iterations} : absisse divergente"
        );
        assert_eq!(
            a.y, b.y,
            "{algorithm:?} it={iterations} : ordonnée divergente"
        );
        assert_eq!(a.radius, b.radius, "{algorithm:?} : rayon divergent");
    }
}

#[test]
fn le_chemin_sans_apercus_rend_le_meme_resultat() {
    for algorithm in [Algorithm::Kmeans, Algorithm::Voronoi] {
        for iterations in [1, 3, 10] {
            for variance in [0.0, 0.5, 1.0] {
                identique(algorithm, iterations, variance);
            }
        }
    }
}

#[test]
fn le_chemin_sans_apercus_annonce_la_progression_attendue() {
    let src = source(64, 48);
    let mut params = base(Algorithm::Voronoi);
    params.iterations = 7;
    params.num_points = 60;

    let cancel = AtomicBool::new(false);
    let mut vues: Vec<(usize, usize)> = Vec::new();
    filter::apply_with_iteration_counts(&src, &params, &cancel, |iter, total| {
        vues.push((iter, total));
    })
    .expect("chemin sans aperçus");

    assert_eq!(vues.len(), 7, "une annonce par itération, pas plus");
    assert!(
        vues.iter().all(|(_, total)| *total == 7),
        "le total annoncé doit être stable : {vues:?}"
    );
    assert_eq!(
        vues.iter().map(|(iter, _)| *iter).collect::<Vec<_>>(),
        (1..=7).collect::<Vec<_>>(),
        "les itérations se suivent, de 1 à 7"
    );
}

/// Halftone has no Lloyd loop: the preview-less path must render the same
/// result with a single announcement.
#[test]
fn le_halftone_passe_aussi_par_le_chemin_sans_apercus() {
    let src = source(48, 48);
    let mut params = base(Algorithm::Halftone);
    params.halftone = HalftoneMode::Cmyk {
        angles: [15.0, 75.0, 0.0, 45.0],
    };
    params.halftone_frequency = 20.0;

    let cancel = AtomicBool::new(false);
    let mut annonces = 0;
    let (image, _dots) =
        filter::apply_with_iteration_counts(&src, &params, &cancel, |_iter, total| {
            annonces += 1;
            assert_eq!(total, 1, "le halftone n'a qu'une passe");
        })
        .expect("chemin sans aperçus");
    assert_eq!(annonces, 1, "une seule annonce pour une passe unique");
    let attendu = filter::apply(&src, &params).expect("chemin direct");
    assert_eq!(image, attendu, "le rendu halftone doit être identique");
}
