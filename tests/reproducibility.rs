//! Reproductibilité : même graine → mêmes dots ; graines différentes → différents.

use pointimg::filter::FilterParams;
use pointimg::filter::{self, Algorithm, Dot};
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

fn dots_for(seed: Option<u64>) -> Vec<Dot> {
    let img = checkerboard(60, 60);
    let params = FilterParams {
        algorithm: Algorithm::Voronoi,
        num_points: 80,
        iterations: 5,
        rng_seed: seed,
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {})
        .unwrap()
        .1
}

#[test]
fn same_seed_produces_identical_dots() {
    let a = dots_for(Some(42));
    let b = dots_for(Some(42));
    assert_eq!(a.len(), b.len());
    for (da, db) in a.iter().zip(b.iter()) {
        assert_eq!(da.x.to_bits(), db.x.to_bits(), "x identique");
        assert_eq!(da.y.to_bits(), db.y.to_bits(), "y identique");
        assert_eq!(da.color, db.color, "color identique");
        assert_eq!(da.radius.to_bits(), db.radius.to_bits(), "radius identique");
    }
}

#[test]
fn different_seeds_produce_different_dots() {
    let a = dots_for(Some(1));
    let b = dots_for(Some(2));
    assert_eq!(a.len(), b.len());
    // Au moins un dot doit différer en position/color.
    let any_diff = a.iter().zip(b.iter()).any(|(da, db)| {
        da.x.to_bits() != db.x.to_bits() || da.y.to_bits() != db.y.to_bits() || da.color != db.color
    });
    assert!(any_diff, "deux seeds différentes doivent diverger");
}
