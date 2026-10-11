//! Reproductibilité : même graine → mêmes dots ; graines différentes → différents.

use pointimg::filter::FilterParams;
use pointimg::filter::{self, Algorithm, Dot};
use std::sync::atomic::AtomicBool;

/// Image à densité uniforme : un damier.
///
/// J'avais d'abord conclu que ce damier rendait le test vain, la densité devant
/// y être nulle partout et les graines ne jamais bouger. C'est faux, et c'est
/// vérifié : les 80 graines se déplacent toutes entre une et cinq itérations.
/// Le test a des dents. Ce qu'il ne couvre pas, en revanche, c'est le *poids* de
/// Lloyd — `lloyd_weights` y vaut partout la même valeur, donc aucune erreur de
/// calcul de ce poids ne se verrait. D'où `noisy` plus bas.
fn checkerboard(w: u32, h: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        if (x + y) % 2 == 0 {
            image::Rgb([255, 255, 255])
        } else {
            image::Rgb([0, 0, 0])
        }
    })
}

/// Image à densité réellement variable, pour exercer les poids de Lloyd.
fn noisy(w: u32, h: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        let t = (x * 7 + y * 13) % 97;
        image::Rgb([
            (t * 2 % 255) as u8,
            (t * 5 % 255) as u8,
            (t * 3 % 255) as u8,
        ])
    })
}

fn dots_for(seed: Option<u64>) -> Vec<Dot> {
    dots_from(&checkerboard(60, 60), seed, 0.0)
}

/// Même reproductibilité, mais sur une image à densité variable.
fn dots_noisy(seed: Option<u64>) -> Vec<Dot> {
    dots_from(&noisy(60, 60), seed, 0.7)
}

fn dots_from(img: &image::RgbImage, seed: Option<u64>, variance: f32) -> Vec<Dot> {
    let params = FilterParams {
        algorithm: Algorithm::Voronoi,
        num_points: 80,
        iterations: 5,
        rng_seed: seed,
        variance_sensitivity: variance,
        ..FilterParams::default()
    };
    let cancel = AtomicBool::new(false);
    filter::apply_with_progress(img, &params, &cancel, |_, _, _| {})
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

/// Les poids de Lloyd (`density^power`) ne sont pas constants sur cette image :
/// une erreur de calcul de ce poids déplacerait les graines.
#[test]
fn meme_graine_sur_image_a_variance_donne_les_memes_points() {
    let a = dots_noisy(Some(42));
    let b = dots_noisy(Some(42));
    assert_eq!(a.len(), b.len());
    for (da, db) in a.iter().zip(b.iter()) {
        assert_eq!(da.x.to_bits(), db.x.to_bits(), "x identique");
        assert_eq!(da.y.to_bits(), db.y.to_bits(), "y identique");
        assert_eq!(da.color, db.color, "color identique");
        assert_eq!(da.radius.to_bits(), db.radius.to_bits(), "radius identique");
    }
    // La variance doit faire une différence, sinon le nouveau test n'en serait
    // pas un : à poids uniforme, les deux images donneraient le même résultat.
    let sans = dots_for(Some(42));
    assert_ne!(
        sans[0].x.to_bits(),
        a[0].x.to_bits(),
        "la variance doit déplacer les graines, sinon ce test ne prouve rien"
    );
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
