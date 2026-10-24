//! Consistency between the ranges of the GTK interface and those of the engine.
//!
//! Equivalent of `src/gui/ranges.rs` for the GTK4 front-end. The two lists
//! remain distinct and independent: the bounds are a consequence of what the
//! engine accepts, not an interface contract, and each one can lie about the
//! scale (linear or logarithmic) without the other noticing.
//!
//! What the test guarantees is that neither of them offers a value that
//! `filter::validate_params` would refuse.

use std::ops::RangeInclusive;

use pointimg::filter::{Algorithm, DotShape, FilterParams, HalftoneMode, validate_params};

/// Bounds of the `adw::SpinRow`s of `controls.rs`.
///
/// The `ValueSlider`s that double these rows take up the same bounds via
/// `Curve`: changing one without the other shows up here, since the tested
/// bound is that of the `SpinRow` and the slider's follows from it.
mod bornes {
    use super::RangeInclusive;

    pub const VARIANCE: RangeInclusive<f32> = 0.0..=1.0;
    pub const RAYON: RangeInclusive<f32> = 0.0005..=1.0;
    pub const BOOST: RangeInclusive<f32> = 1.0..=8.0;
    pub const POINTS: RangeInclusive<f32> = 10.0..=100_000.0;
    pub const ITERATIONS: RangeInclusive<f32> = 1.0..=100.0;
    pub const COLONNES: RangeInclusive<f32> = 2.0..=8192.0;
    pub const ANGLE_GRILLE: RangeInclusive<f32> = -180.0..=180.0;
    pub const FREQUENCE_TRAME: RangeInclusive<f32> = 4.0..=400.0;
    /// Low and high bounds of the screen radius, separately.
    ///
    /// The engine caps `halftone_min_radius_ratio` at 1.0 when
    /// `halftone_max_dot_ratio` rises to 10 (`validate_params`). A single
    /// shared constant let the slider of the low bound cross that cap: the
    /// user got a bare worker error, and the range tests did not see it
    /// because they only exercised the low end of the low bound and the high
    /// end of the high bound.
    pub const RAYON_TRAME_MIN: RangeInclusive<f32> = 0.0..=1.0;
    pub const RAYON_TRAME_MAX: RangeInclusive<f32> = 0.05..=10.0;
    pub const ANGLE_TRAME: RangeInclusive<f32> = 0.0..=180.0;
    pub const CANAUX: RangeInclusive<f32> = 2.0..=32.0;
    pub const PALETTE: RangeInclusive<f32> = 2.0..=256.0;
    pub const RAPPORT: RangeInclusive<f32> = 0.1..=10.0;
    pub const COTES: RangeInclusive<f32> = 3.0..=12.0;
}

/// Default settings, plus one valid ink mode: `HalftoneMode::Off` combined with
/// the rest of the default values is refused by the engine.
fn base() -> FilterParams {
    FilterParams {
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        ..FilterParams::default()
    }
}

fn accepte(applique: impl Fn(&mut FilterParams)) {
    let mut params = base();
    applique(&mut params);
    validate_params(64, 64, &params).unwrap_or_else(|e| {
        panic!(
            "{}",
            pointimg::frontend::t("réglages refusés : {e}").replace("{e}", &e.to_string())
        )
    });
}

fn refuse(applique: impl Fn(&mut FilterParams)) {
    let mut params = base();
    applique(&mut params);
    assert!(
        validate_params(64, 64, &params).is_err(),
        "{}",
        pointimg::frontend::t(
            "le moteur accepte une valeur que l'interface ne devrait pas proposer"
        )
    );
}

#[test]
fn les_deux_bornes_de_chaque_champ_de_saisie_sont_acceptees() {
    accepte(|p| p.variance_sensitivity = *bornes::VARIANCE.start());
    accepte(|p| p.variance_sensitivity = *bornes::VARIANCE.end());
    accepte(|p| p.min_radius_ratio = *bornes::RAYON.start());
    accepte(|p| {
        p.min_radius_ratio = *bornes::RAYON.start();
        p.max_radius_ratio = *bornes::RAYON.end();
    });
    accepte(|p| p.max_boost = *bornes::BOOST.start());
    accepte(|p| p.max_boost = *bornes::BOOST.end());
    accepte(|p| {
        p.algorithm = Algorithm::Grid;
        p.cols = *bornes::COLONNES.start() as u32;
    });
    accepte(|p| {
        p.algorithm = Algorithm::Grid;
        p.cols = *bornes::COLONNES.end() as u32;
    });
    accepte(|p| {
        p.algorithm = Algorithm::Quadtree;
        p.num_points = *bornes::POINTS.start() as usize;
    });
    accepte(|p| {
        p.algorithm = Algorithm::Quadtree;
        p.num_points = *bornes::POINTS.end() as usize;
    });
    accepte(|p| p.iterations = *bornes::ITERATIONS.start() as usize);
    accepte(|p| p.iterations = *bornes::ITERATIONS.end() as usize);
    accepte(|p| {
        p.algorithm = Algorithm::Grid;
        p.grid_angle_deg = *bornes::ANGLE_GRILLE.start();
    });
    accepte(|p| {
        p.algorithm = Algorithm::Grid;
        p.grid_angle_deg = *bornes::ANGLE_GRILLE.end();
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_frequency = *bornes::FREQUENCE_TRAME.start();
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_frequency = *bornes::FREQUENCE_TRAME.end();
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_min_radius_ratio = *bornes::RAYON_TRAME_MIN.start();
    });
    // The top of the range of the *low* bound: it is the case that sharing a
    // single constant let through. As long as it is not exercised, a shared
    // slider can cross 1.0 without any test turning red.
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_min_radius_ratio = *bornes::RAYON_TRAME_MIN.end();
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_max_dot_ratio = *bornes::RAYON_TRAME_MAX.start();
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_max_dot_ratio = *bornes::RAYON_TRAME_MAX.end();
    });
    refuse(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_min_radius_ratio = *bornes::RAYON_TRAME_MIN.end() + 0.001;
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone = HalftoneMode::Dominant {
            n: *bornes::CANAUX.start() as usize,
            base_angle_deg: *bornes::ANGLE_TRAME.start(),
        };
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone = HalftoneMode::Dominant {
            n: *bornes::CANAUX.end() as usize,
            base_angle_deg: *bornes::ANGLE_TRAME.end(),
        };
    });
    accepte(|p| p.palette_size = Some(*bornes::PALETTE.start() as usize));
    accepte(|p| p.palette_size = Some(*bornes::PALETTE.end() as usize));
    accepte(|p| {
        p.dot_shape = DotShape::Ellipse {
            aspect: *bornes::RAPPORT.start(),
            angle_deg: 0.0,
        };
    });
    accepte(|p| {
        p.dot_shape = DotShape::Ellipse {
            aspect: *bornes::RAPPORT.end(),
            angle_deg: 0.0,
        };
    });
    accepte(|p| {
        p.dot_shape = DotShape::RegularPolygon {
            sides: *bornes::COTES.start() as u8,
        };
    });
    accepte(|p| {
        p.dot_shape = DotShape::RegularPolygon {
            sides: *bornes::COTES.end() as u8,
        };
    });
}

#[test]
fn juste_au_dela_de_chaque_borne_le_moteur_refuse() {
    refuse(|p| p.variance_sensitivity = *bornes::VARIANCE.end() + 0.1);
    refuse(|p| {
        p.algorithm = Algorithm::Grid;
        p.cols = *bornes::COLONNES.end() as u32 + 1;
    });
    refuse(|p| {
        p.algorithm = Algorithm::Quadtree;
        p.num_points = *bornes::POINTS.end() as usize + 1;
    });
    refuse(|p| p.iterations = *bornes::ITERATIONS.end() as usize + 1);
    refuse(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_max_dot_ratio = *bornes::RAYON_TRAME_MAX.end() + 0.5;
    });
    refuse(|p| p.palette_size = Some(*bornes::PALETTE.end() as usize + 1));
    refuse(|p| {
        p.dot_shape = DotShape::RegularPolygon {
            sides: *bornes::COTES.end() as u8 + 1,
        };
    });
}
