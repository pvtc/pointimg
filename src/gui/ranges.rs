//! Consistency between the ranges of the interface and those of the engine.
//!
//! Each interface declares its own bounds, and this module checks that they
//! stay acceptable to `filter::validate_params`. Nothing is shared between the
//! two frontends beyond that check: the ranges are not an interface contract,
//! but a consequence of what the engine accepts.
//!
//! A range that drifts out of the bounds does not break the interface: the field
//! becomes impossible to type into, the preview freezes, and the user discovers
//! the refusal at the worst moment. Hence the test.

use std::ops::RangeInclusive;

use pointimg::filter::{Algorithm, DotShape, FilterParams, HalftoneMode, validate_params};

/// Bounds declared by the control panel sliders.
///
/// Deliberately gathered here rather than scattered across widget calls: that
/// is what makes the list checkable, while staying a declaration specific to
/// the egui interface.
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
    /// `halftone_max_dot_ratio` goes up to 10 (`validate_params`). A shared
    /// constant let the low end's slider cross that ceiling: the user got a
    /// dry error from the worker, and the range tests did not see it since
    /// they only exercised the low end of the low end and the high end of the
    /// high end.
    pub const RAYON_TRAME_MIN: RangeInclusive<f32> = 0.0..=1.0;
    pub const RAYON_TRAME_MAX: RangeInclusive<f32> = 0.05..=10.0;
    pub const PALETTE: RangeInclusive<f32> = 2.0..=256.0;
    pub const RAPPORT: RangeInclusive<f32> = 0.1..=10.0;
    pub const COTES: RangeInclusive<f32> = 3.0..=12.0;
}

/// Default settings, plus a valid ink mode.
///
/// `FilterParams::default()` combines `Algorithm::Grid` and
/// `HalftoneMode::Off`: moving the slider to "Halftone (rosette)" without
/// choosing a mode gives a setting the engine refuses. The halftone range
/// tests must therefore start from a valid base, as the interface does
/// (which forces CMYK on selection).
fn base() -> FilterParams {
    FilterParams {
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        ..FilterParams::default()
    }
}

/// Checks that settings are accepted by the engine.
///
/// Both ends of each slider are tested: it is the high end that drifts the
/// most often (an enlarged radius, a faster screen).
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

/// Checks that a value *outside* the range is refused by the engine.
///
/// The complement of [`accepte`]: a bound that became too wide would go
/// unnoticed if only the low end were tested.
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
fn les_deux_bornes_de_chaque_curseur_sont_acceptees() {
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
    // The top of the *low* end's range: this is the case a single shared
    // constant let through. As long as it is not exercised, a shared slider
    // can cross 1.0 without any test turning red.
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
    refuse(|p| p.variance_sensitivity = *bornes::VARIANCE.start() - 0.1);
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

#[test]
fn voronoi_et_kmeans_ont_leur_propre_plafond_de_points() {
    // Lower than the generic ceiling: `validate_params` refuses 50 001
    // points for these two algorithms, while Quadtree accepts 100 000.
    // The "Number of points" slider is common to all three — it is therefore
    // the validation that must remain the source of truth, not the slider bound.
    let mut params = base();
    params.algorithm = Algorithm::Voronoi;
    params.num_points = 50_001;
    assert!(
        validate_params(64, 64, &params).is_err(),
        "{}",
        pointimg::frontend::t("Voronoi doit rester plafonné à 50 000 points")
    );
    params.algorithm = Algorithm::Quadtree;
    let msg = pointimg::frontend::t("Quadtree accepte 50 001");
    validate_params(64, 64, &params).expect(msg);
}

#[test]
fn le_halftone_exige_un_mode_d_encrage() {
    let mut params = base();
    params.algorithm = Algorithm::Halftone;
    params.halftone = HalftoneMode::Off;
    assert!(
        validate_params(64, 64, &params).is_err(),
        "{}",
        pointimg::frontend::t("Halftone sans mode doit être refusé")
    );
    params.halftone = HalftoneMode::Cmyk {
        angles: [15.0, 75.0, 0.0, 45.0],
    };
    let msg = pointimg::frontend::t("CMJN est valide");
    validate_params(64, 64, &params).expect(msg);
}
