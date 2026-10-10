//! Cohérence entre les plages de l'interface GTK et celles du moteur.
//!
//! Équivalent de `src/gui/ranges.rs` pour le front-end GTK4. Les deux listes
//! restent distinctes et indépendantes : les bornes sont une conséquence de ce
//! que le moteur accepte, pas un contrat d'interface, et chacune peut mentira
//! sur l'échelle (linéaire ou logarithmique) sans que l'autre s'en aperçoive.
//!
//! Ce que le test garantit, c'est qu'aucune des deux ne propose une valeur que
//! `filter::validate_params` refuserait.

use std::ops::RangeInclusive;

use pointimg::filter::{Algorithm, DotShape, FilterParams, HalftoneMode, validate_params};

/// Bornes des `adw::SpinRow` de `controls.rs`.
///
/// Les `ValueSlider` qui doublent ces rangées reprennent les mêmes bornes via
/// `Curve` : changer l'un sans l'autre se voit ici, puisque la borne testée
/// est celle du `SpinRow` et celle du curseur en découle.
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
    pub const RAYON_TRAME: RangeInclusive<f32> = 0.0..=10.0;
    pub const ANGLE_TRAME: RangeInclusive<f32> = 0.0..=180.0;
    pub const CANAUX: RangeInclusive<f32> = 2.0..=32.0;
    pub const PALETTE: RangeInclusive<f32> = 2.0..=256.0;
    pub const RAPPORT: RangeInclusive<f32> = 0.1..=10.0;
    pub const COTES: RangeInclusive<f32> = 3.0..=12.0;
}

/// Réglages par défaut, plus un mode d'encre valide : `HalftoneMode::Off`
/// combiné au reste des valeurs par défaut est refusé par le moteur.
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
    validate_params(64, 64, &params).unwrap_or_else(|e| panic!("réglages refusés : {e}"));
}

fn refuse(applique: impl Fn(&mut FilterParams)) {
    let mut params = base();
    applique(&mut params);
    assert!(
        validate_params(64, 64, &params).is_err(),
        "le moteur accepte une valeur que l'interface ne devrait pas proposer"
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
        p.halftone_min_radius_ratio = *bornes::RAYON_TRAME.start();
    });
    accepte(|p| {
        p.algorithm = Algorithm::Halftone;
        p.halftone_max_dot_ratio = *bornes::RAYON_TRAME.end();
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
        p.halftone_max_dot_ratio = *bornes::RAYON_TRAME.end() + 0.5;
    });
    refuse(|p| p.palette_size = Some(*bornes::PALETTE.end() as usize + 1));
    refuse(|p| {
        p.dot_shape = DotShape::RegularPolygon {
            sides: *bornes::COTES.end() as u8 + 1,
        };
    });
}
