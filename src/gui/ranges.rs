//! Cohérence entre les plages de l'interface et celles du moteur.
//!
//! Chaque interface déclare ses propres bornes, et ce module vérifie qu'elles
//! restent acceptables par `filter::validate_params`. Rien n'est partagé entre
//! les deux fronts au-delà de cette vérification : les plages ne sont pas un
//! contrat d'interface, mais une conséquence de ce que le moteur accepte.
//!
//! Une plage qui glisse hors des bornes ne casse pas l'interface : le champ
//! devient insaisissable, l'aperçu se fige, et l'utilisateur découvre le refus
//! au pire moment. D'où le test.

use std::ops::RangeInclusive;

use pointimg::filter::{Algorithm, DotShape, FilterParams, HalftoneMode, validate_params};

/// Bornes déclarées par les curseurs du panneau de réglages.
///
/// Volontairement regroupées ici plutôt qu'éparpillées dans les appels de
/// widgets : c'est ce qui rend la liste vérifiable, tout en restant une
/// déclaration propre à l'interface egui.
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
    /// Bornes basse et haute du rayon de trame, séparément.
    ///
    /// Le moteur plafonne `halftone_min_radius_ratio` à 1,0 quand
    /// `halftone_max_dot_ratio` monte à 10 (`validate_params`). Une
    /// constante partagée laissait le curseur de la borne basse franchir
    /// ce plafond : l'utilisateur obtenait une erreur sèche du worker,
    /// et les tests de plage ne le voyaient pas puisqu'ils n'exercaient
    /// que la borne basse de la borne basse et la borne haute de la
    /// borne haute.
    pub const RAYON_TRAME_MIN: RangeInclusive<f32> = 0.0..=1.0;
    pub const RAYON_TRAME_MAX: RangeInclusive<f32> = 0.05..=10.0;
    pub const PALETTE: RangeInclusive<f32> = 2.0..=256.0;
    pub const RAPPORT: RangeInclusive<f32> = 0.1..=10.0;
    pub const COTES: RangeInclusive<f32> = 3.0..=12.0;
}

/// Réglages par défaut, plus un mode d'encre valide.
///
/// `FilterParams::default()` combine `Algorithm::Grid` et
/// `HalftoneMode::Off` : passer le curseur sur « Halftone (rosette) » sans
/// choisir de mode donne un réglage que le moteur refuse. Les tests de plage
/// du halftone doivent donc partir d'une base valide, comme le fait l'interface
/// (qui force CMJN à la sélection).
fn base() -> FilterParams {
    FilterParams {
        halftone: HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0],
        },
        ..FilterParams::default()
    }
}

/// Vérifie que des réglages sont acceptés par le moteur.
///
/// Les deux bornes de chaque curseur sont testées : c'est la borne haute qui
/// glisse le plus souvent (un rayon agrandi, une trame accélérée).
fn accepte(applique: impl Fn(&mut FilterParams)) {
    let mut params = base();
    applique(&mut params);
    validate_params(64, 64, &params).unwrap_or_else(|e| panic!("réglages refusés : {e}"));
}

/// Vérifie qu'une valeur *hors* plage est refusée par le moteur.
///
/// Le complément de [`accepte`] : une borne devenue trop large passerait
/// inaperçue si seule la borne basse est testée.
fn refuse(applique: impl Fn(&mut FilterParams)) {
    let mut params = base();
    applique(&mut params);
    assert!(
        validate_params(64, 64, &params).is_err(),
        "le moteur accepte une valeur que l'interface ne devrait pas proposer"
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
    // Le sommet de la plage de la borne *basse* : c'est le cas que le
    // partage d'une constante unique laissait passer. Tant qu'il n'est pas
    // exercé, un curseur partagé peut franchir 1,0 sans qu'aucun test rougeoie.
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
    // Plus bas que le plafond générique : `validate_params` refuse 50 001
    // points pour ces deux algorithmes, alors que Quadtree en accepte 100 000.
    // Le curseur « Nombre de points » est commun aux trois — c'est donc la
    // validation qui doit rester la source de vérité, pas la borne du curseur.
    let mut params = base();
    params.algorithm = Algorithm::Voronoi;
    params.num_points = 50_001;
    assert!(
        validate_params(64, 64, &params).is_err(),
        "Voronoi doit rester plafonné à 50 000 points"
    );
    params.algorithm = Algorithm::Quadtree;
    validate_params(64, 64, &params).expect("Quadtree accepte 50 001");
}

#[test]
fn le_halftone_exige_un_mode_d_encrage() {
    let mut params = base();
    params.algorithm = Algorithm::Halftone;
    params.halftone = HalftoneMode::Off;
    assert!(
        validate_params(64, 64, &params).is_err(),
        "Halftone sans mode doit être refusé"
    );
    params.halftone = HalftoneMode::Cmyk {
        angles: [15.0, 75.0, 0.0, 45.0],
    };
    validate_params(64, 64, &params).expect("CMJN est valide");
}
