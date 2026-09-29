//! Presets intégrés, prêts à l'emploi.
//!
//! Les fichiers `.toml` restent le format d'échange (voir
//! [`FilterParams::to_toml_string`]), mais partir de zéro d'une liste de
//! réglages est un premier obstacle : il faut déjà savoir quel algorithme et
//! quelles valeurs donnent un rendu agréable. Ces presets sont des points de
//! départ validés, exposés à la CLI comme aux deux interfaces.

use super::dither::Dithering;
use super::halftone::{HalftoneMode, Screening};
use super::params::{Algorithm, DotShape, FilterParams};

/// Un preset intégré : un nom affichable et les réglages correspondants.
pub struct BuiltinPreset {
    /// Nom affiché (français, comme les libellés de l'interface).
    pub name: &'static str,
    /// Description courte affichée en sous-titre dans les interfaces.
    pub description: &'static str,
    /// Réglages appliqués quand le preset est sélectionné.
    pub params: FilterParams,
}

fn preset(name: &'static str, description: &'static str, params: FilterParams) -> BuiltinPreset {
    BuiltinPreset {
        name,
        description,
        params,
    }
}

/// Liste des presets intégrés, dans l'ordre d'affichage.
///
/// ```
/// use pointimg::filter::{builtin_presets, validate_params};
///
/// for p in builtin_presets() {
///     // Tous les presets doivent passer la validation des paramètres.
///     validate_params(64, 64, &p.params).unwrap();
/// }
/// assert!(!builtin_presets().is_empty());
/// ```
pub fn builtin_presets() -> Vec<BuiltinPreset> {
    vec![
        preset(
            "Pointillisme classique",
            "Stippling de Lloyd, 1 500 points — le rendu de référence",
            FilterParams {
                algorithm: Algorithm::Voronoi,
                num_points: 1500,
                iterations: 12,
                variance_sensitivity: 0.7,
                max_boost: 2.5,
                ..FilterParams::default()
            },
        ),
        preset(
            "Stippling fin",
            "Beaucoup de petits points : textures, portraits, détails",
            FilterParams {
                algorithm: Algorithm::Voronoi,
                num_points: 6000,
                iterations: 16,
                min_radius_ratio: 0.0015,
                max_radius_ratio: 0.03,
                variance_sensitivity: 0.85,
                ..FilterParams::default()
            },
        ),
        preset(
            "Aplats graphiques",
            "Palette réduite et dithering : effet sérigraphie / offset",
            FilterParams {
                algorithm: Algorithm::Quadtree,
                num_points: 2500,
                min_radius_ratio: 0.006,
                max_radius_ratio: 0.09,
                palette_size: Some(8),
                dithering: Dithering::FloydSteinberg,
                ..FilterParams::default()
            },
        ),
        preset(
            "Rosette CMJN",
            "Trame AM classique 15°/75°/0°/45°",
            FilterParams {
                algorithm: Algorithm::Halftone,
                halftone: HalftoneMode::Cmyk {
                    angles: [15.0, 75.0, 0.0, 45.0],
                },
                screening: Screening::Am,
                halftone_frequency: 60.0,
                halftone_max_dot_ratio: 1.1,
                ..FilterParams::default()
            },
        ),
        preset(
            "Trame FM (blue noise)",
            "Trame stochastique : c'est la densité qui code l'encre",
            FilterParams {
                algorithm: Algorithm::Halftone,
                halftone: HalftoneMode::Cmyk {
                    angles: [15.0, 75.0, 0.0, 45.0],
                },
                screening: Screening::Fm,
                halftone_frequency: 90.0,
                ..FilterParams::default()
            },
        ),
        preset(
            "Moiré",
            "Grille pivotée : le motif est le rendu, pas la photo",
            FilterParams {
                algorithm: Algorithm::Grid,
                cols: 160,
                grid_angle_deg: 45.0,
                min_radius_ratio: 0.004,
                max_radius_ratio: 0.05,
                max_boost: 1.0,
                variance_sensitivity: 0.0,
                ..FilterParams::default()
            },
        ),
        preset(
            "Mosaïque",
            "Grille serrée de carrés, contraste poussé",
            FilterParams {
                algorithm: Algorithm::Grid,
                cols: 60,
                dot_shape: DotShape::Square,
                min_radius_ratio: 0.008,
                max_radius_ratio: 0.12,
                variance_sensitivity: 0.3,
                ..FilterParams::default()
            },
        ),
        preset(
            "Pastilles",
            "Quadtree contrasté, points regularisés en hexagones",
            FilterParams {
                algorithm: Algorithm::Quadtree,
                num_points: 1800,
                dot_shape: DotShape::RegularPolygon { sides: 6 },
                min_radius_ratio: 0.005,
                max_radius_ratio: 0.07,
                variance_sensitivity: 0.9,
                gamma_correct: true,
                ..FilterParams::default()
            },
        ),
        preset(
            "Calques transparents",
            "Rendu RGBA sur fond transparent, pour superposition",
            FilterParams {
                algorithm: Algorithm::Voronoi,
                num_points: 2000,
                transparent: true,
                gamma_correct: true,
                ..FilterParams::default()
            },
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::validate_params;

    #[test]
    fn all_presets_are_valid() {
        for p in builtin_presets() {
            validate_params(512, 512, &p.params)
                .unwrap_or_else(|e| panic!("preset « {} » : {e}", p.name));
        }
    }

    #[test]
    fn preset_names_are_unique() {
        let mut names: Vec<&str> = builtin_presets().iter().map(|p| p.name).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "noms de presets dupliqués");
    }

    #[test]
    fn halftone_presets_activate_a_mode() {
        for p in builtin_presets() {
            if p.params.algorithm == Algorithm::Halftone {
                assert_ne!(
                    p.params.halftone,
                    HalftoneMode::Off,
                    "preset halftone sans mode : {}",
                    p.name
                );
            }
        }
    }

    /// Les presets sont aussi lisibles par le chemin TOML (aller-retour).
    #[test]
    fn presets_round_trip_through_toml() {
        for p in builtin_presets() {
            let toml = p.params.to_toml_string().unwrap();
            let back = FilterParams::from_toml_str(&toml).unwrap();
            assert_eq!(back, p.params, "preset « {} »", p.name);
        }
    }
}
