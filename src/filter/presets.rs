//! Built-in presets, ready to use.
//!
//! `.toml` files remain the exchange format (see
//! [`FilterParams::to_toml_string`]), but starting from scratch from a list of
//! settings is a first hurdle: one must already know which algorithm and which
//! values give a pleasing render. These presets are validated starting points,
//! exposed to the CLI as well as to the two interfaces.

use super::dither::Dithering;
use super::halftone::{HalftoneMode, Screening};
use super::params::{Algorithm, DotShape, FilterParams};

/// A built-in preset: a displayable name and the corresponding settings.
pub struct BuiltinPreset {
    /// Displayed name (in French, like the interface labels).
    pub name: &'static str,
    /// Short description displayed as a subtitle in the interfaces.
    pub description: &'static str,
    /// Settings applied when the preset is selected.
    pub params: FilterParams,
}

fn preset(name: &'static str, description: &'static str, params: FilterParams) -> BuiltinPreset {
    BuiltinPreset {
        name,
        description,
        params,
    }
}

/// List of the built-in presets, in display order.
///
/// ```
/// use pointimg::filter::{builtin_presets, validate_params};
///
/// for p in builtin_presets() {
///     // Every preset must pass parameter validation.
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

    /// The presets are also readable through the TOML path (round trip).
    #[test]
    fn presets_round_trip_through_toml() {
        for p in builtin_presets() {
            let toml = p.params.to_toml_string().unwrap();
            let back = FilterParams::from_toml_str(&toml).unwrap();
            assert_eq!(back, p.params, "preset « {} »", p.name);
        }
    }
}
