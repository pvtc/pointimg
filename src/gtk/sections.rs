//! Synchronisation état → widgets et visibilité/sensibilité des sections.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;

use pointimg::filter::{Algorithm, Dithering, DotShape, FilterParams, HalftoneMode, Screening};
use pointimg::frontend::{Quality, view_index};

use crate::State;
use crate::controls::Widgets;

pub(crate) fn algorithm_index(a: Algorithm) -> u32 {
    match a {
        Algorithm::Grid => 0,
        Algorithm::Kmeans => 1,
        Algorithm::Voronoi => 2,
        Algorithm::Quadtree => 3,
        Algorithm::Halftone => 4,
    }
}

pub(crate) fn algorithm_from_index(i: u32) -> Algorithm {
    match i {
        0 => Algorithm::Grid,
        1 => Algorithm::Kmeans,
        2 => Algorithm::Voronoi,
        3 => Algorithm::Quadtree,
        _ => Algorithm::Halftone,
    }
}

/// Position de l'algorithme de dithering dans la liste déroulante.
pub(crate) fn dithering_index(dithering: Dithering) -> u32 {
    match dithering {
        Dithering::Off => 0,
        Dithering::FloydSteinberg => 1,
        Dithering::Riemersma => 2,
        Dithering::Bayer4 => 3,
        Dithering::Bayer8 => 4,
    }
}

/// Algorithme correspondant à une position de liste déroulante.
pub(crate) fn dithering_from_index(index: u32) -> Dithering {
    match index {
        1 => Dithering::FloydSteinberg,
        2 => Dithering::Riemersma,
        3 => Dithering::Bayer4,
        4 => Dithering::Bayer8,
        _ => Dithering::Off,
    }
}

pub(crate) fn shape_index(s: &DotShape) -> u32 {
    match s {
        DotShape::Circle => 0,
        DotShape::Square => 1,
        DotShape::Ellipse { .. } => 2,
        DotShape::RegularPolygon { .. } => 3,
    }
}

/// Position du mode de qualité dans la liste déroulante.
pub(crate) fn quality_index(quality: Quality) -> u32 {
    match quality {
        Quality::Half => 0,
        Quality::Quarter => 1,
        Quality::Full => 2,
    }
}

/// Mode de qualité correspondant à une position de liste déroulante.
pub(crate) fn quality_from_index(index: u32) -> Quality {
    match index {
        0 => Quality::Half,
        1 => Quality::Quarter,
        _ => Quality::Full,
    }
}

/// Aide courte affichée en sous-titre du sélecteur d'algorithme.
fn algorithm_help(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Grid => "Une cellule = un point. Rapide et prévisible.",
        Algorithm::Kmeans => "Répartition par couleur et position.",
        Algorithm::Voronoi => "Lloyd pondéré : le rendu le plus « naturel ».",
        Algorithm::Quadtree => "Plus de points là où l'image est détaillée.",
        Algorithm::Halftone => "Rosette CMJN ou couleurs dominantes (offset print).",
    }
}

/// Met à jour les valeurs des widgets depuis `state.params` sans déclencher
/// les callbacks de modification.
pub(crate) fn sync_widgets(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let p = state.borrow().params.clone();
    state.borrow_mut().updating = true;

    widgets.algo.set_selected(algorithm_index(p.algorithm));
    widgets.algo.set_subtitle(algorithm_help(p.algorithm));
    widgets
        .view_mode
        .set_selected(view_index(state.borrow().view_mode));
    widgets
        .quality
        .set_selected(quality_index(state.borrow().quality));

    widgets
        .variance
        .adjustment()
        .set_value(f64::from(p.variance_sensitivity));
    widgets
        .variance_slider
        .set_value(f64::from(p.variance_sensitivity));
    widgets
        .min_radius
        .adjustment()
        .set_value(f64::from(p.min_radius_ratio));
    widgets
        .max_radius
        .adjustment()
        .set_value(f64::from(p.max_radius_ratio));
    // Chaque curseur marque l'autre borne : la plage se lit sans le titre.
    widgets
        .min_radius_slider
        .set_value(f64::from(p.min_radius_ratio));
    widgets
        .min_radius_slider
        .set_bound(f64::from(p.max_radius_ratio));
    widgets
        .max_radius_slider
        .set_value(f64::from(p.max_radius_ratio));
    widgets
        .max_radius_slider
        .set_bound(f64::from(p.min_radius_ratio));
    widgets
        .max_boost
        .adjustment()
        .set_value(f64::from(p.max_boost));
    widgets.max_boost_slider.set_value(f64::from(p.max_boost));
    widgets.cols.adjustment().set_value(f64::from(p.cols));
    widgets.cols_slider.set_value(f64::from(p.cols));
    widgets
        .grid_angle
        .adjustment()
        .set_value(f64::from(p.grid_angle_deg));
    widgets
        .num_points
        .adjustment()
        .set_value(p.num_points as f64);
    widgets.num_points_slider.set_value(p.num_points as f64);
    widgets
        .iterations
        .adjustment()
        .set_value(p.iterations as f64);
    widgets.iterations_slider.set_value(p.iterations as f64);

    widgets.shape.set_selected(shape_index(&p.dot_shape));
    match p.dot_shape {
        DotShape::Ellipse { aspect, angle_deg } => {
            widgets
                .ellipse_aspect
                .adjustment()
                .set_value(f64::from(aspect));
            widgets
                .ellipse_angle
                .adjustment()
                .set_value(f64::from(angle_deg));
        }
        DotShape::RegularPolygon { sides } => {
            widgets
                .polygon_sides
                .adjustment()
                .set_value(f64::from(sides));
        }
        _ => {}
    }

    widgets.use_palette.set_active(p.palette_size.is_some());
    if let Some(n) = p.palette_size {
        widgets.palette_size.adjustment().set_value(n as f64);
    }
    widgets.dithering.set_selected(dithering_index(p.dithering));
    widgets.use_seed.set_active(p.rng_seed.is_some());
    if let Some(seed) = p.rng_seed {
        widgets.seed.set_value(seed as f64);
    }

    widgets.bg_button.set_rgba(&gtk::gdk::RGBA::new(
        f32::from(p.bg_color[0]) / 255.0,
        f32::from(p.bg_color[1]) / 255.0,
        f32::from(p.bg_color[2]) / 255.0,
        1.0,
    ));
    widgets.transparent.set_active(p.transparent);
    widgets.gamma.set_active(p.gamma_correct);

    match &p.halftone {
        HalftoneMode::Dominant { .. } => widgets.halftone_mode.set_selected(1),
        _ => widgets.halftone_mode.set_selected(0),
    }
    if let HalftoneMode::Dominant { n, base_angle_deg } = &p.halftone {
        widgets.dominant_n.adjustment().set_value(*n as f64);
        widgets
            .dominant_angle
            .adjustment()
            .set_value(f64::from(*base_angle_deg));
    }
    widgets.screening.set_selected(match p.screening {
        Screening::Am => 0,
        Screening::Fm => 1,
    });
    widgets
        .halftone_freq
        .adjustment()
        .set_value(f64::from(p.halftone_frequency));
    widgets
        .halftone_freq_slider
        .set_value(f64::from(p.halftone_frequency));
    widgets
        .halftone_min_radius
        .adjustment()
        .set_value(f64::from(p.halftone_min_radius_ratio));
    widgets
        .halftone_max_dot
        .adjustment()
        .set_value(f64::from(p.halftone_max_dot_ratio));
    widgets
        .halftone_min_slider
        .set_value(f64::from(p.halftone_min_radius_ratio));
    widgets
        .halftone_min_slider
        .set_bound(f64::from(p.halftone_max_dot_ratio));
    widgets
        .halftone_max_slider
        .set_value(f64::from(p.halftone_max_dot_ratio));
    widgets
        .halftone_max_slider
        .set_bound(f64::from(p.halftone_min_radius_ratio));

    state.borrow_mut().updating = false;
}

/// Ajuste visibilité et sensibilité selon l'algorithme et la forme choisis.
pub(crate) fn update_sections(widgets: &Widgets, p: &FilterParams) {
    let is_halftone = p.algorithm == Algorithm::Halftone;

    widgets
        .grid_group
        .set_visible(p.algorithm == Algorithm::Grid);
    widgets.points_group.set_visible(matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree
    ));
    widgets.iterations.set_visible(matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi
    ));
    // Le halftone ignore tous les paramètres de placement : on masque le bloc
    // plutôt que de le griser, ce qui évite une lecture trompeuse.
    widgets.variance.set_visible(!is_halftone);
    widgets.min_radius.set_visible(!is_halftone);
    widgets.max_radius.set_visible(!is_halftone);
    widgets.max_boost.set_visible(!is_halftone);
    widgets.halftone_group.set_visible(is_halftone);
    // Les curseurs vivent dans des `ActionRow` à part : leur visibilité suit
    // celle du réglage qu'ils doublent, sans quoi une rangée vide subsisterait
    // sous un réglage masqué.
    let is_points = matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree
    );
    for (slider, visible) in [
        (&widgets.variance_slider, !is_halftone),
        (&widgets.min_radius_slider, !is_halftone),
        (&widgets.max_radius_slider, !is_halftone),
        (&widgets.max_boost_slider, !is_halftone),
        (&widgets.cols_slider, p.algorithm == Algorithm::Grid),
        (&widgets.num_points_slider, is_points),
        (
            &widgets.iterations_slider,
            matches!(p.algorithm, Algorithm::Kmeans | Algorithm::Voronoi),
        ),
    ] {
        slider.set_row_visible(visible);
    }

    widgets
        .ellipse_box
        .set_visible(matches!(p.dot_shape, DotShape::Ellipse { .. }));
    widgets
        .polygon_box
        .set_visible(matches!(p.dot_shape, DotShape::RegularPolygon { .. }));

    widgets.palette_size.set_sensitive(p.palette_size.is_some());
    widgets.dithering.set_sensitive(p.palette_size.is_some());
    widgets.seed.set_sensitive(p.rng_seed.is_some());
    widgets.vary_seed.set_sensitive(p.rng_seed.is_some());

    widgets
        .dominant_box
        .set_visible(matches!(p.halftone, HalftoneMode::Dominant { .. }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algorithm_index_round_trips() {
        for algorithm in [
            Algorithm::Grid,
            Algorithm::Kmeans,
            Algorithm::Voronoi,
            Algorithm::Quadtree,
            Algorithm::Halftone,
        ] {
            assert_eq!(algorithm_from_index(algorithm_index(algorithm)), algorithm);
        }
    }

    #[test]
    fn shape_index_maps_each_variant() {
        assert_eq!(shape_index(&DotShape::Circle), 0);
        assert_eq!(shape_index(&DotShape::Square), 1);
        assert_eq!(
            shape_index(&DotShape::Ellipse {
                aspect: 1.0,
                angle_deg: 0.0
            }),
            2
        );
        assert_eq!(shape_index(&DotShape::RegularPolygon { sides: 6 }), 3);
    }
}
