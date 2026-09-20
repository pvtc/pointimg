//! Synchronisation état → widgets et visibilité/sensibilité des sections.

use std::cell::RefCell;
use std::rc::Rc;

use gtk::prelude::*;

use pointimg::filter::{Algorithm, DotShape, FilterParams, HalftoneMode, Screening};

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

pub(crate) fn shape_index(s: &DotShape) -> u32 {
    match s {
        DotShape::Circle => 0,
        DotShape::Square => 1,
        DotShape::Ellipse { .. } => 2,
        DotShape::RegularPolygon { .. } => 3,
    }
}

pub(crate) fn view_from_index(i: u32) -> crate::ViewMode {
    match i {
        1 => crate::ViewMode::ResultOnly,
        2 => crate::ViewMode::SourceOnly,
        3 => crate::ViewMode::DensityMap,
        _ => crate::ViewMode::Side,
    }
}

pub(crate) fn view_index(v: crate::ViewMode) -> u32 {
    match v {
        crate::ViewMode::Side => 0,
        crate::ViewMode::ResultOnly => 1,
        crate::ViewMode::SourceOnly => 2,
        crate::ViewMode::DensityMap => 3,
    }
}

/// Met à jour les valeurs des widgets depuis `state.params` sans déclencher
/// les callbacks de modification.
pub(crate) fn sync_widgets(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let p = state.borrow().params.clone();
    state.borrow_mut().updating = true;

    widgets.algo.set_selected(algorithm_index(p.algorithm));
    widgets
        .view_mode
        .set_selected(view_index(state.borrow().view_mode));
    widgets.zoom.set_value(state.borrow().zoom);
    widgets.zoom_fit.set_active(state.borrow().zoom_fit);

    widgets.variance.set_value(p.variance_sensitivity as f64);
    widgets.min_radius.set_value(p.min_radius_ratio as f64);
    widgets.max_radius.set_value(p.max_radius_ratio as f64);
    widgets.max_boost.set_value(p.max_boost as f64);
    widgets.cols.set_value(p.cols as f64);
    widgets.grid_angle.set_value(p.grid_angle_deg as f64);
    widgets.num_points.set_value(p.num_points as f64);
    widgets.iterations.set_value(p.iterations as f64);

    widgets.shape.set_selected(shape_index(&p.dot_shape));
    match p.dot_shape {
        DotShape::Ellipse { aspect, angle_deg } => {
            widgets.ellipse_aspect.set_value(aspect as f64);
            widgets.ellipse_angle.set_value(angle_deg as f64);
        }
        DotShape::RegularPolygon { sides } => {
            widgets.polygon_sides.set_value(sides as f64);
        }
        _ => {}
    }

    widgets.use_palette.set_active(p.palette_size.is_some());
    if let Some(n) = p.palette_size {
        widgets.palette_size.set_value(n as f64);
    }
    widgets.dithering.set_active(p.dithering);
    widgets.use_seed.set_active(p.rng_seed.is_some());
    if let Some(seed) = p.rng_seed {
        widgets.seed.set_value(seed as f64);
    }

    widgets.bg_button.set_rgba(&gtk::gdk::RGBA::new(
        p.bg_color[0] as f32 / 255.0,
        p.bg_color[1] as f32 / 255.0,
        p.bg_color[2] as f32 / 255.0,
        1.0,
    ));
    widgets.transparent.set_active(p.transparent);
    widgets.gamma.set_active(p.gamma_correct);

    match &p.halftone {
        HalftoneMode::Off => {
            widgets.halftone_mode.set_selected(0);
        }
        HalftoneMode::Cmyk { .. } => {
            widgets.halftone_mode.set_selected(0);
        }
        HalftoneMode::Dominant { n, base_angle_deg } => {
            widgets.halftone_mode.set_selected(1);
            widgets.dominant_n.set_value(*n as f64);
            widgets.dominant_angle.set_value(*base_angle_deg as f64);
        }
    }
    widgets.screening.set_selected(match p.screening {
        Screening::Am => 0,
        Screening::Fm => 1,
    });
    widgets.halftone_freq.set_value(p.halftone_frequency as f64);
    widgets
        .halftone_min_radius
        .set_value(p.halftone_min_radius_ratio as f64);
    widgets
        .halftone_max_dot
        .set_value(p.halftone_max_dot_ratio as f64);

    state.borrow_mut().updating = false;
}

/// Ajuste visibilité et sensibilité selon l'algorithme et la forme choisis.
pub(crate) fn update_sections(widgets: &Widgets, p: &FilterParams) {
    let is_halftone = p.algorithm == Algorithm::Halftone;

    widgets.grid_box.set_visible(p.algorithm == Algorithm::Grid);
    widgets.points_box.set_visible(matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree
    ));
    widgets.iterations_row.set_visible(matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi
    ));
    widgets.placement_box.set_sensitive(!is_halftone);
    widgets.halftone_group.set_visible(is_halftone);

    widgets
        .ellipse_box
        .set_visible(matches!(p.dot_shape, DotShape::Ellipse { .. }));
    widgets
        .polygon_box
        .set_visible(matches!(p.dot_shape, DotShape::RegularPolygon { .. }));

    widgets.palette_size.set_sensitive(p.palette_size.is_some());
    widgets.dithering.set_sensitive(p.palette_size.is_some());
    widgets.seed.set_sensitive(p.rng_seed.is_some());

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
    fn view_index_round_trips() {
        for view in [
            crate::ViewMode::Side,
            crate::ViewMode::ResultOnly,
            crate::ViewMode::SourceOnly,
            crate::ViewMode::DensityMap,
        ] {
            assert_eq!(view_from_index(view_index(view)), view);
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
