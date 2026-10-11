//! Synchronisation état → widgets et visibilité/sensibilité des sections.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;

use pointimg::filter::{Algorithm, Dithering, DotShape, FilterParams, HalftoneMode, Screening};
use pointimg::frontend::{self, Quality, view_index};

use crate::State;
use crate::controls::Widgets;

/// Position de l'algorithme dans la liste déroulante.
///
/// La table vit dans `frontend`, partagée avec egui. Le repli est le *premier*
/// algorithme, donc cohérent avec la table : une liste décalée retombe sur
/// « Grille » plutôt que sur `Halftone` — qui, de plus, exige un mode d'encre et
/// faisait échouer le rendu suivant.
pub(crate) fn algorithm_index(a: Algorithm) -> u32 {
    frontend::algorithm_index(a).unwrap_or(0)
}

/// Algorithme correspondant à une position. Voir [`algorithm_index`].
pub(crate) fn algorithm_from_index(i: u32) -> Algorithm {
    frontend::algorithm_of(i).unwrap_or(frontend::ALGORITHMS[0])
}

/// Sous-titre du sélecteur de dithering quand aucun algorithme n'est signalé.
pub(crate) const DITHERING_SUBTITLE: &str = "Tramage de la palette quantifiée";

/// Avertissement de poids de l'algorithme de dithering choisi.
///
/// Le sous-titre porte la version courte, visible sans survol comme la ligne
/// d'avertissement d'egui ; l'infobulle porte la version longue. Les textes
/// viennent de `Dithering::size_caveat`, donc les deux interfaces disent la
/// même chose.
///
/// Appelé à la fois par `sync_widgets` (undo, préréglages) et par le gestionnaire
/// du sélecteur : `sync_widgets` n'est pas rejoué après un changement proposé
/// par l'utilisateur, puisqu'il sert à resynchroniser les widgets *après* une
/// modification programmatique.
pub(crate) fn sync_dithering_note(dithering: Dithering, widgets: &Widgets) {
    widgets
        .dots
        .dithering
        .set_subtitle(dithering.size_caveat_short().unwrap_or(DITHERING_SUBTITLE));
    widgets
        .dots
        .dithering
        .set_tooltip_text(dithering.size_caveat());
}

/// Position du tramage dans la liste déroulante. Table partagée avec egui.
pub(crate) fn dithering_index(dithering: Dithering) -> u32 {
    frontend::dithering_index(dithering).unwrap_or(0)
}

/// Tramage correspondant à une position. Le repli est le premier de la table
/// (« Aucun »), pas un `_ => Dithering::Off` qui apparaîtrait correct alors
/// qu'une liste décalée vient de choisir à sa place.
pub(crate) fn dithering_from_index(index: u32) -> Dithering {
    frontend::dithering_of(index).unwrap_or(frontend::DITHERINGS[0])
}

pub(crate) fn shape_index(s: &DotShape) -> u32 {
    match s {
        DotShape::Circle => 0,
        DotShape::Square => 1,
        DotShape::Ellipse { .. } => 2,
        DotShape::RegularPolygon { .. } => 3,
    }
}

/// Position du mode de qualité dans la liste déroulante. Table partagée avec egui.
pub(crate) fn quality_index(quality: Quality) -> u32 {
    frontend::quality_index(quality).unwrap_or(0)
}

/// Mode de qualité correspondant à une position. Le repli est le premier de la
/// table, pas `Full` : index inconnue ne veut pas dire « meilleure qualité ».
pub(crate) fn quality_from_index(index: u32) -> Quality {
    frontend::quality_of(index).unwrap_or(frontend::QUALITIES[0])
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

    widgets
        .algorithm
        .algo
        .set_selected(algorithm_index(p.algorithm));
    widgets
        .algorithm
        .algo
        .set_subtitle(algorithm_help(p.algorithm));
    widgets
        .preview
        .view_mode
        .set_selected(view_index(state.borrow().view_mode));
    widgets
        .compute
        .quality
        .set_selected(quality_index(state.borrow().quality));

    widgets
        .algorithm
        .variance
        .adjustment()
        .set_value(f64::from(p.variance_sensitivity));
    widgets
        .algorithm
        .variance_slider
        .set_value(f64::from(p.variance_sensitivity));
    widgets
        .algorithm
        .min_radius
        .adjustment()
        .set_value(f64::from(p.min_radius_ratio));
    widgets
        .algorithm
        .max_radius
        .adjustment()
        .set_value(f64::from(p.max_radius_ratio));
    // Chaque curseur marque l'autre borne : la plage se lit sans le titre.
    widgets
        .algorithm
        .min_radius_slider
        .set_value(f64::from(p.min_radius_ratio));
    widgets
        .algorithm
        .min_radius_slider
        .set_bound(f64::from(p.max_radius_ratio));
    widgets
        .algorithm
        .max_radius_slider
        .set_value(f64::from(p.max_radius_ratio));
    widgets
        .algorithm
        .max_radius_slider
        .set_bound(f64::from(p.min_radius_ratio));
    widgets
        .algorithm
        .max_boost
        .adjustment()
        .set_value(f64::from(p.max_boost));
    widgets
        .algorithm
        .max_boost_slider
        .set_value(f64::from(p.max_boost));
    widgets
        .algorithm
        .cols
        .adjustment()
        .set_value(f64::from(p.cols));
    widgets.algorithm.cols_slider.set_value(f64::from(p.cols));
    widgets
        .algorithm
        .grid_angle
        .adjustment()
        .set_value(f64::from(p.grid_angle_deg));
    widgets
        .algorithm
        .num_points
        .adjustment()
        .set_value(p.num_points as f64);
    widgets
        .algorithm
        .num_points_slider
        .set_value(p.num_points as f64);
    widgets
        .algorithm
        .iterations
        .adjustment()
        .set_value(p.iterations as f64);
    widgets
        .algorithm
        .iterations_slider
        .set_value(p.iterations as f64);

    widgets.dots.shape.set_selected(shape_index(&p.dot_shape));
    match p.dot_shape {
        DotShape::Ellipse { aspect, angle_deg } => {
            widgets
                .dots
                .ellipse_aspect
                .adjustment()
                .set_value(f64::from(aspect));
            widgets
                .dots
                .ellipse_angle
                .adjustment()
                .set_value(f64::from(angle_deg));
        }
        DotShape::RegularPolygon { sides } => {
            widgets
                .dots
                .polygon_sides
                .adjustment()
                .set_value(f64::from(sides));
        }
        _ => {}
    }

    widgets
        .dots
        .use_palette
        .set_active(p.palette_size.is_some());
    if let Some(n) = p.palette_size {
        widgets.dots.palette_size.adjustment().set_value(n as f64);
    }
    widgets
        .dots
        .dithering
        .set_selected(dithering_index(p.dithering));
    // Avertissement de poids, même texte que dans l'interface egui (source
    // commune : `Dithering::size_caveat`), pour qu'il ne diverge pas.
    widgets
        .dots
        .dithering
        .set_tooltip_text(p.dithering.size_caveat());
    widgets.render.use_seed.set_active(p.rng_seed.is_some());
    if let Some(seed) = p.rng_seed {
        widgets.render.seed.set_value(seed as f64);
    }

    widgets.render.bg_button.set_rgba(&gtk::gdk::RGBA::new(
        f32::from(p.bg_color[0]) / 255.0,
        f32::from(p.bg_color[1]) / 255.0,
        f32::from(p.bg_color[2]) / 255.0,
        1.0,
    ));
    widgets.render.transparent.set_active(p.transparent);
    widgets.render.gamma.set_active(p.gamma_correct);

    match &p.halftone {
        HalftoneMode::Dominant { .. } => widgets.halftone.halftone_mode.set_selected(1),
        _ => widgets.halftone.halftone_mode.set_selected(0),
    }
    if let HalftoneMode::Dominant { n, base_angle_deg } = &p.halftone {
        widgets
            .halftone
            .dominant_n
            .adjustment()
            .set_value(*n as f64);
        widgets
            .halftone
            .dominant_angle
            .adjustment()
            .set_value(f64::from(*base_angle_deg));
    }
    widgets.halftone.screening.set_selected(match p.screening {
        Screening::Am => 0,
        Screening::Fm => 1,
    });
    widgets
        .halftone
        .halftone_freq
        .adjustment()
        .set_value(f64::from(p.halftone_frequency));
    widgets
        .halftone
        .halftone_freq_slider
        .set_value(f64::from(p.halftone_frequency));
    widgets
        .halftone
        .halftone_min_radius
        .adjustment()
        .set_value(f64::from(p.halftone_min_radius_ratio));
    widgets
        .halftone
        .halftone_max_dot
        .adjustment()
        .set_value(f64::from(p.halftone_max_dot_ratio));
    widgets
        .halftone
        .halftone_min_slider
        .set_value(f64::from(p.halftone_min_radius_ratio));
    widgets
        .halftone
        .halftone_min_slider
        .set_bound(f64::from(p.halftone_max_dot_ratio));
    widgets
        .halftone
        .halftone_max_slider
        .set_value(f64::from(p.halftone_max_dot_ratio));
    widgets
        .halftone
        .halftone_max_slider
        .set_bound(f64::from(p.halftone_min_radius_ratio));

    state.borrow_mut().updating = false;
}

/// Ajuste visibilité et sensibilité selon l'algorithme et la forme choisis.
pub(crate) fn update_sections(widgets: &Widgets, p: &FilterParams) {
    let is_halftone = p.algorithm == Algorithm::Halftone;

    widgets
        .algorithm
        .grid_group
        .set_visible(p.algorithm == Algorithm::Grid);
    widgets.algorithm.points_group.set_visible(matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree
    ));
    widgets.algorithm.iterations.set_visible(matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi
    ));
    // Le halftone ignore tous les paramètres de placement : on masque le bloc
    // plutôt que de le griser, ce qui évite une lecture trompeuse.
    widgets.algorithm.variance.set_visible(!is_halftone);
    widgets.algorithm.min_radius.set_visible(!is_halftone);
    widgets.algorithm.max_radius.set_visible(!is_halftone);
    widgets.algorithm.max_boost.set_visible(!is_halftone);
    widgets.halftone.halftone_group.set_visible(is_halftone);

    // Réglages(conditionnels : masqués tant que la fonction qui les utilise n'est
    // pas active. Grisés, ils se lisaient comme des réglages disponibles — et
    // l'interface egui les masquait déjà (cf. `show_shape_and_palette_section`),
    // ce qui faisait diverger les deux rendus pour rien.
    let palette = p.palette_size.is_some();
    widgets.dots.palette_size.set_visible(palette);
    widgets.dots.dithering.set_visible(palette);
    widgets.render.seed_row.set_visible(p.rng_seed.is_some());
    // Les curseurs vivent dans des `ActionRow` à part : leur visibilité suit
    // celle du réglage qu'ils doublent, sans quoi une rangée vide subsisterait
    // sous un réglage masqué.
    let is_points = matches!(
        p.algorithm,
        Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree
    );
    for (slider, visible) in [
        (&widgets.algorithm.variance_slider, !is_halftone),
        (&widgets.algorithm.min_radius_slider, !is_halftone),
        (&widgets.algorithm.max_radius_slider, !is_halftone),
        (&widgets.algorithm.max_boost_slider, !is_halftone),
        (
            &widgets.algorithm.cols_slider,
            p.algorithm == Algorithm::Grid,
        ),
        (&widgets.algorithm.num_points_slider, is_points),
        (
            &widgets.algorithm.iterations_slider,
            matches!(p.algorithm, Algorithm::Kmeans | Algorithm::Voronoi),
        ),
    ] {
        slider.set_row_visible(visible);
    }

    widgets
        .dots
        .ellipse_box
        .set_visible(matches!(p.dot_shape, DotShape::Ellipse { .. }));
    widgets
        .dots
        .polygon_box
        .set_visible(matches!(p.dot_shape, DotShape::RegularPolygon { .. }));

    widgets
        .halftone
        .dominant_box
        .set_visible(matches!(p.halftone, HalftoneMode::Dominant { .. }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointimg::frontend::{VIEW_COUNT, ViewMode};

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

    /// `step_view` est piloté par `Tab` et `Maj+Tab` (voir `wire.rs`). Le
    /// calcul est ici refait sur `ViewMode` : un aller-retour doit revenir au
    /// point de départ, et les deux sens doivent boucler.
    #[test]
    fn le_cycle_des_apercus_boucle_dans_les_deux_sens() {
        let modes: Vec<ViewMode> = (0..VIEW_COUNT)
            .map(pointimg::frontend::view_from_index)
            .collect();
        let pas = |index: u32, delta: i32| -> u32 {
            (index as i32 + delta).rem_euclid(VIEW_COUNT as i32) as u32
        };
        for (index, mode) in modes.iter().enumerate() {
            let index = index as u32;
            assert_eq!(pas(pas(index, 1), -1), index, "aller-retour instable");
            assert_eq!(*mode, pointimg::frontend::view_from_index(index));
        }
        // Le dernier reboucle sur le premier, et l'inverse.
        assert_eq!(pas(0, -1), VIEW_COUNT - 1);
        assert_eq!(pas(VIEW_COUNT - 1, 1), 0);
        assert_eq!(modes[pas(0, -1) as usize], ViewMode::DensityMap);
    }
}
