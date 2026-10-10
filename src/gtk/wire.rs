//! Câblage des signaux GTK vers l'état applicatif.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use gtk::glib;

use pointimg::filter::{Algorithm, DotShape, HalftoneMode, Screening};
use pointimg::frontend::sample_image;
use pointimg::frontend::{VIEW_COUNT, view_from_index, view_index};

use crate::State;
use crate::actions;
use crate::controls::{ValueSlider, Widgets};
use crate::files;
use crate::preview;
use crate::sections;

type St = Rc<RefCell<State>>;
type Wi = Rc<Widgets>;

fn clone_state_widgets(state: &St, widgets: &Wi) -> (St, Wi) {
    (Rc::clone(state), Rc::clone(widgets))
}

// ─── Menu et actions ─────────────────────────────────────────────────────────

/// Éléments propres au bouton « Exporter ».
pub(crate) fn export_menu_model() -> gtk::gio::Menu {
    let menu = gtk::gio::Menu::new();
    menu.append(Some("Exporter l'image…"), Some("win.export"));
    menu.append(Some("Exporter en SVG…"), Some("win.export-svg"));
    menu.append(
        Some("Exporter à côté de la source"),
        Some("win.quick-export"),
    );
    menu.append(
        Some("Exporter le SVG à côté de la source"),
        Some("win.quick-export-svg"),
    );
    menu.append(Some("Copier dans le presse-papiers"), Some("win.copy"));
    menu
}

/// Menu principal de l'application, sous le bouton hamburger de l'en-tête.
///
/// Ce menu était jusqu'ici attaché au bouton « Exporter », alors qu'il ne
/// contenait presque rien d'exportiel : ouvrir un dossier, changer d'image,
/// changer l'affichage et consulter les raccourcis n'avaient aucun autre accès.
/// Les séparateurs `_` des libellés fournissent les mnémoniques clavier.
pub(crate) fn menu_model() -> gtk::gio::Menu {
    let menu = gtk::gio::Menu::new();

    menu.append(Some("_Ouvrir une image…"), Some("win.open"));
    menu.append(
        Some("Ouvrir un _dossier d'images…"),
        Some("win.open-folder"),
    );
    let images = gtk::gio::Menu::new();
    images.append(Some("_Image précédente"), Some("win.prev-image"));
    images.append(Some("Image _suivante"), Some("win.next-image"));
    menu.append_section(None, &images);

    let presets = gtk::gio::Menu::new();
    presets.append(Some("_Charger un preset…"), Some("win.load-preset"));
    presets.append(Some("_Sauver le preset…"), Some("win.save-preset"));
    menu.append_section(None, &presets);

    let view = gtk::gio::Menu::new();
    view.append(Some("_Aperçu suivant"), Some("win.view-next"));
    view.append(Some("Aperçu _précédent"), Some("win.view-prev"));
    view.append(Some("Zoom _100 %"), Some("win.zoom-one"));
    view.append(Some("_Ajuster à la fenêtre"), Some("win.zoom-fit"));
    view.append(Some("Plein _écran"), Some("win.fullscreen"));
    view.append(Some("Raccourcis _clavier"), Some("win.shortcuts"));
    menu.append_section(None, &view);

    let reset = gtk::gio::Menu::new();
    reset.append(Some("Réinitialiser les _réglages"), Some("win.reset"));
    menu.append_section(None, &reset);
    menu
}

/// Enregistre le groupe d'actions `win.*` sur la fenêtre et les raccourcis.
pub(crate) fn install_actions(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    let group = gtk::gio::SimpleActionGroup::new();
    let (s, w) = clone_state_widgets(state, widgets);
    let dialogs_window = window.clone();

    // `g_set_accels_for_action` n'est pas exposé par gio-rs : les
    // accélérateurs sont déclarés via un `GtkShortcutController` greffé à la
    // fenêtre, ce qui a en plus l'avantage de fonctionner sans GtkApplication.
    let mut accelerators: Vec<(String, Vec<String>)> = Vec::new();
    let mut add = |name: &str, accels: &[&str], f: Rc<dyn Fn()>| {
        let action = gtk::gio::SimpleAction::new(name, None);
        let callback = Rc::clone(&f);
        action.connect_activate(move |_, _| callback());
        group.add_action(&action);
        if !accels.is_empty() {
            accelerators.push((
                format!("win.{name}"),
                accels.iter().map(|a| a.to_string()).collect(),
            ));
        }
    };

    add("open", &["<Control>o"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        let window = dialogs_window.clone();
        Rc::new(move || actions::choose_open_image(&window, &s, &w))
    });
    add("open-folder", &["<Control><Shift>o"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        let window = dialogs_window.clone();
        Rc::new(move || actions::choose_open_folder(&window, &s, &w))
    });
    add("export", &["<Control>s"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        let window = dialogs_window.clone();
        Rc::new(move || actions::choose_save_image(&window, &s, &w))
    });
    add("export-svg", &["<Control><Shift>s"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        let window = dialogs_window.clone();
        Rc::new(move || actions::choose_save_svg(&window, &s, &w))
    });
    add("quick-export", &["<Control>e"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || files::quick_export(&s, &w))
    });
    add("quick-export-svg", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || files::quick_export_svg(&s, &w))
    });
    add("copy", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::copy_to_clipboard(&s, &w))
    });
    add("load-preset", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        let window = dialogs_window.clone();
        Rc::new(move || actions::choose_load_preset(&window, &s, &w))
    });
    add("save-preset", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        let window = dialogs_window.clone();
        Rc::new(move || actions::choose_save_preset(&window, &s, &w))
    });
    add("prev-image", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::step_folder(-1, &s, &w))
    });
    add("next-image", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::step_folder(1, &s, &w))
    });
    add("reset", &["<Control>r"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::reset_params(&s, &w))
    });
    add("undo", &["<Control>z"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::undo(&s, &w))
    });
    add("redo", &["<Control>y", "<Control><Shift>z"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::redo(&s, &w))
    });
    add("recalc", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::request_full_compute(&s, &w))
    });
    add("cancel", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || actions::cancel_compute(&s, &w))
    });
    add("zoom-in", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || preview::zoom_by(&s, &w, preview::ZOOM_STEP))
    });
    add("zoom-out", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || preview::zoom_by(&s, &w, 1.0 / 1.25))
    });
    add("zoom-fit", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || preview::toggle_fit(&s, &w, true))
    });
    add("zoom-one", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || preview::set_zoom(&s, &w, 1.0))
    });
    add("view-next", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || step_view(&s, &w, 1))
    });
    // Pas d'accélérateur : `Maj+Tab` est traité par le `EventControllerKey`
    // ci-dessous, qui fait la même chose et se déclenche dans les deux sens.
    add("view-prev", &[], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || step_view(&s, &w, -1))
    });
    add("fullscreen", &["F11"], {
        let window = dialogs_window.clone();
        Rc::new(move || toggle_fullscreen(&window))
    });
    add("shortcuts", &["<Control>question"], {
        let w = Rc::clone(widgets);
        Rc::new(move || w.shortcuts.present())
    });
    add("preset-next", &["<Alt>Down"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || step_preset(&s, &w, 1))
    });
    add("preset-prev", &["<Alt>Up"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || step_preset(&s, &w, -1))
    });
    add("apply-preset", &["<Control>p"], {
        let (s, w) = (Rc::clone(&s), Rc::clone(&w));
        Rc::new(move || files::apply_selected_preset(&s, &w))
    });

    window.insert_action_group("win", Some(&group));
    widgets.export.set_menu_model(Some(&menu_model()));

    let controller = gtk::ShortcutController::new();
    controller.set_propagation_phase(gtk::PropagationPhase::Capture);
    for (action_name, accels) in &accelerators {
        for accel in accels {
            let Some(trigger) = gtk::ShortcutTrigger::parse_string(accel) else {
                continue;
            };
            // La syntaxe attendue est `action(win.nom)`.
            let Some(action) = gtk::ShortcutAction::parse_string(&format!("action({action_name})"))
            else {
                continue;
            };
            controller.add_shortcut(gtk::Shortcut::new(Some(trigger), Some(action)));
        }
    }
    window.add_controller(controller);
}

fn toggle_fullscreen(window: &adw::ApplicationWindow) {
    if window.is_fullscreen() {
        window.unfullscreen();
    } else {
        window.fullscreen();
    }
}

/// Passe au mode d'affichage suivant (ou précédent avec `delta < 0`).
fn step_view(state: &St, widgets: &Wi, delta: i32) {
    let next = {
        let current = view_index(state.borrow().view_mode) as i32;
        (current + delta).rem_euclid(VIEW_COUNT as i32) as u32
    };
    {
        let mut st = state.borrow_mut();
        st.view_mode = view_from_index(next);
        st.updating = true;
    }
    widgets.view_mode.set_selected(next);
    state.borrow_mut().updating = false;
    preview::apply_zoom(state, widgets);
}

/// Applique le preset suivant/précédent de la liste.
fn step_preset(state: &St, widgets: &Wi, delta: i32) {
    let total = pointimg::filter::builtin_presets().len() as i32;
    let current = widgets.preset_combo.selected() as i32;
    let next = (current + delta).rem_euclid(total + 1) as u32;
    if next == 0 {
        files::mark_preset_custom(state, widgets);
        files::update_preset_row(state, widgets);
        return;
    }
    state.borrow_mut().updating = true;
    widgets.preset_combo.set_selected(next);
    sync_preset_thumbs(state, widgets, next);
    state.borrow_mut().updating = false;
    actions::apply_preset(next as usize - 1, state, widgets);
}

/// Aligne la grille de vignettes sur le preset courant.
///
/// `next` est la position dans la liste déroulable, où 0 vaut « Personnalisé »
/// et `n + 1` le preset `n`. Un état « personnalisé » décoche toutes les cases
/// — sans quoi une vignette resterait active alors qu'elle ne décrit plus les
/// réglages.
///
/// `updating` encadre la manipulation : `set_active` émet `toggled`, dont le
/// rappel réappliquerait le preset — une seconde fois, et pour le mauvais.
pub(crate) fn sync_preset_thumbs(state: &St, widgets: &Wi, next: u32) {
    state.borrow_mut().updating = true;
    for (index, widget) in widgets.preset_thumbs.iter().enumerate() {
        if let Ok(bouton) = widget.clone().downcast::<gtk::ToggleButton>() {
            bouton.set_active(next == index as u32 + 1);
        }
    }
    state.borrow_mut().updating = false;
}

/// Légende sous la grille de vignettes.
///
/// `texte` est celui que la liste déroulante affiche déjà — le même calcul, une
/// seule source. Sans la grille, l'opération ne fait rien : c'est le
/// sous-titre du `ComboRow` qui tient lieu de légende.
pub(crate) fn update_preset_caption(_state: &St, widgets: &Wi, texte: &str) {
    if widgets.preset_thumbs.is_empty() {
        return;
    }
    widgets.preset_caption.set_text(texte);
}

// ─── Signaux des widgets ─────────────────────────────────────────────────────

/// Branche un réglage numérique : applique la valeur puis relance (debounce).
fn on_value(
    state: &St,
    widgets: &Wi,
    row: &adw::SpinRow,
    apply: fn(&mut pointimg::filter::FilterParams, f64),
    after: Option<fn(&St, &Wi)>,
) {
    let (state, widgets) = clone_state_widgets(state, widgets);
    row.adjustment().connect_value_changed(move |adjustment| {
        if state.borrow().updating {
            return;
        }
        let before = state.borrow().params.clone();
        apply(&mut state.borrow_mut().params, adjustment.value());
        commit_value_change(&state, &widgets, before, after, false);
    });
}

/// Branche le curseur qui double un réglage numérique.
fn on_slider(
    state: &St,
    widgets: &Wi,
    slider: &ValueSlider,
    apply: fn(&mut pointimg::filter::FilterParams, f64),
    after: Option<fn(&St, &Wi)>,
) {
    let (state, widgets) = clone_state_widgets(state, widgets);
    // `spec` est `Copy` : le callback ne peut pas capturer le widget.
    let spec = slider.spec;
    slider.scale.connect_value_changed(move |scale| {
        if state.borrow().updating {
            return;
        }
        let before = state.borrow().params.clone();
        apply(&mut state.borrow_mut().params, spec.value(scale.value()));
        commit_value_change(&state, &widgets, before, after, true);
    });
}

/// Effets d'un changement de valeur numérique.
///
/// GTK réémet `value-changed` au retrait du focus d'une `SpinRow`, avec la
/// valeur ré-analysée du champ : sans ce garde-fou, l'image était recalculée
/// alors que rien n'avait bougé (passer du nombre de points au rayon min, par
/// exemple). On ne relance donc que si les paramètres ont réellement changé.
fn commit_value_change(
    state: &St,
    widgets: &Wi,
    before: pointimg::filter::FilterParams,
    after: Option<fn(&St, &Wi)>,
    from_slider: bool,
) {
    if state.borrow().params == before {
        return;
    }
    if from_slider {
        // La valeur vient du curseur : la `SpinRow` jumelle affiche encore
        // l'ancienne, et le curseur lui-même doit être recalé sur les
        // réglages voisins (bornes min/max, sections visibles).
        sections::sync_widgets(state, widgets);
    }
    if let Some(after) = after {
        after(state, widgets);
    }
    files::mark_preset_custom(state, widgets);
    files::update_preset_row(state, widgets);
    actions::on_params_changed(state, widgets);
}

/// Branche un interrupteur.
fn on_switch(
    state: &St,
    widgets: &Wi,
    row: &adw::SwitchRow,
    apply: fn(&mut pointimg::filter::FilterParams, bool),
) {
    let (state, widgets) = clone_state_widgets(state, widgets);
    row.connect_active_notify(move |row| {
        if state.borrow().updating {
            return;
        }
        apply(&mut state.borrow_mut().params, row.is_active());
        sections::update_sections(&widgets, &state.borrow().params);
        files::mark_preset_custom(&state, &widgets);
        files::update_preset_row(&state, &widgets);
        actions::on_params_changed(&state, &widgets);
    });
}

fn on_click<F>(state: &St, widgets: &Wi, button: &gtk::Button, f: F)
where
    F: Fn(&St, &Wi) + 'static,
{
    let (state, widgets) = clone_state_widgets(state, widgets);
    button.connect_clicked(move |_| f(&state, &widgets));
}

/// Branche l'ensemble des signaux.
pub(crate) fn wire(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    // ── Boutons de l'en-tête et de l'accueil ─────────────────────────────────
    for button in [&widgets.open, &widgets.open_empty] {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        button.connect_clicked(move |_| actions::choose_open_image(&window, &s, &w));
    }
    on_click(state, widgets, &widgets.cancel, actions::cancel_compute);
    on_click(
        state,
        widgets,
        &widgets.recalc,
        actions::request_full_compute,
    );
    on_click(state, widgets, &widgets.undo, actions::undo);
    on_click(state, widgets, &widgets.redo, actions::redo);
    on_click(state, widgets, &widgets.reset, actions::reset_params);
    on_click(state, widgets, &widgets.vary_seed, actions::vary_seed);

    {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        widgets
            .load_preset
            .connect_clicked(move |_| actions::choose_load_preset(&window, &s, &w));
    }
    {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        widgets
            .save_preset
            .connect_clicked(move |_| actions::choose_save_preset(&window, &s, &w));
    }
    {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        widgets
            .open_folder
            .connect_clicked(move |_| actions::choose_open_folder(&window, &s, &w));
    }

    // Exemple fourni avec le programme, s'il est présent sur le disque.
    if let Some(sample) = sample_image() {
        on_click(state, widgets, &widgets.sample_button, move |s, w| {
            actions::load_image(sample.clone(), s, w);
        });
    } else {
        widgets.sample_button.set_visible(false);
    }

    // Boutons de réinitialisation par section.
    for reset in &widgets.group_resets {
        let (s, w) = clone_state_widgets(state, widgets);
        let target = reset.target;
        reset
            .button
            .connect_clicked(move |_| actions::reset_section(target, &s, &w));
    }

    // ── Presets ─────────────────────────────────────────────────────────────
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.preset_combo.connect_selected_notify(move |_| {
            if s.borrow().updating {
                return;
            }
            files::update_preset_row(&s, &w);
            files::apply_selected_preset(&s, &w);
        });
    }
    // Grille de vignettes : un clic applique le preset. La liste déroulante
    // reste branchée en amont — la grille ne fait que la piloter — ce qui évite
    // deux chemins d'application du même preset.
    for (index, widget) in widgets.preset_thumbs.iter().enumerate() {
        let Some(bouton) = widget.clone().downcast::<gtk::ToggleButton>().ok() else {
            continue;
        };
        let (s, w) = clone_state_widgets(state, widgets);
        bouton.connect_toggled(move |bouton| {
            if !bouton.is_active() || s.borrow().updating {
                return;
            }
            // `updating` empêche le rappel de la liste déroulante de
            // réappliquer le preset une seconde fois.
            s.borrow_mut().updating = true;
            w.preset_combo.set_selected(index as u32 + 1);
            s.borrow_mut().updating = false;
            actions::apply_preset(index, &s, &w);
        });
    }

    // ── Algorithme ──────────────────────────────────────────────────────────
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.algo.connect_selected_notify(move |row| {
            if s.borrow().updating {
                return;
            }
            {
                let mut st = s.borrow_mut();
                st.params.algorithm = sections::algorithm_from_index(row.selected());
                if st.params.algorithm == Algorithm::Halftone
                    && st.params.halftone == HalftoneMode::Off
                {
                    st.params.halftone = HalftoneMode::Cmyk {
                        angles: [15.0, 75.0, 0.0, 45.0],
                    };
                }
            }
            sections::sync_widgets(&s, &w);
            sections::update_sections(&w, &s.borrow().params);
            files::mark_preset_custom(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }

    // Les réglages doublés d'un curseur passent par des fonctions nommées :
    // la `SpinRow` et son curseur appliquent exactement la même conversion.
    on_value(
        state,
        widgets,
        &widgets.variance,
        set_variance,
        Some(after_variance),
    );
    on_value(
        state,
        widgets,
        &widgets.min_radius,
        set_min_radius,
        Some(sync_radii),
    );
    on_value(
        state,
        widgets,
        &widgets.max_radius,
        set_max_radius,
        Some(sync_radii),
    );
    on_value(state, widgets, &widgets.max_boost, set_max_boost, None);
    on_value(state, widgets, &widgets.cols, set_cols, None);
    on_value(
        state,
        widgets,
        &widgets.grid_angle,
        |p, v| p.grid_angle_deg = v as f32,
        None,
    );
    on_value(state, widgets, &widgets.num_points, set_num_points, None);
    on_value(state, widgets, &widgets.iterations, set_iterations, None);

    on_slider(
        state,
        widgets,
        &widgets.variance_slider,
        set_variance,
        Some(after_variance),
    );
    on_slider(
        state,
        widgets,
        &widgets.min_radius_slider,
        set_min_radius,
        Some(sync_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.max_radius_slider,
        set_max_radius,
        Some(sync_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.max_boost_slider,
        set_max_boost,
        None,
    );
    on_slider(state, widgets, &widgets.cols_slider, set_cols, None);
    on_slider(
        state,
        widgets,
        &widgets.num_points_slider,
        set_num_points,
        None,
    );
    on_slider(
        state,
        widgets,
        &widgets.iterations_slider,
        set_iterations,
        None,
    );

    // ── Qualité de l'aperçu ─────────────────────────────────────────────────
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.quality.connect_selected_notify(move |row| {
            if s.borrow().updating {
                return;
            }
            let quality = sections::quality_from_index(row.selected());
            let changed = {
                let mut st = s.borrow_mut();
                let changed = st.quality != quality;
                st.quality = quality;
                changed
            };
            if changed {
                actions::request_compute(&s, &w);
            }
        });
    }

    // ── Halftone ────────────────────────────────────────────────────────────
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.halftone_mode.connect_selected_notify(move |row| {
            if s.borrow().updating {
                return;
            }
            s.borrow_mut().params.halftone = if row.selected() == 1 {
                HalftoneMode::Dominant {
                    n: 5,
                    base_angle_deg: 15.0,
                }
            } else {
                HalftoneMode::Cmyk {
                    angles: [15.0, 75.0, 0.0, 45.0],
                }
            };
            sections::sync_widgets(&s, &w);
            sections::update_sections(&w, &s.borrow().params);
            files::mark_preset_custom(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }
    on_value(
        state,
        widgets,
        &widgets.dominant_n,
        |p, v| {
            if let HalftoneMode::Dominant { n, .. } = &mut p.halftone {
                *n = v.round().clamp(2.0, 32.0) as usize;
            }
        },
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.dominant_angle,
        |p, v| {
            if let HalftoneMode::Dominant { base_angle_deg, .. } = &mut p.halftone {
                *base_angle_deg = v as f32;
            }
        },
        None,
    );
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.screening.connect_selected_notify(move |row| {
            if s.borrow().updating {
                return;
            }
            s.borrow_mut().params.screening = if row.selected() == 1 {
                Screening::Fm
            } else {
                Screening::Am
            };
            files::mark_preset_custom(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }
    on_value(
        state,
        widgets,
        &widgets.halftone_freq,
        set_halftone_freq,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.halftone_min_radius,
        set_halftone_min_radius,
        Some(sync_halftone_radii),
    );
    on_value(
        state,
        widgets,
        &widgets.halftone_max_dot,
        set_halftone_max_dot,
        Some(sync_halftone_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.halftone_freq_slider,
        set_halftone_freq,
        None,
    );
    on_slider(
        state,
        widgets,
        &widgets.halftone_min_slider,
        set_halftone_min_radius,
        Some(sync_halftone_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.halftone_max_slider,
        set_halftone_max_dot,
        Some(sync_halftone_radii),
    );

    // ── Forme des points ────────────────────────────────────────────────────
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.shape.connect_selected_notify(move |row| {
            if s.borrow().updating {
                return;
            }
            {
                let mut st = s.borrow_mut();
                st.params.dot_shape = match row.selected() {
                    0 => DotShape::Circle,
                    1 => DotShape::Square,
                    2 => match st.params.dot_shape {
                        DotShape::Ellipse { .. } => st.params.dot_shape,
                        _ => DotShape::Ellipse {
                            aspect: 1.5,
                            angle_deg: 0.0,
                        },
                    },
                    3 => match st.params.dot_shape {
                        DotShape::RegularPolygon { .. } => st.params.dot_shape,
                        _ => DotShape::RegularPolygon { sides: 6 },
                    },
                    _ => DotShape::Circle,
                };
            }
            sections::sync_widgets(&s, &w);
            sections::update_sections(&w, &s.borrow().params);
            files::mark_preset_custom(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }
    on_value(
        state,
        widgets,
        &widgets.ellipse_aspect,
        |p, v| {
            if let DotShape::Ellipse { aspect, .. } = &mut p.dot_shape {
                *aspect = v as f32;
            }
        },
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.ellipse_angle,
        |p, v| {
            if let DotShape::Ellipse { angle_deg, .. } = &mut p.dot_shape {
                *angle_deg = v as f32;
            }
        },
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.polygon_sides,
        |p, v| {
            if let DotShape::RegularPolygon { sides } = &mut p.dot_shape {
                *sides = v.round().clamp(3.0, 12.0) as u8;
            }
        },
        None,
    );

    on_switch(state, widgets, &widgets.use_palette, |p, on| {
        p.palette_size = if on { Some(8) } else { None };
    });
    on_value(
        state,
        widgets,
        &widgets.palette_size,
        |p, v| {
            p.palette_size = Some(v.round().clamp(2.0, 256.0) as usize);
        },
        None,
    );
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.dithering.connect_selected_notify(move |row| {
            if s.borrow().updating {
                return;
            }
            let mode = sections::dithering_from_index(row.selected());
            s.borrow_mut().params.dithering = mode;
            sections::sync_dithering_note(mode, &w);
            files::mark_preset_custom(&s, &w);
            files::update_preset_row(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }

    // ── Rendu ───────────────────────────────────────────────────────────────
    {
        let (s, w) = clone_state_widgets(state, widgets);
        let button = w.bg_button.clone();
        button.connect_rgba_notify(move |b| {
            if s.borrow().updating {
                return;
            }
            let rgba = b.rgba();
            {
                let mut st = s.borrow_mut();
                st.params.bg_color = [
                    (rgba.red() * 255.0).round() as u8,
                    (rgba.green() * 255.0).round() as u8,
                    (rgba.blue() * 255.0).round() as u8,
                ];
                st.params.transparent = false;
                st.updating = true;
            }
            // Les setters GTK émettent des signaux synchrones : on relâche le
            // borrow avant de toucher aux widgets.
            w.transparent.set_active(false);
            s.borrow_mut().updating = false;
            sections::update_sections(&w, &s.borrow().params);
            actions::refresh_src_rgb(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }
    on_switch(state, widgets, &widgets.transparent, |p, on| {
        p.transparent = on;
    });
    on_switch(state, widgets, &widgets.gamma, |p, on| {
        p.gamma_correct = on;
    });
    on_switch(state, widgets, &widgets.use_seed, |p, on| {
        p.rng_seed = if on { Some(42) } else { None };
    });
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.seed.connect_value_changed(move |spin| {
            if s.borrow().updating {
                return;
            }
            s.borrow_mut().params.rng_seed = Some(spin.value().max(0.0) as u64);
            files::mark_preset_custom(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }

    // ── Affichage ───────────────────────────────────────────────────────────
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.view_mode.connect_selected_notify(move |drop| {
            if s.borrow().updating {
                return;
            }
            s.borrow_mut().view_mode = view_from_index(drop.selected());
            preview::apply_zoom(&s, &w);
        });
    }
    on_click(state, widgets, &widgets.zoom_out, |s, w| {
        preview::zoom_by(s, w, 1.0 / 1.25);
    });
    on_click(state, widgets, &widgets.zoom_in, |s, w| {
        preview::zoom_by(s, w, 1.25);
    });
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.zoom_one.connect_toggled(move |button| {
            if s.borrow().updating {
                return;
            }
            if button.is_active() {
                preview::set_zoom(&s, &w, 1.0);
            }
        });
    }
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.zoom_fit.connect_toggled(move |button| {
            if s.borrow().updating {
                return;
            }
            preview::toggle_fit(&s, &w, button.is_active());
        });
    }

    preview::connect_scroll_sync(widgets);
}

/// La density map dépend de la sensibilité à la variance : on la recalcule
/// séparément, sans relancer le filtre.
fn after_variance(state: &St, widgets: &Wi) {
    actions::start_density(state);
    let _ = widgets;
}

fn set_variance(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.variance_sensitivity = v as f32;
}

fn set_min_radius(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.min_radius_ratio = v as f32;
    // La borne basse ne peut pas dépasser la haute.
    if p.max_radius_ratio < p.min_radius_ratio {
        p.max_radius_ratio = p.min_radius_ratio;
    }
}

fn set_max_radius(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.max_radius_ratio = v as f32;
    if p.min_radius_ratio > p.max_radius_ratio {
        p.min_radius_ratio = p.max_radius_ratio;
    }
}

fn set_max_boost(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.max_boost = v as f32;
}

fn set_cols(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.cols = v.round().max(1.0) as u32;
}

fn set_num_points(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.num_points = v.round().max(1.0) as usize;
}

fn set_iterations(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.iterations = v.round().max(1.0) as usize;
}

fn set_halftone_freq(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.halftone_frequency = v as f32;
}

fn set_halftone_min_radius(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.halftone_min_radius_ratio = v as f32;
    if p.halftone_max_dot_ratio < p.halftone_min_radius_ratio {
        p.halftone_max_dot_ratio = p.halftone_min_radius_ratio;
    }
}

fn set_halftone_max_dot(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.halftone_max_dot_ratio = v as f32;
    if p.halftone_min_radius_ratio > p.halftone_max_dot_ratio {
        p.halftone_min_radius_ratio = p.halftone_max_dot_ratio;
    }
}

/// Idem [`sync_radii`] pour la paire de bornes du halftone.
fn sync_halftone_radii(state: &St, widgets: &Wi) {
    if state.borrow().updating {
        return;
    }
    let (min, max) = {
        let st = state.borrow();
        (
            st.params.halftone_min_radius_ratio,
            st.params.halftone_max_dot_ratio,
        )
    };
    state.borrow_mut().updating = true;
    widgets
        .halftone_min_radius
        .adjustment()
        .set_value(f64::from(min));
    widgets
        .halftone_max_dot
        .adjustment()
        .set_value(f64::from(max));
    widgets.halftone_min_slider.set_value(f64::from(min));
    widgets.halftone_max_slider.set_value(f64::from(max));
    widgets.halftone_min_slider.set_bound(f64::from(max));
    widgets.halftone_max_slider.set_bound(f64::from(min));
    state.borrow_mut().updating = false;
}

/// Les deux rayons sont bornés l'un par l'autre : après un changement, la ligne
/// contrainte doit être réalignée sur la valeur retenue.
fn sync_radii(state: &St, widgets: &Wi) {
    if state.borrow().updating {
        return;
    }
    let (min, max) = {
        let st = state.borrow();
        (st.params.min_radius_ratio, st.params.max_radius_ratio)
    };
    state.borrow_mut().updating = true;
    widgets.min_radius.adjustment().set_value(f64::from(min));
    widgets.max_radius.adjustment().set_value(f64::from(max));
    widgets.min_radius_slider.set_value(f64::from(min));
    widgets.max_radius_slider.set_value(f64::from(max));
    widgets.min_radius_slider.set_bound(f64::from(max));
    widgets.max_radius_slider.set_bound(f64::from(min));
    state.borrow_mut().updating = false;
}

// ─── Glisser-déposer ─────────────────────────────────────────────────────────

pub(crate) fn install_dnd(window: &adw::ApplicationWindow, state: &St, widgets: &Wi) {
    let drop = gtk::DropTarget::new(
        gtk::gdk::FileList::static_type(),
        gtk::gdk::DragAction::COPY,
    );
    let drop_state = Rc::clone(state);
    let drop_widgets = Rc::clone(widgets);
    drop.connect_drop(move |_, value, _, _| {
        let Ok(list) = value.get::<gtk::gdk::FileList>() else {
            return false;
        };
        // Chaque fichier est traité, et le geste est accepté dès que la liste
        // contient au moins un élément utile. Le `return true` était à
        // l'intérieur de la boucle : il sortait après le premier fichier, quoi
        // qu'il advienne des suivants. Déposer un dossier et une image ne
        // chargeait donc que le dossier.
        let mut traite = false;
        for file in list.files() {
            let Some(path) = file.path() else {
                continue;
            };
            if path.is_dir() {
                actions::open_folder(path, &drop_state, &drop_widgets);
                traite = true;
            } else if actions::supported_image(&path) {
                actions::load_image(path, &drop_state, &drop_widgets);
                traite = true;
            }
        }
        traite
    });
    window.add_controller(drop);
}

// ─── Raccourcis clavier ──────────────────────────────────────────────────────

pub(crate) fn install_shortcuts(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    let controller = gtk::EventControllerKey::new();
    let key_state = Rc::clone(state);
    let key_widgets = Rc::clone(widgets);
    let key_window = window.clone();
    controller.connect_key_pressed(move |_, keyval, _, modifiers| {
        use gtk::gdk::{Key, ModifierType};
        let ctrl = modifiers.intersects(ModifierType::CONTROL_MASK | ModifierType::META_MASK);
        // Tant qu'un champ de saisie a le focus, les raccourcis « lettre »
        // risqueraient d'être tapés dans le champ : on les laisse passer.
        let editing = key_widgets.seed.has_focus() || key_widgets.bg_button.has_focus();
        if ctrl {
            if keyval == Key::c && !editing {
                actions::copy_to_clipboard(&key_state, &key_widgets);
                return glib::Propagation::Stop;
            }
            return glib::Propagation::Proceed;
        }
        match keyval {
            // `editing` garde déjà les raccourcis « lettre » : l'espace est le
            // seul caractère qu'un champ de saisie peut recevoir au milieu d'un
            // mot (une couleur « #ff cc00 », un nom de fichier). Sans la garde,
            // taper un espace dans « Couleur du fond » relançait un calcul.
            Key::space if !editing => {
                actions::request_full_compute(&key_state, &key_widgets);
                glib::Propagation::Stop
            }
            Key::plus | Key::equal | Key::KP_Add if !editing => {
                preview::zoom_by(&key_state, &key_widgets, 1.25);
                glib::Propagation::Stop
            }
            Key::minus | Key::underscore | Key::KP_Subtract if !editing => {
                preview::zoom_by(&key_state, &key_widgets, 1.0 / 1.25);
                glib::Propagation::Stop
            }
            Key::_0 if !editing => {
                preview::toggle_fit(&key_state, &key_widgets, true);
                glib::Propagation::Stop
            }
            Key::_1 if !editing => {
                preview::set_zoom(&key_state, &key_widgets, 1.0);
                glib::Propagation::Stop
            }
            Key::Tab if !editing => {
                let delta = if modifiers.contains(ModifierType::SHIFT_MASK) {
                    -1
                } else {
                    1
                };
                step_view(&key_state, &key_widgets, delta);
                glib::Propagation::Stop
            }
            Key::Left | Key::Right if !editing => {
                let delta = if keyval == Key::Right { 1 } else { -1 };
                actions::step_folder(delta, &key_state, &key_widgets);
                glib::Propagation::Stop
            }
            Key::Escape => {
                actions::cancel_compute(&key_state, &key_widgets);
                glib::Propagation::Stop
            }
            Key::F11 => {
                if key_window.is_fullscreen() {
                    key_window.unfullscreen();
                } else {
                    key_window.fullscreen();
                }
                glib::Propagation::Stop
            }
            _ => glib::Propagation::Proceed,
        }
    });
    window.add_controller(controller);

    // Molette avec Ctrl : zoom (le zoom conserve le centre du viewport).
    let scroll_state = Rc::clone(state);
    let scroll_widgets = Rc::clone(widgets);
    let scroll = gtk::EventControllerScroll::builder()
        .flags(
            gtk::EventControllerScrollFlags::VERTICAL | gtk::EventControllerScrollFlags::DISCRETE,
        )
        .build();
    scroll.connect_scroll(move |controller, _dx, dy| {
        if !controller
            .current_event_state()
            .contains(gtk::gdk::ModifierType::CONTROL_MASK)
        {
            return glib::Propagation::Proceed;
        }
        let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
        preview::zoom_by(&scroll_state, &scroll_widgets, factor);
        glib::Propagation::Stop
    });
    window.add_controller(scroll);
}
