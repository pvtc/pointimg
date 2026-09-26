//! Câblage des signaux GTK vers l'état applicatif.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use gtk::glib;

use pointimg::filter::{Algorithm, DotShape, HalftoneMode, Screening};

use crate::State;
use crate::actions;
use crate::controls::Widgets;
use crate::files;
use crate::preview;
use crate::sections;

type St = Rc<RefCell<State>>;
type Wi = Rc<Widgets>;

fn clone_state_widgets(state: &St, widgets: &Wi) -> (St, Wi) {
    (Rc::clone(state), Rc::clone(widgets))
}

// ─── Menu et actions ─────────────────────────────────────────────────────────

/// Modèle de menu du bouton « Exporter ».
pub(crate) fn menu_model() -> gtk::gio::Menu {
    let menu = gtk::gio::Menu::new();

    let export = gtk::gio::Menu::new();
    export.append(Some("Exporter l'image…"), Some("win.export"));
    export.append(Some("Exporter en SVG…"), Some("win.export-svg"));
    export.append(
        Some("Exporter à côté de la source"),
        Some("win.quick-export"),
    );
    export.append(
        Some("Exporter le SVG à côté de la source"),
        Some("win.quick-export-svg"),
    );
    export.append(Some("Copier dans le presse-papiers"), Some("win.copy"));
    menu.append_section(None, &export);

    let files_menu = gtk::gio::Menu::new();
    files_menu.append(Some("Ouvrir un dossier d'images…"), Some("win.open-folder"));
    files_menu.append(Some("Image précédente"), Some("win.prev-image"));
    files_menu.append(Some("Image suivante"), Some("win.next-image"));
    files_menu.append(Some("Charger un preset…"), Some("win.load-preset"));
    files_menu.append(Some("Sauver le preset…"), Some("win.save-preset"));
    menu.append_section(None, &files_menu);

    let view = gtk::gio::Menu::new();
    view.append(Some("Aperçu suivant"), Some("win.view-next"));
    view.append(Some("100 %"), Some("win.zoom-one"));
    view.append(Some("Ajuster à la fenêtre"), Some("win.zoom-fit"));
    view.append(Some("Plein écran"), Some("win.fullscreen"));
    view.append(Some("Raccourcis clavier"), Some("win.shortcuts"));
    menu.append_section(None, &view);

    let reset = gtk::gio::Menu::new();
    reset.append(Some("Réinitialiser les réglages"), Some("win.reset"));
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
        Rc::new(move || preview::zoom_by(&s, &w, 1.25))
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
        let current = sections::view_index(state.borrow().view_mode) as i32;
        (current + delta).rem_euclid(sections::VIEW_COUNT as i32) as u32
    };
    {
        let mut st = state.borrow_mut();
        st.view_mode = sections::view_from_index(next);
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
    state.borrow_mut().updating = false;
    actions::apply_preset(next as usize - 1, state, widgets);
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
        apply(&mut state.borrow_mut().params, adjustment.value());
        if let Some(after) = after {
            after(&state, &widgets);
        }
        files::mark_preset_custom(&state, &widgets);
        files::update_preset_row(&state, &widgets);
        actions::on_params_changed(&state, &widgets);
    });
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
    if let Some(sample) = crate::sample_image() {
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

    on_value(
        state,
        widgets,
        &widgets.variance,
        |p, v| p.variance_sensitivity = v as f32,
        Some(after_variance),
    );
    on_value(
        state,
        widgets,
        &widgets.min_radius,
        |p, v| {
            p.min_radius_ratio = v as f32;
            if p.max_radius_ratio < p.min_radius_ratio {
                p.max_radius_ratio = p.min_radius_ratio;
            }
        },
        Some(sync_radii),
    );
    on_value(
        state,
        widgets,
        &widgets.max_radius,
        |p, v| {
            p.max_radius_ratio = v as f32;
            if p.min_radius_ratio > p.max_radius_ratio {
                p.min_radius_ratio = p.max_radius_ratio;
            }
        },
        Some(sync_radii),
    );
    on_value(
        state,
        widgets,
        &widgets.max_boost,
        |p, v| p.max_boost = v as f32,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.cols,
        |p, v| p.cols = v.round().max(1.0) as u32,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.grid_angle,
        |p, v| p.grid_angle_deg = v as f32,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.num_points,
        |p, v| p.num_points = v.round().max(1.0) as usize,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.iterations,
        |p, v| p.iterations = v.round().max(1.0) as usize,
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
        |p, v| p.halftone_frequency = v as f32,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.halftone_min_radius,
        |p, v| p.halftone_min_radius_ratio = v as f32,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.halftone_max_dot,
        |p, v| p.halftone_max_dot_ratio = v as f32,
        None,
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
    on_switch(state, widgets, &widgets.dithering, |p, on| {
        p.dithering = on;
    });

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
            s.borrow_mut().view_mode = sections::view_from_index(drop.selected());
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
        for file in list.files() {
            let Some(path) = file.path() else {
                continue;
            };
            if path.is_dir() {
                actions::open_folder(path, &drop_state, &drop_widgets);
            } else if actions::supported_image(&path) {
                actions::load_image(path, &drop_state, &drop_widgets);
            }
            return true;
        }
        false
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
            Key::space => {
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
