//! Wiring of the GTK signals towards the application state.

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

/// Raises `updating` for the duration of `body`, then puts it back down.
///
/// Writing in a widget re-emits its signal; `updating` prevents the callback
/// from triggering a recompute during that write. Keeping it in a closure
/// rather than in two `borrow_mut`s around a block ensures that the guard is
/// put back even if the block one day gains a `return` — which the open form
/// does not guarantee.
fn with_updating<F>(state: &St, widgets: &Wi, body: F)
where
    F: FnOnce(&Wi),
{
    state.borrow_mut().updating = true;
    body(widgets);
    state.borrow_mut().updating = false;
}

fn clone_state_widgets(state: &St, widgets: &Wi) -> (St, Wi) {
    (Rc::clone(state), Rc::clone(widgets))
}

// ─── Menus and actions ───────────────────────────────────────────────────────

/// Items specific to the "Exporter" button.
pub(crate) fn export_menu_model() -> gtk::gio::Menu {
    let menu = gtk::gio::Menu::new();
    menu.append(
        Some(pointimg::frontend::gettext("Exporter l'image…")),
        Some("win.export"),
    );
    menu.append(
        Some(pointimg::frontend::gettext("Exporter en SVG…")),
        Some("win.export-svg"),
    );
    menu.append(
        Some(pointimg::frontend::gettext("Exporter à côté de la source")),
        Some("win.quick-export"),
    );
    menu.append(
        Some(pointimg::frontend::gettext(
            "Exporter le SVG à côté de la source",
        )),
        Some("win.quick-export-svg"),
    );
    menu.append(
        Some(pointimg::frontend::gettext("Copier dans le presse-papiers")),
        Some("win.copy"),
    );
    menu
}

/// Main menu of the application, under the hamburger button of the header.
///
/// Opening a folder, changing image, changing the display and consulting the
/// shortcuts have no other access than this menu. The `_` separators of the
/// labels provide the keyboard mnemonics.
pub(crate) fn menu_model() -> gtk::gio::Menu {
    let menu = gtk::gio::Menu::new();

    menu.append(
        Some(pointimg::frontend::gettext("_Ouvrir une image…")),
        Some("win.open"),
    );
    menu.append(
        Some(pointimg::frontend::gettext("Ouvrir un _dossier d'images…")),
        Some("win.open-folder"),
    );
    let images = gtk::gio::Menu::new();
    images.append(
        Some(pointimg::frontend::gettext("_Image précédente")),
        Some("win.prev-image"),
    );
    images.append(
        Some(pointimg::frontend::gettext("Image _suivante")),
        Some("win.next-image"),
    );
    menu.append_section(None, &images);

    let presets = gtk::gio::Menu::new();
    presets.append(
        Some(pointimg::frontend::gettext("_Charger un preset…")),
        Some("win.load-preset"),
    );
    presets.append(
        Some(pointimg::frontend::gettext("_Sauver le preset…")),
        Some("win.save-preset"),
    );
    menu.append_section(None, &presets);

    let view = gtk::gio::Menu::new();
    view.append(
        Some(pointimg::frontend::gettext("_Aperçu suivant")),
        Some("win.view-next"),
    );
    view.append(
        Some(pointimg::frontend::gettext("Aperçu _précédent")),
        Some("win.view-prev"),
    );
    view.append(Some("Zoom _100 %"), Some("win.zoom-one"));
    view.append(
        Some(pointimg::frontend::gettext("_Ajuster à la fenêtre")),
        Some("win.zoom-fit"),
    );
    view.append(
        Some(pointimg::frontend::gettext("Plein _écran")),
        Some("win.fullscreen"),
    );
    view.append(
        Some(pointimg::frontend::gettext("Raccourcis _clavier")),
        Some("win.shortcuts"),
    );
    menu.append_section(None, &view);

    let reset = gtk::gio::Menu::new();
    reset.append(
        Some(pointimg::frontend::gettext("Réinitialiser les _réglages")),
        Some("win.reset"),
    );
    menu.append_section(None, &reset);
    menu
}

/// Registers the `win.*` action group on the window and the shortcuts.
pub(crate) fn install_actions(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    let group = gtk::gio::SimpleActionGroup::new();
    let (s, w) = clone_state_widgets(state, widgets);
    let dialogs_window = window.clone();

    // `g_set_accels_for_action` is not exposed by gio-rs: the accelerators
    // are declared via a `GtkShortcutController` grafted onto the window,
    // which has the additional advantage of working without GtkApplication.
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
    // No accelerator: `Maj+Tab` is handled by the `EventControllerKey`
    // below, which does the same thing and fires in both directions.
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
        Rc::new(move || w.window.shortcuts.present())
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
    widgets.window.export.set_menu_model(Some(&menu_model()));

    let controller = gtk::ShortcutController::new();
    controller.set_propagation_phase(gtk::PropagationPhase::Capture);
    for (action_name, accels) in &accelerators {
        for accel in accels {
            let Some(trigger) = gtk::ShortcutTrigger::parse_string(accel) else {
                continue;
            };
            // The expected syntax is `action(win.nom)`.
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

/// Switches to the next display mode (or previous with `delta < 0`).
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
    widgets.preview.view_mode.set_selected(next);
    state.borrow_mut().updating = false;
    preview::apply_zoom(state, widgets);
}

/// Applies the next/previous preset of the list.
fn step_preset(state: &St, widgets: &Wi, delta: i32) {
    let total = pointimg::filter::builtin_presets().len() as i32;
    let current = widgets.presets.preset_combo.selected() as i32;
    let next = (current + delta).rem_euclid(total + 1) as u32;
    if next == 0 {
        files::mark_preset_custom(state, widgets);
        files::update_preset_row(state, widgets);
        return;
    }
    state.borrow_mut().updating = true;
    widgets.presets.preset_combo.set_selected(next);
    sync_preset_thumbs(state, widgets, next);
    state.borrow_mut().updating = false;
    actions::apply_preset(next as usize - 1, state, widgets);
}

/// Aligns the thumbnail grid on the current preset.
///
/// `next` is the position in the drop-down list, where 0 means "Personnalisé"
/// and `n + 1` is preset `n`. A "custom" state unchecks all the boxes —
/// otherwise a thumbnail would stay active while it no longer describes the
/// settings.
///
/// `updating` frames the manipulation: `set_active` emits `toggled`, whose
/// callback would reapply the preset — a second time, and for the wrong one.
pub(crate) fn sync_preset_thumbs(state: &St, widgets: &Wi, next: u32) {
    state.borrow_mut().updating = true;
    for (index, widget) in widgets.presets.preset_thumbs.iter().enumerate() {
        if let Ok(bouton) = widget.clone().downcast::<gtk::ToggleButton>() {
            bouton.set_active(next == index as u32 + 1);
        }
    }
    state.borrow_mut().updating = false;
}

/// Caption under the thumbnail grid.
///
/// `texte` is the one the drop-down list already displays — the same
/// computation, a single source. Without the grid, the operation does
/// nothing: it is the subtitle of the `ComboRow` that serves as the caption.
pub(crate) fn update_preset_caption(_state: &St, widgets: &Wi, texte: &str) {
    if widgets.presets.preset_thumbs.is_empty() {
        return;
    }
    widgets.presets.preset_caption.set_text(texte);
}

// ─── Widget signals ─────────────────────────────────────────────────────────

/// Wires up a numeric setting: applies the value then recomputes (debounce).
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

/// Wires up the slider that doubles a numeric setting.
fn on_slider(
    state: &St,
    widgets: &Wi,
    slider: &ValueSlider,
    apply: fn(&mut pointimg::filter::FilterParams, f64),
    after: Option<fn(&St, &Wi)>,
) {
    let (state, widgets) = clone_state_widgets(state, widgets);
    // `spec` is `Copy`: the callback cannot capture the widget.
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

/// Effects of a numeric value change.
///
/// GTK re-emits `value-changed` when a `SpinRow` loses focus, with the
/// re-parsed value of the field: without that safeguard, the image would be
/// recomputed although nothing moved (going from the number of points to the
/// min radius, for instance). We therefore only recompute if the parameters
/// really changed.
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
        // The value comes from the slider: the twin `SpinRow` displays a stale
        // value, and the neighboring settings (min/max bounds, visible
        // sections) must be resynchronized.
        sections::sync_widgets(state, widgets);
    }
    if let Some(after) = after {
        after(state, widgets);
    }
    files::mark_preset_custom(state, widgets);
    files::update_preset_row(state, widgets);
    actions::on_params_changed(state, widgets);
}

/// Wires up a switch.
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

/// Wires up the whole set of signals.
///
/// Eight groups, one function each: each section already carried its title in
/// a comment, but the 500 lines formed a single block where finding a wiring
/// required scrolling through all of it.
pub(crate) fn wire(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    // Language selector: persisted immediately, applied on the next launch
    // (GTK rebuilds its labels at build time).
    let widgets_for_locale = Rc::clone(widgets);
    widgets
        .window
        .locale_combo
        .connect_selected_notify(move |combo| {
            let locale = if combo.selected() == 1 {
                pointimg::frontend::Locale::Fr
            } else {
                pointimg::frontend::Locale::En
            };
            pointimg::frontend::set_locale(locale);
            crate::prefs::save_locale(locale);
            let toast = adw::Toast::new(pointimg::frontend::gettext(
                "Langue modifiée. Redémarrez l'application pour l'appliquer.",
            ));
            toast.set_timeout(4);
            widgets_for_locale.window.toast_overlay.add_toast(toast);
        });

    wire_header_buttons(state, widgets, window);
    wire_presets(state, widgets);
    wire_algorithm(state, widgets);
    wire_quality(state, widgets);
    wire_halftone(state, widgets);
    wire_shape(state, widgets);
    wire_render(state, widgets);
    wire_display(state, widgets);

    preview::connect_scroll_sync(widgets);
}

/// Signals of the "Boutons de l'en-tête et de l'accueil" group.
fn wire_header_buttons(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    for button in [&widgets.window.open, &widgets.status_bar.open_empty] {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        button.connect_clicked(move |_| actions::choose_open_image(&window, &s, &w));
    }
    on_click(
        state,
        widgets,
        &widgets.window.cancel,
        actions::cancel_compute,
    );
    on_click(
        state,
        widgets,
        &widgets.compute.recalc,
        actions::request_full_compute,
    );
    on_click(state, widgets, &widgets.window.undo, actions::undo);
    on_click(state, widgets, &widgets.window.redo, actions::redo);
    on_click(
        state,
        widgets,
        &widgets.compute.reset,
        actions::reset_params,
    );
    on_click(
        state,
        widgets,
        &widgets.render.vary_seed,
        actions::vary_seed,
    );

    {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        widgets
            .presets
            .load_preset
            .connect_clicked(move |_| actions::choose_load_preset(&window, &s, &w));
    }
    {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        widgets
            .presets
            .save_preset
            .connect_clicked(move |_| actions::choose_save_preset(&window, &s, &w));
    }
    {
        let (s, w) = clone_state_widgets(state, widgets);
        let window = window.clone();
        widgets
            .presets
            .open_folder
            .connect_clicked(move |_| actions::choose_open_folder(&window, &s, &w));
    }

    // Example provided with the program, if it is present on the disk.
    if let Some(sample) = sample_image() {
        on_click(
            state,
            widgets,
            &widgets.status_bar.sample_button,
            move |s, w| {
                actions::load_image(sample.clone(), s, w);
            },
        );
    } else {
        widgets.status_bar.sample_button.set_visible(false);
    }

    // Per-section reset buttons.
    for reset in &widgets.compute.group_resets {
        let (s, w) = clone_state_widgets(state, widgets);
        let target = reset.target;
        reset
            .button
            .connect_clicked(move |_| actions::reset_section(target, &s, &w));
    }
}

/// Signals of the "Presets" group.
fn wire_presets(state: &St, widgets: &Wi) {
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets
            .presets
            .preset_combo
            .connect_selected_notify(move |_| {
                if s.borrow().updating {
                    return;
                }
                files::update_preset_row(&s, &w);
                files::apply_selected_preset(&s, &w);
            });
    }
    // Thumbnail grid: a click applies the preset. The drop-down list stays
    // wired upstream — the grid only drives it — which avoids two paths for
    // applying the same preset.
    for (index, widget) in widgets.presets.preset_thumbs.iter().enumerate() {
        let Some(bouton) = widget.clone().downcast::<gtk::ToggleButton>().ok() else {
            continue;
        };
        let (s, w) = clone_state_widgets(state, widgets);
        bouton.connect_toggled(move |bouton| {
            if !bouton.is_active() || s.borrow().updating {
                return;
            }
            // `updating` prevents the callback of the drop-down list from
            // reapplying the preset a second time.
            s.borrow_mut().updating = true;
            w.presets.preset_combo.set_selected(index as u32 + 1);
            s.borrow_mut().updating = false;
            actions::apply_preset(index, &s, &w);
        });
    }
}

/// Signals of the "Algorithme" group.
fn wire_algorithm(state: &St, widgets: &Wi) {
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.algorithm.algo.connect_selected_notify(move |row| {
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

    // The settings doubled by a slider go through named functions: the
    // `SpinRow` and its slider apply exactly the same conversion.
    on_value(
        state,
        widgets,
        &widgets.algorithm.variance,
        set_variance,
        Some(after_variance),
    );
    on_value(
        state,
        widgets,
        &widgets.algorithm.min_radius,
        set_min_radius,
        Some(sync_radii),
    );
    on_value(
        state,
        widgets,
        &widgets.algorithm.max_radius,
        set_max_radius,
        Some(sync_radii),
    );
    on_value(
        state,
        widgets,
        &widgets.algorithm.max_boost,
        set_max_boost,
        None,
    );
    on_value(state, widgets, &widgets.algorithm.cols, set_cols, None);
    on_value(
        state,
        widgets,
        &widgets.algorithm.grid_angle,
        |p, v| p.grid_angle_deg = v as f32,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.algorithm.num_points,
        set_num_points,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.algorithm.iterations,
        set_iterations,
        None,
    );

    on_slider(
        state,
        widgets,
        &widgets.algorithm.variance_slider,
        set_variance,
        Some(after_variance),
    );
    on_slider(
        state,
        widgets,
        &widgets.algorithm.min_radius_slider,
        set_min_radius,
        Some(sync_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.algorithm.max_radius_slider,
        set_max_radius,
        Some(sync_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.algorithm.max_boost_slider,
        set_max_boost,
        None,
    );
    on_slider(
        state,
        widgets,
        &widgets.algorithm.cols_slider,
        set_cols,
        None,
    );
    on_slider(
        state,
        widgets,
        &widgets.algorithm.num_points_slider,
        set_num_points,
        None,
    );
    on_slider(
        state,
        widgets,
        &widgets.algorithm.iterations_slider,
        set_iterations,
        None,
    );
}

/// Signals of the "Qualité de l'aperçu" group.
fn wire_quality(state: &St, widgets: &Wi) {
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.compute.quality.connect_selected_notify(move |row| {
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
}

/// Signals of the "Trame" group.
fn wire_halftone(state: &St, widgets: &Wi) {
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets
            .halftone
            .halftone_mode
            .connect_selected_notify(move |row| {
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
        &widgets.halftone.dominant_n,
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
        &widgets.halftone.dominant_angle,
        |p, v| {
            if let HalftoneMode::Dominant { base_angle_deg, .. } = &mut p.halftone {
                *base_angle_deg = v as f32;
            }
        },
        None,
    );
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets
            .halftone
            .screening
            .connect_selected_notify(move |row| {
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
        &widgets.halftone.halftone_freq,
        set_halftone_freq,
        None,
    );
    on_value(
        state,
        widgets,
        &widgets.halftone.halftone_min_radius,
        set_halftone_min_radius,
        Some(sync_halftone_radii),
    );
    on_value(
        state,
        widgets,
        &widgets.halftone.halftone_max_dot,
        set_halftone_max_dot,
        Some(sync_halftone_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.halftone.halftone_freq_slider,
        set_halftone_freq,
        None,
    );
    on_slider(
        state,
        widgets,
        &widgets.halftone.halftone_min_slider,
        set_halftone_min_radius,
        Some(sync_halftone_radii),
    );
    on_slider(
        state,
        widgets,
        &widgets.halftone.halftone_max_slider,
        set_halftone_max_dot,
        Some(sync_halftone_radii),
    );
}

/// Signals of the "Forme des points" group.
fn wire_shape(state: &St, widgets: &Wi) {
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.dots.shape.connect_selected_notify(move |row| {
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
        &widgets.dots.ellipse_aspect,
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
        &widgets.dots.ellipse_angle,
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
        &widgets.dots.polygon_sides,
        |p, v| {
            if let DotShape::RegularPolygon { sides } = &mut p.dot_shape {
                *sides = v.round().clamp(3.0, 12.0) as u8;
            }
        },
        None,
    );

    on_switch(state, widgets, &widgets.dots.use_palette, |p, on| {
        p.palette_size = if on { Some(8) } else { None };
    });
    on_value(
        state,
        widgets,
        &widgets.dots.palette_size,
        |p, v| {
            p.palette_size = Some(v.round().clamp(2.0, 256.0) as usize);
        },
        None,
    );
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.dots.dithering.connect_selected_notify(move |row| {
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
}

/// Signals of the "Rendu" group.
fn wire_render(state: &St, widgets: &Wi) {
    {
        let (s, w) = clone_state_widgets(state, widgets);
        let button = w.render.bg_button.clone();
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
            // GTK setters emit synchronous signals: the borrow is released
            // before touching the widgets.
            w.render.transparent.set_active(false);
            s.borrow_mut().updating = false;
            sections::update_sections(&w, &s.borrow().params);
            actions::refresh_src_rgb(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }
    on_switch(state, widgets, &widgets.render.transparent, |p, on| {
        p.transparent = on;
    });
    on_switch(state, widgets, &widgets.render.gamma, |p, on| {
        p.gamma_correct = on;
    });
    on_switch(state, widgets, &widgets.render.use_seed, |p, on| {
        p.rng_seed = if on { Some(42) } else { None };
    });
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.render.seed.connect_value_changed(move |spin| {
            if s.borrow().updating {
                return;
            }
            s.borrow_mut().params.rng_seed = Some(spin.value().max(0.0) as u64);
            files::mark_preset_custom(&s, &w);
            actions::on_params_changed(&s, &w);
        });
    }
}

/// Signals of the "Affichage" group.
fn wire_display(state: &St, widgets: &Wi) {
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets
            .preview
            .view_mode
            .connect_selected_notify(move |drop| {
                if s.borrow().updating {
                    return;
                }
                s.borrow_mut().view_mode = view_from_index(drop.selected());
                preview::apply_zoom(&s, &w);
            });
    }
    on_click(state, widgets, &widgets.preview.zoom_out, |s, w| {
        preview::zoom_by(s, w, 1.0 / 1.25);
    });
    on_click(state, widgets, &widgets.preview.zoom_in, |s, w| {
        preview::zoom_by(s, w, 1.25);
    });
    {
        let (s, w) = clone_state_widgets(state, widgets);
        widgets.preview.zoom_one.connect_toggled(move |button| {
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
        widgets.preview.zoom_fit.connect_toggled(move |button| {
            if s.borrow().updating {
                return;
            }
            preview::toggle_fit(&s, &w, button.is_active());
        });
    }

    preview::connect_scroll_sync(widgets);
}

/// The density map depends on the variance sensitivity: it is recomputed
/// separately, without rerunning the filter.
fn after_variance(state: &St, widgets: &Wi) {
    actions::start_density(state);
    let _ = widgets;
}

fn set_variance(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.variance_sensitivity = v as f32;
}

fn set_min_radius(p: &mut pointimg::filter::FilterParams, v: f64) {
    p.min_radius_ratio = v as f32;
    // The low end cannot exceed the high end.
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

/// Same as [`sync_radii`] for the pair of halftone bounds.
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
    // The guard frames the writing of the two lines: here, `set_value` and
    // `set_bound` both re-emit, and the guard is only lifted at the end.
    // That is exactly what the two copies did, but the closure guarantees that
    // no `return` added later leaves it lifted.
    with_updating(state, widgets, |w| {
        sync_radius_pair(
            &w.halftone.halftone_min_radius,
            &w.halftone.halftone_max_dot,
            &w.halftone.halftone_min_slider,
            &w.halftone.halftone_max_slider,
            min,
            max,
        );
    });
}

/// The two radii are bounded by each other: after a change, the constrained
/// line must be realigned on the retained value.
fn sync_radii(state: &St, widgets: &Wi) {
    if state.borrow().updating {
        return;
    }
    let (min, max) = {
        let st = state.borrow();
        (st.params.min_radius_ratio, st.params.max_radius_ratio)
    };
    with_updating(state, widgets, |w| {
        sync_radius_pair(
            &w.algorithm.min_radius,
            &w.algorithm.max_radius,
            &w.algorithm.min_radius_slider,
            &w.algorithm.max_radius_slider,
            min,
            max,
        );
    });
}

/// Writes a pair (entry line, slider, crossed bound) on both sides.
///
/// The two twin blocks — dot radii and screen radii — differ only by the
/// widgets. The borrow dance (`borrow_mut` in two steps on both sides) is the
/// fragile part: `updating` must stay synchronized during the whole update.
fn sync_radius_pair(
    min_row: &adw::SpinRow,
    max_row: &adw::SpinRow,
    min_slider: &crate::controls::ValueSlider,
    max_slider: &crate::controls::ValueSlider,
    min: f32,
    max: f32,
) {
    min_row.adjustment().set_value(f64::from(min));
    max_row.adjustment().set_value(f64::from(max));
    min_slider.set_value(f64::from(min));
    max_slider.set_value(f64::from(max));
    // The crossed bound: each handle cannot go past the other.
    min_slider.set_bound(f64::from(max));
    max_slider.set_bound(f64::from(min));
}

// ─── Drag and drop ───────────────────────────────────────────────────────────

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
        // Every file is handled, and the gesture is accepted as soon as the
        // list contains at least one useful item: the `return true` stays out
        // of the loop, otherwise only the first file would be loaded.
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

// ─── Keyboard shortcuts ──────────────────────────────────────────────────────

pub(crate) fn install_shortcuts(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    let controller = gtk::EventControllerKey::new();
    let key_state = Rc::clone(state);
    let key_widgets = Rc::clone(widgets);
    let key_window = window.clone();
    controller.connect_key_pressed(move |_, keyval, _, modifiers| {
        use gtk::gdk::{Key, ModifierType};
        let ctrl = modifiers.intersects(ModifierType::CONTROL_MASK | ModifierType::META_MASK);
        // As long as an entry field has the focus, the "letter" shortcuts
        // risk being typed into the field: they are let through.
        let editing =
            key_widgets.render.seed.has_focus() || key_widgets.render.bg_button.has_focus();
        if ctrl {
            if keyval == Key::c && !editing {
                actions::copy_to_clipboard(&key_state, &key_widgets);
                return glib::Propagation::Stop;
            }
            return glib::Propagation::Proceed;
        }
        match keyval {
            // `editing` already guards the "letter" shortcuts: space is the
            // only character an entry field can receive in the middle of a
            // word (a colour "#ff cc00", a file name). Without the guard,
            // typing a space in "Couleur du fond" triggered a recompute.
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

    // Wheel with Ctrl: zoom (the zoom keeps the center of the viewport).
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
