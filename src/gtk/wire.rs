//! Câblage des signaux GTK vers l'état applicatif.

use std::cell::RefCell;
use std::rc::Rc;

use gtk::glib;
use gtk::prelude::*;

use pointimg::filter::{Algorithm, DotShape, FilterParams, HalftoneMode, Screening};

use crate::State;
use crate::actions;
use crate::controls::Widgets;
use crate::sections;

pub(crate) use crate::sections::{sync_widgets, update_sections};

type St = Rc<RefCell<State>>;
type Wi = Rc<Widgets>;

fn clone_state_widgets(state: &St, widgets: &Wi) -> (St, Wi) {
    (Rc::clone(state), Rc::clone(widgets))
}

fn on_scale(state: &St, widgets: &Wi, scale: &gtk::Scale, apply: fn(&mut FilterParams, f64)) {
    let (state, widgets) = clone_state_widgets(state, widgets);
    scale.connect_value_changed(move |s| {
        if state.borrow().updating {
            return;
        }
        {
            let mut st = state.borrow_mut();
            apply(&mut st.params, s.value());
        }
        sections::sync_widgets(&state, &widgets);
        actions::on_params_changed(&state, &widgets);
    });
}

fn on_spin(state: &St, widgets: &Wi, spin: &gtk::SpinButton, apply: fn(&mut FilterParams, f64)) {
    let (state, widgets) = clone_state_widgets(state, widgets);
    spin.connect_value_changed(move |s| {
        if state.borrow().updating {
            return;
        }
        {
            let mut st = state.borrow_mut();
            apply(&mut st.params, s.value());
        }
        sections::sync_widgets(&state, &widgets);
        actions::on_params_changed(&state, &widgets);
    });
}

fn on_switch(
    state: &St,
    widgets: &Wi,
    check: &gtk::CheckButton,
    apply: fn(&mut FilterParams, bool),
) {
    let (state, widgets) = clone_state_widgets(state, widgets);
    check.connect_toggled(move |c| {
        if state.borrow().updating {
            return;
        }
        {
            let mut st = state.borrow_mut();
            apply(&mut st.params, c.is_active());
        }
        sections::update_sections(&widgets, &state.borrow().params);
        actions::on_params_changed(&state, &widgets);
    });
}

fn on_select<F>(state: &St, widgets: &Wi, dropdown: &gtk::DropDown, f: F)
where
    F: Fn(u32, &St, &Wi) + 'static,
{
    let (state, widgets) = clone_state_widgets(state, widgets);
    dropdown.connect_selected_notify(move |d| {
        if state.borrow().updating {
            return;
        }
        f(d.selected(), &state, &widgets);
    });
}

fn on_click<F>(
    state: &St,
    widgets: &Wi,
    window: &adw::ApplicationWindow,
    button: &gtk::Button,
    f: F,
) where
    F: Fn(&St, &Wi, &adw::ApplicationWindow) + 'static,
{
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    let window = window.clone();
    button.connect_clicked(move |_| f(&state, &widgets, &window));
}

pub(crate) fn wire(state: &St, widgets: &Wi, window: &adw::ApplicationWindow) {
    on_click(state, widgets, window, &widgets.open, |s, w, win| {
        actions::choose_open_image(win, s, w)
    });
    on_click(state, widgets, window, &widgets.load_preset, |s, w, win| {
        actions::choose_load_preset(win, s, w)
    });
    on_click(state, widgets, window, &widgets.save_preset, |s, w, win| {
        actions::choose_save_preset(win, s, w)
    });
    on_click(state, widgets, window, &widgets.save_image, |s, w, win| {
        actions::choose_save_image(win, s, w)
    });
    on_click(state, widgets, window, &widgets.save_svg, |s, w, win| {
        actions::choose_save_svg(win, s, w)
    });

    // Algorithme
    on_select(state, widgets, &widgets.algo, |idx, state, widgets| {
        {
            let mut st = state.borrow_mut();
            st.params.algorithm = sections::algorithm_from_index(idx);
            if st.params.algorithm == Algorithm::Halftone && st.params.halftone == HalftoneMode::Off
            {
                st.params.halftone = HalftoneMode::Cmyk {
                    angles: [15.0, 75.0, 0.0, 45.0],
                };
            }
        }
        sections::sync_widgets(state, widgets);
        sections::update_sections(widgets, &state.borrow().params);
        actions::on_params_changed(state, widgets);
    });

    // Placement
    on_scale(state, widgets, &widgets.variance, |p, v| {
        p.variance_sensitivity = v as f32
    });
    {
        let (state, _widgets) = clone_state_widgets(state, widgets);
        widgets.variance.connect_value_changed(move |_| {
            if state.borrow().updating {
                return;
            }
            actions::start_density(&state);
        });
    }
    on_scale(state, widgets, &widgets.min_radius, |p, v| {
        p.min_radius_ratio = v as f32;
        if p.max_radius_ratio < p.min_radius_ratio {
            p.max_radius_ratio = p.min_radius_ratio;
        }
    });
    on_scale(state, widgets, &widgets.max_radius, |p, v| {
        p.max_radius_ratio = v as f32;
        if p.min_radius_ratio > p.max_radius_ratio {
            p.min_radius_ratio = p.max_radius_ratio;
        }
    });
    on_scale(state, widgets, &widgets.max_boost, |p, v| {
        p.max_boost = v as f32
    });
    on_scale(state, widgets, &widgets.cols, |p, v| {
        p.cols = v.round().max(1.0) as u32
    });
    on_scale(state, widgets, &widgets.grid_angle, |p, v| {
        p.grid_angle_deg = v as f32
    });
    on_scale(state, widgets, &widgets.num_points, |p, v| {
        p.num_points = v.round().max(1.0) as usize
    });
    on_scale(state, widgets, &widgets.iterations, |p, v| {
        p.iterations = v.round().max(1.0) as usize
    });

    // Forme
    on_select(state, widgets, &widgets.shape, |idx, state, widgets| {
        {
            let mut st = state.borrow_mut();
            st.params.dot_shape = match idx {
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
        sections::sync_widgets(state, widgets);
        sections::update_sections(widgets, &state.borrow().params);
        actions::on_params_changed(state, widgets);
    });
    on_scale(state, widgets, &widgets.ellipse_aspect, |p, v| {
        if let DotShape::Ellipse { aspect, .. } = &mut p.dot_shape {
            *aspect = v as f32;
        }
    });
    on_scale(state, widgets, &widgets.ellipse_angle, |p, v| {
        if let DotShape::Ellipse { angle_deg, .. } = &mut p.dot_shape {
            *angle_deg = v as f32;
        }
    });
    on_scale(state, widgets, &widgets.polygon_sides, |p, v| {
        if let DotShape::RegularPolygon { sides } = &mut p.dot_shape {
            *sides = v.round().clamp(3.0, 12.0) as u8;
        }
    });

    // Palette / seed
    on_switch(state, widgets, &widgets.use_palette, |p, on| {
        p.palette_size = if on { Some(8) } else { None };
    });
    on_scale(state, widgets, &widgets.palette_size, |p, v| {
        p.palette_size = Some(v.round().clamp(2.0, 256.0) as usize);
    });
    on_switch(state, widgets, &widgets.dithering, |p, on| {
        p.dithering = on;
    });
    on_switch(state, widgets, &widgets.use_seed, |p, on| {
        p.rng_seed = if on { Some(42) } else { None };
    });
    on_spin(state, widgets, &widgets.seed, |p, v| {
        p.rng_seed = Some(v.max(0.0) as u64);
    });

    // Fond / gamma
    {
        let (state, widgets) = clone_state_widgets(state, widgets);
        let button = widgets.bg_button.clone();
        button.connect_rgba_notify(move |b| {
            if state.borrow().updating {
                return;
            }
            let rgba = b.rgba();
            {
                let mut st = state.borrow_mut();
                st.params.bg_color = [
                    (rgba.red() * 255.0).round() as u8,
                    (rgba.green() * 255.0).round() as u8,
                    (rgba.blue() * 255.0).round() as u8,
                ];
                st.params.transparent = false;
            }
            state.borrow_mut().updating = true;
            widgets.transparent.set_active(false);
            state.borrow_mut().updating = false;
            actions::refresh_src_rgb(&state, &widgets);
            actions::on_params_changed(&state, &widgets);
        });
    }
    on_switch(state, widgets, &widgets.transparent, |p, on| {
        p.transparent = on;
    });
    on_switch(state, widgets, &widgets.gamma, |p, on| {
        p.gamma_correct = on;
    });

    // Halftone
    on_select(
        state,
        widgets,
        &widgets.halftone_mode,
        |idx, state, widgets| {
            state.borrow_mut().params.halftone = if idx == 1 {
                HalftoneMode::Dominant {
                    n: 5,
                    base_angle_deg: 15.0,
                }
            } else {
                HalftoneMode::Cmyk {
                    angles: [15.0, 75.0, 0.0, 45.0],
                }
            };
            sections::sync_widgets(state, widgets);
            sections::update_sections(widgets, &state.borrow().params);
            actions::on_params_changed(state, widgets);
        },
    );
    on_scale(state, widgets, &widgets.dominant_n, |p, v| {
        if let HalftoneMode::Dominant { n, .. } = &mut p.halftone {
            *n = v.round().clamp(2.0, 32.0) as usize;
        }
    });
    on_scale(state, widgets, &widgets.dominant_angle, |p, v| {
        if let HalftoneMode::Dominant { base_angle_deg, .. } = &mut p.halftone {
            *base_angle_deg = v as f32;
        }
    });
    on_select(state, widgets, &widgets.screening, |idx, state, widgets| {
        state.borrow_mut().params.screening = if idx == 1 {
            Screening::Fm
        } else {
            Screening::Am
        };
        actions::on_params_changed(state, widgets);
    });
    on_scale(state, widgets, &widgets.halftone_freq, |p, v| {
        p.halftone_frequency = v as f32
    });
    on_scale(state, widgets, &widgets.halftone_min_radius, |p, v| {
        p.halftone_min_radius_ratio = v as f32
    });
    on_scale(state, widgets, &widgets.halftone_max_dot, |p, v| {
        p.halftone_max_dot_ratio = v as f32
    });

    // Calcul
    on_click(state, widgets, window, &widgets.recalc, |s, w, _| {
        actions::request_compute(s, w)
    });
    on_click(state, widgets, window, &widgets.cancel, |s, w, _| {
        actions::cancel_compute(s, w)
    });
    on_click(state, widgets, window, &widgets.undo, |s, w, _| {
        actions::undo(s, w)
    });
    on_click(state, widgets, window, &widgets.redo, |s, w, _| {
        actions::redo(s, w)
    });

    // Affichage / zoom
    on_select(state, widgets, &widgets.view_mode, |idx, state, widgets| {
        state.borrow_mut().view_mode = sections::view_from_index(idx);
        crate::preview::apply_zoom(state, widgets);
    });
    {
        let (state, widgets) = clone_state_widgets(state, widgets);
        let zoom = widgets.zoom.clone();
        zoom.connect_value_changed(move |s| {
            if state.borrow().updating {
                return;
            }
            {
                let mut st = state.borrow_mut();
                st.zoom = s.value();
                st.zoom_fit = false;
            }
            state.borrow_mut().updating = true;
            widgets.zoom_fit.set_active(false);
            state.borrow_mut().updating = false;
            crate::preview::apply_zoom(&state, &widgets);
        });
    }
    {
        let (state, widgets) = clone_state_widgets(state, widgets);
        let zoom_fit = widgets.zoom_fit.clone();
        zoom_fit.connect_toggled(move |b| {
            if state.borrow().updating {
                return;
            }
            state.borrow_mut().zoom_fit = b.is_active();
            crate::preview::apply_zoom(&state, &widgets);
        });
    }
    {
        let (state, widgets) = clone_state_widgets(state, widgets);
        let zoom_one = widgets.zoom_one.clone();
        zoom_one.connect_clicked(move |_| {
            {
                let mut st = state.borrow_mut();
                st.zoom = 1.0;
                st.zoom_fit = false;
                st.updating = true;
            }
            // Les setters ci-dessous émettent des signaux synchrones dont les
            // handlers font `state.borrow()` : ne jamais garder le `borrow_mut`
            // ouvert au-delà de la modification d'état.
            widgets.zoom.set_value(1.0);
            widgets.zoom_fit.set_active(false);
            state.borrow_mut().updating = false;
            crate::preview::apply_zoom(&state, &widgets);
        });
    }
}

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
        let Some(file) = list.files().into_iter().next() else {
            return false;
        };
        let Some(path) = file.path() else {
            return false;
        };
        actions::load_image(path, &drop_state, &drop_widgets);
        true
    });
    window.add_controller(drop);

    let key = gtk::EventControllerKey::new();
    let (state, widgets) = clone_state_widgets(state, widgets);
    let window2 = window.clone();
    key.connect_key_pressed(move |_, keyval, _, modifiers| {
        use gtk::gdk::Key;
        let ctrl = modifiers
            .intersects(gtk::gdk::ModifierType::CONTROL_MASK | gtk::gdk::ModifierType::META_MASK);
        let shift = modifiers.contains(gtk::gdk::ModifierType::SHIFT_MASK);
        match keyval {
            Key::o if ctrl => {
                actions::choose_open_image(&window2, &state, &widgets);
                glib::Propagation::Stop
            }
            Key::s if ctrl => {
                actions::choose_save_image(&window2, &state, &widgets);
                glib::Propagation::Stop
            }
            Key::z if ctrl && shift => {
                actions::redo(&state, &widgets);
                glib::Propagation::Stop
            }
            Key::z if ctrl => {
                actions::undo(&state, &widgets);
                glib::Propagation::Stop
            }
            Key::y if ctrl => {
                actions::redo(&state, &widgets);
                glib::Propagation::Stop
            }
            Key::space if !ctrl => {
                actions::request_compute(&state, &widgets);
                glib::Propagation::Stop
            }
            _ => glib::Propagation::Proceed,
        }
    });
    window.add_controller(key);
}
