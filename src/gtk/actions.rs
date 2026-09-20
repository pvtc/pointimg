//! Logique applicative : workers de calcul, chargement, undo/redo.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use gtk::glib;

use image::RgbImage;
use pointimg::filter::{self, Algorithm, Dot, FilterParams};
use pointimg::frontend;

use crate::controls::Widgets;
use crate::preview;
use crate::sections;
use crate::{Msg, State};

pub(crate) use crate::files::{
    choose_load_preset, choose_open_image, choose_save_image, choose_save_preset, choose_save_svg,
};

/// Appelé à chaque modification de paramètre : planifie un recalcul après un
/// court délai (debounce). Si l'image n'est pas encore chargée, ne fait rien.
pub(crate) fn on_params_changed(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    if state.borrow().updating {
        return;
    }
    if state.borrow().src_rgb.is_none() {
        return;
    }
    if let Some(id) = state.borrow_mut().debounce.take() {
        id.remove();
    }
    state.borrow_mut().pending_commit = true;
    let timer_state = Rc::clone(state);
    let timer_widgets = Rc::clone(widgets);
    let id = glib::timeout_add_local_once(Duration::from_millis(250), move || {
        timer_state.borrow_mut().debounce = None;
        request_compute(&timer_state, &timer_widgets);
    });
    state.borrow_mut().debounce = Some(id);
}

/// Demande un calcul : annule celui en cours et le relance, ou démarre.
pub(crate) fn request_compute(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    if state.borrow().computing {
        state.borrow_mut().restart_pending = true;
        state.borrow().cancel.store(true, Ordering::SeqCst);
        return;
    }
    start_compute(state, widgets);
}

/// Annulation par l'utilisateur.
pub(crate) fn cancel_compute(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    if !state.borrow().computing {
        return;
    }
    state.borrow_mut().user_cancelled = true;
    state.borrow().cancel.store(true, Ordering::SeqCst);
    let mut st = state.borrow_mut();
    st.status = "Annulation…".to_string();
    drop(st);
    preview::update_status_labels(state, widgets);
}

pub(crate) fn start_compute(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (src, params, density, cancel, generation, tx) = {
        let mut st = state.borrow_mut();
        let Some(src) = st.src_rgb.clone() else {
            return;
        };
        let generation = st.generation.fetch_add(1, Ordering::AcqRel) + 1;
        st.computing = true;
        st.user_cancelled = false;
        st.restart_pending = false;
        st.compute_start = Some(Instant::now());
        st.status = "Calcul en cours…".to_string();
        st.last_dots = None;
        (
            src,
            st.params.clone(),
            st.density_data.clone(),
            Arc::clone(&st.cancel),
            generation,
            st.tx.clone(),
        )
    };
    cancel.store(false, Ordering::SeqCst);

    std::thread::spawn(move || run_worker(src, params, density, cancel, generation, tx));

    preview::refresh_actions(state, widgets);
    preview::update_status_labels(state, widgets);
}

/// Recalcule la density map dans un thread séparé.
pub(crate) fn start_density(state: &Rc<RefCell<State>>) {
    let Some(src) = state.borrow().src_rgb.clone() else {
        return;
    };
    let sensitivity = state.borrow().params.variance_sensitivity;
    let (generation, tx) = {
        let mut st = state.borrow_mut();
        let generation = st.density_generation.fetch_add(1, Ordering::AcqRel) + 1;
        st.density_image = None;
        st.density_data = None;
        st.density_texture = None;
        (generation, st.tx.clone())
    };
    std::thread::spawn(move || {
        let (w, h) = src.dimensions();
        let data = filter::compute_density_map(&src, sensitivity);
        let image = filter::density_to_image(&data, w, h);
        let _ = tx.send_blocking(Msg::Density {
            generation,
            image,
            data: Arc::new(data),
        });
    });
}

/// Charge une image depuis le disque.
pub(crate) fn load_image(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    match pointimg::color::decode_to_srgb(&path, "auto") {
        Ok((img, profile_converted, was_resized)) => {
            let (orig_w, orig_h) = (img.width(), img.height());
            let new_dims = {
                let mut st = state.borrow_mut();
                st.generation.fetch_add(1, Ordering::AcqRel);
                st.cancel.store(true, Ordering::SeqCst);
                st.computing = false;
                st.restart_pending = false;
                st.user_cancelled = false;
                let rgb = filter::flatten_to_rgb(&img, st.params.bg_color);
                let dims = rgb.dimensions();
                st.src_dynamic = Some(img);
                st.src_rgb = Some(rgb);
                st.src_path = Some(path.clone());
                st.src_texture = None;
                st.result = None;
                st.result_texture = None;
                st.result_texture_revision = u64::MAX;
                st.last_dots = None;
                st.density_image = None;
                st.density_data = None;
                st.density_texture = None;
                st.history.clear();
                st.future.clear();
                st.last_committed = Some(st.params.clone());
                st.status = if was_resized {
                    format!(
                        "Image réduite de {orig_w}x{orig_h} à {}x{} pour respecter la mémoire.",
                        dims.0, dims.1
                    )
                } else if profile_converted {
                    "Profil ICC converti vers sRGB.".to_string()
                } else {
                    format!("Image chargée : {}", path.display())
                };
                dims
            };
            let _ = new_dims;
            preview::refresh_preview(state, widgets);
            preview::refresh_actions(state, widgets);
            preview::update_status_labels(state, widgets);
            start_density(state);
            start_compute(state, widgets);
        }
        Err(e) => {
            state.borrow_mut().status = format!("Erreur chargement : {e}");
            preview::update_status_labels(state, widgets);
        }
    }
}

/// Recompose `src_rgb` depuis `src_dynamic` (utile quand la couleur de fond change).
pub(crate) fn refresh_src_rgb(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let Some(img) = state.borrow().src_dynamic.clone() else {
        return;
    };
    let rgb = filter::flatten_to_rgb(&img, state.borrow().params.bg_color);
    {
        let mut st = state.borrow_mut();
        st.src_rgb = Some(rgb);
        st.src_texture = None;
        st.density_image = None;
        st.density_data = None;
        st.density_texture = None;
    }
    preview::refresh_source_texture(state, widgets);
    preview::refresh_density_texture(state, widgets);
    preview::apply_zoom(state, widgets);
    start_density(state);
}

fn commit_history(st: &mut State) {
    let cur = st.params.clone();
    if let Some(last) = &st.last_committed
        && last == &cur
    {
        return;
    }
    if st.history.len() >= 50 {
        st.history.remove(0);
    }
    if let Some(previous) = st.last_committed.take() {
        st.history.push(previous);
    }
    st.last_committed = Some(cur);
    st.future.clear();
}

pub(crate) fn undo(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    {
        let mut st = state.borrow_mut();
        if st.pending_commit {
            commit_history(&mut st);
            st.pending_commit = false;
        }
    }
    let previous = state.borrow_mut().history.pop();
    let Some(previous) = previous else {
        state.borrow_mut().status = "Rien à annuler.".to_string();
        preview::update_status_labels(state, widgets);
        return;
    };
    {
        let mut st = state.borrow_mut();
        if let Some(current) = st.last_committed.take() {
            st.future.push(current);
        }
        st.last_committed = Some(previous.clone());
        st.params = previous;
    }
    sections::sync_widgets(state, widgets);
    sections::update_sections(widgets, &state.borrow().params);
    refresh_src_rgb(state, widgets);
    request_compute(state, widgets);
    state.borrow_mut().status = "Annulé.".to_string();
    preview::update_status_labels(state, widgets);
}

pub(crate) fn redo(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let next = state.borrow_mut().future.pop();
    let Some(next) = next else {
        state.borrow_mut().status = "Rien à refaire.".to_string();
        preview::update_status_labels(state, widgets);
        return;
    };
    {
        let mut st = state.borrow_mut();
        if let Some(current) = st.last_committed.take() {
            st.history.push(current);
        }
        st.last_committed = Some(next.clone());
        st.params = next;
    }
    sections::sync_widgets(state, widgets);
    sections::update_sections(widgets, &state.borrow().params);
    refresh_src_rgb(state, widgets);
    request_compute(state, widgets);
    state.borrow_mut().status = "Refait.".to_string();
    preview::update_status_labels(state, widgets);
}

/// Récupère un message d'un worker et met à jour l'interface.
pub(crate) fn handle_message(msg: Msg, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    match msg {
        Msg::Progress {
            generation,
            done,
            total,
        } => {
            if !is_current(state, generation) {
                return;
            }
            if total > 0 {
                widgets.progress.set_fraction(done as f64 / total as f64);
                widgets.progress.set_text(Some(&format!("{done}/{total}")));
            }
        }
        Msg::Preview { generation, image } => {
            if !is_current(state, generation) {
                return;
            }
            state.borrow_mut().result = Some(image);
            preview::refresh_result_texture(state, widgets);
            preview::apply_zoom(state, widgets);
        }
        Msg::Done {
            generation,
            image,
            dots,
            elapsed_ms,
        } => {
            if !is_current(state, generation) {
                return;
            }
            {
                let mut st = state.borrow_mut();
                st.result = Some(image);
                st.last_dots = Some(dots);
                st.computing = false;
                st.compute_start = None;
                st.last_compute_ms = Some(elapsed_ms);
                st.status = format!("Terminé en {}.", crate::format_duration(elapsed_ms));
                if st.pending_commit {
                    commit_history(&mut st);
                    st.pending_commit = false;
                }
            }
            preview::refresh_result_texture(state, widgets);
            preview::apply_zoom(state, widgets);
            preview::refresh_actions(state, widgets);
            preview::update_status_labels(state, widgets);
            maybe_restart(state, widgets);
        }
        Msg::Failed {
            generation,
            message,
            cancelled,
        } => {
            if !is_current(state, generation) {
                return;
            }
            {
                let mut st = state.borrow_mut();
                st.computing = false;
                st.compute_start = None;
                if cancelled && st.user_cancelled {
                    st.status = "Annulé.".to_string();
                } else if !cancelled {
                    st.status = message;
                }
                st.user_cancelled = false;
            }
            preview::refresh_actions(state, widgets);
            preview::update_status_labels(state, widgets);
            maybe_restart(state, widgets);
        }
        Msg::Density {
            generation,
            image,
            data,
        } => {
            if state.borrow().density_generation.load(Ordering::Acquire) != generation {
                return;
            }
            {
                let mut st = state.borrow_mut();
                st.density_image = Some(image);
                st.density_data = Some(data);
                st.density_texture = None;
            }
            preview::refresh_density_texture(state, widgets);
            preview::apply_zoom(state, widgets);
        }
    }
}

fn is_current(state: &Rc<RefCell<State>>, generation: u64) -> bool {
    state.borrow().generation.load(Ordering::Acquire) == generation
}

fn maybe_restart(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let restart = {
        let mut st = state.borrow_mut();
        if st.restart_pending {
            st.restart_pending = false;
            true
        } else {
            false
        }
    };
    if restart {
        start_compute(state, widgets);
    }
}

/// Corps du thread de calcul.
fn run_worker(
    src: RgbImage,
    params: FilterParams,
    density: Option<Arc<Vec<f32>>>,
    cancel: Arc<AtomicBool>,
    generation: u64,
    tx: async_channel::Sender<Msg>,
) {
    let started = Instant::now();
    let elapsed = || started.elapsed().as_millis() as u64;

    if params.transparent || params.algorithm == Algorithm::Halftone {
        match filter::apply_rgba(&src, &params) {
            Ok((image, dots)) => {
                let msg = if cancel.load(Ordering::SeqCst) {
                    Msg::Failed {
                        generation,
                        message: "Annulé".to_string(),
                        cancelled: true,
                    }
                } else {
                    Msg::Done {
                        generation,
                        image,
                        dots,
                        elapsed_ms: elapsed(),
                    }
                };
                let _ = tx.send_blocking(msg);
            }
            Err(e) => {
                let _ = tx.send_blocking(Msg::Failed {
                    generation,
                    message: format!("Erreur : {e}"),
                    cancelled: filter::is_cancelled(&e),
                });
            }
        }
        return;
    }

    let mut last_preview = Instant::now();
    let mut publish = |iter: usize, total: usize, preview: &RgbImage| {
        let _ = tx.send_blocking(Msg::Progress {
            generation,
            done: iter,
            total,
        });
        if iter == total || last_preview.elapsed() >= Duration::from_millis(100) {
            let _ = tx.send_blocking(Msg::Preview {
                generation,
                image: frontend::rgb_to_rgba_opaque(preview),
            });
            last_preview = Instant::now();
        }
    };

    let result = if let Some(density) = density.as_deref() {
        filter::apply_with_progress_cached(&src, &params, &cancel, density, &mut publish)
    } else {
        filter::apply_with_progress(&src, &params, &cancel, &mut publish)
    };

    match result {
        Ok((dst, dots)) => {
            let _ = tx.send_blocking(Msg::Done {
                generation,
                image: frontend::rgb_to_rgba_opaque(&dst),
                dots,
                elapsed_ms: elapsed(),
            });
        }
        Err(e) => {
            let _ = tx.send_blocking(Msg::Failed {
                generation,
                message: format!("Erreur : {e}"),
                cancelled: filter::is_cancelled(&e),
            });
        }
    }
}

#[allow(dead_code)]
fn _assert_types(_: &[Dot], _: &RgbImage, _: &FilterParams) {}
