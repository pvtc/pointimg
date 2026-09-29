//! Logique applicative : workers de calcul, chargement, undo/redo, presets.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use gtk::glib;
use gtk::prelude::*;

use image::{RgbImage, RgbaImage};
use pointimg::filter::{self, Algorithm, Dot, FilterParams};
use pointimg::frontend;
pub(crate) use pointimg::frontend::{
    IMAGE_EXTENSIONS as SUPPORTED_IMAGE_EXTENSIONS, is_supported_image as supported_image,
};

use crate::controls::{ResetTarget, Widgets};
use crate::preview;
use crate::sections;
use crate::{Msg, State};

pub(crate) use crate::files::{
    choose_load_preset, choose_open_folder, choose_open_image, choose_save_image,
    choose_save_preset, choose_save_svg,
};

/// Délai avant de repasser en pleine résolution quand l'utilisateur arrête de
/// manipuler les réglages.
const FINALIZE_DELAY: Duration = Duration::from_millis(900);

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

/// Recalcule immédiatement en pleine résolution (bouton Recalculer, Espace).
///
/// Le mode choisi par l'utilisateur n'est pas modifié : c'est le calcul qui
/// passe en pleine résolution, comme le fait la finalisation automatique.
pub(crate) fn request_full_compute(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    {
        let mut st = state.borrow_mut();
        st.final_pass = true;
        if let Some(id) = st.finalize_pending.take() {
            id.remove();
        }
    }
    request_compute(state, widgets);
}

/// Annulation par l'utilisateur.
pub(crate) fn cancel_compute(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    if !state.borrow().computing {
        return;
    }
    let mut st = state.borrow_mut();
    st.user_cancelled = true;
    st.status = "Annulation…".to_string();
    drop(st);
    state.borrow().cancel.store(true, Ordering::SeqCst);
    preview::update_status_labels(state, widgets);
}

pub(crate) fn start_compute(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (src, params, density, cancel, generation, tx, full) = {
        let mut st = state.borrow_mut();
        let full = st.computes_full_resolution();
        let Some(src) = st.compute_source() else {
            return;
        };
        // La density map n'est valable que pour la résolution où elle a été
        // calculée : un aperçu à 1/2 suivi d'une finale pleine résolution
        // doit recalculer la sienne.
        let scale = if st.final_pass {
            1.0
        } else {
            st.quality.scale()
        };
        let density = if (st.density_scale - scale).abs() < f32::EPSILON {
            st.density_data.clone()
        } else {
            None
        };
        let generation = st.generation.fetch_add(1, Ordering::AcqRel) + 1;
        st.computing = true;
        st.user_cancelled = false;
        st.restart_pending = false;
        if full {
            st.result_is_final = false;
        }
        st.compute_start = Some(Instant::now());
        st.status = if full {
            "Calcul en cours…".to_string()
        } else {
            "Aperçu rapide en cours…".to_string()
        };
        st.last_dots = None;
        (
            src,
            st.params.clone(),
            density,
            Arc::clone(&st.cancel),
            generation,
            st.tx.clone(),
            full,
        )
    };
    cancel.store(false, Ordering::SeqCst);

    std::thread::spawn(move || run_worker(src, params, density, cancel, generation, tx, full));

    preview::refresh_actions(state, widgets);
    preview::update_status_labels(state, widgets);
    preview::refresh_compare(state, widgets);
}

/// Programme la finalisation en pleine résolution après une courte période
/// d'inactivité.
fn schedule_finalize(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    if state.borrow().computes_full_resolution() {
        return;
    }
    if let Some(id) = state.borrow_mut().finalize_pending.take() {
        id.remove();
    }
    let timer_state = Rc::clone(state);
    let timer_widgets = Rc::clone(widgets);
    let id = glib::timeout_add_local_once(FINALIZE_DELAY, move || {
        timer_state.borrow_mut().finalize_pending = None;
        request_full_compute(&timer_state, &timer_widgets);
    });
    state.borrow_mut().finalize_pending = Some(id);
}

/// Recalcule la density map dans un thread séparé.
pub(crate) fn start_density(state: &Rc<RefCell<State>>) {
    let Some(src) = state.borrow().compute_source() else {
        return;
    };
    let sensitivity = state.borrow().params.variance_sensitivity;
    let (generation, tx, scale) = {
        let mut st = state.borrow_mut();
        let generation = st.density_generation.fetch_add(1, Ordering::AcqRel) + 1;
        st.density_image = None;
        st.density_data = None;
        st.density_scale = 0.0;
        st.density_texture = None;
        (generation, st.tx.clone(), st.quality.scale())
    };
    std::thread::spawn(move || {
        let (w, h) = src.dimensions();
        let data = filter::compute_density_map(&src, sensitivity);
        let image = filter::density_to_image(&data, w, h);
        let _ = tx.send_blocking(Msg::Density {
            generation,
            image,
            data: Arc::new(data),
            scale,
        });
    });
}

/// Charge une image depuis le disque.
pub(crate) fn load_image(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    // Si l'image appartient au dossier en cours, on se place dessus.
    if let Some(index) = state
        .borrow()
        .folder_images
        .iter()
        .position(|candidate| candidate == &path)
    {
        state.borrow_mut().folder_index = Some(index);
    }

    match pointimg::color::decode_to_srgb(&path, "auto") {
        Ok((img, profile_converted, was_resized)) => {
            let (orig_w, orig_h) = (img.width(), img.height());
            {
                let mut st = state.borrow_mut();
                st.generation.fetch_add(1, Ordering::AcqRel);
                st.cancel.store(true, Ordering::SeqCst);
                st.computing = false;
                st.restart_pending = false;
                st.user_cancelled = false;
                if let Some(id) = st.finalize_pending.take() {
                    id.remove();
                }
                let rgb = filter::flatten_to_rgb(&img, st.params.bg_color);
                let dims = rgb.dimensions();
                st.src_dynamic = Some(img);
                st.src_rgb = Some(rgb);
                st.src_path = Some(path.clone());
                // Taille du fichier d'origine : référence de la taille d'export.
                st.source_size = std::fs::metadata(&path).ok().map(|meta| meta.len());
                st.export_sizes = None;
                st.src_texture = None;
                st.result = None;
                st.result_texture = None;
                st.result_is_final = true;
                st.last_dots = None;
                st.density_image = None;
                st.density_data = None;
                st.density_texture = None;
                st.history.clear();
                st.future.clear();
                st.last_committed = Some(st.params.clone());
                st.source_was_resized = was_resized;
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
            }
            preview::refresh_preview(state, widgets);
            preview::refresh_actions(state, widgets);
            preview::update_status_labels(state, widgets);
            start_density(state);
            start_compute(state, widgets);
        }
        Err(e) => {
            preview::notify_error(state, widgets, &format!("Erreur de chargement : {e}"));
        }
    }
}

/// Ouvre un dossier : charge la première image et active la navigation.
pub(crate) fn open_folder(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let Ok(entries) = std::fs::read_dir(&path) else {
        preview::notify_error(
            state,
            widgets,
            &format!("Dossier illisible : {}", path.display()),
        );
        return;
    };
    let mut images: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| supported_image(path))
        .collect();
    images.sort();
    if images.is_empty() {
        preview::notify_error(
            state,
            widgets,
            &format!("Aucune image dans {}", path.display()),
        );
        return;
    }
    {
        let mut st = state.borrow_mut();
        st.last_dir = Some(path);
        st.folder_images = images;
        st.folder_index = Some(0);
    }
    let first = state.borrow().folder_images[0].clone();
    load_image(first, state, widgets);
}

/// Image suivante / précédente dans le dossier courant.
pub(crate) fn step_folder(delta: i32, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let target = {
        let st = state.borrow();
        let (Some(index), len) = (st.folder_index, st.folder_images.len()) else {
            return;
        };
        if len < 2 {
            return;
        }
        let next = (index as i32 + delta).rem_euclid(len as i32) as usize;
        st.folder_images[next].clone()
    };
    load_image(target, state, widgets);
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
    preview::refresh_compare(state, widgets);
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
        preview::notify(state, widgets, "Rien à annuler.");
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
    preview::notify(state, widgets, "Annulé.");
}

pub(crate) fn redo(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let next = state.borrow_mut().future.pop();
    let Some(next) = next else {
        preview::notify(state, widgets, "Rien à rétablir.");
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
    preview::notify(state, widgets, "Rétabli.");
}

/// Rétablit les valeurs par défaut d'une section, **sans toucher aux autres** :
/// on part des réglages courants et on ne remplace que les champs de la
/// section visée.
fn with_section_defaults(current: &FilterParams, target: ResetTarget) -> FilterParams {
    let default = FilterParams::default();
    let mut params = current.clone();
    match target {
        ResetTarget::All | ResetTarget::Presets => return default,
        ResetTarget::Algorithm => {
            params.algorithm = default.algorithm;
            params.num_points = default.num_points;
            params.cols = default.cols;
            params.min_radius_ratio = default.min_radius_ratio;
            params.max_radius_ratio = default.max_radius_ratio;
            params.iterations = default.iterations;
            params.variance_sensitivity = default.variance_sensitivity;
            params.max_boost = default.max_boost;
            params.grid_angle_deg = default.grid_angle_deg;
            params.halftone = default.halftone;
            params.screening = default.screening;
            params.halftone_frequency = default.halftone_frequency;
            params.halftone_min_radius_ratio = default.halftone_min_radius_ratio;
            params.halftone_max_dot_ratio = default.halftone_max_dot_ratio;
        }
        ResetTarget::Shape => {
            params.dot_shape = default.dot_shape;
            params.palette_size = default.palette_size;
            params.dithering = default.dithering;
        }
        ResetTarget::Render => {
            params.bg_color = default.bg_color;
            params.transparent = default.transparent;
            params.gamma_correct = default.gamma_correct;
            params.rng_seed = default.rng_seed;
        }
    }
    params
}

/// Rétablit les réglages par défaut. Passe par l'historique pour rester
/// annulable (Ctrl+Z).
pub(crate) fn reset_params(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    if state.borrow().params == FilterParams::default() {
        preview::notify(state, widgets, "Réglages déjà par défaut.");
        return;
    }
    {
        let mut st = state.borrow_mut();
        st.params = FilterParams::default();
        st.pending_commit = false;
        commit_history(&mut st);
    }
    sections::sync_widgets(state, widgets);
    sections::update_sections(widgets, &state.borrow().params);
    refresh_src_rgb(state, widgets);
    request_compute(state, widgets);
    preview::notify(state, widgets, "Réglages réinitialisés.");
}

/// Rétablit une seule section (bouton ↺ dans l'en-tête du groupe).
pub(crate) fn reset_section(
    target: ResetTarget,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    if target == ResetTarget::All {
        reset_params(state, widgets);
        return;
    }
    let defaults = with_section_defaults(&state.borrow().params, target);
    let before = state.borrow().params.clone();
    if before == defaults {
        preview::notify(state, widgets, "Section déjà par défaut.");
        return;
    }
    {
        let mut st = state.borrow_mut();
        st.params = defaults.clone();
        st.pending_commit = false;
        commit_history(&mut st);
    }
    // Le preset affiché n'est plus « personnalisé » seulement si la section
    // qui le définit a changé : on repasse à l'entrée neutre.
    sections::sync_widgets(state, widgets);
    sections::update_sections(widgets, &state.borrow().params);
    refresh_src_rgb(state, widgets);
    request_compute(state, widgets);
    preview::notify(state, widgets, "Section réinitialisée.");
}

/// Applique un preset intégré (annulable, comme n'importe quel changement).
pub(crate) fn apply_preset(index: usize, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let presets = filter::builtin_presets();
    let Some(preset) = presets.get(index) else {
        return;
    };
    let params = preset.params.clone();
    {
        let mut st = state.borrow_mut();
        if st.params == params {
            return;
        }
        st.params = params;
        st.pending_commit = false;
        commit_history(&mut st);
    }
    sections::sync_widgets(state, widgets);
    sections::update_sections(widgets, &state.borrow().params);
    refresh_src_rgb(state, widgets);
    request_compute(state, widgets);
    preview::notify(
        state,
        widgets,
        &format!("Preset « {} » appliqué.", preset.name),
    );
}

/// Applique les paramètres d'un preset chargé depuis un fichier.
pub(crate) fn apply_preset_params(
    params: FilterParams,
    label: &str,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    {
        let mut st = state.borrow_mut();
        st.params = params;
        st.pending_commit = false;
        commit_history(&mut st);
    }
    sections::sync_widgets(state, widgets);
    sections::update_sections(widgets, &state.borrow().params);
    refresh_src_rgb(state, widgets);
    request_compute(state, widgets);
    preview::notify(state, widgets, label);
}

/// Tire une nouvelle graine aléatoire et recalcule.
pub(crate) fn vary_seed(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let seed = {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        );
        hasher.finish()
    };
    {
        let mut st = state.borrow_mut();
        st.params.rng_seed = Some(seed);
        st.pending_commit = true;
    }
    sections::sync_widgets(state, widgets);
    on_params_changed(state, widgets);
}

/// Copie le résultat courant dans le presse-papiers.
pub(crate) fn copy_to_clipboard(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let Some(result) = state.borrow().result.clone() else {
        preview::notify(state, widgets, "Aucun résultat à copier.");
        return;
    };
    let texture = crate::preview::rgba_to_texture(&result);
    // Le presse-papiers est un service de l'affichage : on passe par la
    // fenêtre pour être sûr d'être dans le bon display.
    widgets.toast_overlay.clipboard().set_texture(&texture);
    preview::notify(state, widgets, "Image copiée dans le presse-papiers.");
}

// ─── Messages des workers ────────────────────────────────────────────────────

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
            state.borrow_mut().result = Some(Arc::new(image));
            state.borrow_mut().result_is_final = false;
            preview::refresh_result_texture(state, widgets);
            preview::apply_zoom(state, widgets);
            preview::refresh_compare(state, widgets);
        }
        Msg::Done {
            generation,
            image,
            dots,
            elapsed_ms,
            full_resolution,
        } => {
            if !is_current(state, generation) {
                return;
            }
            {
                let mut st = state.borrow_mut();
                st.result = Some(Arc::new(image));
                st.last_dots = Some(dots);
                st.computing = false;
                st.compute_start = None;
                st.result_is_final = full_resolution;
                st.final_pass = false;
                st.last_compute_ms = Some(elapsed_ms);
                st.result_revision += 1;
                st.status = if full_resolution {
                    "Calcul terminé.".to_string()
                } else {
                    "Aperçu calculé — finalisation en cours…".to_string()
                };
                if st.pending_commit {
                    commit_history(&mut st);
                    st.pending_commit = false;
                }
            }
            preview::refresh_result_texture(state, widgets);
            preview::apply_zoom(state, widgets);
            preview::refresh_compare(state, widgets);
            preview::refresh_actions(state, widgets);
            // Taille d'export : uniquement sur le rendu définitif. Le worker
            // publiera son résultat par le canal.
            let tx = state.borrow().tx.clone();
            crate::exports::refresh(state, &tx);
            preview::update_status_labels(state, widgets);
            maybe_restart(state, widgets);
            if !full_resolution {
                schedule_finalize(state, widgets);
            }
        }
        Msg::Failed {
            generation,
            message,
            cancelled,
        } => {
            if !is_current(state, generation) {
                return;
            }
            let mut error = None;
            {
                let mut st = state.borrow_mut();
                st.computing = false;
                st.compute_start = None;
                st.final_pass = false;
                if cancelled && st.user_cancelled {
                    st.status = "Calcul annulé.".to_string();
                } else if !cancelled {
                    st.status = message.clone();
                    error = Some(message);
                }
                st.user_cancelled = false;
            }
            preview::refresh_actions(state, widgets);
            preview::update_status_labels(state, widgets);
            if let Some(error) = error {
                preview::notify_error(state, widgets, &error);
            }
            maybe_restart(state, widgets);
        }
        Msg::Density {
            generation,
            image,
            data,
            scale,
        } => {
            if state.borrow().density_generation.load(Ordering::Acquire) != generation {
                return;
            }
            {
                let mut st = state.borrow_mut();
                st.density_image = Some(image);
                st.density_data = Some(data);
                st.density_scale = scale;
                st.density_texture = None;
            }
            preview::refresh_density_texture(state, widgets);
            preview::apply_zoom(state, widgets);
        }
        Msg::ExportSizes { generation, sizes } => {
            if state
                .borrow()
                .export_sizes_generation
                .load(Ordering::Acquire)
                != generation
            {
                return;
            }
            {
                let mut st = state.borrow_mut();
                st.export_sizes = Some(sizes);
                st.export_sizes_computing.store(false, Ordering::Release);
            }
            preview::update_status_labels(state, widgets);
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
    full_resolution: bool,
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
                        full_resolution,
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
                full_resolution,
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

/// Export SVG des dots courants (utilisé par le menu et les raccourcis).
pub(crate) fn save_svg_quick(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let Some(path) = state.borrow().src_path.as_ref().map(|p| {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("pointimg");
        p.with_file_name(format!("{stem}-pointimg.svg"))
    }) else {
        preview::notify(state, widgets, "Ouvrez d'abord une image.");
        return;
    };
    crate::files::save_svg(path, state, widgets);
}

#[allow(dead_code)]
fn _assert_types(_: &[Dot], _: &RgbaImage, _: &FilterParams) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Les réinitialisations partielles ne doivent toucher qu'à leur section.
    #[test]
    fn reset_targets_are_scoped() {
        let default = FilterParams::default();
        let tuned = FilterParams {
            algorithm: pointimg::filter::Algorithm::Grid,
            num_points: 4242,
            dot_shape: pointimg::filter::DotShape::Square,
            palette_size: Some(5),
            bg_color: [1, 2, 3],
            gamma_correct: true,
            rng_seed: Some(7),
            transparent: true,
            ..FilterParams::default()
        };

        let algorithm = with_section_defaults(&tuned, ResetTarget::Algorithm);
        assert_eq!(algorithm.num_points, default.num_points);
        assert_eq!(
            algorithm.dot_shape, tuned.dot_shape,
            "la forme est conservée"
        );
        assert_eq!(algorithm.bg_color, tuned.bg_color, "le fond est conservé");

        let shape = with_section_defaults(&tuned, ResetTarget::Shape);
        assert_eq!(shape.dot_shape, default.dot_shape);
        assert_eq!(shape.palette_size, default.palette_size);
        assert_eq!(
            shape.num_points, tuned.num_points,
            "la densité est conservée"
        );

        let render = with_section_defaults(&tuned, ResetTarget::Render);
        assert_eq!(render.bg_color, default.bg_color);
        assert_eq!(render.gamma_correct, default.gamma_correct);
        assert_eq!(render.rng_seed, default.rng_seed);
        assert_eq!(
            render.algorithm, tuned.algorithm,
            "l'algorithme est conservé"
        );
    }

    /// Les presets intégrés doivent rester acceptables par le pipeline.
    #[test]
    fn halftone_presets_carry_a_mode() {
        for preset in filter::builtin_presets() {
            if preset.params.algorithm == pointimg::filter::Algorithm::Halftone {
                assert!(matches!(
                    preset.params.halftone,
                    filter::HalftoneMode::Cmyk { .. } | filter::HalftoneMode::Dominant { .. }
                ));
            }
        }
    }

    #[test]
    fn folder_navigation_wraps() {
        // `step_folder` ne fait rien sans dossier : on vérifie ici le calcul
        // d'index utilisé par la navigation.
        let len = 3usize;
        let index = 2usize;
        for delta in [1, -1] {
            let next = (index as i32 + delta).rem_euclid(len as i32) as usize;
            assert!(next < len);
        }
        assert_eq!((2i32 + 1).rem_euclid(3) as usize, 0);
        assert_eq!((0i32 - 1).rem_euclid(3) as usize, 2);
    }
}
