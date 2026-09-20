//! État applicatif de haut niveau : chargement, historique undo/redo.

use eframe::egui;
use image::ImageReader;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::App;
use super::format::describe_parameter_change;
use super::io;
use pointimg::filter::{self, FilterParams};
use pointimg::frontend;

// ─── Types partagés ───────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy, Debug)]
pub(crate) enum ViewMode {
    Side,       // source | résultat côte à côte
    ResultOnly, // résultat seul
    SourceOnly, // source seule
    DensityMap, // aperçu de la density map
}

pub(crate) struct HistoryEntry {
    pub params: FilterParams,
    pub label: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            params: FilterParams::default(),
            src_dynamic: None,
            src_rgb: None,
            src_path: None,
            src_texture: None,
            result: Default::default(),
            result_texture: None,
            result_revision: Default::default(),
            result_texture_revision: 0,
            last_dots: Default::default(),
            density_image: None,
            density_texture: None,
            density_result: Default::default(),
            density_data: Default::default(),
            progress: Default::default(),
            computing: Default::default(),
            cancel: Default::default(),
            compute_generation: Default::default(),
            density_generation: Default::default(),
            compute_error: Default::default(),
            last_compute_ms: None,
            compute_start: None,
            last_param_change: None,
            history: Vec::new(),
            future: Vec::new(),
            last_committed: None,
            pending_commit: false,
            view_mode: ViewMode::Side,
            zoom: 1.0,
            zoom_fit: true,
            status: "Ouvrez une image pour commencer (ou faites glisser un fichier).".to_string(),
        }
    }
}

impl App {
    pub(crate) fn trigger_compute(&mut self, ctx: &egui::Context) {
        if self.computing.load(Ordering::Relaxed) {
            // Annuler le calcul en cours, puis relancer
            self.cancel.store(true, Ordering::Relaxed);
            // The cached dots belong to the previous parameter set and must not
            // be offered for SVG export while the replacement is pending.
            *self.last_dots.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return;
        }
        self.start_compute(ctx);
    }

    /// Commit d'un nouvel état dans l'historique (undo stack). Le redo stack est
    /// remis à zéro. On évite de pousser des entrées consécutives identiques
    /// (ex: drag sans fin du même slider).
    pub(crate) fn commit_history(&mut self) {
        let cur = self.params.clone();
        if let Some(last) = &self.last_committed
            && last == &cur
        {
            return; // pas de changement réel, on ignore
        }
        // Plafond à 50 entrées (FIFO).
        if self.history.len() >= 50 {
            self.history.remove(0);
        }
        if let Some(p) = self.last_committed.take() {
            let label = describe_parameter_change(&p, &cur);
            self.history.push(HistoryEntry { params: p, label });
        }
        self.last_committed = Some(cur);
        self.future.clear();
    }

    /// `Ctrl+Z` : restore le FilterParams précédent.
    pub(crate) fn undo(&mut self, ctx: &egui::Context) {
        // Commit d'abord tout état pending pour qu'il figure dans l'historique
        // avant qu'on ne rembobine (sinon le dernier réglage n'est pas undo-able).
        if self.pending_commit {
            self.commit_history();
            self.pending_commit = false;
        }
        let Some(prev) = self.history.pop() else {
            self.status = "Rien à annuler.".to_string();
            return;
        };
        // L'état live actuel devient la tête du redo.
        if let Some(prev_committed) = self.last_committed.take() {
            self.future.push(HistoryEntry {
                params: prev_committed,
                label: "État annulé".to_string(),
            });
        }
        self.last_committed = Some(prev.params.clone());
        self.params = prev.params;
        // Annuler le compute en cours (peut venir d'un drag) et relancer.
        if self.computing.load(Ordering::Relaxed) {
            self.cancel.store(true, Ordering::Relaxed);
            // La relance sera déclenchée par la logique de cancel-await, via le
            // même mécanisme qui détecte params_changed. Sinon, on debounce.
        }
        self.last_param_change = Some(Instant::now()); // trigger compute via debounce
        self.status = "Annulé.".to_string();
        ctx.request_repaint();
    }

    /// `Ctrl+Y` (ou `Ctrl+Shift+Z`) : restore l'état suivant.
    pub(crate) fn redo(&mut self, ctx: &egui::Context) {
        let Some(next) = self.future.pop() else {
            self.status = "Rien à refaire.".to_string();
            return;
        };
        if let Some(prev_committed) = self.last_committed.take() {
            self.history.push(HistoryEntry {
                params: prev_committed,
                label: "État rétabli".to_string(),
            });
        }
        self.last_committed = Some(next.params.clone());
        self.params = next.params;
        if self.computing.load(Ordering::Relaxed) {
            self.cancel.store(true, Ordering::Relaxed);
        }
        self.last_param_change = Some(Instant::now());
        self.status = "Refait.".to_string();
        ctx.request_repaint();
    }

    pub(crate) fn load_image(&mut self, path: PathBuf, ctx: &egui::Context) {
        let dimensions = match ImageReader::open(&path) {
            Ok(reader) => match reader.into_dimensions() {
                Ok(dimensions) => dimensions,
                Err(e) => {
                    self.status = format!("Erreur lecture dimensions : {e}");
                    return;
                }
            },
            Err(e) => {
                self.status = format!("Erreur lecture dimensions : {e}");
                return;
            }
        };
        if dimensions.0 == 0 || dimensions.1 == 0 {
            self.status = format!("Image vide ({}x{})", dimensions.0, dimensions.1);
            return;
        }
        match pointimg::color::decode_to_srgb(&path, "auto") {
            Ok((img, profile_converted, was_resized)) => {
                let original_dimensions = dimensions;
                // Invalidate workers before replacing the source. An old worker
                // may still be unwinding, but it must not publish its result.
                self.compute_generation.fetch_add(1, Ordering::AcqRel);
                self.cancel.store(true, Ordering::Release);
                // Use the shared alpha-compositing implementation.
                let rgb = filter::flatten_to_rgb(&img, self.params.bg_color);
                self.density_image = None;
                self.density_texture = None;
                self.src_dynamic = Some(img);
                self.src_rgb = Some(rgb);
                self.src_path = Some(path.clone());
                self.src_texture = None;
                self.result_texture = None;
                *self.result.lock().unwrap_or_else(|e| e.into_inner()) = None;
                self.result_revision.fetch_add(1, Ordering::Release);
                *self.last_dots.lock().unwrap_or_else(|e| e.into_inner()) = None;
                self.status = if was_resized {
                    format!(
                        "Image chargée et réduite de {}x{} à {}x{} pour respecter la mémoire.",
                        original_dimensions.0,
                        original_dimensions.1,
                        self.src_rgb.as_ref().map_or(0, |image| image.width()),
                        self.src_rgb.as_ref().map_or(0, |image| image.height())
                    )
                } else if profile_converted {
                    "Profil ICC converti vers sRGB.".to_string()
                } else {
                    format!("Image chargée : {}", path.display())
                };
                self.start_density_compute(ctx);
                self.start_compute(ctx);
            }
            Err(e) => {
                self.status = format!("Erreur chargement : {e}");
            }
        }
    }

    pub(crate) fn load_preset(&mut self, path: PathBuf, ctx: &egui::Context) {
        match frontend::load_preset(&path) {
            Ok(params) => {
                self.params = params;
                self.refresh_src_rgb(ctx);
                self.last_param_change = Some(Instant::now());
                self.status = format!("Preset chargé : {}", path.display());
            }
            Err(e) => self.status = format!("Erreur chargement preset : {e}"),
        }
    }

    pub(crate) fn save_preset(&mut self, path: PathBuf) {
        let path = frontend::ensure_toml_extension(path);
        if !io::confirm_overwrite(&path) {
            self.status = "Sauvegarde annulée.".to_string();
            return;
        }
        match frontend::save_preset(&path, &self.params) {
            Ok(()) => self.status = format!("Preset sauvegardé : {}", path.display()),
            Err(e) => self.status = format!("Erreur sauvegarde preset : {e}"),
        }
    }

    /// Reconstruit src_rgb depuis src_dynamic si bg_color a changé.
    pub(crate) fn refresh_src_rgb(&mut self, ctx: &egui::Context) {
        if let Some(img) = &self.src_dynamic {
            // The shared alpha-compositing implementation.
            let new_rgb = filter::flatten_to_rgb(img, self.params.bg_color);
            self.density_image = None;
            self.density_texture = None;
            self.src_rgb = Some(new_rgb);
            self.src_texture = None;
            self.start_density_compute(ctx);
        }
    }
}
