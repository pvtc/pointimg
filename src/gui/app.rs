//! État applicatif de haut niveau : chargement, historique undo/redo.

use eframe::egui;
use image::ImageReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::App;
use super::HistoryEntry;
use super::format::describe_parameter_change;
use super::io;
use pointimg::filter::{self, FilterParams};
use pointimg::frontend;
use pointimg::frontend::{IMAGE_EXTENSIONS, describe_image_size, is_supported_image};

impl App {
    // ─── Images ──────────────────────────────────────────────────────────────

    pub(crate) fn choose_image(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new().add_filter("Images", &IMAGE_EXTENSIONS);
        if let Some(folder) = self.initial_folder() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.pick_file() {
            self.load_image(path, ctx);
        }
    }

    pub(crate) fn choose_folder(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new();
        if let Some(folder) = self.initial_folder() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.pick_folder() {
            self.open_folder(path, ctx);
        }
    }

    /// Dossier proposé par défaut : celui de la dernière image ou export.
    fn initial_folder(&self) -> Option<PathBuf> {
        self.last_dir.clone().or_else(|| {
            self.src_path
                .as_ref()
                .and_then(|p| p.parent().map(Path::to_path_buf))
        })
    }

    /// Charge la première image d'un dossier et active la navigation `←`/`→`.
    pub(crate) fn open_folder(&mut self, path: PathBuf, ctx: &egui::Context) {
        let Ok(entries) = std::fs::read_dir(&path) else {
            self.status = format!("Dossier illisible : {}", path.display());
            return;
        };
        let mut images: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|candidate| is_supported_image(candidate))
            .collect();
        images.sort();
        if images.is_empty() {
            self.status = format!("Aucune image dans {}", path.display());
            return;
        }
        self.last_dir = Some(path);
        self.folder_images = images;
        self.folder_index = Some(0);
        let first = self.folder_images[0].clone();
        self.load_image(first, ctx);
    }

    /// Image suivante (`delta > 0`) ou précédente dans le dossier courant.
    pub(crate) fn step_folder(&mut self, delta: i32, ctx: &egui::Context) {
        let target = {
            let (Some(index), len) = (self.folder_index, self.folder_images.len()) else {
                return;
            };
            if len < 2 {
                return;
            }
            let next = (index as i32 + delta).rem_euclid(len as i32) as usize;
            self.folder_images[next].clone()
        };
        self.load_image(target, ctx);
    }

    pub(crate) fn folder_position(&self) -> Option<(usize, usize)> {
        self.folder_index
            .map(|index| (index + 1, self.folder_images.len()))
    }

    /// Description courte de l'image courante (« 1400×1050 · 1,5 Mpx »).
    pub(crate) fn image_info(&self) -> Option<String> {
        let src = self.src_rgb.as_ref()?;
        Some(describe_image_size(src.width(), src.height()))
    }

    pub(crate) fn load_image(&mut self, path: PathBuf, ctx: &egui::Context) {
        if let Some(index) = self
            .folder_images
            .iter()
            .position(|candidate| candidate == &path)
        {
            self.folder_index = Some(index);
        }
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
                self.finalize_at = None;
                self.final_pass = false;
                let rgb = filter::flatten_to_rgb(&img, self.params.bg_color);
                self.density_image = None;
                self.density_texture = None;
                self.src_dynamic = Some(img);
                self.src_rgb = Some(rgb);
                self.src_path = Some(path.clone());
                // Taille du fichier d'origine : c'est la référence contre
                // laquelle la taille de l'export est comparée.
                self.source_size = std::fs::metadata(&path).ok().map(|meta| meta.len());
                self.export_sizes = None;
                self.src_texture = None;
                self.result_texture = None;
                *self.result.lock().unwrap_or_else(|e| e.into_inner()) = None;
                self.result_revision.fetch_add(1, Ordering::Release);
                *self.last_dots.lock().unwrap_or_else(|e| e.into_inner()) = None;
                self.history.clear();
                self.future.clear();
                self.last_committed = Some(self.params.clone());
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

    // ─── Presets ─────────────────────────────────────────────────────────────

    /// Annule puis relance le calcul courant (le nouveau calcul prend le relais).
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

    /// Applique un preset intégré ( annulable comme n'importe quel changement).
    pub(crate) fn apply_preset(&mut self, index: usize, ctx: &egui::Context) {
        let Some(preset) = filter::builtin_presets().into_iter().nth(index) else {
            return;
        };
        if self.params == preset.params {
            return;
        }
        self.params = preset.params;
        self.commit_history();
        self.refresh_src_rgb(ctx);
        self.on_params_changed(ctx);
        self.status = format!("Preset « {} » appliqué.", preset.name);
    }

    /// Index du preset correspondant exactement aux réglages courants.
    pub(crate) fn preset_index(&self) -> Option<usize> {
        filter::builtin_presets()
            .iter()
            .position(|preset| preset.params == self.params)
    }

    pub(crate) fn cycle_preset(&mut self, delta: i32, ctx: &egui::Context) {
        let total = filter::builtin_presets().len() as i32;
        let current = self.preset_index().map_or(-1, |index| index as i32);
        let next = (current + delta).rem_euclid(total + 1);
        if next < 0 {
            return;
        }
        self.apply_preset(next as usize, ctx);
    }

    /// Tire une nouvelle graine aléatoire et relance le calcul.
    pub(crate) fn vary_seed(&mut self, ctx: &egui::Context) {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0),
        );
        self.params.rng_seed = Some(hasher.finish());
        self.on_params_changed(ctx);
    }

    // ─── Presets (fichiers) ──────────────────────────────────────────────────

    pub(crate) fn load_preset(&mut self, path: PathBuf, ctx: &egui::Context) {
        match frontend::load_preset(&path) {
            Ok(params) => {
                self.params = params;
                self.refresh_src_rgb(ctx);
                self.on_params_changed(ctx);
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

    // ─── Historique ──────────────────────────────────────────────────────────

    /// Commit d'un nouvel état dans l'historique (undo stack). Le redo stack est
    /// remis à zéro. On évite de pousser des entrées consécutives identiques
    /// (ex: drag sans fin du même slider).
    pub(crate) fn commit_history(&mut self) {
        let cur = self.params.clone();
        if let Some(last) = &self.last_committed
            && last == &cur
        {
            return;
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
        if self.pending_commit {
            self.commit_history();
            self.pending_commit = false;
        }
        let Some(prev) = self.history.pop() else {
            self.status = "Rien à annuler.".to_string();
            return;
        };
        if let Some(prev_committed) = self.last_committed.take() {
            self.future.push(HistoryEntry {
                params: prev_committed,
                label: "État annulé".to_string(),
            });
        }
        self.last_committed = Some(prev.params.clone());
        self.params = prev.params;
        if self.computing.load(Ordering::Relaxed) {
            self.cancel.store(true, Ordering::Relaxed);
        }
        self.last_param_change = Some(Instant::now());
        self.status = "Annulé.".to_string();
        ctx.request_repaint();
    }

    /// `Ctrl+Y` (ou `Ctrl+Shift+Z`) : restore l'état suivant.
    pub(crate) fn redo(&mut self, ctx: &egui::Context) {
        let Some(next) = self.future.pop() else {
            self.status = "Rien à rétablir.".to_string();
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
        self.status = "Rétabli.".to_string();
        ctx.request_repaint();
    }

    /// Rétablit tous les réglages par défaut. Passe par l'historique pour rester
    /// annulable (`Ctrl+Z`).
    pub(crate) fn reset_params(&mut self, ctx: &egui::Context) {
        if self.params == FilterParams::default() {
            self.status = "Réglages déjà par défaut.".to_string();
            return;
        }
        self.params = FilterParams::default();
        self.commit_history();
        // `bg_color` change : la source aplatie doit être reconstruite.
        self.refresh_src_rgb(ctx);
        self.on_params_changed(ctx);
        self.status = "Réglages réinitialisés.".to_string();
    }

    // ─── Zoom ────────────────────────────────────────────────────────────────

    pub(crate) fn current_zoom(&self) -> f32 {
        if self.zoom_fit {
            self.effective_zoom
        } else {
            self.zoom
        }
    }

    /// Change le zoom en sortant du mode « Ajuster ».
    pub(crate) fn set_zoom(&mut self, value: f32) {
        self.zoom = value.clamp(0.02, 16.0);
        self.zoom_fit = false;
    }

    // ─── Divers ──────────────────────────────────────────────────────────────

    /// Recompose `src_rgb` depuis `src_dynamic` si bg_color a changé.
    pub(crate) fn refresh_src_rgb(&mut self, ctx: &egui::Context) {
        if let Some(img) = &self.src_dynamic {
            let new_rgb = filter::flatten_to_rgb(img, self.params.bg_color);
            self.density_image = None;
            self.density_texture = None;
            self.src_rgb = Some(new_rgb);
            self.src_texture = None;
            self.start_density_compute(ctx);
        }
    }
}
