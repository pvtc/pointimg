//! High-level application state: loading, undo/redo history.

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
use pointimg::frontend::{IMAGE_EXTENSIONS, describe_image_size};

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

    /// Folder offered by default: that of the last image or export.
    fn initial_folder(&self) -> Option<PathBuf> {
        self.last_dir.clone().or_else(|| {
            self.src_path
                .as_ref()
                .and_then(|p| p.parent().map(Path::to_path_buf))
        })
    }

    /// Loads the first image of a folder and enables `←`/`→` navigation.
    pub(crate) fn open_folder(&mut self, path: PathBuf, ctx: &egui::Context) {
        // The sweep and its messages come from `frontend`, shared with GTK.
        let images = match frontend::collect_folder_images(&path) {
            Ok(images) => images,
            Err(e) => {
                self.notify_error(e.to_string());
                return;
            }
        };
        self.last_dir = Some(path);
        self.folder_images = images;
        self.folder_index = Some(0);
        let first = self.folder_images[0].clone();
        self.load_image(first, ctx);
    }

    /// Next (`delta > 0`) or previous image in the current folder.
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

    /// Short description of the current image ("1400×1050 · 1.5 Mpx").
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
                    self.notify_error(format!("Erreur lecture dimensions : {e}"));
                    return;
                }
            },
            Err(e) => {
                self.notify_error(format!("Erreur lecture dimensions : {e}"));
                return;
            }
        };
        if dimensions.0 == 0 || dimensions.1 == 0 {
            self.notify_error(format!("Image vide ({}x{})", dimensions.0, dimensions.1));
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
                // Recent image: the welcome state offers it on the next launch.
                // GTK reads the same information in the desktop manager; egui has
                // no equivalent and keeps its own list.
                if let Some(prefs) = self.prefs.as_mut() {
                    prefs.push_recent(&path);
                }
                // Size of the original file: it is the reference against which
                // the export size is compared.
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
                self.notify_error(format!("Erreur chargement : {e}"));
            }
        }
    }

    // ─── Presets ─────────────────────────────────────────────────────────────

    /// Cancels then restarts the current compute (the new compute takes over).
    pub(crate) fn trigger_compute(&mut self, ctx: &egui::Context) {
        if self.computing.load(Ordering::Relaxed) {
            // Cancel the compute in progress, then restart
            self.cancel.store(true, Ordering::Relaxed);
            // The cached dots belong to the previous parameter set and must not
            // be offered for SVG export while the replacement is pending.
            *self.last_dots.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return;
        }
        self.start_compute(ctx);
    }

    /// Applies a built-in preset ( undoable like any other change).
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
        self.notify(format!("Preset « {} » appliqué.", preset.name));
    }

    /// Index of the preset matching the current settings exactly.
    pub(crate) fn preset_index(&self) -> Option<usize> {
        frontend::matching_preset_index(&self.params)
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

    /// Draws a new random seed and restarts the compute.
    pub(crate) fn vary_seed(&mut self, ctx: &egui::Context) {
        // Same draw as GTK, hence a single definition (see `frontend`).
        self.params.rng_seed = Some(frontend::random_seed());
        self.on_params_changed(ctx);
    }

    // ─── Presets (files) ─────────────────────────────────────────────────────

    pub(crate) fn load_preset(&mut self, path: PathBuf, ctx: &egui::Context) {
        match frontend::load_preset(&path) {
            Ok(params) => {
                self.params = params;
                self.refresh_src_rgb(ctx);
                self.on_params_changed(ctx);
                self.notify(format!("Preset chargé : {}", path.display()));
            }
            Err(e) => self.notify_error(format!("Erreur chargement preset : {e}")),
        }
    }

    pub(crate) fn save_preset(&mut self, path: PathBuf) {
        let path = frontend::ensure_toml_extension(path);
        if !io::confirm_overwrite(&path) {
            self.notify(frontend::msg::SAVE_CANCELLED);
            return;
        }
        match frontend::save_preset(&path, &self.params) {
            Ok(()) => self.notify(format!("Preset sauvegardé : {}", path.display())),
            Err(e) => self.notify_error(format!("Erreur sauvegarde preset : {e}")),
        }
    }

    // ─── History ─────────────────────────────────────────────────────────────

    /// Commit of a new state into the history (undo stack). The redo stack is
    /// reset to zero. We avoid pushing consecutive identical entries
    /// (e.g.: endless drag of the same slider).
    pub(crate) fn commit_history(&mut self) {
        let cur = self.params.clone();
        if let Some(last) = &self.last_committed
            && last == &cur
        {
            return;
        }
        // Ceiling at 50 entries (FIFO).
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

    /// `Ctrl+Z`: restores the previous FilterParams.
    pub(crate) fn undo(&mut self, ctx: &egui::Context) {
        if self.pending_commit {
            self.commit_history();
            self.pending_commit = false;
        }
        let Some(prev) = self.history.pop() else {
            self.notify(frontend::msg::NOTHING_TO_UNDO);
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
        self.notify(frontend::msg::UNDONE);
        ctx.request_repaint();
    }

    /// `Ctrl+Y` (or `Ctrl+Shift+Z`): restores the next state.
    pub(crate) fn redo(&mut self, ctx: &egui::Context) {
        let Some(next) = self.future.pop() else {
            self.notify(frontend::msg::NOTHING_TO_REDO);
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
        self.notify(frontend::msg::REDONE);
        ctx.request_repaint();
    }

    /// Resets every setting to its default. Goes through the history so it stays
    /// undoable (`Ctrl+Z`).
    pub(crate) fn reset_params(&mut self, ctx: &egui::Context) {
        if self.params == FilterParams::default() {
            self.notify(frontend::msg::ALREADY_DEFAULT);
            return;
        }
        self.params = FilterParams::default();
        self.commit_history();
        // `bg_color` changed: the flattened source must be rebuilt.
        self.refresh_src_rgb(ctx);
        self.on_params_changed(ctx);
        self.notify("Réglages réinitialisés.");
    }

    // ─── Zoom ────────────────────────────────────────────────────────────────

    pub(crate) fn current_zoom(&self) -> f32 {
        if self.zoom_fit {
            self.effective_zoom
        } else {
            self.zoom
        }
    }

    /// Changes the zoom while leaving "Fit" mode.
    pub(crate) fn set_zoom(&mut self, value: f32) {
        self.zoom = value.clamp(frontend::ZOOM_MIN, frontend::ZOOM_MAX);
        self.zoom_fit = false;
    }

    // ─── Misc ────────────────────────────────────────────────────────────────

    /// Rebuilds `src_rgb` from `src_dynamic` if bg_color has changed.
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
