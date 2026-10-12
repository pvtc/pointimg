//! Result exports: raster image, SVG, clipboard.

use std::path::{Path, PathBuf};

use eframe::egui;

use super::App;
use super::convert::rgba_to_color_image_checker;
use super::io;
use pointimg::filter;
use pointimg::frontend;

impl App {
    /// Memorizes the directory of a written file, offered to the next dialogs.
    fn remember_dir(&mut self, path: &Path) {
        if let Some(parent) = path.parent().filter(|p| p.is_dir()) {
            self.last_dir = Some(parent.to_path_buf());
        }
    }

    // ─── Dialogs ──────────────────────────────────────────────────────────────

    pub(crate) fn choose_export(&mut self) {
        let Some(name) = self.default_name("png") else {
            self.notify(frontend::msg::OPEN_FIRST);
            return;
        };
        let mut dialog = rfd::FileDialog::new()
            .set_file_name(&name)
            .add_filter("PNG", &["png"])
            .add_filter("JPEG", &["jpg", "jpeg"])
            .add_filter("WebP", &["webp"])
            .add_filter("BMP", &["bmp"])
            .add_filter("TIFF", &["tif", "tiff"]);
        if let Some(folder) = self.last_dir.clone() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.save_file() {
            self.save_result(path);
        }
    }

    pub(crate) fn choose_svg_export(&mut self) {
        let Some(name) = self.default_name("svg") else {
            self.notify(frontend::msg::OPEN_FIRST);
            return;
        };
        let mut dialog = rfd::FileDialog::new()
            .set_file_name(&name)
            .add_filter("SVG", &["svg"]);
        if let Some(folder) = self.last_dir.clone() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.save_file() {
            self.save_svg(path);
        }
    }

    pub(crate) fn choose_load_preset(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new().add_filter("Preset TOML", &["toml"]);
        if let Some(folder) = self.last_dir.clone() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.pick_file() {
            self.load_preset(path, ctx);
        }
    }

    pub(crate) fn choose_save_preset(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_file_name("preset.toml")
            .add_filter("Preset TOML", &["toml"]);
        if let Some(folder) = self.last_dir.clone() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.save_file() {
            self.save_preset(path);
        }
    }

    /// Default proposed name: `photo.jpg` → `photo-pointimg.png`.
    fn default_name(&self, extension: &str) -> Option<String> {
        let nom = frontend::derived_export_name(self.src_path.as_deref(), extension);
        // egui refuses to export without an image, where GTK falls back on
        // `pointimg.<ext>`: the name formula is shared, the decision to accept
        // it belongs to each interface.
        self.src_path.as_ref().map(|_| nom)
    }

    // ─── Exports ──────────────────────────────────────────────────────────────

    /// Export without a dialog, next to the source image.
    pub(crate) fn quick_export(&mut self) {
        let Some(path) = self.src_path.as_ref().map(|source| {
            source.with_file_name(frontend::derived_export_name(Some(source), "png"))
        }) else {
            self.notify(frontend::msg::OPEN_FIRST);
            return;
        };
        self.save_result(path);
    }

    /// Copies the current render into the system clipboard.
    pub(crate) fn copy_to_clipboard(&mut self, ctx: &egui::Context) {
        let image = {
            let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().map(|img| rgba_to_color_image_checker(img))
        };
        let Some(image) = image else {
            self.notify(frontend::msg::NOTHING_TO_COPY);
            return;
        };
        ctx.copy_image(image);
        self.notify(frontend::msg::COPIED);
    }

    pub(crate) fn save_result(&mut self, path: PathBuf) {
        // We never export a downscaled preview by inadvertence.
        if !self.has_final_result() {
            self.status =
                "Rendu en pleine résolution en cours — réessayez dans un instant.".to_string();
            return;
        }
        // Validate and normalize the selected extension.
        let path = frontend::ensure_image_extension(path, "png");
        if !io::confirm_overwrite(&path) {
            self.notify(frontend::msg::SAVE_CANCELLED);
            return;
        }
        // The guard is released before mutating `self` (see `remember_dir`).
        let result = {
            let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
            let Some(img) = guard.as_ref() else {
                return;
            };
            // PNG/WebP/TIFF preserve alpha; JPEG/BMP composite onto the
            // bg_color if the render is transparent (logic shared with GTK).
            frontend::save_rgba_image(&path, img, self.params.transparent, self.params.bg_color)
        };

        match result {
            Ok(_) => {
                self.remember_dir(&path);
                self.notify(format!("Sauvegardé : {}", path.display()));
            }
            Err(e) => self.notify_error(format!("Erreur sauvegarde : {e}")),
        }
    }

    pub(crate) fn save_svg(&mut self, path: PathBuf) {
        let path = frontend::ensure_svg_extension(path);
        if !io::confirm_overwrite(&path) {
            self.notify(frontend::msg::SAVE_CANCELLED);
            return;
        }
        // Reusing the dots of the last render avoids restarting the pipeline.
        let rendered = {
            let dots_guard = self.last_dots.lock().unwrap_or_else(|e| e.into_inner());
            match (dots_guard.as_ref(), &self.src_rgb) {
                (Some(dots), Some(src)) => {
                    let (w, h) = src.dimensions();
                    Some(filter::render_svg_from_dots(w, h, dots, &self.params))
                }
                _ => None,
            }
        };
        if let Some(rendered) = rendered {
            self.write_svg(path, rendered);
            return;
        }
        // Fallback: no cached dot, we recompute from the source image.
        if let Some(src) = &self.src_rgb {
            let rendered = filter::render_svg(src, &self.params);
            self.write_svg(path, rendered);
        }
    }

    fn write_svg(&mut self, path: PathBuf, rendered: anyhow::Result<String>) {
        match rendered {
            Ok(svg) => match frontend::atomic_text_write(&path, &svg) {
                Ok(_) => {
                    self.remember_dir(&path);
                    self.notify(format!("SVG sauvegardé : {}", path.display()));
                }
                Err(e) => self.notify_error(format!("Erreur écriture SVG : {e}")),
            },
            Err(e) => self.notify_error(format!("Erreur rendu SVG : {e}")),
        }
    }
}
