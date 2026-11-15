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

    pub(crate) fn choose_export(&mut self, ctx: &egui::Context) {
        let Some(name) = self.default_name("png") else {
            self.notify(frontend::msg::open_first());
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
            self.save_result(path, ctx);
        }
    }

    pub(crate) fn choose_svg_export(&mut self, ctx: &egui::Context) {
        let Some(name) = self.default_name("svg") else {
            self.notify(frontend::msg::open_first());
            return;
        };
        let mut dialog = rfd::FileDialog::new()
            .set_file_name(&name)
            .add_filter("SVG", &["svg"]);
        if let Some(folder) = self.last_dir.clone() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.save_file() {
            self.save_svg(path, ctx);
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
    pub(crate) fn quick_export(&mut self, ctx: &egui::Context) {
        let Some(path) = self.src_path.as_ref().map(|source| {
            source.with_file_name(frontend::derived_export_name(Some(source), "png"))
        }) else {
            self.notify(frontend::msg::open_first());
            return;
        };
        self.save_result(path, ctx);
    }

    /// Copies the current render into the system clipboard.
    pub(crate) fn copy_to_clipboard(&mut self, ctx: &egui::Context) {
        let image = {
            let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().map(|img| rgba_to_color_image_checker(img))
        };
        let Some(image) = image else {
            self.notify(frontend::msg::nothing_to_copy());
            return;
        };
        ctx.copy_image(image);
        self.notify(frontend::msg::copied());
    }

    pub(crate) fn save_result(&mut self, path: PathBuf, ctx: &egui::Context) {
        // We never export a downscaled preview by inadvertence.
        if !self.has_final_result() {
            self.status = pointimg::frontend::gettext(
                "Rendu en pleine résolution en cours — réessayez dans un instant.",
            )
            .to_string();
            return;
        }
        // Validate and normalize the selected extension.
        let path = frontend::ensure_image_extension(path, "png");
        if !io::confirm_overwrite(&path) {
            self.notify(frontend::msg::save_cancelled());
            return;
        }
        // Clone the `Arc` (not the image) and encode off the UI thread: a
        // lossless WebP on a 8 Mpx render takes seconds.
        let image = {
            let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().map(Clone::clone)
        };
        let Some(image) = image else {
            return;
        };
        let transparent = self.params.transparent;
        let bg = self.params.bg_color;
        let slot = Clone::clone(&self.save_result_slot);
        let thread_path = path.clone();
        let ctx = ctx.clone();
        self.status = pointimg::frontend::gettext("Export en cours…").to_string();
        std::thread::spawn(move || {
            let result = frontend::save_rgba_image(&thread_path, &image, transparent, bg)
                .map_err(|e| e.to_string());
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some((thread_path, false, result));
            ctx.request_repaint();
        });
    }

    pub(crate) fn save_svg(&mut self, path: PathBuf, ctx: &egui::Context) {
        let path = frontend::ensure_svg_extension(path);
        if !io::confirm_overwrite(&path) {
            self.notify(frontend::msg::save_cancelled());
            return;
        }
        // Reusing the dots of the last render avoids restarting the pipeline.
        // The SVG string generation is also moved off the UI thread: it walks
        // every dot and can allocate a large string.
        let dots = self
            .last_dots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(src) = self.src_rgb.as_ref() else {
            self.notify(frontend::msg::open_first());
            return;
        };
        let (w, h) = src.dimensions();
        // The cached dots avoid a full source clone; only the fallback path
        // (no dots yet) needs the image.
        let fallback = if dots.is_none() {
            Some(src.clone())
        } else {
            None
        };
        if self.params.dithering != filter::Dithering::Off {
            self.notify(pointimg::frontend::gettext(
                "Le dithering n'est pas reproduit dans l'export SVG.",
            ));
        }
        let params = self.params.clone();
        let slot = Clone::clone(&self.save_result_slot);
        let thread_path = path.clone();
        let ctx = ctx.clone();
        self.status = pointimg::frontend::gettext("Export SVG en cours…").to_string();
        std::thread::spawn(move || {
            let rendered = match dots.as_deref() {
                Some(dots) => filter::render_svg_from_dots(w, h, dots, &params),
                None => match fallback.as_ref() {
                    Some(src) => filter::render_svg(src, &params),
                    None => Err(anyhow::anyhow!(pointimg::frontend::gettext("Image vide"))),
                },
            };
            let result = rendered.map_err(|e| e.to_string()).and_then(|svg| {
                frontend::atomic_text_write(&thread_path, &svg).map_err(|e| e.to_string())
            });
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some((thread_path, true, result));
            ctx.request_repaint();
        });
    }

    /// Publishes the completion of a background export.
    pub(crate) fn poll_save_result(&mut self) {
        let outcome = self
            .save_result_slot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let Some((path, svg, result)) = outcome else {
            return;
        };
        match result {
            Ok(()) => {
                self.remember_dir(&path);
                let message = if svg {
                    pointimg::frontend::gettext("SVG sauvegardé : {}")
                } else {
                    pointimg::frontend::gettext("Sauvegardé : {}")
                };
                self.notify(message.replace("{}", &path.display().to_string()));
            }
            Err(e) => {
                let message = if svg {
                    pointimg::frontend::gettext("Erreur écriture SVG : {e}").replace("{e}", &e)
                } else {
                    pointimg::frontend::gettext("Erreur sauvegarde : {e}").replace("{e}", &e)
                };
                self.notify_error(message);
            }
        }
    }
}
