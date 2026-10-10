//! Export des résultats : image raster, SVG, presse-papiers.

use std::path::{Path, PathBuf};

use eframe::egui;

use super::App;
use super::convert::rgba_to_color_image_checker;
use super::io;
use pointimg::filter;
use pointimg::frontend;

impl App {
    /// Mémorise le dossier d'un fichier écrit, proposé aux dialogues suivants.
    fn remember_dir(&mut self, path: &Path) {
        if let Some(parent) = path.parent().filter(|p| p.is_dir()) {
            self.last_dir = Some(parent.to_path_buf());
        }
    }

    // ─── Dialogues ───────────────────────────────────────────────────────────

    pub(crate) fn choose_export(&mut self) {
        let Some(name) = self.default_name("png") else {
            self.notify("Ouvrez d'abord une image.");
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
            self.notify("Ouvrez d'abord une image.");
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

    /// Nom proposé par défaut : `photo.jpg` → `photo-pointimg.png`.
    fn default_name(&self, extension: &str) -> Option<String> {
        self.src_path
            .as_ref()
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .map(|stem| format!("{stem}-pointimg.{extension}"))
    }

    // ─── Exports ─────────────────────────────────────────────────────────────

    /// Export sans dialogue, à côté de l'image source.
    pub(crate) fn quick_export(&mut self) {
        let Some(path) = self.src_path.as_ref().map(|path| {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("pointimg");
            path.with_file_name(format!("{stem}-pointimg.png"))
        }) else {
            self.notify("Ouvrez d'abord une image.");
            return;
        };
        self.save_result(path);
    }

    /// Copie le rendu courant dans le presse-papiers système.
    pub(crate) fn copy_to_clipboard(&mut self, ctx: &egui::Context) {
        let image = {
            let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().map(|img| rgba_to_color_image_checker(img))
        };
        let Some(image) = image else {
            self.notify("Aucun résultat à copier.");
            return;
        };
        ctx.copy_image(image);
        self.notify("Image copiée dans le presse-papiers.");
    }

    pub(crate) fn save_result(&mut self, path: PathBuf) {
        // On n'exporte jamais un aperçu réduit par inadvertance.
        if !self.has_final_result() {
            self.status =
                "Rendu en pleine résolution en cours — réessayez dans un instant.".to_string();
            return;
        }
        // Validate and normalize the selected extension.
        let path = frontend::ensure_image_extension(path, "png");
        if !io::confirm_overwrite(&path) {
            self.notify("Sauvegarde annulée.");
            return;
        }
        // Le garde est relâché avant de muter `self` (voir `remember_dir`).
        let result = {
            let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
            let Some(img) = guard.as_ref() else {
                return;
            };
            // PNG/WebP/TIFF préservent l'alpha ; JPEG/BMP compositent sur la
            // bg_color si le rendu est transparent (logique partagée avec GTK).
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
            self.notify("Sauvegarde annulée.");
            return;
        }
        // Utiliser render_svg_from_dots si on a les dots (archi 19)
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
        // Fallback : aucun dot en cache, on recalcule depuis l'image source.
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
