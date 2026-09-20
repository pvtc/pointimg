//! Export des résultats : image raster et SVG.

use std::path::PathBuf;

use super::App;
use super::io;
use pointimg::filter;
use pointimg::frontend;

impl App {
    pub(crate) fn save_result(&mut self, path: PathBuf) {
        // Validate and normalize the selected extension.
        let path = frontend::ensure_image_extension(path, "png");
        if !io::confirm_overwrite(&path) {
            self.status = "Sauvegarde annulée.".to_string();
            return;
        }
        let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        let Some(img) = guard.as_ref() else { return };

        // PNG/WebP/TIFF préservent l'alpha ; JPEG/BMP compositent sur la
        // bg_color si le rendu est transparent (logique partagée avec GTK).
        let result =
            frontend::save_rgba_image(&path, img, self.params.transparent, self.params.bg_color);

        match result {
            Ok(_) => self.status = format!("Sauvegardé : {}", path.display()),
            Err(e) => self.status = format!("Erreur sauvegarde : {e}"),
        }
    }

    pub(crate) fn save_svg(&mut self, path: PathBuf) {
        let path = frontend::ensure_svg_extension(path);
        if !io::confirm_overwrite(&path) {
            self.status = "Sauvegarde annulée.".to_string();
            return;
        }
        // Utiliser render_svg_from_dots si on a les dots (archi 19)
        let dots_guard = self.last_dots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dots) = dots_guard.as_ref()
            && let Some(src) = &self.src_rgb
        {
            let (w, h) = src.dimensions();
            match filter::render_svg_from_dots(w, h, dots, &self.params) {
                Ok(svg) => match frontend::atomic_text_write(&path, &svg) {
                    Ok(_) => self.status = format!("SVG sauvegardé : {}", path.display()),
                    Err(e) => self.status = format!("Erreur écriture SVG : {e}"),
                },
                Err(e) => self.status = format!("Erreur rendu SVG : {e}"),
            }
            return;
        }
        drop(dots_guard);
        // Fallback : recalculer
        if let Some(src) = &self.src_rgb {
            match filter::render_svg(src, &self.params) {
                Ok(svg) => match frontend::atomic_text_write(&path, &svg) {
                    Ok(_) => self.status = format!("SVG sauvegardé : {}", path.display()),
                    Err(e) => self.status = format!("Erreur écriture SVG : {e}"),
                },
                Err(e) => self.status = format!("Erreur rendu SVG : {e}"),
            }
        }
    }
}
