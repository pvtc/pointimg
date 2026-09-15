//! Export des résultats : image raster et SVG.

use image::RgbaImage;
use std::path::PathBuf;

use super::App;
use super::io;
use pointimg::filter;

impl App {
    pub(crate) fn save_result(&mut self, path: PathBuf) {
        // Validate and normalize the selected extension.
        let path = io::ensure_extension(path, "png");
        if !io::confirm_overwrite(&path) {
            self.status = "Sauvegarde annulée.".to_string();
            return;
        }
        let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        let Some(img) = guard.as_ref() else { return };

        // Pour les formats sans alpha (JPEG, BMP), flat sur la bg_color.
        // PNG et WebP préservent l'alpha nativement.
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_else(|| "png".to_string());

        let result = if matches!(ext.as_str(), "png" | "webp" | "tif" | "tiff") {
            // Préserve l'alpha.
            io::atomic_image_save(&path, |tmp| img.save(tmp))
        } else if self.params.transparent {
            // JPG/BMP sans alpha : composite sur fond bg_color avant save.
            let flat = flatten_rgba_for_export(img, self.params.bg_color);
            io::atomic_image_save(&path, |tmp| flat.save(tmp))
        } else {
            // Pas transparent : simple to_rgb8.
            io::atomic_image_save(&path, |tmp| {
                image::DynamicImage::ImageRgba8(img.clone())
                    .to_rgb8()
                    .save(tmp)
            })
        };

        match result {
            Ok(_) => self.status = format!("Sauvegardé : {}", path.display()),
            Err(e) => self.status = format!("Erreur sauvegarde : {e}"),
        }
    }

    pub(crate) fn save_svg(&mut self, path: PathBuf) {
        let path = io::ensure_extension(path, "svg");
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
                Ok(svg) => match io::atomic_text_write(&path, &svg) {
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
                Ok(svg) => match io::atomic_text_write(&path, &svg) {
                    Ok(_) => self.status = format!("SVG sauvegardé : {}", path.display()),
                    Err(e) => self.status = format!("Erreur écriture SVG : {e}"),
                },
                Err(e) => self.status = format!("Erreur rendu SVG : {e}"),
            }
        }
    }
}

/// Compose une image RGBA sur un fond opaque (formats sans alpha : JPEG, BMP).
fn flatten_rgba_for_export(img: &RgbaImage, bg: [u8; 3]) -> image::RgbImage {
    let (w, h) = img.dimensions();
    image::RgbImage::from_fn(w, h, |x, y| {
        let p = img.get_pixel(x, y);
        let a = p[3] as f32 / 255.0;
        let inv = 1.0 - a;
        image::Rgb([
            (p[0] as f32 * a + bg[0] as f32 * inv) as u8,
            (p[1] as f32 * a + bg[1] as f32 * inv) as u8,
            (p[2] as f32 * a + bg[2] as f32 * inv) as u8,
        ])
    })
}
