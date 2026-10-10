//! Conversion des images `image` vers les textures egui.

use eframe::egui;
use egui::ColorImage;
use image::{GrayImage, RgbImage, RgbaImage};
use pointimg::frontend;

pub(crate) fn rgb_to_color_image(img: &RgbImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let pixels: Vec<egui::Color32> = img
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
        .collect();
    ColorImage::new([w as usize, h as usize], pixels)
}

/// Conversion RGBA → ColorImage avec **damier transparent** pour visualiser l'alpha.
/// Les pixels opaques (alpha=255) affichent leur couleur ; les transparents
/// montrent un damier gris clair/gris foncé (effet "Photoshop"), avec les
/// pixels partiellement opaques mélangés proportionnellement.
pub(crate) fn rgba_to_color_image_checker(img: &RgbaImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            // Le composite vit dans `frontend` (partagé avec GTK) ; l'emballage
            // en `Color32` est le seul travail qui reste ici.
            let [r, g, b] = frontend::composite_on_checker(p.0, x, y);
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    ColorImage::new([w as usize, h as usize], pixels)
}

/// Helper: convertir `RgbImage` en `RgbaImage` opaque (alpha=255 partout).
pub(crate) use pointimg::frontend::rgb_to_rgba_opaque;

pub(crate) fn gray_to_color_image(img: &GrayImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let pixels: Vec<egui::Color32> = img
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[0], p[0]))
        .collect();
    ColorImage::new([w as usize, h as usize], pixels)
}
