//! Conversion des images `image` vers les textures egui.

use eframe::egui;
use egui::ColorImage;
use image::{GrayImage, RgbImage, Rgba, RgbaImage};

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
            let a = p[3] as f32 / 255.0;
            // Damier 8px, deux gris.
            let checker_light = 200u8;
            let checker_dark = 240u8;
            let cell = ((x / 8) + (y / 8)) % 2;
            let bg = if cell == 0 {
                checker_light
            } else {
                checker_dark
            };
            // Composite sur le damier.
            let inv = 1.0 - a;
            let r = (p[0] as f32 * a + bg as f32 * inv + 0.5) as u8;
            let g = (p[1] as f32 * a + bg as f32 * inv + 0.5) as u8;
            let b = (p[2] as f32 * a + bg as f32 * inv + 0.5) as u8;
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    ColorImage::new([w as usize, h as usize], pixels)
}

/// Helper: convertir `RgbImage` en `RgbaImage` opaque (alpha=255 partout).
pub(crate) fn rgb_to_rgba_opaque(img: &RgbImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            out.put_pixel(x, y, Rgba([p[0], p[1], p[2], 255]));
        }
    }
    out
}

pub(crate) fn gray_to_color_image(img: &GrayImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let pixels: Vec<egui::Color32> = img
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[0], p[0]))
        .collect();
    ColorImage::new([w as usize, h as usize], pixels)
}
