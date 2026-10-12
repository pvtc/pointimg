//! Conversion of `image` images to egui textures.

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

/// RGBA → ColorImage conversion with a **transparent checkerboard** to visualize
/// alpha. Opaque pixels (alpha=255) show their color; transparent ones
/// display a light gray/dark gray checkerboard ("Photoshop" effect), with
/// partially opaque pixels blended proportionally.
pub(crate) fn rgba_to_color_image_checker(img: &RgbaImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            // The composite lives in `frontend` (shared with GTK); wrapping it
            // into `Color32` is the only work left to do here.
            let [r, g, b] = frontend::composite_on_checker(p.0, x, y);
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    ColorImage::new([w as usize, h as usize], pixels)
}

/// Helper: convert `RgbImage` to opaque `RgbaImage` (alpha=255 everywhere).
pub(crate) use pointimg::frontend::rgb_to_rgba_opaque;

pub(crate) fn gray_to_color_image(img: &GrayImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let pixels: Vec<egui::Color32> = img
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[0], p[0]))
        .collect();
    ColorImage::new([w as usize, h as usize], pixels)
}
