//! Correction gamma sRGB ↔ linéaire via tables de lookup pré-calculées.
//!
//! Quand `FilterParams::gamma_correct` est vrai, le pipeline pré-traite l'image
//! source en linéarisant ses pixels sRGB, lance les algorithmes sur cette
//! version linéaire (les moyennes de couleur deviennent donc perceptuellement
//! correctes), puis ré-encode le résultat final en sRGB. Sans toucher aux
//! algorithmes internes.
//!
//! Les tables couvrent les 256 valeurs u8 dans les deux sens ; un lookup est une
//! simple indexation. Le coût total pour une image W×H est : deux passes
//! (line´arisation + ré-encodage) ≈ 6·W·H indexations.

use image::{Rgb, RgbImage, Rgba, RgbaImage};
use std::sync::LazyLock;

/// Conversion sRGB (u8) → valeur linéaire (u8).
/// Encodage : un byte 0-255 représente une radiance linéaire
/// normalisée par 255. La précision est 8-bit (suffisante pour la visualisation).
fn build_srgb_to_linear_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for i in 0..256u32 {
        let s = i as f32 / 255.0; // entrée sRGB normalisée
        let l = if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        };
        lut[i as usize] = (l * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Conversion linéaire (u8) → sRGB (u8).
fn build_linear_to_srgb_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for i in 0..256u32 {
        let l = i as f32 / 255.0; // entrée linéaire normalisée
        let s = if l <= 0.0031308 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        lut[i as usize] = (s * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

static SRGB_TO_LINEAR: LazyLock<[u8; 256]> = LazyLock::new(build_srgb_to_linear_lut);
static LINEAR_TO_SRGB: LazyLock<[u8; 256]> = LazyLock::new(build_linear_to_srgb_lut);

#[inline]
pub(crate) fn srgb_to_linear_u8(v: u8) -> u8 {
    SRGB_TO_LINEAR[v as usize]
}

#[inline]
pub(crate) fn linear_to_srgb_u8(v: u8) -> u8 {
    LINEAR_TO_SRGB[v as usize]
}

#[inline]
pub(crate) fn srgb_to_linear_rgb(p: [u8; 3]) -> [u8; 3] {
    [
        srgb_to_linear_u8(p[0]),
        srgb_to_linear_u8(p[1]),
        srgb_to_linear_u8(p[2]),
    ]
}

#[inline]
pub(crate) fn linear_to_srgb_rgb(p: [u8; 3]) -> [u8; 3] {
    [
        linear_to_srgb_u8(p[0]),
        linear_to_srgb_u8(p[1]),
        linear_to_srgb_u8(p[2]),
    ]
}

/// Map une `RgbImage` entière via une LUT par canal.
pub(crate) fn map_image<F: Fn(u8) -> u8>(src: &RgbImage, f: F) -> RgbImage {
    let (w, h) = src.dimensions();
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = src.get_pixel(x, y);
            out.put_pixel(x, y, Rgb([f(p[0]), f(p[1]), f(p[2])]));
        }
    }
    out
}

pub(crate) fn srgb_to_linear_image(src: &RgbImage) -> RgbImage {
    map_image(src, srgb_to_linear_u8)
}

pub(crate) fn linear_to_srgb_image(src: &RgbImage) -> RgbImage {
    map_image(src, linear_to_srgb_u8)
}

/// Ré-encode les 3 canaux RGB d'une `RgbaImage` via une LUT, sans toucher le canal alpha.
/// Utilisé pour transformer un buffer RGBA "linéaire" en buffer RGBA "sRGB" après le rendu.
pub(crate) fn map_rgba_image_rgb<F: Fn(u8) -> u8>(src: &RgbaImage, f: F) -> RgbaImage {
    let (w, h) = src.dimensions();
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = src.get_pixel(x, y);
            out.put_pixel(x, y, Rgba([f(p[0]), f(p[1]), f(p[2]), p[3]]));
        }
    }
    out
}

pub(crate) fn linear_to_srgba_image_rgb(src: &RgbaImage) -> RgbaImage {
    map_rgba_image_rgb(src, linear_to_srgb_u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_lut_extremes() {
        assert_eq!(srgb_to_linear_u8(0), 0);
        assert_eq!(srgb_to_linear_u8(255), 255);
    }

    #[test]
    fn linear_lut_extremes() {
        assert_eq!(linear_to_srgb_u8(0), 0);
        assert_eq!(linear_to_srgb_u8(255), 255);
    }

    #[test]
    fn round_trip_is_approx_identity_above_mid_range() {
        // Pour des valeurs sRGB suffisamment éloignées des noirs, le round-trip
        // LUT 8-bit doit être ≈ identité (≤ 2 unités). Les très petites valeurs
        // (sRGB < 32) s'effondrent vers 0 en linéaire 8-bit : c'est attendu et
        // peu visible en pratique, donc hors du test.
        for v in [32u8, 48, 64, 96, 128, 160, 192, 224, 240] {
            let r = linear_to_srgb_u8(srgb_to_linear_u8(v));
            assert!(
                (r as i32 - v as i32).abs() <= 2,
                "round-trip srgb→linear→srgb pour {v} = {r}, diff trop grande"
            );
        }
    }

    #[test]
    fn mid_gray_srgb_to_linear_is_darker() {
        // sRGB 128 → linéaire ≈ 55 (sensiblement plus sombre). C'est le cœur
        // de la correction : la valeur "linéaire" 128 représentée comme sRGB
        // est ~21% de luminance, pas 50%.
        let lin = srgb_to_linear_u8(128);
        assert!(
            lin < 80,
            "128 sRGB doit devenir nettement plus sombre, got {lin}"
        );
        assert!(lin > 40, "mais pas trop, got {lin}");
    }

    #[test]
    fn mid_gray_linear_to_srgb_is_brighter() {
        // Inverse : linéaire 128 → sRGB ≈ 188 (plus clair).
        let srgb = linear_to_srgb_u8(128);
        assert!(srgb > 150, "128 lin→sRGB doit être plus clair, got {srgb}");
        assert!(srgb < 220, "mais pas trop, got {srgb}");
    }
}
