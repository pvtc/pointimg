//! sRGB ↔ linear gamma correction via pre-computed lookup tables.
//!
//! When `FilterParams::gamma_correct` is true, the pipeline pre-processes the
//! source image by linearizing its sRGB pixels, runs the algorithms on that
//! linear version (so the color averages become perceptually correct), then
//! re-encodes the final result into sRGB. Without touching the internal
//! algorithms.
//!
//! The tables cover the 256 u8 values in both directions; a lookup is a plain
//! indexing. The total cost for a W×H image is: two passes (linearization +
//! re-encoding) ≈ 6·W·H indexings.

use image::{Rgb, RgbImage, Rgba, RgbaImage};
use std::sync::LazyLock;

/// sRGB (u8) → linear value (u8) conversion.
/// Encoding: a 0-255 byte represents a linear radiance normalized by 255.
/// The precision is 8-bit (enough for display).
fn build_srgb_to_linear_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for i in 0..256u32 {
        let s = i as f32 / 255.0; // normalized sRGB input
        let l = if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        };
        lut[i as usize] = (l * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Linear (u8) → sRGB (u8) conversion.
fn build_linear_to_srgb_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for i in 0..256u32 {
        let l = i as f32 / 255.0; // normalized linear input
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
pub(crate) fn srgb_to_linear_f32(v: u8) -> f32 {
    let s = v as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// High-precision linear source used by the CMYK halftone path. The regular
/// pointillist algorithms intentionally keep their compact RGB8 pipeline.
pub(crate) fn srgb_to_linear_f32_image(src: &RgbImage) -> image::Rgb32FImage {
    image::Rgb32FImage::from_fn(src.width(), src.height(), |x, y| {
        let p = src.get_pixel(x, y);
        Rgb([
            srgb_to_linear_f32(p[0]),
            srgb_to_linear_f32(p[1]),
            srgb_to_linear_f32(p[2]),
        ])
    })
}

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

/// Maps a whole `RgbImage` through a per-channel LUT.
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

/// Re-encodes the 3 RGB channels of an `RgbaImage` through a LUT, leaving the
/// alpha channel untouched. Used to turn a "linear" RGBA buffer into an "sRGB"
/// RGBA buffer after rendering.
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
        // For sRGB values far enough from black, the 8-bit LUT round-trip must
        // be ≈ identity (≤ 2 units). The very small values (sRGB < 32) collapse
        // to 0 in 8-bit linear: that is expected and barely visible in
        // practice, hence outside of the test.
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
        // sRGB 128 → linear ≈ 55 (noticeably darker). This is the heart of the
        // correction: the "linear" value 128 displayed as sRGB is ~21% of
        // luminance, not 50%.
        let lin = srgb_to_linear_u8(128);
        assert!(
            lin < 80,
            "128 sRGB doit devenir nettement plus sombre, got {lin}"
        );
        assert!(lin > 40, "mais pas trop, got {lin}");
    }

    #[test]
    fn mid_gray_linear_to_srgb_is_brighter() {
        // Inverse: linear 128 → sRGB ≈ 188 (brighter).
        let srgb = linear_to_srgb_u8(128);
        assert!(srgb > 150, "128 lin→sRGB doit être plus clair, got {srgb}");
        assert!(srgb < 220, "mais pas trop, got {srgb}");
    }
}
