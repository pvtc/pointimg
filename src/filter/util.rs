use image::{DynamicImage, GenericImageView, Rgb, RgbImage};

use super::halftone::HalftoneMode;
use super::params::{Algorithm, FilterParams, MAX_IMAGE_DIMENSION, MAX_IMAGE_PIXELS};

/// Estimates the peak working-set size of the CPU pipeline.
/// This is deliberately conservative: it includes source, density, integral
/// tables (2 f64 pour la density map + 6 u64 pour le quadtree), result
/// buffers and a fixed decoder/GUI overhead.
pub fn estimate_memory_bytes(width: u32, height: u32) -> u64 {
    estimate_memory_base(width, height, 0)
}

/// Comme [`estimate_memory_bytes`], mais tient compte des cartes de couverture
/// du mode halftone : 4 octets/pixel en CMYK (`u8` × 4 canaux), `n` en mode
/// dominant. Sans cela, l'estimation affichée par la GUI sous-évalue fortement
/// le mode dominant.
pub fn estimate_memory_bytes_for(width: u32, height: u32, params: &FilterParams) -> u64 {
    let extra_per_pixel = if params.algorithm == Algorithm::Halftone {
        match &params.halftone {
            HalftoneMode::Cmyk { .. } => 4,
            HalftoneMode::Dominant { n, .. } => (*n).min(32) as u64,
            HalftoneMode::Off => 0,
        }
    } else {
        0
    };
    estimate_memory_base(width, height, extra_per_pixel)
}

fn estimate_memory_base(width: u32, height: u32, extra_per_pixel: u64) -> u64 {
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    pixels
        .saturating_mul(3 + 4 + 16 + 48 + 4 + 4 + extra_per_pixel)
        .saturating_add(64 * 1024 * 1024)
}

/// Resizes an image so it satisfies the pipeline limits while preserving its
/// aspect ratio. Returns the resized image and whether a resize occurred.
pub fn resize_to_limits(img: DynamicImage) -> (DynamicImage, bool) {
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return (img, false);
    }
    let pixel_scale = (MAX_IMAGE_PIXELS as f64 / (u64::from(width) * u64::from(height)) as f64)
        .sqrt()
        .min(1.0);
    let dimension_scale = (f64::from(MAX_IMAGE_DIMENSION) / f64::from(width))
        .min(f64::from(MAX_IMAGE_DIMENSION) / f64::from(height))
        .min(1.0);
    let scale = pixel_scale.min(dimension_scale);
    if scale >= 1.0 {
        return (img, false);
    }
    let new_width = (f64::from(width) * scale).floor().max(1.0) as u32;
    let new_height = (f64::from(height) * scale).floor().max(1.0) as u32;
    (
        img.resize(new_width, new_height, image::imageops::FilterType::Triangle),
        true,
    )
}

pub(crate) fn luminance(r: u8, g: u8, b: u8) -> f32 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

pub(crate) fn pixel_sum(src: &RgbImage, x0: u32, y0: u32, w: u32, h: u32) -> (u64, u64, u64, u64) {
    let (iw, ih) = src.dimensions();
    let (mut sr, mut sg, mut sb, mut n) = (0u64, 0u64, 0u64, 0u64);
    for py in y0..(y0 + h).min(ih) {
        for px in x0..(x0 + w).min(iw) {
            let p = src.get_pixel(px, py);
            sr += p[0] as u64;
            sg += p[1] as u64;
            sb += p[2] as u64;
            n += 1;
        }
    }
    (sr, sg, sb, n)
}

/// Aplatir n'importe quelle `DynamicImage` vers RGB8 en composant l'alpha sur `bg`.
/// La conversion passe d'abord par RGBA8 pour traiter uniformément tous les
/// formats à canal alpha (RGBA16, LumaA8, etc.), pas seulement `ImageRgba8`.
pub fn flatten_to_rgb(img: &DynamicImage, bg: [u8; 3]) -> RgbImage {
    // Composite only when the source actually contains an alpha channel.
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    // Convertir en RGBA8 pour compositeur uniformément
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    RgbImage::from_fn(w, h, |x, y| {
        let p = rgba.get_pixel(x, y);
        let a = p[3] as f32 / 255.0;
        let inv = 1.0 - a;
        // Arrondi (et non troncature) pour rester cohérent avec le
        // compositing de `flatten_rgba_image` côté CLI.
        Rgb([
            (p[0] as f32 * a + bg[0] as f32 * inv).round() as u8,
            (p[1] as f32 * a + bg[1] as f32 * inv).round() as u8,
            (p[2] as f32 * a + bg[2] as f32 * inv).round() as u8,
        ])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halftone_memory_accounts_for_coverage_maps() {
        let base = estimate_memory_bytes(1000, 1000);
        let mut params = FilterParams {
            algorithm: Algorithm::Halftone,
            ..FilterParams::default()
        };

        params.halftone = HalftoneMode::Dominant {
            n: 8,
            base_angle_deg: 0.0,
        };
        assert_eq!(
            estimate_memory_bytes_for(1000, 1000, &params) - base,
            1_000_000 * 8
        );

        params.halftone = HalftoneMode::Cmyk { angles: [0.0; 4] };
        assert_eq!(
            estimate_memory_bytes_for(1000, 1000, &params) - base,
            1_000_000 * 4
        );

        // Hors halftone, aucun surcoût.
        params.algorithm = Algorithm::Grid;
        assert_eq!(estimate_memory_bytes_for(1000, 1000, &params), base);
    }

    #[test]
    fn flatten_to_rgb_rounds_alpha_compositing() {
        let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            1,
            1,
            image::Rgba([255, 0, 0, 128]),
        ));
        let out = flatten_to_rgb(&img, [255, 255, 255]);
        assert_eq!(out.get_pixel(0, 0)[0], 255);
        // Rouge à ~50 % sur blanc : arrondi vers 127/128, jamais tronqué à 126.
        assert!(out.get_pixel(0, 0)[1] >= 127);
    }
}
