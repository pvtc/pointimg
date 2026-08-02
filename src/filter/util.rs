use image::{DynamicImage, Rgb, RgbImage};

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

pub(crate) fn pixel_variance(
    src: &RgbImage,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    avg: &[u8; 3],
) -> f32 {
    let (iw, ih) = src.dimensions();
    let mut sum = 0f32;
    let mut n = 0u32;
    for py in y0..(y0 + h).min(ih) {
        for px in x0..(x0 + w).min(iw) {
            let p = src.get_pixel(px, py);
            let dr = p[0] as f32 - avg[0] as f32;
            let dg = p[1] as f32 - avg[1] as f32;
            let db = p[2] as f32 - avg[2] as f32;
            sum += dr * dr + dg * dg + db * db;
            n += 1;
        }
    }
    if n > 0 { sum / n as f32 } else { 0.0 }
}

/// Aplatir n'importe quelle `DynamicImage` vers RGB8 en composant l'alpha sur `bg`.
/// Fix #4 : convertit d'abord en RGBA8 pour gérer tous les formats alpha
/// (RGBA16, LumaA8, etc.), pas seulement ImageRgba8.
pub fn flatten_to_rgb(img: &DynamicImage, bg: [u8; 3]) -> RgbImage {
    // Q7: simplified — just check if the format has alpha and composite if so
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    // Convertir en RGBA8 pour compositeur uniformément
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    RgbImage::from_fn(w, h, |x, y| {
        let p = rgba.get_pixel(x, y);
        let a = p[3] as f32 / 255.0;
        Rgb([
            (p[0] as f32 * a + bg[0] as f32 * (1.0 - a)) as u8,
            (p[1] as f32 * a + bg[1] as f32 * (1.0 - a)) as u8,
            (p[2] as f32 * a + bg[2] as f32 * (1.0 - a)) as u8,
        ])
    })
}
