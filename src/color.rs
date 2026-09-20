//! Color-profile aware image loading.
//!
//! The filter works in sRGB, but input files may carry an ICC profile. This
//! module converts tagged RGB/RGBA input to sRGB before alpha compositing.

use anyhow::{Context, Result, anyhow};
use image::{DynamicImage, ImageDecoder, ImageReader, Limits, RgbImage, RgbaImage};
use moxcms::{ColorProfile, Layout, TransformOptions};
use std::path::Path;

pub fn decode_to_srgb(path: &Path, profile_spec: &str) -> Result<(DynamicImage, bool, bool)> {
    let mut reader = ImageReader::open(path)
        .with_context(|| format!("Impossible de lire '{}'", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("Format inconnu pour '{}'", path.display()))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(131_072);
    limits.max_image_height = Some(131_072);
    limits.max_alloc = Some(512 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .context("Impossible de créer le décodeur")?;
    let embedded = decoder.icc_profile().context("Profil ICC invalide")?;
    let image = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("Impossible de décoder '{}'", path.display()))?;

    // Resize before the f32 ICC transform so oversized inputs cannot create
    // several full-size working buffers at once.
    let (image, was_resized) = crate::filter::resize_to_limits(image);
    let profile = profile_from_spec(profile_spec, embedded.as_deref())?;
    // Une conversion réelle n'a lieu que si un profil embarqué existe ou si un
    // profil source non-sRGB est demandé. Avec le défaut `auto` sans profil
    // embarqué, l'image est déjà sRGB : ne pas annoncer une conversion.
    let spec = profile_spec.to_ascii_lowercase();
    let uses_non_srgb = !matches!(spec.as_str(), "auto" | "srgb");
    let was_converted = embedded.is_some() || uses_non_srgb;
    convert_to_srgb(image, &profile).map(|image| (image, was_converted, was_resized))
}

pub fn output_profile_from_spec(spec: &str) -> Result<Option<(ColorProfile, Vec<u8>)>> {
    if spec.eq_ignore_ascii_case("srgb") || spec.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    let bytes = std::fs::read(spec)
        .with_context(|| format!("Impossible de lire le profil ICC de sortie '{spec}'"))?;
    let profile = ColorProfile::new_from_slice(&bytes)
        .map_err(|e| anyhow!("Profil ICC de sortie invalide '{spec}' : {e:?}"))?;
    Ok(Some((profile, bytes)))
}

pub fn profile_from_spec(spec: &str, embedded: Option<&[u8]>) -> Result<ColorProfile> {
    match spec.to_ascii_lowercase().as_str() {
        "auto" => match embedded {
            Some(bytes) => ColorProfile::new_from_slice(bytes)
                .map_err(|e| anyhow!("Profil ICC embarqué invalide : {e:?}")),
            None => Ok(ColorProfile::new_srgb()),
        },
        "srgb" => Ok(ColorProfile::new_srgb()),
        "display-p3" | "display_p3" | "p3" => Ok(ColorProfile::new_display_p3()),
        path => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Impossible de lire le profil ICC '{path}'"))?;
            ColorProfile::new_from_slice(&bytes)
                .map_err(|e| anyhow!("Profil ICC invalide '{path}' : {e:?}"))
        }
    }
}

fn convert_to_srgb(image: DynamicImage, profile: &ColorProfile) -> Result<DynamicImage> {
    let destination = ColorProfile::new_srgb();
    convert_image(image, profile, &destination)
}

pub fn convert_from_srgb(image: DynamicImage, profile: &ColorProfile) -> Result<DynamicImage> {
    let source = ColorProfile::new_srgb();
    convert_image(image, &source, profile)
}

fn convert_image(
    image: DynamicImage,
    source_profile: &ColorProfile,
    destination_profile: &ColorProfile,
) -> Result<DynamicImage> {
    if image.color().has_alpha() {
        let source = image.to_rgba8();
        let converted = transform_rgba(&source, source_profile, destination_profile)?;
        Ok(DynamicImage::ImageRgba8(converted))
    } else {
        let source = image.to_rgb8();
        let converted = transform_rgb(&source, source_profile, destination_profile)?;
        Ok(DynamicImage::ImageRgb8(converted))
    }
}

/// Pixels traités par appel de transformation ICC. Borne le pic mémoire des
/// buffers `f32` intermédiaires au lieu de réserver deux buffers pleine image
/// (24 octets/pixel pour RGB, 32 pour RGBA).
const COLOR_CHUNK_PIXELS: usize = 64 * 1024;

fn transform_rgb(
    source: &RgbImage,
    profile: &ColorProfile,
    destination: &ColorProfile,
) -> Result<RgbImage> {
    transform_rgb_chunked(source, profile, destination, COLOR_CHUNK_PIXELS)
}

fn transform_rgb_chunked(
    source: &RgbImage,
    profile: &ColorProfile,
    destination: &ColorProfile,
    chunk_pixels: usize,
) -> Result<RgbImage> {
    let transform = profile
        .create_transform_f32(
            Layout::Rgb,
            destination,
            Layout::Rgb,
            TransformOptions::default(),
        )
        .map_err(|e| anyhow!("Impossible de créer la transformation colorimétrique : {e:?}"))?;
    // `saturating_mul` : permet d'utiliser `usize::MAX` comme « tout d'un coup »
    // dans les tests sans déborder ; le plafond par `source.len()` évite une
    // capacité de vecteur absurde.
    let chunk_len = chunk_pixels
        .max(1)
        .saturating_mul(3)
        .max(3)
        .min(source.len().max(3));
    let mut input: Vec<f32> = Vec::with_capacity(chunk_len);
    let mut output: Vec<f32> = vec![0.0; chunk_len];
    let mut bytes = Vec::with_capacity(source.len());
    for chunk in source.as_raw().chunks(chunk_len) {
        let len = (chunk.len() / 3) * 3;
        input.clear();
        input.extend(chunk.iter().map(|&value| value as f32 / 255.0));
        output.resize(len, 0.0);
        transform
            .transform(&input, &mut output)
            .map_err(|e| anyhow!("Transformation colorimétrique échouée : {e:?}"))?;
        bytes.extend(
            output[..len]
                .iter()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8),
        );
    }
    RgbImage::from_raw(source.width(), source.height(), bytes)
        .ok_or_else(|| anyhow!("Taille d'image invalide après conversion colorimétrique"))
}

fn transform_rgba(
    source: &RgbaImage,
    profile: &ColorProfile,
    destination: &ColorProfile,
) -> Result<RgbaImage> {
    transform_rgba_chunked(source, profile, destination, COLOR_CHUNK_PIXELS)
}

fn transform_rgba_chunked(
    source: &RgbaImage,
    profile: &ColorProfile,
    destination: &ColorProfile,
    chunk_pixels: usize,
) -> Result<RgbaImage> {
    let transform = profile
        .create_transform_f32(
            Layout::Rgba,
            destination,
            Layout::Rgba,
            TransformOptions::default(),
        )
        .map_err(|e| anyhow!("Impossible de créer la transformation colorimétrique : {e:?}"))?;
    let chunk_len = chunk_pixels
        .max(1)
        .saturating_mul(4)
        .max(4)
        .min(source.len().max(4));
    let mut input: Vec<f32> = Vec::with_capacity(chunk_len);
    let mut output: Vec<f32> = vec![0.0; chunk_len];
    let mut bytes = Vec::with_capacity(source.len());
    for chunk in source.as_raw().chunks(chunk_len) {
        let len = (chunk.len() / 4) * 4;
        input.clear();
        input.extend(chunk.iter().map(|&value| value as f32 / 255.0));
        output.resize(len, 0.0);
        transform
            .transform(&input, &mut output)
            .map_err(|e| anyhow!("Transformation colorimétrique échouée : {e:?}"))?;
        bytes.extend(
            output[..len]
                .iter()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8),
        );
    }
    RgbaImage::from_raw(source.width(), source.height(), bytes)
        .ok_or_else(|| anyhow!("Taille d'image invalide après conversion colorimétrique"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn srgb_profile_is_accepted() {
        assert!(profile_from_spec("srgb", None).is_ok());
        assert!(profile_from_spec("display-p3", None).is_ok());
    }

    #[test]
    fn srgb_conversion_preserves_dimensions_and_alpha() {
        let image =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 3, image::Rgba([20, 40, 80, 128])));
        let converted = convert_to_srgb(image, &ColorProfile::new_srgb()).unwrap();
        assert_eq!(converted.dimensions(), (2, 3));
        assert!(converted.color().has_alpha());
    }

    #[test]
    fn chunked_rgb_transform_matches_single_shot() {
        let source = RgbImage::from_fn(200, 130, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let srgb = ColorProfile::new_srgb();
        let p3 = ColorProfile::new_display_p3();
        // Un chunk minuscule force de nombreux appels ; `usize::MAX` fait tout
        // d'un coup. Les deux doivent produire exactement les mêmes octets.
        let chunked = transform_rgb_chunked(&source, &srgb, &p3, 1).unwrap();
        let single = transform_rgb_chunked(&source, &srgb, &p3, usize::MAX).unwrap();
        assert_eq!(chunked.as_raw(), single.as_raw());
    }

    #[test]
    fn chunked_rgba_transform_matches_single_shot() {
        let source = RgbaImage::from_fn(90, 70, |x, y| {
            image::Rgba([
                (x % 256) as u8,
                (y % 256) as u8,
                ((x * y) % 256) as u8,
                ((x + y) % 256) as u8,
            ])
        });
        let srgb = ColorProfile::new_srgb();
        let p3 = ColorProfile::new_display_p3();
        let chunked = transform_rgba_chunked(&source, &srgb, &p3, 3).unwrap();
        let single = transform_rgba_chunked(&source, &srgb, &p3, usize::MAX).unwrap();
        assert_eq!(chunked.as_raw(), single.as_raw());
    }
}
