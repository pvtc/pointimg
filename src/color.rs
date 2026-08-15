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
    let was_converted = profile_spec != "srgb" || embedded.is_some();
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

fn transform_rgb(
    source: &RgbImage,
    profile: &ColorProfile,
    destination: &ColorProfile,
) -> Result<RgbImage> {
    let transform = profile
        .create_transform_f32(
            Layout::Rgb,
            destination,
            Layout::Rgb,
            TransformOptions::default(),
        )
        .map_err(|e| anyhow!("Impossible de créer la transformation colorimétrique : {e:?}"))?;
    let mut input = Vec::with_capacity(source.len() * 3);
    for pixel in source.pixels() {
        input.extend(pixel.0.map(|value| value as f32 / 255.0));
    }
    let mut output = vec![0.0; input.len()];
    transform
        .transform(&input, &mut output)
        .map_err(|e| anyhow!("Transformation colorimétrique échouée : {e:?}"))?;
    let bytes = output
        .chunks_exact(3)
        .flat_map(|pixel| {
            pixel
                .iter()
                .copied()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect();
    RgbImage::from_raw(source.width(), source.height(), bytes)
        .ok_or_else(|| anyhow!("Taille d'image invalide après conversion colorimétrique"))
}

fn transform_rgba(
    source: &RgbaImage,
    profile: &ColorProfile,
    destination: &ColorProfile,
) -> Result<RgbaImage> {
    let transform = profile
        .create_transform_f32(
            Layout::Rgba,
            destination,
            Layout::Rgba,
            TransformOptions::default(),
        )
        .map_err(|e| anyhow!("Impossible de créer la transformation colorimétrique : {e:?}"))?;
    let mut input = Vec::with_capacity(source.len() * 4);
    for pixel in source.pixels() {
        input.extend(pixel.0.map(|value| value as f32 / 255.0));
    }
    let mut output = vec![0.0; input.len()];
    transform
        .transform(&input, &mut output)
        .map_err(|e| anyhow!("Transformation colorimétrique échouée : {e:?}"))?;
    let bytes = output
        .chunks_exact(4)
        .flat_map(|pixel| {
            pixel
                .iter()
                .copied()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect();
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
}
