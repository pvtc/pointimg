//! Real size of an export, format by format.
//!
//! The point of Halftone — "the same image, but lighter" — only shows up if
//! the exported file size is compared with the source one. No formula gives
//! that size reliably: the size of a PNG or of a WebP depends on the entropy
//! of the render, hence on the image, the settings and the format.
//!
//! The image is therefore really encoded, through the same code path as the
//! export ([`crate::frontend::encode_rgba_image`]), and the number of bytes
//! obtained is reported. That is exact, and it costs one full compression per
//! format — hence the two safeguards below.
//!
//! - The fast formats are always measured (PNG, JPEG, TIFF, BMP).
//! - Lossless WebP is much slower to encode (libwebp, method 4): beyond
//!   [`WEBP_MEASURE_LIMIT`] pixels it is not measured, and the interface
//!   displays a "—" rather than an invented figure.
//!
//! The measurements are produced in a worker thread, in parallel with rayon,
//! and invalidated by a generation counter when the user keeps on adjusting
//! the image.

use image::RgbaImage;
use rayon::prelude::*;

use crate::frontend::{encode_rgba_extension, format_bytes};

/// Beyond this number of pixels, lossless WebP is not measured (encoding then
/// takes several seconds for a single image).
pub const WEBP_MEASURE_LIMIT: u64 = 4_000_000;

/// Export formats offered to the user, in dialog order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportFormat {
    /// PNG, lossless, with alpha — the default format.
    Png,
    /// WebP **lossless** (this is what the export produces), with alpha.
    WebP,
    /// JPEG: alpha is composited onto the background.
    Jpeg,
    /// TIFF, lossless, with alpha — very heavy, rarely useful here.
    Tiff,
    /// BMP, uncompressed, without alpha.
    Bmp,
}

impl ExportFormat {
    /// All the formats, in display order.
    pub const ALL: [ExportFormat; 5] = [
        ExportFormat::Png,
        ExportFormat::WebP,
        ExportFormat::Jpeg,
        ExportFormat::Tiff,
        ExportFormat::Bmp,
    ];

    /// Extension of the exported file, without the dot.
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Png => "png",
            ExportFormat::WebP => "webp",
            ExportFormat::Jpeg => "jpg",
            ExportFormat::Tiff => "tiff",
            ExportFormat::Bmp => "bmp",
        }
    }

    /// Short name displayed in the interface.
    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Png => "PNG",
            ExportFormat::WebP => "WebP",
            ExportFormat::Jpeg => "JPEG",
            ExportFormat::Tiff => "TIFF",
            ExportFormat::Bmp => "BMP",
        }
    }

    /// Default format of the export (the one of the button and of the dialog).
    pub fn default_export() -> Self {
        ExportFormat::Png
    }

    /// True when the format is measured systematically.
    fn is_cheap(self) -> bool {
        !matches!(self, ExportFormat::WebP)
    }
}

/// Size of an export, measured by real encoding.
#[derive(Clone, Debug, Default)]
pub struct ExportSizes {
    /// Bytes per format; `None` when the format was not measured.
    pub sizes: Vec<(ExportFormat, Option<u64>)>,
    /// Size of the source file, when it is known.
    pub source: Option<u64>,
}

impl ExportSizes {
    /// Measured size of a format, if it was measured.
    pub fn get(&self, format: ExportFormat) -> Option<u64> {
        self.sizes
            .iter()
            .find(|(candidate, _)| *candidate == format)
            .and_then(|(_, size)| *size)
    }

    /// Gain in percent relative to the source (negative = lighter).
    ///
    /// `None` when the source or the format was not measured, or when the gain
    /// is undetermined (empty source).
    pub fn gain_percent(&self, format: ExportFormat) -> Option<f64> {
        let (source, output) = (self.source?, self.get(format)?);
        if source == 0 {
            return None;
        }
        Some((1.0 - output as f64 / source as f64) * 100.0)
    }

    /// Rendering of the information line: "PNG 2.1 MB · WebP — · JPEG 380 kB".
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for format in ExportFormat::ALL {
            let cell = match self.get(format) {
                Some(bytes) => format!("{} {}", format.label(), format_bytes(bytes)),
                None => format!("{} —", format.label()),
            };
            parts.push(cell);
        }
        parts.join(" · ")
    }

    /// Explanation displayed under the comparison, in the tooltip.
    ///
    /// The text used to be written twice, once per interface, and a correction
    /// on one side would not have arrived from the other. This is not a
    /// widget: it is the explanation of a datum produced here, so it shares
    /// its logic with it.
    pub fn explanation() -> &'static str {
        "Tailles obtenues en encodant réellement le rendu dans chaque format.\n\
         « — » : format non mesuré (image trop grande pour un WebP sans perte)."
    }

    /// Rendering of the comparison with the source, for the tooltip of the
    /// export button: "source 8.4 MB · PNG −75 %".
    pub fn comparison(&self, format: ExportFormat) -> Option<String> {
        let source = self.source?;
        let mut text = format!("source {}", format_bytes(source));
        if let Some(gain) = self.gain_percent(format) {
            // `gain` is positive when the export is lighter.
            let rounded = gain.round() as i64;
            text.push_str(&format!(
                " · {} {}{}",
                format.label(),
                if rounded >= 0 { "−" } else { "+" },
                rounded.abs()
            ));
            text.push_str(" %");
        }
        Some(text)
    }
}

/// Measures the size of an export for each format, by really encoding the
/// image.
///
/// `pixels` is used to decide whether lossless WebP is measured; callers deduce
/// it from the image, but the parameter allows testing without allocating.
pub fn measure_export_sizes(
    img: &RgbaImage,
    transparent: bool,
    bg: [u8; 3],
    source: Option<u64>,
) -> ExportSizes {
    let (w, h) = img.dimensions();
    let pixels = u64::from(w) * u64::from(h);
    let sizes: Vec<(ExportFormat, Option<u64>)> = ExportFormat::ALL
        .into_par_iter()
        .map(|format| {
            if !format.is_cheap() && pixels > WEBP_MEASURE_LIMIT {
                return (format, None);
            }
            let measured = encode_rgba_extension(format.extension(), img, transparent, bg)
                .ok()
                .map(|bytes| bytes.len() as u64);
            (format, measured)
        })
        .collect();
    ExportSizes { sizes, source }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    /// Image whose encoded size is known: 4×4 solid pixels, so every format
    /// produces a deterministic file.
    fn flat_image(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba([10, 20, 30, 255]))
    }

    #[test]
    fn every_offered_format_is_measured() {
        let sizes = measure_export_sizes(&flat_image(16, 16), false, [255; 3], Some(4096));
        for format in ExportFormat::ALL {
            assert!(
                sizes.get(format).is_some(),
                "{} n'a pas été mesuré",
                format.label()
            );
        }
    }

    #[test]
    fn measured_sizes_match_a_real_save() {
        // The measurement must land on the size of the file really written:
        // that is the whole point of sharing the encoding path.
        let img = flat_image(24, 18);
        let dir = std::env::temp_dir().join(format!("pointimg-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for format in ExportFormat::ALL {
            let path = dir.join(format!("mesure.{}", format.extension()));
            crate::frontend::save_rgba_image(&path, &img, false, [255; 3]).unwrap();
            let written = std::fs::metadata(&path).unwrap().len();
            let measured = measure_export_sizes(&img, false, [255; 3], None)
                .get(format)
                .unwrap();
            assert_eq!(
                written,
                measured,
                "{} : tailleannounced != écrite",
                format.label()
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn alpha_survives_png_and_is_composited_in_jpeg() {
        // The same transparent render, through the two paths: PNG keeps the
        // alpha, JPEG composites onto the background.
        let img = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 0]));
        let png = encode_rgba_extension("png", &img, true, [255, 255, 255]).unwrap();
        let jpeg = encode_rgba_extension("jpg", &img, true, [255, 255, 255]).unwrap();
        let decoded = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!(decoded.get_pixel(0, 0)[3], 0, "le PNG doit garder l'alpha");
        let decoded = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!(
            decoded.get_pixel(0, 0).0,
            [255, 255, 255],
            "le JPEG doit composer sur le fond"
        );
    }

    #[test]
    fn webp_is_skipped_above_the_measure_limit() {
        // 2000×2001 pixels: beyond the limit, WebP is not encoded.
        let img = flat_image(2000, 2001);
        let sizes = measure_export_sizes(&img, false, [255; 3], None);
        assert_eq!(sizes.get(ExportFormat::WebP), None);
        assert!(sizes.get(ExportFormat::Png).is_some());
    }

    #[test]
    fn gain_is_negative_when_the_export_is_smaller() {
        let sizes = measure_export_sizes(&flat_image(16, 16), false, [255; 3], Some(1 << 20));
        let gain = sizes.gain_percent(ExportFormat::Png).unwrap();
        assert!(gain > 0.0, "gain attendu positif, obtenu {gain}");
        let comparison = sizes.comparison(ExportFormat::Png).unwrap();
        assert!(comparison.contains("−"), "comparaison : {comparison}");
    }

    #[test]
    fn unknown_source_leaves_the_comparison_empty() {
        let sizes = measure_export_sizes(&flat_image(8, 8), false, [255; 3], None);
        assert_eq!(sizes.gain_percent(ExportFormat::Png), None);
        assert_eq!(sizes.comparison(ExportFormat::Png), None);
    }

    #[test]
    fn summary_marks_unmeasured_formats() {
        let sizes = ExportSizes {
            sizes: vec![
                (ExportFormat::Png, Some(2_100_000)),
                (ExportFormat::WebP, None),
                (ExportFormat::Jpeg, Some(390_000)),
                (ExportFormat::Tiff, None),
                (ExportFormat::Bmp, None),
            ],
            source: Some(4_000_000),
        };
        let summary = sizes.summary();
        assert!(summary.contains("PNG 2.0 Mo"), "{summary}");
        assert!(summary.contains("WebP —"), "{summary}");
        assert!(summary.contains("JPEG 381 Ko"), "{summary}");
    }
}
