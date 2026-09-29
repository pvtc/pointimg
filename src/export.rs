//! Taille réelle d'un export, format par format.
//!
//! Le point d'appui de Halftone — « la même image, mais plus légère » — ne se
//! constate que si l'on compare la taille du fichier exporté à celle de la
//! source. Aucune formule ne donne cette taille de façon fiable : la taille
//! d'un PNG ou d'un WebP dépend de l'entropie du rendu, donc de l'image, des
//! réglages et du format.
//!
//! On encode donc réellement l'image, dans le même chemin de code que
//! l'export ([`crate::frontend::encode_rgba_image`]), et on rapporte le nombre
//! d'octets obtenu. C'est exact, et cela coûte une compression complète par
//! format — d'où les deux garde-fous ci-dessous.
//!
//! - Les formats rapides sont toujours mesurés (PNG, JPEG, TIFF, BMP).
//! - Le WebP **sans perte** est bien plus lent à encoder (libwebp, méthode 4) :
//!   au-delà de [`WEBP_MEASURE_LIMIT`] pixels il n'est pas mesuré, et
//!   l'interface affiche « — » plutôt qu'un chiffre inventé.
//!
//! Les mesures sont produites dans un thread de travail, en parallèle avec
//! rayon, et invalidées par un compteur de génération quand l'utilisateur
//! continue de régler l'image.

use image::RgbaImage;
use rayon::prelude::*;

use crate::frontend::{encode_rgba_extension, format_bytes};

/// Au-delà de ce nombre de pixels, le WebP sans perte n'est pas mesuré
/// (l'encodage prend alors plusieurs secondes pour une seule image).
pub const WEBP_MEASURE_LIMIT: u64 = 4_000_000;

/// Formats d'export proposés à l'utilisateur, dans l'ordre du dialogue.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportFormat {
    /// PNG, sans perte, avec alpha — le format par défaut.
    Png,
    /// WebP **sans perte** (c'est ce que produit l'export), avec alpha.
    WebP,
    /// JPEG : l'alpha est composé sur le fond.
    Jpeg,
    /// TIFF, sans perte, avec alpha — très lourd, rarement utile ici.
    Tiff,
    /// BMP, sans compression, sans alpha.
    Bmp,
}

impl ExportFormat {
    /// Tous les formats, dans l'ordre d'affichage.
    pub const ALL: [ExportFormat; 5] = [
        ExportFormat::Png,
        ExportFormat::WebP,
        ExportFormat::Jpeg,
        ExportFormat::Tiff,
        ExportFormat::Bmp,
    ];

    /// Extension du fichier exporté, sans le point.
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Png => "png",
            ExportFormat::WebP => "webp",
            ExportFormat::Jpeg => "jpg",
            ExportFormat::Tiff => "tiff",
            ExportFormat::Bmp => "bmp",
        }
    }

    /// Nom court affiché dans l'interface.
    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Png => "PNG",
            ExportFormat::WebP => "WebP",
            ExportFormat::Jpeg => "JPEG",
            ExportFormat::Tiff => "TIFF",
            ExportFormat::Bmp => "BMP",
        }
    }

    /// Format par défaut de l'export (celui du bouton et du dialogue).
    pub fn default_export() -> Self {
        ExportFormat::Png
    }

    /// Vrai si le format est mesuré systématiquement.
    fn is_cheap(self) -> bool {
        !matches!(self, ExportFormat::WebP)
    }
}

/// Taille d'un export, mesurée par encodage réel.
#[derive(Clone, Debug, Default)]
pub struct ExportSizes {
    /// Octets par format ; `None` quand le format n'a pas été mesuré.
    pub sizes: Vec<(ExportFormat, Option<u64>)>,
    /// Taille du fichier source, si elle est connue.
    pub source: Option<u64>,
}

impl ExportSizes {
    /// Taille mesurée d'un format, si elle l'a été.
    pub fn get(&self, format: ExportFormat) -> Option<u64> {
        self.sizes
            .iter()
            .find(|(candidate, _)| *candidate == format)
            .and_then(|(_, size)| *size)
    }

    /// Gain en pourcentage par rapport à la source (négatif = plus léger).
    ///
    /// `None` si la source ou le format n'a pas été mesuré, ou si le gain est
    /// indéterminé (source vide).
    pub fn gain_percent(&self, format: ExportFormat) -> Option<f64> {
        let (source, output) = (self.source?, self.get(format)?);
        if source == 0 {
            return None;
        }
        Some((1.0 - output as f64 / source as f64) * 100.0)
    }

    /// Rendu de la ligne d'information : « PNG 2,1 Mo · WebP — · JPEG 380 Ko ».
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

    /// Rendu de la comparaison avec la source, pour l'infobulle du bouton
    /// d'export : « source 8,4 Mo · PNG −75 % ».
    pub fn comparison(&self, format: ExportFormat) -> Option<String> {
        let source = self.source?;
        let mut text = format!("source {}", format_bytes(source));
        if let Some(gain) = self.gain_percent(format) {
            // `gain` est positif quand l'export est plus léger.
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

/// Mesure la taille d'un export pour chaque format, en encodant réellement
/// l'image.
///
/// `pixels` sert à décider si le WebP sans perte est mesuré ; les appelants le
/// déduisent de l'image, mais le paramètre permet de tester sans.allouer.
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

    /// Image dont on connaît la taille encodée : 4×4 pixels unis, donc chaque
    /// format produit un fichier déterministe.
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
        // La mesure doit tomber sur la taille du fichier réellement écrit :
        // c'est tout l'intérêt de partager le chemin d'encodage.
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
        // Le même rendu transparent, dans les deux chemins : le PNG conserve
        // l'alpha, le JPEG compose sur le fond.
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
        // 2000×2001 pixels : au-delà de la limite, on n'encode pas le WebP.
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
