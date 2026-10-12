use anyhow::{Result, anyhow};
use clap::ValueEnum;
use serde::{Deserialize, Deserializer, Serialize};

use super::dither::Dithering;

use crate::filter::halftone::{HalftoneConfig, HalftoneMode, Screening};

const PRESET_FORMAT_VERSION: u32 = 1;
/// Maximum length of an image side accepted by the pipeline (pixels).
pub const MAX_IMAGE_DIMENSION: u32 = 65_535;
/// Maximum number of pixels accepted by the pipeline (8388608 = 2048×4096).
pub const MAX_IMAGE_PIXELS: u64 = 8 * 1024 * 1024;

#[derive(Deserialize, Serialize)]
struct PresetDocument {
    preset_version: u32,
    params: FilterParams,
}

/// Reads `dithering` while accepting the two historical spellings.
///
/// `dithering = true` (0.1 presets) means [`Dithering::FloydSteinberg`], and
/// `false` like the absence of the key means [`Dithering::Off`]; algorithm
/// names are accepted as they are written in a preset.
fn deserialize_dithering<'de, D>(deserializer: D) -> Result<Dithering, D::Error>
where
    D: Deserializer<'de>,
{
    /// Visitor accepting a boolean or a string.
    struct Visitor;

    impl serde::de::Visitor<'_> for Visitor {
        type Value = Dithering;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("un algorithme de dithering (ou l'ancien booléen)")
        }

        fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Dithering, E> {
            Ok(if value {
                Dithering::FloydSteinberg
            } else {
                Dithering::Off
            })
        }

        fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Dithering, E> {
            match value {
                "off" | "Off" => Ok(Dithering::Off),
                "floyd-steinberg" | "FloydSteinberg" | "floyd_steinberg" => {
                    Ok(Dithering::FloydSteinberg)
                }
                "riemersma" | "Riemersma" => Ok(Dithering::Riemersma),
                "bayer4" | "Bayer4" => Ok(Dithering::Bayer4),
                "bayer8" | "Bayer8" => Ok(Dithering::Bayer8),
                other => Err(E::unknown_variant(other, &["dithering"])),
            }
        }
    }

    deserializer.deserialize_any(Visitor)
}

/// Dot placement algorithm.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize, ValueEnum)]
pub enum Algorithm {
    /// Regular grid of `cols` × `rows` cells.
    Grid,
    /// Spatial k-means over (x, y, r, g, b).
    Kmeans,
    /// Lloyd stippling weighted by the density map.
    Voronoi,
    /// Recursive subdivision by contrast (integral tables).
    Quadtree,
    /// Multi-channel halftone (CMYK rosette or dominant colors).
    /// The sub-parameters (`HalftoneMode`, `Screening`, `halftone_frequency`,
    /// …) live on `FilterParams`. When this algorithm is selected, the other
    /// placement parameters (`num_points`, `cols`, `iterations`,
    /// `variance_sensitivity`, `max_boost`, `grid_angle_deg`) are ignored.
    Halftone,
}

/// Shape of the drawn dots.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
pub enum DotShape {
    /// Filled disk (historical behavior)
    #[default]
    Circle,
    /// Filled square, side = 2×radius
    Square,
    /// Ellipse, `aspect` = width/height ratio ∈ (0, 10], `angle_deg` = rotation in degrees
    Ellipse {
        /// Width/height ratio (0, 10].
        aspect: f32,
        /// Rotation in degrees.
        angle_deg: f32,
    },
    /// Regular polygon with `sides` sides (3 = triangle, 4 = diamond, 6 = hexagon, …)
    RegularPolygon {
        /// Number of sides (3 to 12).
        sides: u8,
    },
}

/// All the filter settings, serializable into a TOML preset.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FilterParams {
    /// Placement algorithm.
    pub algorithm: Algorithm,
    /// Target number of dots (Kmeans / Voronoi / Quadtree)
    pub num_points: usize,
    /// Number of columns (Grid only)
    pub cols: u32,
    /// Minimum radius: fraction of min(width, height) — e.g. 0.002 = 0.2% of the image
    pub min_radius_ratio: f32,
    /// Maximum radius: fraction of min(width, height) — e.g. 0.08 = 8% of the image
    pub max_radius_ratio: f32,
    /// RGB background color — single source of truth
    pub bg_color: [u8; 3],
    /// Number of iterations for Voronoi / Kmeans
    pub iterations: usize,
    /// Sensitivity to local variance (0 = ignored, 1 = strong redistribution)
    pub variance_sensitivity: f32,
    /// Maximum size multiplier in uniform areas (1.0 = no boost)
    pub max_boost: f32,
    /// RNG seed for reproducibility (None = clock-based random)
    pub rng_seed: Option<u64>,
    /// Number of colors for quantization (None = no quantization)
    pub palette_size: Option<usize>,
    /// Shape of the dots
    pub dot_shape: DotShape,
    /// Transparent background on output (RGBA, alpha 0 between the dots).
    /// When true, `bg_color` is ignored for the final render (uncovered areas
    /// stay transparent). It is still used to flatten the source image to RGB
    /// (computing the average dot colors).
    pub transparent: bool,
    /// Gamma correction of the average dot colors. When true, the averages are
    /// taken in linear space (sRGB→linear→average→sRGB), which avoids the
    /// "too dark" bias on contrasted areas (e.g. the edge of a gradient).
    /// Cost: 2 passes over the image (source linearization + dst re-encoding).
    pub gamma_correct: bool,
    /// Dithering algorithm applied when quantizing the palette. When it is
    /// active (and `palette_size` is `Some(n)`), the final rendered image is
    /// post-quantized with the chosen screening — an "offset print" or
    /// "pixel art" effect. Ignored when `palette_size` is `None`.
    ///
    /// Old presets write `dithering = true`: it is read as
    /// [`Dithering::FloydSteinberg`] (cf. `deserialize_dithering`).
    #[serde(default, deserialize_with = "deserialize_dithering")]
    pub dithering: Dithering,
    /// Grid rotation angle (degrees), applied to the Grid algorithm only.
    /// The dots are placed on a grid pivoted around the image center, an
    /// experimental "halftone screen" effect. The color stays the average of
    /// the un-rotated cell. 0 = no rotation (default).
    pub grid_angle_deg: f32,
    /// Multi-channel halftone mode (CMYK rosette, dominant colors, or off).
    pub halftone: HalftoneMode,
    /// Screening (AM placement on a rotated grid, or FM blue noise).
    pub screening: Screening,
    /// AM screen frequency in "cells per min(W,H)". 60 → 60 dots along the
    /// smallest dimension (~60 lpp screen at 1 dpi).
    pub halftone_frequency: f32,
    /// Minimum halftone dot radius (fraction of `min(W,H)`).
    pub halftone_min_radius_ratio: f32,
    /// Maximum halftone dot radius (fraction of the screen step).
    pub halftone_max_dot_ratio: f32,
}

impl FilterParams {
    /// Builds a `HalftoneConfig` ready to hand to the halftone engine.
    pub fn halftone_config(&self) -> HalftoneConfig<'_> {
        HalftoneConfig {
            mode: &self.halftone,
            screening: self.screening,
            screen_frequency: self.halftone_frequency,
            dot_shape: self.dot_shape,
            min_radius_ratio: self.halftone_min_radius_ratio,
            max_dot_ratio: self.halftone_max_dot_ratio,
            rng_seed: self.rng_seed,
        }
    }

    /// Serializes the parameters into a TOML string, ready to write into a file.
    ///
    /// ```
    /// use pointimg::filter::{Algorithm, FilterParams};
    ///
    /// let params = FilterParams { algorithm: Algorithm::Quadtree, ..Default::default() };
    /// let toml = params.to_toml_string().unwrap();
    /// let back = FilterParams::from_toml_str(&toml).unwrap();
    /// assert_eq!(back.algorithm, Algorithm::Quadtree);
    /// ```
    pub fn to_toml_string(&self) -> Result<String> {
        // Presets are also created without an input image. Validate all
        // parameter-level constraints using a minimal valid image size.
        validate_params(1, 1, self)?;
        toml::to_string_pretty(&PresetDocument {
            preset_version: PRESET_FORMAT_VERSION,
            params: self.clone(),
        })
        .map_err(|e| anyhow!("erreur de sérialisation presets : {e}"))
    }

    /// Rebuilds the parameters from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        if let Ok(document) = toml::from_str::<PresetDocument>(s) {
            if document.preset_version != PRESET_FORMAT_VERSION {
                return Err(anyhow!(
                    "version de preset {} non supportée (version actuelle : {})",
                    document.preset_version,
                    PRESET_FORMAT_VERSION
                ));
            }
            return Ok(document.params);
        }

        // Flat presets, without the `document` wrapper, stay readable.
        toml::from_str(s).map_err(|e| anyhow!("preset TOML invalide : {e}"))
    }
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Voronoi,
            num_points: 800,
            cols: 80,
            min_radius_ratio: 0.003,
            max_radius_ratio: 0.06,
            bg_color: [255, 255, 255],
            iterations: 10,
            variance_sensitivity: 0.7,
            max_boost: 2.5,
            rng_seed: None,
            palette_size: None,
            dot_shape: DotShape::Circle,
            transparent: false,
            gamma_correct: false,
            dithering: Dithering::Off,
            grid_angle_deg: 0.0,
            halftone: HalftoneMode::Off,
            screening: Screening::Am,
            halftone_frequency: 60.0,
            halftone_min_radius_ratio: 0.002,
            halftone_max_dot_ratio: 0.85,
        }
    }
}

/// A dot produced by the filter
#[derive(Clone, Debug)]
pub struct Dot {
    /// Center position in pixels.
    pub x: f32,
    /// Center position in pixels.
    pub y: f32,
    /// Average RGB8 color of the covered area.
    pub color: [u8; 3],
    /// Radius in pixels.
    pub radius: f32,
}

/// Checks the image dimensions then all the parameters.
///
/// Invalid parameters (negative, non-finite, out-of-bounds values) are
/// rejected here before any computation, by the GUIs and the CLI alike.
///
/// Seven named checks instead of a series of 156 inline `if`s. The order of the
/// checks and the messages are unchanged, character for character: only the
/// grouping changes, and each check becomes testable on its own.
/// Dot radii: ordered, positive, and at the scale of the image.
fn validate_radii(p: &FilterParams) -> Result<()> {
    if p.min_radius_ratio <= 0.0 {
        return Err(anyhow!("min_radius_ratio doit etre > 0"));
    }
    if p.max_radius_ratio < p.min_radius_ratio {
        return Err(anyhow!(
            "max_radius_ratio ({}) doit etre >= min_radius_ratio ({})",
            p.max_radius_ratio,
            p.min_radius_ratio
        ));
    }
    // A radius larger than the image scale is not meaningful.
    if p.max_radius_ratio > 1.0 {
        return Err(anyhow!(
            "max_radius_ratio ({}) doit etre <= 1.0",
            p.max_radius_ratio
        ));
    }
    Ok(())
}

/// Placement: number of dots, grid, iterations.
fn validate_density(p: &FilterParams) -> Result<()> {
    if p.num_points == 0 {
        return Err(anyhow!("num_points doit etre > 0"));
    }
    if p.num_points > 100_000 {
        return Err(anyhow!("num_points doit etre <= 100000"));
    }
    if matches!(p.algorithm, Algorithm::Kmeans | Algorithm::Voronoi) && p.num_points > 50_000 {
        return Err(anyhow!(
            "num_points doit etre <= 50000 pour K-means/Voronoi"
        ));
    }
    if p.cols == 0 {
        return Err(anyhow!("cols doit etre > 0"));
    }
    if p.cols > 8192 {
        return Err(anyhow!("cols doit etre <= 8192"));
    }
    if p.iterations > 100 {
        return Err(anyhow!("iterations doit etre <= 100"));
    }
    Ok(())
}

/// Palette and sensitivity to variance.
fn validate_palette(p: &FilterParams) -> Result<()> {
    if let Some(n) = p.palette_size
        && n < 2
    {
        return Err(anyhow!("palette_size doit etre >= 2, got {}", n));
    }
    if p.palette_size.is_some_and(|n| n > 256) {
        return Err(anyhow!("palette_size doit etre <= 256"));
    }
    if !(0.0..=1.0).contains(&p.variance_sensitivity) {
        return Err(anyhow!(
            "variance_sensitivity doit etre dans [0, 1], got {}",
            p.variance_sensitivity
        ));
    }
    if p.max_boost < 1.0 {
        return Err(anyhow!("max_boost doit etre >= 1.0, got {}", p.max_boost));
    }
    Ok(())
}

/// Settings specific to halftone.
fn validate_halftone(p: &FilterParams) -> Result<()> {
    if p.algorithm == Algorithm::Halftone && p.halftone == HalftoneMode::Off {
        return Err(anyhow!(
            "l'algorithme Halftone nécessite un mode cmyk ou dominant"
        ));
    }
    if p.halftone_frequency <= 0.0 {
        return Err(anyhow!(
            "halftone_frequency doit etre > 0, got {}",
            p.halftone_frequency
        ));
    }
    if p.halftone_min_radius_ratio < 0.0 {
        return Err(anyhow!(
            "halftone_min_radius_ratio doit etre >= 0, got {}",
            p.halftone_min_radius_ratio
        ));
    }
    if p.halftone_max_dot_ratio <= 0.0 {
        return Err(anyhow!(
            "halftone_max_dot_ratio doit etre > 0, got {}",
            p.halftone_max_dot_ratio
        ));
    }
    // Upper bounds: a halftone radius is not capped by the nearest-neighbor,
    // unlike the pointillist algorithms. Arbitrarily large values (e.g.
    // `--halftone-max-dot 1e9`) would produce dots covering the whole image →
    // a pathologically slow render.
    if p.halftone_min_radius_ratio > 1.0 {
        return Err(anyhow!(
            "halftone_min_radius_ratio doit etre <= 1.0, got {}",
            p.halftone_min_radius_ratio
        ));
    }
    if p.halftone_max_dot_ratio > 10.0 {
        return Err(anyhow!(
            "halftone_max_dot_ratio doit etre <= 10.0, got {}",
            p.halftone_max_dot_ratio
        ));
    }
    Ok(())
}

/// Dot shape.
fn validate_shapes(p: &FilterParams) -> Result<()> {
    if let DotShape::RegularPolygon { sides } = p.dot_shape
        && !(3..=12).contains(&sides)
    {
        return Err(anyhow!(
            "polygon_sides doit etre entre 3 et 12, got {}",
            sides
        ));
    }
    if let DotShape::Ellipse { aspect, .. } = p.dot_shape
        && (aspect <= 0.0 || aspect > 10.0)
    {
        return Err(anyhow!(
            "ellipse_aspect doit etre dans (0, 10], got {}",
            aspect
        ));
    }
    Ok(())
}

/// No infinite or NaN value: they would travel through the whole pipeline
/// unseen, and come out as black pixels or random artifacts.
fn validate_finite(p: &FilterParams) -> Result<()> {
    for (name, value) in [
        ("min_radius_ratio", p.min_radius_ratio),
        ("max_radius_ratio", p.max_radius_ratio),
        ("variance_sensitivity", p.variance_sensitivity),
        ("max_boost", p.max_boost),
        ("grid_angle_deg", p.grid_angle_deg),
        ("halftone_frequency", p.halftone_frequency),
        ("halftone_min_radius_ratio", p.halftone_min_radius_ratio),
        ("halftone_max_dot_ratio", p.halftone_max_dot_ratio),
    ] {
        if !value.is_finite() {
            return Err(anyhow!("{} doit etre fini, got {}", name, value));
        }
    }
    Ok(())
}

/// Checks everything a render refuses to accept.
///
/// Seven named checks instead of a series of 156 inline `if`s. The order of the
/// checks and the messages are unchanged, character for character: only the
/// grouping changes, and each check becomes testable on its own.
/// the grouping changes, and each check becomes testable on its own.
/// the grouping changes, and each check becomes testable on its own.
pub fn validate_params(w: u32, h: u32, params: &FilterParams) -> Result<()> {
    validate_image_dimensions(w, h)?;
    validate_radii(params)?;
    validate_density(params)?;
    validate_palette(params)?;
    validate_halftone(params)?;
    validate_shapes(params)?;
    validate_finite(params)
}

/// Validates the dimensions before decoding the image into memory.
pub fn validate_image_dimensions(w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(anyhow!("Image vide ({}x{})", w, h));
    }
    if w > MAX_IMAGE_DIMENSION || h > MAX_IMAGE_DIMENSION {
        return Err(anyhow!(
            "Image trop grande ({}x{}), maximum autorise: {}x{}",
            w,
            h,
            MAX_IMAGE_DIMENSION,
            MAX_IMAGE_DIMENSION
        ));
    }
    let pixels = (w as u64) * (h as u64);
    if pixels > MAX_IMAGE_PIXELS {
        return Err(anyhow!(
            "Image trop grande ({} pixels), maximum: {} pixels",
            pixels,
            MAX_IMAGE_PIXELS
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A complete preset whose `dithering` line is replaced (or removed when
    /// `line` is empty).
    fn preset_with_dithering(line: &str) -> String {
        let base = FilterParams::default().to_toml_string().unwrap();
        base.lines()
            .filter(|current| !current.starts_with("dithering"))
            .map(|current| {
                if current == "[params]" && !line.is_empty() {
                    format!("{current}\n{line}")
                } else {
                    current.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 0.1 presets wrote `dithering = true`: they must keep producing
    /// Floyd-Steinberg rather than failing to load.
    #[test]
    fn legacy_boolean_dithering_is_still_read() {
        let legacy = preset_with_dithering("dithering = true");
        let params = FilterParams::from_toml_str(&legacy).expect("preset 0.1");
        assert_eq!(params.dithering, Dithering::FloydSteinberg);
    }

    #[test]
    fn dithering_names_round_trip() {
        for (name, expected) in [
            ("off", Dithering::Off),
            ("floyd-steinberg", Dithering::FloydSteinberg),
            ("riemersma", Dithering::Riemersma),
            ("bayer4", Dithering::Bayer4),
            ("bayer8", Dithering::Bayer8),
        ] {
            let text = preset_with_dithering(&format!("dithering = \"{name}\""));
            let params = FilterParams::from_toml_str(&text)
                .unwrap_or_else(|e| panic!("dithering = {name} : {e}"));
            assert_eq!(params.dithering, expected, "{name}");
            // And the written form must stay readable.
            let written = params.to_toml_string().unwrap();
            assert!(
                written.contains(&format!("dithering = \"{name}\"")),
                "écriture : {written}"
            );
        }
    }

    #[test]
    fn missing_dithering_defaults_to_off() {
        let text = preset_with_dithering("");
        let params = FilterParams::from_toml_str(&text).expect("preset sans dithering");
        assert_eq!(params.dithering, Dithering::Off);
    }
}
