use anyhow::{Result, anyhow};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::filter::halftone::{HalftoneConfig, HalftoneMode, Screening};

const PRESET_FORMAT_VERSION: u32 = 1;
pub const MAX_IMAGE_DIMENSION: u32 = 65_535;
pub const MAX_IMAGE_PIXELS: u64 = 8 * 1024 * 1024;

#[derive(Deserialize, Serialize)]
struct PresetDocument {
    preset_version: u32,
    params: FilterParams,
}

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize, ValueEnum)]
pub enum Algorithm {
    Grid,
    Kmeans,
    Voronoi,
    Quadtree,
    /// Halftone multi-canal (rosette CMJN ou couleurs dominantes).
    /// Les sous-paramètres (`HalftoneMode`, `Screening`, `halftone_frequency`,
    /// …) vivent sur `FilterParams`. Quand cet algorithme est sélectionné, les
    /// autres paramètres de placement (`num_points`, `cols`, `iterations`,
    /// `variance_sensitivity`, `max_boost`, `grid_angle_deg`) sont ignorés.
    Halftone,
}

/// Forme des dots dessinés.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
pub enum DotShape {
    /// Disque plein (comportement historique)
    #[default]
    Circle,
    /// Carré plein, côté = 2×rayon
    Square,
    /// Ellipse, `aspect` = ratio width/height ∈ (0, 10], `angle_deg` = rotation en degrés
    Ellipse { aspect: f32, angle_deg: f32 },
    /// Polygone régulier à `sides` côtés (3 = triangle, 4 = losange, 6 = hexagone, …)
    RegularPolygon { sides: u8 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FilterParams {
    pub algorithm: Algorithm,
    /// Nombre de points cible (Kmeans / Voronoi / Quadtree)
    pub num_points: usize,
    /// Nombre de colonnes (Grid uniquement)
    pub cols: u32,
    /// Rayon minimum : fraction de min(largeur, hauteur) — ex. 0.002 = 0.2% de l'image
    pub min_radius_ratio: f32,
    /// Rayon maximum : fraction de min(largeur, hauteur) — ex. 0.08 = 8% de l'image
    pub max_radius_ratio: f32,
    /// Couleur de fond RGB — source de vérité unique (remplace l'ancien bg_white)
    pub bg_color: [u8; 3],
    /// Nombre d'itérations pour Voronoi / Kmeans
    pub iterations: usize,
    /// Sensibilité à la variance locale (0 = ignorée, 1 = forte redistribution)
    pub variance_sensitivity: f32,
    /// Multiplicateur maximum de la taille dans les zones uniformes (1.0 = pas de boost)
    pub max_boost: f32,
    /// Graine du RNG pour reproductibilité (None = aléatoire basé sur l'horloge)
    pub rng_seed: Option<u64>,
    /// Nombre de couleurs pour la quantification (None = pas de quantification)
    pub palette_size: Option<usize>,
    /// Forme des dots
    pub dot_shape: DotShape,
    /// Fond transparent en sortie (RGBA, alpha 0 entre les dots).
    /// Quand vrai, `bg_color` est ignoré pour le rendu final (les zones non couvertes
    /// restent transparentes). Il reste utilisé pour flattener l'image source vers
    /// RGB (computations des moyennes de couleur des dots).
    pub transparent: bool,
    /// Correction gamma des moyennes de couleur des dots. Quand vrai, les
    /// moyennes sont effectuées en espace linéaire (sRGB→linear→moyenne→sRGB),
    /// ce qui évite le biais "trop sombre" des zones contrastées (ex: bord de
    /// dégradé). Coût : 2 passes de l'image (line´arisation source + ré-encodage dst).
    pub gamma_correct: bool,
    /// Dithering Floyd-Steinberg lors de la quantification de palette. Quand
    /// vrai (et `palette_size` est Some(n)), l'image rendue finale est
    /// post-quantifiée en propageant l'erreur par pixel — effet "offset print".
    /// Ignoré si `palette_size` est None.
    pub dithering: bool,
    /// Angle de rotation de la grille (degrés), appliqué à l'algorithme Grid
    /// uniquement. Les dots sont placés sur une grille pivotée autour du centre
    /// de l'image — effet "halftone screen" expérimental. La couleur reste la
    /// moyenne de la cellule non rotationnée. 0 = pas de rotation (défaut).
    pub grid_angle_deg: f32,
    /// Mode halftone multi-canaux (rosette CMJK, couleurs dominantes, ou off).
    pub halftone: HalftoneMode,
    /// Screening (placement AM grille rotationnée ou FM blue noise).
    pub screening: Screening,
    /// Fréquence de trame AM en "cells per min(W,H)". 60 → 60 dots par la plus
    /// petite dimension (~trame 60 lpp à 1 dpi).
    pub halftone_frequency: f32,
    /// Rayon minimum du dot halftone (fraction de `min(W,H)`).
    pub halftone_min_radius_ratio: f32,
    /// Rayon maximum du dot halftone (fraction du step de trame).
    pub halftone_max_dot_ratio: f32,
}

impl FilterParams {
    /// Le fond est-il clair ? Dérivé de `bg_color` (faux si `transparent`).
    pub fn is_bg_light(&self) -> bool {
        if self.transparent {
            return false;
        }
        let lum = 0.2126 * self.bg_color[0] as f32
            + 0.7152 * self.bg_color[1] as f32
            + 0.0722 * self.bg_color[2] as f32;
        lum > 127.5
    }

    /// Construit une `HalftoneConfig` prête à passer au moteur halftone.
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

    /// Sérialise les paramètres en une chaîne TOML, prête à écrire dans un fichier.
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

    /// Reconstruit les paramètres depuis une chaîne TOML.
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

        // Les presets plats produits avant la version 1 restent lisibles.
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
            dithering: false,
            grid_angle_deg: 0.0,
            halftone: HalftoneMode::Off,
            screening: Screening::Am,
            halftone_frequency: 60.0,
            halftone_min_radius_ratio: 0.002,
            halftone_max_dot_ratio: 0.85,
        }
    }
}

/// Un point résultant du filtre
#[derive(Clone, Debug)]
pub struct Dot {
    pub x: f32,
    pub y: f32,
    pub color: [u8; 3],
    pub radius: f32,
}

pub fn validate_params(w: u32, h: u32, params: &FilterParams) -> Result<()> {
    validate_image_dimensions(w, h)?;
    if params.min_radius_ratio <= 0.0 {
        return Err(anyhow!("min_radius_ratio doit etre > 0"));
    }
    if params.max_radius_ratio < params.min_radius_ratio {
        return Err(anyhow!(
            "max_radius_ratio ({}) doit etre >= min_radius_ratio ({})",
            params.max_radius_ratio,
            params.min_radius_ratio
        ));
    }
    // A radius larger than the image scale is not meaningful.
    if params.max_radius_ratio > 1.0 {
        return Err(anyhow!(
            "max_radius_ratio ({}) doit etre <= 1.0",
            params.max_radius_ratio
        ));
    }
    if params.num_points == 0 {
        return Err(anyhow!("num_points doit etre > 0"));
    }
    if params.num_points > 100_000 {
        return Err(anyhow!("num_points doit etre <= 100000"));
    }
    if matches!(params.algorithm, Algorithm::Kmeans | Algorithm::Voronoi)
        && params.num_points > 50_000
    {
        return Err(anyhow!(
            "num_points doit etre <= 50000 pour K-means/Voronoi"
        ));
    }
    if params.cols == 0 {
        return Err(anyhow!("cols doit etre > 0"));
    }
    if params.cols > 8192 {
        return Err(anyhow!("cols doit etre <= 8192"));
    }
    if params.iterations > 100 {
        return Err(anyhow!("iterations doit etre <= 100"));
    }
    if let Some(n) = params.palette_size
        && n < 2
    {
        return Err(anyhow!("palette_size doit etre >= 2, got {}", n));
    }
    if !(0.0..=1.0).contains(&params.variance_sensitivity) {
        return Err(anyhow!(
            "variance_sensitivity doit etre dans [0, 1], got {}",
            params.variance_sensitivity
        ));
    }
    if params.palette_size.is_some_and(|n| n > 256) {
        return Err(anyhow!("palette_size doit etre <= 256"));
    }
    if params.max_boost < 1.0 {
        return Err(anyhow!(
            "max_boost doit etre >= 1.0, got {}",
            params.max_boost
        ));
    }
    if params.algorithm == Algorithm::Halftone && params.halftone == HalftoneMode::Off {
        return Err(anyhow!(
            "l'algorithme Halftone nécessite un mode cmyk ou dominant"
        ));
    }
    if params.halftone_frequency <= 0.0 {
        return Err(anyhow!(
            "halftone_frequency doit etre > 0, got {}",
            params.halftone_frequency
        ));
    }
    if params.halftone_min_radius_ratio < 0.0 {
        return Err(anyhow!(
            "halftone_min_radius_ratio doit etre >= 0, got {}",
            params.halftone_min_radius_ratio
        ));
    }
    if params.halftone_max_dot_ratio <= 0.0 {
        return Err(anyhow!(
            "halftone_max_dot_ratio doit etre > 0, got {}",
            params.halftone_max_dot_ratio
        ));
    }
    if let DotShape::RegularPolygon { sides } = params.dot_shape
        && !(3..=12).contains(&sides)
    {
        return Err(anyhow!(
            "polygon_sides doit etre entre 3 et 12, got {}",
            sides
        ));
    }
    if let DotShape::Ellipse { aspect, .. } = params.dot_shape
        && (aspect <= 0.0 || aspect > 10.0)
    {
        return Err(anyhow!(
            "ellipse_aspect doit etre dans (0, 10], got {}",
            aspect
        ));
    }
    for (name, value) in [
        ("min_radius_ratio", params.min_radius_ratio),
        ("max_radius_ratio", params.max_radius_ratio),
        ("variance_sensitivity", params.variance_sensitivity),
        ("max_boost", params.max_boost),
        ("grid_angle_deg", params.grid_angle_deg),
        ("halftone_frequency", params.halftone_frequency),
        (
            "halftone_min_radius_ratio",
            params.halftone_min_radius_ratio,
        ),
        ("halftone_max_dot_ratio", params.halftone_max_dot_ratio),
    ] {
        if !value.is_finite() {
            return Err(anyhow!("{} doit etre une valeur finie", name));
        }
    }
    match params.dot_shape {
        DotShape::Ellipse { aspect, angle_deg }
            if !aspect.is_finite() || !angle_deg.is_finite() =>
        {
            return Err(anyhow!("les parametres de l'ellipse doivent etre finis"));
        }
        _ => {}
    }
    match &params.halftone {
        HalftoneMode::Dominant { n, base_angle_deg } => {
            if *n > 32 {
                return Err(anyhow!("le nombre de couleurs dominantes doit etre <= 32"));
            }
            if !base_angle_deg.is_finite() {
                return Err(anyhow!("base_angle_deg doit etre une valeur finie"));
            }
        }
        HalftoneMode::Cmyk { angles } if angles.iter().any(|angle| !angle.is_finite()) => {
            return Err(anyhow!("les angles CMJN doivent etre finis"));
        }
        _ => {}
    }
    Ok(())
}

/// Valide les dimensions avant de decoder l'image en memoire.
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
