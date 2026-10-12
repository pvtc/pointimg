//! Generates the preset thumbnails in `assets/presets/`.
//!
//! Each thumbnail is the rendering of a sample image by a built-in preset.
//! The same image serves for all of them: comparing two presets only makes
//! sense on the same source.
//!
//! The thumbnails are generated from `filter::builtin_presets()` itself, and
//! not from hand-copied command line options - a modified preset would give an
//! up-to-date thumbnail without any option being able to diverge from it. A
//! test (`src/preset_thumbs.rs`) checks that there is indeed one thumbnail per
//! preset.
//!
//! Usage: `cargo run --example preset_thumbnails`

use std::path::{Path, PathBuf};

use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use pointimg::filter::{self, FilterParams};

/// Thumbnail size. Big enough for a HiDPI screen in a list of ~48 px,
/// small enough for the nine files to stay light.
const LARGEUR: u32 = 240;
const HAUTEUR: u32 = 180;

/// Source image.
///
/// A portrait, not a landscape: pointillism is judged on a face, and
/// this is the only sample in the repository where one sees what a preset does
/// with a recognizable subject. The more contrasted volcanic scene gave
/// almost black thumbnails - the CMYK Rosette most of all.
const SOURCE: &str = "assets/examples/pexels-kofishelbyfotos-38152015.jpg";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let racine = racine_du_projet()?;
    let sortie = racine.join("assets/presets");
    std::fs::create_dir_all(&sortie)?;

    let source = redimensionner(
        &image::open(racine.join(SOURCE))?.to_rgb8(),
        LARGEUR,
        HAUTEUR,
    )?;

    let presets = filter::builtin_presets();
    for (index, preset) in presets.iter().enumerate() {
        let rendu = filtre(&source, &preset.params)?;
        let chemin: PathBuf = sortie.join(format!("{:02}-{}.png", index, fichier_sur(preset.name)));
        rendu.save(&chemin)?;
        println!(
            "{:<22} {} ({}×{})",
            preset.name,
            chemin.display(),
            rendu.width(),
            rendu.height()
        );
    }
    println!("\n{} vignettes dans {}", presets.len(), sortie.display());
    Ok(())
}

/// Applies a preset. The fast mode is disabled: a thumbnail must
/// show the final rendering, not a preview.
fn filtre(src: &RgbImage, params: &FilterParams) -> Result<RgbImage, Box<dyn std::error::Error>> {
    // The presets are validated by the library; a failure here would signal a
    // broken preset, not a badly rendered thumbnail.
    filter::validate_params(src.width(), src.height(), params)
        .map_err(|e| format!("preset invalide : {e}"))?;
    Ok(filter::apply(src, params)?)
}

/// Resizes by cropping the requested ratio, then to the target size.
///
/// A crop rather than a stretch: the thumbnail shows the same framing the
/// user will see, and a distorted portrait would make the preset judged unfairly.
///
/// The vertical anchor is the upper third rather than the middle. As the
/// source is a vertical portrait, a centered crop would cut the face in two;
/// the upper-third anchor keeps it whole inside the frame.
fn redimensionner(
    src: &RgbImage,
    largeur: u32,
    hauteur: u32,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let ratio = f64::from(largeur) / f64::from(hauteur);
    let (mut l, mut h) = (src.width(), src.height());
    if f64::from(l) / f64::from(h) > ratio {
        l = (f64::from(h) * ratio).round() as u32;
    } else {
        h = (f64::from(l) / ratio).round() as u32;
    }
    let haut = (src.height() - h) / 3;
    let recadre = image::DynamicImage::ImageRgb8(src.clone()).crop_imm(0, haut, l, h);
    let reduit = imageops::resize(&recadre, largeur, hauteur, FilterType::Lanczos3);
    Ok(image::DynamicImage::ImageRgba8(reduit).to_rgb8())
}

/// File name: lowercase, without accents or spaces.
fn fichier_sur(nom: &str) -> String {
    nom.to_lowercase()
        .chars()
        .map(|c| match c {
            'a'..='z' | '0'..='9' => c,
            'à' | 'â' | 'ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'î' | 'ï' => 'i',
            'ô' | 'ö' => 'o',
            'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            ' ' | '-' | '(' | ')' | ',' | '\'' => '-',
            _ => '-',
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// Repository root, deduced from the directory of the example binary
/// (`target/<profile>/examples/preset_thumbnails`).
fn racine_du_projet() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let mut chemin: &Path = exe.as_path();
    for _ in 0..4 {
        if let Some(parent) = chemin.parent() {
            chemin = parent;
        }
    }
    if !chemin.join("Cargo.toml").is_file() {
        return Err(format!("racine du dépôt introuvable depuis {}", exe.display()).into());
    }
    Ok(chemin.to_path_buf())
}
