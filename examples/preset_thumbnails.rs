//! Génère les vignettes des presets dans `assets/presets/`.
//!
//! Chaque vignette est le rendu d'une image d'exemple par un preset intégré.
//! La même image sert pour tous : comparer deux presets n'a de sens que sur la
//! même source.
//!
//! Les vignettes sont générées depuis `filter::builtin_presets()` lui-même, et
//! non depuis des options de ligne de commande recopiées — un preset modifié
//! donnerait une vignette à jour sans qu'aucune option puisse en diverger. Un
//! test (`src/preset_thumbs.rs`) vérifie qu'il y a bien une vignette par
//! preset.
//!
//! Usage : `cargo run --example preset_thumbnails`

use std::path::{Path, PathBuf};

use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use pointimg::filter::{self, FilterParams};

/// Taille de la vignette. Assez grand pour un écran HiDPI dans une liste de
/// ~48 px, assez petit pour que les neuf fichiers restent légers.
const LARGEUR: u32 = 240;
const HAUTEUR: u32 = 180;

/// Image source.
///
/// Un portrait, et non un paysage : le pointillisme se juge sur un visage, et
/// c'est le seul exemple du dépôt où l'on voit ce qu'un preset fait d'un sujet
/// reconnaissable. La scène volcanique, plus contrastée, donnait des vignettes
/// presque noires — la Rosette CMJN surtout.
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
        let chemin: PathBuf =
            sortie.join(format!("{:02}-{}.png", index, fichier_sur(&preset.name)));
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

/// Applique un preset. Le mode rapide est désactivé : une vignette doit
/// montrer le rendu final, pas un aperçu.
fn filtre(src: &RgbImage, params: &FilterParams) -> Result<RgbImage, Box<dyn std::error::Error>> {
    // Les presets sont validés par la bibliothèque ; un échec ici signalerait
    // un preset cassé, pas une vignette mal rendue.
    filter::validate_params(src.width(), src.height(), params)
        .map_err(|e| format!("preset invalide : {e}"))?;
    Ok(filter::apply(src, params)?)
}

/// Redimensionne en roster le ratio demandé, puis à la taille.
///
/// Un recadrage plutôt qu'un étirement : la vignette montre le même cadrage que
/// l'utilisateur verra, et un portrait déformé ferait juger le preset à tort.
///
/// L'ancrage vertical est au tiers haut plutôt qu'au milieu. La source étant
/// un portrait vertical, un recadrage centré couperait le visage en deux ;
/// l'ancrage au tiers le garde entier dans le cadre.
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

/// Nom de fichier : minuscules, sans accents ni espaces.
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

/// Racine du dépôt, déduite du répertoire du binaire d'exemple
/// (`target/<profil>/examples/preset_thumbnails`).
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
