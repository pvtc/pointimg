//! Helpers partagés entre les binaires (CLI, egui et GTK4).
//!
//! Les binaires `pointimg`, `pointimg-gui` et `pointimg-gtk` sont des crates
//! distinctes et ne peuvent pas partager du code `pub(crate)`. Ce module expose
//! donc les helpers communs — formatage, normalisation des chemins de sortie,
//! écritures atomiques, export RGBA et presets — depuis la bibliothèque.
//!
//! Il est public parce que les binaires en ont besoin, mais il ne fait pas
//! partie de l'API stable de `pointimg`.

use crate::filter::FilterParams;
use image::{RgbImage, RgbaImage};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Éléments partagés par les deux interfaces ───────────────────────────────

/// Extensions d'images acceptées par les deux interfaces (ouverture, dossier,
/// glisser-déposer).
pub const IMAGE_EXTENSIONS: [&str; 7] = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];

/// Vrai si le chemin pointe vers un fichier image connu des interfaces.
pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Décrit une image pour l'interface (« 1400×1050 · 1,5 Mpx »).
pub fn describe_image_size(width: u32, height: u32) -> String {
    let megapixels = f64::from(width) * f64::from(height) / 1_000_000.0;
    format!("{width}×{height} · {megapixels:.1} Mpx")
}

/// Mode d'affichage de la zone d'aperçu, commun aux deux interfaces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    /// Source et résultat superposés, séparés par un curseur.
    Compare,
    /// Source et résultat côte à côte.
    Side,
    /// Résultat seul.
    ResultOnly,
    /// Source seule.
    SourceOnly,
    /// Carte de détail local.
    DensityMap,
}

/// Nombre de modes d'affichage (cyclés par `Tab` et par le menu).
pub const VIEW_COUNT: u32 = 5;

/// Position du mode dans les listes déroulantes.
pub fn view_index(mode: ViewMode) -> u32 {
    match mode {
        ViewMode::Compare => 0,
        ViewMode::Side => 1,
        ViewMode::ResultOnly => 2,
        ViewMode::SourceOnly => 3,
        ViewMode::DensityMap => 4,
    }
}

/// Mode correspondant à une position de liste déroulante.
pub fn view_from_index(index: u32) -> ViewMode {
    match index {
        1 => ViewMode::Side,
        2 => ViewMode::ResultOnly,
        3 => ViewMode::SourceOnly,
        4 => ViewMode::DensityMap,
        _ => ViewMode::Compare,
    }
}

/// Résolution de calcul choisie par l'utilisateur.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    /// Aperçu à 1/2 puis finale automatique au repos.
    Half,
    /// Aperçu à 1/4 puis finale automatique au repos.
    Quarter,
    /// Toujours en pleine résolution.
    Full,
}

impl Quality {
    /// Facteur de sous-échantillonnage appliqué à l'image source.
    pub fn scale(self) -> f32 {
        match self {
            Quality::Half => 0.5,
            Quality::Quarter => 0.25,
            Quality::Full => 1.0,
        }
    }

    /// Le mode correspond-il à un calcul en pleine résolution ?
    pub fn is_full(self) -> bool {
        self == Quality::Full
    }
}

/// Image d'exemple proposée dans l'état d'accueil.
///
/// Cherche les quelques images livrées avec le projet à côté du binaire (usage
/// depuis les sources) ; renvoie `None` pour une installation qui ne les
/// contient pas, auquel cas l'interface masque le bouton.
pub fn sample_image() -> Option<PathBuf> {
    const NAMES: [&str; 3] = [
        "pexels-ruyan-ayten-153760-4190725.jpg",
        "pexels-kofishelbyfotos-38152015.jpg",
        "kilauea25_0.jpeg",
    ];
    let exe = std::env::current_exe().ok()?;
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(dir) = exe.parent() {
        roots.push(dir.to_path_buf());
        if let Some(parent) = dir.parent() {
            roots.push(parent.to_path_buf());
            if let Some(grand) = parent.parent() {
                roots.push(grand.to_path_buf());
            }
        }
    }
    roots.push(std::env::current_dir().unwrap_or_default());
    for root in roots {
        for name in NAMES {
            for candidate in [
                root.join("assets/examples").join(name),
                root.join("examples").join(name),
            ] {
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

// ─── Formatage ───────────────────────────────────────────────────────────────

/// Formate une durée en millisecondes de façon lisible.
pub fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        let secs = ms / 1000;
        format!("{}m {}s", secs / 60, secs % 60)
    }
}

/// Formate une taille mémoire en octets de façon lisible.
pub fn format_memory(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} Go", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{} Mo", bytes / (1024 * 1024))
    }
}

/// Vrai si la génération de worker attendue est toujours la génération courante.
pub fn generation_is_current(generation: &AtomicU64, expected: u64) -> bool {
    generation.load(Ordering::Acquire) == expected
}

/// Décrit en une phrase le changement de paramètres le plus notable, pour les
/// libellés d'undo/redo.
pub fn describe_parameter_change(before: &FilterParams, after: &FilterParams) -> String {
    if before.algorithm != after.algorithm {
        return format!(
            "Algorithme : {:?} → {:?}",
            before.algorithm, after.algorithm
        );
    }
    if before.num_points != after.num_points {
        return format!(
            "Nombre de points : {} → {}",
            before.num_points, after.num_points
        );
    }
    if before.cols != after.cols {
        return format!("Colonnes : {} → {}", before.cols, after.cols);
    }
    if before.iterations != after.iterations {
        return format!("Itérations : {} → {}", before.iterations, after.iterations);
    }
    if before.dot_shape != after.dot_shape {
        return "Forme des points modifiée".to_string();
    }
    if before.palette_size != after.palette_size {
        return "Palette modifiée".to_string();
    }
    if before.bg_color != after.bg_color || before.transparent != after.transparent {
        return "Fond modifié".to_string();
    }
    if before.gamma_correct != after.gamma_correct {
        return "Correction gamma modifiée".to_string();
    }
    if before.halftone != after.halftone || before.screening != after.screening {
        return "Paramètres halftone modifiés".to_string();
    }
    "Paramètres modifiés".to_string()
}

// ─── Chemins de sortie ───────────────────────────────────────────────────────

fn with_allowed_extension(mut path: PathBuf, default_ext: &str, allowed: &[&str]) -> PathBuf {
    let keep = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| allowed.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false);
    if !keep {
        path.set_extension(default_ext);
    }
    path
}

/// Normalise l'extension d'un chemin de sortie image : une extension reconnue
/// est conservée, sinon `default_ext` est appliquée.
pub fn ensure_image_extension(path: PathBuf, default_ext: &str) -> PathBuf {
    with_allowed_extension(
        path,
        default_ext,
        &["png", "jpg", "jpeg", "svg", "webp", "bmp", "tif", "tiff"],
    )
}

/// Force l'extension `.svg` (un export SVG ne doit pas atterrir dans un `.png`).
pub fn ensure_svg_extension(path: PathBuf) -> PathBuf {
    with_allowed_extension(path, "svg", &["svg"])
}

/// Force l'extension `.toml` (presets).
pub fn ensure_toml_extension(path: PathBuf) -> PathBuf {
    with_allowed_extension(path, "toml", &["toml"])
}

/// Vrai si le format de sortie (déduit de l'extension) préserve l'alpha.
pub fn extension_preserves_alpha(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "webp" | "tif" | "tiff")
    )
}

// ─── Conversions d'images ────────────────────────────────────────────────────

/// Compose une image RGBA sur un fond opaque (formats sans alpha : JPEG, BMP).
/// Arrondit au lieu de tronquer, comme le compositing du CLI.
pub fn flatten_rgba_on_bg(img: &RgbaImage, bg: [u8; 3]) -> RgbImage {
    let (w, h) = img.dimensions();
    RgbImage::from_fn(w, h, |x, y| {
        let p = img.get_pixel(x, y);
        let a = p[3] as f32 / 255.0;
        let inv = 1.0 - a;
        image::Rgb([
            (p[0] as f32 * a + bg[0] as f32 * inv).round() as u8,
            (p[1] as f32 * a + bg[1] as f32 * inv).round() as u8,
            (p[2] as f32 * a + bg[2] as f32 * inv).round() as u8,
        ])
    })
}

/// Convertit une `RgbImage` en `RgbaImage` opaque (alpha = 255 partout).
pub fn rgb_to_rgba_opaque(img: &RgbImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels() {
        out.put_pixel(x, y, image::Rgba([p[0], p[1], p[2], 255]));
    }
    out
}

// ─── Écritures atomiques ─────────────────────────────────────────────────────

/// Chemin temporaire adjacent à `path`, unique par processus/appel.
pub fn temporary_path(path: &Path) -> PathBuf {
    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("png");
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    path.with_file_name(format!(
        ".{name}.pointimg-{}-{timestamp}-{id}.tmp.{ext}",
        std::process::id()
    ))
}

/// Remplace `to` par `from` (supprime la cible d'abord sous Windows).
pub fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    if to.exists() {
        std::fs::remove_file(to)?;
    }
    std::fs::rename(from, to)
}

/// Écrit une image via un fichier temporaire puis un remplacement atomique.
/// Le fichier temporaire est supprimé en cas d'erreur.
pub fn atomic_image_save<F>(path: &Path, save: F) -> image::ImageResult<()>
where
    F: FnOnce(&Path) -> image::ImageResult<()>,
{
    let tmp = temporary_path(path);
    if let Err(error) = save(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = replace_file(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(image::ImageError::IoError(error));
    }
    Ok(())
}

/// Écrit du texte via un fichier temporaire puis un remplacement atomique.
pub fn atomic_text_write(path: &Path, contents: &str) -> std::io::Result<()> {
    let tmp = temporary_path(path);
    if let Err(error) = std::fs::write(&tmp, contents) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = replace_file(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    Ok(())
}

// ─── Export de haut niveau ───────────────────────────────────────────────────

/// Sauvegarde un résultat RGBA en respectant les capacités du format :
/// PNG/WebP/TIFF préservent l'alpha ; JPEG/BMP compositent sur `bg` quand le
/// rendu est transparent, sinon convertissent simplement en RGB.
pub fn save_rgba_image(
    path: &Path,
    img: &RgbaImage,
    transparent: bool,
    bg: [u8; 3],
) -> image::ImageResult<()> {
    atomic_image_save(path, |tmp| {
        if extension_preserves_alpha(path) {
            img.save(tmp)
        } else if transparent {
            flatten_rgba_on_bg(img, bg).save(tmp)
        } else {
            image::DynamicImage::ImageRgba8(img.clone())
                .to_rgb8()
                .save(tmp)
        }
    })
}

/// Sérialise `params` en TOML et l'écrit de façon atomique.
pub fn save_preset(path: &Path, params: &FilterParams) -> Result<(), String> {
    let contents = params.to_toml_string().map_err(|e| e.to_string())?;
    atomic_text_write(path, &contents).map_err(|e| e.to_string())
}

/// Lit et désérialise un preset TOML.
pub fn load_preset(path: &Path) -> Result<FilterParams, String> {
    let contents = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    FilterParams::from_toml_str(&contents).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Algorithm;

    #[test]
    fn history_labels_changed_algorithm() {
        let before = FilterParams::default();
        let after = FilterParams {
            algorithm: Algorithm::Grid,
            ..before.clone()
        };
        assert!(describe_parameter_change(&before, &after).contains("Algorithme"));
    }

    #[test]
    fn memory_format_is_human_readable() {
        assert_eq!(format_memory(8 * 1024 * 1024), "8 Mo");
    }

    #[test]
    fn duration_format_scales() {
        assert_eq!(format_duration(500), "500ms");
        assert_eq!(format_duration(1500), "1.5s");
        assert_eq!(format_duration(125_000), "2m 5s");
    }

    #[test]
    fn stale_worker_generation_is_rejected() {
        let generation = AtomicU64::new(7);
        assert!(generation_is_current(&generation, 7));
        generation.store(8, Ordering::Release);
        assert!(!generation_is_current(&generation, 7));
    }

    #[test]
    fn view_index_round_trips() {
        for view in [
            ViewMode::Compare,
            ViewMode::Side,
            ViewMode::ResultOnly,
            ViewMode::SourceOnly,
            ViewMode::DensityMap,
        ] {
            assert_eq!(view_from_index(view_index(view)), view);
        }
    }

    #[test]
    fn quality_scales_and_flags_full_resolution() {
        assert_eq!(Quality::Half.scale(), 0.5);
        assert_eq!(Quality::Quarter.scale(), 0.25);
        assert_eq!(Quality::Full.scale(), 1.0);
        assert!(Quality::Full.is_full());
        assert!(!Quality::Half.is_full());
    }

    #[test]
    fn supported_images_are_recognized_by_extension() {
        assert!(is_supported_image(Path::new("photo.JPG")));
        assert!(is_supported_image(Path::new("scan.tiff")));
        assert!(!is_supported_image(Path::new("notes.txt")));
        assert!(!is_supported_image(Path::new("sans_extension")));
    }

    #[test]
    fn image_size_is_described_with_megapixels() {
        assert_eq!(describe_image_size(1400, 1050), "1400×1050 · 1.5 Mpx");
    }

    #[test]
    fn image_extension_is_normalized() {
        assert_eq!(
            ensure_image_extension(PathBuf::from("out.jpg"), "png"),
            PathBuf::from("out.jpg")
        );
        assert_eq!(
            ensure_image_extension(PathBuf::from("out"), "png"),
            PathBuf::from("out.png")
        );
        assert_eq!(
            ensure_image_extension(PathBuf::from("out.xyz"), "png"),
            PathBuf::from("out.png")
        );
    }

    #[test]
    fn svg_and_toml_extensions_are_forced() {
        assert_eq!(
            ensure_svg_extension(PathBuf::from("out.png")),
            PathBuf::from("out.svg")
        );
        assert_eq!(
            ensure_toml_extension(PathBuf::from("preset.json")),
            PathBuf::from("preset.toml")
        );
    }

    #[test]
    fn alpha_capable_formats_detected() {
        assert!(extension_preserves_alpha(Path::new("a.PNG")));
        assert!(extension_preserves_alpha(Path::new("a.webp")));
        assert!(!extension_preserves_alpha(Path::new("a.jpg")));
        assert!(!extension_preserves_alpha(Path::new("a.bmp")));
    }

    #[test]
    fn flatten_composites_on_background() {
        let img = RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 128]));
        let flat = flatten_rgba_on_bg(&img, [255, 255, 255]);
        // Rouge à 50% sur blanc → rose clair, jamais tronqué vers le bas.
        assert_eq!(flat.get_pixel(0, 0)[0], 255);
        assert!(flat.get_pixel(0, 0)[1] >= 127);
    }

    #[test]
    fn preset_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "pointimg-preset-test-{}-{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let params = FilterParams {
            algorithm: Algorithm::Grid,
            cols: 42,
            ..FilterParams::default()
        };
        save_preset(&path, &params).unwrap();
        let loaded = load_preset(&path).unwrap();
        assert_eq!(loaded.algorithm, Algorithm::Grid);
        assert_eq!(loaded.cols, 42);
        let _ = std::fs::remove_file(path);
    }
}
