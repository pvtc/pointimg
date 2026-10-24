//! Helpers shared between the binaries (CLI, egui and GTK4).
//!
//! The `pointimg`, `pointimg-gui` and `pointimg-gtk` binaries are distinct
//! crates and cannot share `pub(crate)` code. This module therefore exposes
//! the common helpers — formatting, normalization of the output paths, atomic
//! writes, RGBA export and presets — from the library.
//!
//! It is public because the binaries need it, but it is not part of the stable
//! API of `pointimg`.

use crate::filter::{self, FilterParams};
use image::{RgbImage, RgbaImage};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Elements shared by the two interfaces ───────────────────────────────────

/// Image extensions accepted by the two interfaces (opening, folder,
/// drag-and-drop).
/// Extensions recognized by the interfaces.
///
/// `gif` appears here because the CLI already accepted it (`main.rs` kept its
/// own list, one entry ahead) and because `image::open` decodes it: an animated
/// GIF gives its first frame, which is exactly the promise of a still-image
/// render. A single list, so no more divergence possible between the
/// interfaces and the CLI.
pub const IMAGE_EXTENSIONS: [&str; 8] = ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"];

/// True when the path points to an image file known to the interfaces.
pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Describes an image for the interface ("1400×1050 · 1.5 Mpx").
pub fn describe_image_size(width: u32, height: u32) -> String {
    let megapixels = f64::from(width) * f64::from(height) / 1_000_000.0;
    format!("{width}×{height} · {megapixels:.1} Mpx")
}

/// Display mode of the preview area, common to both interfaces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    /// Source and result overlaid, separated by a slider.
    Compare,
    /// Source and result side by side.
    Side,
    /// Result only.
    ResultOnly,
    /// Source only.
    SourceOnly,
    /// Local detail map.
    DensityMap,
}

/// Number of display modes (cycled by `Tab` and by the menu).
pub const VIEW_COUNT: u32 = 5;

/// Position of the mode in the dropdown lists.
pub fn view_index(mode: ViewMode) -> u32 {
    match mode {
        ViewMode::Compare => 0,
        ViewMode::Side => 1,
        ViewMode::ResultOnly => 2,
        ViewMode::SourceOnly => 3,
        ViewMode::DensityMap => 4,
    }
}

/// Mode matching a dropdown list position.
///
/// Falls back on the first mode explicitly, rather than through a `_ =>` that
/// would hide a list that got out of sync with [`VIEW_COUNT`].
pub fn view_from_index(index: u32) -> ViewMode {
    match index {
        1 => ViewMode::Side,
        2 => ViewMode::ResultOnly,
        3 => ViewMode::SourceOnly,
        4 => ViewMode::DensityMap,
        // Zero being the first mode, the rule and the boundary case coincide.
        _ => ViewMode::Compare,
    }
}

// ─── Variant tables ──────────────────────────────────────────────────────────
//
// The dropdown lists of the two interfaces span the same enums. Each
// conversion direction used to be written in every interface, with a `_ =>`
// for the "impossible case" — and those cases did happen:
// `algorithm_from_index` fell back on `Halftone`, which moreover requires an
// ink mode, so a list got out of sync with its enum and produced a failed
// render; `dithering_from_index` fell back on `Off`, so a shifted list
// disabled dithering without saying so.
//
// The tables below are the only list. Both directions follow from them, and
// an unknown index gives `None`: the caller decides explicitly what to do,
// instead of inheriting a variant chosen by a `_`.

/// Order of the placement algorithms in the dropdown lists.
pub const ALGORITHMS: [filter::Algorithm; 5] = [
    filter::Algorithm::Grid,
    filter::Algorithm::Kmeans,
    filter::Algorithm::Voronoi,
    filter::Algorithm::Quadtree,
    filter::Algorithm::Halftone,
];

/// Position of an algorithm. `None` if it is not in the table — which cannot
/// happen for an enum variant, and which a test verifies.
pub fn algorithm_index(algorithm: filter::Algorithm) -> Option<u32> {
    ALGORITHMS
        .iter()
        .position(|entree| *entree == algorithm)
        .map(|index| index as u32)
}

/// Algorithm at that position, or `None` if the position does not exist.
pub fn algorithm_of(index: u32) -> Option<filter::Algorithm> {
    ALGORITHMS.get(index as usize).copied()
}

/// Order of the dithering algorithms in the dropdown lists.
pub const DITHERINGS: [filter::Dithering; 5] = [
    filter::Dithering::Off,
    filter::Dithering::FloydSteinberg,
    filter::Dithering::Riemersma,
    filter::Dithering::Bayer4,
    filter::Dithering::Bayer8,
];

/// Position of a dithering. See [`algorithm_index`].
pub fn dithering_index(dithering: filter::Dithering) -> Option<u32> {
    DITHERINGS
        .iter()
        .position(|entree| *entree == dithering)
        .map(|index| index as u32)
}

/// Dithering at that position, or `None`.
pub fn dithering_of(index: u32) -> Option<filter::Dithering> {
    DITHERINGS.get(index as usize).copied()
}

/// Order of the quality levels.
pub const QUALITIES: [Quality; 3] = [Quality::Half, Quality::Quarter, Quality::Full];

/// Position of a quality level. See [`algorithm_index`].
pub fn quality_index(quality: Quality) -> Option<u32> {
    QUALITIES
        .iter()
        .position(|entree| *entree == quality)
        .map(|index| index as u32)
}

/// Quality level at that position, or `None`.
pub fn quality_of(index: u32) -> Option<Quality> {
    QUALITIES.get(index as usize).copied()
}

/// Compute resolution chosen by the user.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    /// Preview at 1/2 then automatic final render at rest.
    Half,
    /// Preview at 1/4 then automatic final render at rest.
    Quarter,
    /// Always at full resolution.
    Full,
}

impl Quality {
    /// Downsampling factor applied to the source image.
    pub fn scale(self) -> f32 {
        match self {
            Quality::Half => 0.5,
            Quality::Quarter => 0.25,
            Quality::Full => 1.0,
        }
    }

    /// Does the mode correspond to a full-resolution computation?
    pub fn is_full(self) -> bool {
        self == Quality::Full
    }
}

/// Distribution of a value along a slider.
///
/// A linear track would be unusable on the setting ranges of pointimg:
/// the minimum radius (0.0005) and the maximum one (1.0) are three orders of
/// magnitude apart, and the halftone range (0 → 10) is dominated by zero.
/// Both interfaces use it to place the handle — and to deduce the value of a
/// click on the rail, GTK having no logarithmic slider.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Curve {
    /// Proportional to the range, for already uniform ranges (0 → 1).
    Linear,
    /// Logarithm, for entirely positive ranges (0.0005 → 1.0).
    Log,
    /// Square root, for ranges that start at 0 (0 → 10): the logarithm is
    /// undefined there, and the linear one confines the useful values to the
    /// first tenth of the rail.
    Sqrt,
}

impl Curve {
    /// Relative position (0 → 1) of a value inside the range.
    ///
    /// Values outside the range are brought back inside the bounds — including
    /// a negative or zero value passed to [`Curve::Log`], for which the
    /// logarithm would be undefined.
    pub fn position(self, value: f64, start: f64, end: f64) -> f64 {
        let t = match self {
            Curve::Linear => (value - start) / (end - start),
            Curve::Log => (value.max(0.0).ln() - start.ln()) / (end.ln() - start.ln()),
            Curve::Sqrt => (value / end).max(0.0).sqrt(),
        };
        t.clamp(0.0, 1.0)
    }

    /// Value of the range matching a relative position (0 → 1).
    pub fn value(self, position: f64, start: f64, end: f64) -> f64 {
        let t = position.clamp(0.0, 1.0);
        match self {
            Curve::Linear => start + t * (end - start),
            Curve::Log => (start.ln() + t * (end.ln() - start.ln())).exp(),
            Curve::Sqrt => t * t * end,
        }
        .clamp(start.min(end), end.max(start))
    }
}

/// Sample image offered in the welcome state.
///
/// Looks for the few images shipped with the project next to the binary (use
/// from the sources); returns `None` for an installation that does not
/// contain them, in which case the interface hides the button.
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

/// Thumbnail of the preset `index`, if it is present.
///
/// The thumbnails are generated by `cargo run --example preset_thumbnails`,
/// which renders each built-in preset on a sample image. They are optional:
/// an installation without the `assets/presets` folder simply displays the
/// lists without an image.
///
/// This is only a *file lookup*: each interface then decides on its own
/// widgets — a grid in egui, `ActionRow`s in GTK.
pub fn preset_thumbnail(index: usize) -> Option<PathBuf> {
    let prefix = format!("{index:02}-");
    let dir = asset_dir("presets")?;
    let entree = std::fs::read_dir(&dir).ok()?;
    let mut trouvee: Option<PathBuf> = None;
    for chemin in entree.flatten() {
        let nom = chemin.file_name().to_string_lossy().into_owned();
        if nom.starts_with(&prefix) && nom.ends_with(".png") {
            // A single file per prefix: the first one will do.
            if trouvee.is_none() {
                trouvee = Some(chemin.path());
            }
        }
    }
    trouvee
}

/// Draws a random seed.
///
/// Both interfaces need it for "Varier la graine", and the identical formula
/// on both sides (`RandomState` on the clock): a seed drawn differently
/// from one interface to the next would make no sense, and two copies leave
/// room for a divergence.
pub fn random_seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0),
    );
    hasher.finish()
}

/// Export file name derived from the source image: `photo.jpg` →
/// `photo-pointimg.png`.
///
/// The formula used to be written in five places (two `default_name`, two
/// quick exports, a quick SVG). The `pointimg.<ext>` fallback is explicitly
/// parachuted in: an image with no name must not make the export fail.
pub fn derived_export_name(src: Option<&Path>, extension: &str) -> String {
    src.and_then(|path| path.file_stem())
        .and_then(|stem| stem.to_str())
        .map(|stem| format!("{stem}-pointimg.{extension}"))
        .unwrap_or_else(|| format!("pointimg.{extension}"))
}

/// Index of the built-in preset matching these settings exactly.
///
/// This is the question "ai-je personnalisé un preset ?", asked three
/// times in each interface. A single definition prevents a fourth call from
/// reinventing the comparison.
pub fn matching_preset_index(params: &filter::FilterParams) -> Option<usize> {
    filter::builtin_presets()
        .iter()
        .position(|preset| preset.params == *params)
}

/// Bounds of the preview zoom. Same span on both sides: why offer
/// 8× on one side and 32× on the other for the same wheel gesture?
/// Smallest preview zoom: 2 % of the real size.
pub const ZOOM_MIN: f32 = 0.02;
/// Largest preview zoom: 16×.
pub const ZOOM_MAX: f32 = 16.0;

/// Zoom factor for the keyboard and the buttons.
///
/// Wheel zoom remains specific to GTK (egui has none), so its factor is not
/// here — there is nothing to share as long as it does not exist on one side.
pub const ZOOM_STEP: f32 = 1.25;

/// Messages displayed by the two interfaces, word for word.
///
/// These are not widgets: they are strings. Duplicating them guarantees that
/// one day one will say "Rien à annuler." and the other "Aucune annulation
/// possible", for the same action. Each interface then chooses its way of
/// displaying it (status line on one side, ephemeral notification on the
/// other).
pub mod msg {
    /// A computation is in progress.
    pub fn computing() -> &'static str {
        super::t("Calcul en cours…")
    }
    /// The displayed render predates the last setting change.
    pub fn stale() -> &'static str {
        super::t("aperçu obsolète — recalcul en cours")
    }
    /// The full-resolution pass follows the reduced preview.
    pub fn finalizing() -> &'static str {
        super::t("pleine résolution…")
    }
    /// No previous state to restore.
    pub fn nothing_to_undo() -> &'static str {
        super::t("Rien à annuler.")
    }
    /// An undo has just been applied.
    pub fn undone() -> &'static str {
        super::t("Annulé.")
    }
    /// No undone state to redo.
    pub fn nothing_to_redo() -> &'static str {
        super::t("Rien à rétablir.")
    }
    /// A redo has just been applied.
    pub fn redone() -> &'static str {
        super::t("Rétabli.")
    }
    /// The settings are already the default ones.
    pub fn already_default() -> &'static str {
        super::t("Réglages déjà par défaut.")
    }
    /// No image is loaded.
    pub fn open_first() -> &'static str {
        super::t("Ouvrez d'abord une image.")
    }
    /// No render to copy.
    pub fn nothing_to_copy() -> &'static str {
        super::t("Aucun résultat à copier.")
    }
    /// The render is in the clipboard.
    pub fn copied() -> &'static str {
        super::t("Image copiée dans le presse-papiers.")
    }
    /// The user gave up on overwriting a file.
    pub fn save_cancelled() -> &'static str {
        super::t("Sauvegarde annulée.")
    }
}

/// Looks for an assets folder near the binary then in the current directory.
///
/// Same search as [`sample_image`]: a run from the sources finds `assets/`, an
/// installation finds a folder next to the binary.
fn asset_dir(nom: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(dir) = exe.parent() {
        roots.push(dir.to_path_buf());
        let mut courant = dir.to_path_buf();
        for _ in 0..3 {
            let Some(parent) = courant.parent() else {
                break;
            };
            courant = parent.to_path_buf();
            roots.push(courant.clone());
        }
    }
    roots.push(std::env::current_dir().unwrap_or_default());
    for root in roots {
        let candidat = root.join("assets").join(nom);
        if candidat.is_dir() {
            return Some(candidat);
        }
    }
    None
}

/// Subsamples an image for a quick preview.
///
/// Same filter (`Triangle`) and same rounding on both sides: two previews of
/// the same image must show the same pixels, otherwise comparing what a
/// setting produces in one interface against the other would make no sense.
/// The scale >= 1 and the finalization pass return the image as is (clone
/// included).
pub fn downsample(src: &RgbImage, scale: f32) -> RgbImage {
    if scale >= 1.0 {
        return src.clone();
    }
    let (w, h) = src.dimensions();
    let target_w = ((f64::from(w) * f64::from(scale)).round() as u32).max(1);
    let target_h = ((f64::from(h) * f64::from(scale)).round() as u32).max(1);
    image::imageops::resize(
        src,
        target_w,
        target_h,
        image::imageops::FilterType::Triangle,
    )
}

/// Images of a folder, sorted by name, or the error that explains why empty.
///
/// The scan used to be written twice, down to the error messages — a folder
/// without an image had to state the same thing in the two interfaces. Only
/// the way of displaying it remains specific to each one.
pub fn collect_folder_images(dir: &Path) -> Result<Vec<PathBuf>, FolderError> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Err(FolderError::Unreadable(dir.to_path_buf()));
    };
    let mut images: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|candidate| is_supported_image(candidate))
        .collect();
    images.sort();
    if images.is_empty() {
        return Err(FolderError::Empty(dir.to_path_buf()));
    }
    Ok(images)
}

/// Why a folder gave no images.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FolderError {
    /// The folder could not be read.
    Unreadable(PathBuf),
    /// The folder is readable but contains no recognized image.
    Empty(PathBuf),
}

impl std::fmt::Display for FolderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FolderError::Unreadable(dir) => write!(f, "Dossier illisible : {}", dir.display()),
            FolderError::Empty(dir) => write!(f, "Aucune image dans {}", dir.display()),
        }
    }
}

// ─── Transparency checker ────────────────────────────────────────────────────

/// Side of one cell of the transparency checker, in pixels.
pub const CHECKER_SIZE: u32 = 8;
/// Light gray of a cell.
pub const CHECKER_LIGHT: f32 = 200.0;
/// Dark gray of a cell.
pub const CHECKER_DARK: f32 = 240.0;

/// Checker gray under the pixel `(x, y)`.
///
/// The two interfaces and the comparison panel share the same definition:
/// they must display the same checker for the same image.
#[inline]
pub fn checker_shade(x: u32, y: u32) -> f32 {
    if (x / CHECKER_SIZE + y / CHECKER_SIZE).is_multiple_of(2) {
        CHECKER_LIGHT
    } else {
        CHECKER_DARK
    }
}

/// Composites an RGBA pixel on the checker, and returns the opaque
/// `[r, g, b]`.
///
/// The order of the bytes on output belongs to each interface: egui stacks
/// `Color32`s, GTK R8g8b8a8 bytes, cairo BGRX. The *color* itself is unique —
/// it is the only thing this module imposes.
#[inline]
pub fn composite_on_checker(pixel: [u8; 4], x: u32, y: u32) -> [u8; 3] {
    let alpha = f32::from(pixel[3]) / 255.0;
    let inv = 1.0 - alpha;
    let bg = checker_shade(x, y);
    [
        (f32::from(pixel[0]) * alpha + bg * inv + 0.5) as u8,
        (f32::from(pixel[1]) * alpha + bg * inv + 0.5) as u8,
        (f32::from(pixel[2]) * alpha + bg * inv + 0.5) as u8,
    ]
}

// ─── Formatting ──────────────────────────────────────────────────────────────

/// Formats a duration in milliseconds in a readable way.
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

/// Formats a memory size in bytes in a readable way.
pub fn format_memory(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} Go", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{} Mo", bytes / (1024 * 1024))
    }
}

/// Formats a file size with the useful precision: "381 kB", "2.1 MB".
/// [`format_memory`] stays in whole units for the memory estimates, where the
/// figure is approximate anyway.
pub fn format_bytes(bytes: u64) -> String {
    const KO: f64 = 1024.0;
    const MO: f64 = 1024.0 * 1024.0;
    const GO: f64 = 1024.0 * 1024.0 * 1024.0;
    let value = bytes as f64;
    if value >= GO {
        format!("{:.1} Go", value / GO)
    } else if value >= MO {
        format!("{:.1} Mo", value / MO)
    } else if value >= KO {
        format!("{:.0} Ko", value / KO)
    } else {
        format!("{bytes} o")
    }
}

/// True when the expected worker generation is still the current generation.
pub fn generation_is_current(generation: &AtomicU64, expected: u64) -> bool {
    generation.load(Ordering::Acquire) == expected
}

/// Describes in one sentence the most notable parameter change, for the
/// undo/redo labels.
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
        return t("Forme des points modifiée").to_string();
    }
    if before.palette_size != after.palette_size {
        return t("Palette modifiée").to_string();
    }
    if before.bg_color != after.bg_color || before.transparent != after.transparent {
        return t("Fond modifié").to_string();
    }
    if before.gamma_correct != after.gamma_correct {
        return t("Correction gamma modifiée").to_string();
    }
    if before.halftone != after.halftone || before.screening != after.screening {
        return t("Paramètres halftone modifiés").to_string();
    }
    t("Paramètres modifiés").to_string()
}

// ─── Output paths ────────────────────────────────────────────────────────────

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

/// Normalizes the extension of an image output path: a recognized extension
/// is kept, otherwise `default_ext` is applied.
pub fn ensure_image_extension(path: PathBuf, default_ext: &str) -> PathBuf {
    with_allowed_extension(
        path,
        default_ext,
        &["png", "jpg", "jpeg", "svg", "webp", "bmp", "tif", "tiff"],
    )
}

/// Forces the `.svg` extension (an SVG export must not land in a `.png`).
pub fn ensure_svg_extension(path: PathBuf) -> PathBuf {
    with_allowed_extension(path, "svg", &["svg"])
}

/// Forces the `.toml` extension (presets).
pub fn ensure_toml_extension(path: PathBuf) -> PathBuf {
    with_allowed_extension(path, "toml", &["toml"])
}

/// True when the output format (deduced from the extension) preserves alpha.
pub fn extension_preserves_alpha(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "webp" | "tif" | "tiff")
    )
}

// ─── Image conversions ───────────────────────────────────────────────────────

/// Composites an RGBA image on an opaque background (formats without alpha:
/// JPEG, BMP). Rounds instead of truncating, like the CLI's compositing.
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

/// Converts an `RgbImage` into an opaque `RgbaImage` (alpha = 255 everywhere).
pub fn rgb_to_rgba_opaque(img: &RgbImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels() {
        out.put_pixel(x, y, image::Rgba([p[0], p[1], p[2], 255]));
    }
    out
}

// ─── Atomic writes ───────────────────────────────────────────────────────────

/// Temporary path adjacent to `path`, unique per process/call.
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

/// Replaces `to` with `from` (removes the target first on Windows).
pub fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    if to.exists() {
        std::fs::remove_file(to)?;
    }
    std::fs::rename(from, to)
}

/// Writes to a temporary file, then replaces the target atomically.
/// The temporary file is deleted in case of failure, in both phases.
///
/// The skeleton used to be copied three times (image, bytes, text). The
/// copies differed by the type of error they return — `image::ImageError`
/// for the first two, `std::io::Error` for text — and only the first one
/// reported the write error as is: a change of this cleanup (for example
/// keeping the folder of a failure) would not be applied to more than one
/// copy. The type of error is therefore a parameter, and the three public
/// entry points keep their signature.
fn atomic_write<E, F>(
    path: &Path,
    write: F,
    on_rename_error: fn(std::io::Error) -> E,
) -> Result<(), E>
where
    F: FnOnce(&Path) -> Result<(), E>,
{
    let tmp = temporary_path(path);
    if let Err(error) = write(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = replace_file(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(on_rename_error(error));
    }
    Ok(())
}

/// Writes an image via a temporary file then an atomic replacement.
/// The temporary file is deleted in case of error.
pub fn atomic_image_save<F>(path: &Path, save: F) -> image::ImageResult<()>
where
    F: FnOnce(&Path) -> image::ImageResult<()>,
{
    atomic_write(path, save, image::ImageError::IoError)
}

/// Writes bytes via a temporary file then an atomic replacement.
pub fn atomic_bytes_write(path: &Path, bytes: &[u8]) -> image::ImageResult<()> {
    atomic_write(
        path,
        |tmp| std::fs::write(tmp, bytes).map_err(image::ImageError::IoError),
        image::ImageError::IoError,
    )
}

/// Writes text via a temporary file then an atomic replacement.
pub fn atomic_text_write(path: &Path, contents: &str) -> std::io::Result<()> {
    atomic_write(path, |tmp| std::fs::write(tmp, contents), |error| error)
}

// ─── High-level export ───────────────────────────────────────────────────────

/// Saves an RGBA result while respecting the capabilities of the format:
/// PNG/WebP/TIFF preserve alpha; JPEG/BMP composite on `bg` when the
/// render is transparent, otherwise simply convert to RGB.
pub fn save_rgba_image(
    path: &Path,
    img: &RgbaImage,
    transparent: bool,
    bg: [u8; 3],
) -> image::ImageResult<()> {
    // The export goes through in-memory encoding: it is the same code as
    // [`measure_export_sizes`], so the announced size is that of the file
    // really written.
    let bytes = encode_rgba_image(path, img, transparent, bg)?;
    atomic_bytes_write(path, &bytes)
}

/// Encodes an RGBA result in the format designated by the extension of
/// `path`, in memory.
///
/// Formats without alpha (JPEG, BMP): compositing on `bg` when the render is
/// transparent, simple conversion to RGB otherwise. Formats with alpha (PNG,
/// WebP, TIFF): image kept as is.
pub fn encode_rgba_image(
    path: &Path,
    img: &RgbaImage,
    transparent: bool,
    bg: [u8; 3],
) -> image::ImageResult<Vec<u8>> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    encode_rgba_extension(extension, img, transparent, bg)
}

/// Same thing, from the extension alone (without the dot): this is what
/// the measurement of the sizes needs, since no path has been chosen yet.
pub fn encode_rgba_extension(
    extension: &str,
    img: &RgbaImage,
    transparent: bool,
    bg: [u8; 3],
) -> image::ImageResult<Vec<u8>> {
    let format = image_format(extension)?;
    let alpha = format_preserves_alpha(extension);
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    if alpha {
        img.write_to(&mut cursor, format)?;
    } else if transparent {
        flatten_rgba_on_bg(img, bg).write_to(&mut cursor, format)?;
    } else {
        image::DynamicImage::ImageRgba8(img.clone())
            .to_rgb8()
            .write_to(&mut cursor, format)?;
    }
    Ok(buffer)
}

/// `image` format matching an extension (without the dot).
fn image_format(extension: &str) -> image::ImageResult<image::ImageFormat> {
    match extension.to_ascii_lowercase().as_str() {
        "png" => Ok(image::ImageFormat::Png),
        "jpg" | "jpeg" => Ok(image::ImageFormat::Jpeg),
        "webp" => Ok(image::ImageFormat::WebP),
        "bmp" => Ok(image::ImageFormat::Bmp),
        "tif" | "tiff" => Ok(image::ImageFormat::Tiff),
        _ => Err(image::ImageError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("format d'export non pris en charge : {extension}"),
        ))),
    }
}

fn format_preserves_alpha(extension: &str) -> bool {
    matches!(extension, "png" | "webp" | "tif" | "tiff")
}

/// Serializes `params` as TOML and writes it atomically.
pub fn save_preset(path: &Path, params: &FilterParams) -> Result<(), String> {
    let contents = params.to_toml_string().map_err(|e| e.to_string())?;
    atomic_text_write(path, &contents).map_err(|e| e.to_string())
}

/// Reads and deserializes a TOML preset.
pub fn load_preset(path: &Path) -> Result<FilterParams, String> {
    let contents = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    FilterParams::from_toml_str(&contents).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Algorithm;

    #[test]
    fn curve_round_trips_over_its_range() {
        for (curve, (start, end)) in [
            (Curve::Linear, (-180.0_f64, 180.0)),
            (Curve::Log, (0.0005_f64, 1.0_f64)),
            (Curve::Sqrt, (0.0, 10.0)),
        ] {
            for value in [start, (start + end) / 2.0, end] {
                let back = curve.value(curve.position(value, start, end), start, end);
                assert!(
                    (back - value).abs() <= end * 1e-6,
                    "{curve:?} : {value} est revenu {back}"
                );
            }
        }
    }

    #[test]
    fn curve_positions_are_clamped() {
        assert_eq!(Curve::Log.position(-1.0, 0.0005, 1.0), 0.0);
        assert_eq!(Curve::Log.position(99.0, 0.0005, 1.0), 1.0);
        assert_eq!(Curve::Sqrt.value(2.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn log_curve_spreads_small_values() {
        // 0.06 in 0.0005 → 1.0: halfway up the three orders of magnitude,
        // where the linear one would glue it to the edge of the slider.
        let position = Curve::Log.position(0.06, 0.0005, 1.0);
        assert!((0.55..=0.7).contains(&position), "position : {position}");
    }

    #[test]
    fn sqrt_curve_uses_the_first_tenth() {
        // 0.85 on 0 → 10: 29 % of the slider instead of 8 %.
        let position = Curve::Sqrt.position(0.85, 0.0, 10.0);
        assert!((0.28..=0.3).contains(&position), "position : {position}");
    }

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

    /// Every table must contain every variant of its enum, exactly once,
    /// and the two directions of the conversions must agree. Adding a variant
    /// without listing it makes this test fail — it is the only safeguard
    /// against a shifted list, which a `_ =>` would let through in silence.
    #[test]
    fn les_tables_couvrent_toutes_leurs_variantes() {
        use filter::{Algorithm, Dithering};
        for algorithm in [
            Algorithm::Grid,
            Algorithm::Kmeans,
            Algorithm::Voronoi,
            Algorithm::Quadtree,
            Algorithm::Halftone,
        ] {
            let index = algorithm_index(algorithm).expect("variante absente de la table");
            assert_eq!(algorithm_of(index), Some(algorithm), "aller-retour cassé");
            // A single occurrence: twice the same variant would shift everything.
            assert_eq!(
                ALGORITHMS.iter().filter(|a| **a == algorithm).count(),
                1,
                "{algorithm:?} figure plus d'une fois"
            );
        }
        for dithering in [
            Dithering::Off,
            Dithering::FloydSteinberg,
            Dithering::Riemersma,
            Dithering::Bayer4,
            Dithering::Bayer8,
        ] {
            let index = dithering_index(dithering).expect("variante absente de la table");
            assert_eq!(dithering_of(index), Some(dithering), "aller-retour cassé");
            assert_eq!(
                DITHERINGS.iter().filter(|d| **d == dithering).count(),
                1,
                "{dithering:?} figure plus d'une fois"
            );
        }
        for quality in [Quality::Half, Quality::Quarter, Quality::Full] {
            let index = quality_index(quality).expect("variante absente de la table");
            assert_eq!(quality_of(index), Some(quality), "aller-retour cassé");
            assert_eq!(QUALITIES.iter().filter(|q| **q == quality).count(), 1);
        }
        // And an out-of-bounds index returns nothing, instead of choosing.
        assert_eq!(algorithm_of(99), None);
        assert_eq!(dithering_of(99), None);
        assert_eq!(quality_of(99), None);
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
        // Red at 50% on white → light pink, never truncated downwards.
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

    /// The checker alternates over a period of two cells, and the composite
    /// follows the alpha: opaque = the pixel, transparent = the gray of the
    /// cell.
    #[test]
    fn le_damier_alterne_et_suit_lalpha() {
        // Along the same row, the background color changes from one cell to
        // the next.
        assert_eq!(checker_shade(0, 0), CHECKER_LIGHT);
        assert_eq!(checker_shade(CHECKER_SIZE, 0), CHECKER_DARK);
        assert_eq!(checker_shade(0, 0), checker_shade(2 * CHECKER_SIZE, 0));
        // And along a column too: the formula combines both axes.
        assert_eq!(
            checker_shade(0, 0),
            checker_shade(CHECKER_SIZE, CHECKER_SIZE)
        );
        assert_eq!(
            checker_shade(CHECKER_SIZE, 0),
            checker_shade(0, CHECKER_SIZE)
        );

        // Opaque: the pixel itself, the checker does not show through.
        assert_eq!(
            composite_on_checker([10, 200, 30, 255], 0, 0),
            [10, 200, 30],
            "un pixel opaque ne dépend pas du damier"
        );
        // Transparent: exactly the gray of the cell.
        assert_eq!(composite_on_checker([9, 9, 9, 0], 0, 0).len(), 3);
        assert_eq!(
            composite_on_checker([9, 9, 9, 0], CHECKER_SIZE, 0),
            [CHECKER_DARK as u8, CHECKER_DARK as u8, CHECKER_DARK as u8]
        );
        // Semi-transparent: between the two, whatever the channel.
        let demi = composite_on_checker([0, 0, 0, 128], 0, 0);
        assert!(
            demi[0] > 0 && demi[0] < CHECKER_LIGHT as u8,
            "un pixel à moitié opaque doit tomber entre le noir et le gris clair, pas à {}",
            demi[0]
        );
        // The three channels undergo the same treatment.
        assert_eq!(demi[0], demi[1]);
        assert_eq!(demi[1], demi[2]);
    }

    /// The derived export name gives the same thing with or without a source
    /// file, and falls back cleanly when there is none.
    #[test]
    fn le_nom_derive_suit_le_repli() {
        use std::path::Path;
        assert_eq!(
            derived_export_name(Some(Path::new("/photos/vacances.jpg")), "png"),
            "vacances-pointimg.png"
        );
        assert_eq!(derived_export_name(None, "svg"), "pointimg.svg");
        // A file name without extension: the stem is enough.
        assert_eq!(
            derived_export_name(Some(Path::new("/tmp/photo")), "png"),
            "photo-pointimg.png"
        );
    }

    /// A readable folder with no known image is not "unreadable": the
    /// two messages are distinct, and the two interfaces display them as is.
    #[test]
    fn un_dossier_vide_se_distingue_dun_dossier_illisible() {
        let dir = std::env::temp_dir().join(format!(
            "pointimg-folder-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), b"pas une image").unwrap();

        let e = collect_folder_images(&dir).unwrap_err();
        assert_eq!(
            e,
            FolderError::Empty(dir.clone()),
            "du texte n'est pas une image"
        );
        assert!(
            e.to_string().contains("Aucune image"),
            "le message doit expliquer le vide, pas l'illisibilité : {e}"
        );

        assert!(collect_folder_images(&dir.join("inexistant")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Downsampling respects the requested scale, and does not move
    /// when the scale does not reduce anything.
    #[test]
    fn le_sous_echantillonnage_respecte_lechelle() {
        let img = image::RgbImage::new(200, 100);
        assert_eq!(downsample(&img, 1.0).dimensions(), (200, 100));
        assert_eq!(downsample(&img, 2.0).dimensions(), (200, 100));
        let reduit = downsample(&img, 0.5);
        assert_eq!(reduit.dimensions(), (100, 50));
        // Never a zero dimension, even at an absurd scale.
        assert_eq!(downsample(&img, 0.0001).dimensions(), (1, 1));
    }

    /// The three atomic writes share the same cleanup: a temporary
    /// must never survive a failure, and the replacement must have happened.
    #[test]
    fn les_ecritures_atomiques_nettoyent_leur_temporaire() {
        let dir = std::env::temp_dir().join(format!(
            "pointimg-atomic-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cible = dir.join("texte.txt");

        atomic_text_write(&cible, "bonjour").expect("écriture");
        assert_eq!(std::fs::read_to_string(&cible).unwrap(), "bonjour");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "le temporaire a disparu après le remplacement"
        );

        // Overwriting: the replacement must have happened, not only the
        // creation.
        atomic_text_write(&cible, "au revoir").expect("réécriture");
        assert_eq!(std::fs::read_to_string(&cible).unwrap(), "au revoir");

        // Write failure: `write` fails, so nothing must remain.
        let vide = dir.join("vide.bin");
        let echec: image::ImageResult<()> = atomic_write(
            &vide,
            |_| Err(image::ImageError::IoError(std::io::Error::other("refusé"))),
            image::ImageError::IoError,
        );
        assert!(echec.is_err());
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "un temporaire d'un échec doit être supprimé"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ─── Internationalisation ────────────────────────────────────────────────────

/// Interface language supported by the application.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Locale {
    /// English.
    #[default]
    En,
    /// French.
    Fr,
}

impl Locale {
    /// Short code used in preferences and the system environment.
    pub fn code(self) -> &'static str {
        match self {
            Locale::En => "en",
            Locale::Fr => "fr",
        }
    }

    /// Parses a code; anything that is not `fr*` becomes English.
    pub fn from_code(code: &str) -> Self {
        let code = code.trim().to_ascii_lowercase();
        if code.starts_with("fr") {
            Locale::Fr
        } else {
            Locale::En
        }
    }
}

static LOCALE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Currently selected interface language.
pub fn locale() -> Locale {
    match LOCALE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Locale::Fr,
        _ => Locale::En,
    }
}

/// Changes the interface language. The change is immediate for egui (which
/// rebuilds every frame); GTK applies it on the next launch.
pub fn set_locale(locale: Locale) {
    LOCALE.store(
        match locale {
            Locale::En => 0,
            Locale::Fr => 1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Guesses the language from the environment (`LC_ALL`, `LC_MESSAGES`, `LANG`).
pub fn detect_system_locale() -> Locale {
    for var in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(value) = std::env::var(var) {
            let value = value.to_ascii_lowercase();
            if value.starts_with("fr") {
                return Locale::Fr;
            }
            if !value.is_empty() && value != "c" && value != "posix" {
                return Locale::En;
            }
        }
    }
    Locale::En
}

include!("i18n_table.rs");

/// Translates a source string. The source string is the French text: it is
/// returned as-is in French, and looked up in the table in English. An entry
/// missing from the table falls back to the source string, so a not-yet
/// translated string never disappears.
pub fn t(key: &'static str) -> &'static str {
    match locale() {
        Locale::Fr => key,
        Locale::En => TRANSLATIONS
            .iter()
            .find(|(source, _)| *source == key)
            .map(|(_, translation)| *translation)
            .unwrap_or(key),
    }
}
