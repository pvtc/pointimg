//! Helpers partagés entre les binaires (CLI, egui et GTK4).
//!
//! Les binaires `pointimg`, `pointimg-gui` et `pointimg-gtk` sont des crates
//! distinctes et ne peuvent pas partager du code `pub(crate)`. Ce module expose
//! donc les helpers communs — formatage, normalisation des chemins de sortie,
//! écritures atomiques, export RGBA et presets — depuis la bibliothèque.
//!
//! Il est public parce que les binaires en ont besoin, mais il ne fait pas
//! partie de l'API stable de `pointimg`.

use crate::filter::{self, FilterParams};
use image::{RgbImage, RgbaImage};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Éléments partagés par les deux interfaces ───────────────────────────────

/// Extensions d'images acceptées par les deux interfaces (ouverture, dossier,
/// glisser-déposer).
/// Extensions reconnues par les interfaces.
///
/// `gif` figure ici parce que la CLI l'acceptait déjà (`main.rs` tenait sa
/// propre liste, en avance d'une entrée) et que `image::open` le décode :
/// un GIF animé donne son premier plan, ce qui est exactement la promesse
/// d'un rendu sur image fixe. Une seule liste, donc plus de divergence
/// possible entre les interfaces et la CLI.
pub const IMAGE_EXTENSIONS: [&str; 8] = ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"];

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
///
/// Retombe sur le premier mode, explicitement, plutôt que via un `_ =>` qui
/// masquerait une liste désynchronisée de [`VIEW_COUNT`].
pub fn view_from_index(index: u32) -> ViewMode {
    match index {
        1 => ViewMode::Side,
        2 => ViewMode::ResultOnly,
        3 => ViewMode::SourceOnly,
        4 => ViewMode::DensityMap,
        // Le zéro étant le premier mode, la règle et le cas limite coïncident.
        _ => ViewMode::Compare,
    }
}

// ─── Tables de variantes ────────────────────────────────────────────────────
//
// Les listes déroulantes des deux interfaces portent sur les mêmes enums. Chaque
// sens de conversion était écrit dans chaque interface, avec un `_ =>` pour
// « cas impossible » — et ces cas sont arrivés : `algorithm_from_index`
// retombait sur `Halftone`, qui exige de plus un mode d'encre, donc une liste
// désynchronisée de son enum produisait un rendu en échec ; `dithering_from_index`
// retombait sur `Off`, donc une liste décalée désactivait le tramage sans le dire.
//
// Les tables ci-dessous sont la seule liste. Les deux sens en découlent, et un
// index inconnu donne `None` : l'appelant décide explicitement quoi faire, au
// lieu d'hériter d'une variante choisie par un `_`.

/// Ordre des algorithmes de placement dans les listes déroulantes.
pub const ALGORITHMS: [filter::Algorithm; 5] = [
    filter::Algorithm::Grid,
    filter::Algorithm::Kmeans,
    filter::Algorithm::Voronoi,
    filter::Algorithm::Quadtree,
    filter::Algorithm::Halftone,
];

/// Position d'un algorithme. `None` s'il n'est pas dans la table — ce qui ne peut
/// pas arriver pour une variante de l'enum, et qu'un test vérifie.
pub fn algorithm_index(algorithm: filter::Algorithm) -> Option<u32> {
    ALGORITHMS
        .iter()
        .position(|entree| *entree == algorithm)
        .map(|index| index as u32)
}

/// Algorithme à cette position, ou `None` si la position n'existe pas.
pub fn algorithm_of(index: u32) -> Option<filter::Algorithm> {
    ALGORITHMS.get(index as usize).copied()
}

/// Ordre des algorithmes de tramage dans les listes déroulantes.
pub const DITHERINGS: [filter::Dithering; 5] = [
    filter::Dithering::Off,
    filter::Dithering::FloydSteinberg,
    filter::Dithering::Riemersma,
    filter::Dithering::Bayer4,
    filter::Dithering::Bayer8,
];

/// Position d'un tramage. Voir [`algorithm_index`].
pub fn dithering_index(dithering: filter::Dithering) -> Option<u32> {
    DITHERINGS
        .iter()
        .position(|entree| *entree == dithering)
        .map(|index| index as u32)
}

/// Tramage à cette position, ou `None`.
pub fn dithering_of(index: u32) -> Option<filter::Dithering> {
    DITHERINGS.get(index as usize).copied()
}

/// Ordre des niveaux de qualité.
pub const QUALITIES: [Quality; 3] = [Quality::Half, Quality::Quarter, Quality::Full];

/// Position d'un niveau de qualité. Voir [`algorithm_index`].
pub fn quality_index(quality: Quality) -> Option<u32> {
    QUALITIES
        .iter()
        .position(|entree| *entree == quality)
        .map(|index| index as u32)
}

/// Niveau de qualité à cette position, ou `None`.
pub fn quality_of(index: u32) -> Option<Quality> {
    QUALITIES.get(index as usize).copied()
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

/// Répartition d'une valeur le long d'un curseur.
///
/// Une course linéaire serait inexploitable sur les plages de réglages de
/// pointimg : le rayon minimum (0,0005) et le maximum (1,0) sont séparés de
/// trois ordres de grandeur, et la plage du halftone (0 → 10) est à dominante
/// zéro. Les deux interfaces s'en servent pour placer la poignée — et pour
/// déduire la valeur d'un clic sur le rail, GTK n'ayant pas de curseur
/// logarithmique.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Curve {
    /// Proportionnelle à la plage, pour les plages déjà homogènes (0 → 1).
    Linear,
    /// Logarithme, pour les plages entièrement positives (0,0005 → 1,0).
    Log,
    /// Racine, pour les plages qui commencent à 0 (0 → 10) : le logarithme y
    /// est indéfini, et le linéaire confine les valeurs utiles dans le premier
    /// dixième du rail.
    Sqrt,
}

impl Curve {
    /// Position relative (0 → 1) d'une valeur à l'intérieur de la plage.
    ///
    /// Les valeurs hors plage sont ramenées dans les bornes — y compris une
    /// valeur négative ou nulle passée à [`Curve::Log`], pour laquelle le
    /// logarithme serait indéfini.
    pub fn position(self, value: f64, start: f64, end: f64) -> f64 {
        let t = match self {
            Curve::Linear => (value - start) / (end - start),
            Curve::Log => (value.max(0.0).ln() - start.ln()) / (end.ln() - start.ln()),
            Curve::Sqrt => (value / end).max(0.0).sqrt(),
        };
        t.clamp(0.0, 1.0)
    }

    /// Valeur de la plage correspondant à une position relative (0 → 1).
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

/// Vignette du preset `index`, si elle est présente.
///
/// Les vignettes sont générées par `cargo run --example preset_thumbnails`, qui
/// rend chaque preset intégré sur une image d'exemple. Elles sont optionnelles :
/// une installation sans le dossier `assets/presets` affiche simplement les
/// listes sans image.
///
/// Ce n'est qu'une *recherche de fichier* : chaque interface décide ensuite de
/// ses propres widgets — une grille dans egui, des `ActionRow` dans GTK.
pub fn preset_thumbnail(index: usize) -> Option<PathBuf> {
    let prefix = format!("{index:02}-");
    let dir = asset_dir("presets")?;
    let entree = std::fs::read_dir(&dir).ok()?;
    let mut trouvee: Option<PathBuf> = None;
    for chemin in entree.flatten() {
        let nom = chemin.file_name().to_string_lossy().into_owned();
        if nom.starts_with(&prefix) && nom.ends_with(".png") {
            // Un seul fichier par préfixe : le premier convient.
            if trouvee.is_none() {
                trouvee = Some(chemin.path());
            }
        }
    }
    trouvee
}

/// Tire une graine aléatoire.
///
/// Les deux interfaces en ont besoin pour « Varier la graine », et la formule
/// identique des deux côtés (`RandomState` sur l'horloge) : une graine tirée
/// différemment d'une interface à l'autre n'aurait aucun sens, et deux copies
/// laissent la place à une divergence.
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

/// Nom de fichier d'export dérivé de l'image source : `photo.jpg` →
/// `photo-pointimg.png`.
///
/// La formule était écrite à cinq endroits (deux `default_name`, deux exports
/// rapides, un SVG rapide). Le repli `pointimg.<ext>` est explicitement parachuté
/// : une image sans nom ne doit pas faire échouer l'export.
pub fn derived_export_name(src: Option<&Path>, extension: &str) -> String {
    src.and_then(|path| path.file_stem())
        .and_then(|stem| stem.to_str())
        .map(|stem| format!("{stem}-pointimg.{extension}"))
        .unwrap_or_else(|| format!("pointimg.{extension}"))
}

/// Index du preset intégré correspondant exactement à ces réglages.
///
/// C'est la question « ai-je personnalisé un preset ? », posée trois fois dans
/// chaque interface. Une seule définition évite qu'un quatrième appel
/// réinvente la comparaison.
pub fn matching_preset_index(params: &filter::FilterParams) -> Option<usize> {
    filter::builtin_presets()
        .iter()
        .position(|preset| preset.params == *params)
}

/// Bornes du zoom d'aperçu. Même étendue des deux côtés : à quoi bon proposer
/// 8× d'un côté et 32× de l'autre pour le même geste de molette ?
/// Plus petit zoom d'aperçu : 2 % de la taille réelle.
pub const ZOOM_MIN: f32 = 0.02;
/// Plus grand zoom d'aperçu : 16×.
pub const ZOOM_MAX: f32 = 16.0;

/// Facteur de zoom au clavier et aux boutons.
///
/// Le zoom à la molette reste propre à GTK (egui n'en a pas), donc son facteur
/// n'est pas ici — il n'y a rien à partager tant qu'il n'existe pas d'un côté.
pub const ZOOM_STEP: f32 = 1.25;

/// Messages affichés par les deux interfaces, mot pour mot.
///
/// Ce ne sont pas des widgets : ce sont des chaînes. Les dupliquer garantit
/// qu'un jour l'une dira « Rien à annuler. » et l'autre « Aucune annulation
/// possible », pour la même action. Chaque interface choisit ensuite son
/// moyen de l'afficher (ligne d'état d'un côté, notification éphémère de
/// l'autre).
pub mod msg {
    /// Un calcul est en cours.
    pub const COMPUTING: &str = "Calcul en cours…";
    /// Le rendu affiché date d'avant le dernier réglage.
    pub const STALE: &str = "aperçu obsolète — recalcul en cours";
    /// La passe en pleine résolution suit l'aperçu réduit.
    pub const FINALIZING: &str = "pleine résolution…";
    /// Aucun état précédent à restaurer.
    pub const NOTHING_TO_UNDO: &str = "Rien à annuler.";
    /// Une annulation vient d'être appliquée.
    pub const UNDONE: &str = "Annulé.";
    /// Aucun état annulé à rétablir.
    pub const NOTHING_TO_REDO: &str = "Rien à rétablir.";
    /// Un rétablissement vient d'être appliqué.
    pub const REDONE: &str = "Rétabli.";
    /// Les réglages sont déjà ceux par défaut.
    pub const ALREADY_DEFAULT: &str = "Réglages déjà par défaut.";
    /// Aucune image n'est chargée.
    pub const OPEN_FIRST: &str = "Ouvrez d'abord une image.";
    /// Aucun rendu à copier.
    pub const NOTHING_TO_COPY: &str = "Aucun résultat à copier.";
    /// Le rendu est dans le presse-papiers.
    pub const COPIED: &str = "Image copiée dans le presse-papiers.";
    /// L'utilisateur a renoncé à écrire par-dessus un fichier.
    pub const SAVE_CANCELLED: &str = "Sauvegarde annulée.";
}

/// Cherche un dossier d'assets près du binaire puis du répertoire courant.
///
/// Même recherche que [`sample_image`] : une exécution depuis les sources
/// trouve `assets/`, une installation trouve un dossier voisin du binaire.
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

/// Sous-échantillonne une image pour un aperçu rapide.
///
/// Même filtre (`Triangle`) et même arrondi des deux côtés : deux aperçus de la
/// même image doivent montrer les mêmes pixels, sinon comparer ce que produit
/// un réglage d'une interface à l'autre n'aurait pas de sens. L'échelle >= 1 et
/// la passe de finalisation renvoient l'image telle quelle (clone compris).
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

/// Images d'un dossier, triées par nom, ou l'erreur qui explique pourquoi vide.
///
/// Le balayage était écrit deux fois, jusqu'aux messages d'erreur — un dossier
/// sans image devait expliciter la même chose dans les deux interfaces. Seul le
/// moyen d'afficher reste propre à chacune.
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

/// Pourquoi un dossier n'a pas donné d'images.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FolderError {
    /// Le dossier n'a pas pu être lu.
    Unreadable(PathBuf),
    /// Le dossier est lisible mais ne contient aucune image reconnue.
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

// ─── Damier de transparence ────────────────────────────────────────────────

/// Côté d'une case du damier de transparence, en pixels.
pub const CHECKER_SIZE: u32 = 8;
/// Gris clair d'une case.
pub const CHECKER_LIGHT: f32 = 200.0;
/// Gris foncé d'une case.
pub const CHECKER_DARK: f32 = 240.0;

/// Gris du damier sous le pixel `(x, y)`.
///
/// Les deux interfaces et le panneau de comparaison partagent la même
/// définition : elles doivent afficher le même damier pour la même image.
#[inline]
pub fn checker_shade(x: u32, y: u32) -> f32 {
    if (x / CHECKER_SIZE + y / CHECKER_SIZE).is_multiple_of(2) {
        CHECKER_LIGHT
    } else {
        CHECKER_DARK
    }
}

/// Composite un pixel RGBA sur le damier, et renvoie le `[r, g, b]` opaque.
///
/// L'ordre des octets en sortie appartient à chaque interface : egui empile des
/// `Color32`, GTK des octets R8g8b8a8, cairo du BGRX. La *couleur*, elle, est
/// unique — c'est la seule chose que ce module impose.
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

/// Formate une taille de fichier avec la précision utile : « 381 Ko »,
/// « 2,1 Mo ». [`format_memory`] reste en unités entières pour les estimations
/// de mémoire, où le chiffre est de toute façon approximatif.
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

/// Écrit dans un fichier temporaire, puis remplace la cible atomiquement.
/// Le temporaire est supprimé en cas d'échec, dans les deux phases.
///
/// Le squelette était recopié trois fois (image, octets, texte). Les copies
/// différaient par le type d'erreur qu'elles renvoient — `image::ImageError` pour
/// les deux premières, `std::io::Error` pour le texte — et seule la première
/// rapportait l'erreur d'écriture telle quelle : un changement de ce nettoyage
/// (par exemple retenir le dossier d'un échec) ne serait appliqué qu'à une copie.
/// Le type d'erreur est donc un paramètre, et les trois entrées publiques
/// gardent leur signature.
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

/// Écrit une image via un fichier temporaire puis un remplacement atomique.
/// Le fichier temporaire est supprimé en cas d'erreur.
pub fn atomic_image_save<F>(path: &Path, save: F) -> image::ImageResult<()>
where
    F: FnOnce(&Path) -> image::ImageResult<()>,
{
    atomic_write(path, save, image::ImageError::IoError)
}

/// Écrit des octets via un fichier temporaire puis un remplacement atomique.
pub fn atomic_bytes_write(path: &Path, bytes: &[u8]) -> image::ImageResult<()> {
    atomic_write(
        path,
        |tmp| std::fs::write(tmp, bytes).map_err(image::ImageError::IoError),
        image::ImageError::IoError,
    )
}

/// Écrit du texte via un fichier temporaire puis un remplacement atomique.
pub fn atomic_text_write(path: &Path, contents: &str) -> std::io::Result<()> {
    atomic_write(path, |tmp| std::fs::write(tmp, contents), |error| error)
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
    // L'export passe par l'encodage en mémoire : c'est le même code que
    // [`measure_export_sizes`], donc la taille annoncée est celle du fichier
    // réellement écrit.
    let bytes = encode_rgba_image(path, img, transparent, bg)?;
    atomic_bytes_write(path, &bytes)
}

/// Encode un résultat RGBA dans le format désigné par l'extension de `path`,
/// en mémoire.
///
/// Formats sans alpha (JPEG, BMP) : composition sur `bg` quand le rendu est
/// transparent, simple conversion en RGB sinon. Formats avec alpha (PNG, WebP,
/// TIFF) : image conservée telle quelle.
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

/// Idem, à partir de la seule extension (sans le point) : c'est ce dont la
/// mesure des tailles a besoin, puisqu'aucun chemin n'est encore choisi.
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

/// Format `image` correspondant à une extension (sans le point).
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
        // 0,06 dans 0,0005 → 1,0 : à mi-chemin des trois ordres de grandeur,
        // là où le linéaire la collerait au bord du curseur.
        let position = Curve::Log.position(0.06, 0.0005, 1.0);
        assert!((0.55..=0.7).contains(&position), "position : {position}");
    }

    #[test]
    fn sqrt_curve_uses_the_first_tenth() {
        // 0,85 sur 0 → 10 : 29 % du curseur au lieu de 8 %.
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

    /// Chaque table doit contenir chaque variante de son enum, une seule fois,
    /// et les deux sens de conversion doivent s'accorder. Ajouter une variante
    /// sans l'inscrire fait échouer ce test — c'est le seul garde-fou contre
    /// une liste décalée, qu'un `_ =>` laisserait passer en silence.
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
            // Une seule occurrence : deux fois la même variante décalerait tout.
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
        // Et un index hors bornes ne renvoie rien, au lieu de choisir.
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

    /// Le damier alterne sur une période de deux cases, et le composite suit
    /// l'alpha : opaque = le pixel, transparent = le gris de la case.
    #[test]
    fn le_damier_alterne_et_suit_lalpha() {
        // Sur une même ligne, la couleur de fond change d'une case à l'autre.
        assert_eq!(checker_shade(0, 0), CHECKER_LIGHT);
        assert_eq!(checker_shade(CHECKER_SIZE, 0), CHECKER_DARK);
        assert_eq!(checker_shade(0, 0), checker_shade(2 * CHECKER_SIZE, 0));
        // Et sur une colonne aussi : la formule combine les deux axes.
        assert_eq!(
            checker_shade(0, 0),
            checker_shade(CHECKER_SIZE, CHECKER_SIZE)
        );
        assert_eq!(
            checker_shade(CHECKER_SIZE, 0),
            checker_shade(0, CHECKER_SIZE)
        );

        // Opaque : le pixel lui-même, le damier ne se voit pas.
        assert_eq!(
            composite_on_checker([10, 200, 30, 255], 0, 0),
            [10, 200, 30],
            "un pixel opaque ne dépend pas du damier"
        );
        // Transparent : exactement le gris de la case.
        assert_eq!(composite_on_checker([9, 9, 9, 0], 0, 0).len(), 3);
        assert_eq!(
            composite_on_checker([9, 9, 9, 0], CHECKER_SIZE, 0),
            [CHECKER_DARK as u8, CHECKER_DARK as u8, CHECKER_DARK as u8]
        );
        // Semi-transparent : entre les deux, quel que soit le canal.
        let demi = composite_on_checker([0, 0, 0, 128], 0, 0);
        assert!(
            demi[0] > 0 && demi[0] < CHECKER_LIGHT as u8,
            "un pixel à moitié opaque doit tomber entre le noir et le gris clair, pas à {}",
            demi[0]
        );
        // Les trois canaux subissent le même traitement.
        assert_eq!(demi[0], demi[1]);
        assert_eq!(demi[1], demi[2]);
    }

    /// Le nom d'export dérivé donne la même chose avec ou sans fichier source,
    /// et replie proprement quand il n'y en a pas.
    #[test]
    fn le_nom_derive_suit_le_repli() {
        use std::path::Path;
        assert_eq!(
            derived_export_name(Some(Path::new("/photos/vacances.jpg")), "png"),
            "vacances-pointimg.png"
        );
        assert_eq!(derived_export_name(None, "svg"), "pointimg.svg");
        // Un nom de fichier sans extension : le tronc suffit.
        assert_eq!(
            derived_export_name(Some(Path::new("/tmp/photo")), "png"),
            "photo-pointimg.png"
        );
    }

    /// Un dossier lisible mais sans image connue n'est pas « illisible » : les
    /// deux messages sont distincts, et les deux interfaces les affichent tels
    /// quels.
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

    /// Le sous-échantillonnage respecte l'échelle demandée, et ne bouge pas
    /// quand l'échelle ne réduit rien.
    #[test]
    fn le_sous_echantillonnage_respecte_lechelle() {
        let img = image::RgbImage::new(200, 100);
        assert_eq!(downsample(&img, 1.0).dimensions(), (200, 100));
        assert_eq!(downsample(&img, 2.0).dimensions(), (200, 100));
        let reduit = downsample(&img, 0.5);
        assert_eq!(reduit.dimensions(), (100, 50));
        // Jamais de dimension nulle, même à une échelle absurde.
        assert_eq!(downsample(&img, 0.0001).dimensions(), (1, 1));
    }

    /// Les trois écritures atomiques partagent le même nettoyage : un temporaire
    /// ne doit jamais survivre à un échec, et le remplacement doit avoir lieu.
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

        // Écrasement : la remplacement doit avoir lieu, pas seulement la création.
        atomic_text_write(&cible, "au revoir").expect("réécriture");
        assert_eq!(std::fs::read_to_string(&cible).unwrap(), "au revoir");

        // Échec d'écriture : `write` échoue, donc rien ne doit rester.
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
