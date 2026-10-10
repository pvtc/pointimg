//! Point d'entrée de la GUI pointimg.
//!
//! Architecture :
//! - Thread principal = thread egui (obligatoire sur macOS/Windows)
//! - Calcul du filtre dans un thread séparé via `std::thread::spawn`
//! - Communication : `Arc<Mutex<Option<DynamicImage>>>` + `AtomicBool` computing
//!   + cancel token
//! - Density map de preview : calculée hors thread GUI et publiée quand prête
//! - Preview progressive : pour Voronoï/K-means, chaque itération publie un
//!   résultat intermédiaire
//! - Dots cachés : stockés dans App après chaque calcul complet, utilisés pour
//!   l'export SVG
//!
//! Découpage :
//! - [`convert`] : conversion d'images `image` → textures egui
//! - `format` : formatage humain (durées) + garde des générations
//! - [`io`] : écritures atomiques et validation des chemins de sortie
//! - [`app`] : état applicatif et logique de haut niveau (chargement, undo/redo)
//! - [`compute`] : workers de calcul du filtre et de la density map
//! - [`save`] : export des résultats (preset, image, SVG)
//! - [`ui`] : rendu egui (en-tête, panneau de contrôle, zone centrale)

mod app;
mod compute;
mod convert;
mod exports;
mod format;
mod io;
mod notify;
mod palette;
mod prefs;
#[cfg(test)]
mod ranges;
mod save;
mod sliders;
mod ui;

use eframe::egui;
use image::{DynamicImage, GrayImage as ImgGrayImage, RgbImage, RgbaImage};
use pointimg::filter::Dot;
pub(crate) use pointimg::frontend::{Quality, ViewMode};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() -> eframe::Result {
    env_logger::init();
    use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};

    // Backend GPU : sur Linux on force Vulkan + GL (fallback pour Wayland/EGL
    // selon le driver NVIDIA/AMD/Intel). Sur macOS on laisse wgpu choisir
    // Metal automatiquement. Sur Windows, le backend par défaut (DX12/Vulkan)
    // fonctionne nativement.
    #[cfg(target_os = "linux")]
    let backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
    #[cfg(not(target_os = "linux"))]
    let backends = wgpu::Backends::PRIMARY;

    let mut wgpu_setup = WgpuSetupCreateNew::without_display_handle();
    wgpu_setup.instance_descriptor.backends = backends;
    let wgpu_options = WgpuConfiguration {
        wgpu_setup: WgpuSetup::CreateNew(wgpu_setup),
        ..Default::default()
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("pointimg")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([860.0, 560.0])
            .with_drag_and_drop(true)
            .with_icon(window_icon()),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };
    // Chemin passé en argument : `pointimg-gui photo.jpg`. La fenêtre n'existe
    // pas encore ici, le chargement attend la première frame — c'est
    // `App::pending_open` qui s'en charge.
    let requested = pending_open_from_args();
    eframe::run_native(
        "pointimg",
        options,
        Box::new(move |_cc| {
            let mut app = App::new();
            app.pending_open = requested;
            Ok(Box::new(app))
        }),
    )
}

/// Icône de la fenêtre, lue dans le binaire.
///
/// egui n'a pas de thème d'icônes : contrairement à GTK, qui va chercher
/// `pointimg` dans le thème du bureau, il faut fournir les pixels. Le PNG est
/// embarqué — c'est le coût de 20 ko — plutôt que lu sur disque, pour ne pas
/// dépendre du répertoire d'installation.
///
/// Le SVG maître est dans `assets/logo/` ; les PNG en sont dérivés par
/// `packaging/icons/regenerate.sh`.
fn window_icon() -> egui::IconData {
    let png = include_bytes!("../../assets/logo/pointimg-256.png");
    match image::load_from_memory_with_format(png, image::ImageFormat::Png) {
        Ok(image) => {
            let image = image.into_rgba8();
            let (width, height) = image.dimensions();
            egui::IconData {
                rgba: image.into_raw(),
                width,
                height,
            }
        }
        Err(e) => {
            // Une icône absente ne doit pas empêcher l'application de
            // démarrer : on le signale et on ouvre sans.
            log::warn!("icône de fenêtre illisible : {e}");
            egui::IconData {
                rgba: Vec::new(),
                width: 0,
                height: 0,
            }
        }
    }
}

/// Chemins d'images ou de dossiers reçus sur la ligne de commande.
///
/// egui ne gère pas l'activation D-Bus comme GTK : seule la ligne de commande
/// est lue ici, ce qui couvre `pointimg-gui photo.jpg` et l'association de
/// fichiers du bureau.
fn pending_open_from_args() -> Option<PathBuf> {
    std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .find(|path| pointimg::frontend::is_supported_image(path) || path.is_dir())
}

/// Entrée d'historique : les paramètres d'un état passé + son libellé.
pub(crate) struct HistoryEntry {
    pub params: pointimg::filter::FilterParams,
    /// « Algorithme : Grid → Voronoi », « Nombre de points : 800 → 1500 »…
    pub label: String,
}

/// État de l'application : partage le type entre les modules `app`, `compute`,
/// `save` et `ui` (un `impl App` par module pour sa partie de logique).
pub(crate) struct App {
    pub params: pointimg::filter::FilterParams,

    // Image source originale (DynamicImage pour supporter RGBA)
    pub src_dynamic: Option<DynamicImage>,
    // Image source convertie en RGB8 (mise en cache, recalculée si bg_color change)
    pub src_rgb: Option<RgbImage>,
    pub src_path: Option<PathBuf>,
    /// Taille du fichier source en octets, pour comparer l'export à l'original.
    pub source_size: Option<u64>,
    /// Dernier dossier d'enregistrement (proposé ensuite aux dialogues).
    pub last_dir: Option<PathBuf>,
    /// Images du dossier courant, pour la navigation `←`/`→`.
    pub folder_images: Vec<PathBuf>,
    pub folder_index: Option<usize>,
    pub src_texture: Option<egui::TextureHandle>,

    // Résultat du filtre (mis à jour à chaque preview intermédiaire aussi).
    // Stocké en RGBA pour supporter le mode transparent (canal alpha préservé).
    // L'`Arc` intérieur permet à la mesure de taille d'export d'emprunter
    // l'image sans la copier (elle pèse jusqu'à ~33 Mo) — c'est le même motif
    // que `density_data`.
    pub result: Arc<Mutex<Option<Arc<RgbaImage>>>>,
    pub result_texture: Option<egui::TextureHandle>,
    pub result_revision: Arc<AtomicU64>,
    pub result_texture_revision: u64,

    // Dots du dernier calcul terminé (utilisés pour export SVG)
    pub last_dots: Arc<Mutex<Option<Vec<Dot>>>>,

    // Density map de preview, calculée hors thread GUI.
    pub density_image: Option<ImgGrayImage>,
    pub density_texture: Option<egui::TextureHandle>,
    pub density_result: Arc<Mutex<Option<ImgGrayImage>>>,
    pub density_data: Arc<Mutex<Option<Arc<Vec<f32>>>>>,
    /// Échelle de l'image servant au calcul en cours (1.0 = pleine résolution).
    pub density_scale: f32,

    // Progression (iter_courant, iter_total)
    pub progress: Arc<Mutex<(usize, usize)>>,

    // Calcul en cours ?
    pub computing: Arc<AtomicBool>,
    // Token d'annulation
    pub cancel: Arc<AtomicBool>,
    // Identifie le dernier calcul demandé. Les workers obsolètes ne publient rien.
    pub compute_generation: Arc<AtomicU64>,
    // Identifie la dernière density map demandée.
    pub density_generation: Arc<AtomicU64>,

    // Erreur du thread de calcul
    pub compute_error: Arc<Mutex<Option<String>>>,

    // Temps de calcul du dernier rendu terminé
    pub last_compute_ms: Option<u64>,
    pub compute_start: Option<Instant>,

    // Last instant at which a parameter changed, used for debounce.
    pub last_param_change: Option<Instant>,

    // Mode aperçu rapide + finalisation automatique.
    pub quality: Quality,
    /// Passe de finalisation en cours (pleine résolution malgré le mode rapide).
    pub final_pass: bool,
    /// Le résultat affiché est-il en pleine résolution ? Partagé avec le
    /// worker, qui le publie en fin de calcul.
    pub result_is_final: Arc<AtomicBool>,
    /// Instant auquel déclencher la finalisation.
    pub finalize_at: Option<Instant>,

    // Undo/redo : historique des FilterParams commités.
    pub history: Vec<HistoryEntry>,
    pub future: Vec<HistoryEntry>,
    // `last_committed` est la version "live" au moment du dernier commit
    // (sert à éviter de pousser 50 entrées consécutives sur un même slider).
    pub last_committed: Option<pointimg::filter::FilterParams>,
    // Drapeau : un changement vient d'être détecté, mais le debounce est en
    // cours. Quand le compute se termine (et sans nouveau commit prévu),
    // on pousse l'état actuel dans l'historique.
    pub pending_commit: bool,
    /// La density map est à recalculer (la variance a bougé). Elle part avec
    /// le debounce du filtre : sans cela, chaque pixel de glissement d'un
    /// curseur lancerait un calcul de density map sur toute l'image.
    pub density_pending: bool,

    // Taille d'export, mesurée par encodage réel dans un thread de travail
    // (cf. `exports.rs`). `export_sizes_revision` est la révision du résultat
    // pour laquelle la mesure affichée a été faite.
    pub export_sizes: Option<pointimg::export::ExportSizes>,
    pub export_sizes_revision: u64,
    pub export_sizes_generation: Arc<AtomicU64>,
    pub export_sizes_computing: Arc<std::sync::atomic::AtomicBool>,
    pub export_sizes_slot: Arc<Mutex<Option<pointimg::export::ExportSizes>>>,

    // Mode d'affichage
    pub view_mode: ViewMode,

    // Niveau de zoom (1.0 = 100 %)
    pub zoom: f32,
    // Whether the preview should fit the available panel.
    pub zoom_fit: bool,
    // Zoom réellement appliqué (mode « Ajuster » : déduit du viewport).
    pub effective_zoom: f32,
    // Position du curseur de comparaison (0 = tout à droite).
    pub compare_split: f32,

    // Message de statut
    pub status: String,
    // Titre de la fenêtre (nom du fichier), pour éviter de le renvoyer à chaque frame.
    pub window_title: String,
    /// Notifications éphémères (erreurs, confirmations).
    pub toasts: notify::Toasts,
    /// Palette de commandes ouverte (`Ctrl+K`) et son texte de recherche.
    pub palette: Option<String>,
    /// Chemin donné en ligne de commande, chargé à la première frame : la
    /// fenêtre n'existe pas encore au moment où `main` le reçoit.
    pub pending_open: Option<PathBuf>,
    /// Échelle d'interface appliquée au style egui (1.0 = confortable).
    pub ui_scale: f32,
    /// Thème imposé par l'utilisateur ; `None` suit la préférence du système.
    pub ui_theme: Option<egui::Theme>,
    /// Préférences d'interface persistées (échelle, thème).
    pub prefs: Option<prefs::InterfacePrefs>,
    /// Vignettes des presets, décodées et montées au GPU une seule fois.
    ///
    /// Elles étaient reconstruites à chaque frame (neuf `read_dir`, neuf
    /// décodages PNG de ~130 Ko et neuf montées de texture), ce qu'un panneau
    /// redessiné plusieurs fois par seconde ne peut pas se permettre. `None`
    /// tant que le chargement n'a pas eu lieu — c'est ce qui distingue « pas
    /// encore fait » de « fait et vide ».
    pub preset_thumbnails: Option<Vec<Option<egui::TextureHandle>>>,
    /// Échelle pour laquelle le style a été appliqué (evite de le refaire à
    /// chaque frame).
    pub applied_style: Option<f32>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            params: pointimg::filter::FilterParams::default(),
            src_dynamic: None,
            src_rgb: None,
            src_path: None,
            last_dir: None,
            folder_images: Vec::new(),
            folder_index: None,
            src_texture: None,
            result: Default::default(),
            result_texture: None,
            result_revision: Default::default(),
            result_texture_revision: 0,
            last_dots: Default::default(),
            density_image: None,
            density_texture: None,
            density_result: Default::default(),
            density_data: Default::default(),
            density_scale: 0.0,
            progress: Default::default(),
            computing: Default::default(),
            cancel: Default::default(),
            compute_generation: Default::default(),
            density_generation: Default::default(),
            compute_error: Default::default(),
            last_compute_ms: None,
            compute_start: None,
            last_param_change: None,
            quality: Quality::Half,
            final_pass: false,
            result_is_final: Arc::new(AtomicBool::new(true)),
            finalize_at: None,
            history: Vec::new(),
            future: Vec::new(),
            last_committed: None,
            pending_commit: false,
            density_pending: false,
            source_size: None,
            export_sizes: None,
            export_sizes_revision: 0,
            export_sizes_generation: Arc::new(AtomicU64::new(0)),
            export_sizes_computing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            export_sizes_slot: Arc::new(Mutex::new(None)),
            view_mode: ViewMode::Compare,
            zoom: 1.0,
            zoom_fit: true,
            effective_zoom: 1.0,
            compare_split: 0.5,
            status: "Ouvrez une image pour commencer (ou faites glisser un fichier).".to_string(),
            window_title: String::new(),
            toasts: notify::Toasts::new(),
            palette: None,
            pending_open: None,
            ui_scale: 1.0,
            ui_theme: None,
            prefs: None,
            preset_thumbnails: None,
            applied_style: None,
        }
    }
}

impl App {
    /// Construit l'application en relisant les préférences d'interface
    /// enregistrées au lancement précédent.
    pub fn new() -> Self {
        let mut app = Self::default();
        if let Some(prefs) = prefs::InterfacePrefs::load() {
            if let Some(scale) = prefs.scale() {
                app.ui_scale = scale.clamp(0.85, 1.6);
            }
            // Une absence de clé signifie « suivre le système ».
            app.ui_theme = prefs.theme();
            app.prefs = Some(prefs);
        }
        app
    }

    /// Enregistre une préférence d'interface (immédiatement : quitter aussitôt
    /// après le changement ne perd rien).
    pub fn persist_display_preferences(&mut self) {
        if let Some(prefs) = &mut self.prefs {
            prefs.save(self.ui_scale, self.ui_theme);
        }
    }

    /// Applique le thème choisi au contexte. « Auto » rend la main à egui,
    /// qui suit alors la préférence du système.
    pub(crate) fn apply_theme(&self, ctx: &egui::Context) {
        let preference = match self.ui_theme {
            Some(egui::Theme::Dark) => egui::ThemePreference::Dark,
            Some(egui::Theme::Light) => egui::ThemePreference::Light,
            None => egui::ThemePreference::System,
        };
        // Écrire les options à chaque rendu redemanderait un rendu : on ne
        // touche au contexte que si la préférence a vraiment changé.
        if ctx.options(|options| options.theme_preference) != preference {
            ctx.set_theme(preference);
        }
    }

    /// Un changement de réglage : armement du debounce + historique.
    pub(crate) fn on_params_changed(&mut self, ctx: &egui::Context) {
        if self.src_rgb.is_none() {
            return;
        }
        self.last_param_change = Some(Instant::now());
        self.pending_commit = true;
        // Une modification annule la finalisation en attente : c'est le
        // nouveau réglage qui sera passé en pleine résolution.
        self.finalize_at = None;
        ctx.request_repaint();
    }

    /// Change la qualité de l'aperçu et relance un calcul.
    pub(crate) fn set_quality(&mut self, quality: Quality, ctx: &egui::Context) {
        if self.quality == quality {
            return;
        }
        self.quality = quality;
        self.final_pass = false;
        self.finalize_at = None;
        if self.src_rgb.is_some() {
            self.start_compute(ctx);
        }
    }

    /// Recalcule en pleine résolution sans changer le mode choisi ( bouton
    /// Recalculer, Espace, finalisation automatique).
    pub(crate) fn request_full_compute(&mut self, ctx: &egui::Context) {
        if self.src_rgb.is_none() {
            return;
        }
        if self.computing.load(Ordering::Relaxed) {
            self.cancel.store(true, Ordering::Relaxed);
        }
        self.final_pass = true;
        self.finalize_at = None;
        self.start_compute(ctx);
    }

    /// Le calcul en cours est-il en pleine résolution ?
    pub(crate) fn computes_full_resolution(&self) -> bool {
        self.quality.is_full() || self.final_pass
    }

    /// Image source utilisée pour un calcul, sous-échantillonnée en mode rapide.
    pub(crate) fn compute_source(&self) -> Option<RgbImage> {
        let src = self.src_rgb.as_ref()?;
        if self.final_pass {
            return Some(src.clone());
        }
        // Le sous-échantillonnage est partagé avec GTK : un aperçu doit montrer
        // les mêmes pixels des deux côtés.
        Some(pointimg::frontend::downsample(src, self.quality.scale()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L'icône embarquée doit être décodable, sinon la fenêtre s'ouvre sans
    /// icône et rien ne le signale à l'utilisateur.
    #[test]
    fn licone_de_fenetre_est_decodable() {
        let icon = window_icon();
        assert_eq!(icon.width, 256);
        assert_eq!(icon.height, 256);
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
        // Le centre du grand point cyan, à (163, 169) sur la grille 512,
        // soit (81, 84) en 256. Le coin haut-gauche est transparent : le
        // cercle n'y passe pas.
        let x = 81 * 4;
        let y = 84 * icon.width as usize * 4;
        let centre = &icon.rgba[x + y..x + y + 4];
        assert_eq!(
            centre,
            &[0x00, 0xAE, 0xEF, 0xFF],
            "le centre du logo devrait être le cyan du point principal"
        );
        // Et le coin, lui, doit être transparent : une icône de bureau ne doit
        // pas poser un carré opaque.
        assert_eq!(&icon.rgba[..4], &[0, 0, 0, 0]);
    }
}
