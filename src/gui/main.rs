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
mod format;
mod io;
mod save;
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
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };
    eframe::run_native(
        "pointimg",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
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
    /// Dernier dossier d'enregistrement (proposé ensuite aux dialogues).
    pub last_dir: Option<PathBuf>,
    /// Images du dossier courant, pour la navigation `←`/`→`.
    pub folder_images: Vec<PathBuf>,
    pub folder_index: Option<usize>,
    pub src_texture: Option<egui::TextureHandle>,

    // Résultat du filtre (mis à jour à chaque preview intermédiaire aussi).
    // Stocké en RGBA pour supporter le mode transparent (canal alpha préservé).
    pub result: Arc<Mutex<Option<RgbaImage>>>,
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
    /// Échelle d'interface appliquée au style egui (1.0 = confortable).
    pub ui_scale: f32,
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
            view_mode: ViewMode::Compare,
            zoom: 1.0,
            zoom_fit: true,
            effective_zoom: 1.0,
            compare_split: 0.5,
            status: "Ouvrez une image pour commencer (ou faites glisser un fichier).".to_string(),
            window_title: String::new(),
            ui_scale: 1.0,
            applied_style: None,
        }
    }
}

impl App {
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
        let scale = self.quality.scale();
        if scale >= 1.0 {
            return Some(src.clone());
        }
        let (w, h) = src.dimensions();
        let target_w = ((w as f32 * scale).round() as u32).max(1);
        let target_h = ((h as f32 * scale).round() as u32).max(1);
        Some(image::imageops::resize(
            src,
            target_w,
            target_h,
            image::imageops::FilterType::Triangle,
        ))
    }
}
