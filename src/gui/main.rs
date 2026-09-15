//! Point d'entrée de la GUI pointimg.
//!
//! Architecture :
//! - Thread principal = thread egui (obligatoire sur macOS/Windows)
//! - Calcul du filtre dans un thread séparé via `std::thread::spawn`
//! - Communication : `Arc<Mutex<Option<RgbImage>>>` + `AtomicBool` computing + cancel token
//! - Density map de preview : calculée hors thread GUI et publiée quand prête
//! - Preview progressive : pour Voronoï/K-means, chaque itération publie un résultat intermédiaire
//! - Dots cachés : stockés dans App après chaque calcul complet, utilisés pour l'export SVG
//!
//! Découpage :
//! - [`convert`] : conversion d'images `image` → textures egui
//! - [`format`] : formatage humain (durées, mémoire) + garde des générations
//! - [`io`] : écritures atomiques et validation des chemins de sortie
//! - [`app`] : état applicatif et logique de haut niveau (chargement, undo/redo)
//! - [`compute`] : workers de calcul du filtre et de la density map
//! - [`save`] : export des résultats (preset, image, SVG)
//! - [`ui`] : rendu egui (panneau de contrôle, zone centrale)

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
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use app::HistoryEntry;
use app::ViewMode;
use pointimg::filter::FilterParams;

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

    let wgpu_options = WgpuConfiguration {
        wgpu_setup: WgpuSetup::CreateNew(WgpuSetupCreateNew {
            instance_descriptor: wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            },
            ..Default::default()
        }),
        ..Default::default()
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("pointimg")
            .with_inner_size([1200.0, 750.0])
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

/// État de l'application : partage le type entre les modules `app`, `compute`,
/// `save` et `ui` (un `impl App` par module pour sa partie de logique).
pub(crate) struct App {
    pub params: FilterParams,

    // Image source originale (DynamicImage pour supporter RGBA)
    pub src_dynamic: Option<DynamicImage>,
    // Image source convertie en RGB8 (mise en cache, recalculée si bg_color change)
    pub src_rgb: Option<RgbImage>,
    pub src_path: Option<PathBuf>,
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

    // Undo/redo : historique des FilterParams commités.
    pub history: Vec<HistoryEntry>,
    pub future: Vec<HistoryEntry>,
    // `last_committed` est la version "live" au moment du dernier commit
    // (sert à éviter de pousser 50 entrées consécutives sur un même slider).
    pub last_committed: Option<FilterParams>,
    // Drapeau : un changement vient d'être détecté, mais le debounce est en
    // cours. Quand le compute se termine (et sans nouveau commit prévu),
    // on pousse l'état actuel dans l'historique.
    pub pending_commit: bool,

    // Mode d'affichage
    pub view_mode: ViewMode,

    // Niveau de zoom (1.0 = 100%)
    pub zoom: f32,
    // Whether the preview should fit the available panel.
    pub zoom_fit: bool,

    // Message de statut
    pub status: String,
}
