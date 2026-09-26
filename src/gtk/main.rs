//! Interface GTK4 / libadwaita de pointimg.
//!
//! Architecture :
//! - Le thread principal GTK possède l'état applicatif (`State`, derrière
//!   `Rc<RefCell<..>>`) et tous les widgets (`Widgets`).
//! - Les calculs du filtre et de la density map tournent dans des threads
//!   dédiés et publient leurs résultats via un canal `async-channel`.
//! - Un futur local (via `glib::MainContext::spawn_local`) consomme le canal
//!   sur le thread principal et met à jour l'interface.
//!
//! Découpage :
//! - [`ui`] : construction de la fenêtre et des raccourcis.
//! - [`actions`] : workers, chargement/export, presets, undo/redo, calculs.
//! - [`controls`] : fabrication et disposition des widgets.
//! - [`sections`] : synchronisation état → widgets, visibilité des sections.
//! - [`preview`] : textures, zoom, aperçus, barre d'état.
//! - [`compare`] : comparaison avant/après par balayage.
//! - [`wire`] : signaux GTK → état, menu et raccourcis clavier.

mod actions;
mod compare;
mod controls;
mod files;
mod preview;
mod sections;
mod shortcuts;
mod ui;
mod wire;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Instant;

use gtk::glib;
use gtk::prelude::*;

use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use pointimg::filter::{Dot, FilterParams};

const APP_ID: &str = "org.pointimg.Pointimg";

/// Mode d'affichage de la zone d'aperçu.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ViewMode {
    /// Source et résultat superposés, séparés par un curseur.
    Compare,
    Side,
    ResultOnly,
    SourceOnly,
    DensityMap,
}

/// Messages envoyés par les workers vers le thread GTK.
pub(crate) enum Msg {
    Progress {
        generation: u64,
        done: usize,
        total: usize,
    },
    Preview {
        generation: u64,
        image: RgbaImage,
    },
    Done {
        generation: u64,
        image: RgbaImage,
        dots: Vec<Dot>,
        elapsed_ms: u64,
        full_resolution: bool,
    },
    Failed {
        generation: u64,
        message: String,
        cancelled: bool,
    },
    Density {
        generation: u64,
        image: GrayImage,
        data: Arc<Vec<f32>>,
        /// Échelle de l'image utilisée (la density map en dépend).
        scale: f32,
    },
}

/// État applicatif partagé entre les modules.
pub(crate) struct State {
    pub params: FilterParams,

    pub src_dynamic: Option<DynamicImage>,
    pub src_rgb: Option<RgbImage>,
    pub src_path: Option<PathBuf>,
    /// Dernier dossier utilisé pour ouvrir/enregistrer (proposé ensuite par les
    /// dialogues de fichiers).
    pub last_dir: Option<PathBuf>,
    /// Images du dossier courant (navigation `←`/`→`), si le fichier a été
    /// ouvert via « Ouvrir un dossier ».
    pub folder_images: Vec<PathBuf>,
    pub folder_index: Option<usize>,
    /// Taille d'origine avant le sous-échantillonnage éventuel, pour l'info-bulle.
    pub source_was_resized: bool,

    pub result: Option<RgbaImage>,
    pub last_dots: Option<Vec<Dot>>,
    pub density_image: Option<GrayImage>,
    pub density_data: Option<Arc<Vec<f32>>>,
    /// Échelle à laquelle `density_data` a été calculée : la density map
    /// n'est réutilisable que par un calcul de même résolution.
    pub density_scale: f32,

    pub view_mode: ViewMode,
    pub zoom: f64,
    pub zoom_fit: bool,
    /// Zoom réellement appliqué (mode « Ajuster » : calculé du viewport).
    pub effective_zoom: f64,
    pub status: String,
    pub last_compute_ms: Option<u64>,
    pub compute_start: Option<Instant>,

    pub history: Vec<FilterParams>,
    pub future: Vec<FilterParams>,
    pub last_committed: Option<FilterParams>,
    pub pending_commit: bool,
    /// Empêche les callbacks de widgets de réagir à une mise à jour
    /// programmatique (undo, chargement de preset).
    pub updating: bool,
    pub debounce: Option<glib::SourceId>,

    pub computing: bool,
    pub user_cancelled: bool,
    pub restart_pending: bool,

    /// Mode choisi par l'utilisateur (rapide ou pleine résolution).
    pub quality: Quality,
    /// Passe de finalisation en cours : le rendu affiché est en pleine
    /// résolution même si l'aperçu rapide est le mode retenu.
    pub final_pass: bool,
    /// Le résultat affiché est-il en pleine résolution ?
    pub result_is_final: bool,
    /// Minuterie de finalisation : repasse en pleine résolution quand
    /// l'utilisateur arrête de manipuler les réglages.
    pub finalize_pending: Option<glib::SourceId>,

    pub cancel: Arc<AtomicBool>,
    pub generation: Arc<AtomicU64>,
    pub density_generation: Arc<AtomicU64>,
    pub tx: async_channel::Sender<Msg>,

    // Cache des textures d'aperçu.
    pub src_texture: Option<gtk::gdk::Texture>,
    pub result_texture: Option<gtk::gdk::Texture>,
    pub density_texture: Option<gtk::gdk::Texture>,
}

/// Résolution de calcul choisie par l'utilisateur.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Quality {
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

    pub fn is_full(self) -> bool {
        self == Quality::Full
    }
}

impl State {
    fn new(tx: async_channel::Sender<Msg>) -> Self {
        Self {
            params: FilterParams::default(),
            src_dynamic: None,
            src_rgb: None,
            src_path: None,
            last_dir: None,
            folder_images: Vec::new(),
            folder_index: None,
            source_was_resized: false,
            result: None,
            last_dots: None,
            density_image: None,
            density_data: None,
            density_scale: 0.0,
            view_mode: ViewMode::Compare,
            zoom: 1.0,
            zoom_fit: true,
            effective_zoom: 1.0,
            status: "Ouvrez une image pour commencer (ou glissez-déposez un fichier).".to_string(),
            last_compute_ms: None,
            compute_start: None,
            history: Vec::new(),
            future: Vec::new(),
            last_committed: None,
            pending_commit: false,
            updating: false,
            debounce: None,
            computing: false,
            user_cancelled: false,
            restart_pending: false,
            quality: Quality::Half,
            final_pass: false,
            result_is_final: true,
            finalize_pending: None,
            cancel: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicU64::new(0)),
            density_generation: Arc::new(AtomicU64::new(0)),
            tx,
            src_texture: None,
            result_texture: None,
            density_texture: None,
        }
    }

    /// Le résultat affiché ne correspond plus aux réglages courants.
    pub fn result_is_stale(&self) -> bool {
        (self.result.is_some() && !self.result_is_final) || self.computing
    }

    /// Le calcul en cours est-il en pleine résolution ?
    pub fn computes_full_resolution(&self) -> bool {
        self.quality.is_full() || self.final_pass
    }

    /// Image utilisée pour les calculs, sous-échantillonnée si l'aperçu est
    /// en mode rapide.
    pub fn compute_source(&self) -> Option<RgbImage> {
        let src = self.src_rgb.as_ref()?;
        let scale = self.quality.scale();
        if self.final_pass {
            return Some(src.clone());
        }
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

    /// Index de l'image courante dans la navigation de dossier.
    pub fn folder_position(&self) -> Option<(usize, usize)> {
        self.folder_index
            .map(|index| (index + 1, self.folder_images.len()))
    }
}

/// Formate une durée en millisecondes de façon lisible.
pub(crate) use pointimg::frontend::{format_duration, format_memory};

/// Image d'exemple proposée dans l'état d'accueil, si le programme a été
/// lancé depuis les sources (ou si des exemples ont été installés à côté).
pub(crate) fn sample_image() -> Option<PathBuf> {
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

fn main() -> glib::ExitCode {
    env_logger::init();

    let app = adw::Application::builder().application_id(APP_ID).build();
    let (tx, rx) = async_channel::unbounded::<Msg>();

    app.connect_activate(move |app| {
        if app.active_window().is_some() {
            return;
        }
        let state = Rc::new(RefCell::new(State::new(tx.clone())));
        let (window, widgets) = ui::build_ui(app, Rc::clone(&state));

        let rx = rx.clone();
        let state_rx = Rc::clone(&state);
        let widgets_rx = Rc::clone(&widgets);
        glib::MainContext::default().spawn_local(async move {
            while let Ok(msg) = rx.recv().await {
                actions::handle_message(msg, &state_rx, &widgets_rx);
            }
        });

        window.present();
    });

    app.run()
}
