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
//! - [`ui`] : construction de la fenêtre, panneau de réglages, aperçu, signaux.
//! - [`actions`] : workers, chargement/export, presets, undo/redo, calculs.

mod actions;
mod controls;
mod files;
mod preview;
mod sections;
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
    },
}

/// État applicatif partagé entre les modules.
pub(crate) struct State {
    pub params: FilterParams,

    pub src_dynamic: Option<DynamicImage>,
    pub src_rgb: Option<RgbImage>,
    pub src_path: Option<PathBuf>,

    pub result: Option<RgbaImage>,
    pub last_dots: Option<Vec<Dot>>,
    pub density_image: Option<GrayImage>,
    pub density_data: Option<Arc<Vec<f32>>>,

    pub view_mode: ViewMode,
    pub zoom: f64,
    pub zoom_fit: bool,
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

    pub cancel: Arc<AtomicBool>,
    pub generation: Arc<AtomicU64>,
    pub density_generation: Arc<AtomicU64>,
    pub tx: async_channel::Sender<Msg>,

    // Cache des textures d'aperçu.
    pub src_texture: Option<gtk::gdk::Texture>,
    pub result_texture: Option<gtk::gdk::Texture>,
    pub result_texture_revision: u64,
    pub density_texture: Option<gtk::gdk::Texture>,
}

impl State {
    fn new(tx: async_channel::Sender<Msg>) -> Self {
        Self {
            params: FilterParams::default(),
            src_dynamic: None,
            src_rgb: None,
            src_path: None,
            result: None,
            last_dots: None,
            density_image: None,
            density_data: None,
            view_mode: ViewMode::Side,
            zoom: 1.0,
            zoom_fit: true,
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
            cancel: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicU64::new(0)),
            density_generation: Arc::new(AtomicU64::new(0)),
            tx,
            src_texture: None,
            result_texture: None,
            result_texture_revision: u64::MAX,
            density_texture: None,
        }
    }
}

/// Formate une durée en millisecondes de façon lisible.
pub(crate) use pointimg::frontend::{format_duration, format_memory};

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

        let state = Rc::clone(&state);
        let widgets = Rc::clone(&widgets);
        let rx = rx.clone();
        glib::MainContext::default().spawn_local(async move {
            while let Ok(msg) = rx.recv().await {
                actions::handle_message(msg, &state, &widgets);
            }
        });

        window.present();
    });

    app.run()
}
