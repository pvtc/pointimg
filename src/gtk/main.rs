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
mod confirm;
mod controls;
mod exports;
mod files;
mod preview;
#[cfg(test)]
mod ranges;
mod recent;
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
use pointimg::export::ExportSizes;
use pointimg::filter::{Dot, FilterParams};
use pointimg::frontend::{Quality, ViewMode};

const APP_ID: &str = "org.pointimg.Pointimg";

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
    /// Tailles d'export mesurées (encodage réel, hors thread principal).
    ExportSizes {
        generation: u64,
        sizes: ExportSizes,
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

    pub result: Option<Arc<RgbaImage>>,
    pub last_dots: Option<Vec<Dot>>,
    /// Taille du fichier source en octets (référence de la taille d'export).
    pub source_size: Option<u64>,
    /// Taille d'export mesurée par encodage réel, et son état de calcul.
    /// (cf. `exports.rs`)
    pub export_sizes: Option<ExportSizes>,
    /// Révision du résultat : la mesure d'export est refaite quand elle change.
    pub result_revision: u64,
    pub export_sizes_revision: u64,
    pub export_sizes_generation: Arc<AtomicU64>,
    pub export_sizes_computing: Arc<AtomicBool>,
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

impl State {
    fn new(tx: async_channel::Sender<Msg>) -> Self {
        Self {
            params: FilterParams::default(),
            src_dynamic: None,
            src_rgb: None,
            src_path: None,
            source_size: None,
            export_sizes: None,
            result_revision: 0,
            export_sizes_revision: 0,
            export_sizes_generation: Arc::new(AtomicU64::new(0)),
            export_sizes_computing: Arc::new(AtomicBool::new(false)),
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
        if self.final_pass {
            return Some(src.clone());
        }
        // Même sous-échantillonnage qu'egui : comparer deux aperçus n'aurait
        // aucun sens s'ils ne montraient pas les mêmes pixels.
        Some(pointimg::frontend::downsample(src, self.quality.scale()))
    }

    /// Index de l'image courante dans la navigation de dossier.
    pub fn folder_position(&self) -> Option<(usize, usize)> {
        self.folder_index
            .map(|index| (index + 1, self.folder_images.len()))
    }
}

/// Formate une durée en millisecondes de façon lisible.
pub(crate) use pointimg::frontend::{format_duration, format_memory};

/// La fenêtre ouverte, accessible depuis le signal `open`.
///
/// `GApplication` n'a qu'une fenêtre : le signal `activate` la construit, le
/// signal `open` (chemins reçus en argument ou d'un autre programme) charge
/// ensuite les fichiers dedans. Ces poignées font le lien entre les deux.
type WindowHandles = (Rc<RefCell<State>>, Rc<controls::Widgets>);

fn main() -> glib::ExitCode {
    env_logger::init();

    let app = adw::Application::builder()
        .application_id(APP_ID)
        // `HANDLES_OPEN` : le chemin est transmis par le signal `open` plutôt
        // que par `activate`. C'est ce qui permet à la fois
        // `pointimg-gtk photo.jpg` et l'ouverture depuis le gestionnaire de
        // fichiers (activation D-Bus), qui passe par le même chemin.
        .flags(gtk::gio::ApplicationFlags::HANDLES_OPEN)
        .build();
    let (tx, rx) = async_channel::unbounded::<Msg>();

    // `activate` construit la fenêtre, `open` doit ensuite y charger des
    // fichiers. Le pont entre les deux signaux : `activate` est émis de façon
    // synchrone, donc le remplissage a lieu avant le retour de
    // `app.activate()` ci-dessous.
    let handles: Rc<RefCell<Option<WindowHandles>>> = Rc::new(RefCell::new(None));
    let handles_for_activate = Rc::clone(&handles);

    app.connect_activate(move |app| {
        if app.active_window().is_some() {
            return;
        }
        let state = Rc::new(RefCell::new(State::new(tx.clone())));
        let (window, widgets) = ui::build_ui(app, Rc::clone(&state));
        *handles_for_activate.borrow_mut() = Some((Rc::clone(&state), Rc::clone(&widgets)));

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

    // Fichiers (et dossiers) passés en argument, ou choisis dans un autre
    // programme. Le premier est ouvert dans la fenêtre, les suivants la
    // remplissent à leur tour — même comportement qu'un glisser-déposer.
    app.connect_open(move |app, files, _hint| {
        app.activate();
        let borrowed = handles.borrow();
        let Some((state, widgets)) = borrowed.as_ref() else {
            return;
        };
        let paths: Vec<PathBuf> = files.iter().filter_map(|file| file.path()).collect();
        for path in paths {
            if path.is_dir() {
                actions::open_folder(path, state, widgets);
            } else {
                actions::load_image(path, state, widgets);
            }
        }
    });

    // `run_with_args_os` plutôt que `run()` : c'est lui qui transmettra les
    // chemins de la ligne de commande au signal `open`.
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    app.run_with_args_os(&args)
}
