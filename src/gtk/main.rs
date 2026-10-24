//! GTK4 / libadwaita interface of pointimg.
//!
//! Architecture:
//! - The GTK main thread owns the application state (`State`, behind
//!   `Rc<RefCell<..>>`) and all the widgets (`Widgets`).
//! - The filter and density map computations run in dedicated threads and
//!   publish their results through an `async-channel` channel.
//! - A local future (via `glib::MainContext::spawn_local`) consumes the
//!   channel on the main thread and updates the interface.
//!
//! Breakdown:
//! - [`ui`]: building the window and the shortcuts.
//! - [`actions`]: workers, loading/export, presets, undo/redo, computes.
//! - [`controls`]: building and laying out the widgets.
//! - [`sections`]: state → widgets synchronization, section visibility.
//! - [`preview`]: textures, zoom, previews, status bar.
//! - [`compare`]: before/after comparison by sweeping.
//! - [`wire`]: GTK signals → state, menus and keyboard shortcuts.

mod actions;
mod compare;
mod confirm;
mod controls;
mod exports;
mod files;
mod prefs;
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

/// Messages sent by the workers to the GTK thread.
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
        /// Scale of the image used (the density map depends on it).
        scale: f32,
    },
    /// Measured export sizes (real encoding, off the main thread).
    ExportSizes {
        generation: u64,
        sizes: ExportSizes,
    },
}

/// Shared application state between the modules.
pub(crate) struct State {
    pub params: FilterParams,

    pub src_dynamic: Option<DynamicImage>,
    pub src_rgb: Option<RgbImage>,
    pub src_path: Option<PathBuf>,
    /// Last folder used to open/save (then offered by the file dialogs).
    pub last_dir: Option<PathBuf>,
    /// Images of the current folder (`←`/`→` navigation), if the file was
    /// opened through "Ouvrir un dossier".
    pub folder_images: Vec<PathBuf>,
    pub folder_index: Option<usize>,
    /// Original size before any downsampling, for the tooltip.
    pub source_was_resized: bool,

    pub result: Option<Arc<RgbaImage>>,
    pub last_dots: Option<Vec<Dot>>,
    /// Source file size in bytes (reference for the export size).
    pub source_size: Option<u64>,
    /// Export size measured by real encoding, and its computation state.
    /// (see `exports.rs`)
    pub export_sizes: Option<ExportSizes>,
    /// Revision of the result: the export measurement is redone when it
    /// changes.
    pub result_revision: u64,
    pub export_sizes_revision: u64,
    pub export_sizes_generation: Arc<AtomicU64>,
    pub export_sizes_computing: Arc<AtomicBool>,
    pub density_image: Option<GrayImage>,
    pub density_data: Option<Arc<Vec<f32>>>,
    /// Scale at which `density_data` was computed: the density map is only
    /// reusable by a compute of the same resolution.
    pub density_scale: f32,

    pub view_mode: ViewMode,
    pub zoom: f64,
    pub zoom_fit: bool,
    /// Zoom actually applied ("Ajuster" mode: computed from the viewport).
    pub effective_zoom: f64,
    pub status: String,
    pub last_compute_ms: Option<u64>,
    pub compute_start: Option<Instant>,

    pub history: Vec<FilterParams>,
    pub future: Vec<FilterParams>,
    pub last_committed: Option<FilterParams>,
    pub pending_commit: bool,
    /// Prevents widget callbacks from reacting to a programmatic update
    /// (undo, preset loading).
    pub updating: bool,
    pub debounce: Option<glib::SourceId>,

    pub computing: bool,
    pub user_cancelled: bool,
    pub restart_pending: bool,

    /// Mode chosen by the user (fast or full resolution).
    pub quality: Quality,
    /// Finalization pass in progress: the displayed render is at full
    /// resolution even if fast preview is the mode being used.
    pub final_pass: bool,
    /// Is the displayed result at full resolution?
    pub result_is_final: bool,
    /// Finalization timer: returns to full resolution when the user stops
    /// adjusting the settings.
    pub finalize_pending: Option<glib::SourceId>,

    pub cancel: Arc<AtomicBool>,
    pub generation: Arc<AtomicU64>,
    pub density_generation: Arc<AtomicU64>,
    pub tx: async_channel::Sender<Msg>,

    // Cache of the preview textures.
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
            status: pointimg::frontend::gettext(
                "Ouvrez une image pour commencer (ou glissez-déposez un fichier).",
            )
            .to_string(),
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

    /// The displayed result no longer matches the current settings.
    pub fn result_is_stale(&self) -> bool {
        (self.result.is_some() && !self.result_is_final) || self.computing
    }

    /// Is the running compute at full resolution?
    pub fn computes_full_resolution(&self) -> bool {
        self.quality.is_full() || self.final_pass
    }

    /// Image used for the computes, downsampled if the preview is in fast
    /// mode.
    pub fn compute_source(&self) -> Option<RgbImage> {
        let src = self.src_rgb.as_ref()?;
        if self.final_pass {
            return Some(src.clone());
        }
        // Same downsampling as egui: comparing two previews would make no
        // sense if they did not show the same pixels.
        Some(pointimg::frontend::downsample(src, self.quality.scale()))
    }

    /// Index of the current image in the folder navigation.
    pub fn folder_position(&self) -> Option<(usize, usize)> {
        self.folder_index
            .map(|index| (index + 1, self.folder_images.len()))
    }
}

/// Formats a duration in milliseconds in a readable way.
pub(crate) use pointimg::frontend::{format_duration, format_memory};

/// The open window, accessible from the `open` signal.
///
/// `GApplication` has only one window: the `activate` signal builds it, the
/// `open` signal (paths received as an argument or from another program) then
/// loads the files into it. These handles connect the two.
type WindowHandles = (
    Rc<RefCell<State>>,
    Rc<RefCell<Option<Rc<controls::Widgets>>>>,
);

fn main() -> glib::ExitCode {
    env_logger::init();
    pointimg::frontend::init_i18n();
    if let Some(locale) = prefs::load_locale() {
        pointimg::frontend::set_locale(locale);
    } else {
        pointimg::frontend::set_locale(pointimg::frontend::detect_system_locale());
    }

    let app = adw::Application::builder()
        .application_id(APP_ID)
        // `HANDLES_OPEN`: the path is transmitted by the `open` signal rather
        // than by `activate`. That is what allows both
        // `pointimg-gtk photo.jpg` and opening from the file
        // manager (D-Bus activation), which go through the same path.
        .flags(gtk::gio::ApplicationFlags::HANDLES_OPEN)
        .build();
    let (tx, rx) = async_channel::unbounded::<Msg>();

    // `activate` builds the window, `open` must then load files into it. The
    // bridge between the two signals: `activate` is emitted synchronously, so
    // the filling takes place before `app.activate()` returns below.
    let handles: Rc<RefCell<Option<WindowHandles>>> = Rc::new(RefCell::new(None));
    let handles_for_activate = Rc::clone(&handles);

    app.connect_activate(move |app| {
        if app.active_window().is_some() {
            return;
        }
        let state = Rc::new(RefCell::new(State::new(tx.clone())));
        let (window, widgets) = ui::build_ui(app, Rc::clone(&state));
        let widgets_cell: Rc<RefCell<Option<Rc<controls::Widgets>>>> =
            Rc::new(RefCell::new(Some(Rc::clone(&widgets))));
        *handles_for_activate.borrow_mut() = Some((Rc::clone(&state), Rc::clone(&widgets_cell)));

        let rx = rx.clone();
        let state_rx = Rc::clone(&state);
        let widgets_rx = Rc::clone(&widgets_cell);
        glib::MainContext::default().spawn_local(async move {
            while let Ok(msg) = rx.recv().await {
                let widgets = widgets_rx.borrow().clone();
                if let Some(widgets) = widgets {
                    actions::handle_message(msg, &state_rx, &widgets);
                }
            }
        });

        window.present();
    });

    // Files (and folders) passed as an argument, or chosen in another
    // program. The first one is opened in the window, the next ones fill it
    // in turn — same behaviour as a drag and drop.
    app.connect_open(move |app, files, _hint| {
        app.activate();
        let borrowed = handles.borrow();
        let Some((state, widgets_cell)) = borrowed.as_ref() else {
            return;
        };
        let Some(widgets) = widgets_cell.borrow().clone() else {
            return;
        };
        let paths: Vec<PathBuf> = files.iter().filter_map(|file| file.path()).collect();
        for path in paths {
            if path.is_dir() {
                actions::open_folder(path, state, &widgets);
            } else {
                actions::load_image(path, state, &widgets);
            }
        }
    });

    // `run_with_args_os` rather than `run()`: it is the one that will transmit
    // the command line paths to the `open` signal.
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    app.run_with_args_os(&args)
}
