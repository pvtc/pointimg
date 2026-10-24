//! Entry point of the pointimg GUI.
//!
//! Architecture:
//! - Main thread = egui thread (mandatory on macOS/Windows)
//! - Filter compute in a separate thread via `std::thread::spawn`
//! - Communication: `Arc<Mutex<Option<DynamicImage>>>` + `AtomicBool` computing
//!   + cancel token
//! - Preview density map: computed outside the GUI thread and published when ready
//! - Progressive preview: for Voronoi/K-means, each iteration publishes an
//!   intermediate result
//! - Cached dots: stored in App after every full compute, used for
//!   the SVG export
//!
//! Breakdown:
//! - [`convert`]: conversion of `image` images → egui textures
//! - `format`: human formatting (durations) + generation guard
//! - [`io`]: atomic writes and validation of output paths
//! - [`app`]: application state and high-level logic (loading, undo/redo)
//! - [`compute`]: compute workers of the filter and of the density map
//! - [`save`]: result exports (preset, image, SVG)
//! - [`ui`]: egui rendering (header, control panel, central area)

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

    // GPU backend: on Linux we force Vulkan + GL (fallback for Wayland/EGL
    // depending on the NVIDIA/AMD/Intel driver). On macOS we let wgpu choose
    // Metal automatically. On Windows, the default backend (DX12/Vulkan)
    // works natively.
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
    // Path given as an argument: `pointimg-gui photo.jpg`. The window does not
    // exist yet here, loading waits for the first frame — that is
    // `App::pending_open` that takes care of it.
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

/// Window icon, read into the binary.
///
/// egui has no icon theme: unlike GTK, which looks for `pointimg` in the
/// desktop theme, the pixels must be provided. The PNG is embedded — that is
/// the cost of 20 kB — rather than read from disk, so as not to depend on the
/// installation directory.
///
/// The master SVG is in `assets/logo/`; the PNGs are derived from it by
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
            // A missing icon must not prevent the application from
            // starting: we report it and open without it.
            log::warn!(
                "{}",
                pointimg::frontend::t("icône de fenêtre illisible : {e}")
                    .replace("{e}", &e.to_string())
            );
            egui::IconData {
                rgba: Vec::new(),
                width: 0,
                height: 0,
            }
        }
    }
}

/// Image or folder paths received on the command line.
///
/// egui does not handle D-Bus activation like GTK: only the command line is
/// read here, which covers `pointimg-gui photo.jpg` and the desktop's file
/// associations.
fn pending_open_from_args() -> Option<PathBuf> {
    std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .find(|path| pointimg::frontend::is_supported_image(path) || path.is_dir())
}

/// History entry: the parameters of a past state + its label.
pub(crate) struct HistoryEntry {
    pub params: pointimg::filter::FilterParams,
    /// "Algorithm: Grid → Voronoi", "Number of points: 800 → 1500"…
    pub label: String,
}

/// Application state: shares the type between the `app`, `compute`,
/// `save` and `ui` modules (one `impl App` per module for its part of logic).
pub(crate) struct App {
    pub params: pointimg::filter::FilterParams,

    // Original source image (DynamicImage to support RGBA)
    pub src_dynamic: Option<DynamicImage>,
    // Source image converted to RGB8 (cached, recomputed if bg_color changes)
    pub src_rgb: Option<RgbImage>,
    pub src_path: Option<PathBuf>,
    /// Size of the source file in bytes, to compare the export to the original.
    pub source_size: Option<u64>,
    /// Last save directory (then offered to the dialogs).
    pub last_dir: Option<PathBuf>,
    /// Images of the current folder, for `←`/`→` navigation.
    pub folder_images: Vec<PathBuf>,
    pub folder_index: Option<usize>,
    pub src_texture: Option<egui::TextureHandle>,

    // Filter result (also updated at every intermediate preview).
    // Stored in RGBA to support transparent mode (alpha channel preserved).
    // The inner `Arc` lets the export size measurement borrow the image without
    // copying it (it weighs up to ~33 MB) — the same pattern as `density_data`.
    pub result: Arc<Mutex<Option<Arc<RgbaImage>>>>,
    pub result_texture: Option<egui::TextureHandle>,
    pub result_revision: Arc<AtomicU64>,
    pub result_texture_revision: u64,

    // Dots of the last completed compute (used for the SVG export)
    pub last_dots: Arc<Mutex<Option<Vec<Dot>>>>,

    // Preview density map, computed outside the GUI thread.
    pub density_image: Option<ImgGrayImage>,
    pub density_texture: Option<egui::TextureHandle>,
    pub density_result: Arc<Mutex<Option<ImgGrayImage>>>,
    pub density_data: Arc<Mutex<Option<Arc<Vec<f32>>>>>,
    /// Scale of the image used by the current compute (1.0 = full resolution).
    pub density_scale: f32,

    // Progress (iter_current, iter_total)
    pub progress: Arc<Mutex<(usize, usize)>>,

    // Compute in progress?
    pub computing: Arc<AtomicBool>,
    // Cancel token
    pub cancel: Arc<AtomicBool>,
    // Identifies the last requested compute. Obsolete workers publish nothing.
    pub compute_generation: Arc<AtomicU64>,
    // Identifies the last requested density map.
    pub density_generation: Arc<AtomicU64>,

    // Error from the compute thread
    pub compute_error: Arc<Mutex<Option<String>>>,

    // Compute time of the last finished render
    pub last_compute_ms: Option<u64>,
    pub compute_start: Option<Instant>,

    // Last instant at which a parameter changed, used for debounce.
    pub last_param_change: Option<Instant>,

    // Quick preview mode + automatic finalization.
    pub quality: Quality,
    /// Finalization pass in progress (full resolution despite quick mode).
    pub final_pass: bool,
    /// Is the displayed result in full resolution? Shared with the worker,
    /// which publishes it at the end of the compute.
    pub result_is_final: Arc<AtomicBool>,
    /// Instant at which to trigger the finalization.
    pub finalize_at: Option<Instant>,

    // Undo/redo: history of the committed FilterParams.
    pub history: Vec<HistoryEntry>,
    pub future: Vec<HistoryEntry>,
    // `last_committed` is the "live" version at the time of the last commit
    // (serves to avoid pushing 50 consecutive entries on the same slider).
    pub last_committed: Option<pointimg::filter::FilterParams>,
    // Flag: a change has just been detected, but the debounce is in
    // progress. When the compute finishes (and with no new commit planned),
    // the current state is pushed into the history.
    pub pending_commit: bool,
    /// The density map needs recomputing (the variance changed). It goes with
    /// the filter debounce: without that, every pixel of a slider drag would
    /// launch a density map compute over the whole image.
    pub density_pending: bool,

    // Export size, measured by real encoding in a worker thread
    // (see `exports.rs`). `export_sizes_revision` is the revision of the
    // result for which the displayed measurement was made.
    pub export_sizes: Option<pointimg::export::ExportSizes>,
    pub export_sizes_revision: u64,
    pub export_sizes_generation: Arc<AtomicU64>,
    pub export_sizes_computing: Arc<std::sync::atomic::AtomicBool>,
    pub export_sizes_slot: Arc<Mutex<Option<pointimg::export::ExportSizes>>>,

    // Display mode
    pub view_mode: ViewMode,

    // Zoom level (1.0 = 100 %)
    pub zoom: f32,
    // Whether the preview should fit the available panel.
    pub zoom_fit: bool,
    // Zoom actually applied ("Fit" mode: deduced from the viewport).
    pub effective_zoom: f32,
    // Position of the comparison handle (0 = all the way to the right).
    pub compare_split: f32,

    // Status message
    pub status: String,
    // Window title (file name), to avoid sending it again on every frame.
    pub window_title: String,
    /// Ephemeral notifications (errors, confirmations).
    pub toasts: notify::Toasts,
    /// Command palette open (`Ctrl+K`) and its search text.
    pub palette: Option<String>,
    /// Path given on the command line, loaded on the first frame: the window
    /// does not exist yet when `main` receives it.
    pub pending_open: Option<PathBuf>,
    /// Interface scale applied to the egui style (1.0 = comfortable).
    pub ui_scale: f32,
    /// Theme forced by the user; `None` follows the system preference.
    pub ui_theme: Option<egui::Theme>,
    /// Interface language.
    pub ui_locale: pointimg::frontend::Locale,
    /// Persisted interface preferences (scale, theme).
    pub prefs: Option<prefs::InterfacePrefs>,
    /// Preset thumbnails, decoded and uploaded to the GPU only once.
    ///
    /// They were rebuilt on every frame (nine `read_dir`, nine PNG decodes of
    /// ~130 kB and nine texture uploads), which a panel redrawn several times
    /// per second cannot afford. `None` as long as loading has not happened —
    /// that is what distinguishes "not done yet" from "done and empty".
    pub preset_thumbnails: Option<Vec<Option<egui::TextureHandle>>>,
    /// Scale for which the style has been applied (avoids redoing it on
    /// every frame).
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
            status: pointimg::frontend::t(
                "Ouvrez une image pour commencer (ou faites glisser un fichier).",
            )
            .to_string(),
            window_title: String::new(),
            toasts: notify::Toasts::new(),
            palette: None,
            pending_open: None,
            ui_scale: 1.0,
            ui_theme: None,
            ui_locale: pointimg::frontend::Locale::default(),
            prefs: None,
            preset_thumbnails: None,
            applied_style: None,
        }
    }
}

impl App {
    /// Builds the application by rereading the interface preferences saved at
    /// the previous launch.
    pub fn new() -> Self {
        let mut app = Self::default();
        if let Some(prefs) = prefs::InterfacePrefs::load() {
            if let Some(scale) = prefs.scale() {
                app.ui_scale = scale.clamp(0.85, 1.6);
            }
            // A missing key means "follow the system".
            app.ui_theme = prefs.theme();
            app.ui_locale = prefs
                .locale()
                .unwrap_or_else(pointimg::frontend::detect_system_locale);
            pointimg::frontend::set_locale(app.ui_locale);
            app.prefs = Some(prefs);
        } else {
            app.ui_locale = pointimg::frontend::detect_system_locale();
            pointimg::frontend::set_locale(app.ui_locale);
        }
        app
    }

    /// Saves an interface preference (immediately: quitting right after the
    /// change loses nothing).
    pub fn persist_display_preferences(&mut self) {
        if let Some(prefs) = &mut self.prefs {
            prefs.save(self.ui_scale, self.ui_theme, self.ui_locale);
        }
    }

    /// Applies the chosen theme to the context. "Auto" hands control back to
    /// egui, which then follows the system preference.
    pub(crate) fn apply_theme(&self, ctx: &egui::Context) {
        let preference = match self.ui_theme {
            Some(egui::Theme::Dark) => egui::ThemePreference::Dark,
            Some(egui::Theme::Light) => egui::ThemePreference::Light,
            None => egui::ThemePreference::System,
        };
        // Writing the options on every render would ask for another render: we
        // only touch the context if the preference has really changed.
        if ctx.options(|options| options.theme_preference) != preference {
            ctx.set_theme(preference);
        }
    }

    /// A setting change: arming the debounce + history.
    pub(crate) fn on_params_changed(&mut self, ctx: &egui::Context) {
        if self.src_rgb.is_none() {
            return;
        }
        self.last_param_change = Some(Instant::now());
        self.pending_commit = true;
        // A modification cancels the pending finalization: it is the new
        // setting that will be passed to full resolution.
        self.finalize_at = None;
        ctx.request_repaint();
    }

    /// Changes the preview quality and restarts a compute.
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

    /// Recomputes in full resolution without changing the chosen mode (
    /// Recompute button, Space, automatic finalization).
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

    /// Is the compute in progress in full resolution?
    pub(crate) fn computes_full_resolution(&self) -> bool {
        self.quality.is_full() || self.final_pass
    }

    /// Source image used for a compute, downsampled in quick mode.
    pub(crate) fn compute_source(&self) -> Option<RgbImage> {
        let src = self.src_rgb.as_ref()?;
        if self.final_pass {
            return Some(src.clone());
        }
        // The downsampling is shared with GTK: a preview must show the same
        // pixels on both sides.
        Some(pointimg::frontend::downsample(src, self.quality.scale()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded icon must be decodable, otherwise the window opens without
    /// an icon and nothing signals it to the user.
    #[test]
    fn licone_de_fenetre_est_decodable() {
        let icon = window_icon();
        assert_eq!(icon.width, 256);
        assert_eq!(icon.height, 256);
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
        // The center of the big cyan dot, at (163, 169) on the 512 grid,
        // i.e. (81, 84) in 256. The top-left corner is transparent: the
        // circle does not go through it.
        let x = 81 * 4;
        let y = 84 * icon.width as usize * 4;
        let centre = &icon.rgba[x + y..x + y + 4];
        assert_eq!(
            centre,
            &[0x00, 0xAE, 0xEF, 0xFF],
            "le centre du logo devrait être le cyan du point principal"
        );
        // And the corner, itself, must be transparent: a desktop icon must not
        // set an opaque square.
        assert_eq!(&icon.rgba[..4], &[0, 0, 0, 0]);
    }
}
