/// Point d'entrée de la GUI pointimg.
///
/// Architecture :
/// - Thread principal = thread egui (obligatoire sur macOS/Windows)
/// - Calcul du filtre dans un thread séparé via `std::thread::spawn`
/// - Communication : `Arc<Mutex<Option<RgbImage>>>` + `AtomicBool` computing + cancel token
/// - Density map cachée : recalculée uniquement au chargement d'une nouvelle image
/// - Preview progressive : pour Voronoï/K-means, chaque itération publie un résultat intermédiaire
/// - Dots cachés : stockés dans App après chaque calcul complet, utilisés pour l'export SVG
use pointimg::filter::{self, Algorithm, Dot, DotShape, FilterParams};

use eframe::egui;
use egui::{ColorImage, TextureHandle, TextureOptions};
use image::{DynamicImage, GrayImage as ImgGrayImage, RgbImage, GenericImageView};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn main() -> eframe::Result {
    env_logger::init();
    use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};

    let wgpu_options = WgpuConfiguration {
        wgpu_setup: WgpuSetup::CreateNew(WgpuSetupCreateNew {
            instance_descriptor: wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
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

// ─── Mode d'affichage ─────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum ViewMode {
    Side,       // source | résultat côte à côte
    ResultOnly, // résultat seul
    SourceOnly, // source seule
    DensityMap, // aperçu de la density map
}

// ─── État de l'application ────────────────────────────────────────────────────

struct App {
    params: FilterParams,

    // Image source originale (DynamicImage pour supporter RGBA)
    src_dynamic: Option<DynamicImage>,
    // Image source convertie en RGB8 (mise en cache, recalculée si bg_color change)
    src_rgb: Option<RgbImage>,
    src_path: Option<PathBuf>,
    src_texture: Option<TextureHandle>,

    // Résultat du filtre (mis à jour à chaque preview intermédiaire aussi)
    result: Arc<Mutex<Option<RgbImage>>>,
    result_texture: Option<TextureHandle>,

    // Dots du dernier calcul terminé (utilisés pour export SVG)
    last_dots: Arc<Mutex<Option<Vec<Dot>>>>,

    // Density map mise en cache (recalculée uniquement au chargement d'image)
    density_image: Option<ImgGrayImage>,
    density_texture: Option<TextureHandle>,

    // Progression (iter_courant, iter_total)
    progress: Arc<Mutex<(usize, usize)>>,

    // Calcul en cours ?
    computing: Arc<AtomicBool>,
    // Token d'annulation
    cancel: Arc<AtomicBool>,

    // Erreur du thread de calcul
    compute_error: Arc<Mutex<Option<String>>>,

    // Temps de calcul du dernier rendu terminé
    last_compute_ms: Option<u64>,
    compute_start: Option<Instant>,

    // Debounce UX 11 : dernier instant où un paramètre a changé
    last_param_change: Option<Instant>,

    // Mode d'affichage
    view_mode: ViewMode,

    // Niveau de zoom (1.0 = 100%)
    zoom: f32,
    // U3: flag for "fit to panel" mode
    zoom_fit: bool,

    // Message de statut
    status: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            params: FilterParams::default(),
            src_dynamic: None,
            src_rgb: None,
            src_path: None,
            src_texture: None,
            result: Arc::new(Mutex::new(None)),
            result_texture: None,
            last_dots: Arc::new(Mutex::new(None)),
            density_image: None,
            density_texture: None,
            progress: Arc::new(Mutex::new((0, 0))),
            computing: Arc::new(AtomicBool::new(false)),
            cancel: Arc::new(AtomicBool::new(false)),
            compute_error: Arc::new(Mutex::new(None)),
            last_compute_ms: None,
            compute_start: None,
            last_param_change: None,
            view_mode: ViewMode::Side,
            zoom: 1.0,
            zoom_fit: true,
            status: "Ouvrez une image pour commencer (ou faites glisser un fichier).".to_string(),
        }
    }
}

// ─── Conversion image → texture egui ─────────────────────────────────────────

fn rgb_to_color_image(img: &RgbImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let pixels: Vec<egui::Color32> = img
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
        .collect();
    ColorImage {
        size: [w as usize, h as usize],
        pixels,
    }
}

fn gray_to_color_image(img: &ImgGrayImage) -> ColorImage {
    let (w, h) = img.dimensions();
    let pixels: Vec<egui::Color32> = img
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[0], p[0]))
        .collect();
    ColorImage {
        size: [w as usize, h as usize],
        pixels,
    }
}

// ─── Formatage du temps ───────────────────────────────────────────────────────

fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        let secs = ms / 1000;
        format!("{}m {}s", secs / 60, secs % 60)
    }
}

// ─── Logique de l'application ─────────────────────────────────────────────────

impl App {
    fn trigger_compute(&mut self, ctx: &egui::Context) {
        if self.computing.load(Ordering::Relaxed) {
            // Annuler le calcul en cours, puis relancer
            self.cancel.store(true, Ordering::Relaxed);
            return;
        }
        self.start_compute(ctx);
    }

    fn start_compute(&mut self, ctx: &egui::Context) {
        let src = match &self.src_rgb {
            Some(s) => s.clone(),
            None => return,
        };
        let params = self.params.clone();
        let result = Arc::clone(&self.result);
        let computing = Arc::clone(&self.computing);
        let cancel = Arc::clone(&self.cancel);
        let progress = Arc::clone(&self.progress);
        let status_err = Arc::clone(&self.compute_error);
        let last_dots = Arc::clone(&self.last_dots);
        let ctx = ctx.clone();

        computing.store(true, Ordering::Relaxed);
        cancel.store(false, Ordering::Relaxed);
        *self.progress.lock().unwrap_or_else(|e| e.into_inner()) = (0, params.iterations);
        self.status = "Calcul en cours…".to_string();
        self.compute_start = Some(Instant::now());
        self.last_param_change = None;

        std::thread::spawn(move || {
            let iters = params.iterations;
            // P3: throttle preview cloning to at most once per 100ms to avoid
            // cloning a full image (potentially 36MB on 4K) at every iteration
            let mut last_preview = Instant::now();
            let preview_interval = Duration::from_millis(100);

            let res = filter::apply_with_progress(&src, &params, &cancel, |iter, total, preview: &RgbImage| {
                *progress.lock().unwrap_or_else(|e| e.into_inner()) = (iter, total);
                let now = Instant::now();
                // Always clone on last iteration, throttle intermediate previews
                if iter == total || now.duration_since(last_preview) >= preview_interval {
                    *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(preview.clone());
                    last_preview = now;
                    ctx.request_repaint();
                }
            });
            match res {
                Ok((dst, dots)) => {
                    *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(dst);
                    *progress.lock().unwrap_or_else(|e| e.into_inner()) = (iters, iters);
                    *last_dots.lock().unwrap_or_else(|e| e.into_inner()) = Some(dots);
                }
                Err(e) => {
                    if !cancel.load(Ordering::Relaxed) {
                        *status_err.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("Erreur : {e}"));
                    }
                }
            }
            computing.store(false, Ordering::Relaxed);
            ctx.request_repaint();
        });
    }

    fn load_image(&mut self, path: PathBuf, ctx: &egui::Context) {
        match image::open(&path) {
            Ok(img) => {
                const MAX_DIMENSION: u32 = 65535;
                const MAX_PIXELS: u64 = 256 * 1024 * 1024; // 256M pixels ≈ 1GB RAM (RGBA)
                let (w, h) = img.dimensions();
                let pixels = (w as u64) * (h as u64);
                
                if w > MAX_DIMENSION || h > MAX_DIMENSION {
                    self.status = format!("Image trop grande : {}x{} (max {}x{})", w, h, MAX_DIMENSION, MAX_DIMENSION);
                    return;
                }
                if pixels > MAX_PIXELS {
                    self.status = format!("Image trop grande : {} pixels (max {})", pixels, MAX_PIXELS);
                    return;
                }
                // Utiliser filter::flatten_to_rgb (bug 2 : pas de duplication)
                let rgb = filter::flatten_to_rgb(&img, self.params.bg_color);
                // Density map mise en cache ici
                let density = filter::compute_density_image(&rgb, self.params.variance_sensitivity);
                self.density_image = Some(density);
                self.density_texture = None;
                self.src_dynamic = Some(img);
                self.src_rgb = Some(rgb);
                self.src_path = Some(path.clone());
                self.src_texture = None;
                self.result_texture = None;
                *self.result.lock().unwrap_or_else(|e| e.into_inner()) = None;
                *self.last_dots.lock().unwrap_or_else(|e| e.into_inner()) = None;
                self.status = format!("Image chargée : {}", path.display());
                self.start_compute(ctx);
            }
            Err(e) => {
                self.status = format!("Erreur chargement : {e}");
            }
        }
    }

    fn save_result(&mut self, path: PathBuf) {
        // UX 16 : valider/forcer l'extension
        let path = ensure_extension(path, "png");
        let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(img) = guard.as_ref() {
            match img.save(&path) {
                Ok(_) => self.status = format!("Sauvegardé : {}", path.display()),
                Err(e) => self.status = format!("Erreur sauvegarde : {e}"),
            }
        }
    }

    fn save_svg(&mut self, path: PathBuf) {
        let path = ensure_extension(path, "svg");
        // Utiliser render_svg_from_dots si on a les dots (archi 19)
        let dots_guard = self.last_dots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dots) = dots_guard.as_ref()
            && let Some(src) = &self.src_rgb
        {
            let (w, h) = src.dimensions();
            match filter::render_svg_from_dots(w, h, dots, &self.params) {
                Ok(svg) => match std::fs::write(&path, svg) {
                    Ok(_) => self.status = format!("SVG sauvegardé : {}", path.display()),
                    Err(e) => self.status = format!("Erreur écriture SVG : {e}"),
                },
                Err(e) => self.status = format!("Erreur rendu SVG : {e}"),
            }
            return;
        }
        drop(dots_guard);
        // Fallback : recalculer
        if let Some(src) = &self.src_rgb {
            match filter::render_svg(src, &self.params) {
                Ok(svg) => match std::fs::write(&path, svg) {
                    Ok(_) => self.status = format!("SVG sauvegardé : {}", path.display()),
                    Err(e) => self.status = format!("Erreur écriture SVG : {e}"),
                },
                Err(e) => self.status = format!("Erreur rendu SVG : {e}"),
            }
        }
    }

    /// Reconstruit src_rgb depuis src_dynamic si bg_color a changé.
    fn refresh_src_rgb(&mut self) {
        if let Some(img) = &self.src_dynamic {
            // filter::flatten_to_rgb gère la composition alpha (bug 2)
            let new_rgb = filter::flatten_to_rgb(img, self.params.bg_color);
            let density = filter::compute_density_image(&new_rgb, self.params.variance_sensitivity);
            self.density_image = Some(density);
            self.density_texture = None;
            self.src_rgb = Some(new_rgb);
            self.src_texture = None;
        }
    }
}

// ─── UX 16 : validation d'extension ──────────────────────────────────────────

fn ensure_extension(mut path: PathBuf, default_ext: &str) -> PathBuf {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if matches!(ext.to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "svg") => path,
        _ => {
            path.set_extension(default_ext);
            path
        }
    }
}

// ─── Interface egui ───────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── U4: Keyboard shortcuts ────────────────────────────────────────────
        let (kb_open, kb_save, kb_recalc) = ctx.input(|i| {
            let open = i.modifiers.command && i.key_pressed(egui::Key::O);
            let save = i.modifiers.command && i.key_pressed(egui::Key::S);
            let recalc = i.key_pressed(egui::Key::Space) && !i.modifiers.command;
            (open, save, recalc)
        });

        if kb_open
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
                .pick_file()
        {
            self.load_image(path, ctx);
        }
        if kb_save {
            let has_result = self.result.lock().unwrap_or_else(|e| e.into_inner()).is_some();
            if has_result && !self.computing.load(Ordering::Relaxed)
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("PNG", &["png"])
                    .add_filter("JPEG", &["jpg", "jpeg"])
                    .save_file()
            {
                self.save_result(path);
            }
        }
        if kb_recalc && self.src_rgb.is_some() && !self.computing.load(Ordering::Relaxed) {
            self.last_param_change = None;
            self.start_compute(ctx);
        }

        // ── Drag & drop ───────────────────────────────────────────────────────
        let dropped_path = ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                i.raw.dropped_files[0].path.clone()
            } else {
                None
            }
        });
        if let Some(path) = dropped_path {
            self.load_image(path, ctx);
        }

        // ── Gestion du cancel + relance ───────────────────────────────────────
        if self.cancel.load(Ordering::Relaxed) && !self.computing.load(Ordering::Relaxed) {
            self.cancel.store(false, Ordering::Relaxed);
            self.start_compute(ctx);
        }

        // ── Debounce UX 11 : déclencher le calcul 300ms après le dernier changement ──
        if let Some(t) = self.last_param_change {
            if t.elapsed() >= Duration::from_millis(300) && self.src_rgb.is_some() {
                self.last_param_change = None;
                self.trigger_compute(ctx);
            } else {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
        }

        // ── Résultat prêt ─────────────────────────────────────────────────────
        let computing = self.computing.load(Ordering::Relaxed);
        if !computing {
            // Invalider la texture résultat à chaque frame pour prendre les previews
            let has_new = self.result.lock().unwrap_or_else(|e| e.into_inner()).is_some();
            if has_new {
                self.result_texture = None;
            }
            // Calculer le temps si on vient de terminer
            if let Some(start) = self.compute_start.take() {
                let ms = start.elapsed().as_millis() as u64;
                self.last_compute_ms = Some(ms);
                self.status = format!("Terminé en {}.", format_duration(ms));
            }
        } else {
            // Pendant le calcul : invalider la texture pour prendre les nouvelles previews
            // (le worker appelle ctx.request_repaint() à chaque itération)
            self.result_texture = None;
        }

        // ── Erreur du thread ──────────────────────────────────────────────────
        if let Some(err) = self.compute_error.lock().unwrap_or_else(|e| e.into_inner()).take() {
            self.status = err;
        }

        // ── Panneau de contrôle ───────────────────────────────────────────────
        egui::SidePanel::left("controls")
            .resizable(true)   // UX 15
            .min_width(290.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("pointimg");
                    ui.separator();

                    // Bouton ouvrir
                    if ui.button("Ouvrir une image…").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
                            .pick_file()
                    {
                        self.load_image(path, ctx);
                    }
                    ui.small("(ou glisser-déposer une image dans la fenêtre)");

                    ui.separator();
                    ui.label("Algorithme");

                    let mut algo_changed = false;
                    egui::Grid::new("algo_grid").num_columns(2).show(ui, |ui| {
                        for (label, algo) in [
                            ("Grille", Algorithm::Grid),
                            ("K-means", Algorithm::Kmeans),
                            ("Voronoi (Lloyd)", Algorithm::Voronoi),
                            ("Quadtree", Algorithm::Quadtree),
                        ] {
                            if ui
                                .selectable_label(self.params.algorithm == algo, label)
                                .clicked()
                            {
                                self.params.algorithm = algo;
                                algo_changed = true;
                            }
                            ui.end_row();
                        }
                    });

                    ui.separator();

                    let mut params_changed = algo_changed;

                    ui.label("Sensibilite variance");
                    let vs_changed = ui
                        .add(egui::Slider::new(&mut self.params.variance_sensitivity, 0.0..=1.0).step_by(0.01))
                        .changed();
                    if vs_changed {
                        // C1: recalculate density image when variance_sensitivity changes
                        if let Some(src) = &self.src_rgb {
                            self.density_image = Some(filter::compute_density_image(src, self.params.variance_sensitivity));
                            self.density_texture = None;
                        }
                    }
                    params_changed |= vs_changed;

                    ui.label("Rayon min (fraction image)");
                    params_changed |= ui
                        .add(egui::Slider::new(&mut self.params.min_radius_ratio, 0.001..=0.02).step_by(0.001))
                        .changed();

                    ui.label("Rayon max (fraction image)");
                    params_changed |= ui
                        .add(egui::Slider::new(&mut self.params.max_radius_ratio, 0.01..=0.3).step_by(0.005))
                        .changed();

                    ui.label("Boost zones uniformes (×max)");
                    params_changed |= ui
                        .add(egui::Slider::new(&mut self.params.max_boost, 1.0..=5.0).step_by(0.1))
                        .changed();

                    match self.params.algorithm {
                        Algorithm::Grid => {
                            ui.label("Colonnes");
                            params_changed |= ui
                                .add(egui::Slider::new(&mut self.params.cols, 10..=300))
                                .changed();
                        }
                        Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree => {
                            ui.label("Nombre de points");
                            params_changed |= ui
                                .add(egui::Slider::new(&mut self.params.num_points, 50..=5000).logarithmic(true))
                                .changed();

                            if matches!(self.params.algorithm, Algorithm::Kmeans | Algorithm::Voronoi) {
                                ui.label("Itérations");
                                params_changed |= ui
                                    .add(egui::Slider::new(&mut self.params.iterations, 1..=30))
                                    .changed();
                            }
                        }
                    }

                    ui.separator();

                    // ── Forme des dots ────────────────────────────────────────
                    ui.label("Forme des points");
                    let shape_changed = show_shape_selector(ui, &mut self.params.dot_shape);
                    params_changed |= shape_changed;

                    ui.separator();

                    // Palette réduite
                    ui.horizontal(|ui| {
                        let mut use_palette = self.params.palette_size.is_some();
                        if ui.checkbox(&mut use_palette, "Palette réduite").changed() {
                            self.params.palette_size = if use_palette { Some(8) } else { None };
                            params_changed = true;
                        }
                        if let Some(ref mut n) = self.params.palette_size {
                            params_changed |= ui.add(egui::Slider::new(n, 2..=32)).changed();
                        }
                    });

                    // Seed reproductible
                    ui.horizontal(|ui| {
                        let mut use_seed = self.params.rng_seed.is_some();
                        if ui.checkbox(&mut use_seed, "Seed fixé").changed() {
                            self.params.rng_seed = if use_seed { Some(42) } else { None };
                            params_changed = true;
                        }
                        if let Some(ref mut s) = self.params.rng_seed {
                            let mut seed_i64 = *s as i64;
                            if ui.add(egui::DragValue::new(&mut seed_i64).speed(1.0)).changed() {
                                *s = seed_i64.unsigned_abs();
                                params_changed = true;
                            }
                        }
                    });

                    ui.separator();

                    // ── Fond UX 14 : color picker ─────────────────────────────
                    ui.label("Couleur de fond");
                    ui.horizontal(|ui| {
                        let old_color = self.params.bg_color;
                        let mut color = egui::Color32::from_rgb(
                            old_color[0], old_color[1], old_color[2],
                        );
                        if egui::color_picker::color_edit_button_srgba(
                            ui,
                            &mut color,
                            egui::color_picker::Alpha::Opaque,
                        ).changed() {
                            self.params.bg_color = [color.r(), color.g(), color.b()];
                            self.refresh_src_rgb();
                            params_changed = true;
                        }
                        // Raccourcis Blanc / Noir
                        if ui.small_button("Blanc").clicked() {
                            self.params.bg_color = [255, 255, 255];
                            self.refresh_src_rgb();
                            params_changed = true;
                        }
                        if ui.small_button("Noir").clicked() {
                            self.params.bg_color = [0, 0, 0];
                            self.refresh_src_rgb();
                            params_changed = true;
                        }
                    });

                    ui.separator();

                    // ── Zoom ──────────────────────────────────────────────────
                    ui.horizontal(|ui| {
                        ui.label("Zoom");
                        if ui.add(egui::Slider::new(&mut self.zoom, 0.1..=4.0).step_by(0.1)).changed() {
                            self.zoom_fit = false; // U3: manual zoom disables fit
                        }
                        if ui.small_button("1:1").clicked() {
                            self.zoom = 1.0;
                            self.zoom_fit = false;
                        }
                        if ui.small_button("Fit").clicked() {
                            self.zoom_fit = true; // U3: use flag instead of zoom=0.0
                        }
                    });

                    // ── Boutons Recalculer / Annuler ──────────────────────────
                    let has_src = self.src_rgb.is_some();
                    let is_computing = computing;
                    let is_cancelling = self.cancel.load(Ordering::Relaxed);

                    ui.separator();
                    ui.horizontal(|ui| {
                        if is_computing || is_cancelling {
                            if ui.button("Annuler").clicked() {
                                self.cancel.store(true, Ordering::Relaxed);
                            }
                            ui.spinner();
                            ctx.request_repaint();
                        } else {
                            if ui.add_enabled(has_src, egui::Button::new("Recalculer")).clicked() {
                                self.last_param_change = None;
                                self.start_compute(ctx);
                            }
                            // Debounce : déclencher avec délai si params changed
                            if params_changed && has_src {
                                self.last_param_change = Some(Instant::now());
                            }
                        }
                    });

                    // Barre de progression pour Voronoï / K-means (UX 12)
                    if is_computing || is_cancelling {
                        let (cur, tot) = *self.progress.lock().unwrap_or_else(|e| e.into_inner());
                        if tot > 0 {
                            let frac = cur as f32 / tot as f32;
                            ui.add(egui::ProgressBar::new(frac).text(format!("{cur}/{tot}")));
                        }
                    }

                    ui.separator();

                    // Mode d'affichage
                    ui.label("Affichage");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.view_mode, ViewMode::Side, "Côte à côte");
                        ui.selectable_value(&mut self.view_mode, ViewMode::ResultOnly, "Résultat");
                        ui.selectable_value(&mut self.view_mode, ViewMode::SourceOnly, "Source");
                        ui.selectable_value(&mut self.view_mode, ViewMode::DensityMap, "Density");
                    });

                    ui.separator();

                    // Sauvegarder PNG
                    let has_result = self.result.lock().unwrap_or_else(|e| e.into_inner()).is_some();
                    if ui
                        .add_enabled(has_result && !is_computing, egui::Button::new("Sauvegarder PNG…"))
                        .clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("PNG", &["png"])
                            .add_filter("JPEG", &["jpg", "jpeg"])
                            .save_file()
                    {
                        self.save_result(path);
                    }

                    // Sauvegarder SVG
                    let has_dots = self.last_dots.lock().unwrap_or_else(|e| e.into_inner()).is_some();
                    if ui
                        .add_enabled((has_src || has_dots) && !is_computing, egui::Button::new("Sauvegarder SVG…"))
                        .clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter("SVG", &["svg"])
                            .save_file()
                    {
                        self.save_svg(path);
                    }

                    ui.separator();

                    // Statut + temps de calcul
                    if let Some(ms) = self.last_compute_ms {
                        ui.small(format!("Dernier calcul : {}", format_duration(ms)));
                    }
                    ui.label(&self.status);
                });
            });

        // ── Zone centrale ─────────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();

            // Lazy build textures
            if let Some(src) = &self.src_rgb
                && self.src_texture.is_none()
            {
                self.src_texture = Some(ctx.load_texture(
                    "src",
                    rgb_to_color_image(src),
                    TextureOptions::default(),
                ));
            }
            {
                let guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
                if self.result_texture.is_none()
                    && let Some(img) = guard.as_ref()
                {
                    self.result_texture = Some(ctx.load_texture(
                        "result",
                        rgb_to_color_image(img),
                        TextureOptions::default(),
                    ));
                }
            }
            if let Some(density) = &self.density_image
                && self.density_texture.is_none()
            {
                self.density_texture = Some(ctx.load_texture(
                    "density",
                    gray_to_color_image(density),
                    TextureOptions::default(),
                ));
            }

            // UX 13 : zoom/pan via ScrollArea
            // U3: use zoom_fit flag instead of magic 0.0 value
            let zoom = if self.zoom_fit { 0.0 } else { self.zoom };

            match self.view_mode {
                ViewMode::Side => {
                    let panel_w = available.x / 2.0 - 4.0;
                    let panel_h = available.y;
                    ui.horizontal(|ui| {
                        show_panel_zoomable(ui, "Source", &self.src_texture, panel_w, panel_h, false, zoom);
                        ui.separator();
                        show_panel_zoomable(ui, "Résultat", &self.result_texture, panel_w, panel_h,
                            computing, zoom);
                    });
                }
                ViewMode::ResultOnly => {
                    show_panel_zoomable(ui, "Résultat", &self.result_texture,
                        available.x, available.y,
                        computing, zoom);
                }
                ViewMode::SourceOnly => {
                    show_panel_zoomable(ui, "Source", &self.src_texture,
                        available.x, available.y, false, zoom);
                }
                ViewMode::DensityMap => {
                    show_panel_zoomable(ui, "Density map", &self.density_texture,
                        available.x, available.y, false, zoom);
                }
            }
        });
    }
}

// ─── Sélecteur de forme ───────────────────────────────────────────────────────

/// Affiche les contrôles de sélection de forme. Retourne true si la forme a changé.
fn show_shape_selector(ui: &mut egui::Ui, shape: &mut DotShape) -> bool {
    let mut changed = false;

    ui.horizontal(|ui| {
        for (label, variant) in [
            ("Cercle", DotShape::Circle),
            ("Carré", DotShape::Square),
        ] {
            let selected = std::mem::discriminant(shape) == std::mem::discriminant(&variant);
            if ui.selectable_label(selected, label).clicked() && !selected {
                *shape = variant;
                changed = true;
            }
        }
    });
    ui.horizontal(|ui| {
        let is_ellipse = matches!(shape, DotShape::Ellipse { .. });
        if ui.selectable_label(is_ellipse, "Ellipse").clicked() && !is_ellipse {
            *shape = DotShape::Ellipse { aspect: 1.5, angle_deg: 0.0 };
            changed = true;
        }
        let is_poly = matches!(shape, DotShape::RegularPolygon { .. });
        if ui.selectable_label(is_poly, "Polygone").clicked() && !is_poly {
            *shape = DotShape::RegularPolygon { sides: 6 };
            changed = true;
        }
    });

    // Paramètres de forme secondaires
    match shape {
        DotShape::Ellipse { aspect, angle_deg } => {
            ui.horizontal(|ui| {
                ui.label("Aspect");
                changed |= ui.add(egui::Slider::new(aspect, 0.2..=5.0).step_by(0.05)).changed();
            });
            ui.horizontal(|ui| {
                ui.label("Angle (°)");
                changed |= ui.add(egui::Slider::new(angle_deg, -180.0..=180.0).step_by(1.0)).changed();
            });
        }
        DotShape::RegularPolygon { sides } => {
            ui.horizontal(|ui| {
                ui.label("Côtés");
                let mut s = *sides as u32;
                if ui.add(egui::Slider::new(&mut s, 3..=12)).changed() {
                    *sides = s as u8;
                    changed = true;
                }
            });
        }
        _ => {}
    }

    changed
}

// ─── Affichage image avec zoom/pan ────────────────────────────────────────────

/// Affiche une image dans un panel avec zoom et scrollbar (UX 13).
/// zoom=0.0 signifie "fit" (comportement original).
fn show_panel_zoomable(
    ui: &mut egui::Ui,
    label: &str,
    texture: &Option<TextureHandle>,
    max_w: f32,
    max_h: f32,
    spinning: bool,
    zoom: f32,
) {
    ui.allocate_ui(egui::vec2(max_w, max_h), |ui| {
        ui.vertical(|ui| {
            ui.label(label);
            let remaining_h = ui.available_height();
            let remaining_w = ui.available_width();
            if let Some(tex) = texture {
                let (tw, th) = (tex.size()[0] as f32, tex.size()[1] as f32);
                let (img_w, img_h) = if zoom <= 0.0 {
                    // fit : respecte la largeur ET la hauteur restante
                    let scale = (remaining_w / tw).min(remaining_h / th);
                    (tw * scale, th * scale)
                } else {
                    (tw * zoom, th * zoom)
                };

                egui::ScrollArea::both()
                    .max_width(remaining_w)
                    .max_height(remaining_h)
                    .show(ui, |ui| {
                        ui.image((tex.id(), egui::vec2(img_w, img_h)));
                    });
            } else if spinning {
                ui.spinner();
            } else {
                ui.label("Aucune image");
            }
        });
    });
}
