//! Rendu egui : panneau de contrôle, zone centrale, raccourcis clavier.

use eframe::egui;
use egui::TextureOptions;
use pointimg::filter::{Algorithm, DotShape, HalftoneMode, Screening};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::convert::{gray_to_color_image, rgb_to_color_image, rgba_to_color_image_checker};
use super::format::{format_duration, format_memory};
use super::{App, ViewMode};
use pointimg::filter;

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        self.poll_worker_results();
        self.handle_shortcuts(&ctx);
        self.handle_drag_and_drop(&ctx);
        self.handle_cancel_and_debounce(&ctx);

        let computing = self.computing.load(Ordering::Relaxed);
        if !computing {
            // Calculer le temps si on vient de terminer
            if let Some(start) = self.compute_start.take() {
                let ms = start.elapsed().as_millis() as u64;
                self.last_compute_ms = Some(ms);
                self.status = format!("Terminé en {}.", format_duration(ms));
                // Commit undo-history : à chaque calcul terminé avec
                // params réellement différents depuis le dernier commit, on
                // pousse. `pending_commit` est mis à true quand params_changed.
                if self.pending_commit {
                    self.commit_history();
                    self.pending_commit = false;
                }
            }
        }

        // ── Erreur du thread ──────────────────────────────────────────────────
        if let Some(err) = self
            .compute_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            self.status = err;
        }

        self.show_control_panel(ui, &ctx, computing);
        self.show_central_panel(ui, &ctx, computing);
    }
}

impl App {
    /// Récupère les résultats publiés par les workers (density map).
    fn poll_worker_results(&mut self) {
        if let Some(density) = self
            .density_result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            self.density_image = Some(density);
            self.density_texture = None;
        }
    }

    /// Raccourcis clavier : ouvrir, sauver, recalculer, undo/redo.
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        let (kb_open, kb_save, kb_recalc, kb_undo, kb_redo) = ctx.input(|i| {
            let open = i.modifiers.command && i.key_pressed(egui::Key::O);
            let save = i.modifiers.command && i.key_pressed(egui::Key::S);
            let recalc = i.key_pressed(egui::Key::Space) && !i.modifiers.command;
            // Undo : Ctrl+Z (sans Shift). Redo : Ctrl+Y ou Ctrl+Shift+Z.
            let undo = i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z);
            let redo = i.modifiers.command
                && (i.key_pressed(egui::Key::Y)
                    || (i.modifiers.shift && i.key_pressed(egui::Key::Z)));
            (open, save, recalc, undo, redo)
        });

        if kb_open
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
                .pick_file()
        {
            self.load_image(path, ctx);
        }
        if kb_save {
            let has_result = self
                .result
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some();
            if has_result
                && !self.computing.load(Ordering::Relaxed)
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

        // ── Undo / Redo (Ctrl+Z / Ctrl+Y) ──────────────────────────────────
        // On n'agit que si l'utilisateur n'est pas en train de taper dans un champ.
        let wants_text = ctx.input(|i| {
            i.pointer.any_down()
                || i.raw
                    .events
                    .iter()
                    .any(|e| matches!(e, egui::Event::PointerMoved(_)))
        });
        if kb_undo && !wants_text {
            self.undo(ctx);
        }
        if kb_redo && !wants_text {
            self.redo(ctx);
        }
    }

    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        let dropped_path = ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                Some(i.raw.dropped_files[0].path().to_path_buf())
            } else {
                None
            }
        });
        if let Some(path) = dropped_path {
            self.load_image(path, ctx);
        }
    }

    /// Relance après annulation + debounce des changements de paramètres.
    fn handle_cancel_and_debounce(&mut self, ctx: &egui::Context) {
        if self.cancel.load(Ordering::Relaxed) && !self.computing.load(Ordering::Relaxed) {
            self.cancel.store(false, Ordering::Relaxed);
            self.start_compute(ctx);
        }

        if let Some(t) = self.last_param_change {
            if t.elapsed() >= Duration::from_millis(300) && self.src_rgb.is_some() {
                self.last_param_change = None;
                self.trigger_compute(ctx);
            } else {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
        }
    }

    // ─── Panneau de contrôle ──────────────────────────────────────────────────

    fn show_control_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, computing: bool) {
        egui::Panel::left("controls")
            .resizable(true)
            .min_size(290.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("pointimg");
                    ui.separator();

                    let mut params_changed = self.show_file_section(ui, ctx);
                    ui.separator();
                    params_changed |= self.show_algorithm_section(ui, ctx);
                    ui.separator();
                    params_changed |= self.show_shape_and_palette_section(ui);
                    ui.separator();
                    params_changed |= self.show_bg_section(ui, ctx);
                    params_changed |= self.show_gamma_and_zoom_section(ui);
                    ui.separator();
                    if self.params.algorithm == Algorithm::Halftone {
                        params_changed |= self.show_halftone_section(ui);
                    }
                    self.show_compute_buttons(ui, ctx, computing, &mut params_changed);
                    ui.separator();
                    self.show_view_mode_section(ui);
                    ui.separator();
                    self.show_export_buttons(ui, computing);
                    ui.separator();
                    self.show_status_section(ui);
                });
            });
    }

    /// Section fichiers/presets. Retourne true si un changement nécessite un compute.
    fn show_file_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> bool {
        if ui.button("Ouvrir une image…").clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
                .pick_file()
        {
            self.load_image(path, ctx);
        }
        ui.small("(ou glisser-déposer une image dans la fenêtre)");
        ui.horizontal(|ui| {
            if ui.button("Charger preset").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Preset TOML", &["toml"])
                    .pick_file()
            {
                self.load_preset(path, ctx);
            }
            if ui.button("Sauver preset").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Preset TOML", &["toml"])
                    .save_file()
            {
                self.save_preset(path);
            }
        });
        false
    }

    /// Section algorithme + paramètres de placement. Retourne true si changement.
    fn show_algorithm_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> bool {
        ui.label("Algorithme");

        let mut algo_changed = false;
        egui::Grid::new("algo_grid").num_columns(2).show(ui, |ui| {
            for (label, algo) in [
                ("Grille", Algorithm::Grid),
                ("K-means", Algorithm::Kmeans),
                ("Voronoi (Lloyd)", Algorithm::Voronoi),
                ("Quadtree", Algorithm::Quadtree),
                ("Halftone (rosette)", Algorithm::Halftone),
            ] {
                if ui
                    .selectable_label(self.params.algorithm == algo, label)
                    .clicked()
                {
                    self.params.algorithm = algo;
                    // Si on active Halftone et que le mode interne est Off,
                    // on force Cmyk par défaut.
                    if algo == Algorithm::Halftone && self.params.halftone == HalftoneMode::Off {
                        self.params.halftone = HalftoneMode::Cmyk {
                            angles: [15.0, 75.0, 0.0, 45.0],
                        };
                    }
                    algo_changed = true;
                }
                ui.end_row();
            }
        });

        let mut params_changed = algo_changed;
        let is_halftone = self.params.algorithm == Algorithm::Halftone;
        if self.params.max_radius_ratio < self.params.min_radius_ratio {
            self.params.max_radius_ratio = self.params.min_radius_ratio;
            params_changed = true;
        }
        // En mode Halftone, les sliders de placement (variance, rayons,
        // boost, cols, num_points, iterations, grid_angle) sont inutiles
        // car le pipeline les ignore. On les grise pour clarté.

        ui.label("Sensibilite variance");
        let vs_changed = ui
            .add_enabled(
                !is_halftone,
                egui::Slider::new(&mut self.params.variance_sensitivity, 0.0..=1.0).step_by(0.01),
            )
            .changed();
        if vs_changed {
            self.start_density_compute(ctx);
        }
        params_changed |= vs_changed;

        ui.label("Rayons des points (fraction image)");
        let mut min_radius = self.params.min_radius_ratio;
        let mut max_radius = self.params.max_radius_ratio.max(min_radius);
        let radius_limit = 0.3_f32.max(max_radius).min(1.0);
        let mut radius_changed = false;
        ui.horizontal(|ui| {
            ui.label("Min");
            radius_changed |= ui
                .add_enabled(
                    !is_halftone,
                    egui::Slider::new(&mut min_radius, 0.001..=max_radius)
                        .step_by(0.001)
                        .show_value(true),
                )
                .changed();
            ui.label("Max");
            radius_changed |= ui
                .add_enabled(
                    !is_halftone,
                    egui::Slider::new(&mut max_radius, min_radius..=radius_limit)
                        .step_by(0.005)
                        .show_value(true),
                )
                .changed();
        });
        if radius_changed {
            // Each slider is constrained by the other one, and the
            // final clamp also protects values loaded from presets.
            self.params.min_radius_ratio = min_radius.min(max_radius);
            self.params.max_radius_ratio = max_radius.max(min_radius);
        }
        params_changed |= radius_changed;

        ui.label("Boost zones uniformes (×max)");
        params_changed |= ui
            .add_enabled(
                !is_halftone,
                egui::Slider::new(&mut self.params.max_boost, 1.0..=5.0).step_by(0.1),
            )
            .changed();

        match self.params.algorithm {
            Algorithm::Grid => {
                ui.label("Colonnes");
                params_changed |= ui
                    .add(egui::Slider::new(&mut self.params.cols, 10..=300))
                    .changed();
                ui.label("Angle grille (°)");
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.params.grid_angle_deg, -90.0..=90.0)
                            .step_by(1.0),
                    )
                    .changed();
            }
            Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree => {
                ui.label("Nombre de points");
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.params.num_points, 50..=5000).logarithmic(true),
                    )
                    .changed();

                if matches!(
                    self.params.algorithm,
                    Algorithm::Kmeans | Algorithm::Voronoi
                ) {
                    ui.label("Itérations");
                    params_changed |= ui
                        .add(egui::Slider::new(&mut self.params.iterations, 1..=30))
                        .changed();
                }
            }
            Algorithm::Halftone => {
                // Pas de sous-params placement. Les sous-params
                // halftone (mode, screening, fréquence, rayons) sont
                // affichés plus bas dans la section dédiée.
                ui.small("(les paramètres de placement ci-dessus sont ignorés)");
            }
        }
        params_changed
    }

    /// Section forme des dots, palette, dithering et seed. Retourne true si changement.
    fn show_shape_and_palette_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut params_changed = false;

        ui.label("Forme des points");
        params_changed |= show_shape_selector(ui, &mut self.params.dot_shape);
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
        // Dithering Floyd-Steinberg : ne s'applique que si palette est activée.
        ui.horizontal(|ui| {
            ui.label("Dithering FS");
            let enabled = self.params.palette_size.is_some();
            let mut dither = self.params.dithering;
            if ui
                .add_enabled(enabled, egui::Checkbox::new(&mut dither, ""))
                .changed()
            {
                self.params.dithering = dither;
                params_changed = true;
            }
            if !enabled {
                ui.small("(active la palette pour utiliser le dithering)");
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
                if ui
                    .add(egui::DragValue::new(&mut seed_i64).speed(1.0))
                    .changed()
                {
                    *s = seed_i64.unsigned_abs();
                    params_changed = true;
                }
            }
        });
        params_changed
    }

    /// Section couleur de fond (le changement invalide la density → refresh).
    /// Retourne true si un recompute du filtre est nécessaire.
    fn show_bg_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> bool {
        let mut params_changed = false;
        ui.label("Couleur de fond");
        ui.horizontal(|ui| {
            let old_color = self.params.bg_color;
            let mut color = egui::Color32::from_rgb(old_color[0], old_color[1], old_color[2]);
            if egui::color_picker::color_edit_button_srgba(
                ui,
                &mut color,
                egui::color_picker::Alpha::Opaque,
            )
            .changed()
            {
                self.params.bg_color = [color.r(), color.g(), color.b()];
                self.params.transparent = false;
                self.refresh_src_rgb(ctx);
                params_changed = true;
            }
            // Raccourcis Blanc / Noir
            if ui.small_button("Blanc").clicked() {
                self.params.bg_color = [255, 255, 255];
                self.params.transparent = false;
                self.refresh_src_rgb(ctx);
                params_changed = true;
            }
            if ui.small_button("Noir").clicked() {
                self.params.bg_color = [0, 0, 0];
                self.params.transparent = false;
                self.refresh_src_rgb(ctx);
                params_changed = true;
            }
        });
        // Mode transparent : produit RGBA à la sauvegarde. La preview
        // GUI reste sur fond coloré (le panneau résultat affiche du RGB).
        params_changed |= ui
            .checkbox(&mut self.params.transparent, "Fond transparent (RGBA)")
            .changed();
        params_changed
    }

    /// Section gamma + zoom. Retourne true si changement (gamma uniquement).
    fn show_gamma_and_zoom_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut params_changed = false;
        ui.horizontal(|ui| {
            ui.label("Correction gamma");
            params_changed |= ui
                .checkbox(&mut self.params.gamma_correct, "espace linéaire")
                .changed();
        });
        ui.small("(moyennes perceptuelles, évite les mi-tons trop sombres)");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Zoom");
            if ui
                .add(egui::Slider::new(&mut self.zoom, 0.1..=4.0).step_by(0.1))
                .changed()
            {
                self.zoom_fit = false; // Manual zoom disables fit.
            }
            if ui.small_button("1:1").clicked() {
                self.zoom = 1.0;
                self.zoom_fit = false;
            }
            if ui.small_button("Fit").clicked() {
                self.zoom_fit = true; // Re-enable fit mode.
            }
        });
        params_changed
    }

    /// Sous-contrôles du mode halftone. Retourne true si changement.
    fn show_halftone_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut params_changed = false;
        ui.label("Mode halftone");
        ui.horizontal(|ui| {
            if ui
                .selectable_label(
                    matches!(self.params.halftone, HalftoneMode::Cmyk { .. }),
                    "CMYK",
                )
                .clicked()
            {
                self.params.halftone = HalftoneMode::Cmyk {
                    angles: [15.0, 75.0, 0.0, 45.0],
                };
                params_changed = true;
            }
            if ui
                .selectable_label(
                    matches!(self.params.halftone, HalftoneMode::Dominant { .. }),
                    "Dominant",
                )
                .clicked()
            {
                self.params.halftone = HalftoneMode::Dominant {
                    n: 5,
                    base_angle_deg: 15.0,
                };
                params_changed = true;
            }
        });
        // Sous-params dépendant du mode.
        match &mut self.params.halftone {
            HalftoneMode::Dominant { n, base_angle_deg } => {
                ui.horizontal(|ui| {
                    ui.label("Canaux");
                    let mut nv = *n as i32;
                    if ui
                        .add(egui::Slider::new(&mut nv, 2..=12).step_by(1.0))
                        .changed()
                    {
                        *n = nv as usize;
                        params_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Angle base (°)");
                    params_changed |= ui
                        .add(egui::Slider::new(base_angle_deg, 0.0..=180.0).step_by(1.0))
                        .changed();
                });
            }
            HalftoneMode::Cmyk { angles } => {
                ui.small(format!(
                    "Angles C/M/Y/K : {:.0}° / {:.0}° / {:.0}° / {:.0}°",
                    angles[0], angles[1], angles[2], angles[3]
                ));
            }
            HalftoneMode::Off => {
                ui.small("(sélectionnez CMYK ou Dominant)");
            }
        }
        // Screening AM/FM
        ui.horizontal(|ui| {
            ui.label("Screening");
            if ui
                .selectable_label(self.params.screening == Screening::Am, "AM (grille)")
                .clicked()
            {
                self.params.screening = Screening::Am;
                params_changed = true;
            }
            if ui
                .selectable_label(self.params.screening == Screening::Fm, "FM (blue noise)")
                .clicked()
            {
                self.params.screening = Screening::Fm;
                params_changed = true;
            }
        });
        // Fréquence de trame
        ui.horizontal(|ui| {
            ui.label("Fréquence trame");
            params_changed |= ui
                .add(
                    egui::Slider::new(&mut self.params.halftone_frequency, 20.0..=200.0)
                        .step_by(5.0),
                )
                .changed();
        });
        // Rayons halftone
        ui.horizontal(|ui| {
            ui.label("Rayon min (frac. min)");
            params_changed |= ui
                .add(
                    egui::Slider::new(&mut self.params.halftone_min_radius_ratio, 0.001..=0.05)
                        .step_by(0.001),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Rayon max (frac. step)");
            params_changed |= ui
                .add(
                    egui::Slider::new(&mut self.params.halftone_max_dot_ratio, 0.3..=1.5)
                        .step_by(0.05),
                )
                .changed();
        });
        params_changed
    }

    /// Boutons Annuler/Recalculer + barre de progression + armement du debounce.
    fn show_compute_buttons(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        computing: bool,
        params_changed: &mut bool,
    ) {
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
                if ui
                    .add_enabled(has_src, egui::Button::new("Recalculer"))
                    .clicked()
                {
                    self.last_param_change = None;
                    self.start_compute(ctx);
                }
                // Debounce : déclencher avec délai si params changed
                if *params_changed && has_src {
                    self.last_param_change = Some(Instant::now());
                    self.pending_commit = true;
                }
            }
        });
        *params_changed = false;

        // Progress bar for Voronoi / K-means.
        if is_computing || is_cancelling {
            let (cur, tot) = *self.progress.lock().unwrap_or_else(|e| e.into_inner());
            if tot > 0 {
                let frac = cur as f32 / tot as f32;
                ui.add(egui::ProgressBar::new(frac).text(format!("{cur}/{tot}")));
            }
        }
    }

    fn show_view_mode_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Affichage");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view_mode, ViewMode::Side, "Côte à côte");
            ui.selectable_value(&mut self.view_mode, ViewMode::ResultOnly, "Résultat");
            ui.selectable_value(&mut self.view_mode, ViewMode::SourceOnly, "Source");
            ui.selectable_value(&mut self.view_mode, ViewMode::DensityMap, "Density");
        });
    }

    fn show_export_buttons(&mut self, ui: &mut egui::Ui, computing: bool) {
        let has_src = self.src_rgb.is_some();
        // Sauvegarder PNG
        let has_result = self
            .result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        if ui
            .add_enabled(
                has_result && !computing,
                egui::Button::new("Sauvegarder image…"),
            )
            .clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .add_filter("JPEG", &["jpg", "jpeg"])
                .add_filter("WebP", &["webp"])
                .add_filter("BMP", &["bmp"])
                .add_filter("TIFF", &["tif", "tiff"])
                .save_file()
        {
            self.save_result(path);
        }

        // Sauvegarder SVG
        let has_dots = self
            .last_dots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        if ui
            .add_enabled(
                (has_src || has_dots) && !computing,
                egui::Button::new("Sauvegarder SVG…"),
            )
            .clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("SVG", &["svg"])
                .save_file()
        {
            self.save_svg(path);
        }
    }

    fn show_status_section(&mut self, ui: &mut egui::Ui) {
        // Statut + temps de calcul
        if let Some(src) = &self.src_rgb {
            ui.small(format!(
                "Mémoire estimée : {}",
                format_memory(filter::estimate_memory_bytes(src.width(), src.height()))
            ));
        }
        if let Some(ms) = self.last_compute_ms {
            ui.small(format!("Dernier calcul : {}", format_duration(ms)));
        }
        ui.collapsing("Historique des réglages", |ui| {
            if self.history.is_empty() && self.future.is_empty() {
                ui.small("Aucune modification enregistrée.");
            } else {
                for entry in self.history.iter().rev().take(8) {
                    ui.small(format!("✓ {}", entry.label));
                }
                for entry in self.future.iter().rev().take(8) {
                    ui.small(format!("↶ {}", entry.label));
                }
            }
        });
        ui.label(&self.status);
    }

    // ─── Zone centrale ────────────────────────────────────────────────────────

    fn show_central_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, computing: bool) {
        egui::CentralPanel::default().show(ui, |ui| {
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
                let revision = self.result_revision.load(Ordering::Acquire);
                if self.result_texture_revision != revision
                    && let Some(img) = guard.as_ref()
                {
                    self.result_texture = Some(ctx.load_texture(
                        "result",
                        rgba_to_color_image_checker(img),
                        TextureOptions::default(),
                    ));
                    self.result_texture_revision = revision;
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

            // Zoom and pan via ScrollArea.
            let zoom = if self.zoom_fit { 0.0 } else { self.zoom };

            match self.view_mode {
                ViewMode::Side => {
                    let panel_w = available.x / 2.0 - 4.0;
                    let panel_h = available.y;
                    ui.horizontal(|ui| {
                        show_panel_zoomable(
                            ui,
                            "Source",
                            &self.src_texture,
                            panel_w,
                            panel_h,
                            false,
                            zoom,
                        );
                        ui.separator();
                        show_panel_zoomable(
                            ui,
                            "Résultat",
                            &self.result_texture,
                            panel_w,
                            panel_h,
                            computing,
                            zoom,
                        );
                    });
                }
                ViewMode::ResultOnly => {
                    show_panel_zoomable(
                        ui,
                        "Résultat",
                        &self.result_texture,
                        available.x,
                        available.y,
                        computing,
                        zoom,
                    );
                }
                ViewMode::SourceOnly => {
                    show_panel_zoomable(
                        ui,
                        "Source",
                        &self.src_texture,
                        available.x,
                        available.y,
                        false,
                        zoom,
                    );
                }
                ViewMode::DensityMap => {
                    show_panel_zoomable(
                        ui,
                        "Density map",
                        &self.density_texture,
                        available.x,
                        available.y,
                        false,
                        zoom,
                    );
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
        for (label, variant) in [("Cercle", DotShape::Circle), ("Carré", DotShape::Square)] {
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
            *shape = DotShape::Ellipse {
                aspect: 1.5,
                angle_deg: 0.0,
            };
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
                changed |= ui
                    .add(egui::Slider::new(aspect, 0.2..=5.0).step_by(0.05))
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("Angle (°)");
                changed |= ui
                    .add(egui::Slider::new(angle_deg, -180.0..=180.0).step_by(1.0))
                    .changed();
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

/// Displays an image in a panel with zoom and scrolling.
/// zoom=0.0 signifie "fit" (comportement original).
fn show_panel_zoomable(
    ui: &mut egui::Ui,
    label: &str,
    texture: &Option<egui::TextureHandle>,
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
                    // Salt unique par panneau : en vue côte-à-côte, les deux
                    // ScrollAreas identiques recevaient le même ID egui.
                    .id_salt(label)
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
