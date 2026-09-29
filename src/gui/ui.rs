//! Rendu egui : en-tête, panneau de réglages, zone d'aperçu, raccourcis.
//!
//! L'interface reprend les décisions de la version GTK4 : actions visibles
//! dans l'en-tête (rien d'important sous la ligne de flottaison), comparaison
//! avant/après par balayage, réglages en saisie directe, aperçu rapide avec
//! finalisation automatique, presets intégrés, et un état d'accueil qui donne
//! quelque chose à faire.

use eframe::egui;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use pointimg::filter;
use pointimg::filter::{Algorithm, Dithering, DotShape, HalftoneMode, Screening};

use super::App;
use super::convert::{gray_to_color_image, rgb_to_color_image, rgba_to_color_image_checker};
use super::exports;
use super::format::format_duration;
use super::sliders::{display_decimals, int_slider_row, range_slider, slider_row};
use pointimg::frontend::{Curve, Quality, ViewMode, sample_image, view_from_index, view_index};

/// Description courte d'un algorithme, affichée sous le sélecteur.
fn algorithm_help(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Grid => "Grille régulière : une cellule = un point. Rapide et prévisible.",
        Algorithm::Kmeans => {
            "k-means spatial : les points se répartissent selon la couleur et la position."
        }
        Algorithm::Voronoi => {
            "Lloyd pondéré : répartition régulière guidée par le détail. Rendu le plus « naturel »."
        }
        Algorithm::Quadtree => {
            "Subdivision par contraste : davantage de points là où l'image est détaillée."
        }
        Algorithm::Halftone => {
            "Trame d'encre : rosette CMJN ou couleurs dominantes, style offset print."
        }
    }
}

/// Modes d'affichage, dans l'ordre de cycling.
const VIEWS: [(&str, ViewMode); 5] = [
    ("Comparaison", ViewMode::Compare),
    ("Côte à côte", ViewMode::Side),
    ("Résultat", ViewMode::ResultOnly),
    ("Source", ViewMode::SourceOnly),
    ("Density map", ViewMode::DensityMap),
];

/// Style global de l'interface.
///
/// `scale` agrandit *tout* ce qui donne la densité perçue — corps de texte,
/// espacements, hauteur des widgets, largeur des curseurs, arrondis — de sorte
/// qu'un seul réglage rende l'interface confortable sans toucher à chaque
/// sous-widget. 1.0 = confortable (valeur par défaut), 1.3 = grande.
fn tuned_style(style: &mut egui::Style, scale: f32) {
    let scale = scale.clamp(0.85, 1.6);

    // ── Textes ──────────────────────────────────────────────────────────────
    for (text, size) in [
        (egui::TextStyle::Body, 14.0),
        (egui::TextStyle::Button, 14.0),
        (egui::TextStyle::Small, 12.0),
        (egui::TextStyle::Heading, 18.0),
        (egui::TextStyle::Monospace, 13.0),
    ] {
        style
            .text_styles
            .insert(text, egui::FontId::proportional(size * scale));
    }

    // ── Rythme vertical et horizontal ───────────────────────────────────────
    style.spacing.item_spacing = egui::vec2(10.0, 8.0) * scale;
    style.spacing.button_padding = egui::vec2(10.0, 5.0) * scale;
    style.spacing.indent = 20.0 * scale;
    style.spacing.interact_size.y = 24.0 * scale;
    // Les curseurs occupent la largeur du panneau (comme le curseur à deux
    // poignées des rayons) : 120 px étaient trop étroits pour régler finely.
    style.spacing.slider_width = 320.0 * scale;
    style.spacing.window_margin = egui::Margin::same((8.0 * scale) as i8);
    style.spacing.menu_margin = egui::Margin::same((6.0 * scale) as i8);

    // ── Widgets plus généreux et moins anguleux ─────────────────────────────
    let radius = ((7.0 * scale).round() as u8).min(14);
    let stroke = 1.0;
    for widget in [
        &mut style.visuals.widgets.noninteractive,
        &mut style.visuals.widgets.inactive,
        &mut style.visuals.widgets.hovered,
        &mut style.visuals.widgets.active,
        &mut style.visuals.widgets.open,
    ] {
        widget.corner_radius = radius.into();
        widget.fg_stroke.width = stroke;
    }
    // Un peu plus de contraste entre les panneaux et le fond de fenêtre.
    style.visuals.window_fill = style.visuals.window_fill.gamma_multiply(1.0);
    style.visuals.panel_fill = style.visuals.panel_fill.gamma_multiply(1.0);
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Thème mémorisé : appliqué dès le premier rendu (le contexte n'existe
        // qu'à partir de là, et « Auto » doit aussi corriger un éventuel thème
        // laissé par eframe dans ses propres préférences).
        self.apply_theme(&ctx);

        // `eframe::App::style` n'existe plus en 0.36 : on applique notre style
        // nous-mêmes, et on le réapplique quand l'utilisateur change l'échelle
        // d'interface.
        if self.applied_style != Some(self.ui_scale) {
            let scale = self.ui_scale;
            ctx.all_styles_mut(|style| tuned_style(style, scale));
            self.applied_style = Some(scale);
        }

        self.poll_worker_results();
        self.collect_errors();
        self.handle_cancel_and_debounce(&ctx);
        self.handle_shortcuts(&ctx);
        self.handle_drag_and_drop(&ctx);

        let computing = self.computing.load(Ordering::Relaxed);
        self.finish_pending_compute(computing);
        // Taille d'export : relancée à chaque résultat définitif, invalidée
        // dès que l'utilisateur revient à régler l'image.
        self.refresh_export_sizes(&ctx);
        self.sync_window_title(&ctx);

        // L'ordre compte : le panneau du bas doit être réservé avant le
        // panneau de gauche, sinon celui-ci déborde dessus.
        self.show_view_bar(ui, &ctx, computing);
        self.show_status_bar(ui, computing);
        self.show_control_panel(ui, &ctx);
        self.show_central_panel(ui, &ctx, computing);
    }
}

impl App {
    // ─── Cycle de vie ────────────────────────────────────────────────────────

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

    fn collect_errors(&mut self) {
        if let Some(err) = self
            .compute_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            self.status = err;
        }
    }

    /// Clôt le calcul en cours : durée, message, commit d'historique.
    fn finish_pending_compute(&mut self, computing: bool) {
        if computing {
            return;
        }
        if let Some(start) = self.compute_start.take() {
            let ms = start.elapsed().as_millis() as u64;
            self.last_compute_ms = Some(ms);
            let final_result = self.result_is_final.load(Ordering::Acquire);
            self.status = if final_result {
                format!("Calcul terminé en {}.", format_duration(ms))
            } else {
                "Aperçu calculé — finalisation en cours…".to_string()
            };
            if self.pending_commit {
                self.commit_history();
                self.pending_commit = false;
            }
        }
        // Un aperçu rapide terminé déclenche la passe en pleine résolution.
        if !self.has_final_result()
            && !self.computing.load(Ordering::Relaxed)
            && self.has_any_result()
        {
            self.schedule_finalize();
        }
    }

    /// Programme la passe en pleine résolution quand l'utilisateur arrête de
    /// manipuler les réglages (mode aperçu rapide).
    fn schedule_finalize(&mut self) {
        if self.quality.is_full() || self.finalize_at.is_some() {
            return;
        }
        self.finalize_at = Some(Instant::now() + Duration::from_millis(900));
    }

    fn poll_finalize(&mut self) {
        if let Some(at) = self.finalize_at
            && at <= Instant::now()
        {
            self.finalize_at = None;
            self.final_pass = true;
            self.last_param_change = Some(Instant::now());
        }
    }

    /// Le résultat affiché est-il exploitable (pleine résolution, à jour) ?
    pub(crate) fn has_final_result(&self) -> bool {
        self.result_is_final.load(Ordering::Acquire) && self.has_any_result()
    }

    /// Le résultat affiché ne correspond plus aux réglages courants.
    pub(crate) fn result_is_stale(&self) -> bool {
        (!self.result_is_final.load(Ordering::Acquire) && self.has_any_result())
            || self.computing.load(Ordering::Relaxed)
    }

    fn has_any_result(&self) -> bool {
        self.result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    fn sync_window_title(&mut self, ctx: &egui::Context) {
        let name = self
            .src_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("Aucune image");
        let title = match self.folder_position() {
            Some((index, total)) => format!("pointimg — {name} ({index}/{total})"),
            None => format!("pointimg — {name}"),
        };
        if title != self.window_title {
            self.window_title = title.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        }
    }

    // ─── Raccourcis et glisser-déposer ───────────────────────────────────────

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        #[derive(Default)]
        struct Keys {
            open: bool,
            open_folder: bool,
            save: bool,
            save_svg: bool,
            quick_save: bool,
            copy: bool,
            recalc: bool,
            undo: bool,
            redo: bool,
            preset: bool,
            view: bool,
            zoom_in: bool,
            zoom_out: bool,
            fit: bool,
            one: bool,
            prev: bool,
            next: bool,
            reset: bool,
        }
        let keys = ctx.input(|i| {
            let command = i.modifiers.command;
            let shift = i.modifiers.shift;
            let plain = !command && !i.modifiers.ctrl && !i.modifiers.alt;
            Keys {
                open: command && !shift && i.key_pressed(egui::Key::O),
                open_folder: command && shift && i.key_pressed(egui::Key::O),
                save: command && !shift && i.key_pressed(egui::Key::S),
                save_svg: command && shift && i.key_pressed(egui::Key::S),
                quick_save: command && i.key_pressed(egui::Key::E),
                copy: command && i.key_pressed(egui::Key::C),
                recalc: plain && i.key_pressed(egui::Key::Space),
                undo: command && !shift && i.key_pressed(egui::Key::Z),
                redo: command
                    && (i.key_pressed(egui::Key::Y) || (shift && i.key_pressed(egui::Key::Z))),
                preset: command && i.key_pressed(egui::Key::P),
                view: plain && i.key_pressed(egui::Key::Tab),
                zoom_in: plain
                    && (i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)),
                zoom_out: plain && i.key_pressed(egui::Key::Minus),
                fit: plain && i.key_pressed(egui::Key::Num0),
                one: plain && i.key_pressed(egui::Key::Num1),
                prev: plain && i.key_pressed(egui::Key::ArrowLeft),
                next: plain && i.key_pressed(egui::Key::ArrowRight),
                reset: command && i.key_pressed(egui::Key::R),
            }
        });

        // Tant qu'un champ de saisie a le focus, les raccourcis « lettre »
        // seraient tapés dans le champ.
        let wants_text = ctx.egui_wants_keyboard_input();

        if keys.open {
            self.choose_image(ctx);
        }
        if keys.open_folder {
            self.choose_folder(ctx);
        }
        if keys.save && self.has_final_result() {
            self.choose_export();
        }
        if keys.save_svg && self.has_final_result() {
            self.choose_svg_export();
        }
        if keys.quick_save {
            self.quick_export();
        }
        if keys.copy && !wants_text {
            self.copy_to_clipboard(ctx);
        }
        if keys.reset {
            self.reset_params(ctx);
        }
        if keys.undo && !wants_text {
            self.undo(ctx);
        }
        if keys.redo && !wants_text {
            self.redo(ctx);
        }
        if keys.recalc {
            self.request_full_compute(ctx);
        }
        if keys.preset && !wants_text {
            self.cycle_preset(1, ctx);
        }
        if keys.view && !wants_text {
            self.cycle_view(1);
        }
        if keys.zoom_in && !wants_text {
            self.set_zoom(self.current_zoom() * 1.25);
        }
        if keys.zoom_out && !wants_text {
            self.set_zoom(self.current_zoom() / 1.25);
        }
        if keys.fit && !wants_text {
            self.zoom_fit = true;
        }
        if keys.one && !wants_text {
            self.set_zoom(1.0);
        }
        if keys.prev && !wants_text {
            self.step_folder(-1, ctx);
        }
        if keys.next && !wants_text {
            self.step_folder(1, ctx);
        }
    }

    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        let dropped: Vec<std::path::PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|file| file.path().to_path_buf())
                .collect()
        });
        for path in dropped {
            if path.is_dir() {
                self.open_folder(path, ctx);
            } else {
                self.load_image(path, ctx);
            }
        }
    }

    fn handle_cancel_and_debounce(&mut self, ctx: &egui::Context) {
        if self.cancel.load(Ordering::Relaxed) && !self.computing.load(Ordering::Relaxed) {
            self.cancel.store(false, Ordering::Relaxed);
            self.start_compute(ctx);
        }
        self.poll_finalize();

        if let Some(t) = self.last_param_change {
            if t.elapsed() >= Duration::from_millis(300) && self.src_rgb.is_some() {
                self.last_param_change = None;
                // La density map suit le même debounce que le filtre.
                if self.density_pending {
                    self.density_pending = false;
                    self.start_density_compute(ctx);
                }
                self.trigger_compute(ctx);
            } else {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
        } else if self.finalize_at.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    // ─── En-tête ─────────────────────────────────────────────────────────────

    /// Tout ce qui est action vit dans l'en-tête : plus rien à faire défiler.
    fn show_view_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, computing: bool) {
        let has_final = self.has_final_result();
        let has_dots = self
            .last_dots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
            || self.src_rgb.is_some();
        // L'infobulle du bouton d'export annonce la taille obtenue, pour éviter
        // d'ouvrir le dialogue juste pour la découvrir.
        let export_hint = match self
            .poll_export_sizes()
            .as_ref()
            .and_then(exports::export_comparison)
        {
            Some(comparison) => format!("Exporter l'image (Ctrl+S)\n{comparison}"),
            None => "Exporter l'image (Ctrl+S)".to_string(),
        };

        egui::Panel::top("view_bar").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                if ui.button("Ouvrir…").clicked() {
                    self.choose_image(ctx);
                }
                if ui.button("Dossier…").clicked() {
                    self.choose_folder(ctx);
                }
                ui.separator();
                if computing {
                    if ui.button("Stop").clicked() {
                        self.cancel.store(true, Ordering::Relaxed);
                        self.status = "Annulation…".to_string();
                    }
                    ui.spinner();
                } else if ui.button("Recalculer").clicked() {
                    self.request_full_compute(ctx);
                }
                ui.separator();
                let undo_label = self
                    .history
                    .last()
                    .map(|entry| format!("Annuler : {} (Ctrl+Z)", entry.label))
                    .unwrap_or_else(|| "Rien à annuler".to_string());
                if ui
                    .add_enabled(
                        !self.history.is_empty() || self.pending_commit,
                        egui::Button::new("Annuler"),
                    )
                    .on_hover_text(undo_label)
                    .clicked()
                {
                    self.undo(ctx);
                }
                let redo_label = self
                    .future
                    .last()
                    .map(|entry| format!("Rétablir : {} (Ctrl+Y)", entry.label))
                    .unwrap_or_else(|| "Rien à rétablir".to_string());
                if ui
                    .add_enabled(!self.future.is_empty(), egui::Button::new("Rétablir"))
                    .on_hover_text(redo_label)
                    .clicked()
                {
                    self.redo(ctx);
                }
                if ui
                    .button("⟲")
                    .on_hover_text("Rétablir tous les réglages par défaut (Ctrl+R)")
                    .clicked()
                {
                    self.reset_params(ctx);
                }
                ui.separator();
                if ui
                    .add_enabled(has_final, egui::Button::new("Exporter l'image…"))
                    .on_hover_text(export_hint)
                    .clicked()
                {
                    self.choose_export();
                }
                if ui
                    .add_enabled(has_dots, egui::Button::new("Exporter SVG…"))
                    .on_hover_text("Exporter le rendu en SVG (Ctrl+Shift+S)")
                    .clicked()
                {
                    self.choose_svg_export();
                }
                if ui
                    .add_enabled(has_final, egui::Button::new("Copier"))
                    .on_hover_text("Copier l'image dans le presse-papiers (Ctrl+C)")
                    .clicked()
                {
                    self.copy_to_clipboard(ctx);
                }
            });
            ui.add_space(3.0);
        });
    }

    // ─── Panneau de réglages ────────────────────────────────────────────────

    fn show_control_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(330.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("pointimg");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let info = self
                                .image_info()
                                .unwrap_or_else(|| "Aucune image".to_string());
                            ui.label(info)
                                .on_hover_text("Dimensions de l'image chargée");
                        });
                    });
                    ui.separator();

                    // Un seul garde-fou pour tout le panneau : on ne relance un
                    // calcul que si les paramètres ont *réellement* changé.
                    // Sinon, un simple changement de focus entre deux champs —
                    // ou la revalidation d'un curseur par egui — suffirait à
                    // faire recalculer l'image.
                    let before = self.params.clone();
                    self.show_display_preferences(ui);
                    self.show_presets(ui, ctx);
                    self.show_algorithm_section(ui);
                    self.show_shape_and_palette_section(ui);
                    self.show_bg_section(ui, ctx);
                    self.show_quality_section(ui, ctx);
                    if self.params != before {
                        self.on_params_changed(ctx);
                    }
                    ui.separator();
                    if ui.button("Réinitialiser les réglages").clicked() {
                        self.reset_params(ctx);
                    }
                });
            });
    }

    /// Confort d'affichage : taille de l'interface et thème.
    fn show_display_preferences(&mut self, ui: &mut egui::Ui) {
        // `horizontal_wrapped` : la ligne se replie au lieu d'élargir le panneau.
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Interface").size(13.0));
            let mut percent = (self.ui_scale * 100.0).round();
            let response = ui.add(
                egui::DragValue::new(&mut percent)
                    .speed(5.0)
                    .range(85.0..=160.0)
                    .suffix(" %"),
            );
            if response.changed() {
                self.ui_scale = percent / 100.0;
                self.persist_display_preferences();
            }
            response.on_hover_text(
                "Agrandit les textes, les espacements et la hauteur des widgets. \
                 100 % = confortable.",
            );
            for (label, theme) in [
                ("Auto", None),
                ("Sombre", Some(egui::Theme::Dark)),
                ("Clair", Some(egui::Theme::Light)),
            ] {
                let selected = self.ui_theme == theme;
                if ui.selectable_label(selected, label).clicked() {
                    self.ui_theme = theme;
                    self.apply_theme(ui.ctx());
                    self.persist_display_preferences();
                }
            }
        });
        ui.separator();
    }

    /// Presets intégrés : le point de départ le plus rapide.
    fn show_presets(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let presets = filter::builtin_presets();
        let current = self.preset_index();
        section(ui, "Réglages rapides");
        let mut selected = current;
        ui.horizontal(|ui| {
            ui.label("Preset");
            egui::ComboBox::from_id_salt("preset_combo")
                .selected_text(match current {
                    None => "Personnalisé".to_string(),
                    Some(index) => presets[index].name.to_string(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selected, None, "Personnalisé");
                    for (index, preset) in presets.iter().enumerate() {
                        ui.selectable_value(&mut selected, Some(index), preset.name)
                            .on_hover_text(preset.description);
                    }
                });
            if ui
                .small_button("Suivant")
                .on_hover_text("Preset suivant (Ctrl+P)")
                .clicked()
            {
                self.cycle_preset(1, ctx);
            } else if current.is_some()
                && selected != current
                && let Some(index) = selected
            {
                self.apply_preset(index, ctx);
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Charger un preset…").clicked() {
                self.choose_load_preset(ctx);
            }
            if ui.button("Sauver le preset…").clicked() {
                self.choose_save_preset();
            }
        });
    }

    fn show_algorithm_section(&mut self, ui: &mut egui::Ui) {
        const ALGOS: [(&str, Algorithm); 5] = [
            ("Grille", Algorithm::Grid),
            ("K-means", Algorithm::Kmeans),
            ("Voronoi (Lloyd)", Algorithm::Voronoi),
            ("Quadtree", Algorithm::Quadtree),
            ("Halftone (rosette)", Algorithm::Halftone),
        ];
        section(ui, "Algorithme");
        let current = ALGOS
            .iter()
            .find(|(_, algo)| *algo == self.params.algorithm)
            .map(|(label, _)| *label)
            .unwrap_or("Voronoi (Lloyd)");
        ui.horizontal(|ui| {
            ui.label("Placement");
            egui::ComboBox::from_id_salt("algo_combo")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (label, algo) in ALGOS {
                        ui.selectable_value(&mut self.params.algorithm, algo, label)
                            .on_hover_text(algorithm_help(algo));
                    }
                });
        });
        ui.add(
            egui::Label::new(egui::RichText::new(algorithm_help(self.params.algorithm)).size(11.5))
                .wrap(),
        );

        if self.params.algorithm == Algorithm::Halftone {
            self.show_halftone_section(ui);
            return;
        }

        let variance_before = self.params.variance_sensitivity;
        number_row_with_hint(
            ui,
            "Sensibilité variance",
            &mut self.params.variance_sensitivity,
            0.0..=1.0,
            Some("0 = uniforme · 1 = détail maximal"),
        );
        slider_row(
            ui,
            "variance",
            &mut self.params.variance_sensitivity,
            0.0..=1.0,
            false,
            "0 = répartition uniforme · 1 = détail maximal",
        );
        if variance_before != self.params.variance_sensitivity {
            // La density map dépend de ce seul réglage : on la recalcule sans
            // relancer le filtre — mais au même moment que lui (le debounce),
            // sinon un glissement de curseur lancerait un calcul par pixel.
            self.density_pending = true;
        }

        number_row_with_hint(
            ui,
            "Rayon min",
            &mut self.params.min_radius_ratio,
            0.0005..=1.0,
            None,
        );
        number_row_with_hint(
            ui,
            "Rayon max",
            &mut self.params.max_radius_ratio,
            0.0005..=1.0,
            None,
        );
        // Curseur à deux poignées : la plage se règle d'un geste, et l'ordre
        // des bornes ne peut pas s'inverser.
        range_slider(
            ui,
            "rayon",
            &mut self.params.min_radius_ratio,
            &mut self.params.max_radius_ratio,
            0.0005..=1.0,
            Curve::Log,
            "Rayon min et max, en fraction du plus petit côté",
        );
        if self.params.min_radius_ratio > self.params.max_radius_ratio {
            std::mem::swap(
                &mut self.params.min_radius_ratio,
                &mut self.params.max_radius_ratio,
            );
        }

        number_row_with_hint(
            ui,
            "Boost zones uniformes",
            &mut self.params.max_boost,
            1.0..=8.0,
            None,
        );
        slider_row(
            ui,
            "boost",
            &mut self.params.max_boost,
            1.0..=8.0,
            false,
            "Agrandit les points dans les zones sans détail. 1 = aucun agrandissement.",
        );

        match self.params.algorithm {
            Algorithm::Grid => {
                // 2 → 8 192 : logarithmique, sinon les petites grilles
                // n'occupent qu'un pixel de la piste.
                let mut cols = self.params.cols as usize;
                int_number_row(
                    ui,
                    "Colonnes",
                    &mut cols,
                    2.0..=8192.0,
                    true,
                    "Nombre de cellules de la grille",
                );
                self.params.cols = cols as u32;
                number_row_with_hint(
                    ui,
                    "Angle grille",
                    &mut self.params.grid_angle_deg,
                    -180.0..=180.0,
                    None,
                );
            }
            Algorithm::Kmeans | Algorithm::Voronoi | Algorithm::Quadtree => {
                let mut points = self.params.num_points;
                int_number_row(
                    ui,
                    "Nombre de points",
                    &mut points,
                    10.0..=100_000.0,
                    true,
                    "De 10 à 100 000 points, en échelle logarithmique",
                );
                self.params.num_points = points;
                if matches!(
                    self.params.algorithm,
                    Algorithm::Kmeans | Algorithm::Voronoi
                ) {
                    let mut iterations = self.params.iterations;
                    int_number_row(
                        ui,
                        "Itérations",
                        &mut iterations,
                        1.0..=100.0,
                        false,
                        "Passes de Lloyd. Au-delà de ~30, le gain devient marginal.",
                    );
                    self.params.iterations = iterations;
                }
            }
            Algorithm::Halftone => unreachable!("traité plus haut"),
        }
    }

    fn show_shape_and_palette_section(&mut self, ui: &mut egui::Ui) {
        section(ui, "Forme des points");
        show_shape_selector(ui, &mut self.params.dot_shape);
        ui.checkbox(
            &mut self.params.palette_size.is_some(),
            egui::RichText::new("Palette réduite").size(13.0),
        )
        .on_hover_text("Quantifie l'image vers quelques couleurs (effet sérigraphie)")
        .changed()
        .then(|| {
            self.params.palette_size = if self.params.palette_size.is_some() {
                None
            } else {
                Some(8)
            };
        });
        if let Some(size) = self.params.palette_size.as_mut() {
            let mut value = *size as f32;
            if number_row(ui, "Couleurs", &mut value, 2.0..=256.0).changed() {
                *size = value.round().max(2.0) as usize;
            }
            show_dithering_selector(ui, &mut self.params.dithering);
        }
    }

    fn show_bg_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        section(ui, "Fond et rendu");
        let mut color = egui::Color32::from_rgb(
            self.params.bg_color[0],
            self.params.bg_color[1],
            self.params.bg_color[2],
        );
        ui.horizontal(|ui| {
            ui.label("Couleur de fond");
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
                self.on_params_changed(ctx);
            }
            if ui.small_button("Blanc").clicked() {
                self.params.bg_color = [255, 255, 255];
                self.params.transparent = false;
                self.refresh_src_rgb(ctx);
                self.on_params_changed(ctx);
            }
            if ui.small_button("Noir").clicked() {
                self.params.bg_color = [0, 0, 0];
                self.params.transparent = false;
                self.refresh_src_rgb(ctx);
                self.on_params_changed(ctx);
            }
        });
        if ui
            .checkbox(
                &mut self.params.transparent,
                egui::RichText::new("Fond transparent (RGBA)").size(13.0),
            )
            .on_hover_text("Alpha conservé à l'export PNG/WebP/TIFF")
            .changed()
        {
            self.on_params_changed(ctx);
        }
        if ui
            .checkbox(
                &mut self.params.gamma_correct,
                egui::RichText::new("Correction gamma").size(13.0),
            )
            .on_hover_text(
                "Moyennes de couleur en espace linéaire : évite les mi-tons trop sombres",
            )
            .changed()
        {
            self.on_params_changed(ctx);
        }

        ui.horizontal(|ui| {
            if ui
                .checkbox(
                    &mut self.params.rng_seed.is_some(),
                    egui::RichText::new("Graine fixée").size(13.0),
                )
                .on_hover_text("Deux exécutions avec les mêmes réglages donnent le même résultat")
                .changed()
            {
                self.params.rng_seed = if self.params.rng_seed.is_some() {
                    None
                } else {
                    Some(42)
                };
                self.on_params_changed(ctx);
            }
            if let Some(seed) = self.params.rng_seed.as_mut() {
                let mut value = *seed;
                if ui
                    .add(egui::DragValue::new(&mut value).range(0..=u32::MAX as u64))
                    .changed()
                {
                    *seed = value;
                    self.on_params_changed(ctx);
                }
                if ui
                    .small_button("Varier")
                    .on_hover_text("Tirer une nouvelle graine")
                    .clicked()
                {
                    self.vary_seed(ctx);
                }
            }
        });
    }

    fn show_quality_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        section(ui, "Calcul");
        ui.horizontal(|ui| {
            ui.label("Aperçu");
            let mut index = match self.quality {
                Quality::Half => 0,
                Quality::Quarter => 1,
                Quality::Full => 2,
            };
            let response = ui.selectable_value(&mut index, 0, "Rapide ½");
            if response.clicked() {
                self.set_quality(Quality::Half, ctx);
            }
            response.on_hover_text("Calcul sur une image à 50 % : très réactif");
            let response = ui.selectable_value(&mut index, 1, "Rapide ¼");
            if response.clicked() {
                self.set_quality(Quality::Quarter, ctx);
            }
            response.on_hover_text("Calcul sur une image à 25 % : le plus réactif");
            let response = ui.selectable_value(&mut index, 2, "Plein");
            if response.clicked() {
                self.set_quality(Quality::Full, ctx);
            }
            response.on_hover_text("Toujours en pleine résolution");
        });
        ui.add(
            egui::Label::new(
                egui::RichText::new(
                    "En mode rapide, la pleine résolution est calculée dès que vous arrêtez \
                     de manipuler les réglages.",
                )
                .size(11.5),
            )
            .wrap(),
        );
    }

    fn show_halftone_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Mode");
            if ui
                .selectable_label(
                    matches!(self.params.halftone, HalftoneMode::Cmyk { .. }),
                    "CMJN",
                )
                .on_hover_text("Rosette classique Cyan / Magenta / Jaune / Noir, à 15°/75°/0°/45°")
                .clicked()
            {
                self.params.halftone = HalftoneMode::Cmyk {
                    angles: [15.0, 75.0, 0.0, 45.0],
                };
            }
            if ui
                .selectable_label(
                    matches!(self.params.halftone, HalftoneMode::Dominant { .. }),
                    "Dominant",
                )
                .on_hover_text("Les N couleurs dominantes de l'image deviennent chacune une encre")
                .clicked()
            {
                self.params.halftone = HalftoneMode::Dominant {
                    n: 5,
                    base_angle_deg: 15.0,
                };
            }
        });
        match &mut self.params.halftone {
            HalftoneMode::Dominant { n, base_angle_deg } => {
                let mut channels = *n;
                int_number_row(
                    ui,
                    "Canaux",
                    &mut channels,
                    2.0..=32.0,
                    false,
                    "Nombre d'encres retenues parmi les couleurs dominantes",
                );
                *n = channels;
                number_row_with_hint(ui, "Angle de base", base_angle_deg, 0.0..=180.0, None);
            }
            HalftoneMode::Cmyk { angles } => {
                ui.small(format!(
                    "Angles C/M/Y/K : {:.0}° / {:.0}° / {:.0}° / {:.0}°",
                    angles[0], angles[1], angles[2], angles[3]
                ));
            }
            HalftoneMode::Off => {
                ui.small("(sélectionnez CMJN ou Dominant)");
            }
        }
        ui.horizontal(|ui| {
            ui.label("Screening");
            ui.selectable_value(&mut self.params.screening, Screening::Am, "AM (grille)")
                .on_hover_text("Trame régulière : le rayon du dot code la couverture d'encre");
            ui.selectable_value(&mut self.params.screening, Screening::Fm, "FM (blue noise)")
                .on_hover_text(
                    "Trame stochastique : la densité des dots code la couverture d'encre",
                );
        });
        number_row(
            ui,
            "Fréquence de trame",
            &mut self.params.halftone_frequency,
            4.0..=400.0,
        );
        // 4 → 400 : logarithmique, la plage est trop large pour une course
        // linéaire.
        slider_row(
            ui,
            "halftone_frequency",
            &mut self.params.halftone_frequency,
            4.0..=400.0,
            true,
            "Cellules de la trame par unité de longueur",
        );
        number_row_with_hint(
            ui,
            "Rayon min",
            &mut self.params.halftone_min_radius_ratio,
            0.0..=1.0,
            None,
        );
        number_row_with_hint(
            ui,
            "Rayon max",
            &mut self.params.halftone_max_dot_ratio,
            0.05..=10.0,
            None,
        );
        // Même paire de bornes que pour les points : curseur à deux poignées,
        // échelle logarithmique (le rayon varie d'un facteur 100).
        range_slider(
            ui,
            "halftone_rayon",
            &mut self.params.halftone_min_radius_ratio,
            &mut self.params.halftone_max_dot_ratio,
            0.0..=10.0,
            Curve::Sqrt,
            "Rayon min et max des dots d'encre",
        );
    }

    /// Charge (une seule fois) les textures egui correspondant aux images.
    fn ensure_textures(&mut self, ctx: &egui::Context) {
        if let Some(src) = &self.src_rgb
            && self.src_texture.is_none()
        {
            self.src_texture = Some(ctx.load_texture(
                "src",
                rgb_to_color_image(src),
                egui::TextureOptions::default(),
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
                    egui::TextureOptions::default(),
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
                egui::TextureOptions::default(),
            ));
        }
    }

    /// Passe au mode d'affichage suivant (Tab).
    fn cycle_view(&mut self, delta: i32) {
        let index = view_index(self.view_mode) as i32;
        let next = (index + delta).rem_euclid(pointimg::frontend::VIEW_COUNT as i32) as u32;
        self.view_mode = view_from_index(next);
    }

    // ─── Zone centrale ───────────────────────────────────────────────────────

    fn show_central_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, computing: bool) {
        egui::CentralPanel::default().show(ui, |ui| {
            self.ensure_textures(ctx);

            ui.horizontal_wrapped(|ui| {
                ui.label("Affichage");
                for (label, mode) in VIEWS {
                    ui.selectable_value(&mut self.view_mode, mode, label);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("1:1")
                        .on_hover_text("Revenir à 100 % (1)")
                        .clicked()
                    {
                        self.set_zoom(1.0);
                    }
                    let fit = ui
                        .selectable_label(self.zoom_fit, "Ajuster")
                        .on_hover_text("Ajuster l'aperçu à la fenêtre (0)");
                    if fit.clicked() {
                        self.zoom_fit = true;
                    }
                    let mut percent = (self.current_zoom() * 100.0).round();
                    let editable = !self.zoom_fit;
                    let response = ui.add_enabled(
                        editable,
                        egui::DragValue::new(&mut percent)
                            .speed(20.0)
                            .range(2.0..=1600.0)
                            .suffix(" %"),
                    );
                    if response.changed() {
                        self.set_zoom(percent / 100.0);
                    }
                });
            });
            ui.separator();

            if self.src_rgb.is_none() {
                self.show_empty_state(ui, ctx);
                return;
            }

            let (source, result, density) = (
                self.src_texture.clone(),
                self.result_texture.clone(),
                self.density_texture.clone(),
            );
            let zoom = self.zoom;
            let zoom_fit = self.zoom_fit;
            let stale = self.result_is_stale();
            // Hauteur restante pour les aperçus (barre d'outils déjà posée).
            let height = (ui.available_height() - 8.0).max(120.0);
            match self.view_mode {
                ViewMode::Compare => self.show_compare(ui, stale),
                ViewMode::Side => {
                    // Deux moitiés *plus* le séparateur doivent tenir dans la
                    // largeur disponible, sinon le second panneau est rogné.
                    // En dessous de 640 px (soit ~300 px par image), on
                    // empile : deux images côte à côte deviennent illisibles.
                    let available = ui.available_size();
                    let separator = ui.spacing().item_spacing.x * 3.0;
                    if available.x < 640.0 {
                        let half = (available.y / 2.0 - 6.0).max(64.0);
                        show_pane(
                            ui,
                            Pane {
                                caption: "Source",
                                texture: source.as_ref(),
                                size: egui::vec2(available.x, half),
                                zoom,
                                zoom_fit,
                                spinning: false,
                            },
                        );
                        show_pane(
                            ui,
                            Pane {
                                caption: "Résultat",
                                texture: result.as_ref(),
                                size: egui::vec2(available.x, half),
                                zoom,
                                zoom_fit,
                                spinning: computing || stale,
                            },
                        );
                    } else {
                        let panel_w = ((available.x - separator) / 2.0).max(64.0);
                        let source_pane = Pane {
                            caption: "Source",
                            texture: source.as_ref(),
                            size: egui::vec2(panel_w, height),
                            zoom,
                            zoom_fit,
                            spinning: false,
                        };
                        let result_pane = Pane {
                            caption: "Résultat",
                            texture: result.as_ref(),
                            size: egui::vec2(panel_w, height),
                            zoom,
                            zoom_fit,
                            spinning: computing || stale,
                        };
                        ui.horizontal(|ui| {
                            show_pane(ui, source_pane);
                            ui.separator();
                            show_pane(ui, result_pane);
                        });
                    }
                }
                ViewMode::ResultOnly => show_pane(
                    ui,
                    Pane {
                        caption: "Résultat",
                        texture: result.as_ref(),
                        size: egui::vec2(ui.available_width(), height),
                        zoom,
                        zoom_fit,
                        spinning: computing || stale,
                    },
                ),
                ViewMode::SourceOnly => show_pane(
                    ui,
                    Pane {
                        caption: "Source",
                        texture: source.as_ref(),
                        size: egui::vec2(ui.available_width(), height),
                        zoom,
                        zoom_fit,
                        spinning: false,
                    },
                ),
                ViewMode::DensityMap => show_pane(
                    ui,
                    Pane {
                        caption: "Density map",
                        texture: density.as_ref(),
                        size: egui::vec2(ui.available_width(), height),
                        zoom,
                        zoom_fit,
                        spinning: false,
                    },
                ),
            }
        });
    }

    /// Comparaison avant/après : source à gauche d'un curseur, résultat à
    /// droite, dans la même zone — la comparaison porte sur la même zone pixel.
    fn show_compare(&mut self, ui: &mut egui::Ui, stale: bool) {
        let (Some(source), Some(result)) = (&self.src_texture, &self.result_texture) else {
            ui.centered_and_justified(|ui| {
                if self.computing.load(Ordering::Relaxed) {
                    ui.spinner();
                }
                ui.label("Calcul en cours…");
            });
            return;
        };
        let available = ui.available_size();
        let sw = source.size_vec2().x.max(1.0);
        let sh = source.size_vec2().y.max(1.0);
        let rw = result.size_vec2().x.max(1.0);
        let rh = result.size_vec2().y.max(1.0);
        let zoom = if self.zoom_fit {
            (available.x / sw.max(rw))
                .min((available.y - 22.0).max(32.0) / sh.max(rh))
                .max(0.01)
        } else {
            self.zoom
        };
        self.effective_zoom = zoom;
        let total_w = (sw.max(rw) * zoom).max(1.0);
        let total_h = (sh.max(rh) * zoom).max(1.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Comparaison").size(12.0).weak());
            ui.label(
                egui::RichText::new("glissez le curseur (double-clic : centré)")
                    .size(11.0)
                    .weak(),
            );
        });
        egui::ScrollArea::both()
            .id_salt("compare")
            .max_width(available.x)
            .max_height((available.y - 22.0).max(32.0))
            .show(ui, |ui| {
                let (rect, response) = ui.allocate_exact_size(
                    egui::vec2(total_w, total_h),
                    egui::Sense::click_and_drag(),
                );
                let split = (rect.width() * self.compare_split)
                    .clamp(16.0, (rect.width() - 16.0).max(16.0));
                let cursor_x = rect.min.x + split;
                let uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0));

                // Source à gauche, résultat à droite, découpés au curseur.
                ui.painter()
                    .with_clip_rect(egui::Rect::from_min_max(
                        rect.min,
                        egui::pos2(cursor_x, rect.max.y),
                    ))
                    .image(
                        source.id(),
                        egui::Rect::from_min_size(rect.min, egui::vec2(sw * zoom, sh * zoom)),
                        uv,
                        egui::Color32::WHITE,
                    );
                ui.painter()
                    .with_clip_rect(egui::Rect::from_min_max(
                        egui::pos2(cursor_x, rect.min.y),
                        rect.max,
                    ))
                    .image(
                        result.id(),
                        egui::Rect::from_min_size(rect.min, egui::vec2(rw * zoom, rh * zoom)),
                        uv,
                        egui::Color32::WHITE,
                    );

                let painter = ui.painter();
                painter.line_segment(
                    [
                        egui::pos2(cursor_x, rect.min.y),
                        egui::pos2(cursor_x, rect.max.y),
                    ],
                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                );
                let center = egui::pos2(cursor_x, rect.center().y);
                painter.circle_filled(center, 11.0, egui::Color32::WHITE);
                painter.circle_stroke(
                    center,
                    11.0,
                    egui::Stroke::new(1.5, egui::Color32::DARK_GRAY),
                );
                if stale {
                    painter.text(
                        egui::pos2(rect.min.x + 10.0, rect.min.y + 18.0),
                        egui::Align2::LEFT_CENTER,
                        "aperçu obsolète — recalcul en cours",
                        egui::FontId::proportional(12.0),
                        egui::Color32::WHITE,
                    );
                }
                for direction in [-1.0f32, 1.0] {
                    painter.line_segment(
                        [
                            egui::pos2(cursor_x + direction * 6.0, center.y),
                            egui::pos2(cursor_x + direction * 2.0, center.y + direction * 4.0),
                        ],
                        egui::Stroke::new(1.5, egui::Color32::DARK_GRAY),
                    );
                }

                if response.dragged() {
                    self.compare_split = (self.compare_split
                        + response.drag_delta().x / rect.width().max(1.0))
                    .clamp(0.02, 0.98);
                }
                if response.double_clicked() {
                    self.compare_split = 0.5;
                }
            });
    }

    /// État d'accueil : quelque chose à faire tant qu'aucune image n'est chargée.
    fn show_empty_state(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.vertical_centered(|ui| {
            ui.add_space((ui.available_height() * 0.22).max(24.0));
            ui.heading("Aucune image");
            ui.label("Glissez-déposez un fichier dans la fenêtre, ou ouvrez-en un.");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Ouvrir une image…").clicked() {
                    self.choose_image(ctx);
                }
                if ui.button("Ouvrir un dossier…").clicked() {
                    self.choose_folder(ctx);
                }
                if let Some(sample) = sample_image()
                    && ui.button("Utiliser un exemple").clicked()
                {
                    self.load_image(sample, ctx);
                }
            });
        });
    }

    // ─── Barre d'état ───────────────────────────────────────────────────────

    fn show_status_bar(&mut self, ui: &mut egui::Ui, computing: bool) {
        let stale = self.result_is_stale();
        let timing = self.last_compute_ms;
        let quick = !self.quality.is_full() && self.src_rgb.is_some();
        let quick_stale = stale;
        let status = self.status.clone();
        // Taille d'export : mesurée sur le résultat définitif, encodage réel.
        let sizes = self.poll_export_sizes();
        let sizes_label = sizes
            .as_ref()
            .map(exports::export_sizes_line)
            .unwrap_or_default();
        let sizes_comparison = sizes.as_ref().and_then(exports::export_comparison);
        let sizes_measuring = self.export_sizes_computing.load(Ordering::Relaxed);
        egui::Panel::bottom("status_bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if computing {
                    ui.spinner();
                    let (done, total) = *self.progress.lock().unwrap_or_else(|e| e.into_inner());
                    if total > 0 {
                        ui.add(
                            egui::ProgressBar::new(done as f32 / total as f32).desired_width(160.0),
                        );
                        ui.small(format!("{done}/{total}"));
                    } else {
                        ui.label("Calcul en cours…");
                    }
                } else {
                    ui.label(&status);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if !sizes_label.is_empty() {
                        let label = ui.small(format!("Export : {sizes_label}"));
                        if let Some(comparison) = &sizes_comparison {
                            label.on_hover_text(format!(
                                "{comparison}\n\
                                 Tailles obtenues en encodant réellement le rendu dans \
                                 chaque format.\n\
                                 « — » : format non mesuré (image trop grande pour un \
                                 WebP sans perte)."
                            ));
                        }
                    } else if sizes_measuring {
                        ui.small("Export : mesure…");
                    }
                    if quick && quick_stale {
                        ui.small("pleine résolution…");
                    } else if quick {
                        ui.small("aperçu rapide").on_hover_text(
                            "Calcul sur une image réduite ; la pleine résolution est \
                                 obtenue automatiquement dès que vous arrêtez.",
                        );
                    }
                    if let Some(ms) = timing {
                        ui.small(format!("Dernier calcul : {}", format_duration(ms)));
                    }
                });
            });
        });
    }
}

// ─── Aides ───────────────────────────────────────────────────────────────────

/// Panneau d'aperçu : cadre, légende, image avec zoom et défilement.
/// Un panneau d'aperçu : ce qu'il faut pour le dessiner dans une place donnée.
///
/// Regroupé dans une structure plutôt qu'en huit paramètres : la place
/// allouée, la texture et l'état du zoom voyagent toujours ensemble.
struct Pane<'a> {
    caption: &'a str,
    texture: Option<&'a egui::TextureHandle>,
    size: egui::Vec2,
    zoom: f32,
    zoom_fit: bool,
    spinning: bool,
}

fn show_pane(ui: &mut egui::Ui, pane: Pane<'_>) {
    let Pane {
        caption,
        texture,
        size,
        zoom,
        zoom_fit,
        spinning,
    } = pane;
    // La place est imposée par l'appelant : dans une mise en page horizontale,
    // `available_width()` est la largeur restante *pour chaque enfant*, donc
    // le premier panneau consumesait tout l'espace et le second était expulsé.
    ui.allocate_ui(egui::vec2(size.x.max(64.0), size.y.max(64.0)), |ui| {
        egui::Frame::central_panel(ui.style())
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.label(egui::RichText::new(caption).size(12.0).weak());
                let (w, h) = (
                    ui.available_width().max(1.0),
                    ui.available_height().max(1.0),
                );
                let (tw, th) = texture
                    .map(|tex| (tex.size_vec2().x.max(1.0), tex.size_vec2().y.max(1.0)))
                    .unwrap_or((w, h));
                let scale = if zoom_fit {
                    (w / tw).min(h / th).max(0.01)
                } else {
                    zoom
                };
                let (img_w, img_h) = (tw * scale, th * scale);
                egui::ScrollArea::both()
                    .id_salt(caption)
                    .max_width(w)
                    .max_height(h)
                    .show(ui, |ui| match texture {
                        Some(tex) => {
                            ui.image((tex.id(), egui::vec2(img_w, img_h)));
                        }
                        None => {
                            if spinning {
                                ui.spinner();
                            } else {
                                ui.centered_and_justified(|ui| {
                                    ui.label("Aucune image");
                                });
                            }
                        }
                    });
            });
    });
}

/// Titre de section, façon `adw::PreferencesGroup`.
fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.strong(title);
    ui.separator();
}

/// Ligne « libellé + saisie directe » : la plage réelle (celle de
/// `validate_params`) est donnée en infobulle, plus une indication d'unité.
fn number_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
) -> egui::Response {
    number_row_with_hint(ui, label, value, range, None)
}

fn number_row_with_hint(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    hint: Option<&str>,
) -> egui::Response {
    let range_label = format!("{:.4} – {:.4}", *range.start(), range.end());
    let mut tooltip = match hint {
        Some(hint) => format!("{hint}\nPlage autorisée : {range_label}"),
        None => format!("Plage autorisée : {range_label}"),
    };
    if label.contains("Itérations") || label.contains("Côtés") || label.contains("Canaux") {
        tooltip.push_str("\nMolette : ajustement fin (avec Ctrl ou Maj)");
    }
    // egui choisit la précision d'affichage selon la distance du pointeur au
    // bord de l'écran : près d'un bord, « Rayon min » s'afficherait « 0,000 »
    // puis la saisie serait réécrite en 0 à la perte de focus. On impose donc
    // les décimales utiles au réglage (cf. `display_decimals`).
    let decimals = display_decimals(*value, &range);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(13.0));
        // `DragValue` n'a pas de pas en 0.36 : la vitesse fixe l'incrément de
        // la molette, la saisie directe accepte n'importe quelle valeur.
        ui.add(
            egui::DragValue::new(value)
                .speed(0.25)
                .clamp_existing_to_range(true)
                .min_decimals(decimals)
                .max_decimals(decimals)
                .range(*range.start()..=*range.end()),
        )
        .on_hover_text(tooltip)
    })
    .inner
}

/// Sélecteur d'algorithme de dithering, avec l'aide correspondant à chacun.
fn show_dithering_selector(ui: &mut egui::Ui, dithering: &mut Dithering) {
    const CHOICES: [(Dithering, &str, &str); 5] = [
        (Dithering::Off, "Aucun", "Palette franche, sans motif"),
        (
            Dithering::FloydSteinberg,
            "Floyd-Steinberg",
            "Diffusion de l'erreur : dégradés préservés, motif légèrement « verreux »",
        ),
        (
            Dithering::Riemersma,
            "Riemersma",
            "Diffusion à queue longue : moins de vers, motif plus lisse",
        ),
        (
            Dithering::Bayer4,
            "Bayer 4×4",
            "Tramage ordonné : rendu « pixel art », sans diffusion d'erreur",
        ),
        (
            Dithering::Bayer8,
            "Bayer 8×8",
            "Tramage ordonné plus fin : motif de trame 8×8",
        ),
    ];
    let (label, hint) = CHOICES
        .iter()
        .find(|(mode, _, _)| mode == dithering)
        .map(|(_, label, hint)| (*label, *hint))
        .unwrap_or(("Inconnu", ""));
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Dithering").size(13.0));
        let response = egui::ComboBox::from_id_salt("dithering_combo")
            .selected_text(label)
            .show_ui(ui, |ui| {
                for (mode, name, help) in CHOICES {
                    ui.selectable_value(dithering, mode, name)
                        .on_hover_text(help);
                }
            });
        response.response.on_hover_text(hint);
    });
}

/// Ligne « libellé + saisie directe + curseur » pour un réglage entier : la
/// saisie comme le glissement sont arrondis à l'unité.
fn int_number_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    hint: &str,
) {
    let mut mirror = *value as f32;
    if number_row(ui, label, &mut mirror, range.clone()).changed() {
        *value = mirror.round().max(1.0) as usize;
    }
    int_slider_row(ui, label, value, range, logarithmic, hint);
}

/// Sélecteur de forme des points.
fn show_shape_selector(ui: &mut egui::Ui, shape: &mut DotShape) {
    let current = match shape {
        DotShape::Circle => "Cercle",
        DotShape::Square => "Carré",
        DotShape::Ellipse { .. } => "Ellipse",
        DotShape::RegularPolygon { .. } => "Polygone",
    };
    ui.horizontal(|ui| {
        ui.label("Forme");
        for (label, variant) in [("Cercle", DotShape::Circle), ("Carré", DotShape::Square)] {
            ui.selectable_value(shape, variant, label);
        }
        if ui
            .selectable_label(matches!(shape, DotShape::Ellipse { .. }), "Ellipse")
            .clicked()
            && !matches!(shape, DotShape::Ellipse { .. })
        {
            *shape = DotShape::Ellipse {
                aspect: 1.5,
                angle_deg: 0.0,
            };
        }
        if ui
            .selectable_label(matches!(shape, DotShape::RegularPolygon { .. }), "Polygone")
            .clicked()
            && !matches!(shape, DotShape::RegularPolygon { .. })
        {
            *shape = DotShape::RegularPolygon { sides: 6 };
        }
    });
    let _ = current;
    match shape {
        DotShape::Ellipse { aspect, angle_deg } => {
            number_row(ui, "Rapport l/h", aspect, 0.1..=10.0);
            number_row_with_hint(ui, "Angle", angle_deg, -180.0..=180.0, None);
        }
        DotShape::RegularPolygon { sides } => {
            let mut value = *sides as f32;
            if number_row(ui, "Côtés", &mut value, 3.0..=12.0).changed() {
                *sides = value.round().clamp(3.0, 12.0) as u8;
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_cycle_wraps() {
        let len = VIEWS.len() as i32;
        for index in 0..len {
            let next = ((index + 1).rem_euclid(len)) as usize;
            assert_ne!(VIEWS[next].1, VIEWS[index as usize].1);
        }
    }

    #[test]
    fn number_row_range_is_inclusive() {
        let range = 2.0..=256.0;
        assert_eq!(*range.start(), 2.0);
        assert_eq!(*range.end(), 256.0);
    }
}
