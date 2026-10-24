//! egui rendering: header, settings panel, preview area, shortcuts.
//!
//! The interface takes over the decisions of the GTK4 version: actions visible
//! in the header (nothing important below the waterline), before/after
//! comparison by sweeping, settings in direct entry, quick preview with
//! automatic finalization, built-in presets, and a welcome state that gives
//! something to do.

use eframe::egui;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use pointimg::filter;
use pointimg::filter::{Algorithm, Dithering, DotShape, HalftoneMode, Screening};

use super::App;
use super::convert::{gray_to_color_image, rgb_to_color_image, rgba_to_color_image_checker};
use super::exports;
use super::format::format_duration;
use super::sliders::{Bounds, display_decimals, range_slider, round_to};
use pointimg::frontend::{
    self, Curve, Quality, ViewMode, sample_image, view_from_index, view_index,
};

/// Short description of an algorithm, displayed under the selector.
/// Label of an algorithm in the selector.
///
/// Total `match`, with no `_`: adding a variant to `Algorithm` without
/// labeling it here is a compilation error, not a shifted list.
fn algorithm_label(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Grid => "Grille",
        Algorithm::Kmeans => "K-means",
        Algorithm::Voronoi => "Voronoi (Lloyd)",
        Algorithm::Quadtree => "Quadtree",
        Algorithm::Halftone => "Halftone (rosette)",
    }
}

fn algorithm_help(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Grid => pointimg::frontend::t(
            "Grille régulière : une cellule = un point. Rapide et prévisible.",
        ),
        Algorithm::Kmeans => pointimg::frontend::t(
            "k-means spatial : les points se répartissent selon la couleur et la position.",
        ),
        Algorithm::Voronoi => pointimg::frontend::t(
            "Lloyd pondéré : répartition régulière guidée par le détail. Rendu le plus « naturel ».",
        ),
        Algorithm::Quadtree => pointimg::frontend::t(
            "Subdivision par contraste : davantage de points là où l'image est détaillée.",
        ),
        Algorithm::Halftone => {
            "Trame d'encre : rosette CMJN ou couleurs dominantes, style offset print."
        }
    }
}

/// Display modes, in cycling order.
fn views() -> [(&'static str, ViewMode); 5] {
    [
        (pointimg::frontend::t("Comparaison"), ViewMode::Compare),
        (pointimg::frontend::t("Côte à côte"), ViewMode::Side),
        (pointimg::frontend::t("Résultat"), ViewMode::ResultOnly),
        (pointimg::frontend::t("Source"), ViewMode::SourceOnly),
        (
            pointimg::frontend::t("Carte de densité"),
            ViewMode::DensityMap,
        ),
    ]
}

/// Global style of the interface.
///
/// `scale` enlarges *everything* that gives the perceived density — text body,
/// spacings, widget height, slider width, corner radii — so that a single
/// setting makes the interface comfortable without touching every sub-widget.
/// 1.0 = comfortable (default value), 1.3 = large.
fn tuned_style(style: &mut egui::Style, scale: f32) {
    let scale = scale.clamp(0.85, 1.6);

    // ── Texts ────────────────────────────────────────────────────────────────
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

    // ── Vertical and horizontal rhythm ───────────────────────────────────────
    style.spacing.item_spacing = egui::vec2(10.0, 8.0) * scale;
    style.spacing.button_padding = egui::vec2(10.0, 5.0) * scale;
    style.spacing.indent = 20.0 * scale;
    style.spacing.interact_size.y = 24.0 * scale;
    // The sliders take the width of the panel (like the two-handle slider of
    // the radii): 120 px were too narrow to adjust finely. It is a ceiling,
    // not an imposed width: `show_control_panel` bounds it to the available
    // width, otherwise the panel could not go below ~440 px and ate half of a
    // 900 px window.
    style.spacing.slider_width = 320.0 * scale;
    style.spacing.window_margin = egui::Margin::same((8.0 * scale) as i8);
    style.spacing.menu_margin = egui::Margin::same((6.0 * scale) as i8);

    // ── More generous and less angular widgets ───────────────────────────────
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
    // A bit more contrast between the panels and the window background.
    style.visuals.window_fill = style.visuals.window_fill.gamma_multiply(1.0);
    style.visuals.panel_fill = style.visuals.panel_fill.gamma_multiply(1.0);
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Interface language: applied first, so every `t()` call below resolves
        // in the right language.
        pointimg::frontend::set_locale(self.ui_locale);

        // Memorized theme: applied from the very first render (the context only
        // exists from there, and "Auto" must also correct an eventual theme left
        // by eframe in its own preferences).
        self.apply_theme(&ctx);

        // `eframe::App::style` is not exposed by eframe 0.36: the style is
        // applied here, and reapplied when the user changes the interface
        // scale.
        if self.applied_style != Some(self.ui_scale) {
            let scale = self.ui_scale;
            ctx.all_styles_mut(|style| tuned_style(style, scale));
            self.applied_style = Some(scale);
        }

        self.poll_worker_results();
        self.collect_errors();
        self.expire_toasts(&ctx);
        self.handle_pending_open(&ctx);
        self.show_palette(&ctx);
        self.handle_cancel_and_debounce(&ctx);
        self.handle_shortcuts(&ctx);
        self.handle_drag_and_drop(&ctx);

        let computing = self.computing.load(Ordering::Relaxed);
        self.finish_pending_compute(computing);
        // Export size: relaunched at every definitive result, invalidated as
        // soon as the user goes back to adjusting the image.
        self.refresh_export_sizes(&ctx);
        self.sync_window_title(&ctx);

        // The order matters: the bottom panel must be reserved before the left
        // panel, otherwise the latter overflows onto it.
        self.show_view_bar(ui, &ctx, computing);
        self.show_status_bar(ui, computing);
        self.show_control_panel(ui, &ctx);
        self.show_central_panel(ui, &ctx, computing);
    }
}

impl App {
    // ─── Lifecycle ───────────────────────────────────────────────────────────

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
        // The borrow of the lock closes before calling `notify_error`: the
        // guard must be released first.
        let error = self
            .compute_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(err) = error {
            // A compute error must outlive the duration of the message, which
            // the status line would replace on the next frame.
            self.notify_error(err);
        }
    }

    /// Closes the compute in progress: duration, message, history commit.
    fn finish_pending_compute(&mut self, computing: bool) {
        if computing {
            return;
        }
        if let Some(start) = self.compute_start.take() {
            let ms = start.elapsed().as_millis() as u64;
            self.last_compute_ms = Some(ms);
            let final_result = self.result_is_final.load(Ordering::Acquire);
            self.status = if final_result {
                pointimg::frontend::t("Calcul terminé en {}.").replace("{}", &format_duration(ms))
            } else {
                pointimg::frontend::t("Aperçu calculé — finalisation en cours…").to_string()
            };
            if self.pending_commit {
                self.commit_history();
                self.pending_commit = false;
            }
        }
        // A finished quick preview triggers the full resolution pass.
        if !self.has_final_result()
            && !self.computing.load(Ordering::Relaxed)
            && self.has_any_result()
        {
            self.schedule_finalize();
        }
    }

    /// Schedules the full resolution pass when the user stops
    /// manipulating the settings (quick preview mode).
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

    /// Is the displayed result usable (full resolution, up to date)?
    pub(crate) fn has_final_result(&self) -> bool {
        self.result_is_final.load(Ordering::Acquire) && self.has_any_result()
    }

    /// The displayed result no longer matches the current settings.
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

    /// Command palette (`Ctrl+K`).
    ///
    /// It intercepts the keyboard and draws above everything: it is an egui
    /// `Window`, while the GTK equivalent is a menu and a shortcuts window.
    fn show_palette(&mut self, ctx: &egui::Context) {
        let Some(query) = self.palette.take() else {
            return;
        };
        // `None` = the palette closes; the remaining text reopens it on the
        // next frame.
        self.palette = super::palette::show(self, ctx, query);
    }

    /// Removes the expired notifications and redraws on time.
    ///
    /// egui having no clock, a frame must be explicitly requested at the
    /// expiry: without this `request_repaint_after`, a message would stay
    /// displayed indefinitely as long as the user did not interact.
    fn expire_toasts(&mut self, ctx: &egui::Context) {
        if self.toasts.expire() {
            ctx.request_repaint();
        }
        if let Some(remaining) = self.toasts.next_expiry() {
            ctx.request_repaint_after(remaining);
        }
    }

    /// Opens the path received on the command line, once the window is ready.
    ///
    /// `main` receives it before the context exists: it can only load it now.
    /// A folder enables navigation between images, like a drag and drop.
    fn handle_pending_open(&mut self, ctx: &egui::Context) {
        let Some(path) = self.pending_open.take() else {
            return;
        };
        if path.is_dir() {
            self.open_folder(path, ctx);
        } else {
            self.load_image(path, ctx);
        }
    }

    // ─── Shortcuts and drag and drop ─────────────────────────────────────────

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
            palette: bool,
            view_back: bool,
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
                // `Tab` moves forward, `Shift+Tab` moves backward: the shortcuts
                // window of the GTK interface announces both, and it is also the
                // expected gesture for any tab list.
                view: plain && i.key_pressed(egui::Key::Tab),
                view_back: plain && shift && i.key_pressed(egui::Key::Tab),
                zoom_in: plain
                    && (i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)),
                zoom_out: plain && i.key_pressed(egui::Key::Minus),
                fit: plain && i.key_pressed(egui::Key::Num0),
                one: plain && i.key_pressed(egui::Key::Num1),
                prev: plain && i.key_pressed(egui::Key::ArrowLeft),
                next: plain && i.key_pressed(egui::Key::ArrowRight),
                reset: command && i.key_pressed(egui::Key::R),
                palette: command && i.key_pressed(egui::Key::K),
            }
        });

        // As long as an input field has focus, "letter" shortcuts would be
        // typed into the field.
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
        if keys.palette && !wants_text {
            // The palette intercepts the whole keyboard: the guard is
            // essential, since its own fields are the only ones where `Ctrl+K`
            // can be a command (in a file name, for example).
            self.palette = if self.palette.is_some() {
                None
            } else {
                Some(String::new())
            };
        }
        if keys.undo && !wants_text {
            self.undo(ctx);
        }
        if keys.redo && !wants_text {
            self.redo(ctx);
        }
        if keys.recalc && !wants_text {
            self.request_full_compute(ctx);
        }
        if keys.preset && !wants_text {
            self.cycle_preset(1, ctx);
        }
        if keys.view && !wants_text {
            self.cycle_view(1);
        }
        if keys.view_back && !wants_text {
            self.cycle_view(-1);
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
                // The density map follows the same debounce as the filter.
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

    // ─── Header ──────────────────────────────────────────────────────────────

    /// Everything that is an action lives in the header: nothing left to scroll.
    fn show_view_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, computing: bool) {
        let has_final = self.has_final_result();
        let has_dots = self
            .last_dots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
            || self.src_rgb.is_some();
        // The tooltip of the export button announces the resulting size, to
        // avoid opening the dialog just to discover it.
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
                    .unwrap_or_else(|| pointimg::frontend::t("Rien à annuler").to_string());
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
                    .map(|entry| {
                        pointimg::frontend::t("Rétablir : {} (Ctrl+Y)").replace("{}", &entry.label)
                    })
                    .unwrap_or_else(|| pointimg::frontend::t("Rien à rétablir").to_string());
                if ui
                    .add_enabled(
                        !self.future.is_empty(),
                        egui::Button::new(pointimg::frontend::t("Rétablir")),
                    )
                    .on_hover_text(redo_label)
                    .clicked()
                {
                    self.redo(ctx);
                }
                if ui
                    .button("⟲")
                    .on_hover_text(pointimg::frontend::t(
                        "Rétablir tous les réglages par défaut (Ctrl+R)",
                    ))
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
                    .on_hover_text(pointimg::frontend::t(
                        "Exporter le rendu en SVG (Ctrl+Shift+S)",
                    ))
                    .clicked()
                {
                    self.choose_svg_export();
                }
                if ui
                    .add_enabled(has_final, egui::Button::new("Copier"))
                    .on_hover_text(pointimg::frontend::t(
                        "Copier l'image dans le presse-papiers (Ctrl+C)",
                    ))
                    .clicked()
                {
                    self.copy_to_clipboard(ctx);
                }
            });
            ui.add_space(3.0);
        });
    }

    // ─── Settings panel ──────────────────────────────────────────────────────

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
                            ui.label(info).on_hover_text(pointimg::frontend::t(
                                "Dimensions de l'image chargée",
                            ));
                        });
                    });
                    ui.separator();

                    // A single guard for the whole panel: a compute is only
                    // restarted if the parameters have *really* changed.
                    // Otherwise, a simple focus change between two fields — or
                    // the revalidation of a slider by egui — would be enough to
                    // recompute the image.
                    let before = self.params.clone();
                    self.show_display_preferences(ui);
                    self.show_presets(ui, ctx);
                    section(ui, "Algorithme", "algorithm", true, |ui| {
                        self.show_algorithm_section(ui);
                    });
                    section(
                        ui,
                        pointimg::frontend::t("Forme des points"),
                        "shape",
                        true,
                        |ui| {
                            self.show_shape_and_palette_section(ui);
                        },
                    );
                    section(ui, "Fond et rendu", "bg", false, |ui| {
                        self.show_bg_section(ui, ctx);
                    });
                    section(ui, "Calcul", "quality", false, |ui| {
                        self.show_quality_section(ui, ctx);
                    });
                    if self.params != before {
                        self.on_params_changed(ctx);
                    }
                    ui.separator();
                    if ui
                        .button(pointimg::frontend::t("Réinitialiser les réglages"))
                        .clicked()
                    {
                        self.reset_params(ctx);
                    }
                });
            });
    }

    /// Display comfort: interface size and theme.
    fn show_display_preferences(&mut self, ui: &mut egui::Ui) {
        // `horizontal_wrapped`: the row folds instead of widening the panel.
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
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(pointimg::frontend::t("Langue")).size(13.0));
            for (code, label) in [("en", "English"), ("fr", "Français")] {
                let selected = self.ui_locale.code() == code;
                if ui.selectable_label(selected, label).clicked() {
                    self.ui_locale = pointimg::frontend::Locale::from_code(code);
                    pointimg::frontend::set_locale(self.ui_locale);
                    self.persist_display_preferences();
                }
            }
        });
        ui.separator();
    }

    /// Built-in presets: the fastest starting point.
    fn show_presets(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let presets = filter::builtin_presets();
        let current = self.preset_index();
        group_title(ui, pointimg::frontend::t("Réglages rapides"));

        // Thumbnail grid: a preset is chosen on its render, not on its name.
        // Without an image (`assets/presets/` missing), we fall back on a list
        // of names — same information, different form.
        // Once, not on every frame: the panel is redrawn much more often than
        // the thumbnails change.
        self.ensure_preset_thumbnails(ctx);
        // The clone is one `Arc` handle per thumbnail, not an image. Without it,
        // the borrow of `self` would cross the `&mut self` calls below.
        let textures: Vec<Option<egui::TextureHandle>> =
            self.preset_thumbnails.clone().unwrap_or_default();
        let avec_vignettes = textures.iter().any(|t| t.is_some());

        if avec_vignettes {
            self.show_preset_grid(ui, ctx, &presets, &textures, current);
        } else {
            self.show_preset_list(ui, ctx, &presets, current);
        }

        ui.horizontal(|ui| {
            if ui.button("Charger un preset…").clicked() {
                self.choose_load_preset(ctx);
            }
            if ui
                .button(pointimg::frontend::t("Sauver le preset…"))
                .clicked()
            {
                self.choose_save_preset();
            }
        });
    }

    /// Thumbnail grid, three per row.
    ///
    /// The current thumbnail is framed: without that marker, one does not know
    /// which preset is applied once the settings have been customized — the
    /// name alone no longer says it.
    fn show_preset_grid(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        presets: &[filter::BuiltinPreset],
        textures: &[Option<egui::TextureHandle>],
        current: Option<usize>,
    ) {
        let colonnes = 3;
        egui::Grid::new("preset_grid")
            .num_columns(colonnes)
            .spacing(egui::vec2(6.0, 6.0))
            .show(ui, |ui| {
                for (ligne, chunk) in textures.chunks(colonnes).enumerate() {
                    for (colonne, texture) in chunk.iter().enumerate() {
                        let index = ligne * colonnes + colonne;
                        let Some(preset) = presets.get(index) else {
                            break;
                        };
                        let selectionne = current == Some(index);
                        let reponse = self.preset_button(ui, texture.as_ref(), selectionne);
                        let reponse = reponse
                            .on_hover_text(format!("{}\n\n{}", preset.name, preset.description));
                        if reponse.clicked() {
                            self.apply_preset(index, ctx);
                        }
                    }
                    ui.end_row();
                }
            });
        // The name of the current preset, and what "Custom" means.
        let legende = match current {
            Some(index) => presets[index].name.to_string(),
            None => {
                pointimg::frontend::t("Personnalisé — les réglages ne correspondent à aucun preset")
                    .to_string()
            }
        };
        ui.label(
            egui::RichText::new(legende)
                .size(11.5)
                .color(ui.visuals().weak_text_color()),
        );
    }

    /// A clickable thumbnail, framed if it is the current selection.
    fn preset_button(
        &self,
        ui: &mut egui::Ui,
        texture: Option<&egui::TextureHandle>,
        selectionne: bool,
    ) -> egui::Response {
        let taille = egui::vec2(96.0, 72.0);
        let (rect, reponse) = ui.allocate_exact_size(taille, egui::Sense::click());
        let rayon = egui::CornerRadius::same(4);
        if selectionne {
            ui.painter().rect_stroke(
                rect,
                rayon,
                egui::Stroke::new(2.0, ui.visuals().selection.stroke.color),
                egui::StrokeKind::Outside,
            );
        }
        if let Some(texture) = texture {
            // Full-image `uv`, and a rectangle computed to keep the ratio:
            // the thumbnails are in 4:3, and so is the cell.
            ui.painter().image(
                texture.id(),
                egui::Rect::from_min_size(rect.min, egui::vec2(taille.x, taille.x * 0.75)),
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        reponse
    }

    /// List of names, for an installation without thumbnails.
    fn show_preset_list(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        presets: &[filter::BuiltinPreset],
        current: Option<usize>,
    ) {
        let mut selected = current;
        ui.horizontal(|ui| {
            ui.label("Preset");
            egui::ComboBox::from_id_salt("preset_combo")
                .selected_text(match current {
                    None => pointimg::frontend::t("Personnalisé").to_string(),
                    Some(index) => presets[index].name.to_string(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut selected, None, pointimg::frontend::t("Personnalisé"));
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
    }

    fn show_algorithm_section(&mut self, ui: &mut egui::Ui) {
        // The order comes from `frontend::ALGORITHMS`, shared with GTK — only
        // the list is common, the labels stay specific to egui. The fallback is
        // explicit rather than an `unwrap_or("Voronoi")` that would hide an algo
        // missing from the table.
        let current = algorithm_label(self.params.algorithm);
        ui.horizontal(|ui| {
            ui.label("Placement");
            egui::ComboBox::from_id_salt("algo_combo")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for algo in frontend::ALGORITHMS {
                        ui.selectable_value(
                            &mut self.params.algorithm,
                            algo,
                            algorithm_label(algo),
                        )
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
        slider_number_row(
            ui,
            pointimg::frontend::t("Sensibilité variance"),
            &mut self.params.variance_sensitivity,
            0.0..=1.0,
            false,
            pointimg::frontend::t("0 = répartition uniforme · 1 = détail maximal"),
        );
        if variance_before != self.params.variance_sensitivity {
            // The density map depends on this single setting: we recompute it
            // without restarting the filter — but at the same time as it (the
            // debounce), otherwise a slider drag would launch a compute per
            // pixel.
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
        // Two-handle slider: the range is set with a single gesture, and the
        // order of the ends cannot be inverted.
        range_slider(
            ui,
            "rayon",
            &mut self.params.min_radius_ratio,
            &mut self.params.max_radius_ratio,
            Bounds::both(0.0005..=1.0),
            Curve::Log,
            pointimg::frontend::t("Rayon min et max, en fraction du plus petit côté"),
        );
        if self.params.min_radius_ratio > self.params.max_radius_ratio {
            std::mem::swap(
                &mut self.params.min_radius_ratio,
                &mut self.params.max_radius_ratio,
            );
        }

        slider_number_row(
            ui,
            "Boost zones uniformes",
            &mut self.params.max_boost,
            1.0..=8.0,
            // Logarithmic, as in the GTK interface: `max_boost` is a
            // multiplier, whose perception is logarithmic. Going from 1 to 2
            // must cost the same gesture as from 4 to 8.
            true,
            pointimg::frontend::t(
                "Agrandit les points dans les zones sans détail. 1 = aucun agrandissement.",
            ),
        );

        match self.params.algorithm {
            Algorithm::Grid => {
                // 2 → 8 192: logarithmic, otherwise the small grids
                // occupy a single pixel of the track.
                let mut cols = self.params.cols as usize;
                slider_int_row(
                    ui,
                    "Colonnes",
                    &mut cols,
                    2.0..=8192.0,
                    true,
                    pointimg::frontend::t("Nombre de cellules de la grille"),
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
                slider_int_row(
                    ui,
                    "Nombre de points",
                    &mut points,
                    10.0..=100_000.0,
                    true,
                    pointimg::frontend::t("De 10 à 100 000 points, en échelle logarithmique"),
                );
                self.params.num_points = points;
                if matches!(
                    self.params.algorithm,
                    Algorithm::Kmeans | Algorithm::Voronoi
                ) {
                    let mut iterations = self.params.iterations;
                    slider_int_row(
                        ui,
                        pointimg::frontend::t("Itérations"),
                        &mut iterations,
                        1.0..=100.0,
                        false,
                        pointimg::frontend::t(
                            "Passes de Lloyd. Au-delà de ~30, le gain devient marginal.",
                        ),
                    );
                    self.params.iterations = iterations;
                }
            }
            Algorithm::Halftone => unreachable!("{}", pointimg::frontend::t("traité plus haut")),
        }
    }

    fn show_shape_and_palette_section(&mut self, ui: &mut egui::Ui) {
        show_shape_selector(ui, &mut self.params.dot_shape);
        ui.checkbox(
            &mut self.params.palette_size.is_some(),
            egui::RichText::new(pointimg::frontend::t("Palette réduite")).size(13.0),
        )
        .on_hover_text(pointimg::frontend::t(
            "Quantifie l'image vers quelques couleurs (effet sérigraphie)",
        ))
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
            .on_hover_text(pointimg::frontend::t(
                "Alpha conservé à l'export PNG/WebP/TIFF",
            ))
            .changed()
        {
            self.on_params_changed(ctx);
        }
        if ui
            .checkbox(
                &mut self.params.gamma_correct,
                egui::RichText::new("Correction gamma").size(13.0),
            )
            .on_hover_text(pointimg::frontend::t(
                "Moyennes de couleur en espace linéaire : évite les mi-tons trop sombres",
            ))
            .changed()
        {
            self.on_params_changed(ctx);
        }

        ui.horizontal(|ui| {
            if ui
                .checkbox(
                    &mut self.params.rng_seed.is_some(),
                    egui::RichText::new(pointimg::frontend::t("Graine fixée")).size(13.0),
                )
                .on_hover_text(pointimg::frontend::t(
                    "Deux exécutions avec les mêmes réglages donnent le même résultat",
                ))
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
                    .on_hover_text(pointimg::frontend::t("Tirer une nouvelle graine"))
                    .clicked()
                {
                    self.vary_seed(ctx);
                }
            }
        });
    }

    fn show_quality_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.label(pointimg::frontend::t("Aperçu"));
            let mut index = match self.quality {
                Quality::Half => 0,
                Quality::Quarter => 1,
                Quality::Full => 2,
            };
            let response = ui.selectable_value(&mut index, 0, "Rapide ½");
            if response.clicked() {
                self.set_quality(Quality::Half, ctx);
            }
            response.on_hover_text(pointimg::frontend::t(
                "Calcul sur une image à 50 % : très réactif",
            ));
            let response = ui.selectable_value(&mut index, 1, "Rapide ¼");
            if response.clicked() {
                self.set_quality(Quality::Quarter, ctx);
            }
            response.on_hover_text(pointimg::frontend::t(
                "Calcul sur une image à 25 % : le plus réactif",
            ));
            let response = ui.selectable_value(&mut index, 2, "Plein");
            if response.clicked() {
                self.set_quality(Quality::Full, ctx);
            }
            response.on_hover_text(pointimg::frontend::t("Toujours en pleine résolution"));
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
                .on_hover_text(pointimg::frontend::t(
                    "Rosette classique Cyan / Magenta / Jaune / Noir, à 15°/75°/0°/45°",
                ))
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
                .on_hover_text(pointimg::frontend::t(
                    "Les N couleurs dominantes de l'image deviennent chacune une encre",
                ))
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
                // `slider_int_row` and not `int_number_row`: the latter also
                // drew a second slider without a label, and "Channels"
                // appeared twice in the same section.
                slider_int_row(
                    ui,
                    "Canaux",
                    &mut channels,
                    2.0..=32.0,
                    false,
                    pointimg::frontend::t("Nombre d'encres retenues parmi les couleurs dominantes"),
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
                ui.small(pointimg::frontend::t("(sélectionnez CMJN ou Dominant)"));
            }
        }
        ui.horizontal(|ui| {
            ui.label("Screening");
            ui.selectable_value(&mut self.params.screening, Screening::Am, "AM (grille)")
                .on_hover_text(pointimg::frontend::t(
                    "Trame régulière : le rayon du dot code la couverture d'encre",
                ));
            ui.selectable_value(&mut self.params.screening, Screening::Fm, "FM (blue noise)")
                .on_hover_text(pointimg::frontend::t(
                    "Trame stochastique : la densité des dots code la couverture d'encre",
                ));
        });
        // 4 → 400: logarithmic, the range is too wide for a linear travel.
        slider_number_row(
            ui,
            pointimg::frontend::t("Fréquence de trame"),
            &mut self.params.halftone_frequency,
            4.0..=400.0,
            true,
            pointimg::frontend::t("Cellules de la trame par unité de longueur"),
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
        // Same pair of ends as for the points: two-handle slider, logarithmic
        // scale (the radius varies by a factor of 100).
        range_slider(
            ui,
            "halftone_rayon",
            &mut self.params.halftone_min_radius_ratio,
            &mut self.params.halftone_max_dot_ratio,
            Bounds {
                low: 0.0..=1.0,
                high: 0.05..=10.0,
            },
            Curve::Sqrt,
            pointimg::frontend::t("Rayon min et max des dots d'encre"),
        );
    }

    /// Loads (only once) the egui textures matching the images.
    /// Loads once and for all the thumbnails of the presets.
    ///
    /// An unreadable file gives `None`: if none of them works, the grid will
    /// switch to the list of names, and the panel will have lost nothing.
    fn ensure_preset_thumbnails(&mut self, ctx: &egui::Context) {
        if self.preset_thumbnails.is_some() {
            return;
        }
        let textures = filter::builtin_presets()
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let chemin = frontend::preset_thumbnail(index)?;
                match image::open(&chemin) {
                    Ok(img) => Some(ctx.load_texture(
                        format!("preset_{index}"),
                        super::convert::rgba_to_color_image_checker(&img.to_rgba8()),
                        egui::TextureOptions::NEAREST,
                    )),
                    Err(e) => {
                        log::warn!("vignette de preset illisible ({}) : {e}", chemin.display());
                        None
                    }
                }
            })
            .collect();
        self.preset_thumbnails = Some(textures);
    }

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

    /// Moves to the next (`Tab`) or previous (`Shift+Tab`) display mode.
    ///
    /// `rem_euclid` makes it loop in both directions: going from the last mode
    /// to the first, and from the first to the last.
    pub(crate) fn cycle_view(&mut self, delta: i32) {
        let index = view_index(self.view_mode) as i32;
        let next = (index + delta).rem_euclid(pointimg::frontend::VIEW_COUNT as i32) as u32;
        self.view_mode = view_from_index(next);
    }

    // ─── Central area ────────────────────────────────────────────────────────

    fn show_central_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, computing: bool) {
        egui::CentralPanel::default().show(ui, |ui| {
            self.ensure_textures(ctx);

            ui.horizontal_wrapped(|ui| {
                ui.label("Affichage");
                for (label, mode) in views() {
                    ui.selectable_value(&mut self.view_mode, mode, label);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("1:1")
                        .on_hover_text(pointimg::frontend::t("Revenir à 100 % (1)"))
                        .clicked()
                    {
                        self.set_zoom(1.0);
                    }
                    let fit = ui
                        .selectable_label(self.zoom_fit, "Ajuster")
                        .on_hover_text(pointimg::frontend::t("Ajuster l'aperçu à la fenêtre (0)"));
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
                self.toasts.show(ui);
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
            // Remaining height for the previews (toolbar already laid out).
            let height = (ui.available_height() - 8.0).max(120.0);
            match self.view_mode {
                ViewMode::Compare => self.show_compare(ui, stale),
                ViewMode::Side => {
                    // Two halves *plus* the separator must fit into the
                    // available width, otherwise the second panel is cropped.
                    // Below 640 px (i.e. ~300 px per image), we
                    // stack them: two images side by side become unreadable.
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
                                caption: pointimg::frontend::t("Résultat"),
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
                            caption: pointimg::frontend::t("Résultat"),
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
                        caption: pointimg::frontend::t("Résultat"),
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
                        caption: pointimg::frontend::t("Carte de densité"),
                        texture: density.as_ref(),
                        size: egui::vec2(ui.available_width(), height),
                        zoom,
                        zoom_fit,
                        spinning: false,
                    },
                ),
            }
            // The notification stack draws over the preview: it must depend
            // neither on the displayed view nor on the loaded image.
            self.toasts.show(ui);
        });
    }

    /// Before/after comparison: source on the left of a handle, result on the
    /// right, in the same area — the comparison bears on the same pixel area.
    fn show_compare(&mut self, ui: &mut egui::Ui, stale: bool) {
        let (Some(source), Some(result)) = (&self.src_texture, &self.result_texture) else {
            ui.centered_and_justified(|ui| {
                if self.computing.load(Ordering::Relaxed) {
                    ui.spinner();
                }
                ui.label(frontend::msg::computing());
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
                egui::RichText::new(pointimg::frontend::t(
                    "glissez le curseur (double-clic : centré)",
                ))
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

                // Source on the left, result on the right, cut at the handle.
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
                // Double stroke, dark then light: the handle and the separation
                // line were invisible on a white-background render, even though
                // they are the sole reference of the comparison. The halo
                // detaches them from any background, the light line detaches
                // them from the halo.
                painter.line_segment(
                    [
                        egui::pos2(cursor_x, rect.min.y),
                        egui::pos2(cursor_x, rect.max.y),
                    ],
                    egui::Stroke::new(4.0, egui::Color32::from_black_alpha(140)),
                );
                painter.line_segment(
                    [
                        egui::pos2(cursor_x, rect.min.y),
                        egui::pos2(cursor_x, rect.max.y),
                    ],
                    egui::Stroke::new(1.5, egui::Color32::WHITE),
                );
                let center = egui::pos2(cursor_x, rect.center().y);
                painter.circle_filled(center, 12.0, egui::Color32::from_black_alpha(140));
                painter.circle_filled(center, 10.0, egui::Color32::WHITE);
                if stale {
                    // Opaque plate behind the text: bare white, it was
                    // unreadable on a light sky.
                    let text = frontend::msg::stale();
                    let galley = ui.painter().layout_no_wrap(
                        text.to_owned(),
                        egui::FontId::proportional(12.0),
                        egui::Color32::WHITE,
                    );
                    let pad = egui::vec2(8.0, 5.0);
                    let anchor = egui::pos2(rect.min.x + 10.0, rect.min.y + 10.0);
                    let plate = egui::Rect::from_min_size(anchor, galley.size() + 2.0 * pad);
                    painter.rect_filled(
                        plate,
                        egui::CornerRadius::same(4),
                        egui::Color32::from_black_alpha(170),
                    );
                    painter.galley(plate.min + pad, galley, egui::Color32::WHITE);
                }
                // Drag chevrons, on the white disk.
                for direction in [-1.0f32, 1.0] {
                    painter.line_segment(
                        [
                            egui::pos2(cursor_x + direction * 5.0, center.y),
                            egui::pos2(cursor_x + direction * 1.5, center.y + direction * 4.0),
                        ],
                        egui::Stroke::new(1.6, egui::Color32::DARK_GRAY),
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

    /// Welcome state: something to do as long as no image is loaded.
    fn show_empty_state(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.vertical_centered(|ui| {
            ui.add_space((ui.available_height() * 0.18).max(24.0));
            ui.heading("Aucune image");
            ui.label(pointimg::frontend::t(
                "Glissez-déposez un fichier dans la fenêtre, ou ouvrez-en un.",
            ));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .button(pointimg::frontend::t("Ouvrir une image…"))
                    .clicked()
                {
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
            self.show_recent_images(ui, ctx);
        });
    }

    /// Recently opened images, under the buttons of the welcome state.
    ///
    /// GTK reads the same information in `GtkRecentManager`, the desktop's
    /// list; egui has no equivalent, its own lives in the preferences file.
    /// Same intent, each frontend its own mechanism.
    fn show_recent_images(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(prefs) = self.prefs.as_ref() else {
            return;
        };
        let recentes = prefs.recent();
        if recentes.is_empty() {
            return;
        }
        ui.add_space(18.0);
        ui.label(
            egui::RichText::new(pointimg::frontend::t("Images récentes"))
                .strong()
                .size(13.0),
        );
        ui.label(
            egui::RichText::new(pointimg::frontend::t("Reprendre où vous en étiez"))
                .size(11.5)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(2.0);
        // A narrow column: the full paths would make the list unreadable, and
        // the folder name is enough to tell two homonymous images apart.
        egui::Grid::new("recent_images")
            .num_columns(1)
            .spacing([0.0, 2.0])
            .max_col_width(280.0)
            .show(ui, |ui| {
                for path in recentes {
                    let nom = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.display().to_string());
                    let dossier = path
                        .parent()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default();
                    let texte = egui::RichText::new(nom).size(12.5);
                    let sous_titre = egui::RichText::new(dossier)
                        .size(10.5)
                        .color(ui.visuals().weak_text_color());
                    // Two rows per image: the name, then its folder. egui has no
                    // button with a subtitle, the layout is done here.
                    let reponse = ui
                        .vertical(|ui| {
                            ui.set_min_width(280.0);
                            ui.label(texte);
                            ui.label(sous_titre);
                        })
                        .response
                        .interact(egui::Sense::click());
                    if reponse.clicked() {
                        self.load_image(path, ctx);
                    }
                    ui.end_row();
                }
            });
    }

    // ─── Status bar ──────────────────────────────────────────────────────────

    fn show_status_bar(&mut self, ui: &mut egui::Ui, computing: bool) {
        let stale = self.result_is_stale();
        let timing = self.last_compute_ms;
        let quick = !self.quality.is_full() && self.src_rgb.is_some();
        let status = self.status.clone();
        // Export size: measured on the definitive result, real encoding.
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
                    // Drawing order: progress bar then spinner. In the reverse
                    // order the two overlap.
                    let (done, total) = *self.progress.lock().unwrap_or_else(|e| e.into_inner());
                    if total > 0 {
                        // The default fill color is too dark on the panel
                        // background: we take the accent of the theme.
                        let fill = ui.visuals().selection.bg_fill;
                        ui.add(
                            egui::ProgressBar::new(done as f32 / total as f32)
                                .desired_width(160.0)
                                .fill(fill),
                        );
                        ui.spinner();
                        ui.small(format!("{done}/{total}"));
                    } else {
                        ui.label(frontend::msg::computing());
                    }
                } else {
                    ui.label(&status);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if !sizes_label.is_empty() {
                        let label = ui.small(format!("Export : {sizes_label}"));
                        if let Some(comparison) = &sizes_comparison {
                            label.on_hover_text(format!(
                                "{comparison}\n{}",
                                pointimg::export::ExportSizes::explanation()
                            ));
                        }
                    } else if sizes_measuring {
                        ui.small("Export : mesure…");
                    }
                    if quick && stale {
                        ui.small(frontend::msg::finalizing());
                    } else if quick {
                        ui.small(pointimg::frontend::t("aperçu rapide"))
                            .on_hover_text(
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Preview panel: frame, caption, image with zoom and scrolling.
/// A preview panel: what is needed to draw it in a given space.
///
/// Gathered into a structure rather than in eight parameters: the allocated
/// space, the texture and the zoom state always travel together.
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
    // The space is imposed by the caller: in a horizontal layout,
    // `available_width()` is the remaining width *for each child*, so one
    // child would take all the space and expel the other.
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

/// Subgroup title, inside a collapsible section.
fn group_title(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.strong(title);
    ui.separator();
}

/// Collapsible section of the settings panel.
///
/// The panel lines up about thirty settings in a single column: without
/// folding, one has to walk through it entirely every time one looks for an
/// option. `CollapsingState` is stored in egui's memory, which eframe
/// persists from one session to the next — the section open at the previous
/// launch still is.
///
/// The folding is purely visual: nothing is hidden on the engine side, and the
/// GTK interface keeps its own `PreferencesGroup` — the two frontends stay
/// independent, including on this point.
fn section<R>(
    ui: &mut egui::Ui,
    title: &str,
    id_salt: &'static str,
    open_by_default: bool,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::CollapsingResponse<R> {
    // The header is the only navigation reference of the panel: it must stand
    // out from the body of the settings, without however looking like a window
    // title. 15 px in bold, against 13 px for a setting label.
    let header = egui::CollapsingHeader::new(egui::RichText::new(title).strong().size(15.0))
        .id_salt(id_salt)
        .default_open(open_by_default);
    header.show(ui, add)
}

impl App {
    /// Ephemeral message: the status line only described it for the duration of
    /// a frame, and disappeared entirely during a compute.
    pub(crate) fn notify(&mut self, text: impl Into<String>) {
        self.toasts.push(text, super::notify::ToastKind::Info);
    }

    /// Error message, with a longer reading duration.
    pub(crate) fn notify_error(&mut self, text: impl Into<String>) {
        self.toasts.push(text, super::notify::ToastKind::Error);
    }
}

/// Row "label + slider with integrated value".
///
/// One row per setting: input field and slider merged, `egui::Slider`
/// displaying its value *in* the rail, enterable by click.
///
/// The layout is `[label] [rail + value]`, like the input rows:
/// `Slider::text` would have put the label *to the right* of the rail, and the
/// panel would have displayed two conventions side by side. The rail takes the
/// remaining space rather than the 320 px of the style, without which it would
/// fix the panel minimum again.
fn slider_number_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    hint: &str,
) -> egui::Response {
    let decimals = display_decimals(*value, &range);
    let step = 10f64.powi(-(decimals as i32));
    let range_label = format!("{:.4} – {:.4}", *range.start(), range.end());
    let rail = (ui.available_width() * 0.55).max(80.0);
    ui.spacing_mut().slider_width = rail;
    let rail = (ui.available_width() * 0.55).max(80.0);
    ui.spacing_mut().slider_width = rail;
    // The response of the slider is captured: that of `ui.horizontal` is that
    // of the last widget drawn, and what we want to know is whether the *value*
    // moved.
    let mut curseur = None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(13.0));
        curseur = Some(
            ui.add(
                egui::Slider::new(value, range.clone())
                    .show_value(true)
                    .logarithmic(logarithmic)
                    .trailing_fill(true)
                    .step_by(step)
                    .min_decimals(decimals)
                    .max_decimals(decimals),
            ),
        );
    });
    let msg = pointimg::frontend::t("le curseur vient d'être dessiné");
    let reponse = curseur.expect(msg);
    let reponse = reponse.on_hover_text(
        pointimg::frontend::t("{hint}\nPlage autorisée : {range_label}")
            .replace("{hint}", hint)
            .replace("{range_label}", &range_label),
    );
    if reponse.changed() {
        // The step applies to the drag, not to the written value: without this
        // rounding the display carried the stray decimals of the pointer.
        *value = round_to(*value, decimals);
    }
    reponse
}

/// Integer variant of [`slider_number_row`]: the value is rounded to
/// the unit after each gesture.
fn slider_int_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    hint: &str,
) -> egui::Response {
    let mut miroir = *value as f32;
    let reponse = slider_number_row(ui, label, &mut miroir, range, logarithmic, hint);
    let arrondi = miroir.round().max(1.0) as usize;
    if arrondi != *value {
        *value = arrondi;
    }
    reponse
}

/// Row "label + direct entry": the real range (that of
/// `validate_params`) is given as a tooltip, plus a unit indication.
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
    let tooltip = match hint {
        Some(hint) => pointimg::frontend::t("{hint}\nPlage autorisée : {range_label}")
            .replace("{hint}", hint)
            .replace("{range_label}", &range_label),
        None => pointimg::frontend::t("Plage autorisée : {range_label}")
            .replace("{range_label}", &range_label),
    };
    // egui picks the display precision according to the distance from the
    // pointer to the screen edge: near an edge, "Min radius" would display
    // "0.000" then the entry would be rewritten to 0 on loss of focus. We
    // therefore impose the decimals useful to the setting (see
    // `display_decimals`).
    let decimals = display_decimals(*value, &range);
    // The field is sized to its content, not to the remaining width: without
    // that bound, `DragValue` stretched and the panel could not go below
    // ~440 px — it then took half of a 900 px window.
    let field_width = 56.0 + 7.0 * decimals as f32;
    // The inner `right_to_left` returns the response of the field, which the
    // row gives back: callers test `.changed()` to detect an entry.
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(13.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // `DragValue` has no step in 0.36: the speed sets the wheel
            // increment, direct entry accepts any value.
            ui.add_sized(
                [field_width, ui.spacing().interact_size.y],
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
    })
    .inner
}

/// Dithering algorithm selector, with the help matching each one.
fn show_dithering_selector(ui: &mut egui::Ui, dithering: &mut Dithering) {
    let choices: [(Dithering, &str, &str); 5] = [
        (
            Dithering::Off,
            pointimg::frontend::t("Aucun"),
            pointimg::frontend::t("Palette franche, sans motif"),
        ),
        (
            Dithering::FloydSteinberg,
            "Floyd-Steinberg",
            pointimg::frontend::t(
                "Diffusion de l'erreur : dégradés préservés, motif légèrement « verreux »",
            ),
        ),
        (
            Dithering::Riemersma,
            "Riemersma",
            pointimg::frontend::t("Diffusion à queue longue : moins de vers, motif plus lisse"),
        ),
        (
            Dithering::Bayer4,
            "Bayer 4×4",
            pointimg::frontend::t("Tramage ordonné : rendu « pixel art », sans diffusion d'erreur"),
        ),
        (
            Dithering::Bayer8,
            "Bayer 8×8",
            pointimg::frontend::t("Tramage ordonné plus fin : motif de trame 8×8"),
        ),
    ];
    let (label, hint) = choices
        .iter()
        .find(|(mode, _, _)| *mode == *dithering)
        .map(|(_, label, hint)| (*label, *hint))
        .unwrap_or((pointimg::frontend::t("Inconnu"), ""));
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Dithering").size(13.0));
        let response = egui::ComboBox::from_id_salt("dithering_combo")
            .selected_text(label)
            .show_ui(ui, |ui| {
                for (mode, name, help) in choices {
                    ui.selectable_value(dithering, mode, name)
                        .on_hover_text(help);
                }
            });
        // The help of the chosen algorithm, then its size warning if there is
        // one (text shared by both interfaces, see `size_caveat`).
        let text = match dithering.size_caveat() {
            Some(caveat) => format!("{hint}\n\n{caveat}"),
            None => hint.to_string(),
        };
        response.response.on_hover_text(text);
    });
    // The warning is also visible without hovering: it is information about
    // the file size, not an interface detail. Short version, the detail
    // remaining in the tooltip.
    if let Some(short) = dithering.size_caveat_short() {
        ui.add_space(2.0);
        ui.add(
            egui::Label::new(
                egui::RichText::new(short)
                    .size(11.0)
                    .color(ui.visuals().warn_fg_color),
            )
            .wrap(),
        )
        .on_hover_text(dithering.size_caveat().unwrap_or(short));
    }
}

/// Row "label + direct entry + slider" for an integer setting: entry and
/// dragging are both rounded to the unit.
/// Point shape selector.
fn show_shape_selector(ui: &mut egui::Ui, shape: &mut DotShape) {
    let current = match shape {
        DotShape::Circle => "Cercle",
        DotShape::Square => pointimg::frontend::t("Carré"),
        DotShape::Ellipse { .. } => "Ellipse",
        DotShape::RegularPolygon { .. } => "Polygone",
    };
    ui.horizontal(|ui| {
        ui.label("Forme");
        for (label, variant) in [
            ("Cercle", DotShape::Circle),
            (pointimg::frontend::t("Carré"), DotShape::Square),
        ] {
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
            if number_row(ui, pointimg::frontend::t("Côtés"), &mut value, 3.0..=12.0).changed() {
                *sides = value.round().clamp(3.0, 12.0) as u8;
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Tab` moves forward and `Shift+Tab` backward, looping in both
    /// directions — that is what the GTK shortcuts window announces on both
    /// sides.
    #[test]
    fn le_cycle_des_apercus_boucle_dans_les_deux_sens() {
        let total = pointimg::frontend::VIEW_COUNT as i32;
        let modes: Vec<ViewMode> = (0..pointimg::frontend::VIEW_COUNT)
            .map(pointimg::frontend::view_from_index)
            .collect();
        assert_eq!(modes.len() as i32, total);

        // A step then an inverse step comes back to the starting point.
        for mode in &modes {
            let mut app = App {
                view_mode: *mode,
                ..App::default()
            };
            app.cycle_view(1);
            app.cycle_view(-1);
            assert_eq!(app.view_mode, *mode, "un aller-retour doit revenir");
        }
        // The last mode loops onto the first, and the reverse.
        let mut app = App {
            view_mode: modes[0],
            ..App::default()
        };
        app.cycle_view(-1);
        assert_eq!(app.view_mode, modes[modes.len() - 1]);
        app.cycle_view(1);
        assert_eq!(app.view_mode, modes[0]);
        // Both directions cover the whole list without repeating it.
        let mut vus = Vec::new();
        app.cycle_view(1);
        for _ in 0..total {
            vus.push(app.view_mode);
            app.cycle_view(1);
        }
        assert_eq!(vus.len(), total as usize);
        for mode in &modes {
            assert!(vus.contains(mode), "{mode:?} n'a pas été parcouru");
        }
    }
}
