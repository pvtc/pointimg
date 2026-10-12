//! Command palette (`Ctrl+K`).
//!
//! The GTK interface has a main menu and a shortcuts window
//! (`src/gtk/wire.rs`, `src/gtk/shortcuts.rs`). egui had neither: its
//! seventeen shortcuts were only discoverable by hovering, and the density map
//! like full screen were only reachable through the `Tab` cycle or not at all.
//!
//! The palette gives access to every action from a single field. It stays an
//! egui-specific implementation — `egui::Window`, no `adw::Dialog`, no
//! `GMenu` — so that both interfaces keep using their own toolkit.

use eframe::egui;

/// An action that can be run from the palette.
///
/// `enabled` decides the appearance: an unavailable action stays visible but
/// greyed out, so that the user discovers what exists.
pub(crate) struct Command {
    pub label: &'static str,
    pub hint: &'static str,
    pub enabled: bool,
    pub run: fn(&mut crate::App, &egui::Context),
}

/// All the commands, in display order.
///
/// Each entry carries its shortcut in `hint`: it is the only way, in egui, to
/// make the shortcuts discoverable.
pub(crate) fn all(app: &crate::App) -> Vec<Command> {
    let has_final = app.has_final_result();
    let has_dots = app.src_rgb.is_some();
    let in_folder = app.folder_images.len() > 1;
    let computing = app.computing.load(std::sync::atomic::Ordering::Relaxed);
    vec![
        Command {
            label: "Ouvrir une image…",
            hint: "Ctrl+O",
            enabled: true,
            run: |app, ctx| app.choose_image(ctx),
        },
        Command {
            label: "Ouvrir un dossier d'images…",
            hint: "Ctrl+Maj+O",
            enabled: true,
            run: |app, ctx| app.choose_folder(ctx),
        },
        Command {
            label: "Image précédente",
            hint: "←",
            enabled: in_folder,
            run: |app, ctx| app.step_folder(-1, ctx),
        },
        Command {
            label: "Image suivante",
            hint: "→",
            enabled: in_folder,
            run: |app, ctx| app.step_folder(1, ctx),
        },
        Command {
            label: "Exporter l'image…",
            hint: "Ctrl+S",
            enabled: has_final,
            run: |app, _| app.choose_export(),
        },
        Command {
            label: "Exporter en SVG…",
            hint: "Ctrl+Maj+S",
            enabled: has_dots,
            run: |app, _| app.choose_svg_export(),
        },
        Command {
            label: "Exporter à côté de la source",
            hint: "Ctrl+E",
            enabled: has_final,
            run: |app, _| app.quick_export(),
        },
        Command {
            label: "Copier dans le presse-papiers",
            hint: "Ctrl+C",
            enabled: has_final,
            run: |app, ctx| app.copy_to_clipboard(ctx),
        },
        Command {
            label: "Recalculer",
            hint: "Espace",
            enabled: app.src_rgb.is_some(),
            run: |app, ctx| app.request_full_compute(ctx),
        },
        Command {
            label: "Annuler le calcul",
            hint: "Échap",
            enabled: computing,
            run: |app, _| {
                app.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                app.notify("Annulation…");
            },
        },
        Command {
            label: "Annuler la dernière modification",
            hint: "Ctrl+Z",
            enabled: !app.history.is_empty(),
            run: |app, ctx| app.undo(ctx),
        },
        Command {
            label: "Rétablir",
            hint: "Ctrl+Y",
            enabled: !app.future.is_empty(),
            run: |app, ctx| app.redo(ctx),
        },
        Command {
            label: "Réinitialiser les réglages",
            hint: "Ctrl+R",
            enabled: true,
            run: |app, ctx| app.reset_params(ctx),
        },
        Command {
            label: "Preset suivant",
            hint: "Ctrl+P",
            enabled: true,
            run: |app, ctx| app.cycle_preset(1, ctx),
        },
        Command {
            label: "Aperçu suivant",
            hint: "Tab",
            enabled: true,
            run: |app, _| app.cycle_view(1),
        },
        Command {
            label: "Aperçu précédent",
            hint: "Maj+Tab",
            enabled: true,
            run: |app, _| app.cycle_view(-1),
        },
        Command {
            label: "Carte de densité",
            hint: "",
            enabled: true,
            run: |app, _| app.view_mode = crate::ViewMode::DensityMap,
        },
        Command {
            label: "Zoomer",
            hint: "+",
            enabled: true,
            run: |app, _| {
                let zoom = app.current_zoom() * 1.25;
                app.set_zoom(zoom);
            },
        },
        Command {
            label: "Dézoomer",
            hint: "−",
            enabled: true,
            run: |app, _| {
                let zoom = app.current_zoom() / 1.25;
                app.set_zoom(zoom);
            },
        },
        Command {
            label: "Ajuster à la fenêtre",
            hint: "0",
            enabled: true,
            run: |app, _| app.zoom_fit = true,
        },
        Command {
            label: "100 %",
            hint: "1",
            enabled: true,
            run: |app, _| app.set_zoom(1.0),
        },
    ]
}

/// Draws the palette for one frame.
///
/// Returns `Some(text)` as long as it stays open, `None` as soon as a command
/// has been played or the user closed it (cross or `Esc`).
pub(crate) fn show(app: &mut crate::App, ctx: &egui::Context, mut query: String) -> Option<String> {
    // Enter triggers the first proposed command, as in every palette.
    let enter_pressed = ctx.input(|i| i.key_pressed(egui::Key::Enter));
    let commands = all(app);
    let filter = query.to_lowercase();
    let matching: Vec<&Command> = commands
        .iter()
        .filter(|command| {
            filter.is_empty()
                || command.label.to_lowercase().contains(&filter)
                || command.hint.to_lowercase().contains(&filter)
        })
        .collect();

    let mut open = true;
    let mut run: Option<fn(&mut crate::App, &egui::Context)> = None;
    // `open` is borrowed by `.open(&mut open)` for the whole drawing; closing
    // requested from the inside goes through a separate flag.
    let mut triggered = false;

    egui::Window::new("Commandes")
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, -60.0))
        .collapsible(false)
        .resizable(false)
        .default_width(430.0)
        .open(&mut open)
        .show(ctx, |ui| {
            let response = ui.add(
                egui::TextEdit::singleline(&mut query)
                    .hint_text("Rechercher une commande…")
                    .desired_width(f32::INFINITY),
            );
            // Focus follows the field on every frame: without that, one has to
            // click inside it once the palette is open.
            response.request_focus();
            ui.separator();
            if matching.is_empty() {
                ui.small("Aucune commande ne correspond.");
            } else {
                egui::ScrollArea::vertical()
                    .max_height(280.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (index, command) in matching.iter().enumerate() {
                            let label = if command.hint.is_empty() {
                                egui::RichText::new(command.label)
                            } else {
                                egui::RichText::new(format!(
                                    "{}    {}",
                                    command.label, command.hint
                                ))
                                .color(ui.visuals().weak_text_color())
                            };
                            let text = if command.enabled {
                                label
                            } else {
                                label.color(ui.visuals().weak_text_color())
                            };
                            let response = ui.add_enabled(command.enabled, egui::Button::new(text));
                            if response.clicked()
                                || (index == 0 && enter_pressed && command.enabled)
                            {
                                run = Some(command.run);
                                triggered = true;
                            }
                        }
                    });
                ui.separator();
                ui.weak("Entrée : première commande · Échap : fermer");
            }
        });

    if !open || triggered {
        return None;
    }
    if let Some(run) = run {
        run(app, ctx);
        return None;
    }
    Some(query)
}
