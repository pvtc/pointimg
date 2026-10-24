//! "Raccourcis clavier" window.
//!
//! `adw::ShortcutsWindow` is available from libadwaita 1.8; the page is
//! therefore built by hand (`adw::ShortcutsItem` not being available in 1.4
//! either), which gives the same rendering.

use adw::prelude::*;

/// Builds the content of the shortcuts window.
pub(crate) fn build_page() -> impl IsA<gtk::Widget> {
    let page = adw::PreferencesPage::new();
    #[allow(clippy::type_complexity)]
    let sections: &[(&str, &[(&str, &[&str])])] = &[
        (
            pointimg::frontend::t("Fichier"),
            &[
                (pointimg::frontend::t("Ouvrir une image"), &["Ctrl", "O"]),
                (
                    pointimg::frontend::t("Ouvrir un dossier"),
                    &["Ctrl", "Shift", "O"],
                ),
                (
                    pointimg::frontend::t("Image précédente / suivante"),
                    &["←", "→"],
                ),
                (pointimg::frontend::t("Exporter l'image…"), &["Ctrl", "S"]),
                (
                    pointimg::frontend::t("Exporter en SVG…"),
                    &["Ctrl", "Shift", "S"],
                ),
                (
                    pointimg::frontend::t("Exporter sans dialogue"),
                    &["Ctrl", "E"],
                ),
                (pointimg::frontend::t("Copier l'image"), &["Ctrl", "C"]),
            ],
        ),
        (
            pointimg::frontend::t("Réglages"),
            &[
                (
                    pointimg::frontend::t("Recalculer"),
                    &[pointimg::frontend::t("Espace")],
                ),
                (
                    pointimg::frontend::t("Preset précédent / suivant"),
                    &["Alt", "↑ / ↓"],
                ),
                (pointimg::frontend::t("Preset suivant"), &["Ctrl", "P"]),
                (
                    pointimg::frontend::t("Annuler / Rétablir"),
                    &["Ctrl", "Z", "Ctrl", "Y"],
                ),
                (
                    pointimg::frontend::t("Réinitialiser les réglages"),
                    &["Ctrl", "R"],
                ),
            ],
        ),
        (
            pointimg::frontend::t("Affichage"),
            &[
                (
                    pointimg::frontend::t("Aperçu suivant / précédent"),
                    &["Tab"],
                ),
                (pointimg::frontend::t("Zoomer / Dézoomer"), &["+", "−"]),
                (pointimg::frontend::t("Ajuster à la fenêtre"), &["0"]),
                ("100 %", &["1"]),
                (
                    pointimg::frontend::t("Zoom à la souris"),
                    &["Ctrl", pointimg::frontend::t("molette")],
                ),
                (pointimg::frontend::t("Plein écran"), &["F11"]),
            ],
        ),
        (
            pointimg::frontend::t("Calcul"),
            &[
                (
                    pointimg::frontend::t("Annuler le calcul"),
                    &[pointimg::frontend::t("Échap")],
                ),
                (
                    pointimg::frontend::t("Recalculer en pleine résolution"),
                    &[pointimg::frontend::t("Espace")],
                ),
            ],
        ),
    ];
    for (title, shortcuts) in sections {
        let group = adw::PreferencesGroup::new();
        group.set_title(title);
        for (label, keys) in *shortcuts {
            let row = adw::ActionRow::new();
            row.set_title(label);
            row.set_subtitle(&keys.join(" "));
            row.set_activatable(false);
            group.add(&row);
        }
        page.add(&group);
    }
    page
}
