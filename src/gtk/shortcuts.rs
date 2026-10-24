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
            pointimg::frontend::gettext("Fichier"),
            &[
                (
                    pointimg::frontend::gettext("Ouvrir une image"),
                    &["Ctrl", "O"],
                ),
                (
                    pointimg::frontend::gettext("Ouvrir un dossier"),
                    &["Ctrl", "Shift", "O"],
                ),
                (
                    pointimg::frontend::gettext("Image précédente / suivante"),
                    &["←", "→"],
                ),
                (
                    pointimg::frontend::gettext("Exporter l'image…"),
                    &["Ctrl", "S"],
                ),
                (
                    pointimg::frontend::gettext("Exporter en SVG…"),
                    &["Ctrl", "Shift", "S"],
                ),
                (
                    pointimg::frontend::gettext("Exporter sans dialogue"),
                    &["Ctrl", "E"],
                ),
                (
                    pointimg::frontend::gettext("Copier l'image"),
                    &["Ctrl", "C"],
                ),
            ],
        ),
        (
            pointimg::frontend::gettext("Réglages"),
            &[
                (
                    pointimg::frontend::gettext("Recalculer"),
                    &[pointimg::frontend::gettext("Espace")],
                ),
                (
                    pointimg::frontend::gettext("Preset précédent / suivant"),
                    &["Alt", "↑ / ↓"],
                ),
                (
                    pointimg::frontend::gettext("Preset suivant"),
                    &["Ctrl", "P"],
                ),
                (
                    pointimg::frontend::gettext("Annuler / Rétablir"),
                    &["Ctrl", "Z", "Ctrl", "Y"],
                ),
                (
                    pointimg::frontend::gettext("Réinitialiser les réglages"),
                    &["Ctrl", "R"],
                ),
            ],
        ),
        (
            pointimg::frontend::gettext("Affichage"),
            &[
                (
                    pointimg::frontend::gettext("Aperçu suivant / précédent"),
                    &["Tab"],
                ),
                (
                    pointimg::frontend::gettext("Zoomer / Dézoomer"),
                    &["+", "−"],
                ),
                (pointimg::frontend::gettext("Ajuster à la fenêtre"), &["0"]),
                ("100 %", &["1"]),
                (
                    pointimg::frontend::gettext("Zoom à la souris"),
                    &["Ctrl", pointimg::frontend::gettext("molette")],
                ),
                (pointimg::frontend::gettext("Plein écran"), &["F11"]),
            ],
        ),
        (
            pointimg::frontend::gettext("Calcul"),
            &[
                (
                    pointimg::frontend::gettext("Annuler le calcul"),
                    &[pointimg::frontend::gettext("Échap")],
                ),
                (
                    pointimg::frontend::gettext("Recalculer en pleine résolution"),
                    &[pointimg::frontend::gettext("Espace")],
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
