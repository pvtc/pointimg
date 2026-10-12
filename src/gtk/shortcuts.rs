//! "Raccourcis clavier" window.
//!
//! `adw::ShortcutsWindow` is available from libadwaita 1.8; the page is
//! therefore built by hand (`adw::ShortcutsItem` not being available in 1.4
//! either), which gives the same rendering.

use adw::prelude::*;

/// A shortcut: label + keys.
type Shortcut = (&'static str, &'static [&'static str]);

/// The displayed shortcuts, grouped by section.
const SECTIONS: &[(&str, &[Shortcut])] = &[
    (
        "Fichier",
        &[
            ("Ouvrir une image", &["Ctrl", "O"]),
            ("Ouvrir un dossier", &["Ctrl", "Shift", "O"]),
            ("Image précédente / suivante", &["←", "→"]),
            ("Exporter l'image…", &["Ctrl", "S"]),
            ("Exporter en SVG…", &["Ctrl", "Shift", "S"]),
            ("Exporter sans dialogue", &["Ctrl", "E"]),
            ("Copier l'image", &["Ctrl", "C"]),
        ],
    ),
    (
        "Réglages",
        &[
            ("Recalculer", &["Espace"]),
            ("Preset précédent / suivant", &["Alt", "↑ / ↓"]),
            ("Preset suivant", &["Ctrl", "P"]),
            ("Annuler / Rétablir", &["Ctrl", "Z", "Ctrl", "Y"]),
            ("Réinitialiser les réglages", &["Ctrl", "R"]),
        ],
    ),
    (
        "Affichage",
        &[
            ("Aperçu suivant / précédent", &["Tab"]),
            ("Zoomer / Dézoomer", &["+", "−"]),
            ("Ajuster à la fenêtre", &["0"]),
            ("100 %", &["1"]),
            ("Zoom à la souris", &["Ctrl", "molette"]),
            ("Plein écran", &["F11"]),
        ],
    ),
    (
        "Calcul",
        &[
            ("Annuler le calcul", &["Échap"]),
            ("Recalculer en pleine résolution", &["Espace"]),
        ],
    ),
];

/// Builds the content of the shortcuts window.
pub(crate) fn build_page() -> impl IsA<gtk::Widget> {
    let page = adw::PreferencesPage::new();
    for (title, shortcuts) in SECTIONS {
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
