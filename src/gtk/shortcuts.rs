//! Fenêtre « Raccourcis clavier ».
//!
//! `adw::ShortcutsWindow` est disponible à partir de libadwaita 1.8 ; on
//! construit donc la page à la main (`adw::ShortcutsItem` n'étant pas non plus
//! disponible en 1.4), ce qui donne le même rendu.

use adw::prelude::*;

/// Un raccourci : libellé + touches.
type Shortcut = (&'static str, &'static [&'static str]);

/// Les raccourcis affichés, regroupés par section.
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
            ("Preset suivant / précédent", &["Alt", "↑ / ↓"]),
            ("Choisir un preset", &["Ctrl", "P"]),
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
            ("Agrandir la fenêtre", &["Ctrl", "↑ / ↓"]),
        ],
    ),
];

/// Construit le contenu de la fenêtre des raccourcis.
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
