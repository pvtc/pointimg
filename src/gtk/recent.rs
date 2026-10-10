//! Fichiers récents.
//!
//! GTK dispose de `GtkRecentManager`, qui alimente le menu « Documents
//! récents » de toutes les applications GNOME et survit aux sessions : c'est
//! l'outil natif, et il est utilisé tel quel.
//!
//! egui n'a pas d'équivalent — pas de `GMenu`, pas de menu du bureau — donc
//! l'interface tient sa propre liste, dans le fichier de préférences qui
//! existe déjà (`prefs.rs`).
//!
//! Chaque front-end garde ainsi la manière de son toolkit ; seule l'intention
//! — proposer les images récentes — est commune.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use adw::prelude::*;

use crate::State;
use crate::actions;
use crate::controls::Widgets;

/// Nombre d'entrées proposées au maximum.
///
/// La liste vit dans l'état d'accueil, sous les boutons : quatre rangées
/// tiennent avec l'icône et le titre de la page d'accueil. Au-delà, il
/// faudrait un défilement — et un `ScrolledWindow` ne prend pas de hauteur
/// dans le `StatusPage` d'adwaita, il s'effondre.
const MAX_RECENT: usize = 4;

/// Ajoute une image aux documents récents du bureau.
pub(crate) fn remember(path: &Path) {
    let uri = gtk::gio::File::for_path(path).uri().to_string();
    gtk::RecentManager::default().add_item(&uri);
}

/// Chemins des images récentes, du plus récent au plus ancien.
///
/// Le gestionnaire du bureau peut contenir des documents sans rapport : on
/// ne garde que les images encore présentes sur le disque. `RecentInfo` ne
/// rend que l'URI, d'où la conversion — un chemin local, ce que fournit
/// toujours `file://`.
pub(crate) fn recent_images() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for item in gtk::RecentManager::default().items() {
        if paths.len() == MAX_RECENT {
            break;
        }
        let uri = item.uri().to_string();
        let Some(path) = uri
            .strip_prefix("file://")
            .map(|raw| PathBuf::from(percent_decode(raw)))
        else {
            continue;
        };
        if pointimg::frontend::is_supported_image(&path) && path.is_file() {
            paths.push(path);
        }
    }
    paths
}

/// Décode les séquences `%XX` d'un URI `file://` : un chemin contenant un
/// espace ou une lettre accentuée en contient.
fn percent_decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(byte) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Propose les images récentes dans l'état d'accueil.
///
/// `adw::StatusPage` n'accepte qu'un titre, une description et une icône : les
/// rangées vont donc dans un `PreferencesGroup` placé sous elle, ce qui reste
/// dans l'adwaita et évite un dialogue séparé au premier lancement.
pub(crate) fn build_recent_widget(
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) -> Option<gtk::Widget> {
    let images = recent_images();
    if images.is_empty() {
        return None;
    }
    let group = adw::PreferencesGroup::new();
    group.set_title("Images récentes");
    group.set_description(Some("Reprendre où vous en étiez"));
    for path in images {
        let row = adw::ActionRow::new();
        // Le nom du fichier, pas son chemin complet : la liste doit rester
        // lisible, et deux images homonymes se distinguent par leur dossier.
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        row.set_title(&name);
        if let Some(parent) = path.parent() {
            row.set_subtitle(&parent.display().to_string());
        }
        row.set_activatable(true);
        let (s, w) = (state.clone(), widgets.clone());
        let target = path.clone();
        row.connect_activated(move |_| {
            actions::load_image(target.clone(), &s, &w);
        });
        group.add(&row);
    }
    Some(group.upcast())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn les_uris_de_fichier_sont_decodes() {
        assert_eq!(percent_decode("/tmp/photo.png"), "/tmp/photo.png");
        assert_eq!(percent_decode("/tmp/une%20photo.png"), "/tmp/une photo.png");
        assert_eq!(percent_decode("/tmp/%C3%A9t%C3%A9.png"), "/tmp/été.png");
        // Une séquence incomplète est laissée telle quelle plutôt que perdue.
        assert_eq!(percent_decode("/tmp/100%.png"), "/tmp/100%.png");
    }
}
