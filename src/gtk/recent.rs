//! Recent files.
//!
//! GTK has `GtkRecentManager`, which feeds the "Documents récents" menu of
//! all GNOME applications and survives sessions: it is the native tool, and
//! it is used as is.
//!
//! egui has no equivalent — no `GMenu`, no desktop menu — so the interface
//! keeps its own list, in the preferences file that already exists
//! (`prefs.rs`).
//!
//! Each front-end thus keeps the way of its toolkit; only the intent —
//! offering the recent images — is shared.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use adw::prelude::*;

use crate::State;
use crate::actions;
use crate::controls::Widgets;

/// Maximum number of entries offered.
///
/// The list lives in the welcome state, under the buttons: four rows fit with
/// the icon and the title of the welcome page. Beyond that, scrolling would
/// be needed — and a `ScrolledWindow` takes no height inside the adwaita
/// `StatusPage`, it collapses.
const MAX_RECENT: usize = 4;

/// Adds an image to the desktop recent documents.
pub(crate) fn remember(path: &Path) {
    let uri = gtk::gio::File::for_path(path).uri().to_string();
    gtk::RecentManager::default().add_item(&uri);
}

/// Paths of the recent images, from the most recent to the oldest.
///
/// The desktop manager can contain unrelated documents: only the images
/// still present on disk are kept. `RecentInfo` only returns the URI, hence
/// the conversion — a local path, which is what `file://` always provides.
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

/// Decodes the `%XX` sequences of a `file://` URI: a path containing a space
/// or an accented letter contains some.
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

/// Offers the recent images in the welcome state.
///
/// `adw::StatusPage` only accepts a title, a description and an icon: the rows
/// therefore go into a `PreferencesGroup` placed under it, which stays within
/// adwaita and avoids a separate dialog on the first launch.
pub(crate) fn build_recent_widget(
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) -> Option<gtk::Widget> {
    let images = recent_images();
    if images.is_empty() {
        return None;
    }
    let group = adw::PreferencesGroup::new();
    group.set_title(pointimg::frontend::gettext("Images récentes"));
    group.set_description(Some(pointimg::frontend::gettext(
        "Reprendre où vous en étiez",
    )));
    for path in images {
        let row = adw::ActionRow::new();
        // The file name, not its full path: the list must stay readable,
        // and two images with the same name are told apart by their folder.
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
        // An incomplete sequence is left as it is rather than lost.
        assert_eq!(percent_decode("/tmp/100%.png"), "/tmp/100%.png");
    }
}
