//! Confirmation before overwriting a file.
//!
//! A quick export writes next to the source image, without the user choosing
//! the path. Overwriting that file without telling them would be a silent
//! loss. The exports chosen in the dialog do not need this step — the portal
//! already confirms.
//!
//! `choose` is asynchronous, hence the "ask the question, then do the work"
//! shape instead of a boolean return value.

use std::path::Path;

use gtk::prelude::*;

use crate::controls::Widgets;

/// The state and widget type aliases, as in the rest of the GTK module.
type Wi = std::rc::Rc<Widgets>;

/// "Annuler" button — index 0, hence `cancel_button`.
const ANNULER: i32 = 0;
/// "Remplacer" button — index 1.
const REMPLACER: i32 = 1;

/// Asks for confirmation if `path` exists, then calls `then`.
///
/// A missing file is not confirmed — there is nothing to lose. Neither is an
/// unfindable window: exporting is better than blocking on a dialog that
/// cannot be shown.
pub(crate) fn confirm_overwrite<F>(widgets: &Wi, path: &Path, then: F)
where
    F: FnOnce() + 'static,
{
    if !path.exists() {
        then();
        return;
    }
    let fenetre = widgets
        .window
        .split
        .root()
        .and_downcast::<gtk::Window>()
        .or_else(|| {
            widgets
                .window
                .window_title
                .root()
                .and_downcast::<gtk::Window>()
        });
    let Some(fenetre) = fenetre else {
        then();
        return;
    };
    let dialogue = gtk::AlertDialog::builder()
        .message(pointimg::frontend::gettext("Remplacer le fichier ?"))
        .detail(
            pointimg::frontend::gettext("« {} » existe déjà, son contenu sera remplacé.")
                .replace("{}", &path.display().to_string()),
        )
        .buttons(["Annuler", "Remplacer"])
        // "Annuler" is the default button: confirming by accident must never
        // destroy the file.
        .cancel_button(ANNULER)
        .default_button(ANNULER)
        .build();
    dialogue.choose(
        Some(&fenetre),
        gtk::gio::Cancellable::NONE,
        move |reponse| {
            if reponse == Ok(REMPLACER) {
                then();
            }
        },
    );
}
