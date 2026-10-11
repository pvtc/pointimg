//! Confirmation avant d'écraser un fichier.
//!
//! Un export rapide écrit à côté de l'image source, sans que l'utilisateur
//! choisisse le chemin. Écraser ce fichier sans le lui dire serait une perte
//! silencieuse. Les exports choisis dans le dialogue n'ont pas besoin de cette
//! étape — le portail confirme déjà.
//!
//! `choose` est asynchrone, d'où la forme « pose la question, puis fait le
//! travail » au lieu d'un booléen de retour.

use std::path::Path;

use gtk::prelude::*;

use crate::controls::Widgets;

/// Les raccourcis d'état et de widgets, comme dans le reste du module GTK.
type Wi = std::rc::Rc<Widgets>;

/// Bouton « Annuler » — l'index 0, donc `cancel_button`.
const ANNULER: i32 = 0;
/// Bouton « Remplacer » — l'index 1.
const REMPLACER: i32 = 1;

/// Demande confirmation si `path` existe, puis appelle `then`.
///
/// Le fichier absent n'est pas confirmé — il n'y a rien à perdre. La fenêtre
/// introuvable non plus : mieux vaut exporter que bloquer sur un dialogue qu'on
/// ne peut pas poser.
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
        .message("Remplacer le fichier ?")
        .detail(format!(
            "« {} » existe déjà, son contenu sera remplacé.",
            path.display()
        ))
        .buttons(["Annuler", "Remplacer"])
        // « Annuler » est le bouton par défaut : valider par mégarde ne doit
        // jamais détruire le fichier.
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
