//! Mesure de la taille d'export, dans un thread de travail.
//!
//! La mesure encode réellement l'image pour chaque format
//! ([`pointimg::export`]) : c'est une compression complète par format, donc
//! elle est faite à part, uniquement sur le résultat *définitif* (une taille
//! d'aperçu ne voudrait rien dire), invalidée dès que l'utilisateur revient à
//! régler, et déposée dans un emplacement que le thread principal reprend.
//!
//! Le résultat alimente une étiquette de la barre d'état et l'infobulle du
//! menu d'export : sans elle, le bénéfice central de `pointimg` — « la même
//! image, en beaucoup plus léger » — resterait invisible jusqu'au premier
//! enregistrement.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use pointimg::export::{ExportFormat, measure_export_sizes};

use crate::Msg;

use crate::State;

type St = Rc<RefCell<State>>;

/// Lance la mesure si le résultat affiché est définitif et pas déjà mesuré.
pub(crate) fn refresh(state: &St, tx: &async_channel::Sender<Msg>) {
    let (revision, source_size, transparent, bg, image) = {
        let st = state.borrow();
        if !st.result_is_final {
            return;
        }
        // Déjà mesuré (ou en cours) pour ce résultat : on n'encode pas deux fois.
        if st.export_sizes_revision == st.result_revision
            && (st.export_sizes.is_some() || st.export_sizes_computing.load(Ordering::Relaxed))
        {
            return;
        }
        (
            st.result_revision,
            st.source_size,
            st.params.transparent,
            st.params.bg_color,
            st.result.clone(),
        )
    };
    let (Some(source_size), Some(image)) = (source_size, image) else {
        return;
    };
    let (generation_counter, generation, computing) = {
        let mut st = state.borrow_mut();
        // Un nouveau résultat invalide toute mesure en cours.
        st.export_sizes_generation.fetch_add(1, Ordering::AcqRel);
        let generation = st.export_sizes_generation.load(Ordering::Acquire);
        st.export_sizes_computing.store(true, Ordering::Release);
        st.export_sizes_revision = revision;
        st.export_sizes = None;
        (
            Arc::clone(&st.export_sizes_generation),
            generation,
            Arc::clone(&st.export_sizes_computing),
        )
    };
    let tx = tx.clone();
    std::thread::spawn(move || {
        let sizes = measure_export_sizes(&image, transparent, bg, Some(source_size));
        // Mesure périmée (l'utilisateur a changé de rendu entre-temps) : on
        // jette le résultat, comme le fait la density map.
        if generation_counter.load(Ordering::Acquire) != generation {
            return;
        }
        computing.store(false, Ordering::Release);
        // Publication par le canal : c'est la boucle GTK qui rafraîchit les
        // étiquettes, jamais un thread de travail qui touche un widget.
        let _ = tx.send_blocking(Msg::ExportSizes { generation, sizes });
    });
}

/// Texte de l'étiquette de barre d'état : `Export : PNG 1.4 Mo · JPEG 791 Ko`.
pub(crate) fn sizes_label(state: &St) -> String {
    let (sizes, measuring) = {
        let st = state.borrow();
        (
            st.export_sizes.as_ref().cloned(),
            st.export_sizes_computing.load(Ordering::Relaxed),
        )
    };
    match sizes {
        Some(sizes) => format!("Export : {}", sizes.summary()),
        None if measuring => "Export : mesure…".to_string(),
        None => String::new(),
    }
}

/// Comparaison avec la source pour le format par défaut, pour l'infobulle du
/// menu d'export.
pub(crate) fn comparison(state: &St) -> Option<String> {
    state
        .borrow()
        .export_sizes
        .as_ref()
        .and_then(|sizes| sizes.comparison(ExportFormat::default_export()))
}
