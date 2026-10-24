//! Measurement of the export size, in a worker thread.
//!
//! The measurement really encodes the image for each format
//! ([`pointimg::export`]): it is a full compression per format, so
//! it is done apart, only on the *final* result (a preview size would mean
//! nothing), invalidated as soon as the user goes back to adjusting, and
//! dropped in a slot that the main thread picks up.
//!
//! The result feeds a label of the status bar and the tooltip of the export
//! menu: without it, the central benefit of `pointimg` — "the same image, much
//! lighter" — would stay invisible until the first save.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use pointimg::export::{ExportFormat, measure_export_sizes};

use crate::Msg;

use crate::State;

type St = Rc<RefCell<State>>;

/// Starts the measurement if the displayed result is final and not already
/// measured.
pub(crate) fn refresh(state: &St, tx: &async_channel::Sender<Msg>) {
    let (revision, source_size, transparent, bg, image) = {
        let st = state.borrow();
        if !st.result_is_final {
            return;
        }
        // Already measured (or in progress) for this result: we do not encode
        // twice.
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
        // A new result invalidates any measurement in progress.
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
        // Stale measurement (the user changed the render in the meantime): the
        // result is thrown away, as the density map does.
        if generation_counter.load(Ordering::Acquire) != generation {
            return;
        }
        computing.store(false, Ordering::Release);
        // Publication through the channel: it is the GTK loop that refreshes
        // the labels, never a worker thread touching a widget.
        let _ = tx.send_blocking(Msg::ExportSizes { generation, sizes });
    });
}

/// Text of the status bar label: `Export : PNG 1.4 MB · JPEG 791 kB`.
pub(crate) fn sizes_label(state: &St) -> String {
    let (sizes, measuring) = {
        let st = state.borrow();
        (
            st.export_sizes.as_ref().cloned(),
            st.export_sizes_computing.load(Ordering::Relaxed),
        )
    };
    match sizes {
        Some(sizes) => pointimg::frontend::t("Export : {}").replace("{}", &sizes.summary()),
        None if measuring => pointimg::frontend::t("Export : mesure…").to_string(),
        None => String::new(),
    }
}

/// Comparison with the source for the default format, for the tooltip of the
/// export menu.
pub(crate) fn comparison(state: &St) -> Option<String> {
    state
        .borrow()
        .export_sizes
        .as_ref()
        .and_then(|sizes| sizes.comparison(ExportFormat::default_export()))
}
