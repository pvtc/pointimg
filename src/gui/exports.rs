//! Export size measurement, in a worker thread.
//!
//! The measurement really encodes the image for each format
//! ([`pointimg::export`]), which costs a full compression per format. It is
//! therefore done apart, on the *final* result (never on a quick preview),
//! invalidated as soon as the user keeps adjusting, and its result is a
//! simple [`ExportSizes`] shown in the status bar.
//!
//! Without it, the interface can only show the size of the source — and the
//! central benefit of `pointimg`, "the same image, much lighter", would
//! stay invisible until the first save.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use eframe::egui;

use pointimg::export::{ExportFormat, ExportSizes, measure_export_sizes};

use super::App;

impl App {
    /// Starts the measurement if the displayed result is final and not already measured.
    pub(crate) fn refresh_export_sizes(&mut self, ctx: &egui::Context) {
        if !self.has_final_result() {
            return;
        }
        let revision = self.result_revision.load(Ordering::Acquire);
        // Already measured (or in progress) for this result: we do not encode twice.
        if self.export_sizes_revision == revision
            && (self.export_sizes.is_some() || self.export_sizes_computing.load(Ordering::Relaxed))
        {
            return;
        }
        let Some(source_size) = self.source_size else {
            return;
        };
        // The image is shared (`Arc`) instead of being copied: on 8 Mpx, a
        // `cloned()` cost ~33 MB copied back on every measurement.
        let Some(img) = self
            .result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(Arc::clone)
        else {
            return;
        };
        let (transparent, bg) = (self.params.transparent, self.params.bg_color);
        // A new result invalidates the measurement in progress.
        self.export_sizes_generation.fetch_add(1, Ordering::AcqRel);
        let generation = self.export_sizes_generation.load(Ordering::Acquire);
        self.export_sizes_computing.store(true, Ordering::Release);
        self.export_sizes_revision = revision;
        self.export_sizes = None;
        let generation_counter = Arc::clone(&self.export_sizes_generation);
        let computing = Arc::clone(&self.export_sizes_computing);
        let slot = Arc::clone(&self.export_sizes_slot);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let sizes = measure_export_sizes(&img, transparent, bg, Some(source_size));
            if generation_counter.load(Ordering::Acquire) == generation {
                *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(sizes);
                computing.store(false, Ordering::Release);
                ctx.request_repaint();
            }
        });
    }

    /// Retrieves the measurement result if the worker has deposited it.
    pub(crate) fn poll_export_sizes(&mut self) -> Option<ExportSizes> {
        let taken = self
            .export_sizes_slot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(sizes) = taken {
            self.export_sizes = Some(sizes.clone());
        }
        self.export_sizes.clone()
    }
}

/// Information line "export size": `PNG 2.1 MB · JPEG 380 kB`.
pub(crate) fn export_sizes_line(sizes: &ExportSizes) -> String {
    sizes.summary()
}

/// Comparison with the source, for the default format.
pub(crate) fn export_comparison(sizes: &ExportSizes) -> Option<String> {
    sizes.comparison(ExportFormat::default_export())
}
