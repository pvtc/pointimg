//! Mesure de la taille d'export, dans un thread de travail.
//!
//! La mesure encode réellement l'image pour chaque format ([`pointimg::export`]),
//! ce qui coûte une compression complète par format. Elle est donc faite à
//! part, sur le résultat *définitif* (jamais sur un aperçu rapide),
//! invalidée dès que l'utilisateur continue de régler, et son résultat est un
//! simple [`ExportSizes`] affiché dans la barre d'état.
//!
//! Sans elle, l'interface ne peut montrer que la taille de la source — et le
//! bénéfice central de `pointimg`, « la même image en beaucoup plus léger »,
//! resterait invisible jusqu'au premier enregistrement.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use eframe::egui;

use pointimg::export::{ExportFormat, ExportSizes, measure_export_sizes};

use super::App;

impl App {
    /// Lance la mesure si le résultat affiché est définitif et pas déjà mesuré.
    pub(crate) fn refresh_export_sizes(&mut self, ctx: &egui::Context) {
        if !self.has_final_result() {
            return;
        }
        let revision = self.result_revision.load(Ordering::Acquire);
        // Déjà mesuré (ou en cours) pour ce résultat : on n'encode pas deux fois.
        if self.export_sizes_revision == revision
            && (self.export_sizes.is_some() || self.export_sizes_computing.load(Ordering::Relaxed))
        {
            return;
        }
        let Some(source_size) = self.source_size else {
            return;
        };
        let Some(img) = self
            .result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .cloned()
        else {
            return;
        };
        let (transparent, bg) = (self.params.transparent, self.params.bg_color);
        // Un nouveau résultat invalide la mesure en cours.
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

    /// Récupère le résultat de la mesure si le worker l'a déposé.
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

/// Ligne d'information « taille de l'export » : `PNG 2.1 Mo · JPEG 380 Ko`.
pub(crate) fn export_sizes_line(sizes: &ExportSizes) -> String {
    sizes.summary()
}

/// Comparaison avec la source, pour le format par défaut.
pub(crate) fn export_comparison(sizes: &ExportSizes) -> Option<String> {
    sizes.comparison(ExportFormat::default_export())
}
