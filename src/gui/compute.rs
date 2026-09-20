//! Workers de calcul : filtre et density map, hors thread GUI.

use eframe::egui;
use image::RgbImage;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::App;
use super::convert::rgb_to_rgba_opaque;
use super::format::generation_is_current;
use pointimg::filter::{self, Algorithm, FilterParams};

/// Ensemble des canaux de communication partagés entre le thread GUI et un
/// worker de calcul (fige les `Arc::clone` de `start_compute`).
struct ComputeChannels {
    result: Arc<Mutex<Option<image::RgbaImage>>>,
    result_revision: Arc<std::sync::atomic::AtomicU64>,
    last_dots: Arc<Mutex<Option<Vec<filter::Dot>>>>,
    progress: Arc<Mutex<(usize, usize)>>,
    computing: Arc<std::sync::atomic::AtomicBool>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    status_err: Arc<Mutex<Option<String>>>,
}

impl App {
    pub(crate) fn start_compute(&mut self, ctx: &egui::Context) {
        if self.computing.load(Ordering::Acquire) {
            self.cancel.store(true, Ordering::Release);
            return;
        }
        let src = match &self.src_rgb {
            Some(s) => s.clone(),
            None => return,
        };
        *self.last_dots.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let params = self.params.clone();
        let density_data = Arc::clone(&self.density_data);
        let compute_generation = Arc::clone(&self.compute_generation);
        let ctx = ctx.clone();
        let generation = compute_generation.fetch_add(1, Ordering::AcqRel) + 1;
        let channels = ComputeChannels {
            result: Arc::clone(&self.result),
            result_revision: Arc::clone(&self.result_revision),
            last_dots: Arc::clone(&self.last_dots),
            progress: Arc::clone(&self.progress),
            computing: Arc::clone(&self.computing),
            cancel: Arc::clone(&self.cancel),
            status_err: Arc::clone(&self.compute_error),
        };

        channels.computing.store(true, Ordering::Relaxed);
        channels.cancel.store(false, Ordering::Relaxed);
        *self.progress.lock().unwrap_or_else(|e| e.into_inner()) = (0, params.iterations);
        self.status = "Calcul en cours…".to_string();
        self.compute_start = Some(Instant::now());
        self.last_param_change = None;

        std::thread::spawn(move || {
            run_compute_worker(
                src,
                params,
                channels,
                density_data,
                compute_generation,
                generation,
                ctx,
            );
        });
    }

    pub(crate) fn start_density_compute(&self, ctx: &egui::Context) {
        let Some(src) = self.src_rgb.clone() else {
            return;
        };
        let sensitivity = self.params.variance_sensitivity;
        let result = Arc::clone(&self.density_result);
        let data = Arc::clone(&self.density_data);
        let density_generation = Arc::clone(&self.density_generation);
        let ctx = ctx.clone();
        let generation = density_generation.fetch_add(1, Ordering::AcqRel) + 1;
        *result.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *data.lock().unwrap_or_else(|e| e.into_inner()) = None;
        std::thread::spawn(move || {
            let (w, h) = src.dimensions();
            let density = filter::compute_density_map(&src, sensitivity);
            if density_generation.load(Ordering::Acquire) == generation {
                // Construire l'image avant de déplacer `density` dans l'`Arc` :
                // évite un clone complet de la density map (jusqu'à ~33 Mo).
                let image = filter::density_to_image(&density, w, h);
                *data.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(density));
                *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(image);
                ctx.request_repaint();
            }
        });
    }
}

/// Corps du worker de calcul (exécuté hors thread GUI).
fn run_compute_worker(
    src: RgbImage,
    params: FilterParams,
    channels: ComputeChannels,
    density_data: Arc<Mutex<Option<Arc<Vec<f32>>>>>,
    compute_generation: Arc<std::sync::atomic::AtomicU64>,
    generation: u64,
    ctx: egui::Context,
) {
    let iters = params.iterations;
    let ComputeChannels {
        result,
        result_revision,
        last_dots,
        progress,
        computing,
        cancel,
        status_err,
    } = channels;
    // Throttle preview cloning to at most once per 100ms to avoid
    // cloning a full image (potentially 36MB on 4K) at every iteration
    let mut last_preview = Instant::now();
    let preview_interval = Duration::from_millis(100);

    // Halftone + transparent : pas de preview progressive, on appelle
    // apply_rgba une fois et on convertit en RGBA pour stockage.
    if params.transparent || params.algorithm == Algorithm::Halftone {
        let res = filter::apply_rgba(&src, &params);
        match res {
            Ok((dst_rgba, dots)) => {
                if generation_is_current(&compute_generation, generation) {
                    // Publier au moins une fois pour que la GUI voie une preview.
                    *progress.lock().unwrap_or_else(|e| e.into_inner()) = (1, 1);
                    *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(dst_rgba);
                    result_revision.fetch_add(1, Ordering::Release);
                    *last_dots.lock().unwrap_or_else(|e| e.into_inner()) = Some(dots);
                }
            }
            Err(e) => {
                if generation_is_current(&compute_generation, generation)
                    && !cancel.load(Ordering::Relaxed)
                {
                    *status_err.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(format!("Erreur : {e}"));
                }
            }
        }
        computing.store(false, Ordering::Release);
        ctx.request_repaint();
        return;
    }

    let publish_preview = |iter: usize,
                           total: usize,
                           preview: &RgbImage,
                           progress: &Arc<Mutex<(usize, usize)>>,
                           result: &Arc<Mutex<Option<image::RgbaImage>>>,
                           result_revision: &Arc<std::sync::atomic::AtomicU64>,
                           last_preview: &mut Instant,
                           ctx: &egui::Context| {
        if !generation_is_current(&compute_generation, generation) {
            return;
        }
        *progress.lock().unwrap_or_else(|e| e.into_inner()) = (iter, total);
        let now = Instant::now();
        // Always clone on last iteration, throttle intermediate previews
        if iter == total || now.duration_since(*last_preview) >= preview_interval {
            *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(rgb_to_rgba_opaque(preview));
            result_revision.fetch_add(1, Ordering::Release);
            *last_preview = now;
            ctx.request_repaint();
        }
    };

    let cached_density = density_data
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let res = if let Some(density) = cached_density.as_deref() {
        filter::apply_with_progress_cached(
            &src,
            &params,
            &cancel,
            density,
            |iter, total, preview: &RgbImage| {
                publish_preview(
                    iter,
                    total,
                    preview,
                    &progress,
                    &result,
                    &result_revision,
                    &mut last_preview,
                    &ctx,
                );
            },
        )
    } else {
        filter::apply_with_progress(&src, &params, &cancel, |iter, total, preview: &RgbImage| {
            publish_preview(
                iter,
                total,
                preview,
                &progress,
                &result,
                &result_revision,
                &mut last_preview,
                &ctx,
            );
        })
    };
    match res {
        Ok((dst, dots)) => {
            if generation_is_current(&compute_generation, generation) {
                *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(rgb_to_rgba_opaque(&dst));
                result_revision.fetch_add(1, Ordering::Release);
                *progress.lock().unwrap_or_else(|e| e.into_inner()) = (iters, iters);
                *last_dots.lock().unwrap_or_else(|e| e.into_inner()) = Some(dots);
            }
        }
        Err(e) => {
            if generation_is_current(&compute_generation, generation)
                && !cancel.load(Ordering::Relaxed)
            {
                *status_err.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(format!("Erreur : {e}"));
            }
        }
    }
    computing.store(false, Ordering::Release);
    ctx.request_repaint();
}
