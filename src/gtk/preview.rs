//! Preview textures, zoom, panes, empty state and status bar.

use std::cell::RefCell;
use std::rc::Rc;

use gtk::prelude::*;

use image::{GrayImage, RgbImage, RgbaImage};
use pointimg::filter;
use pointimg::frontend;

use crate::compare;
use crate::controls::Widgets;
use crate::{State, ViewMode};

/// Maximum preview zoom: beyond it, one only sees a flat patch of color.
/// Zoom bounds, aligned with egui via `frontend`. Two different ranges for
/// the same gesture would make you compare what is not comparable.
/// (`f64::from` is not `const` yet, hence the explicit `as`.)
const ZOOM_MIN: f64 = frontend::ZOOM_MIN as f64;
const ZOOM_MAX: f64 = frontend::ZOOM_MAX as f64;
/// One zoom notch, by keyboard and by buttons (identical on both sides).
pub(crate) const ZOOM_STEP: f64 = frontend::ZOOM_STEP as f64;

fn texture_from_bytes(width: u32, height: u32, data: Vec<u8>) -> gtk::gdk::Texture {
    let bytes = gtk::glib::Bytes::from_owned(data);
    gtk::gdk::MemoryTexture::new(
        width as i32,
        height as i32,
        gtk::gdk::MemoryFormat::R8g8b8a8,
        &bytes,
        (width * 4) as usize,
    )
    .upcast()
}

fn texture_from_rgb(img: &RgbImage) -> gtk::gdk::Texture {
    let (w, h) = img.dimensions();
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for p in img.pixels() {
        data.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    texture_from_bytes(w, h, data)
}

fn texture_from_gray(img: &GrayImage) -> gtk::gdk::Texture {
    let (w, h) = img.dimensions();
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for p in img.pixels() {
        data.extend_from_slice(&[p[0], p[0], p[0], 255]);
    }
    texture_from_bytes(w, h, data)
}

/// Converts an RGBA image to a texture, compositing the alpha onto a checkerboard.
fn texture_from_rgba_checker(img: &RgbaImage) -> gtk::gdk::Texture {
    texture_from_bytes(img.width(), img.height(), rgba_checker_bytes(img))
}

/// RGBA bytes (R8g8b8a8) of an image, alpha flattened onto a checkerboard.
pub(crate) fn rgba_checker_bytes(img: &RgbaImage) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            // Composite shared with egui; only the byte order of the RGBA
            // bytes is specific to GTK.
            let [r, g, b] = frontend::composite_on_checker(p.0, x, y);
            let i = ((y * w + x) * 4) as usize;
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
            data[i + 3] = 255;
        }
    }
    data
}

/// Texture of the result, for the clipboard.
pub(crate) fn rgba_to_texture(img: &RgbaImage) -> gtk::gdk::Texture {
    texture_from_rgba_checker(img)
}

// ─── Textures ────────────────────────────────────────────────────────────────

pub(crate) fn refresh_source_texture(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let texture = state.borrow().src_rgb.as_ref().map(texture_from_rgb);
    // Do not hold a `borrow_mut` during `set_paintable` (GTK setters can
    // emit synchronous signals).
    match texture {
        Some(texture) => {
            widgets.preview.src_pane.image.set_paintable(Some(&texture));
            state.borrow_mut().src_texture = Some(texture);
        }
        None => {
            widgets
                .preview
                .src_pane
                .image
                .set_paintable(None::<&gtk::gdk::Texture>);
            state.borrow_mut().src_texture = None;
        }
    }
}

pub(crate) fn refresh_result_texture(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let texture = state
        .borrow()
        .result
        .as_ref()
        .map(|img| texture_from_rgba_checker(img));
    match texture {
        Some(texture) => {
            widgets.preview.dst_pane.image.set_paintable(Some(&texture));
            state.borrow_mut().result_texture = Some(texture);
        }
        None => {
            widgets
                .preview
                .dst_pane
                .image
                .set_paintable(None::<&gtk::gdk::Texture>);
            state.borrow_mut().result_texture = None;
        }
    }
}

pub(crate) fn refresh_density_texture(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let texture = state.borrow().density_image.as_ref().map(texture_from_gray);
    match texture {
        Some(texture) => {
            widgets
                .preview
                .density_pane
                .image
                .set_paintable(Some(&texture));
            state.borrow_mut().density_texture = Some(texture);
        }
        None => {
            widgets
                .preview
                .density_pane
                .image
                .set_paintable(None::<&gtk::gdk::Texture>);
            state.borrow_mut().density_texture = None;
        }
    }
}

/// Rebuilds all the preview textures and reapplies the zoom.
pub(crate) fn refresh_preview(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    refresh_source_texture(state, widgets);
    refresh_result_texture(state, widgets);
    refresh_density_texture(state, widgets);
    apply_zoom(state, widgets);
    update_status_labels(state, widgets);
}

// ─── Zoom ────────────────────────────────────────────────────────────────────

/// Size of an image, taking the current render mode into account.
fn reference_size(state: &State) -> Option<(u32, u32)> {
    if let Some(src) = state.src_rgb.as_ref() {
        return Some((src.width(), src.height()));
    }
    state.result.as_ref().map(|img| img.dimensions())
}

/// Recomputes the "Ajuster" (fit) zoom from the available size.
fn compute_fit_zoom(state: &State, width: i32, height: i32) -> f64 {
    let Some((w, h)) = reference_size(state) else {
        return 1.0;
    };
    if w == 0 || h == 0 || width <= 0 || height <= 0 {
        return 1.0;
    }
    let scale = ((width as f64 - 12.0) / f64::from(w)).min((height as f64 - 12.0) / f64::from(h));
    scale.clamp(ZOOM_MIN, ZOOM_MAX)
}

/// Applies the visibility, the scale and the zoom of the preview panes.
pub(crate) fn apply_zoom(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (visible_src, visible_dst, visible_density, visible_compare) = {
        let st = state.borrow();
        match st.view_mode {
            ViewMode::Compare => (false, false, false, true),
            ViewMode::Side => (true, true, false, false),
            ViewMode::ResultOnly => (false, true, false, false),
            ViewMode::SourceOnly => (true, false, false, false),
            ViewMode::DensityMap => (false, false, true, false),
        }
    };

    for (root, visible) in [
        (&widgets.preview.src_pane.root, visible_src),
        (&widgets.preview.dst_pane.root, visible_dst),
        (&widgets.preview.density_pane.root, visible_density),
        (&widgets.preview.compare_pane.root, visible_compare),
    ] {
        root.set_visible(visible);
    }

    let (fit, zoom, size) = {
        let st = state.borrow();
        (st.zoom_fit, st.zoom, reference_size(&st))
    };

    // In "Ajuster" mode, the factor depends on the viewport: it is measured
    // once on a visible pane then applied to all of them (comparison
    // included).
    let mut effective = if fit {
        let (w, h) = viewport(
            &widgets.preview.src_pane.scroll,
            &widgets.preview.dst_pane.scroll,
            &widgets.preview.density_pane.scroll,
            &widgets.preview.compare_pane.scroll,
        );
        compute_fit_zoom(&state.borrow(), w, h)
    } else {
        zoom.clamp(ZOOM_MIN, ZOOM_MAX)
    };
    if effective <= 0.0 {
        effective = 1.0;
    }
    state.borrow_mut().effective_zoom = effective;

    if let Some((w, h)) = size {
        let target = ((w as f64 * effective).round() as i32).max(1);
        let target_h = ((h as f64 * effective).round() as i32).max(1);
        for image in [
            &widgets.preview.src_pane.image,
            &widgets.preview.dst_pane.image,
            &widgets.preview.density_pane.image,
        ] {
            if image.pixel_size() != target {
                image.set_pixel_size(target);
            }
        }
        let area = &widgets.preview.compare_pane.area;
        let (cw, ch) = (target, target_h);
        {
            let mut data = widgets.preview.compare_pane.data.borrow_mut();
            if data.content_width != cw || data.content_height != ch {
                data.content_width = cw;
                data.content_height = ch;
            }
        }
        if area.content_width() != cw {
            area.set_content_width(cw);
        }
        if area.content_height() != ch {
            area.set_content_height(ch);
        }
    } else {
        for image in [
            &widgets.preview.src_pane.image,
            &widgets.preview.dst_pane.image,
            &widgets.preview.density_pane.image,
        ] {
            image.set_pixel_size(0);
        }
        widgets.preview.compare_pane.area.set_content_width(0);
        widgets.preview.compare_pane.area.set_content_height(0);
    }
    widgets.preview.compare_pane.area.queue_draw();

    // The result is not veiled during the compute: the status bar alone
    // signals the provisional nature of the render ("aperçu rapide",
    // "pleine résolution…"), as in the egui interface.

    // Zoom labels: the percentage actually applied, not the stored value
    // (unused in "Ajuster" mode).
    widgets
        .preview
        .zoom_label
        .set_text(&format!("{:.0} %", (effective * 100.0).round()));
    widgets.preview.zoom_out.set_sensitive(!fit);
    widgets.preview.zoom_in.set_sensitive(!fit);
    widgets.preview.zoom_fit.set_tooltip_text(Some(if fit {
        frontend::gettext("Zoom ajusté à la fenêtre (0)")
    } else {
        frontend::gettext("Ajuster l'aperçu à la fenêtre (0)")
    }));
}

/// Size of the preview viewport, measured on the first visible pane.
fn viewport(
    src: &gtk::ScrolledWindow,
    dst: &gtk::ScrolledWindow,
    density: &gtk::ScrolledWindow,
    compare: &gtk::ScrolledWindow,
) -> (i32, i32) {
    for scroll in [compare, src, dst, density] {
        if scroll.is_visible() && scroll.width() > 1 && scroll.height() > 1 {
            return (scroll.width(), scroll.height());
        }
    }
    (0, 0)
}

/// Recomputes the "Ajuster" (fit) zoom when the viewport changes size.
pub(crate) fn connect_fit_signals(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let scrolls = [
        widgets.preview.src_pane.scroll.clone(),
        widgets.preview.dst_pane.scroll.clone(),
        widgets.preview.density_pane.scroll.clone(),
        widgets.preview.compare_pane.scroll.clone(),
    ];
    for scroll in scrolls {
        for adjustment in [scroll.hadjustment(), scroll.vadjustment()] {
            let state = Rc::clone(state);
            let widgets = Rc::clone(widgets);
            adjustment.connect_changed(move |_| {
                if state.borrow().zoom_fit {
                    apply_zoom(&state, &widgets);
                }
            });
        }
    }
}

// ─── Synchronization of the views ───────────────────────────────────────────

// Prevents the adjustment mirrors from triggering each other in a loop.
thread_local! {
    static SYNCING: RefCell<bool> = const { RefCell::new(false) };
}

/// Mirrors the scrolling of one pane onto all the others.
pub(crate) fn connect_scroll_sync(widgets: &Rc<Widgets>) {
    let scrolls = [
        widgets.preview.src_pane.scroll.clone(),
        widgets.preview.dst_pane.scroll.clone(),
        widgets.preview.density_pane.scroll.clone(),
        widgets.preview.compare_pane.scroll.clone(),
    ];
    for (index, scroll) in scrolls.iter().enumerate() {
        for horizontal in [true, false] {
            let adjustment = if horizontal {
                scroll.hadjustment()
            } else {
                scroll.vadjustment()
            };
            let peers: Vec<_> = scrolls
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != index)
                .map(|(_, s)| s.clone())
                .collect();
            adjustment.connect_value_changed(move |source| {
                if SYNCING.with(|flag| *flag.borrow()) {
                    return;
                }
                SYNCING.with(|flag| *flag.borrow_mut() = true);
                for peer in &peers {
                    if !peer.is_visible() {
                        continue;
                    }
                    let target = if horizontal {
                        peer.hadjustment()
                    } else {
                        peer.vadjustment()
                    };
                    mirror(source, &target);
                }
                SYNCING.with(|flag| *flag.borrow_mut() = false);
            });
        }
    }
}

fn mirror(source: &gtk::Adjustment, target: &gtk::Adjustment) {
    if (target.value() - source.value()).abs() < f64::EPSILON {
        return;
    }
    // We align on the relative position, not on the absolute value: the panes
    // may have different content sizes.
    let source_page = source.page_size().max(1.0);
    let target_page = target.page_size().max(1.0);
    let ratio = (source.value() + source_page / 2.0) / source_page;
    let upper = (target.upper() - target_page).max(target.lower());
    target.set_value((ratio * target_page - target_page / 2.0).clamp(target.lower(), upper));
}

// ─── Preview, empty state and badges ─────────────────────────────────────────

/// Shows/hides the welcome state and updates the badges of the preview.
pub(crate) fn update_overlays(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (has_image, computing, stale, quality) = {
        let st = state.borrow();
        (
            st.src_rgb.is_some(),
            st.computing,
            st.result_is_stale(),
            st.quality,
        )
    };

    widgets.status_bar.empty_page.set_visible(!has_image);
    widgets.preview.busy.set_visible(computing);
    widgets.preview.busy.set_spinning(computing);
    widgets.preview.busy_label.set_text(if computing {
        frontend::gettext("Calcul…")
    } else {
        ""
    });

    if quality.is_full() {
        widgets.preview.quality_badge.set_text("");
        widgets.preview.quality_badge.set_visible(false);
    } else {
        let text = if stale {
            frontend::msg::finalizing()
        } else {
            frontend::gettext("aperçu rapide").to_string()
        };
        widgets.preview.quality_badge.set_text(&text);
        widgets.preview.quality_badge.set_visible(has_image);
        widgets
            .preview
            .quality_badge
            .set_tooltip_text(Some(frontend::gettext(
                "Calcul sur une image réduite. La pleine résolution est calculée \
             automatiquement dès que vous arrêtez de manipuler les réglages.",
            )));
    }

    // Window title: file name + position in the folder.
    let (title, subtitle) = {
        let st = state.borrow();
        let name = st
            .src_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or(frontend::gettext("Aucune image"))
            .to_string();
        let suffix = st
            .folder_position()
            .map(|(index, total)| format!(" · {index}/{total}"))
            .unwrap_or_default();
        (format!("pointimg — {name}{suffix}"), window_subtitle(&st))
    };
    widgets.window.window_title.set_title(&title);
    widgets.window.window_title.set_subtitle(&subtitle);

    // Folder: position of the current image.
    let folder = state.borrow().folder_position();
    widgets.presets.folder_label.set_text(
        &folder
            .map(|(index, total)| format!("{index} / {total}"))
            .unwrap_or_default(),
    );
    widgets.presets.folder_row.set_visible(folder.is_some());
}

fn window_subtitle(state: &State) -> String {
    match state.src_rgb.as_ref() {
        Some(src) => pointimg::frontend::describe_image_size(src.width(), src.height()),
        None => frontend::gettext("Aucune image").to_string(),
    }
}

// ─── Actions and status bar ──────────────────────────────────────────────────

pub(crate) fn refresh_actions(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (computing, has_result, final_result, undoable, redoable) = {
        let st = state.borrow();
        (
            st.computing,
            st.result.is_some(),
            st.result.is_some() && st.result_is_final && !st.computing,
            !st.history.is_empty() || st.pending_commit,
            !st.future.is_empty(),
        )
    };

    widgets.window.cancel.set_visible(computing);
    widgets.window.progress.set_visible(computing);
    widgets.compute.recalc.set_sensitive(!computing);
    if !computing {
        widgets.window.progress.set_fraction(0.0);
    }
    widgets.window.undo.set_sensitive(undoable);
    widgets.window.redo.set_sensitive(redoable);
    // Only the full-resolution render is exported: in fast preview mode, the
    // export waits for the automatic finalization.
    widgets
        .window
        .export
        .set_sensitive(has_result && !computing);
    widgets
        .window
        .export
        .set_tooltip_text(Some(if final_result {
            frontend::gettext("Exporter l'image (Ctrl+S)")
        } else {
            frontend::gettext("Export en attente du rendu en pleine résolution…")
        }));
    update_overlays(state, widgets);
}

pub(crate) fn update_status_labels(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (memory, timing, status) = {
        let st = state.borrow();
        let memory = st.src_rgb.as_ref().map(|src| {
            let estimate = filter::estimate_memory_bytes_for(src.width(), src.height(), &st.params);
            let warning = estimate > 1_500 * 1024 * 1024;
            // The figure alone alarmed without explaining anything: we say
            // what it announces, that is what the full-resolution pass will
            // require — this is not the memory already used.
            let text = frontend::gettext("Pleine résolution : ~{}")
                .replace("{}", &crate::format_memory(estimate));
            if warning { format!("⚠ {text}") } else { text }
        });
        let timing = st.last_compute_ms.map(|ms| {
            frontend::gettext("Dernier calcul : {}").replace("{}", &crate::format_duration(ms))
        });
        (memory, timing, st.status.clone())
    };

    widgets
        .status_bar
        .memory
        .set_text(memory.as_deref().unwrap_or(""));
    widgets
        .status_bar
        .memory
        .set_tooltip_text(Some(frontend::gettext(
            "Pic mémoire estimé pour le calcul en cours.",
        )));
    // Export size: measured by real encoding of the final render.
    let sizes = crate::exports::sizes_label(state);
    widgets.preview.export_sizes.set_text(&sizes);
    if let Some(comparison) = crate::exports::comparison(state) {
        widgets.preview.export_sizes.set_tooltip_text(Some(&format!(
            "{comparison}\n{}",
            pointimg::export::ExportSizes::explanation()
        )));
    } else {
        widgets.preview.export_sizes.set_tooltip_text(None);
    }
    // The export menu tooltip announces the size before even opening it.
    if let Some(comparison) = crate::exports::comparison(state) {
        widgets.window.export.set_tooltip_text(Some(
            &frontend::gettext("Exporter le rendu.\n{comparison}")
                .replace("{comparison}", &comparison),
        ));
    } else {
        widgets.window.export.set_tooltip_text(None);
    }
    widgets
        .status_bar
        .timing
        .set_text(timing.as_deref().unwrap_or(""));
    widgets.status_bar.status.set_text(&status);
}

/// Ephemeral error message (errors must not steal a status line that
/// describes the current state).
pub(crate) fn notify_error(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, message: &str) {
    let toast = adw::Toast::new(message);
    toast.set_timeout(6);
    widgets.window.toast_overlay.add_toast(toast);
    state.borrow_mut().status = message.to_string();
    update_status_labels(state, widgets);
}

/// Information notification (save, copy, image change).
pub(crate) fn notify(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, message: &str) {
    let toast = adw::Toast::new(message);
    widgets.window.toast_overlay.add_toast(toast);
    state.borrow_mut().status = message.to_string();
    update_status_labels(state, widgets);
}

// ─── Zoom / views: high-level actions ───────────────────────────────────────

/// Changes the zoom keeping the center of the viewport (a jump to a corner
/// would be disorienting).
pub(crate) fn zoom_by(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, factor: f64) {
    let current = {
        let st = state.borrow();
        if st.zoom_fit {
            st.effective_zoom
        } else {
            st.zoom
        }
    };
    let target = (current * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    set_zoom(state, widgets, target);
}

pub(crate) fn set_zoom(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, value: f64) {
    let (center_x, center_y) = viewport_center(widgets);
    {
        let mut st = state.borrow_mut();
        st.zoom = value.clamp(ZOOM_MIN, ZOOM_MAX);
        st.zoom_fit = false;
    }
    sync_zoom_toggles(state, widgets);
    apply_zoom(state, widgets);
    keep_center(widgets, center_x, center_y);
}

/// Toggles the "Ajuster" (fit) mode.
pub(crate) fn toggle_fit(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, fit: bool) {
    {
        let mut st = state.borrow_mut();
        st.zoom_fit = fit;
        if !fit {
            st.zoom = st.effective_zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        }
    }
    sync_zoom_toggles(state, widgets);
    apply_zoom(state, widgets);
}

/// Aligns the "1:1" / "Ajuster" buttons on the state.
pub(crate) fn sync_zoom_toggles(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (fit, at_one) = {
        let st = state.borrow();
        let zoom = if st.zoom_fit {
            st.effective_zoom
        } else {
            st.zoom
        };
        (st.zoom_fit, (zoom - 1.0).abs() < 0.001)
    };
    let guard = {
        let mut st = state.borrow_mut();
        if st.updating {
            None
        } else {
            st.updating = true;
            Some(())
        }
    };
    if guard.is_some() {
        widgets.preview.zoom_fit.set_active(fit);
        widgets.preview.zoom_one.set_active(!fit && at_one);
        state.borrow_mut().updating = false;
    }
}

/// Center of the viewport, in adjustment coordinates.
///
/// Neither `state` nor the adjustments themselves come into it: only the
/// visible scrolling matters, which would hold as well for the keyboard as
/// for the wheel.
fn viewport_center(widgets: &Rc<Widgets>) -> (f64, f64) {
    let scroll = visible_scroll(widgets);
    match scroll {
        Some(scroll) => {
            let (page_w, page_h) = (scroll.width().max(1), scroll.height().max(1));
            let x = scroll.hadjustment().value() + f64::from(page_w) / 2.0;
            let y = scroll.vadjustment().value() + f64::from(page_h) / 2.0;
            (x, y)
        }
        None => (0.0, 0.0),
    }
}

fn keep_center(widgets: &Rc<Widgets>, center_x: f64, center_y: f64) {
    let Some(scroll) = visible_scroll(widgets) else {
        return;
    };
    for (adjustment, center) in [
        (&scroll.hadjustment(), center_x),
        (&scroll.vadjustment(), center_y),
    ] {
        let page = adjustment.page_size().max(1.0);
        let upper = (adjustment.upper() - page).max(adjustment.lower());
        adjustment.set_value((center - page / 2.0).clamp(adjustment.lower(), upper));
    }
}

fn visible_scroll(widgets: &Rc<Widgets>) -> Option<gtk::ScrolledWindow> {
    for scroll in [
        &widgets.preview.compare_pane.scroll,
        &widgets.preview.src_pane.scroll,
        &widgets.preview.dst_pane.scroll,
        &widgets.preview.density_pane.scroll,
    ] {
        if scroll.is_visible() && scroll.width() > 1 {
            return Some(scroll.clone());
        }
    }
    None
}

/// Refreshes the comparator (surfaces + "stale" state).
pub(crate) fn refresh_compare(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (source, result, has_image, stale) = {
        let st = state.borrow();
        (
            st.src_rgb.clone(),
            st.result.clone(),
            st.src_rgb.is_some(),
            st.result_is_stale(),
        )
    };
    compare::refresh(
        &widgets.preview.compare_pane,
        source.as_ref(),
        result.as_deref(),
        has_image,
        stale,
    );
}
