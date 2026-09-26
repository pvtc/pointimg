//! Textures d'aperçu, zoom, panneaux, état vide et barre d'état.

use std::cell::RefCell;
use std::rc::Rc;

use gtk::prelude::*;

use image::{GrayImage, RgbImage, RgbaImage};
use pointimg::filter;

use crate::compare;
use crate::controls::Widgets;
use crate::{State, ViewMode};

/// Zoom maximal de l'aperçu : au-delà, on ne voit plus qu'un aplat de couleur.
const ZOOM_MAX: f64 = 16.0;
const ZOOM_MIN: f64 = 0.02;

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

/// Convertit une image RGBA en texture, en composant l'alpha sur un damier.
fn texture_from_rgba_checker(img: &RgbaImage) -> gtk::gdk::Texture {
    texture_from_bytes(img.width(), img.height(), rgba_checker_bytes(img))
}

/// Bytes RGBA (R8g8b8a8) d'une image, alpha aplati sur un damier.
pub(crate) fn rgba_checker_bytes(img: &RgbaImage) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            let cell = ((x / 8) + (y / 8)) % 2;
            let bg = if cell == 0 { 200.0 } else { 240.0 };
            let inv = 1.0 - a;
            let i = ((y * w + x) * 4) as usize;
            data[i] = (p[0] as f32 * a + bg * inv + 0.5) as u8;
            data[i + 1] = (p[1] as f32 * a + bg * inv + 0.5) as u8;
            data[i + 2] = (p[2] as f32 * a + bg * inv + 0.5) as u8;
            data[i + 3] = 255;
        }
    }
    data
}

/// Texture du résultat, pour le presse-papiers.
pub(crate) fn rgba_to_texture(img: &RgbaImage) -> gtk::gdk::Texture {
    texture_from_rgba_checker(img)
}

// ─── Textures ────────────────────────────────────────────────────────────────

pub(crate) fn refresh_source_texture(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let texture = state.borrow().src_rgb.as_ref().map(texture_from_rgb);
    // Ne pas garder de `borrow_mut` pendant `set_paintable` (les setters GTK
    // peuvent émettre des signaux synchrones).
    match texture {
        Some(texture) => {
            widgets.src_pane.image.set_paintable(Some(&texture));
            state.borrow_mut().src_texture = Some(texture);
        }
        None => {
            widgets
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
        .map(texture_from_rgba_checker);
    match texture {
        Some(texture) => {
            widgets.dst_pane.image.set_paintable(Some(&texture));
            state.borrow_mut().result_texture = Some(texture);
        }
        None => {
            widgets
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
            widgets.density_pane.image.set_paintable(Some(&texture));
            state.borrow_mut().density_texture = Some(texture);
        }
        None => {
            widgets
                .density_pane
                .image
                .set_paintable(None::<&gtk::gdk::Texture>);
            state.borrow_mut().density_texture = None;
        }
    }
}

/// Reconstruit toutes les textures d'aperçu et réapplique le zoom.
pub(crate) fn refresh_preview(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    refresh_source_texture(state, widgets);
    refresh_result_texture(state, widgets);
    refresh_density_texture(state, widgets);
    apply_zoom(state, widgets);
    update_status_labels(state, widgets);
}

// ─── Zoom ────────────────────────────────────────────────────────────────────

/// Taille d'une image, en tenant compte du mode de rendu courant.
fn reference_size(state: &State) -> Option<(u32, u32)> {
    if let Some(src) = state.src_rgb.as_ref() {
        return Some((src.width(), src.height()));
    }
    state.result.as_ref().map(|img| img.dimensions())
}

/// Recalcule le zoom « Ajuster » à partir de la taille disponible.
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

/// Applique visibilité, échelle et zoom des panneaux d'aperçu.
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
        (&widgets.src_pane.root, visible_src),
        (&widgets.dst_pane.root, visible_dst),
        (&widgets.density_pane.root, visible_density),
        (&widgets.compare_pane.root, visible_compare),
    ] {
        root.set_visible(visible);
    }

    let (fit, zoom, size) = {
        let st = state.borrow();
        (st.zoom_fit, st.zoom, reference_size(&st))
    };

    // En mode « Ajuster », le facteur dépend du viewport : on le mesure une fois
    // sur un panneau visible puis on l'applique à tous (comparaison comprise).
    let mut effective = if fit {
        let (w, h) = viewport(
            &widgets.src_pane.scroll,
            &widgets.dst_pane.scroll,
            &widgets.density_pane.scroll,
            &widgets.compare_pane.scroll,
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
            &widgets.src_pane.image,
            &widgets.dst_pane.image,
            &widgets.density_pane.image,
        ] {
            if image.pixel_size() != target {
                image.set_pixel_size(target);
            }
        }
        let area = &widgets.compare_pane.area;
        let (cw, ch) = (target, target_h);
        {
            let mut data = widgets.compare_pane.data.borrow_mut();
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
            &widgets.src_pane.image,
            &widgets.dst_pane.image,
            &widgets.density_pane.image,
        ] {
            image.set_pixel_size(0);
        }
        widgets.compare_pane.area.set_content_width(0);
        widgets.compare_pane.area.set_content_height(0);
    }
    widgets.compare_pane.area.queue_draw();

    // Le résultat affiché mais non final (ou en cours de calcul) est.voilé.
    let stale = state.borrow().result_is_stale();
    widgets
        .dst_pane
        .image
        .set_opacity(if stale { 0.55 } else { 1.0 });
    widgets
        .compare_pane
        .area
        .set_opacity(if stale { 0.7 } else { 1.0 });

    // Étiquettes de zoom : le pourcentage réellement appliqué, pas la valeur
    // mémorisée (inutilisée en mode « Ajuster »).
    widgets
        .zoom_label
        .set_text(&format!("{:.0} %", (effective * 100.0).round()));
    widgets.zoom_out.set_sensitive(!fit);
    widgets.zoom_in.set_sensitive(!fit);
    widgets.zoom_fit.set_tooltip_text(Some(if fit {
        "Zoom ajusté à la fenêtre (0)"
    } else {
        "Ajuster l'aperçu à la fenêtre (0)"
    }));
}

/// Taille du viewport d'aperçu, mesurée sur le premier panneau visible.
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

/// Recalcule le « Ajuster » quand le viewport change de taille.
pub(crate) fn connect_fit_signals(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let scrolls = [
        widgets.src_pane.scroll.clone(),
        widgets.dst_pane.scroll.clone(),
        widgets.density_pane.scroll.clone(),
        widgets.compare_pane.scroll.clone(),
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

// ─── Synchronisation des vues ────────────────────────────────────────────────

// Empêche les miroirs de adjustment de se déclencher en boucle.
thread_local! {
    static SYNCING: RefCell<bool> = const { RefCell::new(false) };
}

/// Miroite le défilement d'un panneau vers tous les autres.
pub(crate) fn connect_scroll_sync(widgets: &Rc<Widgets>) {
    let scrolls = [
        widgets.src_pane.scroll.clone(),
        widgets.dst_pane.scroll.clone(),
        widgets.density_pane.scroll.clone(),
        widgets.compare_pane.scroll.clone(),
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
    // On se cale sur la position relative, pas sur la valeur absolue : les
    // panneaux peuvent avoir des tailles de contenu différentes.
    let source_page = source.page_size().max(1.0);
    let target_page = target.page_size().max(1.0);
    let ratio = (source.value() + source_page / 2.0) / source_page;
    let upper = (target.upper() - target_page).max(target.lower());
    target.set_value((ratio * target_page - target_page / 2.0).clamp(target.lower(), upper));
}

// ─── Aperçu, état vide et badges ─────────────────────────────────────────────

/// Affiche/masque l'état d'accueil et met à jour les badges de l'aperçu.
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

    widgets.empty_page.set_visible(!has_image);
    widgets.busy.set_visible(computing);
    widgets.busy.set_spinning(computing);
    widgets
        .busy_label
        .set_text(if computing { "Calcul…" } else { "" });

    if quality.is_full() {
        widgets.quality_badge.set_text("");
        widgets.quality_badge.set_visible(false);
    } else {
        widgets.quality_badge.set_text(if stale {
            "pleine résolution…"
        } else {
            "aperçu rapide"
        });
        widgets.quality_badge.set_visible(has_image);
        widgets.quality_badge.set_tooltip_text(Some(
            "Calcul sur une image réduite. La pleine résolution est calculée \
             automatiquement dès que vous arrêtez de manipuler les réglages.",
        ));
    }

    // Titre de la fenêtre : nom du fichier + position dans le dossier.
    let (title, subtitle) = {
        let st = state.borrow();
        let name = st
            .src_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("Aucune image")
            .to_string();
        let suffix = st
            .folder_position()
            .map(|(index, total)| format!(" · {index}/{total}"))
            .unwrap_or_default();
        (format!("pointimg — {name}{suffix}"), window_subtitle(&st))
    };
    widgets.window_title.set_title(&title);
    widgets.window_title.set_subtitle(&subtitle);

    // Dossier : position de l'image courante.
    let folder = state.borrow().folder_position();
    widgets.folder_label.set_text(
        &folder
            .map(|(index, total)| format!("{index} / {total}"))
            .unwrap_or_default(),
    );
    widgets.folder_row.set_visible(folder.is_some());
}

fn window_subtitle(state: &State) -> String {
    match state.src_rgb.as_ref() {
        Some(src) => pointimg::frontend::describe_image_size(src.width(), src.height()),
        None => "Aucune image".to_string(),
    }
}

// ─── Actions et barre d'état ────────────────────────────────────────────────

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

    widgets.cancel.set_visible(computing);
    widgets.progress.set_visible(computing);
    widgets.recalc.set_sensitive(!computing);
    if !computing {
        widgets.progress.set_fraction(0.0);
    }
    widgets.undo.set_sensitive(undoable);
    widgets.redo.set_sensitive(redoable);
    // On n'exporte que le rendu en pleine résolution : en mode aperçu rapide,
    // l'export attend la finalisation automatique.
    widgets.export.set_sensitive(has_result && !computing);
    widgets.export.set_tooltip_text(Some(if final_result {
        "Exporter l'image (Ctrl+S)"
    } else {
        "Export en attente du rendu en pleine résolution…"
    }));
    update_overlays(state, widgets);
}

pub(crate) fn update_status_labels(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let (memory, timing, status) = {
        let st = state.borrow();
        let memory = st.src_rgb.as_ref().map(|src| {
            let estimate = filter::estimate_memory_bytes_for(src.width(), src.height(), &st.params);
            let warning = estimate > 1_500 * 1024 * 1024;
            let text = format!("Mémoire estimée : {}", crate::format_memory(estimate));
            if warning { format!("⚠ {text}") } else { text }
        });
        let timing = st
            .last_compute_ms
            .map(|ms| format!("Dernier calcul : {}", crate::format_duration(ms)));
        (memory, timing, st.status.clone())
    };

    widgets.memory.set_text(memory.as_deref().unwrap_or(""));
    widgets
        .memory
        .set_tooltip_text(Some("Pic mémoire estimé pour le calcul en cours."));
    widgets.timing.set_text(timing.as_deref().unwrap_or(""));
    widgets.status.set_text(&status);
}

/// Message d'erreur éphémère (les erreurs ne doivent pas voler une ligne de
/// statut qui décrit l'état courant).
pub(crate) fn notify_error(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, message: &str) {
    let toast = adw::Toast::new(message);
    toast.set_timeout(6);
    widgets.toast_overlay.add_toast(toast);
    state.borrow_mut().status = message.to_string();
    update_status_labels(state, widgets);
}

/// Notification d'information (enregistrement, copie, changement d'image).
pub(crate) fn notify(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, message: &str) {
    let toast = adw::Toast::new(message);
    widgets.toast_overlay.add_toast(toast);
    state.borrow_mut().status = message.to_string();
    update_status_labels(state, widgets);
}

// ─── Zoom / vues : actions de haut niveau ────────────────────────────────────

/// Change le zoom en gardant le centre du viewport (un saut vers le coin
/// serait désorientant).
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
    let (center_x, center_y) = viewport_center(state, widgets);
    {
        let mut st = state.borrow_mut();
        st.zoom = value.clamp(ZOOM_MIN, ZOOM_MAX);
        st.zoom_fit = false;
    }
    sync_zoom_toggles(state, widgets);
    apply_zoom(state, widgets);
    keep_center(state, widgets, center_x, center_y);
}

/// Bascule le mode « Ajuster ».
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

/// Aligne les boutons « 1:1 » / « Ajuster » sur l'état.
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
        widgets.zoom_fit.set_active(fit);
        widgets.zoom_one.set_active(!fit && at_one);
        state.borrow_mut().updating = false;
    }
}

fn viewport_center(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) -> (f64, f64) {
    let _ = state;
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

fn keep_center(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>, center_x: f64, center_y: f64) {
    let _ = state;
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
        &widgets.compare_pane.scroll,
        &widgets.src_pane.scroll,
        &widgets.dst_pane.scroll,
        &widgets.density_pane.scroll,
    ] {
        if scroll.is_visible() && scroll.width() > 1 {
            return Some(scroll.clone());
        }
    }
    None
}

/// Rafraîchit le comparateur (surfaces + état « obsolète »).
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
        &widgets.compare_pane,
        source.as_ref(),
        result.as_ref(),
        has_image,
        stale,
    );
}
