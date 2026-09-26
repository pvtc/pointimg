//! Comparaison avant/après par balayage (« wipe »).
//!
//! Un seul `DrawingArea` dessine la source à gauche d'un curseur et le résultat
//! à droite, ce qui permet de comparer la même zone pixel par pixel. Le rendu
//! passe par cairo : les `ImageSurface` sont construites à partir des images
//! déjà en mémoire (pas de re-téléchargement GPU) et la zone visible est
//! découpée par clipping rectangulaire de part et d'autre du curseur.
//!
//! La taille du contenu est pilotée par le zoom (comme les autres panneaux) :
//! c'est `preview::apply_zoom` qui la fixe, le `ScrolledWindow` gère le
//! défilement et le curseur est simplement dessiné à `split * largeur`.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use gtk::cairo;

use image::{RgbImage, RgbaImage};

use crate::controls::{ComparePane, make_scroll};

/// Ce que le dessin du comparateur doit savoir sur l'état courant.
pub(crate) struct CompareData {
    pub source: Option<cairo::ImageSurface>,
    pub result: Option<cairo::ImageSurface>,
    /// Dimensions d'affichage (image × zoom) partagées par les deux couches.
    pub content_width: i32,
    pub content_height: i32,
    pub stale: bool,
    pub has_image: bool,
    /// Position du curseur, synchronisée avec le widget interactif.
    pub split: f64,
}

/// Surface cairo `Rgb24` (BGRX en mémoire) depuis une image RVB.
fn rgb_surface(img: &RgbImage) -> Option<cairo::ImageSurface> {
    if img.width() == 0 || img.height() == 0 {
        return None;
    }
    let stride = img.width() as usize * 4;
    let mut data = vec![0u8; stride * img.height() as usize];
    for (x, y, p) in img.enumerate_pixels() {
        let i = y as usize * stride + x as usize * 4;
        data[i] = p[2];
        data[i + 1] = p[1];
        data[i + 2] = p[0];
        data[i + 3] = 0;
    }
    cairo::ImageSurface::create_for_data(
        data,
        cairo::Format::Rgb24,
        img.width() as i32,
        img.height() as i32,
        stride as i32,
    )
    .ok()
}

/// Surface cairo depuis une image RGBA, alpha aplati sur un damier (comme le
/// rendu des autres panneaux).
fn rgba_checker_surface(img: &RgbaImage) -> Option<cairo::ImageSurface> {
    if img.width() == 0 || img.height() == 0 {
        return None;
    }
    let stride = img.width() as usize * 4;
    let mut data = vec![0u8; stride * img.height() as usize];
    for (x, y, p) in img.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let inv = 1.0 - a;
        let cell = ((x / 8) + (y / 8)) % 2;
        let bg = if cell == 0 { 200.0 } else { 240.0 };
        let i = y as usize * stride + x as usize * 4;
        data[i] = (p[2] as f32 * a + bg * inv + 0.5) as u8;
        data[i + 1] = (p[1] as f32 * a + bg * inv + 0.5) as u8;
        data[i + 2] = (p[0] as f32 * a + bg * inv + 0.5) as u8;
        data[i + 3] = 0;
    }
    cairo::ImageSurface::create_for_data(
        data,
        cairo::Format::Rgb24,
        img.width() as i32,
        img.height() as i32,
        stride as i32,
    )
    .ok()
}

/// Construit le panneau « comparaison ».
pub(crate) fn build_pane() -> ComparePane {
    let caption = gtk::Label::new(Some("Comparaison"));
    caption.set_xalign(0.0);
    caption.add_css_class("dim-label");
    caption.set_margin_start(8);
    caption.set_margin_top(4);
    caption.set_tooltip_text(Some(
        "Source à gauche du curseur, résultat à droite — glissez pour le déplacer.",
    ));

    let area = gtk::DrawingArea::new();
    area.set_hexpand(true);
    area.set_vexpand(true);

    let data: Rc<RefCell<CompareData>> = Rc::new(RefCell::new(CompareData {
        source: None,
        result: None,
        content_width: 0,
        content_height: 0,
        stale: false,
        has_image: false,
        split: 0.5,
    }));

    {
        let draw_data = Rc::clone(&data);
        area.set_draw_func(move |_area, cr, width, height| {
            paint(cr, width, height, &draw_data.borrow());
        });
    }

    // Curseur déplaçable : c'est un réglage d'affichage, il vit dans le widget
    // (miroiré dans `CompareData` pour survive au redessin).
    let set_split = std::rc::Rc::new({
        let data = Rc::clone(&data);
        let area = area.clone();
        move |fraction: f64| {
            let fraction = fraction.clamp(0.02, 0.98);
            data.borrow_mut().split = fraction;
            area.queue_draw();
        }
    });
    {
        let set_click = set_split.clone();
        let area_ref = area.clone();
        let click = gtk::GestureClick::new();
        click.connect_pressed(move |gesture, _, x, _| {
            if gesture.current_button() == 1 {
                set_click(x / area_ref.width().max(1) as f64);
            }
        });
        area.add_controller(click);

        let set_start = std::rc::Rc::new(set_split.clone());
        let area_ref = area.clone();
        let drag = gtk::GestureDrag::new();
        drag.set_button(1);
        drag.connect_drag_begin(move |_, x, _| {
            set_start(x / area_ref.width().max(1) as f64);
        });
        let set_update = set_split.clone();
        let area_ref = area.clone();
        drag.connect_drag_update(move |gesture, dx, _| {
            if let Some(start) = gesture.start_point() {
                set_update((start.0 + dx) / area_ref.width().max(1) as f64);
            }
        });
        area.add_controller(drag);
    }

    let scroll = make_scroll(&area);

    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.append(&caption);
    root.append(&scroll);

    ComparePane {
        root,
        scroll,
        area,
        data,
    }
}

/// Rafraîchit les surfaces et la taille de contenu du comparateur.
pub(crate) fn refresh(
    pane: &ComparePane,
    source: Option<&RgbImage>,
    result: Option<&RgbaImage>,
    has_image: bool,
    stale: bool,
) {
    {
        let mut data = pane.data.borrow_mut();
        data.source = source.and_then(rgb_surface);
        data.result = result.and_then(rgba_checker_surface);
        data.stale = stale;
        data.has_image = has_image;
    }
    pane.area.queue_draw();
}

/// Dessine la comparaison.
fn paint(cr: &cairo::Context, width: i32, height: i32, data: &CompareData) {
    let (Some(source), Some(result)) = (data.source.as_ref(), data.result.as_ref()) else {
        if data.has_image {
            // Image chargée mais rendu pas encore arrivé : le message du
            // comparateur reste, l'état d'accueil couvre le cas sans image.
            message(cr, width as f64, height as f64, "Calcul en cours…");
        }
        return;
    };

    let (sw, sh) = (source.width() as f64, source.height() as f64);
    let (rw, rh) = (result.width() as f64, result.height() as f64);
    let content_w = data.content_width.max(1) as f64;
    let content_h = data.content_height.max(1) as f64;
    // Même échelle pour les deux couches : la comparaison porte sur la même zone.
    let scale = (content_w / sw.max(rw)).min(content_h / sh.max(rh));
    let dw = (sw * scale).round().max(1.0);
    let dh = (sh * scale).round().max(1.0);
    let dw_r = (rw * scale).round().max(1.0);
    let dh_r = (rh * scale).round().max(1.0);
    let total_w = dw.max(dw_r);
    let total_h = dh.max(dh_r);
    let split_x = (total_w * data.split).clamp(1.0, total_w - 1.0);

    // Résultat à droite du curseur.
    cr.save().ok();
    cr.rectangle(split_x, 0.0, total_w - split_x, total_h);
    cr.clip();
    paint_surface(cr, result, 0.0, 0.0, dw_r, dh_r);
    cr.restore().ok();

    // Source à gauche du curseur.
    cr.save().ok();
    cr.rectangle(0.0, 0.0, split_x, total_h);
    cr.clip();
    paint_surface(cr, source, 0.0, 0.0, dw, dh);
    cr.restore().ok();

    // Curseur : trait + poignée.
    cr.set_line_width(2.0);
    cr.set_source_rgb(1.0, 1.0, 1.0);
    cr.move_to(split_x, 0.0);
    cr.line_to(split_x, total_h);
    cr.stroke().ok();
    cr.set_source_rgb(0.15, 0.15, 0.15);
    cr.move_to(split_x + 1.5, 0.0);
    cr.line_to(split_x + 1.5, total_h);
    cr.stroke().ok();

    let cy = total_h / 2.0;
    cr.arc(split_x, cy, 13.0, 0.0, std::f64::consts::TAU);
    cr.set_source_rgb(1.0, 1.0, 1.0);
    cr.fill_preserve().ok();
    cr.set_source_rgb(0.2, 0.2, 0.2);
    cr.set_line_width(1.5);
    cr.stroke().ok();
    cr.move_to(split_x - 6.0, cy);
    cr.line_to(split_x - 2.0, cy - 4.0);
    cr.move_to(split_x - 6.0, cy);
    cr.line_to(split_x - 2.0, cy + 4.0);
    cr.move_to(split_x + 6.0, cy);
    cr.line_to(split_x + 2.0, cy - 4.0);
    cr.move_to(split_x + 6.0, cy);
    cr.line_to(split_x + 2.0, cy + 4.0);
    cr.stroke().ok();

    if data.stale {
        badge(cr, 12.0, 12.0, "aperçu obsolète — recalcul en cours");
    }
    let _ = (width, height);
}

fn paint_surface(
    cr: &cairo::Context,
    surface: &cairo::ImageSurface,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) {
    cr.save().ok();
    cr.translate(x, y);
    cr.scale(
        width / surface.width() as f64,
        height / surface.height() as f64,
    );
    cr.set_source_surface(surface, 0.0, 0.0).ok();
    cr.paint().ok();
    cr.restore().ok();
}

fn message(cr: &cairo::Context, width: f64, height: f64, text: &str) {
    cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
    cr.set_font_size(14.0);
    let extents = cr.text_extents(text).map(|e| e.width()).unwrap_or(0.0);
    cr.set_source_rgb(0.55, 0.55, 0.55);
    cr.move_to((width - extents) / 2.0, height / 2.0);
    cr.show_text(text).ok();
}

fn badge(cr: &cairo::Context, x: f64, y: f64, text: &str) {
    cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
    cr.set_font_size(12.0);
    let width = cr.text_extents(text).map(|e| e.width()).unwrap_or(0.0) + 16.0;
    cr.rectangle(x, y, width, 22.0);
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.7);
    cr.fill().ok();
    cr.set_source_rgb(1.0, 1.0, 1.0);
    cr.move_to(x + 8.0, y + 15.0);
    cr.show_text(text).ok();
}
