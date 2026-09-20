//! Conversion d'images en textures GDK, aperçu et zoom.

use std::cell::RefCell;
use std::rc::Rc;

use gtk::prelude::*;

use image::{GrayImage, RgbImage, RgbaImage};
use pointimg::filter;

use crate::controls::Widgets;
use crate::{State, ViewMode};

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
    texture_from_bytes(w, h, data)
}

pub(crate) fn refresh_source_texture(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let texture = state.borrow().src_rgb.as_ref().map(texture_from_rgb);
    // Ne pas garder de `borrow_mut` pendant `set_paintable` (les setters GTK
    // peuvent émettre des signaux synchrones).
    match texture {
        Some(texture) => {
            widgets.src_image.set_paintable(Some(&texture));
            state.borrow_mut().src_texture = Some(texture);
        }
        None => {
            widgets.src_image.set_paintable(None::<&gtk::gdk::Texture>);
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
            widgets.dst_image.set_paintable(Some(&texture));
            state.borrow_mut().result_texture = Some(texture);
        }
        None => {
            widgets.dst_image.set_paintable(None::<&gtk::gdk::Texture>);
            state.borrow_mut().result_texture = None;
        }
    }
}

pub(crate) fn refresh_density_texture(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let texture = state.borrow().density_image.as_ref().map(texture_from_gray);
    match texture {
        Some(texture) => {
            widgets.density_image.set_paintable(Some(&texture));
            state.borrow_mut().density_texture = Some(texture);
        }
        None => {
            widgets
                .density_image
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
}

/// Applique la visibilité et l'échelle des trois panneaux d'aperçu.
pub(crate) fn apply_zoom(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let st = state.borrow();
    let side = st.view_mode == ViewMode::Side;
    let show_src = side || st.view_mode == ViewMode::SourceOnly;
    let show_dst = side || st.view_mode == ViewMode::ResultOnly;
    let show_density = st.view_mode == ViewMode::DensityMap;

    let panels = [
        (
            &widgets.src_image,
            &widgets.src_scroll,
            st.src_texture.as_ref(),
            show_src,
        ),
        (
            &widgets.dst_image,
            &widgets.dst_scroll,
            st.result_texture.as_ref(),
            show_dst,
        ),
        (
            &widgets.density_image,
            &widgets.density_scroll,
            st.density_texture.as_ref(),
            show_density,
        ),
    ];

    for (image, scroll, texture, visible) in panels {
        scroll.set_visible(visible);
        image.set_visible(visible);
        let Some(texture) = texture else {
            continue;
        };
        let tw = texture.width() as f64;
        let th = texture.height() as f64;
        let scale = if st.zoom_fit {
            let aw = scroll.width() as f64 - 12.0;
            let ah = scroll.height() as f64 - 12.0;
            if aw > 1.0 && ah > 1.0 {
                (aw / tw).min(ah / th).max(0.01)
            } else {
                1.0
            }
        } else {
            st.zoom
        };
        let longer = tw.max(th);
        let desired = (longer * scale).round().max(1.0) as i32;
        // Évite de redimensionner inutilement (et donc de relancer une frame)
        // quand rien n'a changé.
        if image.pixel_size() != desired {
            image.set_pixel_size(desired);
        }
    }
}

/// Recalcule le "fit" quand le viewport change de taille (au lieu de poller).
pub(crate) fn connect_fit_signals(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    for scroll in [
        &widgets.src_scroll,
        &widgets.dst_scroll,
        &widgets.density_scroll,
    ] {
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

pub(crate) fn refresh_actions(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let st = state.borrow();
    let computing = st.computing;
    widgets.recalc.set_visible(!computing);
    widgets.cancel.set_visible(computing);
    widgets.progress.set_visible(computing);
    if !computing {
        widgets.progress.set_fraction(0.0);
    }
    widgets
        .save_image
        .set_sensitive(st.result.is_some() && !computing);
    widgets
        .save_svg
        .set_sensitive((st.last_dots.is_some() || st.src_rgb.is_some()) && !computing);
}

pub(crate) fn update_status_labels(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let st = state.borrow();
    if let Some(src) = &st.src_rgb {
        widgets.memory.set_text(&format!(
            "Mémoire estimée : {}",
            crate::format_memory(filter::estimate_memory_bytes_for(
                src.width(),
                src.height(),
                &st.params
            ))
        ));
    }
    if let Some(ms) = st.last_compute_ms {
        widgets
            .timing
            .set_text(&format!("Dernier calcul : {}", crate::format_duration(ms)));
    }
    widgets.status.set_text(&st.status);
}
