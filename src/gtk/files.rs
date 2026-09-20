//! Dialogues de fichiers, presets et export.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use gtk::prelude::*;

use pointimg::filter;
use pointimg::frontend;

use crate::State;
use crate::actions;
use crate::controls::Widgets;
use crate::preview;
use crate::sections;

fn make_filter(name: &str, suffixes: &[&str]) -> gtk::FileFilter {
    let filter = gtk::FileFilter::new();
    filter.set_name(Some(name));
    for suffix in suffixes {
        filter.add_suffix(suffix);
    }
    filter
}

fn filter_store(filters: &[gtk::FileFilter]) -> gtk::gio::ListStore {
    let store = gtk::gio::ListStore::new::<gtk::FileFilter>();
    for filter in filters {
        store.append(filter);
    }
    store
}

pub(crate) fn choose_open_image(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let dialog = gtk::FileDialog::builder()
        .title("Ouvrir une image")
        .modal(true)
        .build();
    let images = make_filter(
        "Images",
        &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
    );
    dialog.set_filters(Some(&filter_store(&[images])));
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    dialog.open(Some(window), gtk::gio::Cancellable::NONE, move |res| {
        if let Ok(file) = res
            && let Some(path) = file.path()
        {
            actions::load_image(path, &state, &widgets);
        }
    });
}

pub(crate) fn choose_save_image(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let dialog = gtk::FileDialog::builder()
        .title("Sauvegarder l'image")
        .modal(true)
        .initial_name("result.png")
        .build();
    let png = make_filter("PNG", &["png"]);
    let jpeg = make_filter("JPEG", &["jpg", "jpeg"]);
    let webp = make_filter("WebP", &["webp"]);
    let bmp = make_filter("BMP", &["bmp"]);
    let tiff = make_filter("TIFF", &["tif", "tiff"]);
    dialog.set_filters(Some(&filter_store(&[png.clone(), jpeg, webp, bmp, tiff])));
    dialog.set_default_filter(Some(&png));
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    dialog.save(Some(window), gtk::gio::Cancellable::NONE, move |res| {
        if let Ok(file) = res
            && let Some(path) = file.path()
        {
            save_result(path, &state, &widgets);
        }
    });
}

pub(crate) fn choose_save_svg(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let dialog = gtk::FileDialog::builder()
        .title("Sauvegarder le SVG")
        .modal(true)
        .initial_name("result.svg")
        .build();
    let svg = make_filter("SVG", &["svg"]);
    dialog.set_filters(Some(&filter_store(std::slice::from_ref(&svg))));
    dialog.set_default_filter(Some(&svg));
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    dialog.save(Some(window), gtk::gio::Cancellable::NONE, move |res| {
        if let Ok(file) = res
            && let Some(path) = file.path()
        {
            save_svg(path, &state, &widgets);
        }
    });
}

pub(crate) fn choose_load_preset(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let dialog = gtk::FileDialog::builder()
        .title("Charger un preset")
        .modal(true)
        .build();
    let toml = make_filter("Preset TOML", &["toml"]);
    dialog.set_filters(Some(&filter_store(&[toml])));
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    dialog.open(Some(window), gtk::gio::Cancellable::NONE, move |res| {
        if let Ok(file) = res
            && let Some(path) = file.path()
        {
            apply_preset(path, &state, &widgets);
        }
    });
}

pub(crate) fn choose_save_preset(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let dialog = gtk::FileDialog::builder()
        .title("Sauvegarder le preset")
        .modal(true)
        .initial_name("preset.toml")
        .build();
    let toml = make_filter("Preset TOML", &["toml"]);
    dialog.set_filters(Some(&filter_store(std::slice::from_ref(&toml))));
    dialog.set_default_filter(Some(&toml));
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    dialog.save(Some(window), gtk::gio::Cancellable::NONE, move |res| {
        if let Ok(file) = res
            && let Some(path) = file.path()
        {
            save_preset(path, &state, &widgets);
        }
    });
}

fn save_preset(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_toml_extension(path);
    let result = frontend::save_preset(&path, &state.borrow().params);
    let mut st = state.borrow_mut();
    st.status = match result {
        Ok(()) => format!("Preset sauvegardé : {}", path.display()),
        Err(e) => format!("Erreur sauvegarde preset : {e}"),
    };
    drop(st);
    preview::update_status_labels(state, widgets);
}

fn apply_preset(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    match frontend::load_preset(&path) {
        Ok(params) => {
            {
                let mut st = state.borrow_mut();
                st.params = params;
                st.last_committed = Some(st.params.clone());
                st.future.clear();
            }
            sections::sync_widgets(state, widgets);
            sections::update_sections(widgets, &state.borrow().params);
            actions::refresh_src_rgb(state, widgets);
            actions::request_compute(state, widgets);
            state.borrow_mut().status = format!("Preset chargé : {}", path.display());
        }
        Err(e) => state.borrow_mut().status = format!("Erreur chargement preset : {e}"),
    }
    preview::update_status_labels(state, widgets);
}

fn save_result(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_image_extension(path, "png");
    let result = {
        let st = state.borrow();
        let Some(img) = st.result.as_ref() else {
            return;
        };
        frontend::save_rgba_image(&path, img, st.params.transparent, st.params.bg_color)
    };
    let mut st = state.borrow_mut();
    st.status = match result {
        Ok(()) => format!("Sauvegardé : {}", path.display()),
        Err(e) => format!("Erreur sauvegarde : {e}"),
    };
    drop(st);
    preview::update_status_labels(state, widgets);
}

fn save_svg(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_svg_extension(path);
    let st = state.borrow();
    let Some(src) = st.src_rgb.as_ref() else {
        drop(st);
        return;
    };
    let (w, h) = src.dimensions();
    let rendered = if let Some(dots) = st.last_dots.as_ref() {
        filter::render_svg_from_dots(w, h, dots, &st.params)
    } else {
        filter::render_svg(src, &st.params)
    };
    let result = rendered
        .map_err(|e| e.to_string())
        .and_then(|svg| frontend::atomic_text_write(&path, &svg).map_err(|e| e.to_string()));
    drop(st);
    state.borrow_mut().status = match result {
        Ok(()) => format!("SVG sauvegardé : {}", path.display()),
        Err(e) => format!("Erreur écriture SVG : {e}"),
    };
    preview::update_status_labels(state, widgets);
}
