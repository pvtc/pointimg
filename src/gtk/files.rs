//! Dialogues de fichiers, presets et export.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use adw::prelude::*;

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

/// Propose le dernier dossier utilisé, pour éviter de le ressaisir à chaque
/// ouverture ou export.
fn apply_initial_folder(dialog: &gtk::FileDialog, state: &Rc<RefCell<State>>) {
    let folder = state.borrow().last_dir.clone().or_else(|| {
        state
            .borrow()
            .src_path
            .as_ref()
            .and_then(|p| p.parent().map(PathBuf::from))
    });
    if let Some(dir) = folder {
        let file = gtk::gio::File::for_path(dir);
        dialog.set_initial_folder(Some(&file));
    }
}

/// Nom de fichier par défaut dérivé de l'image source (`photo.jpg` →
/// `photo-pointimg.png`).
fn default_name(state: &Rc<RefCell<State>>, ext: &str) -> String {
    state
        .borrow()
        .src_path
        .as_ref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .map(|stem| format!("{stem}-pointimg.{ext}"))
        .unwrap_or_else(|| format!("pointimg.{ext}"))
}

/// Mémorise le dossier d'un fichier écrit, pour les dialogues suivants.
fn remember_dir(state: &Rc<RefCell<State>>, path: &std::path::Path) {
    if let Some(parent) = path.parent().filter(|p| p.is_dir()) {
        state.borrow_mut().last_dir = Some(parent.to_path_buf());
    }
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
    apply_initial_folder(&dialog, state);
    let images = make_filter("Images", &actions::SUPPORTED_IMAGE_EXTENSIONS);
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

pub(crate) fn choose_open_folder(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let dialog = gtk::FileDialog::builder()
        .title("Ouvrir un dossier d'images")
        .modal(true)
        .build();
    apply_initial_folder(&dialog, state);
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    dialog.select_folder(Some(window), gtk::gio::Cancellable::NONE, move |res| {
        if let Ok(file) = res
            && let Some(path) = file.path()
        {
            actions::open_folder(path, &state, &widgets);
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
        .initial_name(default_name(state, "png"))
        .build();
    apply_initial_folder(&dialog, state);
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
        .initial_name(default_name(state, "svg"))
        .build();
    apply_initial_folder(&dialog, state);
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
    apply_initial_folder(&dialog, state);
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
    apply_initial_folder(&dialog, state);
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
    let ok = result.is_ok();
    let message = match result {
        Ok(()) => format!("Preset sauvegardé : {}", path.display()),
        Err(e) => format!("Erreur sauvegarde preset : {e}"),
    };
    if ok {
        remember_dir(state, &path);
    }
    if ok {
        preview::notify(state, widgets, &message);
    } else {
        preview::notify_error(state, widgets, &message);
    }
}

fn apply_preset(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    match frontend::load_preset(&path) {
        Ok(params) => {
            remember_dir(state, &path);
            actions::apply_preset_params(
                params,
                &format!("Preset chargé : {}", path.display()),
                state,
                widgets,
            );
        }
        Err(e) => {
            preview::notify_error(
                state,
                widgets,
                &format!("Erreur de chargement du preset : {e}"),
            );
        }
    }
}

pub(crate) fn save_result(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_image_extension(path, "png");
    let result = {
        let st = state.borrow();
        let Some(img) = st.result.as_ref() else {
            preview::notify(state, widgets, "Aucun résultat à sauvegarder.");
            return;
        };
        if !st.result_is_final {
            drop(st);
            preview::notify_error(
                state,
                widgets,
                "Rendu en pleine résolution en cours — réessayez dans un instant.",
            );
            return;
        }
        frontend::save_rgba_image(&path, img, st.params.transparent, st.params.bg_color)
    };
    let ok = result.is_ok();
    let message = match result {
        Ok(()) => format!("Sauvegardé : {}", path.display()),
        Err(e) => format!("Erreur de sauvegarde : {e}"),
    };
    if ok {
        remember_dir(state, &path);
        preview::notify(state, widgets, &message);
    } else {
        preview::notify_error(state, widgets, &message);
    }
}

pub(crate) fn save_svg(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_svg_extension(path);
    let result = {
        let st = state.borrow();
        let Some(src) = st.src_rgb.as_ref() else {
            drop(st);
            preview::notify(state, widgets, "Ouvrez d'abord une image.");
            return;
        };
        let (w, h) = src.dimensions();
        let rendered = if let Some(dots) = st.last_dots.as_ref() {
            filter::render_svg_from_dots(w, h, dots, &st.params)
        } else {
            filter::render_svg(src, &st.params)
        };
        rendered
            .map_err(|e| e.to_string())
            .and_then(|svg| frontend::atomic_text_write(&path, &svg).map_err(|e| e.to_string()))
    };
    let ok = result.is_ok();
    let message = match result {
        Ok(()) => format!("SVG sauvegardé : {}", path.display()),
        Err(e) => format!("Erreur d'écriture SVG : {e}"),
    };
    if ok {
        remember_dir(state, &path);
        preview::notify(state, widgets, &message);
    } else {
        preview::notify_error(state, widgets, &message);
    }
}

/// Export sans dialogue : à côté de l'image source, en PNG.
pub(crate) fn quick_export(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let Some(path) = state.borrow().src_path.as_ref().map(|p| {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("pointimg");
        p.with_file_name(format!("{stem}-pointimg.png"))
    }) else {
        preview::notify(state, widgets, "Ouvrez d'abord une image.");
        return;
    };
    save_result(path, state, widgets);
}

/// Export SVG sans dialogue, à côté de l'image source.
pub(crate) fn quick_export_svg(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    actions::save_svg_quick(state, widgets);
}

/// Réinitialise l'état « preset personnalisé » de la liste déroulante.
pub(crate) fn mark_preset_custom(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.preset_combo.selected();
    if selected == 0 {
        return;
    }
    let preset_params: Vec<_> = filter::builtin_presets()
        .into_iter()
        .map(|p| p.params)
        .collect();
    if preset_params
        .get(selected as usize - 1)
        .is_some_and(|params| *params == state.borrow().params)
    {
        return;
    }
    state.borrow_mut().updating = true;
    widgets.preset_combo.set_selected(0);
    state.borrow_mut().updating = false;
}

/// Applique le preset sélectionné dans la liste déroulante.
pub(crate) fn apply_selected_preset(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.preset_combo.selected();
    if selected == 0 {
        return;
    }
    actions::apply_preset(selected as usize - 1, state, widgets);
}

/// Affiche le sous-titre du preset sélectionné (aide contextuelle).
pub(crate) fn update_preset_row(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.preset_combo.selected();
    let preset = filter::builtin_presets();
    let subtitle = match selected {
        0 => "Réglages personnalisés".to_string(),
        index => preset
            .get(index as usize - 1)
            .map(|p| p.description.to_string())
            .unwrap_or_default(),
    };
    let matches_params = match selected {
        0 => true,
        index => preset
            .get(index as usize - 1)
            .map(|p| p.params == state.borrow().params)
            .unwrap_or(false),
    };
    state.borrow_mut().updating = true;
    let text = if matches_params {
        subtitle
    } else {
        format!("{subtitle} (modifié)")
    };
    widgets.preset_combo.set_subtitle(&text);
    state.borrow_mut().updating = false;
    sections::update_sections(widgets, &state.borrow().params);
}
