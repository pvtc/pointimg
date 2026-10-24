//! File, preset and export dialogs.

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

/// Offers the last used folder, to avoid retyping it at every open or
/// export.
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

/// Default file name derived from the source image (`photo.jpg` →
/// `photo-pointimg.png`).
fn default_name(state: &Rc<RefCell<State>>, ext: &str) -> String {
    // The formula is shared with egui; only the fallback differs, and it
    // already lives in `derived_export_name`.
    frontend::derived_export_name(state.borrow().src_path.as_deref(), ext)
}

/// Remembers the folder of a written file, for the next dialogs.
fn remember_dir(state: &Rc<RefCell<State>>, path: &std::path::Path) {
    if let Some(parent) = path.parent().filter(|p| p.is_dir()) {
        state.borrow_mut().last_dir = Some(parent.to_path_buf());
    }
}

/// What we ask the dialog for: an existing file, a folder, or a write
/// name.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pick {
    Open,
    Folder,
    Save,
}

/// Description of a dialog: what we ask for, and how to present it.
///
/// The six dialogs of the module only differed by these fields and by the
/// function called — about 120 lines for twenty lines of real difference.
/// Everything else (starting folder, clones, path extraction) is here once.
struct Ask {
    title: &'static str,
    pick: Pick,
    /// Name offered on writing (`photo-pointimg.png`).
    initial_name: Option<String>,
    /// Filters, and the one selected by default — same object, hence cloned
    /// separately.
    filters: Vec<gtk::FileFilter>,
    default: Option<gtk::FileFilter>,
}

impl Ask {
    fn new(title: &'static str, pick: Pick) -> Self {
        Self {
            title,
            pick,
            initial_name: None,
            filters: Vec::new(),
            default: None,
        }
    }

    fn named(mut self, name: String) -> Self {
        self.initial_name = Some(name);
        self
    }

    fn filters(mut self, filters: Vec<gtk::FileFilter>, default: Option<gtk::FileFilter>) -> Self {
        self.filters = filters;
        self.default = default;
        self
    }
}

/// Opens a dialog and passes the chosen path to `picked`.
///
/// `picked` is an `fn` rather than a closure: each caller already has its
/// function, and the `fn` avoids carrying redundant clones into the
/// dialog's asynchronous closure.
fn pick_file<F>(
    window: &adw::ApplicationWindow,
    ask: &Ask,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
    picked: F,
) where
    F: Fn(PathBuf, &Rc<RefCell<State>>, &Rc<Widgets>) + 'static,
{
    let mut builder = gtk::FileDialog::builder().title(ask.title).modal(true);
    if let Some(name) = &ask.initial_name {
        builder = builder.initial_name(name);
    }
    let dialog = builder.build();
    apply_initial_folder(&dialog, state);
    dialog.set_filters(Some(&filter_store(&ask.filters)));
    dialog.set_default_filter(ask.default.as_ref());
    let state = Rc::clone(state);
    let widgets = Rc::clone(widgets);
    // A cancelled dialog, or a file without a local path (remote URI), does
    // nothing — just like in the six original copies.
    let rappel = move |res: Result<gtk::gio::File, gtk::glib::Error>| {
        let chemins = res.ok().and_then(|file| file.path());
        if let Some(path) = chemins {
            picked(path, &state, &widgets);
        }
    };
    match ask.pick {
        Pick::Open => dialog.open(Some(window), gtk::gio::Cancellable::NONE, rappel),
        Pick::Folder => dialog.select_folder(Some(window), gtk::gio::Cancellable::NONE, rappel),
        Pick::Save => dialog.save(Some(window), gtk::gio::Cancellable::NONE, rappel),
    }
}

/// Opens an image.
pub(crate) fn choose_open_image(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    pick_file(
        window,
        &Ask::new(pointimg::frontend::gettext("Ouvrir une image"), Pick::Open).filters(
            vec![make_filter("Images", &actions::SUPPORTED_IMAGE_EXTENSIONS)],
            None,
        ),
        state,
        widgets,
        actions::load_image,
    );
}

/// Opens a folder of images.
pub(crate) fn choose_open_folder(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    pick_file(
        window,
        &Ask::new(
            pointimg::frontend::gettext("Ouvrir un dossier d'images"),
            Pick::Folder,
        ),
        state,
        widgets,
        actions::open_folder,
    );
}

/// Saves the render, in one of the raster formats.
pub(crate) fn choose_save_image(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let png = make_filter("PNG", &["png"]);
    let jpeg = make_filter("JPEG", &["jpg", "jpeg"]);
    let webp = make_filter("WebP", &["webp"]);
    let bmp = make_filter("BMP", &["bmp"]);
    let tiff = make_filter("TIFF", &["tif", "tiff"]);
    pick_file(
        window,
        &Ask::new(
            pointimg::frontend::gettext("Sauvegarder l'image"),
            Pick::Save,
        )
        .named(default_name(state, "png"))
        .filters(vec![png.clone(), jpeg, webp, bmp, tiff], Some(png.clone())),
        state,
        widgets,
        save_result,
    );
}

/// Saves the render as SVG.
pub(crate) fn choose_save_svg(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let svg = make_filter("SVG", &["svg"]);
    pick_file(
        window,
        &Ask::new(
            pointimg::frontend::gettext("Sauvegarder le SVG"),
            Pick::Save,
        )
        .named(default_name(state, "svg"))
        .filters(vec![svg.clone()], Some(svg.clone())),
        state,
        widgets,
        save_svg,
    );
}

/// Loads a TOML preset.
pub(crate) fn choose_load_preset(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let toml = make_filter("Preset TOML", &["toml"]);
    pick_file(
        window,
        &Ask::new(pointimg::frontend::gettext("Charger un preset"), Pick::Open)
            .filters(vec![toml], None),
        state,
        widgets,
        apply_preset,
    );
}

/// Saves a TOML preset.
pub(crate) fn choose_save_preset(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let toml = make_filter("Preset TOML", &["toml"]);
    pick_file(
        window,
        &Ask::new(
            pointimg::frontend::gettext("Sauvegarder le preset"),
            Pick::Save,
        )
        .named("preset.toml".to_string())
        .filters(vec![toml.clone()], Some(toml.clone())),
        state,
        widgets,
        save_preset,
    );
}

/// Records the outcome of a write: remembers the folder on success, and
/// displays the matching message.
///
/// A single point for the three writes (preset, image, SVG): the folder is
/// remembered on success and the message is chosen by the caller, so a
/// policy change applies to all three at once.
fn report_save(
    result: Result<(), String>,
    path: &std::path::Path,
    on_ok: impl FnOnce() -> String,
    on_err: impl FnOnce(&str) -> String,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let ok = result.is_ok();
    let message = match result {
        Ok(()) => on_ok(),
        Err(e) => on_err(&e),
    };
    if ok {
        remember_dir(state, path);
    }
    let affichage = message;
    if ok {
        preview::notify(state, widgets, &affichage);
    } else {
        preview::notify_error(state, widgets, &affichage);
    }
}

fn save_preset(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_toml_extension(path);
    report_save(
        frontend::save_preset(&path, &state.borrow().params).map_err(|e| e.to_string()),
        &path,
        || {
            pointimg::frontend::gettext("Preset sauvegardé : {}")
                .replace("{}", &path.display().to_string())
        },
        |e| pointimg::frontend::gettext("Erreur sauvegarde preset : {e}").replace("{e}", e),
        state,
        widgets,
    );
}

fn apply_preset(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    match frontend::load_preset(&path) {
        Ok(params) => {
            remember_dir(state, &path);
            actions::apply_preset_params(
                params,
                &pointimg::frontend::gettext("Preset chargé : {}")
                    .replace("{}", &path.display().to_string()),
                state,
                widgets,
            );
        }
        Err(e) => {
            preview::notify_error(
                state,
                widgets,
                &pointimg::frontend::gettext("Erreur de chargement du preset : {e}")
                    .replace("{e}", e.as_str()),
            );
        }
    }
}

pub(crate) fn save_result(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_image_extension(path, "png");
    let result = {
        let st = state.borrow();
        let Some(img) = st.result.as_ref() else {
            preview::notify(
                state,
                widgets,
                pointimg::frontend::gettext("Aucun résultat à sauvegarder."),
            );
            return;
        };
        if !st.result_is_final {
            drop(st);
            preview::notify_error(
                state,
                widgets,
                pointimg::frontend::gettext(
                    "Rendu en pleine résolution en cours — réessayez dans un instant.",
                ),
            );
            return;
        }
        frontend::save_rgba_image(&path, img, st.params.transparent, st.params.bg_color)
    };
    report_save(
        result.map_err(|e| e.to_string()),
        &path,
        || {
            pointimg::frontend::gettext("Sauvegardé : {}")
                .replace("{}", &path.display().to_string())
        },
        |e| pointimg::frontend::gettext("Erreur de sauvegarde : {e}").replace("{e}", e),
        state,
        widgets,
    );
}

pub(crate) fn save_svg(path: PathBuf, state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let path = frontend::ensure_svg_extension(path);
    let result = {
        let st = state.borrow();
        let Some(src) = st.src_rgb.as_ref() else {
            drop(st);
            preview::notify(state, widgets, &frontend::msg::open_first());
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
    report_save(
        result,
        &path,
        || {
            pointimg::frontend::gettext("SVG sauvegardé : {}")
                .replace("{}", &path.display().to_string())
        },
        |e| pointimg::frontend::gettext("Erreur d'écriture SVG : {e}").replace("{e}", e),
        state,
        widgets,
    );
}

/// Export without a dialog: next to the source image, as PNG.
pub(crate) fn quick_export(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let Some(path) =
        state.borrow().src_path.as_ref().map(|source| {
            source.with_file_name(frontend::derived_export_name(Some(source), "png"))
        })
    else {
        preview::notify(state, widgets, &frontend::msg::open_first());
        return;
    };
    // No file chooser: the confirmation happens here. The path is cloned
    // because the dialog borrows it and the save consumes it.
    let (destination, s, w) = (path.clone(), Rc::clone(state), Rc::clone(widgets));
    crate::confirm::confirm_overwrite(widgets, &path, move || save_result(destination, &s, &w));
}

/// SVG export without a dialog, next to the source image.
pub(crate) fn quick_export_svg(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    actions::save_svg_quick(state, widgets);
}

/// Resets the "custom preset" state of the drop-down list.
pub(crate) fn mark_preset_custom(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.presets.preset_combo.selected();
    if selected == 0 {
        return;
    }
    // The "are these settings exactly this preset?" predicate lives in
    // `frontend`, shared by both interfaces.
    if frontend::matching_preset_index(&state.borrow().params) == Some(selected as usize - 1) {
        return;
    }
    state.borrow_mut().updating = true;
    widgets.presets.preset_combo.set_selected(0);
    state.borrow_mut().updating = false;
}

/// Applies the preset selected in the drop-down list.
pub(crate) fn apply_selected_preset(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.presets.preset_combo.selected();
    if selected == 0 {
        return;
    }
    actions::apply_preset(selected as usize - 1, state, widgets);
}

/// Displays the subtitle of the selected preset (contextual help).
pub(crate) fn update_preset_row(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.presets.preset_combo.selected();
    let preset = filter::builtin_presets();
    let subtitle = match selected {
        0 => pointimg::frontend::gettext("Réglages personnalisés").to_string(),
        index => preset
            .get(index as usize - 1)
            .map(|p| p.description.to_string())
            .unwrap_or_default(),
    };
    let matches_params = match selected {
        0 => true,
        index => preset
            .get(index as usize - 1)
            .is_some_and(|p| p.params == state.borrow().params),
    };
    state.borrow_mut().updating = true;
    let text = if matches_params {
        subtitle
    } else {
        pointimg::frontend::gettext("{subtitle} (modifié)").replace("{subtitle}", &subtitle)
    };
    widgets.presets.preset_combo.set_subtitle(&text);
    state.borrow_mut().updating = false;
    sections::update_sections(widgets, &state.borrow().params);
    // The grid replaces the drop-down list when it is present: it therefore
    // carries the caption too, and the active state of the thumbnails.
    crate::wire::sync_preset_thumbs(state, widgets, selected);
    crate::wire::update_preset_caption(state, widgets, &text);
}
