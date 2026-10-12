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
    // La formule est partagée avec egui ; seul le repli diffère, et il vit déjà
    // dans `derived_export_name`.
    frontend::derived_export_name(state.borrow().src_path.as_deref(), ext)
}

/// Mémorise le dossier d'un fichier écrit, pour les dialogues suivants.
fn remember_dir(state: &Rc<RefCell<State>>, path: &std::path::Path) {
    if let Some(parent) = path.parent().filter(|p| p.is_dir()) {
        state.borrow_mut().last_dir = Some(parent.to_path_buf());
    }
}

/// Ce qu'on demande au dialogue : un fichier existant, un dossier, ou un nom
/// d'écriture.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pick {
    Open,
    Folder,
    Save,
}

/// Description d'un dialogue : ce qu'on demande, et comment le présenter.
///
/// Les six dialogues du module ne différaient que par ces champs et par la
/// fonction appelée — soit ~120 lignes pour une vingtaine de lignes de
/// différence réelle. Tout le reste (dossier de départ, clones, extraction du
/// chemin) est ici une fois.
struct Ask {
    title: &'static str,
    pick: Pick,
    /// Nom proposé à l'écriture (`photo-pointimg.png`).
    initial_name: Option<String>,
    /// Filtres, et celui sélectionné par défaut — même objet, donc cloné à part.
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

/// Ouvre un dialogue et passe le chemin choisi à `picked`.
///
/// `picked` est un `fn` plutôt qu'une closure : chaque appelant a déjà sa
/// fonction, et le `fn` évite d'embarquer des clones superflus dans la closure
/// asynchrone du dialogue.
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
    // Un dialogue annulé, ou un fichier sans chemin local (URI distante), ne fait
    // rien — comme dans les six copies d'origine.
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

/// Ouvre une image.
pub(crate) fn choose_open_image(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    pick_file(
        window,
        &Ask::new("Ouvrir une image", Pick::Open).filters(
            vec![make_filter("Images", &actions::SUPPORTED_IMAGE_EXTENSIONS)],
            None,
        ),
        state,
        widgets,
        actions::load_image,
    );
}

/// Ouvre un dossier d'images.
pub(crate) fn choose_open_folder(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    pick_file(
        window,
        &Ask::new("Ouvrir un dossier d'images", Pick::Folder),
        state,
        widgets,
        actions::open_folder,
    );
}

/// Sauvegarde le rendu, dans l'un des formats matriciels.
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
        &Ask::new("Sauvegarder l'image", Pick::Save)
            .named(default_name(state, "png"))
            .filters(vec![png.clone(), jpeg, webp, bmp, tiff], Some(png.clone())),
        state,
        widgets,
        save_result,
    );
}

/// Sauvegarde le rendu en SVG.
pub(crate) fn choose_save_svg(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let svg = make_filter("SVG", &["svg"]);
    pick_file(
        window,
        &Ask::new("Sauvegarder le SVG", Pick::Save)
            .named(default_name(state, "svg"))
            .filters(vec![svg.clone()], Some(svg.clone())),
        state,
        widgets,
        save_svg,
    );
}

/// Charge un preset TOML.
pub(crate) fn choose_load_preset(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let toml = make_filter("Preset TOML", &["toml"]);
    pick_file(
        window,
        &Ask::new("Charger un preset", Pick::Open).filters(vec![toml], None),
        state,
        widgets,
        apply_preset,
    );
}

/// Sauvegarde un preset TOML.
pub(crate) fn choose_save_preset(
    window: &adw::ApplicationWindow,
    state: &Rc<RefCell<State>>,
    widgets: &Rc<Widgets>,
) {
    let toml = make_filter("Preset TOML", &["toml"]);
    pick_file(
        window,
        &Ask::new("Sauvegarder le preset", Pick::Save)
            .named("preset.toml".to_string())
            .filters(vec![toml.clone()], Some(toml.clone())),
        state,
        widgets,
        save_preset,
    );
}

/// Consigne le résultat d'une écriture : retient le dossier en cas de succès,
/// et affiche le message correspondant.
///
/// Un point unique pour les trois écritures (preset, image, SVG) : le dossier
/// est retenu en cas de succès et le message est choisi par l'appelant, si bien
/// qu'un changement de politique s'applique aux trois d'un coup.
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
        || format!("Preset sauvegardé : {}", path.display()),
        |e| format!("Erreur sauvegarde preset : {e}"),
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
    report_save(
        result.map_err(|e| e.to_string()),
        &path,
        || format!("Sauvegardé : {}", path.display()),
        |e| format!("Erreur de sauvegarde : {e}"),
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
            preview::notify(state, widgets, frontend::msg::OPEN_FIRST);
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
        || format!("SVG sauvegardé : {}", path.display()),
        |e| format!("Erreur d'écriture SVG : {e}"),
        state,
        widgets,
    );
}

/// Export sans dialogue : à côté de l'image source, en PNG.
pub(crate) fn quick_export(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let Some(path) =
        state.borrow().src_path.as_ref().map(|source| {
            source.with_file_name(frontend::derived_export_name(Some(source), "png"))
        })
    else {
        preview::notify(state, widgets, frontend::msg::OPEN_FIRST);
        return;
    };
    // Pas de dialogue de choix : la confirmation se fait ici. Le chemin est
    // cloné parce que le dialogue l'emprunte et la sauvegarde le consomme.
    let (destination, s, w) = (path.clone(), Rc::clone(state), Rc::clone(widgets));
    crate::confirm::confirm_overwrite(widgets, &path, move || save_result(destination, &s, &w));
}

/// Export SVG sans dialogue, à côté de l'image source.
pub(crate) fn quick_export_svg(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    actions::save_svg_quick(state, widgets);
}

/// Réinitialise l'état « preset personnalisé » de la liste déroulante.
pub(crate) fn mark_preset_custom(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.presets.preset_combo.selected();
    if selected == 0 {
        return;
    }
    // Le prédicat « ces réglages sont-ils exactement ce preset ? » vit dans
    // `frontend`, partagé par les deux interfaces.
    if frontend::matching_preset_index(&state.borrow().params) == Some(selected as usize - 1) {
        return;
    }
    state.borrow_mut().updating = true;
    widgets.presets.preset_combo.set_selected(0);
    state.borrow_mut().updating = false;
}

/// Applique le preset sélectionné dans la liste déroulante.
pub(crate) fn apply_selected_preset(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.presets.preset_combo.selected();
    if selected == 0 {
        return;
    }
    actions::apply_preset(selected as usize - 1, state, widgets);
}

/// Affiche le sous-titre du preset sélectionné (aide contextuelle).
pub(crate) fn update_preset_row(state: &Rc<RefCell<State>>, widgets: &Rc<Widgets>) {
    let selected = widgets.presets.preset_combo.selected();
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
            .is_some_and(|p| p.params == state.borrow().params),
    };
    state.borrow_mut().updating = true;
    let text = if matches_params {
        subtitle
    } else {
        format!("{subtitle} (modifié)")
    };
    widgets.presets.preset_combo.set_subtitle(&text);
    state.borrow_mut().updating = false;
    sections::update_sections(widgets, &state.borrow().params);
    // La grille remplace la liste déroulante quand elle est présente : elle
    // porte donc elle aussi la légende, et l'état actif des vignettes.
    crate::wire::sync_preset_thumbs(state, widgets, selected);
    crate::wire::update_preset_caption(state, widgets, &text);
}
