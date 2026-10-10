//! Point d'entrée de la construction de la fenêtre.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;

use crate::State;
use crate::controls;
use crate::files;
use crate::preview;
use crate::recent;
use crate::shortcuts;
use crate::wire;

/// Construit la fenêtre complète et renvoie ses poignées.
pub(crate) fn build_ui(
    app: &adw::Application,
    state: Rc<RefCell<State>>,
) -> (adw::ApplicationWindow, Rc<controls::Widgets>) {
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("pointimg")
        .default_width(1280)
        .default_height(820)
        .build();

    let (toast_overlay, widgets) = controls::build();

    window.set_content(Some(&toast_overlay));

    // La fenêtre des raccourcis partage le même overlay de notifications.
    widgets.shortcuts.set_transient_for(Some(&window));
    widgets.shortcuts.set_modal(true);
    widgets.shortcuts.set_title(Some("Raccourcis clavier"));
    widgets.shortcuts.set_default_size(520, 560);
    widgets
        .shortcuts
        .set_content(Some(&shortcuts::build_page()));

    wire::install_actions(&state, &widgets, &window);
    wire::wire(&state, &widgets, &window);
    wire::install_dnd(&window, &state, &widgets);
    wire::install_shortcuts(&state, &widgets, &window);
    preview::connect_fit_signals(&state, &widgets);
    install_breakpoint(&window, &widgets);
    show_recent_images(&state, &widgets);

    crate::sections::sync_widgets(&state, &widgets);
    crate::sections::update_sections(&widgets, &state.borrow().params);
    // La légende du preset n'est écrite que par ce chemin, appelé aux
    // changements : sans cet appel au démarrage, elle resterait vide tant que
    // l'utilisateur ne toucherait à aucun réglage.
    files::update_preset_row(&state, &widgets);
    preview::refresh_preview(&state, &widgets);
    preview::refresh_actions(&state, &widgets);
    preview::update_status_labels(&state, &widgets);

    (window, widgets)
}

/// Propose les images récentes dans l'état d'accueil.
///
/// `GtkRecentManager` est déjà rempli si l'application a été ouverte avec une
/// image : il suffit de le lire une fois, au démarrage.
fn show_recent_images(state: &Rc<RefCell<State>>, widgets: &Rc<controls::Widgets>) {
    let Some(group) = recent::build_recent_widget(state, widgets) else {
        return;
    };
    widgets.empty_extra.append(&group);
}

/// Sous 900 px de large, l'aperçu et le panneau de réglages ne peuvent plus
/// tenir côte à côte : le panneau se replie au-dessus de l'aperçu et un bouton
/// de l'en-tête permet de le retrouver.
///
/// La condition est déclarée par l'adwaita et réévaluée par GTK à chaque
/// redimensionnement — aucun code à maintenir ici, contrairement à un
/// `connect_map` sur la largeur de la fenêtre.
fn install_breakpoint(window: &adw::ApplicationWindow, widgets: &controls::Widgets) {
    let condition = adw::BreakpointCondition::parse("max-width: 900px")
        .expect("condition de point de rupture valide");
    let breakpoint = adw::Breakpoint::new(condition);
    breakpoint.add_setter(&widgets.split, "collapsed", Some(&true.to_value()));
    breakpoint.add_setter(&widgets.panel_toggle, "visible", Some(&true.to_value()));
    breakpoint.add_setter(&widgets.panel_toggle, "active", Some(&true.to_value()));
    window.add_breakpoint(breakpoint);

    // Le bouton et le panneau se reflètent l'un l'autre dans les deux sens : le
    // point de rupture active le bouton en entrant dans la zone étroite, un clic
    // referme le panneau, et le geste de balayage le rouvre.
    widgets.panel_toggle.connect_toggled({
        let split = widgets.split.clone();
        move |toggle| split.set_collapsed(!toggle.is_active())
    });
    widgets.split.connect_collapsed_notify({
        let toggle = widgets.panel_toggle.clone();
        move |split| toggle.set_active(!split.is_collapsed())
    });
}
