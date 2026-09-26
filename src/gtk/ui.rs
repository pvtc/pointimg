//! Point d'entrée de la construction de la fenêtre.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;

use crate::State;
use crate::controls;
use crate::preview;
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

    crate::sections::sync_widgets(&state, &widgets);
    crate::sections::update_sections(&widgets, &state.borrow().params);
    preview::refresh_preview(&state, &widgets);
    preview::refresh_actions(&state, &widgets);
    preview::update_status_labels(&state, &widgets);

    (window, widgets)
}
