//! Point d'entrée de la construction de la fenêtre.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;

use crate::State;
use crate::controls;
use crate::preview;
use crate::wire;

/// Construit la fenêtre complète et renvoie ses poignées.
pub(crate) fn build_ui(
    app: &adw::Application,
    state: Rc<RefCell<State>>,
) -> (adw::ApplicationWindow, Rc<controls::Widgets>) {
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("pointimg")
        .default_width(1240)
        .default_height(800)
        .build();

    let (paned, widgets) = controls::build();

    let header = adw::HeaderBar::new();
    header.set_title_widget(Some(&adw::WindowTitle::new(
        "pointimg",
        "Filtre pointilliste",
    )));
    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_content(Some(&paned));
    window.set_content(Some(&toolbar));

    wire::wire(&state, &widgets, &window);
    wire::install_dnd(&window, &state, &widgets);
    preview::connect_fit_signals(&state, &widgets);

    wire::sync_widgets(&state, &widgets);
    wire::update_sections(&widgets, &state.borrow().params);
    preview::refresh_preview(&state, &widgets);
    preview::refresh_actions(&state, &widgets);
    preview::update_status_labels(&state, &widgets);

    (window, widgets)
}
