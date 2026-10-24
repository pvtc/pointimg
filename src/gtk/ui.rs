//! Entry point of the window construction.

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

/// Builds the complete window and returns its handles.
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

    // The shortcuts window shares the same notification overlay.
    widgets.window.shortcuts.set_transient_for(Some(&window));
    widgets.window.shortcuts.set_modal(true);
    widgets
        .window
        .shortcuts
        .set_title(Some(pointimg::frontend::t("Raccourcis clavier")));
    widgets.window.shortcuts.set_default_size(520, 560);
    widgets
        .window
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
    // The caption of the preset is only written through this path, called on
    // changes: without this call at startup, it would stay empty as long as
    // the user does not touch any setting.
    files::update_preset_row(&state, &widgets);
    preview::refresh_preview(&state, &widgets);
    preview::refresh_actions(&state, &widgets);
    preview::update_status_labels(&state, &widgets);

    (window, widgets)
}

/// Offers the recent images in the welcome state.
///
/// `GtkRecentManager` is already filled if the application was opened with an
/// image: it is enough to read it once, at startup.
fn show_recent_images(state: &Rc<RefCell<State>>, widgets: &Rc<controls::Widgets>) {
    let Some(group) = recent::build_recent_widget(state, widgets) else {
        return;
    };
    widgets.status_bar.empty_extra.append(&group);
}

/// Below 900 px of width, the preview and the settings panel can no longer sit
/// side by side: the panel folds above the preview and a button in the header
/// makes it findable again.
///
/// The condition is declared by adwaita and re-evaluated by GTK at every
/// resize — no code to maintain here, unlike a `connect_map` on the width of
/// the window.
fn install_breakpoint(window: &adw::ApplicationWindow, widgets: &controls::Widgets) {
    let condition = adw::BreakpointCondition::parse("max-width: 900px")
        .expect("condition de point de rupture valide");
    let breakpoint = adw::Breakpoint::new(condition);
    breakpoint.add_setter(&widgets.window.split, "collapsed", Some(&true.to_value()));
    breakpoint.add_setter(
        &widgets.window.panel_toggle,
        "visible",
        Some(&true.to_value()),
    );
    breakpoint.add_setter(
        &widgets.window.panel_toggle,
        "active",
        Some(&true.to_value()),
    );
    window.add_breakpoint(breakpoint);

    // The button and the panel mirror each other both ways: the breakpoint
    // activates the button when entering the narrow area, a click folds the
    // panel back, and the swipe gesture reopens it.
    widgets.window.panel_toggle.connect_toggled({
        let split = widgets.window.split.clone();
        move |toggle| split.set_collapsed(!toggle.is_active())
    });
    widgets.window.split.connect_collapsed_notify({
        let toggle = widgets.window.panel_toggle.clone();
        move |split| toggle.set_active(!split.is_collapsed())
    });
}
