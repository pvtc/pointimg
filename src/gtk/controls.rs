//! Building the widgets and the layout of the GTK4 window.
//!
//! The left panel is an `adw::PreferencesPage`: each setting occupies a
//! single line (`adw::SpinRow` = keyboard entry at the real ranges of
//! `validate_params`), the sections stay visible and the main actions live
//! in the header rather than at the bottom of a panel that has to scroll.
//!
//! The wide-range settings (number of points, radii) are doubled with a
//! `GtkScale`: the `SpinRow` remains the exact-value tool, the slider the
//! mouse one — impossible with the `SpinButton` wheel alone, whose step is
//! fixed.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use gtk::glib;

use pointimg::frontend::Curve;

/// Conversion between a slider position (0 → 1) and the filter value.
///
/// `Copy` to be copied into the GTK callbacks, which cannot capture the
/// widget itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SliderSpec {
    curve: Curve,
    start: f64,
    end: f64,
    digits: u32,
}

impl SliderSpec {
    /// Slider position corresponding to a filter value.
    pub(crate) fn position(self, value: f64) -> f64 {
        self.curve.position(value, self.start, self.end)
    }

    /// Filter value corresponding to a slider position, rounded to the
    /// precision of the neighboring entry field: without that, dragging
    /// would leave `0.25317937` in the settings.
    pub(crate) fn value(self, position: f64) -> f64 {
        let factor = 10f64.powi(self.digits as i32);
        let value = (self.curve.value(position, self.start, self.end) * factor).round() / factor;
        value.clamp(self.start, self.end)
    }
}

/// Horizontal slider doubling an `adw::SpinRow`.
#[derive(Clone)]
pub(crate) struct ValueSlider {
    pub scale: gtk::Scale,
    pub spec: SliderSpec,
    /// Row hosting the slider, to hide it along with the setting it doubles
    /// (the slider alone would not be enough: the row would stay visible,
    /// empty of its slider).
    row: Option<adw::ActionRow>,
}

impl ValueSlider {
    /// `digits` is the precision of the associated `SpinRow`, so that both
    /// representations agree.
    pub(crate) fn new(start: f64, end: f64, digits: u32, curve: Curve) -> Self {
        // The slider works in relative position: GTK has no logarithmic
        // scale, and a linear travel over 0.0005 → 1.0 would make the small
        // radii impossible to reach with the mouse.
        let scale = gtk::Scale::with_range(gtk::Orientation::Horizontal, 0.0, 1.0, 0.002);
        scale.set_draw_value(false);
        scale.set_value_pos(gtk::PositionType::Right);
        scale.set_has_origin(false);
        scale.set_hexpand(true);
        scale.set_digits(3);
        // A `GtkScale` captures the wheel: without this interception,
        // scrolling the settings panel silently changed the values.
        forward_wheel_to_scroller(&scale);
        Self {
            scale,
            spec: SliderSpec {
                curve,
                start,
                end,
                digits,
            },
            row: None,
        }
    }

    /// Places the handle on a filter value.
    pub(crate) fn set_value(&self, value: f64) {
        self.scale.set_value(self.spec.position(value));
    }

    /// Marks the position of the other end of a pair (one mark at a time:
    /// `gtk_scale_add_mark` returns no handle).
    pub(crate) fn set_bound(&self, other: f64) {
        self.scale.clear_marks();
        self.scale
            .add_mark(self.spec.position(other), gtk::PositionType::Bottom, None);
    }

    /// Row hosting the slider.
    ///
    /// `adw::PreferencesGroup` only accepts `AdwPreferencesRow`s, and the
    /// suffix of an `AdwActionRow` is not extensible: the slider therefore
    /// takes the right part of the row, like the sliders of the GNOME
    /// settings.
    ///
    /// Neither title nor subtitle: the row doubles a `SpinRow` sitting just
    /// above, whose label and help already name the setting. The slider is
    /// only a second way to adjust it, not a setting in its own right —
    /// the label and above all the word "curseur", already present in the
    /// right half of the window, taught nothing. The scale is explained in
    /// the tooltip.
    pub(crate) fn row(&mut self, tooltip: &str) -> adw::ActionRow {
        let row = adw::ActionRow::new();
        row.set_activatable(false);
        row.set_tooltip_text(Some(tooltip));
        row.add_suffix(&self.scale);
        self.row = Some(row.clone());
        row
    }

    /// Hides the whole row when the doubled setting is not visible.
    pub(crate) fn set_row_visible(&self, visible: bool) {
        if let Some(row) = &self.row {
            row.set_visible(visible);
        }
    }
}

/// A slider row for a min/max pair.
///
/// GTK has no two-handle slider (this is also true of libadwaita, whose
/// `AdwSliderRow` only exists from 1.6). The two sliders are therefore
/// stacked, each marking the position of the other end: reading the range
/// and its spread stays immediate, and the numeric entry happens in the two
/// `SpinRow`s above.
fn pair_row(low: &mut ValueSlider, high: &mut ValueSlider, tooltip: &str) -> adw::ActionRow {
    let row = adw::ActionRow::new();
    row.set_activatable(false);
    row.set_tooltip_text(Some(tooltip));
    let box_ = gtk::Box::new(gtk::Orientation::Vertical, 0);
    box_.set_hexpand(true);
    for (label, slider) in [("min", &*low), ("max", &*high)] {
        let line = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        line.set_hexpand(true);
        let caption = gtk::Label::new(Some(label));
        caption.add_css_class("dim-label");
        caption.set_width_chars(4);
        caption.set_xalign(0.0);
        line.append(&caption);
        line.append(&slider.scale);
        box_.append(&line);
    }
    row.add_suffix(&box_);
    // The row is remembered by the low end: it is the one driving the
    // visibility of the pair.
    low.row = Some(row.clone());
    row
}

/// Scopes of "reset": one button per section of the panel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ResetTarget {
    Presets,
    Algorithm,
    Shape,
    Render,
    All,
}

/// A section reset button.
#[derive(Clone)]
pub(crate) struct GroupReset {
    pub button: gtk::Button,
    pub target: ResetTarget,
}

/// A preview pane: caption + scrolling area.
#[derive(Clone)]
pub(crate) struct Pane {
    pub root: gtk::Box,
    pub scroll: gtk::ScrolledWindow,
    pub image: gtk::Image,
}

/// The "comparaison" pane: one image, two layers, one cursor.
#[derive(Clone)]
pub(crate) struct ComparePane {
    pub root: gtk::Box,
    pub scroll: gtk::ScrolledWindow,
    pub area: gtk::DrawingArea,
    pub data: Rc<RefCell<crate::compare::CompareData>>,
}

/// Handles towards all the widgets controlled by the application.
/// Window: split view, notification overlay, shortcuts.
#[derive(Clone)]
pub(crate) struct WindowWidgets {
    pub window_title: adw::WindowTitle,
    pub menu: gtk::MenuButton,
    pub split: adw::OverlaySplitView,
    pub panel_toggle: gtk::ToggleButton,
    pub open: gtk::Button,
    pub progress: gtk::ProgressBar,
    pub cancel: gtk::Button,
    pub undo: gtk::Button,
    pub redo: gtk::Button,
    pub export: gtk::MenuButton,
    pub shortcuts: adw::Window,
    pub toast_overlay: adw::ToastOverlay,
    pub locale_combo: adw::ComboRow,
}

/// Built-in presets, opening of files and folders.
#[derive(Clone)]
pub(crate) struct PresetWidgets {
    pub preset_combo: adw::ComboRow,
    pub preset_thumbs: Vec<gtk::Widget>,
    pub preset_caption: gtk::Label,
    pub load_preset: gtk::Button,
    pub save_preset: gtk::Button,
    pub open_folder: gtk::Button,
    pub folder_label: gtk::Label,
    pub folder_row: adw::ActionRow,
}

/// Placement algorithm and common settings.
#[derive(Clone)]
pub(crate) struct AlgorithmWidgets {
    pub algo: adw::ComboRow,
    pub variance: adw::SpinRow,
    pub variance_slider: ValueSlider,
    pub min_radius: adw::SpinRow,
    pub max_radius: adw::SpinRow,
    pub min_radius_slider: ValueSlider,
    pub max_radius_slider: ValueSlider,
    pub max_boost: adw::SpinRow,
    pub max_boost_slider: ValueSlider,
    pub grid_group: adw::PreferencesGroup,
    pub cols: adw::SpinRow,
    pub cols_slider: ValueSlider,
    pub grid_angle: adw::SpinRow,
    pub points_group: adw::PreferencesGroup,
    pub num_points: adw::SpinRow,
    pub num_points_slider: ValueSlider,
    pub iterations: adw::SpinRow,
    pub iterations_slider: ValueSlider,
}

/// Settings specific to the screen (halftone).
#[derive(Clone)]
pub(crate) struct HalftoneWidgets {
    pub halftone_group: adw::PreferencesGroup,
    pub halftone_mode: adw::ComboRow,
    pub dominant_box: gtk::ListBox,
    pub dominant_n: adw::SpinRow,
    pub dominant_angle: adw::SpinRow,
    pub screening: adw::ComboRow,
    pub halftone_freq: adw::SpinRow,
    pub halftone_freq_slider: ValueSlider,
    pub halftone_min_radius: adw::SpinRow,
    pub halftone_max_dot: adw::SpinRow,
    pub halftone_min_slider: ValueSlider,
    pub halftone_max_slider: ValueSlider,
}

/// Dot shape and palette.
#[derive(Clone)]
pub(crate) struct DotWidgets {
    pub shape: adw::ComboRow,
    pub ellipse_box: gtk::ListBox,
    pub ellipse_aspect: adw::SpinRow,
    pub ellipse_angle: adw::SpinRow,
    pub polygon_box: gtk::ListBox,
    pub polygon_sides: adw::SpinRow,
    pub use_palette: adw::SwitchRow,
    pub palette_size: adw::SpinRow,
    pub dithering: adw::ComboRow,
}

/// Background, gamma correction, seed.
#[derive(Clone)]
pub(crate) struct RenderWidgets {
    pub bg_button: gtk::ColorDialogButton,
    pub transparent: adw::SwitchRow,
    pub gamma: adw::SwitchRow,
    pub use_seed: adw::SwitchRow,
    pub seed: gtk::SpinButton,
    pub seed_row: adw::ActionRow,
    pub vary_seed: gtk::Button,
}

/// Preview quality and per-section resets.
#[derive(Clone)]
pub(crate) struct ComputeWidgets {
    pub quality: adw::ComboRow,
    pub recalc: gtk::Button,
    pub reset: gtk::Button,
    pub group_resets: Vec<GroupReset>,
}

/// Toolbar and preview panes.
#[derive(Clone)]
pub(crate) struct PreviewWidgets {
    pub view_mode: gtk::DropDown,
    pub zoom_out: gtk::Button,
    pub zoom_label: gtk::Label,
    pub zoom_in: gtk::Button,
    pub zoom_one: gtk::ToggleButton,
    pub zoom_fit: gtk::ToggleButton,
    pub busy: gtk::Spinner,
    pub export_sizes: gtk::Label,
    pub busy_label: gtk::Label,
    pub quality_badge: gtk::Label,
    pub src_pane: Pane,
    pub dst_pane: Pane,
    pub density_pane: Pane,
    pub compare_pane: ComparePane,
}

/// Welcome page and status bar.
#[derive(Clone)]
pub(crate) struct StatusWidgets {
    pub empty_page: adw::StatusPage,
    pub empty_extra: gtk::Box,
    pub open_empty: gtk::Button,
    pub sample_button: gtk::Button,
    pub memory: gtk::Label,
    pub timing: gtk::Label,
    pub status: gtk::Label,
}

/// All the widgets, grouped by settings section.
///
/// `widgets.algorithm.variance` reads as what it is, and a widget added to
/// a section files itself. Section names cannot collide with a field name —
/// that is what ruled out `shape` and `status`, both fields *and* sections.
#[derive(Clone)]
pub(crate) struct Widgets {
    pub window: WindowWidgets,
    pub presets: PresetWidgets,
    pub algorithm: AlgorithmWidgets,
    pub halftone: HalftoneWidgets,
    pub dots: DotWidgets,
    pub render: RenderWidgets,
    pub compute: ComputeWidgets,
    pub preview: PreviewWidgets,
    pub status_bar: StatusWidgets,
}

fn algos() -> [&'static str; 5] {
    [
        pointimg::frontend::t("Grille"),
        "K-means",
        "Voronoi (Lloyd)",
        "Quadtree",
        "Halftone (rosette)",
    ]
}
fn algo_help() -> [&'static str; 5] {
    [
        pointimg::frontend::t("Grille régulière : une cellule = un point. Rapide et prévisible."),
        pointimg::frontend::t(
            "k-means spatial : les points se répartissent selon la couleur et la position.",
        ),
        pointimg::frontend::t(
            "Lloyd pondéré : répartition régulière guidée par le détail. Rendu le plus « naturel ».",
        ),
        pointimg::frontend::t(
            "Subdivision par contraste : davantage de points là où l'image est détaillée.",
        ),
        pointimg::frontend::t(
            "Trame d'encre : rosette CMJN ou couleurs dominantes, style offset print.",
        ),
    ]
}
fn shapes() -> [&'static str; 4] {
    [
        pointimg::frontend::t("Cercle"),
        pointimg::frontend::t("Carré"),
        "Ellipse",
        pointimg::frontend::t("Polygone"),
    ]
}
fn views() -> [&'static str; 5] {
    [
        pointimg::frontend::t("Comparaison"),
        pointimg::frontend::t("Côte à côte"),
        pointimg::frontend::t("Résultat"),
        "Source",
        pointimg::frontend::t("Carte de densité"),
    ]
}
fn qualities() -> [&'static str; 3] {
    [
        pointimg::frontend::t("Rapide (1/2)"),
        pointimg::frontend::t("Rapide (1/4)"),
        pointimg::frontend::t("Pleine résolution"),
    ]
}

/// Composes the upper toolbar.
///
/// Everything in it is already built: this function only dresses the buttons
/// (icon, label, tooltip) and orders them. Extracting it removes fifty lines
/// of formatting from the middle of `build()`, where they hid the structure
/// of the window.
///
/// `adw::ButtonContent` imposes the content (icon + label): `set_label`
/// followed by `set_icon_name` on a GtkButton keeps only the icon.
///
/// It takes `&Widgets` and not nine parameters: the fields it touches are
/// all in the `window` and `presets` sections, and a tenth button does not
/// add one more parameter.
fn header_bar(widgets: &Widgets) -> adw::HeaderBar {
    let Widgets {
        window:
            WindowWidgets {
                window_title,
                menu,
                panel_toggle,
                open,
                progress,
                cancel,
                undo,
                redo,
                export,
                ..
            },
        presets: PresetWidgets { open_folder, .. },
        ..
    } = widgets.clone();
    // `adw::ButtonContent` imposes the content (icon + label): `set_label`
    // followed by `set_icon_name` on a GtkButton keeps only the icon.
    set_button_content(
        &open,
        "document-open-symbolic",
        pointimg::frontend::t("Ouvrir…"),
    );
    open.set_tooltip_text(Some(pointimg::frontend::t("Ouvrir une image (Ctrl+O)")));
    set_button_content(
        &open_folder,
        "folder-open-symbolic",
        pointimg::frontend::t("Dossier…"),
    );
    open_folder.set_tooltip_text(Some(pointimg::frontend::t(
        "Ouvrir un dossier et passer d'une image à l'autre avec ← et → (Ctrl+Maj+O)",
    )));
    progress.set_show_text(true);
    progress.set_valign(gtk::Align::Center);
    progress.set_size_request(200, -1);
    progress.set_visible(false);
    set_button_content(
        &cancel,
        "process-stop-symbolic",
        pointimg::frontend::t("Annuler"),
    );
    cancel.set_tooltip_text(Some(pointimg::frontend::t("Annuler le calcul en cours")));
    set_icon_button(
        &undo,
        "edit-undo-symbolic",
        pointimg::frontend::t("Annuler la dernière modification (Ctrl+Z)"),
    );
    set_icon_button(
        &redo,
        "edit-redo-symbolic",
        pointimg::frontend::t("Rétablir la modification annulée (Ctrl+Y)"),
    );
    set_button_content(
        &export,
        "document-save-symbolic",
        pointimg::frontend::t("Exporter"),
    );
    export.add_css_class("suggested-action");
    export.set_tooltip_text(Some(pointimg::frontend::t("Exporter l'image (Ctrl+S)")));
    // The button only offers the exports: the full menu (folder, presets,
    // display, shortcuts) is under the hamburger button.
    export.set_menu_model(Some(&crate::wire::export_menu_model()));
    menu.set_tooltip_text(Some(pointimg::frontend::t("Menu principal")));

    let header = adw::HeaderBar::new();
    header.set_title_widget(Some(&window_title));
    header.pack_start(&menu);
    header.pack_start(&panel_toggle);
    header.pack_start(&open);
    header.pack_start(&open_folder);
    header.pack_end(&export);
    header.pack_end(&redo);
    header.pack_end(&undo);
    header.pack_end(&cancel);
    header.pack_end(&progress);

    header
}

/// Composes the status bar: three measurements, a quality badge, a message.
///
/// The order matters — the status message is last because it is the only one
/// that stretches (`hexpand`) and has to stay visible. Returns the container
/// ready to be placed at the bottom of the window.
fn status_bar(
    memory: &gtk::Label,
    export_sizes: &gtk::Label,
    timing: &gtk::Label,
    quality_badge: &gtk::Label,
    status: &gtk::Label,
) -> gtk::Box {
    let status_bar = gtk::Box::new(gtk::Orientation::Horizontal, 14);
    status_bar.set_margin_top(4);
    status_bar.set_margin_bottom(6);
    status_bar.set_margin_start(12);
    status_bar.set_margin_end(12);
    for label in [memory, export_sizes, timing] {
        label.set_xalign(0.0);
        label.add_css_class("dim-label");
        // The list of formats is long ("Export : PNG 1.4 MB · WebP — · JPEG…")
        // and does not fit in a narrow window: without elision it was cut off
        // short, with no sign of continuation.
        label.set_ellipsize(gtk::pango::EllipsizeMode::End);
        // The label must not expand either: it takes what it needs and
        // yields the rest to the status message, which is the field that
        // changes.
        label.set_max_width_chars(40);
        status_bar.append(label);
    }
    status_bar.append(quality_badge);
    status.set_xalign(0.0);
    status.set_hexpand(true);
    status.set_wrap(false);
    status.set_ellipsize(gtk::pango::EllipsizeMode::End);
    status_bar.append(status);
    status_bar
}

/// Builds the complete layout: header, settings panel, preview.
/// Builds the complete layout: header, settings panel, preview.
pub(crate) fn build() -> (adw::ToastOverlay, Rc<Widgets>) {
    // ── Base widgets ──────────────────────────────────────────────────────────
    let open = gtk::Button::with_label(pointimg::frontend::t("Ouvrir…"));
    let open_empty = gtk::Button::with_label(pointimg::frontend::t("Ouvrir une image…"));
    let sample_button = gtk::Button::with_label(pointimg::frontend::t("Utiliser un exemple"));
    let window_title = adw::WindowTitle::new("pointimg", pointimg::frontend::t("Aucune image"));
    let progress = gtk::ProgressBar::new();
    let cancel = gtk::Button::with_label(pointimg::frontend::t("Annuler"));
    let undo = gtk::Button::new();
    let redo = gtk::Button::new();
    let export = gtk::MenuButton::new();
    let menu = gtk::MenuButton::new();
    menu.set_menu_model(Some(&crate::wire::menu_model()));
    // Reveals the settings panel when it is folded away (narrow window): the
    // button only exists in that case, see the breakpoint further down. A
    // field of `Widgets`, hence declared before building the literal.
    let panel_toggle = gtk::ToggleButton::new();
    panel_toggle.set_icon_name("sidebar-show-symbolic");
    panel_toggle.set_tooltip_text(Some(pointimg::frontend::t("Afficher les réglages")));
    panel_toggle.set_visible(false);
    let toast_overlay = adw::ToastOverlay::new();
    let shortcuts = adw::Window::new();

    let preset_combo = combo_row("Preset", &[pointimg::frontend::t("Personnalisé")], None);
    let load_preset = gtk::Button::with_label(pointimg::frontend::t("Charger un preset…"));
    let save_preset = gtk::Button::with_label(pointimg::frontend::t("Sauver le preset…"));
    let open_folder = gtk::Button::new();
    let folder_label = gtk::Label::new(None);

    let algo = combo_row(
        pointimg::frontend::t("Algorithme"),
        &algos(),
        Some(algo_help()[0]),
    );
    let variance = spin_row(
        pointimg::frontend::t("Sensibilité variance"),
        Some(pointimg::frontend::t("0 = uniforme · 1 = détail maximal")),
        pointimg::frontend::t(
            "Redistribue les points vers les zones détaillées. 0 = répartition uniforme, 1 = maximum de détail.",
        ),
        0.0,
        1.0,
        0.01,
        2,
    );
    let min_radius = spin_row(
        pointimg::frontend::t("Rayon min"),
        Some(pointimg::frontend::t("fraction du petit côté")),
        pointimg::frontend::t(
            "Rayon minimal des points, en fraction du plus petit côté de l'image.",
        ),
        0.0005,
        1.0,
        0.0005,
        4,
    );
    let max_radius = spin_row(
        pointimg::frontend::t("Rayon max"),
        Some(pointimg::frontend::t("fraction du petit côté")),
        pointimg::frontend::t(
            "Rayon maximal des points, en fraction du plus petit côté de l'image.",
        ),
        0.0005,
        1.0,
        0.0025,
        4,
    );
    let max_boost = spin_row(
        pointimg::frontend::t("Boost zones uniformes"),
        Some("× maximum"),
        pointimg::frontend::t(
            "Agrandit les points dans les zones sans détail. 1 = aucun agrandissement, 2.5 = réglage par défaut.",
        ),
        1.0,
        8.0,
        0.1,
        2,
    );
    // Sliders: the `SpinButton` wheel has a fixed step (0.1), which makes
    // the 4 decimals of the min radius painful to reach. The slider is
    // adjusted with the mouse; the `SpinRow` stays the exact-value tool.
    let mut variance_slider = ValueSlider::new(0.0, 1.0, 2, Curve::Linear);
    let mut min_radius_slider = ValueSlider::new(0.0005, 1.0, 4, Curve::Log);
    let mut max_radius_slider = ValueSlider::new(0.0005, 1.0, 4, Curve::Log);
    let mut max_boost_slider = ValueSlider::new(1.0, 8.0, 2, Curve::Log);
    let cols = spin_row(
        pointimg::frontend::t("Colonnes"),
        Some("2 – 8 192"),
        pointimg::frontend::t("Nombre de cellules de la grille (algorithme Grille)."),
        2.0,
        8192.0,
        1.0,
        0,
    );
    let grid_angle = spin_row(
        pointimg::frontend::t("Angle de la grille"),
        Some("°"),
        pointimg::frontend::t("Rotation de la grille pour un effet de trame (algorithme Grille)."),
        -180.0,
        180.0,
        1.0,
        0,
    );
    let num_points = spin_row(
        pointimg::frontend::t("Nombre de points"),
        Some("10 – 100 000"),
        pointimg::frontend::t("Nombre de points générés (K-means, Voronoï, Quadtree)."),
        10.0,
        100_000.0,
        10.0,
        0,
    );
    let mut cols_slider = ValueSlider::new(2.0, 8192.0, 0, Curve::Log);
    let mut num_points_slider = ValueSlider::new(10.0, 100_000.0, 0, Curve::Log);
    let iterations = spin_row(
        pointimg::frontend::t("Itérations"),
        Some("1 – 100"),
        pointimg::frontend::t(
            "Itérations de relaxation de Lloyd (K-means, Voronoï). Davantage = plus régulier, plus lent.",
        ),
        1.0,
        100.0,
        1.0,
        0,
    );
    // Linear, as in the egui interface: over 1 → 100, a logarithmic scale
    // wastes 80% of the travel between 1 and 10, whereas that is precisely
    // where the setting is decided ("5 or 10 iterations?").
    let mut iterations_slider = ValueSlider::new(1.0, 100.0, 0, Curve::Linear);

    let halftone_mode = combo_row(
        "Mode",
        &[
            pointimg::frontend::t("CMJN (rosette)"),
            pointimg::frontend::t("Couleurs dominantes"),
        ],
        None,
    );
    let dominant_n = spin_row(
        pointimg::frontend::t("Canaux"),
        Some(pointimg::frontend::t("couleurs dominantes")),
        pointimg::frontend::t("Nombre de couleurs de la trame (une trame par couleur)."),
        2.0,
        32.0,
        1.0,
        0,
    );
    let dominant_angle = spin_row(
        pointimg::frontend::t("Angle de base"),
        Some("°"),
        pointimg::frontend::t(
            "Angle de la première trame ; les suivantes sont réparties régulièrement pour éviter le moiré.",
        ),
        0.0,
        180.0,
        1.0,
        0,
    );
    let dominant_box = rows_box(&[&dominant_n, &dominant_angle]);
    let screening = combo_row(
        "Screening",
        &[pointimg::frontend::t("AM (grille)"), "FM (blue noise)"],
        Some(pointimg::frontend::t(
            "AM : la taille du dot code l'encre. FM : c'est la densité qui la code.",
        )),
    );
    let halftone_freq = spin_row(
        pointimg::frontend::t("Fréquence de trame"),
        Some("20 – 400"),
        pointimg::frontend::t(
            "Nombre de cellules de trame sur le plus petit côté. Plus c'est élevé, plus le pointillé est fin.",
        ),
        4.0,
        400.0,
        1.0,
        0,
    );
    let halftone_min_radius = spin_row(
        pointimg::frontend::t("Rayon min"),
        Some(pointimg::frontend::t("fraction du petit côté")),
        pointimg::frontend::t("Rayon minimal des dots de trame, en fraction du plus petit côté."),
        0.0,
        1.0,
        0.001,
        3,
    );
    let halftone_max_dot = spin_row(
        pointimg::frontend::t("Rayon max"),
        Some(pointimg::frontend::t("× pas de trame")),
        pointimg::frontend::t(
            "Rayon maximal, en fraction du pas de trame. Au-delà de 1, les dots se touchent.",
        ),
        0.05,
        10.0,
        0.05,
        2,
    );
    let mut halftone_freq_slider = ValueSlider::new(4.0, 400.0, 0, Curve::Log);
    // The 0 → 10 range goes to square root: linearly, the useful values
    // (0.002 and 0.85) would be crammed into the first tenth of the rail.
    //
    // The low end stops at 1.0, not at 10: `validate_params` caps
    // `halftone_min_radius_ratio` at 1.0 when `halftone_max_dot_ratio` rises
    // to 10. Both shared the same range, and the low handle crossed the cap
    // — a bare worker error, with no visible cause. The `SpinRow` just above
    // already indicated the right end.
    let mut halftone_min_slider = ValueSlider::new(0.0, 1.0, 3, Curve::Sqrt);
    let mut halftone_max_slider = ValueSlider::new(0.05, 10.0, 2, Curve::Sqrt);

    let shape = combo_row(pointimg::frontend::t("Forme"), &shapes(), None);
    let ellipse_aspect = spin_row(
        pointimg::frontend::t("Rapport largeur/hauteur"),
        None,
        pointimg::frontend::t("Rapport largeur/hauteur de l'ellipse."),
        0.1,
        10.0,
        0.1,
        2,
    );
    let ellipse_angle = spin_row(
        "Angle",
        Some("°"),
        pointimg::frontend::t("Rotation de l'ellipse en degrés."),
        -180.0,
        180.0,
        1.0,
        0,
    );
    let polygon_sides = spin_row(
        pointimg::frontend::t("Côtés"),
        Some("3 – 12"),
        pointimg::frontend::t("Nombre de côtés du polygone régulier."),
        3.0,
        12.0,
        1.0,
        0,
    );
    let ellipse_box = rows_box(&[&ellipse_aspect, &ellipse_angle]);
    let polygon_box = rows_box(&[&polygon_sides]);
    let use_palette = switch_row(
        pointimg::frontend::t("Palette réduite"),
        pointimg::frontend::t("Quantifie l'image vers quelques couleurs (effet sérigraphie)"),
    );
    let palette_size = spin_row(
        pointimg::frontend::t("Couleurs de la palette"),
        Some("2 – 256"),
        pointimg::frontend::t("Nombre de couleurs de la palette réduite."),
        2.0,
        256.0,
        1.0,
        0,
    );
    // Dithering is chosen among four algorithms: Floyd-Steinberg,
    // Riemersma, Bayer 4×4 and Bayer 8×8, in addition to "Aucun".
    let dithering = combo_row(
        "Dithering",
        &[
            pointimg::frontend::t("Aucun"),
            "Floyd-Steinberg",
            "Riemersma",
            "Bayer 4×4",
            "Bayer 8×8",
        ],
        Some(pointimg::frontend::t("Tramage de la palette quantifiée")),
    );

    let color_dialog = gtk::ColorDialog::builder()
        .with_alpha(false)
        .title(pointimg::frontend::t("Couleur de fond"))
        .build();
    let bg_button = gtk::ColorDialogButton::new(Some(color_dialog));
    bg_button.set_tooltip_text(Some(pointimg::frontend::t("Couleur de fond du rendu")));
    let transparent = switch_row(
        pointimg::frontend::t("Fond transparent (RGBA)"),
        pointimg::frontend::t("Alpha conservé à l'export PNG/WebP/TIFF"),
    );
    let gamma = switch_row(
        pointimg::frontend::t("Correction gamma"),
        pointimg::frontend::t("Moyennes en espace linéaire : évite les mi-tons trop sombres"),
    );
    let use_seed = switch_row(
        pointimg::frontend::t("Graine fixée"),
        pointimg::frontend::t("Deux exécutions identiques donnent le même résultat"),
    );
    let seed = gtk::SpinButton::with_range(0.0, 4_294_967_295.0, 1.0);
    seed.set_digits(0);
    seed.set_tooltip_text(Some(pointimg::frontend::t(
        "Graine du générateur aléatoire",
    )));
    forward_wheel_to_scroller(&seed);
    let vary_seed = gtk::Button::new();
    set_icon_button(
        &vary_seed,
        "view-refresh-symbolic",
        pointimg::frontend::t("Tirer une nouvelle graine et recalculer"),
    );

    let quality = combo_row(
        pointimg::frontend::t("Qualité de l'aperçu"),
        &qualities(),
        Some(pointimg::frontend::t(
            "Le mode rapide calcule sur une image réduite, puis repasse en pleine résolution.",
        )),
    );
    let recalc = gtk::Button::new();
    set_button_content(
        &recalc,
        "view-refresh-symbolic",
        pointimg::frontend::t("Recalculer"),
    );
    recalc.set_tooltip_text(Some(pointimg::frontend::t(
        "Recalculer maintenant, en pleine résolution (Espace)",
    )));
    let reset = gtk::Button::with_label(pointimg::frontend::t("Réinitialiser les réglages"));

    let view_mode = gtk::DropDown::from_strings(&views());
    view_mode.set_tooltip_text(Some(pointimg::frontend::t("Quel aperçu afficher")));
    let zoom_out = gtk::Button::from_icon_name("zoom-out-symbolic");
    zoom_out.set_tooltip_text(Some(pointimg::frontend::t("Dézoomer (−)")));
    let zoom_in = gtk::Button::from_icon_name("zoom-in-symbolic");
    zoom_in.set_tooltip_text(Some(pointimg::frontend::t("Zoomer (+)")));
    let zoom_label = gtk::Label::new(Some("100 %"));
    zoom_label.set_width_chars(5);
    zoom_label.set_xalign(0.5);
    zoom_label.add_css_class("dim-label");
    zoom_label.set_tooltip_text(Some(pointimg::frontend::t(
        "Zoom réellement appliqué à l'aperçu",
    )));
    let zoom_one = gtk::ToggleButton::with_label("1:1");
    zoom_one.set_tooltip_text(Some(pointimg::frontend::t("Revenir à 100 % (1)")));
    let zoom_fit = gtk::ToggleButton::with_label(pointimg::frontend::t("Ajuster"));
    zoom_fit.set_tooltip_text(Some(pointimg::frontend::t(
        "Ajuster l'aperçu à la fenêtre (0)",
    )));
    zoom_fit.set_active(true);
    let busy = gtk::Spinner::new();
    busy.set_valign(gtk::Align::Center);
    let busy_label = gtk::Label::new(None);
    let quality_badge = gtk::Label::new(None);
    quality_badge.add_css_class("dim-label");

    let src_pane = preview_pane("Source");
    let dst_pane = preview_pane(pointimg::frontend::t("Résultat"));
    let density_pane = preview_pane(pointimg::frontend::t("Carte de densité"));
    let compare_pane = crate::compare::build_pane();

    let empty_page = adw::StatusPage::new();
    let memory = gtk::Label::new(None);
    let export_sizes = gtk::Label::new(None);
    let timing = gtk::Label::new(None);
    let status = gtk::Label::new(None);

    // ── Settings panel ────────────────────────────────────────────────────────
    let page = adw::PreferencesPage::new();

    // Interface language: applied on the next launch (GTK rebuilds its labels
    // at build time), but persisted immediately.
    let interface_group = adw::PreferencesGroup::new();
    interface_group.set_title(pointimg::frontend::t("Interface"));
    let locale_combo = adw::ComboRow::new();
    locale_combo.set_title(pointimg::frontend::t("Langue"));
    let locale_items = gtk::StringList::new(&["English", "Français"]);
    locale_combo.set_model(Some(&locale_items));
    let current_locale = pointimg::frontend::locale();
    locale_combo.set_selected(match current_locale {
        pointimg::frontend::Locale::Fr => 1,
        _ => 0,
    });
    interface_group.add(&locale_combo);
    page.add(&interface_group);

    let preset_group = adw::PreferencesGroup::new();
    preset_group.set_title(pointimg::frontend::t("Réglages rapides"));
    let preset_names: Vec<String> = pointimg::filter::builtin_presets()
        .iter()
        .map(|p| p.name.to_string())
        .collect();
    let mut preset_items = vec![pointimg::frontend::t("Personnalisé")];
    preset_items.extend(preset_names.iter().map(String::as_str));
    preset_combo.set_model(Some(&gtk::StringList::new(&preset_items)));

    // Thumbnail grid: a preset is chosen on its render, not on its name.
    // Present only if the thumbnails are there — otherwise the drop-down
    // list, which remains the reference selector, does the job alone.
    let preset_thumbs: Vec<gtk::Widget> = pointimg::filter::builtin_presets()
        .iter()
        .enumerate()
        .filter_map(|(index, preset)| {
            let chemin = pointimg::frontend::preset_thumbnail(index)?;
            let image = gtk::Image::from_file(&chemin);
            image.set_pixel_size(72);
            let bouton = gtk::ToggleButton::new();
            bouton.set_child(Some(&image));
            bouton.add_css_class("flat");
            bouton.set_tooltip_text(Some(&format!("{}\n\n{}", preset.name, preset.description)));
            Some(bouton.upcast())
        })
        .collect();

    let preset_grid = gtk::FlowBox::new();
    preset_grid.set_selection_mode(gtk::SelectionMode::None);
    preset_grid.set_homogeneous(true);
    preset_grid.set_column_spacing(6);
    preset_grid.set_row_spacing(6);
    preset_grid.set_valign(gtk::Align::Center);
    preset_grid.set_max_children_per_line(3);
    for vignette in &preset_thumbs {
        preset_grid.append(vignette);
    }
    let preset_caption = gtk::Label::new(None);
    preset_caption.set_xalign(0.0);
    preset_caption.set_wrap(true);
    preset_caption.add_css_class("dim-label");

    if preset_thumbs.is_empty() {
        preset_group.add(&preset_combo);
    } else {
        preset_group.add(&preset_grid);
        preset_group.add(&preset_caption);
    }

    let files_buttons = button_row();
    files_buttons.append(&load_preset);
    files_buttons.append(&save_preset);
    preset_group.add(&files_buttons);

    let folder_row = adw::ActionRow::new();
    folder_row.set_title(pointimg::frontend::t("Images du dossier"));
    folder_row.set_visible(false);
    folder_label.set_xalign(1.0);
    folder_label.add_css_class("dim-label");
    folder_row.add_suffix(&folder_label);
    preset_group.add(&folder_row);
    page.add(&preset_group);

    let algo_group = adw::PreferencesGroup::new();
    algo_group.set_title(pointimg::frontend::t("Algorithme"));
    algo_group.add(&algo);
    algo_group.add(&variance);
    algo_group.add(&variance_slider.row(pointimg::frontend::t(
        "Ajuster la sensibilité à la souris ; l'infobulle du réglage explique son effet",
    )));
    algo_group.add(&min_radius);
    algo_group.add(&max_radius);
    algo_group.add(&pair_row(
        &mut min_radius_slider,
        &mut max_radius_slider,
        "Curseurs des bornes min et max. Échelle logarithmique ; le repère \
         marque la position de l'autre borne.",
    ));
    algo_group.add(&max_boost);
    algo_group.add(&max_boost_slider.row(pointimg::frontend::t(
        "Ajuster l'agrandissement des points en zones sans détail",
    )));
    page.add(&algo_group);

    let grid_group = adw::PreferencesGroup::new();
    grid_group.set_title(pointimg::frontend::t("Grille"));
    grid_group.add(&cols);
    grid_group.add(&cols_slider.row(pointimg::frontend::t(
        "Nombre de cellules de la grille, de 2 à 8 192, en échelle logarithmique",
    )));
    grid_group.add(&grid_angle);
    page.add(&grid_group);

    let points_group = adw::PreferencesGroup::new();
    points_group.set_title(pointimg::frontend::t("Densité de points"));
    points_group.add(&num_points);
    points_group.add(&num_points_slider.row(pointimg::frontend::t(
        "Nombre de points, de 10 à 100 000, en échelle logarithmique",
    )));
    points_group.add(&iterations);
    points_group.add(&iterations_slider.row(pointimg::frontend::t(
        "Passes de Lloyd ; au-delà d'une trentaine, le gain devient marginal",
    )));
    page.add(&points_group);

    let halftone_group = adw::PreferencesGroup::new();
    halftone_group.set_title(pointimg::frontend::t("Trame"));
    halftone_group.set_description(Some(pointimg::frontend::t(
        "Séparation de l'image en canaux d'encre, une trame par canal.",
    )));
    halftone_group.add(&halftone_mode);
    halftone_group.add(&dominant_box);
    halftone_group.add(&screening);
    halftone_group.add(&halftone_freq);
    halftone_group.add(&halftone_freq_slider.row(pointimg::frontend::t(
        "Fréquence de trame, de 4 à 400 cellules, en échelle logarithmique",
    )));
    halftone_group.add(&halftone_min_radius);
    halftone_group.add(&halftone_max_dot);
    halftone_group.add(&pair_row(
        &mut halftone_min_slider,
        &mut halftone_max_slider,
        pointimg::frontend::t("Curseurs des bornes min et max du halftone ; le repère marque la position de l'autre borne"),
    ));
    page.add(&halftone_group);

    let shape_group = adw::PreferencesGroup::new();
    shape_group.set_title(pointimg::frontend::t("Forme des points"));
    shape_group.add(&shape);
    shape_group.add(&ellipse_box);
    shape_group.add(&polygon_box);
    shape_group.add(&use_palette);
    shape_group.add(&palette_size);
    shape_group.add(&dithering);
    page.add(&shape_group);

    let render_group = adw::PreferencesGroup::new();
    render_group.set_title(pointimg::frontend::t("Rendu"));
    let bg_row = adw::ActionRow::new();
    bg_row.set_title(pointimg::frontend::t("Couleur de fond"));
    bg_row.set_subtitle(pointimg::frontend::t(
        "Appliquée au rendu et aux zones transparentes de la source",
    ));
    bg_row.add_suffix(&bg_button);
    render_group.add(&bg_row);
    render_group.add(&transparent);
    render_group.add(&gamma);
    render_group.add(&use_seed);
    let seed_row = adw::ActionRow::new();
    seed_row.set_title(pointimg::frontend::t("Graine"));
    seed.set_hexpand(false);
    seed_row.add_suffix(&seed);
    vary_seed.add_css_class("flat");
    seed_row.add_suffix(&vary_seed);
    render_group.add(&seed_row);
    page.add(&render_group);

    let compute_group = adw::PreferencesGroup::new();
    compute_group.set_title(pointimg::frontend::t("Calcul"));
    compute_group.add(&quality);
    let compute_row = adw::ActionRow::new();
    compute_row.set_title(pointimg::frontend::t("Recalculer"));
    compute_row.set_subtitle(pointimg::frontend::t(
        "Les réglages se recalculent automatiquement (Espace)",
    ));
    compute_row.add_suffix(&recalc);
    compute_group.add(&compute_row);
    reset.set_halign(gtk::Align::Start);
    let reset_row = adw::ActionRow::new();
    reset_row.add_suffix(&reset);
    compute_group.add(&reset_row);
    page.add(&compute_group);

    // Per-section reset buttons, in the header of each group.
    let mut group_resets = Vec::new();
    for (group, title, target) in [
        (
            &preset_group,
            pointimg::frontend::t("Réglages rapides"),
            ResetTarget::Presets,
        ),
        (
            &algo_group,
            pointimg::frontend::t("Algorithme"),
            ResetTarget::Algorithm,
        ),
        (
            &halftone_group,
            pointimg::frontend::t("Trame"),
            ResetTarget::Algorithm,
        ),
        (
            &shape_group,
            pointimg::frontend::t("Forme des points"),
            ResetTarget::Shape,
        ),
        (
            &render_group,
            pointimg::frontend::t("Rendu"),
            ResetTarget::Render,
        ),
    ] {
        let button = gtk::Button::new();
        button.set_icon_name("edit-undo-symbolic");
        button.set_tooltip_text(Some(
            &pointimg::frontend::t("Rétablir les valeurs par défaut de « {title} »")
                .replace("{title}", title),
        ));
        button.add_css_class("flat");
        group.set_header_suffix(Some(&button));
        group_resets.push(GroupReset { button, target });
    }

    let controls_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&page)
        .build();
    // Forced width: without it the container stretches to the natural width
    // of the widest row of the panel, and that eats half of the window.
    // `propagate_natural_width(false)` stops the content width from bubbling
    // up to the requested size — that is what made it 540 px.
    controls_scroll.set_propagate_natural_width(false);
    controls_scroll.set_size_request(360, -1);

    // ── Header ────────────────────────────────────────────────────────────────
    // By value: a GTK widget is an `Arc`, the clone copies nothing. The
    // values are taken up again in `Widgets`, without double ownership.
    // ── Preview ──────────────────────────────────────────────────────────────
    let preview_toolbar = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    preview_toolbar.set_margin_top(6);
    preview_toolbar.set_margin_bottom(6);
    preview_toolbar.set_margin_start(10);
    preview_toolbar.set_margin_end(10);
    preview_toolbar.append(&gtk::Label::new(Some(pointimg::frontend::t("Affichage"))));
    preview_toolbar.append(&view_mode);
    preview_toolbar.append(&gtk::Separator::new(gtk::Orientation::Vertical));
    preview_toolbar.append(&zoom_out);
    preview_toolbar.append(&zoom_label);
    preview_toolbar.append(&zoom_in);
    preview_toolbar.append(&zoom_fit);
    preview_toolbar.append(&zoom_one);
    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    preview_toolbar.append(&spacer);
    busy_label.set_xalign(1.0);
    busy_label.add_css_class("dim-label");
    preview_toolbar.append(&busy_label);
    preview_toolbar.append(&busy);

    let images = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    images.set_hexpand(true);
    images.set_vexpand(true);
    for pane_root in [
        &src_pane.root,
        &dst_pane.root,
        &density_pane.root,
        &compare_pane.root,
    ] {
        pane_root.set_hexpand(true);
        pane_root.set_vexpand(true);
        images.append(pane_root);
    }

    empty_page.set_icon_name(Some("image-x-generic-symbolic"));
    empty_page.set_title(pointimg::frontend::t("Aucune image"));
    empty_page.set_description(Some(pointimg::frontend::t(
        "Glissez-déposez un fichier dans la fenêtre, ou ouvrez-en un.",
    )));
    open_empty.add_css_class("pill");
    open_empty.add_css_class("suggested-action");
    sample_button.add_css_class("pill");
    let empty_actions = button_row();
    empty_actions.append(&open_empty);
    empty_actions.append(&sample_button);
    // The recent images are added under the buttons by `ui.rs`, which knows
    // the state: the welcome page hosts them without replacing them.
    let empty_child = gtk::Box::new(gtk::Orientation::Vertical, 0);
    empty_child.set_valign(gtk::Align::Center);
    empty_child.set_halign(gtk::Align::Center);
    empty_child.set_spacing(18);
    empty_child.append(&empty_actions);
    let empty_extra = gtk::Box::new(gtk::Orientation::Vertical, 0);
    empty_child.append(&empty_extra);
    empty_page.set_child(Some(&empty_child));

    let overlay = gtk::Overlay::new();
    overlay.set_child(Some(&images));
    overlay.add_overlay(&empty_page);

    let preview = gtk::Box::new(gtk::Orientation::Vertical, 0);
    preview.set_vexpand(true);
    preview.append(&preview_toolbar);
    preview.append(&overlay);

    // `AdwOverlaySplitView` rather than a `GtkPaned`: the panel keeps a
    // forced width, and folds above the preview when the window becomes
    // narrow. A `Paned` cannot do that, and imposed the natural width of its
    // widest row (540 px, dictated by the button row): it left only 280 px of
    // preview in an 820 px window.
    let split = adw::OverlaySplitView::new();
    split.set_sidebar(Some(&controls_scroll));
    split.set_content(Some(&preview));
    split.set_collapsed(false);
    split.set_min_sidebar_width(320.0);
    split.set_max_sidebar_width(400.0);
    split.set_sidebar_width_unit(adw::LengthUnit::Px);
    split.set_show_sidebar(true);

    // ── Status bar ───────────────────────────────────────────────────────────
    let status_bar = status_bar(&memory, &export_sizes, &timing, &quality_badge, &status);

    let toast_overlay_for_return = toast_overlay.clone();
    let widgets = Widgets {
        window: WindowWidgets {
            window_title,
            menu,
            split,
            panel_toggle,
            open,
            progress,
            cancel,
            undo,
            redo,
            export,
            shortcuts,
            toast_overlay,
            locale_combo: locale_combo.clone(),
        },
        presets: PresetWidgets {
            preset_combo,
            preset_thumbs,
            preset_caption,
            load_preset,
            save_preset,
            open_folder,
            folder_label,
            folder_row,
        },
        algorithm: AlgorithmWidgets {
            algo,
            variance,
            variance_slider,
            min_radius,
            max_radius,
            min_radius_slider,
            max_radius_slider,
            max_boost,
            max_boost_slider,
            grid_group,
            cols,
            cols_slider,
            grid_angle,
            points_group,
            num_points,
            num_points_slider,
            iterations,
            iterations_slider,
        },
        halftone: HalftoneWidgets {
            halftone_group,
            halftone_mode,
            dominant_box,
            dominant_n,
            dominant_angle,
            screening,
            halftone_freq,
            halftone_freq_slider,
            halftone_min_radius,
            halftone_max_dot,
            halftone_min_slider,
            halftone_max_slider,
        },
        dots: DotWidgets {
            shape,
            ellipse_box,
            ellipse_aspect,
            ellipse_angle,
            polygon_box,
            polygon_sides,
            use_palette,
            palette_size,
            dithering,
        },
        render: RenderWidgets {
            bg_button,
            transparent,
            gamma,
            use_seed,
            seed,
            seed_row,
            vary_seed,
        },
        compute: ComputeWidgets {
            quality,
            recalc,
            reset,
            group_resets,
        },
        preview: PreviewWidgets {
            view_mode,
            zoom_out,
            zoom_label,
            zoom_in,
            zoom_one,
            zoom_fit,
            busy,
            export_sizes,
            busy_label,
            quality_badge,
            src_pane,
            dst_pane,
            density_pane,
            compare_pane,
        },
        status_bar: StatusWidgets {
            empty_page,
            empty_extra,
            open_empty,
            sample_button,
            memory,
            timing,
            status,
        },
    };

    // The final assembly comes after the literal: `header_bar` and the
    // status bar take `&Widgets`, which only exists here.
    let header = header_bar(&widgets);

    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_content(Some(&widgets.window.split));
    toolbar.add_bottom_bar(&status_bar);
    widgets.window.toast_overlay.set_child(Some(&toolbar));

    (toast_overlay_for_return, Rc::new(widgets))
}

// ─── Factories ──────────────────────────────────────────────────────────────

/// "Icon + label" button (explicit content, as in GNOME).
fn set_button_content<W: IsA<gtk::Widget>>(button: &W, icon: &str, label: &str) {
    let content = adw::ButtonContent::new();
    content.set_icon_name(icon);
    content.set_label(label);
    // `set_child` exists on `GtkButton` as on `GtkMenuButton`: we go through
    // the concrete widget to stay generic.
    let widget = button.as_ref().clone();
    if let Ok(button) = widget.clone().downcast::<gtk::Button>() {
        button.set_child(Some(&content));
    } else if let Ok(button) = widget.clone().downcast::<gtk::MenuButton>() {
        button.set_child(Some(&content));
    } else if let Ok(button) = widget.clone().downcast::<gtk::ToggleButton>() {
        button.set_child(Some(&content));
    }
}

/// Icon-only button, flat (toolbar).
fn set_icon_button(button: &gtk::Button, icon: &str, tooltip: &str) {
    button.set_icon_name(icon);
    button.set_tooltip_text(Some(tooltip));
    button.add_css_class("flat");
}

/// Row of buttons of equal width (equivalent of `AdwButtonRow`, which
/// requires libadwaita 1.6; we keep the base at 1.4).
fn button_row() -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    row.set_homogeneous(true);
    row
}

/// A `ListBox` without selection used as a "boxed-list" container.
fn rows_box<'a>(rows: &[&'a (impl IsA<gtk::Widget> + 'a)]) -> gtk::ListBox {
    let list = gtk::ListBox::new();
    list.set_selection_mode(gtk::SelectionMode::None);
    for row in rows {
        list.append(*row);
    }
    list.add_css_class("boxed-list");
    list
}

fn combo_row(title: &str, items: &[&str], subtitle: Option<&str>) -> adw::ComboRow {
    let row = adw::ComboRow::new();
    row.set_title(title);
    if let Some(subtitle) = subtitle {
        row.set_subtitle(subtitle);
    }
    row.set_model(Some(&gtk::StringList::new(items)));
    row
}

fn switch_row(title: &str, subtitle: &str) -> adw::SwitchRow {
    let row = adw::SwitchRow::new();
    row.set_title(title);
    row.set_subtitle(subtitle);
    row
}

/// Numeric row: slider *and* keyboard entry, like the system settings.
///
/// The bounds are those of `filter::validate_params`: keyboard entry covers
/// the whole validated range (up to 100 000 points, 32 channels).
pub(crate) fn spin_row(
    title: &str,
    subtitle: Option<&str>,
    tooltip: &str,
    min: f64,
    max: f64,
    step: f64,
    digits: u32,
) -> adw::SpinRow {
    let row = adw::SpinRow::with_range(min, max, step);
    row.set_title(title);
    if let Some(subtitle) = subtitle {
        row.set_subtitle(subtitle);
    }
    row.set_climb_rate(1.0);
    row.set_digits(digits);
    row.set_numeric(true);
    row.set_snap_to_ticks(false);
    row.set_tooltip_text(Some(tooltip));
    // A `SpinRow` captures the wheel: without this interception, scrolling
    // the settings panel silently changed the values.
    forward_wheel_to_scroller(&row);
    row
}

fn preview_pane(caption: &str) -> Pane {
    let label = gtk::Label::new(Some(caption));
    label.set_xalign(0.0);
    label.add_css_class("dim-label");
    label.set_margin_start(8);
    label.set_margin_top(4);

    let image = gtk::Image::new();
    image.set_halign(gtk::Align::Center);
    image.set_valign(gtk::Align::Center);

    let scroll = make_scroll(&image);

    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.append(&label);
    root.append(&scroll);
    Pane {
        root,
        scroll,
        image,
    }
}

pub(crate) fn make_scroll(child: &impl IsA<gtk::Widget>) -> gtk::ScrolledWindow {
    let scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(child)
        .build();
    scroll.set_hexpand(true);
    scroll.set_vexpand(true);
    scroll
}

/// Redirects the wheel of an entry widget towards the parent `ScrolledWindow`.
///
/// `GtkRange`/`GtkSpinButton` consume the wheel event: without this, the
/// wheel does not scroll the panel but changes the value of the setting under
/// the pointer — and triggers a recompute. With Ctrl, Shift or Alt, the notch
/// is applied to the setting (fine adjustment, GNOME convention).
pub(crate) fn forward_wheel_to_scroller<W: IsA<gtk::Widget>>(widget: &W) {
    // The closure must be "static": we capture a GTK handle, not a reference
    // borrowed from the parameter.
    let handle: gtk::Widget = widget.as_ref().clone();
    let controller = gtk::EventControllerScroll::builder()
        .flags(
            gtk::EventControllerScrollFlags::VERTICAL
                | gtk::EventControllerScrollFlags::HORIZONTAL
                | gtk::EventControllerScrollFlags::DISCRETE,
        )
        .propagation_phase(gtk::PropagationPhase::Capture)
        .build();
    controller.connect_scroll(move |controller, dx, dy| {
        let fine = controller.current_event_state().intersects(
            gtk::gdk::ModifierType::CONTROL_MASK
                | gtk::gdk::ModifierType::SHIFT_MASK
                | gtk::gdk::ModifierType::ALT_MASK,
        );
        if fine {
            return glib::Propagation::Proceed;
        }
        let Some(scroller) = enclosing_scroller(&handle) else {
            return glib::Propagation::Proceed;
        };
        step_adjustment(&scroller.vadjustment(), dy);
        step_adjustment(&scroller.hadjustment(), dx);
        glib::Propagation::Stop
    });
    widget.add_controller(controller);
}

/// First `ScrolledWindow` in the widget hierarchy.
fn enclosing_scroller(widget: &impl IsA<gtk::Widget>) -> Option<gtk::ScrolledWindow> {
    let mut current = widget.as_ref().parent();
    while let Some(node) = current {
        if let Ok(scroller) = node.clone().downcast::<gtk::ScrolledWindow>() {
            return Some(scroller);
        }
        current = node.parent();
    }
    None
}

fn step_adjustment(adjustment: &gtk::Adjustment, delta: f64) {
    let step = 48.0;
    let upper = (adjustment.upper() - adjustment.page_size()).max(adjustment.lower());
    let value = (adjustment.value() + delta * step).clamp(adjustment.lower(), upper);
    adjustment.set_value(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Radius slider: logarithmic over 0.0005 → 1, with the step of the
    /// `SpinRow` (4 decimals).
    fn radius_spec() -> SliderSpec {
        SliderSpec {
            curve: Curve::Log,
            start: 0.0005,
            end: 1.0,
            digits: 4,
        }
    }

    #[test]
    fn slider_value_is_quantised_to_the_spin_row_precision() {
        // The slider works in relative position: without rounding, the value
        // would need seven decimals and the neighboring field would show
        // "0.2531794".
        assert_eq!(radius_spec().value(0.7), 0.1023);
    }

    #[test]
    fn slider_position_round_trips() {
        let spec = radius_spec();
        for value in [0.0005, 0.003, 0.06, 1.0] {
            assert_eq!(spec.value(spec.position(value)), value, "valeur {value}");
        }
    }

    #[test]
    fn slider_value_stays_inside_its_range() {
        let spec = radius_spec();
        assert_eq!(spec.value(-1.0), 0.0005);
        assert_eq!(spec.value(2.0), 1.0);
    }
}
