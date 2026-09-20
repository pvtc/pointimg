//! Création des widgets et de la disposition de la fenêtre GTK4.

use std::rc::Rc;

use gtk::glib;
use gtk::prelude::*;

/// Poignées vers tous les widgets contrôlés par l'application.
pub(crate) struct Widgets {
    pub open: gtk::Button,
    pub load_preset: gtk::Button,
    pub save_preset: gtk::Button,

    pub algo: gtk::DropDown,
    pub placement_box: gtk::Box,
    pub grid_box: gtk::Box,
    pub points_box: gtk::Box,
    pub iterations_row: gtk::Box,
    pub variance: gtk::Scale,
    pub min_radius: gtk::Scale,
    pub max_radius: gtk::Scale,
    pub max_boost: gtk::Scale,
    pub cols: gtk::Scale,
    pub grid_angle: gtk::Scale,
    pub num_points: gtk::Scale,
    pub iterations: gtk::Scale,

    pub shape: gtk::DropDown,
    pub ellipse_box: gtk::Box,
    pub ellipse_aspect: gtk::Scale,
    pub ellipse_angle: gtk::Scale,
    pub polygon_box: gtk::Box,
    pub polygon_sides: gtk::Scale,
    pub use_palette: gtk::CheckButton,
    pub palette_size: gtk::Scale,
    pub dithering: gtk::CheckButton,
    pub use_seed: gtk::CheckButton,
    pub seed: gtk::SpinButton,

    pub bg_button: gtk::ColorDialogButton,
    pub transparent: gtk::CheckButton,
    pub gamma: gtk::CheckButton,

    pub halftone_group: gtk::Box,
    pub halftone_mode: gtk::DropDown,
    pub dominant_box: gtk::Box,
    pub dominant_n: gtk::Scale,
    pub dominant_angle: gtk::Scale,
    pub screening: gtk::DropDown,
    pub halftone_freq: gtk::Scale,
    pub halftone_min_radius: gtk::Scale,
    pub halftone_max_dot: gtk::Scale,

    pub recalc: gtk::Button,
    pub cancel: gtk::Button,
    pub progress: gtk::ProgressBar,
    pub save_image: gtk::Button,
    pub save_svg: gtk::Button,
    pub undo: gtk::Button,
    pub redo: gtk::Button,
    pub status: gtk::Label,
    pub memory: gtk::Label,
    pub timing: gtk::Label,

    pub view_mode: gtk::DropDown,
    pub zoom: gtk::Scale,
    pub zoom_fit: gtk::ToggleButton,
    pub zoom_one: gtk::Button,

    pub src_image: gtk::Image,
    pub dst_image: gtk::Image,
    pub density_image: gtk::Image,
    pub src_scroll: gtk::ScrolledWindow,
    pub dst_scroll: gtk::ScrolledWindow,
    pub density_scroll: gtk::ScrolledWindow,
}

/// Construit la disposition complète et renvoie le widget racine + poignées.
pub(crate) fn build() -> (gtk::Paned, Rc<Widgets>) {
    let open = gtk::Button::with_label("Ouvrir une image…");
    let load_preset = gtk::Button::with_label("Charger preset");
    let save_preset = gtk::Button::with_label("Sauver preset");

    let algo = gtk::DropDown::from_strings(&[
        "Grille",
        "K-means",
        "Voronoi (Lloyd)",
        "Quadtree",
        "Halftone (rosette)",
    ]);

    let (variance_row, variance) = scale_row("Sensibilité variance", 0.0, 1.0, 0.01, 2);
    let (min_row, min_radius) = scale_row("Rayon min (fraction image)", 0.001, 0.3, 0.001, 3);
    let (max_row, max_radius) = scale_row("Rayon max (fraction image)", 0.001, 0.3, 0.005, 3);
    let (boost_row, max_boost) = scale_row("Boost zones uniformes (×max)", 1.0, 5.0, 0.1, 1);
    let (cols_row, cols) = scale_row("Colonnes", 10.0, 300.0, 1.0, 0);
    let (grid_angle_row, grid_angle) = scale_row("Angle grille (°)", -90.0, 90.0, 1.0, 0);
    let (num_points_row, num_points) = scale_row("Nombre de points", 50.0, 5000.0, 1.0, 0);
    let (iterations_row, iterations) = scale_row("Itérations", 1.0, 30.0, 1.0, 0);

    let grid_box = vbox(6);
    grid_box.append(&cols_row);
    grid_box.append(&grid_angle_row);
    let points_box = vbox(6);
    points_box.append(&num_points_row);

    let placement_box = vbox(6);
    placement_box.append(&variance_row);
    placement_box.append(&min_row);
    placement_box.append(&max_row);
    placement_box.append(&boost_row);
    placement_box.append(&grid_box);
    placement_box.append(&points_box);
    placement_box.append(&iterations_row);

    let shape = gtk::DropDown::from_strings(&["Cercle", "Carré", "Ellipse", "Polygone"]);
    let (aspect_row, ellipse_aspect) = scale_row("Aspect", 0.2, 5.0, 0.05, 2);
    let (angle_row, ellipse_angle) = scale_row("Angle (°)", -180.0, 180.0, 1.0, 0);
    let ellipse_box = vbox(6);
    ellipse_box.append(&aspect_row);
    ellipse_box.append(&angle_row);
    let (sides_row, polygon_sides) = scale_row("Côtés", 3.0, 12.0, 1.0, 0);
    let polygon_box = vbox(6);
    polygon_box.append(&sides_row);

    let use_palette = gtk::CheckButton::with_label("Palette réduite");
    let (palette_row, palette_size) = scale_row("Couleurs", 2.0, 32.0, 1.0, 0);
    let dithering = gtk::CheckButton::with_label("Dithering Floyd-Steinberg");
    let use_seed = gtk::CheckButton::with_label("Seed fixé");
    let seed = gtk::SpinButton::with_range(0.0, 4_294_967_295.0, 1.0);
    seed.set_digits(0);
    seed.set_hexpand(true);

    let color_dialog = gtk::ColorDialog::builder()
        .with_alpha(false)
        .title("Couleur de fond")
        .build();
    let bg_button = gtk::ColorDialogButton::new(Some(color_dialog));
    let transparent = gtk::CheckButton::with_label("Fond transparent (RGBA)");
    let gamma = gtk::CheckButton::with_label("Correction gamma (espace linéaire)");

    let halftone_mode = gtk::DropDown::from_strings(&["CMYK", "Dominant"]);
    let (dominant_n_row, dominant_n) = scale_row("Couleurs dominantes", 2.0, 12.0, 1.0, 0);
    let (dominant_angle_row, dominant_angle) = scale_row("Angle de base (°)", 0.0, 180.0, 1.0, 0);
    let dominant_box = vbox(6);
    dominant_box.append(&dominant_n_row);
    dominant_box.append(&dominant_angle_row);
    let screening = gtk::DropDown::from_strings(&["AM (grille)", "FM (blue noise)"]);
    let (halftone_freq_row, halftone_freq) = scale_row("Fréquence de trame", 20.0, 200.0, 5.0, 0);
    let (halftone_min_row, halftone_min_radius) =
        scale_row("Rayon min (fraction)", 0.001, 0.05, 0.001, 3);
    let (halftone_max_row, halftone_max_dot) =
        scale_row("Rayon max (fraction step)", 0.3, 1.5, 0.05, 2);
    let halftone_group = section("Halftone");
    halftone_mode.set_hexpand(true);
    screening.set_hexpand(true);
    halftone_group.append(&halftone_mode);
    halftone_group.append(&dominant_box);
    halftone_group.append(&screening);
    halftone_group.append(&halftone_freq_row);
    halftone_group.append(&halftone_min_row);
    halftone_group.append(&halftone_max_row);

    let recalc = gtk::Button::with_label("Recalculer");
    let cancel = gtk::Button::with_label("Annuler");
    let progress = gtk::ProgressBar::new();
    progress.set_show_text(true);
    let save_image = gtk::Button::with_label("Sauvegarder image…");
    let save_svg = gtk::Button::with_label("Sauvegarder SVG…");
    let undo = gtk::Button::with_label("Undo");
    let redo = gtk::Button::with_label("Redo");
    let status = dim_label();
    status.set_wrap(true);
    let memory = dim_label();
    let timing = dim_label();

    let view_mode =
        gtk::DropDown::from_strings(&["Côte à côte", "Résultat", "Source", "Density map"]);
    let zoom = gtk::Scale::with_range(gtk::Orientation::Horizontal, 0.1, 4.0, 0.1);
    zoom.set_digits(1);
    zoom.set_value_pos(gtk::PositionType::Left);
    zoom.set_hexpand(true);
    zoom.set_size_request(140, -1);
    let zoom_fit = gtk::ToggleButton::with_label("Fit");
    zoom_fit.set_active(true);
    let zoom_one = gtk::Button::with_label("1:1");

    let src_image = preview_image();
    let dst_image = preview_image();
    let density_image = preview_image();
    let src_scroll = make_scroll(&src_image);
    let dst_scroll = make_scroll(&dst_image);
    let density_scroll = make_scroll(&density_image);

    let widgets = Rc::new(Widgets {
        open: open.clone(),
        load_preset: load_preset.clone(),
        save_preset: save_preset.clone(),
        algo: algo.clone(),
        placement_box: placement_box.clone(),
        grid_box: grid_box.clone(),
        points_box: points_box.clone(),
        iterations_row: iterations_row.clone(),
        variance: variance.clone(),
        min_radius: min_radius.clone(),
        max_radius: max_radius.clone(),
        max_boost: max_boost.clone(),
        cols: cols.clone(),
        grid_angle: grid_angle.clone(),
        num_points: num_points.clone(),
        iterations: iterations.clone(),
        shape: shape.clone(),
        ellipse_box: ellipse_box.clone(),
        ellipse_aspect: ellipse_aspect.clone(),
        ellipse_angle: ellipse_angle.clone(),
        polygon_box: polygon_box.clone(),
        polygon_sides: polygon_sides.clone(),
        use_palette: use_palette.clone(),
        palette_size: palette_size.clone(),
        dithering: dithering.clone(),
        use_seed: use_seed.clone(),
        seed: seed.clone(),
        bg_button: bg_button.clone(),
        transparent: transparent.clone(),
        gamma: gamma.clone(),
        halftone_group: halftone_group.clone(),
        halftone_mode: halftone_mode.clone(),
        dominant_box: dominant_box.clone(),
        dominant_n: dominant_n.clone(),
        dominant_angle: dominant_angle.clone(),
        screening: screening.clone(),
        halftone_freq: halftone_freq.clone(),
        halftone_min_radius: halftone_min_radius.clone(),
        halftone_max_dot: halftone_max_dot.clone(),
        recalc: recalc.clone(),
        cancel: cancel.clone(),
        progress: progress.clone(),
        save_image: save_image.clone(),
        save_svg: save_svg.clone(),
        undo: undo.clone(),
        redo: redo.clone(),
        status: status.clone(),
        memory: memory.clone(),
        timing: timing.clone(),
        view_mode: view_mode.clone(),
        zoom: zoom.clone(),
        zoom_fit: zoom_fit.clone(),
        zoom_one: zoom_one.clone(),
        src_image: src_image.clone(),
        dst_image: dst_image.clone(),
        density_image: density_image.clone(),
        src_scroll: src_scroll.clone(),
        dst_scroll: dst_scroll.clone(),
        density_scroll: density_scroll.clone(),
    });

    // ── Panneau de réglages ──────────────────────────────────────────────────
    let controls = gtk::Box::new(gtk::Orientation::Vertical, 16);
    controls.set_margin_top(12);
    controls.set_margin_bottom(12);
    controls.set_margin_start(12);
    controls.set_margin_end(12);
    controls.set_width_request(330);

    let file_group = section("Fichier");
    file_group.append(&open);
    let hint = dim_label();
    hint.set_text("(ou glissez-déposez une image dans la fenêtre)");
    file_group.append(&hint);
    let preset_row = hbox(6);
    load_preset.set_hexpand(true);
    save_preset.set_hexpand(true);
    preset_row.append(&load_preset);
    preset_row.append(&save_preset);
    file_group.append(&preset_row);
    controls.append(&file_group);

    let algo_group = section("Algorithme");
    algo.set_hexpand(true);
    algo_group.append(&algo);
    algo_group.append(&placement_box);
    controls.append(&algo_group);

    let shape_group = section("Forme des points");
    shape.set_hexpand(true);
    shape_group.append(&shape);
    shape_group.append(&ellipse_box);
    shape_group.append(&polygon_box);
    shape_group.append(&use_palette);
    shape_group.append(&palette_row);
    shape_group.append(&dithering);
    controls.append(&shape_group);

    let bg_group = section("Fond et rendu");
    let bg_row = hbox(8);
    let bg_label = gtk::Label::new(Some("Couleur de fond"));
    bg_label.set_xalign(0.0);
    bg_label.set_hexpand(true);
    bg_row.append(&bg_label);
    bg_row.append(&bg_button);
    bg_group.append(&bg_row);
    bg_group.append(&transparent);
    bg_group.append(&gamma);
    let seed_row = hbox(8);
    use_seed.set_hexpand(true);
    seed_row.append(&use_seed);
    seed_row.append(&seed);
    bg_group.append(&seed_row);
    controls.append(&bg_group);

    controls.append(&halftone_group);

    let compute_group = section("Calcul");
    let compute_row = hbox(6);
    recalc.set_hexpand(true);
    cancel.set_hexpand(true);
    compute_row.append(&recalc);
    compute_row.append(&cancel);
    compute_group.append(&compute_row);
    compute_group.append(&progress);
    let history_row = hbox(6);
    undo.set_hexpand(true);
    redo.set_hexpand(true);
    history_row.append(&undo);
    history_row.append(&redo);
    compute_group.append(&history_row);
    controls.append(&compute_group);

    let export_group = section("Export");
    save_image.set_hexpand(true);
    save_svg.set_hexpand(true);
    export_group.append(&save_image);
    export_group.append(&save_svg);
    controls.append(&export_group);

    let status_group = section("Informations");
    status_group.append(&memory);
    status_group.append(&timing);
    status_group.append(&status);
    controls.append(&status_group);

    let controls_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&controls)
        .build();

    // ── Zone d'aperçu ────────────────────────────────────────────────────────
    let preview_toolbar = hbox(8);
    preview_toolbar.set_margin_top(8);
    preview_toolbar.set_margin_bottom(4);
    preview_toolbar.set_margin_start(10);
    preview_toolbar.set_margin_end(10);
    preview_toolbar.append(&gtk::Label::new(Some("Affichage")));
    preview_toolbar.append(&view_mode);
    let spacer = hbox(0);
    spacer.set_hexpand(true);
    preview_toolbar.append(&spacer);
    preview_toolbar.append(&gtk::Label::new(Some("Zoom")));
    preview_toolbar.append(&zoom);
    preview_toolbar.append(&zoom_fit);
    preview_toolbar.append(&zoom_one);

    let images = hbox(6);
    images.set_hexpand(true);
    images.set_vexpand(true);
    for scroll in [&src_scroll, &dst_scroll, &density_scroll] {
        scroll.set_hexpand(true);
        scroll.set_vexpand(true);
        images.append(scroll);
    }

    let preview = gtk::Box::new(gtk::Orientation::Vertical, 0);
    preview.append(&preview_toolbar);
    preview.append(&images);

    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_start_child(Some(&controls_scroll));
    paned.set_end_child(Some(&preview));
    paned.set_position(350);
    paned.set_resize_start_child(false);
    paned.set_shrink_start_child(false);
    paned.set_vexpand(true);
    paned.set_hexpand(true);

    (paned, widgets)
}

fn vbox(spacing: i32) -> gtk::Box {
    gtk::Box::new(gtk::Orientation::Vertical, spacing)
}

fn hbox(spacing: i32) -> gtk::Box {
    gtk::Box::new(gtk::Orientation::Horizontal, spacing)
}

fn dim_label() -> gtk::Label {
    let label = gtk::Label::new(None);
    label.set_xalign(0.0);
    label.add_css_class("dim-label");
    label
}

pub(crate) fn section(title: &str) -> gtk::Box {
    let b = vbox(8);
    let label = gtk::Label::new(None);
    label.set_markup(&format!(
        "<b>{}</b>",
        glib::markup_escape_text(title).as_str()
    ));
    label.set_halign(gtk::Align::Start);
    b.append(&label);
    b
}

fn scale_row(label: &str, min: f64, max: f64, step: f64, digits: i32) -> (gtk::Box, gtk::Scale) {
    let row = vbox(2);
    let lbl = dim_label();
    lbl.set_text(label);
    let scale = gtk::Scale::with_range(gtk::Orientation::Horizontal, min, max, step);
    scale.set_digits(digits);
    scale.set_hexpand(true);
    scale.set_value_pos(gtk::PositionType::Right);
    scale.set_draw_value(true);
    row.append(&lbl);
    row.append(&scale);
    (row, scale)
}

fn make_scroll(child: &impl IsA<gtk::Widget>) -> gtk::ScrolledWindow {
    gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Automatic)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(child)
        .build()
}

fn preview_image() -> gtk::Image {
    let image = gtk::Image::new();
    image.set_halign(gtk::Align::Center);
    image.set_valign(gtk::Align::Center);
    image
}
