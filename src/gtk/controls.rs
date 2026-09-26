//! Construction des widgets et de la disposition de la fenêtre GTK4.
//!
//! Le panneau de gauche est une `adw::PreferencesPage` : chaque réglage occupe
//! une seule ligne (`adw::SpinRow` = glissière *et* saisie clavier, donc les
//! plages réelles de `validate_params` sont accessibles), les sections restent
//! visibles et les actions principales vivent dans l'en-tête plutôt qu'en bas
//! d'un panneau à faire défiler.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use gtk::glib;

/// Portées de « réinitialiser » : un bouton par section du panneau.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ResetTarget {
    Presets,
    Algorithm,
    Shape,
    Render,
    All,
}

/// Un bouton de réinitialisation de section.
pub(crate) struct GroupReset {
    pub button: gtk::Button,
    pub target: ResetTarget,
}

/// Un panneau d'aperçu : légende + zone défilante.
pub(crate) struct Pane {
    pub root: gtk::Box,
    pub scroll: gtk::ScrolledWindow,
    pub image: gtk::Image,
}

/// Le panneau « comparaison » : une seule image, deux couches, un curseur.
pub(crate) struct ComparePane {
    pub root: gtk::Box,
    pub scroll: gtk::ScrolledWindow,
    pub area: gtk::DrawingArea,
    pub data: Rc<RefCell<crate::compare::CompareData>>,
}

/// Poignées vers tous les widgets contrôlés par l'application.
pub(crate) struct Widgets {
    // ── En-tête ──────────────────────────────────────────────────────────────
    pub window_title: adw::WindowTitle,
    pub open: gtk::Button,
    pub progress: gtk::ProgressBar,
    pub cancel: gtk::Button,
    pub undo: gtk::Button,
    pub redo: gtk::Button,
    pub export: gtk::MenuButton,
    pub shortcuts: adw::Window,
    pub toast_overlay: adw::ToastOverlay,

    // ── Presets / fichiers ───────────────────────────────────────────────────
    pub preset_combo: adw::ComboRow,
    pub load_preset: gtk::Button,
    pub save_preset: gtk::Button,
    pub open_folder: gtk::Button,
    pub folder_label: gtk::Label,
    pub folder_row: adw::ActionRow,

    // ── Algorithme ───────────────────────────────────────────────────────────
    pub algo: adw::ComboRow,
    pub variance: adw::SpinRow,
    pub min_radius: adw::SpinRow,
    pub max_radius: adw::SpinRow,
    pub max_boost: adw::SpinRow,
    pub grid_group: adw::PreferencesGroup,
    pub cols: adw::SpinRow,
    pub grid_angle: adw::SpinRow,
    pub points_group: adw::PreferencesGroup,
    pub num_points: adw::SpinRow,
    pub iterations: adw::SpinRow,

    // ── Halftone ─────────────────────────────────────────────────────────────
    pub halftone_group: adw::PreferencesGroup,
    pub halftone_mode: adw::ComboRow,
    pub dominant_box: gtk::ListBox,
    pub dominant_n: adw::SpinRow,
    pub dominant_angle: adw::SpinRow,
    pub screening: adw::ComboRow,
    pub halftone_freq: adw::SpinRow,
    pub halftone_min_radius: adw::SpinRow,
    pub halftone_max_dot: adw::SpinRow,

    // ── Forme des points ─────────────────────────────────────────────────────
    pub shape: adw::ComboRow,
    pub ellipse_box: gtk::ListBox,
    pub ellipse_aspect: adw::SpinRow,
    pub ellipse_angle: adw::SpinRow,
    pub polygon_box: gtk::ListBox,
    pub polygon_sides: adw::SpinRow,
    pub use_palette: adw::SwitchRow,
    pub palette_size: adw::SpinRow,
    pub dithering: adw::SwitchRow,

    // ── Rendu ────────────────────────────────────────────────────────────────
    pub bg_button: gtk::ColorDialogButton,
    pub transparent: adw::SwitchRow,
    pub gamma: adw::SwitchRow,
    pub use_seed: adw::SwitchRow,
    pub seed: gtk::SpinButton,
    pub vary_seed: gtk::Button,

    // ── Calcul ───────────────────────────────────────────────────────────────
    pub quality: adw::ComboRow,
    pub recalc: gtk::Button,
    pub reset: gtk::Button,
    pub group_resets: Vec<GroupReset>,

    // ── Aperçu ───────────────────────────────────────────────────────────────
    pub view_mode: gtk::DropDown,
    pub zoom_out: gtk::Button,
    pub zoom_label: gtk::Label,
    pub zoom_in: gtk::Button,
    pub zoom_one: gtk::ToggleButton,
    pub zoom_fit: gtk::ToggleButton,
    pub busy: gtk::Spinner,
    pub busy_label: gtk::Label,
    pub quality_badge: gtk::Label,

    pub src_pane: Pane,
    pub dst_pane: Pane,
    pub density_pane: Pane,
    pub compare_pane: ComparePane,

    // ── État vide / barre d'état ─────────────────────────────────────────────
    pub empty_page: adw::StatusPage,
    pub open_empty: gtk::Button,
    pub sample_button: gtk::Button,
    pub memory: gtk::Label,
    pub timing: gtk::Label,
    pub status: gtk::Label,
}

const ALGOS: [&str; 5] = [
    "Grille",
    "K-means",
    "Voronoi (Lloyd)",
    "Quadtree",
    "Halftone (rosette)",
];
const ALGO_HELP: [&str; 5] = [
    "Grille régulière : une cellule = un point. Rapide et prévisible.",
    "k-means spatial : les points se répartissent selon la couleur et la position.",
    "Lloyd pondéré : répartition régulière guidée par le détail. Rendu le plus « naturel ».",
    "Subdivision par contraste : davantage de points là où l'image est détaillée.",
    "Trame d'encre : rosette CMJN ou couleurs dominantes, style offset print.",
];
const SHAPES: [&str; 4] = ["Cercle", "Carré", "Ellipse", "Polygone"];
const VIEWS: [&str; 5] = [
    "Comparaison",
    "Côte à côte",
    "Résultat",
    "Source",
    "Density map",
];
const QUALITIES: [&str; 3] = ["Rapide (1/2)", "Rapide (1/4)", "Pleine résolution"];

/// Construit la disposition complète : en-tête, panneau de réglages, aperçu.
pub(crate) fn build() -> (adw::ToastOverlay, Rc<Widgets>) {
    // ── Widgets de base ──────────────────────────────────────────────────────
    let open = gtk::Button::with_label("Ouvrir…");
    let open_empty = gtk::Button::with_label("Ouvrir une image…");
    let sample_button = gtk::Button::with_label("Utiliser un exemple");
    let window_title = adw::WindowTitle::new("pointimg", "Aucune image");
    let progress = gtk::ProgressBar::new();
    let cancel = gtk::Button::with_label("Annuler");
    let undo = gtk::Button::new();
    let redo = gtk::Button::new();
    let export = gtk::MenuButton::new();
    let toast_overlay = adw::ToastOverlay::new();
    let shortcuts = adw::Window::new();

    let preset_combo = combo_row("Preset", &["Personnalisé"], None);
    let load_preset = gtk::Button::with_label("Charger un preset…");
    let save_preset = gtk::Button::with_label("Sauver le preset…");
    let open_folder = gtk::Button::with_label("Dossier…");
    let folder_label = gtk::Label::new(None);

    let algo = combo_row("Algorithme", &ALGOS, Some(ALGO_HELP[0]));
    let variance = spin_row(
        "Sensibilité variance",
        Some("0 = uniforme · 1 = détail maximal"),
        "Redistribue les points vers les zones détaillées. 0 = répartition uniforme, 1 = maximum de détail.",
        0.0,
        1.0,
        0.01,
        2,
    );
    let min_radius = spin_row(
        "Rayon min",
        Some("fraction du petit côté"),
        "Rayon minimal des points, en fraction du plus petit côté de l'image.",
        0.0005,
        1.0,
        0.0005,
        4,
    );
    let max_radius = spin_row(
        "Rayon max",
        Some("fraction du petit côté"),
        "Rayon maximal des points, en fraction du plus petit côté de l'image.",
        0.0005,
        1.0,
        0.0025,
        4,
    );
    let max_boost = spin_row(
        "Boost zones uniformes",
        Some("× maximum"),
        "Agrandit les points dans les zones sans détail. 1 = aucun agrandissement, 2.5 = réglage par défaut.",
        1.0,
        8.0,
        0.1,
        2,
    );
    let cols = spin_row(
        "Colonnes",
        Some("2 – 8 192"),
        "Nombre de cellules de la grille (algorithme Grille).",
        2.0,
        8192.0,
        1.0,
        0,
    );
    let grid_angle = spin_row(
        "Angle de la grille",
        Some("°"),
        "Rotation de la grille pour un effet de trame (algorithme Grille).",
        -180.0,
        180.0,
        1.0,
        0,
    );
    let num_points = spin_row(
        "Nombre de points",
        Some("10 – 100 000"),
        "Nombre de points générés (K-means, Voronoï, Quadtree).",
        10.0,
        100_000.0,
        10.0,
        0,
    );
    let iterations = spin_row(
        "Itérations",
        Some("1 – 100"),
        "Itérations de relaxation de Lloyd (K-means, Voronoï). Davantage = plus régulier, plus lent.",
        1.0,
        100.0,
        1.0,
        0,
    );

    let halftone_mode = combo_row("Mode", &["CMJN (rosette)", "Couleurs dominantes"], None);
    let dominant_n = spin_row(
        "Canaux",
        Some("couleurs dominantes"),
        "Nombre de couleurs de la trame (une trame par couleur).",
        2.0,
        32.0,
        1.0,
        0,
    );
    let dominant_angle = spin_row(
        "Angle de base",
        Some("°"),
        "Angle de la première trame ; les suivantes sont réparties régulièrement pour éviter le moiré.",
        0.0,
        180.0,
        1.0,
        0,
    );
    let dominant_box = rows_box(&[&dominant_n, &dominant_angle]);
    let screening = combo_row(
        "Screening",
        &["AM (grille)", "FM (blue noise)"],
        Some("AM : la taille du dot code l'encre. FM : c'est la densité qui la code."),
    );
    let halftone_freq = spin_row(
        "Fréquence de trame",
        Some("20 – 400"),
        "Nombre de cellules de trame sur le plus petit côté. Plus c'est élevé, plus le pointillé est fin.",
        4.0,
        400.0,
        1.0,
        0,
    );
    let halftone_min_radius = spin_row(
        "Rayon min",
        Some("fraction du petit côté"),
        "Rayon minimal des dots de trame, en fraction du plus petit côté.",
        0.0,
        1.0,
        0.001,
        3,
    );
    let halftone_max_dot = spin_row(
        "Rayon max",
        Some("× pas de trame"),
        "Rayon maximal, en fraction du pas de trame. Au-delà de 1, les dots se touchent.",
        0.05,
        10.0,
        0.05,
        2,
    );

    let shape = combo_row("Forme", &SHAPES, None);
    let ellipse_aspect = spin_row(
        "Rapport largeur/hauteur",
        None,
        "Rapport largeur/hauteur de l'ellipse.",
        0.1,
        10.0,
        0.1,
        2,
    );
    let ellipse_angle = spin_row(
        "Angle",
        Some("°"),
        "Rotation de l'ellipse en degrés.",
        -180.0,
        180.0,
        1.0,
        0,
    );
    let polygon_sides = spin_row(
        "Côtés",
        Some("3 – 12"),
        "Nombre de côtés du polygone régulier.",
        3.0,
        12.0,
        1.0,
        0,
    );
    let ellipse_box = rows_box(&[&ellipse_aspect, &ellipse_angle]);
    let polygon_box = rows_box(&[&polygon_sides]);
    let use_palette = switch_row(
        "Palette réduite",
        "Quantifie l'image vers quelques couleurs (effet sérigraphie)",
    );
    let palette_size = spin_row(
        "Couleurs de la palette",
        Some("2 – 256"),
        "Nombre de couleurs de la palette réduite.",
        2.0,
        256.0,
        1.0,
        0,
    );
    let dithering = switch_row(
        "Dithering Floyd-Steinberg",
        "Diffusion d'erreur sur la palette : conserve les dégradés",
    );

    let color_dialog = gtk::ColorDialog::builder()
        .with_alpha(false)
        .title("Couleur de fond")
        .build();
    let bg_button = gtk::ColorDialogButton::new(Some(color_dialog));
    bg_button.set_tooltip_text(Some("Couleur de fond du rendu"));
    let transparent = switch_row(
        "Fond transparent (RGBA)",
        "Alpha conservé à l'export PNG/WebP/TIFF",
    );
    let gamma = switch_row(
        "Correction gamma",
        "Moyennes en espace linéaire : évite les mi-tons trop sombres",
    );
    let use_seed = switch_row(
        "Graine fixée",
        "Deux exécutions identiques donnent le même résultat",
    );
    let seed = gtk::SpinButton::with_range(0.0, 4_294_967_295.0, 1.0);
    seed.set_digits(0);
    seed.set_tooltip_text(Some("Graine du générateur aléatoire"));
    forward_wheel_to_scroller(&seed);
    let vary_seed = gtk::Button::new();
    set_icon_button(
        &vary_seed,
        "view-refresh-symbolic",
        "Tirer une nouvelle graine et recalculer",
    );

    let quality = combo_row(
        "Qualité de l'aperçu",
        &QUALITIES,
        Some("Le mode rapide calcule sur une image réduite, puis repasse en pleine résolution."),
    );
    let recalc = gtk::Button::new();
    set_button_content(&recalc, "view-refresh-symbolic", "Recalculer");
    recalc.set_tooltip_text(Some("Recalculer maintenant, en pleine résolution (Espace)"));
    let reset = gtk::Button::with_label("Réinitialiser les réglages");

    let view_mode = gtk::DropDown::from_strings(&VIEWS);
    view_mode.set_tooltip_text(Some("Quel aperçu afficher"));
    let zoom_out = gtk::Button::from_icon_name("zoom-out-symbolic");
    zoom_out.set_tooltip_text(Some("Dézoomer (−)"));
    let zoom_in = gtk::Button::from_icon_name("zoom-in-symbolic");
    zoom_in.set_tooltip_text(Some("Zoomer (+)"));
    let zoom_label = gtk::Label::new(Some("100 %"));
    zoom_label.set_width_chars(5);
    zoom_label.set_xalign(0.5);
    zoom_label.add_css_class("dim-label");
    zoom_label.set_tooltip_text(Some("Zoom réellement appliqué à l'aperçu"));
    let zoom_one = gtk::ToggleButton::with_label("1:1");
    zoom_one.set_tooltip_text(Some("Revenir à 100 % (1)"));
    let zoom_fit = gtk::ToggleButton::with_label("Ajuster");
    zoom_fit.set_tooltip_text(Some("Ajuster l'aperçu à la fenêtre (0)"));
    zoom_fit.set_active(true);
    let busy = gtk::Spinner::new();
    busy.set_valign(gtk::Align::Center);
    let busy_label = gtk::Label::new(None);
    let quality_badge = gtk::Label::new(None);
    quality_badge.add_css_class("dim-label");

    let src_pane = preview_pane("Source");
    let dst_pane = preview_pane("Résultat");
    let density_pane = preview_pane("Density map");
    let compare_pane = crate::compare::build_pane();

    let empty_page = adw::StatusPage::new();
    let memory = gtk::Label::new(None);
    let timing = gtk::Label::new(None);
    let status = gtk::Label::new(None);

    // ── Panneau de réglages ──────────────────────────────────────────────────
    let page = adw::PreferencesPage::new();

    let preset_group = adw::PreferencesGroup::new();
    preset_group.set_title("Réglages rapides");
    let preset_names: Vec<String> = pointimg::filter::builtin_presets()
        .iter()
        .map(|p| p.name.to_string())
        .collect();
    let mut preset_items = vec!["Personnalisé"];
    preset_items.extend(preset_names.iter().map(String::as_str));
    preset_combo.set_model(Some(&gtk::StringList::new(&preset_items)));
    preset_group.add(&preset_combo);

    let files_buttons = button_row();
    files_buttons.append(&load_preset);
    files_buttons.append(&save_preset);
    files_buttons.append(&open_folder);
    preset_group.add(&files_buttons);

    let folder_row = adw::ActionRow::new();
    folder_row.set_title("Images du dossier");
    folder_row.set_visible(false);
    folder_label.set_xalign(1.0);
    folder_label.add_css_class("dim-label");
    folder_row.add_suffix(&folder_label);
    preset_group.add(&folder_row);
    page.add(&preset_group);

    let algo_group = adw::PreferencesGroup::new();
    algo_group.set_title("Algorithme");
    algo_group.add(&algo);
    algo_group.add(&variance);
    algo_group.add(&min_radius);
    algo_group.add(&max_radius);
    algo_group.add(&max_boost);
    page.add(&algo_group);

    let grid_group = adw::PreferencesGroup::new();
    grid_group.set_title("Grille");
    grid_group.add(&cols);
    grid_group.add(&grid_angle);
    page.add(&grid_group);

    let points_group = adw::PreferencesGroup::new();
    points_group.set_title("Densité de points");
    points_group.add(&num_points);
    points_group.add(&iterations);
    page.add(&points_group);

    let halftone_group = adw::PreferencesGroup::new();
    halftone_group.set_title("Trame");
    halftone_group.set_description(Some(
        "Séparation de l'image en canaux d'encre, une trame par canal.",
    ));
    halftone_group.add(&halftone_mode);
    halftone_group.add(&dominant_box);
    halftone_group.add(&screening);
    halftone_group.add(&halftone_freq);
    halftone_group.add(&halftone_min_radius);
    halftone_group.add(&halftone_max_dot);
    page.add(&halftone_group);

    let shape_group = adw::PreferencesGroup::new();
    shape_group.set_title("Forme des points");
    shape_group.add(&shape);
    shape_group.add(&ellipse_box);
    shape_group.add(&polygon_box);
    shape_group.add(&use_palette);
    shape_group.add(&palette_size);
    shape_group.add(&dithering);
    page.add(&shape_group);

    let render_group = adw::PreferencesGroup::new();
    render_group.set_title("Rendu");
    let bg_row = adw::ActionRow::new();
    bg_row.set_title("Couleur de fond");
    bg_row.set_subtitle("Appliquée au rendu et aux zones transparentes de la source");
    bg_row.add_suffix(&bg_button);
    render_group.add(&bg_row);
    render_group.add(&transparent);
    render_group.add(&gamma);
    render_group.add(&use_seed);
    let seed_row = adw::ActionRow::new();
    seed_row.set_title("Graine");
    seed.set_hexpand(false);
    seed_row.add_suffix(&seed);
    vary_seed.add_css_class("flat");
    seed_row.add_suffix(&vary_seed);
    render_group.add(&seed_row);
    page.add(&render_group);

    let compute_group = adw::PreferencesGroup::new();
    compute_group.set_title("Calcul");
    compute_group.add(&quality);
    let compute_row = adw::ActionRow::new();
    compute_row.set_title("Recalculer");
    compute_row.set_subtitle("Les réglages se recalculent automatiquement (Espace)");
    compute_row.add_suffix(&recalc);
    compute_group.add(&compute_row);
    reset.set_halign(gtk::Align::Start);
    let reset_row = adw::ActionRow::new();
    reset_row.add_suffix(&reset);
    compute_group.add(&reset_row);
    page.add(&compute_group);

    // Boutons de réinitialisation par section, dans l'en-tête de chaque groupe.
    let mut group_resets = Vec::new();
    for (group, title, target) in [
        (&preset_group, "Réglages rapides", ResetTarget::Presets),
        (&algo_group, "Algorithme", ResetTarget::Algorithm),
        (&halftone_group, "Trame", ResetTarget::Algorithm),
        (&shape_group, "Forme des points", ResetTarget::Shape),
        (&render_group, "Rendu", ResetTarget::Render),
    ] {
        let button = gtk::Button::new();
        button.set_icon_name("edit-undo-symbolic");
        button.set_tooltip_text(Some(&format!(
            "Rétablir les valeurs par défaut de « {title} »"
        )));
        button.add_css_class("flat");
        group.set_header_suffix(Some(&button));
        group_resets.push(GroupReset { button, target });
    }

    let controls_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .child(&page)
        .build();
    // Largeur fixe : sans cela le `Paned` s'étire à la largeur naturelle de la
    // ligne la plus large et le panneau mange la moitié de la fenêtre.
    controls_scroll.set_size_request(360, -1);

    // ── En-tête ──────────────────────────────────────────────────────────────
    // `adw::ButtonContent` impose le contenu (icône + libellé) : `set_label`
    // suivi de `set_icon_name` sur un GtkButton ne garde que l'icône.
    set_button_content(&open, "document-open-symbolic", "Ouvrir…");
    open.set_tooltip_text(Some("Ouvrir une image (Ctrl+O)"));
    progress.set_show_text(true);
    progress.set_valign(gtk::Align::Center);
    progress.set_size_request(200, -1);
    progress.set_visible(false);
    set_button_content(&cancel, "process-stop-symbolic", "Annuler");
    cancel.set_tooltip_text(Some("Annuler le calcul en cours"));
    set_icon_button(
        &undo,
        "edit-undo-symbolic",
        "Annuler la dernière modification (Ctrl+Z)",
    );
    set_icon_button(
        &redo,
        "edit-redo-symbolic",
        "Rétablir la modification annulée (Ctrl+Y)",
    );
    set_button_content(&export, "document-save-symbolic", "Exporter");
    export.add_css_class("suggested-action");
    export.set_tooltip_text(Some("Exporter l'image (Ctrl+S)"));
    export.set_menu_model(Some(&crate::wire::menu_model()));

    let header = adw::HeaderBar::new();
    header.set_title_widget(Some(&window_title));
    header.pack_start(&open);
    header.pack_end(&export);
    header.pack_end(&redo);
    header.pack_end(&undo);
    header.pack_end(&cancel);
    header.pack_end(&progress);

    // ── Aperçu ───────────────────────────────────────────────────────────────
    let preview_toolbar = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    preview_toolbar.set_margin_top(6);
    preview_toolbar.set_margin_bottom(6);
    preview_toolbar.set_margin_start(10);
    preview_toolbar.set_margin_end(10);
    preview_toolbar.append(&gtk::Label::new(Some("Affichage")));
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
    empty_page.set_title("Aucune image");
    empty_page.set_description(Some(
        "Glissez-déposez un fichier dans la fenêtre, ou ouvrez-en un.",
    ));
    open_empty.add_css_class("pill");
    open_empty.add_css_class("suggested-action");
    sample_button.add_css_class("pill");
    let empty_actions = button_row();
    empty_actions.append(&open_empty);
    empty_actions.append(&sample_button);
    empty_page.set_child(Some(&empty_actions));

    let overlay = gtk::Overlay::new();
    overlay.set_child(Some(&images));
    overlay.add_overlay(&empty_page);

    let preview = gtk::Box::new(gtk::Orientation::Vertical, 0);
    preview.set_vexpand(true);
    preview.append(&preview_toolbar);
    preview.append(&overlay);

    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_start_child(Some(&controls_scroll));
    paned.set_end_child(Some(&preview));
    paned.set_position(360);
    paned.set_resize_start_child(false);
    paned.set_shrink_start_child(false);
    paned.set_vexpand(true);
    paned.set_hexpand(true);

    // ── Barre d'état ─────────────────────────────────────────────────────────
    let status_bar = gtk::Box::new(gtk::Orientation::Horizontal, 14);
    status_bar.set_margin_top(4);
    status_bar.set_margin_bottom(6);
    status_bar.set_margin_start(12);
    status_bar.set_margin_end(12);
    for label in [&memory, &timing] {
        label.set_xalign(0.0);
        label.add_css_class("dim-label");
        status_bar.append(label);
    }
    status_bar.append(&quality_badge);
    status.set_xalign(0.0);
    status.set_hexpand(true);
    status.set_wrap(false);
    status.set_ellipsize(gtk::pango::EllipsizeMode::End);
    status_bar.append(&status);

    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_content(Some(&paned));
    toolbar.add_bottom_bar(&status_bar);

    toast_overlay.set_child(Some(&toolbar));

    let toast_overlay_for_return = toast_overlay.clone();
    let widgets = Widgets {
        window_title,
        open,
        progress,
        cancel,
        undo,
        redo,
        export,
        shortcuts,
        toast_overlay,
        preset_combo,
        load_preset,
        save_preset,
        open_folder,
        folder_label,
        folder_row,
        algo,
        variance,
        min_radius,
        max_radius,
        max_boost,
        grid_group,
        cols,
        grid_angle,
        points_group,
        num_points,
        iterations,
        halftone_group,
        halftone_mode,
        dominant_box,
        dominant_n,
        dominant_angle,
        screening,
        halftone_freq,
        halftone_min_radius,
        halftone_max_dot,
        shape,
        ellipse_box,
        ellipse_aspect,
        ellipse_angle,
        polygon_box,
        polygon_sides,
        use_palette,
        palette_size,
        dithering,
        bg_button,
        transparent,
        gamma,
        use_seed,
        seed,
        vary_seed,
        quality,
        recalc,
        reset,
        group_resets,
        view_mode,
        zoom_out,
        zoom_label,
        zoom_in,
        zoom_one,
        zoom_fit,
        busy,
        busy_label,
        quality_badge,
        src_pane,
        dst_pane,
        density_pane,
        compare_pane,
        empty_page,
        open_empty,
        sample_button,
        memory,
        timing,
        status,
    };

    (toast_overlay_for_return, Rc::new(widgets))
}

// ─── Fabriques ───────────────────────────────────────────────────────────────

/// Bouton « icône + libellé » (contenu explicite, comme dans GNOME).
fn set_button_content<W: IsA<gtk::Widget>>(button: &W, icon: &str, label: &str) {
    let content = adw::ButtonContent::new();
    content.set_icon_name(icon);
    content.set_label(label);
    // `set_child` existe sur `GtkButton` comme sur `GtkMenuButton` : on passe
    // par le widget concret pour rester générique.
    let widget = button.as_ref().clone();
    if let Ok(button) = widget.clone().downcast::<gtk::Button>() {
        button.set_child(Some(&content));
    } else if let Ok(button) = widget.clone().downcast::<gtk::MenuButton>() {
        button.set_child(Some(&content));
    } else if let Ok(button) = widget.clone().downcast::<gtk::ToggleButton>() {
        button.set_child(Some(&content));
    }
}

/// Bouton icône seule, à plat (barre d'outils).
fn set_icon_button(button: &gtk::Button, icon: &str, tooltip: &str) {
    button.set_icon_name(icon);
    button.set_tooltip_text(Some(tooltip));
    button.add_css_class("flat");
}

/// Rangée de boutons de même largeur (équivalent de `AdwButtonRow`, qui
/// demande libadwaita 1.6 ; on garde le socle à 1.4).
fn button_row() -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    row.set_homogeneous(true);
    row
}

/// Une `ListBox` sans sélection servant de conteneur « boxed-list ».
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

/// Ligne numérique : glissière *et* saisie clavier, comme les réglages système.
///
/// Les bornes reprennent celles de `filter::validate_params` : la saisie
/// clavier donne accès à toute la plage, là où l'ancien curseur s'arrêtait
/// bien avant (5 000 points au lieu de 100 000, 12 canaux au lieu de 32…).
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
    // Une `SpinRow` capte la molette : sans cette interception, faire défiler
    // le panneau des réglages modifiait silencieusement les valeurs.
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

/// Redirige la molette d'un widget de saisie vers le `ScrolledWindow` parent.
///
/// `GtkRange`/`GtkSpinButton` consomment l'événement de molette : sans cela, la
/// molette ne fait pas défiler le panneau mais change la valeur du réglage situé
/// sous le curseur — et relance un calcul. Avec Ctrl, Shift ou Alt, le cran est
/// appliqué au réglage (ajustement fin, convention GNOME).
pub(crate) fn forward_wheel_to_scroller<W: IsA<gtk::Widget>>(widget: &W) {
    // Le closure doit être « statique » : on capture une poignée GTK, pas une
    // référence empruntée au paramètre.
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

/// Premier `ScrolledWindow` dans la hiérarchie d'un widget.
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
