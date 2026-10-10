//! Construction des widgets et de la disposition de la fenêtre GTK4.
//!
//! Le panneau de gauche est une `adw::PreferencesPage` : chaque réglage occupe
//! une seule ligne (`adw::SpinRow` = saisie clavier aux plages réelles de
//! `validate_params`), les sections restent visibles et les actions principales
//! vivent dans l'en-tête plutôt qu'en bas d'un panneau à faire défiler.
//!
//! Les réglages à course large (nombre de points, rayons) sont doublés d'un
//! `GtkScale` : le `SpinRow` reste l'outil de la valeur exacte, le curseur
//! celui du réglage à la souris — impossible avec la seule molette du
//! `SpinButton`, dont le pas est fixe.

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use gtk::glib;

use pointimg::frontend::Curve;

/// Conversion entre la position d'un curseur (0 → 1) et la valeur du filtre.
///
/// `Copy` pour être recopié dans les callbacks GTK, qui ne peuvent pas
/// capturer le widget lui-même.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SliderSpec {
    curve: Curve,
    start: f64,
    end: f64,
    digits: u32,
}

impl SliderSpec {
    /// Position du curseur correspondant à une valeur de filtre.
    pub(crate) fn position(self, value: f64) -> f64 {
        self.curve.position(value, self.start, self.end)
    }

    /// Valeur de filtre correspondant à une position du curseur, arrondie à la
    /// précision du champ de saisie voisin : sans cela, le glissement laisserait
    /// `0,25317937` dans les réglages.
    pub(crate) fn value(self, position: f64) -> f64 {
        let factor = 10f64.powi(self.digits as i32);
        let value = (self.curve.value(position, self.start, self.end) * factor).round() / factor;
        value.clamp(self.start, self.end)
    }
}

/// Curseur horizontal doublant un `adw::SpinRow`.
pub(crate) struct ValueSlider {
    pub scale: gtk::Scale,
    pub spec: SliderSpec,
    /// Rangée qui héberge le curseur, pour la masquer avec le réglage qu'elle
    /// double (le curseur seul ne suffirait pas : la rangée resterait visible,
    /// vide de son curseur).
    row: Option<adw::ActionRow>,
}

impl ValueSlider {
    /// `digits` est la précision du `SpinRow` associé, pour que les deux
    /// représentations tombent d'accord.
    pub(crate) fn new(start: f64, end: f64, digits: u32, curve: Curve) -> Self {
        // Le curseur travaille en position relative : GTK n'a pas d'échelle
        // logarithmique, et une course linéaire sur 0,0005 → 1,0 rendrait les
        // petits rayons impossibles à atteindre à la souris.
        let scale = gtk::Scale::with_range(gtk::Orientation::Horizontal, 0.0, 1.0, 0.002);
        scale.set_draw_value(false);
        scale.set_value_pos(gtk::PositionType::Right);
        scale.set_has_origin(false);
        scale.set_hexpand(true);
        scale.set_digits(3);
        // Un `GtkScale` capte la molette : sans cette interception, faire
        // défiler le panneau des réglages modifiait silencieusement les valeurs.
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

    /// Place la poignée sur une valeur de filtre.
    pub(crate) fn set_value(&self, value: f64) {
        self.scale.set_value(self.spec.position(value));
    }

    /// Marque la position de l'autre borne d'une paire (l'ancienne marque est
    /// remplacée : `gtk_scale_add_mark` ne renvoie pas de handle).
    pub(crate) fn set_bound(&self, other: f64) {
        self.scale.clear_marks();
        self.scale
            .add_mark(self.spec.position(other), gtk::PositionType::Bottom, None);
    }

    /// Rangée qui héberge le curseur.
    ///
    /// `adw::PreferencesGroup` n'accepte que des `AdwPreferencesRow`, et le
    /// suffixe d'un `AdwActionRow` n'est pas extensible : le curseur occupe
    /// donc la partie droite de la rangée, comme les curseurs des réglages
    /// GNOME.
    ///
    /// Ni titre ni sous-titre : la rangée double un `SpinRow` situé juste au
    /// dessus, dont le libellé et l'aide nomment déjà le réglage. Le curseur
    /// n'est qu'un second moyen de le régler, pas un réglage à part entière —
    /// libellé et surtout le mot « curseur », déjà présent dans la moitié
    /// droite de la fenêtre, n'apprenaient rien. L'échelle est expliquée dans
    /// l'infobulle.
    pub(crate) fn row(&mut self, tooltip: &str) -> adw::ActionRow {
        let row = adw::ActionRow::new();
        row.set_activatable(false);
        row.set_tooltip_text(Some(tooltip));
        row.add_suffix(&self.scale);
        self.row = Some(row.clone());
        row
    }

    /// Masque la rangée entière quand le réglage doublé n'est pas visible.
    pub(crate) fn set_row_visible(&self, visible: bool) {
        if let Some(row) = &self.row {
            row.set_visible(visible);
        }
    }
}

/// Une rangée de curseur pour une paire min/max.
///
/// GTK n'a pas de curseur à deux poignées (c'est aussi vrai de libadwaita,
/// dont `AdwSliderRow` n'existe qu'à partir de 1.6). On met donc les deux
/// curseurs l'un sous l'autre, chacun marquant la position de l'autre borne :
/// la lecture de la plage et de son écart reste immédiate, et la saisie
/// chiffrée se fait dans les deux `SpinRow` du dessus.
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
    // La rangée est mémorisée par la borne basse : c'est elle qui pilote la
    // visibilité du couple.
    low.row = Some(row.clone());
    row
}

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
    /// Panneau de réglages / aperçu. Se replie sous 900 px de large.
    pub split: adw::OverlaySplitView,
    /// Bouton d'en-tête qui déploie le panneau lorsqu'il est replié. Masqué
    /// tant que la fenêtre est assez large pour les afficher côte à côte.
    pub panel_toggle: gtk::ToggleButton,
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
    /// Cases de la grille de vignettes, dans l'ordre des presets. Vide si les
    /// vignettes ne sont pas installées — la liste déroulante prend alors le
    /// relais.
    pub preset_thumbs: Vec<gtk::Widget>,
    /// Légende sous la grille : nom du preset courant, ou « Personnalisé ».
    pub preset_caption: gtk::Label,
    pub load_preset: gtk::Button,
    pub save_preset: gtk::Button,
    pub open_folder: gtk::Button,
    pub folder_label: gtk::Label,
    pub folder_row: adw::ActionRow,

    // ── Algorithme ───────────────────────────────────────────────────────────
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

    // ── Halftone ─────────────────────────────────────────────────────────────
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

    // ── Forme des points ─────────────────────────────────────────────────────
    pub shape: adw::ComboRow,
    pub ellipse_box: gtk::ListBox,
    pub ellipse_aspect: adw::SpinRow,
    pub ellipse_angle: adw::SpinRow,
    pub polygon_box: gtk::ListBox,
    pub polygon_sides: adw::SpinRow,
    pub use_palette: adw::SwitchRow,
    pub palette_size: adw::SpinRow,
    pub dithering: adw::ComboRow,

    // ── Rendu ────────────────────────────────────────────────────────────────
    pub bg_button: gtk::ColorDialogButton,
    pub transparent: adw::SwitchRow,
    pub gamma: adw::SwitchRow,
    pub use_seed: adw::SwitchRow,
    pub seed: gtk::SpinButton,
    /// Rangée « Graine » + bouton « Varier » : masquée tant que la graine n'est
    /// pas fixée (cf. `sections::update_sections`).
    pub seed_row: adw::ActionRow,
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
    pub export_sizes: gtk::Label,
    pub busy_label: gtk::Label,
    pub quality_badge: gtk::Label,

    pub src_pane: Pane,
    pub dst_pane: Pane,
    pub density_pane: Pane,
    pub compare_pane: ComparePane,

    // ── État vide / barre d'état ─────────────────────────────────────────────
    pub empty_page: adw::StatusPage,
    /// Zone sous les boutons de l'état d'accueil : accueille les images
    /// récentes.
    pub empty_extra: gtk::Box,
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
    "Carte de densité",
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
    let menu = gtk::MenuButton::new();
    menu.set_menu_model(Some(&crate::wire::menu_model()));
    let toast_overlay = adw::ToastOverlay::new();
    let shortcuts = adw::Window::new();

    let preset_combo = combo_row("Preset", &["Personnalisé"], None);
    let load_preset = gtk::Button::with_label("Charger un preset…");
    let save_preset = gtk::Button::with_label("Sauver le preset…");
    let open_folder = gtk::Button::new();
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
    // Curseurs : la molette du `SpinButton` a un pas fixe (0,1), ce qui rend
    // les 4 décimales du rayon min pénibles à atteindre. Le curseur se règle
    // à la souris ; la `SpinRow` reste l'outil de la valeur exacte.
    let mut variance_slider = ValueSlider::new(0.0, 1.0, 2, Curve::Linear);
    let mut min_radius_slider = ValueSlider::new(0.0005, 1.0, 4, Curve::Log);
    let mut max_radius_slider = ValueSlider::new(0.0005, 1.0, 4, Curve::Log);
    let mut max_boost_slider = ValueSlider::new(1.0, 8.0, 2, Curve::Log);
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
    let mut cols_slider = ValueSlider::new(2.0, 8192.0, 0, Curve::Log);
    let mut num_points_slider = ValueSlider::new(10.0, 100_000.0, 0, Curve::Log);
    let iterations = spin_row(
        "Itérations",
        Some("1 – 100"),
        "Itérations de relaxation de Lloyd (K-means, Voronoï). Davantage = plus régulier, plus lent.",
        1.0,
        100.0,
        1.0,
        0,
    );
    // Linéaire, comme dans l'interface egui : sur 1 → 100, une échelle
    // logarithmique gaspille 80 % de la course entre 1 et 10, alors que
    // c'est justement là que se joue le réglage (« 5 ou 10 itérations ?»).
    let mut iterations_slider = ValueSlider::new(1.0, 100.0, 0, Curve::Linear);

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
    let mut halftone_freq_slider = ValueSlider::new(4.0, 400.0, 0, Curve::Log);
    // La plage 0 → 10 passe en racine : en linéaire, les valeurs utiles
    // (0,002 et 0,85) seraient tassées dans le premier dixième du rail.
    let mut halftone_min_slider = ValueSlider::new(0.0, 10.0, 3, Curve::Sqrt);
    let mut halftone_max_slider = ValueSlider::new(0.0, 10.0, 2, Curve::Sqrt);

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
    // Le dithering est devenu un choix d'algorithme : l'interrupteur ne
    // suffisait plus à distinguer Floyd-Steinberg, Riemersma et Bayer.
    let dithering = combo_row(
        "Dithering",
        &[
            "Aucun",
            "Floyd-Steinberg",
            "Riemersma",
            "Bayer 4×4",
            "Bayer 8×8",
        ],
        Some("Tramage de la palette quantifiée"),
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
    let density_pane = preview_pane("Carte de densité");
    let compare_pane = crate::compare::build_pane();

    let empty_page = adw::StatusPage::new();
    let memory = gtk::Label::new(None);
    let export_sizes = gtk::Label::new(None);
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

    // Grille de vignettes : un preset se choisit sur son rendu, pas sur son nom.
    // Présente seulement si les vignettes sont là — sinon la liste déroulante,
    // qui reste le sélecteur de référence, fait le travail seule.
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
    algo_group.add(
        &variance_slider
            .row("Ajuster la sensibilité à la souris ; l'infobulle du réglage explique son effet"),
    );
    algo_group.add(&min_radius);
    algo_group.add(&max_radius);
    algo_group.add(&pair_row(
        &mut min_radius_slider,
        &mut max_radius_slider,
        "Curseurs des bornes min et max. Échelle logarithmique ; le repère \
         marque la position de l'autre borne.",
    ));
    algo_group.add(&max_boost);
    algo_group
        .add(&max_boost_slider.row("Ajuster l'agrandissement des points en zones sans détail"));
    page.add(&algo_group);

    let grid_group = adw::PreferencesGroup::new();
    grid_group.set_title("Grille");
    grid_group.add(&cols);
    grid_group.add(
        &cols_slider.row("Nombre de cellules de la grille, de 2 à 8 192, en échelle logarithmique"),
    );
    grid_group.add(&grid_angle);
    page.add(&grid_group);

    let points_group = adw::PreferencesGroup::new();
    points_group.set_title("Densité de points");
    points_group.add(&num_points);
    points_group
        .add(&num_points_slider.row("Nombre de points, de 10 à 100 000, en échelle logarithmique"));
    points_group.add(&iterations);
    points_group.add(
        &iterations_slider
            .row("Passes de Lloyd ; au-delà d'une trentaine, le gain devient marginal"),
    );
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
    halftone_group.add(
        &halftone_freq_slider
            .row("Fréquence de trame, de 4 à 400 cellules, en échelle logarithmique"),
    );
    halftone_group.add(&halftone_min_radius);
    halftone_group.add(&halftone_max_dot);
    halftone_group.add(&pair_row(
        &mut halftone_min_slider,
        &mut halftone_max_slider,
        "Curseurs des bornes min et max du halftone ; le repère marque la position de l'autre borne",
    ));
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
    // Largeur imposée : sans cela le conteneur s'étire à la largeur naturelle de
    // la ligne la plus large du panneau et celui-ci mange la moitié de la
    // fenêtre. `propagate_natural_width(false)` empêche la largeur du contenu de
    // remonter jusqu'à la taille demandée — c'est lui qui faisait 540 px.
    controls_scroll.set_propagate_natural_width(false);
    controls_scroll.set_size_request(360, -1);

    // ── En-tête ──────────────────────────────────────────────────────────────
    // `adw::ButtonContent` impose le contenu (icône + libellé) : `set_label`
    // suivi de `set_icon_name` sur un GtkButton ne garde que l'icône.
    set_button_content(&open, "document-open-symbolic", "Ouvrir…");
    open.set_tooltip_text(Some("Ouvrir une image (Ctrl+O)"));
    set_button_content(&open_folder, "folder-open-symbolic", "Dossier…");
    open_folder.set_tooltip_text(Some(
        "Ouvrir un dossier et passer d'une image à l'autre avec ← et → (Ctrl+Maj+O)",
    ));
    // Révèle le panneau de réglages quand il est replié (fenêtre étroite) :
    // le bouton n'existe que dans ce cas, cf. le point de rupture plus bas.
    let panel_toggle = gtk::ToggleButton::new();
    panel_toggle.set_icon_name("sidebar-show-symbolic");
    panel_toggle.set_tooltip_text(Some("Afficher les réglages"));
    panel_toggle.set_visible(false);
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
    // Le bouton ne propose que les exports : le menu complet (dossier,
    // presets, affichage, raccourcis) est sous le bouton hamburger.
    export.set_menu_model(Some(&crate::wire::export_menu_model()));
    menu.set_tooltip_text(Some("Menu principal"));

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
    // Les images récentes s'ajoutent sous les boutons par `ui.rs`, qui
    // connaît l'état : la page d'accueil les accueille sans les remplacer.
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

    // `AdwOverlaySplitView` plutôt qu'un `GtkPaned` : le panneau garde une
    // largeur imposée, et se replie au-dessus de l'aperçu quand la fenêtre
    // devient étroite. Un `Paned` ne sait pas le faire, et imposait la largeur
    // naturelle de sa rangée la plus large (540 px, dictée par la rangée de
    // boutons) : il ne laissait que 280 px d'aperçu en fenêtre 820 px.
    let split = adw::OverlaySplitView::new();
    split.set_sidebar(Some(&controls_scroll));
    split.set_content(Some(&preview));
    split.set_collapsed(false);
    split.set_min_sidebar_width(320.0);
    split.set_max_sidebar_width(400.0);
    split.set_sidebar_width_unit(adw::LengthUnit::Px);
    split.set_show_sidebar(true);

    // ── Barre d'état ─────────────────────────────────────────────────────────
    let status_bar = gtk::Box::new(gtk::Orientation::Horizontal, 14);
    status_bar.set_margin_top(4);
    status_bar.set_margin_bottom(6);
    status_bar.set_margin_start(12);
    status_bar.set_margin_end(12);
    for label in [&memory, &export_sizes, &timing] {
        label.set_xalign(0.0);
        label.add_css_class("dim-label");
        // La liste des formats est longue (« PNG 1.4 Mo · WebP — · JPEG… »)
        // et ne tient pas dans une fenêtre étroite : sans élision elle était
        // coupée net, sans aucun signe de continuation.
        label.set_ellipsize(gtk::pango::EllipsizeMode::End);
        // Le libellé ne doit pas non plus s'étendre : il prend ce dont il a
        // besoin et cède le reste au message d'état, qui est le champ qui
        // change.
        label.set_max_width_chars(40);
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
    toolbar.set_content(Some(&split));
    toolbar.add_bottom_bar(&status_bar);

    toast_overlay.set_child(Some(&toolbar));

    let toast_overlay_for_return = toast_overlay.clone();
    let widgets = Widgets {
        window_title,
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
        preset_combo,
        preset_thumbs,
        preset_caption,
        load_preset,
        save_preset,
        open_folder,
        folder_label,
        folder_row,
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
        seed_row,
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
        export_sizes,
        busy_label,
        quality_badge,
        src_pane,
        dst_pane,
        density_pane,
        compare_pane,
        empty_page,
        empty_extra,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Curseur des rayons : logarithmique sur 0,0005 → 1, au pas de la
    /// `SpinRow` (4 décimales).
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
        // Le curseur travaille en position relative : sans arrondi, la valeur
        // demanderait sept décimales et le champ voisin affiche « 0,2531794 ».
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
