use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use pointimg::filter::{self, Algorithm, DotShape, FilterParams};
use std::sync::atomic::AtomicBool;

// U1: clap::ValueEnum for type-safe CLI parsing
#[derive(Clone, Copy, Debug, ValueEnum)]
enum AlgoArg {
    Grid,
    Kmeans,
    Voronoi,
    Quadtree,
}

impl From<AlgoArg> for Algorithm {
    fn from(a: AlgoArg) -> Self {
        match a {
            AlgoArg::Grid => Algorithm::Grid,
            AlgoArg::Kmeans => Algorithm::Kmeans,
            AlgoArg::Voronoi => Algorithm::Voronoi,
            AlgoArg::Quadtree => Algorithm::Quadtree,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ShapeArg {
    Circle,
    Square,
    Ellipse,
    Polygon,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Filtre pointilliste — CLI")]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long, default_value = "output.png")]
    output: String,

    /// Algorithme de placement des points
    #[arg(short, long, value_enum, default_value_t = AlgoArg::Voronoi)]
    algorithm: AlgoArg,

    /// Nombre de points (kmeans / voronoi / quadtree)
    #[arg(short, long, default_value_t = 800)]
    num_points: usize,

    /// Nombre de colonnes (grid uniquement)
    #[arg(short, long, default_value_t = 80)]
    cols: u32,

    /// Rayon minimum : fraction de min(largeur, hauteur)
    #[arg(long, default_value_t = 0.003)]
    min_radius: f32,

    /// Rayon maximum : fraction de min(largeur, hauteur)
    #[arg(long, default_value_t = 0.06)]
    max_radius: f32,

    #[arg(short, long, default_value = "white")]
    bg: String,

    /// Forme des points
    #[arg(long, value_enum, default_value_t = ShapeArg::Circle)]
    shape: ShapeArg,

    /// Ratio largeur/hauteur pour l'ellipse (ex. 1.5)
    // U2: only relevant for --shape ellipse
    #[arg(long, default_value_t = 1.5, requires_if("ellipse", "shape"))]
    ellipse_aspect: f32,

    /// Angle de rotation en degres pour l'ellipse
    #[arg(long, default_value_t = 0.0, requires_if("ellipse", "shape"))]
    ellipse_angle: f32,

    /// Nombre de cotes pour le polygone regulier (3-12)
    #[arg(long, default_value_t = 6, requires_if("polygon", "shape"))]
    polygon_sides: u8,

    /// Nombre d'iterations (kmeans / voronoi)
    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// Sensibilite a la variance (0.0 = ignoree, 1.0 = forte redistribution)
    #[arg(long, default_value_t = 0.7)]
    variance_sensitivity: f32,

    /// Multiplicateur max de rayon dans les zones uniformes
    #[arg(long, default_value_t = 2.5)]
    max_boost: f32,

    /// Graine RNG pour resultats reproductibles
    #[arg(long)]
    seed: Option<u64>,

    /// Nombre de couleurs pour la quantification de palette
    #[arg(long)]
    palette: Option<usize>,

    /// Exporter en SVG au lieu de PNG/JPEG
    #[arg(long)]
    svg: bool,
}

/// Parse une couleur de fond : "white", "black", ou "#rrggbb" / "rrggbb".
fn parse_bg_color(s: &str) -> anyhow::Result<[u8; 3]> {
    match s.to_lowercase().as_str() {
        "white" => Ok([255, 255, 255]),
        "black" => Ok([0, 0, 0]),
        hex => {
            let hex = hex.strip_prefix('#').unwrap_or(hex);
            if hex.len() != 6 {
                anyhow::bail!(
                    "Couleur invalide '{}'. Valeurs valides : white, black, #rrggbb",
                    s
                );
            }
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| anyhow::anyhow!("Couleur invalide '{}'", s))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| anyhow::anyhow!("Couleur invalide '{}'", s))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| anyhow::anyhow!("Couleur invalide '{}'", s))?;
            Ok([r, g, b])
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let src = image::open(&args.input)
        .with_context(|| format!("Impossible d'ouvrir '{}'", args.input))?;

    // U1: algorithm and shape are now parsed by clap via ValueEnum
    let algo: Algorithm = args.algorithm.into();

    let bg_color: [u8; 3] = parse_bg_color(&args.bg)?;

    let dot_shape = match args.shape {
        ShapeArg::Circle => DotShape::Circle,
        ShapeArg::Square => DotShape::Square,
        ShapeArg::Ellipse => DotShape::Ellipse {
            aspect: args.ellipse_aspect,
            angle_deg: args.ellipse_angle,
        },
        ShapeArg::Polygon => DotShape::RegularPolygon {
            sides: args.polygon_sides.clamp(3, 12),
        },
    };

    let params = FilterParams {
        algorithm: algo,
        num_points: args.num_points,
        cols: args.cols,
        min_radius_ratio: args.min_radius,
        max_radius_ratio: args.max_radius,
        bg_color,
        iterations: args.iterations,
        variance_sensitivity: args.variance_sensitivity,
        max_boost: args.max_boost,
        rng_seed: args.seed,
        palette_size: args.palette,
        dot_shape,
    };

    let rgb = filter::flatten_to_rgb(&src, params.bg_color);
    let never_cancel = AtomicBool::new(false);

    // U5: determine if we should show iteration progress
    let show_progress =
        matches!(algo, Algorithm::Voronoi | Algorithm::Kmeans) && params.iterations > 1;

    if args.svg {
        let (_, dots) =
            filter::apply_with_progress(&rgb, &params, &never_cancel, |iter, total, _| {
                if show_progress {
                    eprint!("\r  Iteration {iter}/{total}");
                }
            })
            .with_context(|| "Erreur lors du calcul du filtre")?;
        if show_progress {
            eprintln!();
        }
        let (w, h) = rgb.dimensions();
        let svg = filter::render_svg_from_dots(w, h, &dots, &params)
            .with_context(|| "Erreur lors du rendu SVG")?;
        let out = if args.output == "output.png" {
            "output.svg".to_string()
        } else {
            args.output.clone()
        };
        std::fs::write(&out, svg).with_context(|| format!("Impossible d'ecrire '{}'", out))?;
        println!("SVG sauvegarde : {}", out);
    } else {
        let (dst, _dots) =
            filter::apply_with_progress(&rgb, &params, &never_cancel, |iter, total, _| {
                if show_progress {
                    eprint!("\r  Iteration {iter}/{total}");
                }
            })
            .with_context(|| "Erreur lors du calcul du filtre")?;
        if show_progress {
            eprintln!();
        }
        dst.save(&args.output)
            .with_context(|| format!("Impossible de sauvegarder '{}'", args.output))?;
        println!("Sauvegarde : {}", args.output);
    }

    Ok(())
}
