use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use image::GenericImageView;
use log::LevelFilter;
use pointimg::filter::{self, Algorithm, DotShape, FilterParams, HalftoneMode, Screening};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

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
    input: Option<String>,

    /// Fichier de sortie. Pattern supporté en batch : `{n}` (index),
    /// `{stem}` (nom source sans ext.), `{name}` (nom source complet).
    #[arg(short, long, default_value = "output.png")]
    output: String,

    /// Algorithme de placement des points
    #[arg(short, long, value_enum, default_value_t = Algorithm::Voronoi)]
    algorithm: Algorithm,

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

    /// Correction gamma des moyennes de couleur des dots (espace linéaire)
    #[arg(long)]
    gamma: bool,

    /// Dithering Floyd-Steinberg sur la palette quantifiée (effet "offset print")
    #[arg(long)]
    dithering: bool,

    /// Aperçu rapide : sous-échantillonne l'image à WxH max avant le pipeline
    /// (préserve l'aspect ratio). Format : `--preview 200x150`.
    #[arg(long, value_name = "WxH")]
    preview: Option<String>,

    /// Angle de rotation de la grille (degrés), effet "halftone screen" (Grid uniquement)
    #[arg(long, default_value_t = 0.0)]
    grid_angle: f32,

    /// Mode halftone : `cmyk` (rosette 4 canaux), `dominant-N` (N couleurs via
    /// k-means, ex: `dominant-6`). Désactivé par défaut. Active automatiquement
    /// l'algorithme Halftone (ignore num_points, cols, iterations, etc.).
    #[arg(long, default_value = "off")]
    halftone: String,

    /// Screening AM (grille rotationnée) ou FM (blue noise stochastique)
    #[arg(long, default_value = "am")]
    screening: String,

    /// Fréquence de trame AM (cells per min(W,H)) — ex: 60 = ~60 dots par le petit côté
    #[arg(long, default_value_t = 60.0)]
    halftone_freq: f32,

    /// Rayon minimum du dot halftone (fraction de min(W,H))
    #[arg(long, default_value_t = 0.002)]
    halftone_min_radius: f32,

    /// Rayon maximum du dot halftone (fraction du step de trame)
    #[arg(long, default_value_t = 0.85)]
    halftone_max_dot: f32,

    /// Niveau de verbosité (`-v` = info, `-vv` = debug). Par défaut : warnings seulement.
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    verbose: u8,

    /// Supprime les messages d'information (warnings seulement).
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Charger les FilterParams depuis un fichier TOML (remplace tous les flags individuels).
    #[arg(long, value_name = "FILE")]
    preset: Option<PathBuf>,

    /// Sauvegarder les FilterParams actuels dans un fichier TOML.
    #[arg(long, value_name = "FILE")]
    save_preset: Option<PathBuf>,
}

/// Parse `--halftone` : `off`, `cmyk`, ou `dominant-N` (ex `dominant-6`).
fn parse_halftone(s: &str) -> anyhow::Result<HalftoneMode> {
    let s = s.to_lowercase();
    if s == "off" || s.is_empty() {
        return Ok(HalftoneMode::Off);
    }
    if s == "cmyk" {
        return Ok(HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0], // Cyan, Magenta, Jaune, Noir
        });
    }
    if let Some(rest) = s.strip_prefix("dominant-") {
        let n: usize = rest.parse().map_err(|_| {
            anyhow::anyhow!("`dominant-N` : N doit être un entier ≥ 2, got '{}'", rest)
        })?;
        if n < 2 {
            anyhow::bail!("`dominant-N` : N doit être ≥ 2, got {n}");
        }
        return Ok(HalftoneMode::Dominant {
            n,
            base_angle_deg: 15.0,
        });
    }
    anyhow::bail!(
        "halftone invalide '{s}'. Valeurs valides : off, cmyk, dominant-N (ex: dominant-6)"
    );
}

/// Parse `--screening` : `am` (grille rotationnée) ou `fm` (blue noise).
fn parse_screening(s: &str) -> anyhow::Result<Screening> {
    match s.to_lowercase().as_str() {
        "am" => Ok(Screening::Am),
        "fm" => Ok(Screening::Fm),
        other => anyhow::bail!("screening invalide '{other}'. Valeurs valides : am, fm"),
    }
}

/// Parse une couleur de fond : "white", "black", "transparent" (alias "none"),
/// ou "#rrggbb" / "rrggbb". Retourne `(bg_color, transparent)`.
fn parse_bg_color(s: &str) -> anyhow::Result<([u8; 3], bool)> {
    match s.to_lowercase().as_str() {
        "white" => Ok(([255, 255, 255], false)),
        "black" => Ok(([0, 0, 0], false)),
        "transparent" | "none" => Ok(([0, 0, 0], true)),
        hex => {
            let hex = hex.strip_prefix('#').unwrap_or(hex);
            if hex.len() != 6 {
                anyhow::bail!(
                    "Couleur invalide '{}'. Valeurs valides : white, black, transparent, #rrggbb",
                    s
                );
            }
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| anyhow::anyhow!("Couleur invalide '{}'", s))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| anyhow::anyhow!("Couleur invalide '{}'", s))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| anyhow::anyhow!("Couleur invalide '{}'", s))?;
            Ok(([r, g, b], false))
        }
    }
}

impl Args {
    /// Construit `FilterParams` depuis les flags CLI. Centralise le mapping
    /// flag-par-flag afin que `main()` ne fasse que router (presets, batch,
    /// process_one). Les valeurs invalides (`--bg`, `--halftone`, `--screening`)
    /// sont rejetées ici via les parseurs dédiés.
    fn to_filter_params(&self) -> Result<FilterParams> {
        let (bg_color, transparent) = parse_bg_color(&self.bg)?;
        let halftone = parse_halftone(&self.halftone)?;
        let screening = parse_screening(&self.screening)?;
        // Si --halftone est spécifié (≠ Off), on force algorithm = Halftone.
        // Inversement, si --algorithm halftone est choisi mais --halftone est
        // off, on active cmyk par défaut pour ne pas avoir un algorithme vide.
        let algorithm = if halftone != HalftoneMode::Off {
            Algorithm::Halftone
        } else if self.algorithm == Algorithm::Halftone {
            // --algorithm halftone sans --halftone explicite → défaut cmyk.
            return Err(anyhow::anyhow!(
                "--algorithm halftone nécessite aussi --halftone cmyk ou --halftone dominant-N"
            ));
        } else {
            self.algorithm
        };
        let halftone = if algorithm == Algorithm::Halftone && halftone == HalftoneMode::Off {
            HalftoneMode::Cmyk {
                angles: [15.0, 75.0, 0.0, 45.0],
            }
        } else {
            halftone
        };
        let dot_shape = match self.shape {
            ShapeArg::Circle => DotShape::Circle,
            ShapeArg::Square => DotShape::Square,
            ShapeArg::Ellipse => DotShape::Ellipse {
                aspect: self.ellipse_aspect,
                angle_deg: self.ellipse_angle,
            },
            ShapeArg::Polygon => DotShape::RegularPolygon {
                sides: self.polygon_sides.clamp(3, 12),
            },
        };
        Ok(FilterParams {
            algorithm,
            num_points: self.num_points,
            cols: self.cols,
            min_radius_ratio: self.min_radius,
            max_radius_ratio: self.max_radius,
            bg_color,
            iterations: self.iterations,
            variance_sensitivity: self.variance_sensitivity,
            max_boost: self.max_boost,
            rng_seed: self.seed,
            palette_size: self.palette,
            dot_shape,
            transparent,
            gamma_correct: self.gamma,
            dithering: self.dithering,
            grid_angle_deg: self.grid_angle,
            halftone,
            screening,
            halftone_frequency: self.halftone_freq,
            halftone_min_radius_ratio: self.halftone_min_radius,
            halftone_max_dot_ratio: self.halftone_max_dot,
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Configuration des logs selon --verbose / --quiet. RUST_LOG peut toujours
    // surcharger ces niveaux (parse_default_env).
    let level = if args.quiet {
        LevelFilter::Warn
    } else {
        match args.verbose {
            0 => LevelFilter::Warn,
            1 => LevelFilter::Info,
            _ => LevelFilter::Debug,
        }
    };
    let mut builder = env_logger::Builder::new();
    builder
        .filter_level(level)
        .filter_module("pointimg", level)
        .parse_default_env()
        .format_timestamp(None);
    let _ = builder.try_init();

    log::info!("pointimg {}", env!("CARGO_PKG_VERSION"));

    // Construction des FilterParams depuis les flags CLI. Centralisée dans
    // `Args::to_filter_params` pour aérer `main()` (le reste du main ne fait que
    // router : presets, batch, process_one).
    let mut params = args.to_filter_params()?;

    // Preset TOML : chargé depuis un fichier, surcharge les flags.
    // Sauvegardé dans un fichier (sortie anticipée après écriture).
    if let Some(ref preset_path) = args.preset {
        let txt = std::fs::read_to_string(preset_path)
            .with_context(|| format!("Impossible de lire le preset '{}'", preset_path.display()))?;
        let loaded = FilterParams::from_toml_str(&txt)
            .with_context(|| format!("Preset invalide : '{}'", preset_path.display()))?;
        params = loaded;
        log::info!("Preset chargé : {}", preset_path.display());
    }
    if let Some(ref save_path) = args.save_preset {
        let txt = params
            .to_toml_string()
            .with_context(|| "Impossible de sérialiser les paramètres")?;
        std::fs::write(save_path, &txt)
            .with_context(|| format!("Impossible d'écrire '{}'", save_path.display()))?;
        println!("Preset sauvegardé : {}", save_path.display());
        // Si --save-preset est fourni sans --input, on sort après l'écriture
        // (pas de pipeline à exécuter).
        if args.input.is_none() {
            return Ok(());
        }
    }

    // Expansion des entrées : fichier unique, glob (`*`, `?`, `[`), ou dossier.
    let input_str = args.input.as_deref().ok_or_else(|| {
        anyhow::anyhow!("--input est requis sauf si --save-preset seul est fourni (utilise --help)")
    })?;
    let inputs = expand_inputs(input_str)
        .with_context(|| format!("Impossible d'énumérer les entrées '{}'", input_str))?;
    if inputs.is_empty() {
        anyhow::bail!("Aucun fichier d'entrée trouvé pour '{}'", input_str);
    }
    log::info!("{} fichier(s) à traiter", inputs.len());

    let preview_size = args
        .preview
        .as_deref()
        .map(parse_preview_size)
        .transpose()
        .context("--preview invalide")?;
    if let Some((pw, ph)) = preview_size {
        log::info!("Mode preview : {:?}x{:?} max", pw, ph);
    }

    let total = inputs.len();
    let default_output = args.output == "output.png";
    for (i, input_path) in inputs.iter().enumerate() {
        let out = resolve_output_path(&args.output, input_path, i, total, default_output, args.svg);
        if total > 1 {
            println!("[{}/{}] {}", i + 1, total, input_path.display());
        }
        if let Err(e) = process_one(input_path, &out, &params, &args, preview_size) {
            log::error!("échec '{}': {}", input_path.display(), e);
            // Continue le batch : on ne stoppe pas toute la file pour un fichier défectueux.
        }
    }

    Ok(())
}

/// Parse `"WxH"` (case-insensitive) en `(u32, u32)`. Ex : `"200x150"` → `(200, 150)`.
fn parse_preview_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X', '×'])
        .ok_or_else(|| anyhow::anyhow!("format attendu : WxH (ex. 200x150), got '{}'", s))?;
    let w: u32 = w
        .parse()
        .map_err(|_| anyhow::anyhow!("largeur invalide dans '{}'", s))?;
    let h: u32 = h
        .parse()
        .map_err(|_| anyhow::anyhow!("hauteur invalide dans '{}'", s))?;
    if w == 0 || h == 0 {
        anyhow::bail!("preview WxH doit être > 0, got '{}'", s);
    }
    Ok((w, h))
}

/// Sous-échantillonne `src` pour qu'elle tienne dans `max_w × max_h` en préservant
/// l'aspect ratio. Retourne `src` inchangée si elle est déjà plus petite ou `None`.
fn downscale_for_preview(src: &image::DynamicImage, max_w: u32, max_h: u32) -> image::DynamicImage {
    let (w, h) = src.dimensions();
    if w <= max_w && h <= max_h {
        return src.clone();
    }
    let scale = (max_w as f32 / w as f32).min(max_h as f32 / h as f32);
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    log::info!(
        "downscale {:?} → {}x{} (preview)",
        src.dimensions(),
        new_w,
        new_h
    );
    image::DynamicImage::ImageRgb8(image::imageops::resize(
        &src.to_rgb8(),
        new_w,
        new_h,
        image::imageops::FilterType::Triangle,
    ))
}

/// Énumère les fichiers d'entrée depuis :
/// - un glob (`*`, `?`, `[` présent) via `glob::glob` ;
/// - un dossier (liste les fichiers图像) ;
/// - sinon un chemin unique (renvoyé tel quel même s'il n'existe pas — l'erreur viendra à l'open).
fn expand_inputs(input: &str) -> Result<Vec<PathBuf>> {
    let has_glob_chars = input.contains('*') || input.contains('?') || input.contains('[');
    if has_glob_chars {
        let mut out = Vec::new();
        for entry in glob::glob(input).with_context(|| format!("glob invalide '{}'", input))? {
            let path = entry?;
            if path.is_file() {
                out.push(path);
            }
        }
        out.sort();
        return Ok(out);
    }
    let p = PathBuf::from(input);
    if p.is_dir() {
        const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"];
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&p)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
                    .unwrap_or(false)
            {
                out.push(path);
            }
        }
        out.sort();
        return Ok(out);
    }
    Ok(vec![p])
}

/// Calcule le chemin de sortie à partir d'un pattern `--output`, d'un chemin d'entrée
/// et de l'index dans le batch.
///
/// Substitutions reconnues :
///   `{n}`    → numéro du fichier (0-padded, largeur = nb de digits du total)
///   `{stem}` → nom du fichier sans extension (`photo.jpg` → `photo`)
///   `{name}` → nom complet du fichier (`photo.jpg`)
///
/// Si `--output` est la valeur par défaut (`output.png`), on utilise `{stem}.png` (batch)
/// ou `output.png` (fichier unique). Si `--svg`, l'extension est forcée à `.svg`.
fn resolve_output_path(
    output_pattern: &str,
    input: &std::path::Path,
    index: usize,
    total: usize,
    default_output: bool,
    svg: bool,
) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output")
        .to_string();
    let name = input
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output")
        .to_string();
    let width = total.to_string().len();
    let n_padded = format!("{:0>width$}", index, width = width);

    let resolved = if total == 1 && default_output {
        if svg {
            "output.svg".to_string()
        } else {
            "output.png".to_string()
        }
    } else {
        let has_per_file_token = output_pattern.contains("{n}")
            || output_pattern.contains("{stem}")
            || output_pattern.contains("{name}");
        let mut s = output_pattern
            .replace("{n}", &n_padded)
            .replace("{stem}", &stem)
            .replace("{name}", &name);
        // Si l'utilisateur n'a mis aucun token par-fichier mais qu'on a plusieurs
        // fichiers, on ajoute _{n} avant l'extension pour éviter d'écraser.
        if total > 1 && !has_per_file_token {
            if let Some(dot) = s.rfind('.') {
                s.insert_str(dot, &format!("_{}", n_padded));
            } else {
                s.push_str(&format!("_{}", n_padded));
            }
        }
        if svg {
            // Forcer l'extension à `.svg` (remplace la dernière).
            if let Some(dot) = s.rfind('.') {
                s.truncate(dot);
            }
            s.push_str(".svg");
        }
        s
    };
    PathBuf::from(resolved)
}

/// Traite un fichier d'entrée unique : ouvre, pipeline (RGB/RGBA/SVG), sauvegarde.
fn process_one(
    input: &std::path::Path,
    output: &std::path::Path,
    params: &FilterParams,
    args: &Args,
    preview_size: Option<(u32, u32)>,
) -> Result<()> {
    let src_orig =
        image::open(input).with_context(|| format!("Impossible d'ouvrir '{}'", input.display()))?;
    // Sous-échantillonnage preview si demandé.
    let src = match preview_size {
        Some((pw, ph)) => downscale_for_preview(&src_orig, pw, ph),
        None => src_orig,
    };

    let rgb = filter::flatten_to_rgb(&src, params.bg_color);
    let never_cancel = AtomicBool::new(false);

    let show_progress =
        matches!(params.algorithm, Algorithm::Voronoi | Algorithm::Kmeans) && params.iterations > 1;

    if args.svg {
        let (_, dots) =
            filter::apply_with_progress(&rgb, params, &never_cancel, |iter, total, _| {
                if show_progress {
                    eprint!("\r  Iteration {iter}/{total}");
                }
            })
            .with_context(|| "Erreur lors du calcul du filtre")?;
        if show_progress {
            eprintln!();
        }
        let (w, h) = rgb.dimensions();
        let svg = filter::render_svg_from_dots(w, h, &dots, params)
            .with_context(|| "Erreur lors du rendu SVG")?;
        std::fs::write(output, svg)
            .with_context(|| format!("Impossible d'ecrire '{}'", output.display()))?;
        println!("SVG sauvegarde : {}", output.display());
    } else if params.transparent {
        let (dst, _dots) = filter::apply_rgba(&rgb, params)
            .with_context(|| "Erreur lors du calcul du filtre (RGBA)")?;
        dst.save(output)
            .with_context(|| format!("Impossible de sauvegarder '{}'", output.display()))?;
        println!("Sauvegarde (RGBA) : {}", output.display());
    } else {
        let (dst, _dots) =
            filter::apply_with_progress(&rgb, params, &never_cancel, |iter, total, _| {
                if show_progress {
                    eprint!("\r  Iteration {iter}/{total}");
                }
            })
            .with_context(|| "Erreur lors du calcul du filtre")?;
        if show_progress {
            eprintln!();
        }
        dst.save(output)
            .with_context(|| format!("Impossible de sauvegarder '{}'", output.display()))?;
        println!("Sauvegarde : {}", output.display());
    }
    Ok(())
}
