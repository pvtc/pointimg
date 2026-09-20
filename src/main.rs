use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use image::{ExtendedColorType, GenericImageView, ImageEncoder, ImageReader, RgbaImage};
use log::LevelFilter;
use pointimg::filter::{self, Algorithm, DotShape, FilterParams, HalftoneMode, Screening};
use pointimg::frontend;
use rayon::prelude::*;
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

    /// Profil ICC d'entrée : auto (profil embarqué), srgb, display-p3 ou chemin .icc.
    #[arg(long, default_value = "auto")]
    input_profile: String,

    /// Profil ICC de sortie : sRGB par défaut ou chemin vers un fichier .icc.
    #[arg(long, default_value = "srgb")]
    output_profile: String,

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

    /// Nombre de fichiers traités en parallèle en mode batch (0 = tous les CPUs).
    /// 1 = séquentiel (défaut, conserve la progression par itération).
    #[arg(long, default_value_t = 1, value_name = "N")]
    jobs: usize,
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
        // Inversement, si --algorithm halftone est choisi sans --halftone
        // explicite, on active le mode cmyk par défaut (voir plus bas) au lieu
        // de produire un algorithme vide.
        let algorithm = if halftone != HalftoneMode::Off {
            Algorithm::Halftone
        } else {
            self.algorithm
        };
        // --algorithm halftone sans --halftone explicite → défaut cmyk.
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
    let parallel = args.jobs != 1 && total > 1;
    if parallel {
        log::info!(
            "batch parallèle : {} worker(s)",
            if args.jobs == 0 {
                num_cpus_hint()
            } else {
                args.jobs
            }
        );
    }
    let failures = if parallel {
        // En parallèle, la progression par itération serait entremêlée sur
        // stderr : on la coupe et on laisse chaque fichier rendre sa ligne de
        // résultat atomiquement.
        // `--jobs 0` laisse rayon choisir le nombre de threads (tous les CPUs).
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build()
            .context("impossible de créer le pool de threads du batch")?;
        pool.install(|| {
            inputs
                .par_iter()
                .enumerate()
                .map(|(i, input_path)| {
                    let out = resolve_output_path(
                        &args.output,
                        input_path,
                        i,
                        total,
                        default_output,
                        args.svg,
                    );
                    match process_one(input_path, &out, &params, &args, preview_size, false) {
                        Ok(()) => 0usize,
                        Err(e) => {
                            log::error!("échec '{}': {}", input_path.display(), e);
                            1
                        }
                    }
                })
                .sum::<usize>()
        })
    } else {
        let mut failures = 0usize;
        for (i, input_path) in inputs.iter().enumerate() {
            let out =
                resolve_output_path(&args.output, input_path, i, total, default_output, args.svg);
            if total > 1 {
                println!("[{}/{}] {}", i + 1, total, input_path.display());
            }
            if let Err(e) = process_one(input_path, &out, &params, &args, preview_size, true) {
                log::error!("échec '{}': {}", input_path.display(), e);
                failures += 1;
                // Continue le batch : on ne stoppe pas toute la file pour un fichier défectueux.
            }
        }
        failures
    };

    if failures > 0 {
        anyhow::bail!("{} fichier(s) n'ont pas pu être traité(s)", failures);
    }
    Ok(())
}

/// Nombre de CPUs disponibles (via le pool global rayon, qui lit `available_parallelism`).
fn num_cpus_hint() -> usize {
    rayon::current_num_threads()
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
/// - un dossier (liste les fichiers image) ;
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
/// `iter_progress` contrôle l'affichage de la progression par itération (coupé
/// en mode batch parallèle pour éviter un stdout/stderr entremêlé).
fn process_one(
    input: &std::path::Path,
    output: &std::path::Path,
    params: &FilterParams,
    args: &Args,
    preview_size: Option<(u32, u32)>,
    iter_progress: bool,
) -> Result<()> {
    let dimensions = ImageReader::open(input)
        .with_context(|| format!("Impossible de lire les dimensions de '{}'", input.display()))?
        .into_dimensions()
        .with_context(|| format!("Dimensions invalides pour '{}'", input.display()))?;
    if dimensions.0 == 0 || dimensions.1 == 0 {
        anyhow::bail!("Image vide ({}x{})", dimensions.0, dimensions.1);
    }
    let (src_orig, profile_converted, was_resized) =
        pointimg::color::decode_to_srgb(input, &args.input_profile)?;
    if profile_converted {
        log::info!("profil colorimétrique converti vers sRGB");
    }
    if was_resized {
        log::warn!(
            "image '{}' réduite automatiquement à {:?} pour respecter les limites mémoire",
            input.display(),
            src_orig.dimensions()
        );
    }
    // Sous-échantillonnage preview si demandé.
    let src = match preview_size {
        Some((pw, ph)) => downscale_for_preview(&src_orig, pw, ph),
        None => src_orig,
    };

    let rgb = filter::flatten_to_rgb(&src, params.bg_color);
    let output_profile = pointimg::color::output_profile_from_spec(&args.output_profile)?;
    if args.svg && output_profile.is_some() {
        anyhow::bail!("--output-profile ne s'applique pas à l'export SVG");
    }
    log::info!(
        "mémoire de travail estimée : {} Mo",
        filter::estimate_memory_bytes_for(rgb.width(), rgb.height(), params) / (1024 * 1024)
    );
    let never_cancel = AtomicBool::new(false);

    let show_progress = iter_progress
        && matches!(params.algorithm, Algorithm::Voronoi | Algorithm::Kmeans)
        && params.iterations > 1;

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
        frontend::atomic_text_write(output, &svg)
            .with_context(|| format!("Impossible d'ecrire '{}'", output.display()))?;
        println!("SVG sauvegarde : {}", output.display());
    } else if params.transparent {
        let (dst, _dots) = filter::apply_rgba(&rgb, params)
            .with_context(|| "Erreur lors du calcul du filtre (RGBA)")?;
        if let Some((profile, bytes)) = output_profile.as_ref() {
            let supports_alpha = matches!(
                output
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("png" | "webp" | "tif" | "tiff")
            );
            let source = if supports_alpha {
                image::DynamicImage::ImageRgba8(dst)
            } else {
                image::DynamicImage::ImageRgb8(frontend::flatten_rgba_on_bg(&dst, params.bg_color))
            };
            let dst = pointimg::color::convert_from_srgb(source, profile)?;
            frontend::atomic_image_save(output, |tmp| {
                save_dynamic_with_profile(&dst, Some(bytes), tmp)
            })
            .with_context(|| format!("Impossible de sauvegarder '{}'", output.display()))?;
        } else {
            frontend::atomic_image_save(output, |tmp| {
                save_rgba_for_output(&dst, params.bg_color, tmp)
            })
            .with_context(|| format!("Impossible de sauvegarder '{}'", output.display()))?;
        }
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
        let dst = if let Some((profile, _)) = output_profile.as_ref() {
            pointimg::color::convert_from_srgb(image::DynamicImage::ImageRgb8(dst), profile)?
        } else {
            image::DynamicImage::ImageRgb8(dst)
        };
        frontend::atomic_image_save(output, |tmp| {
            save_dynamic_with_profile(&dst, output_profile.as_ref().map(|(_, bytes)| bytes), tmp)
        })
        .with_context(|| format!("Impossible de sauvegarder '{}'", output.display()))?;
        println!("Sauvegarde : {}", output.display());
    }
    Ok(())
}

fn save_rgba_for_output(
    image: &RgbaImage,
    bg: [u8; 3],
    path: &std::path::Path,
) -> image::ImageResult<()> {
    // PNG/WebP/TIFF préservent l'alpha ; sinon on composite sur le fond.
    if frontend::extension_preserves_alpha(path) {
        return image.save(path);
    }
    frontend::flatten_rgba_on_bg(image, bg).save(path)
}

fn save_dynamic_with_profile(
    image: &image::DynamicImage,
    profile: Option<&Vec<u8>>,
    path: &std::path::Path,
) -> image::ImageResult<()> {
    let Some(profile) = profile else {
        return image.save(path);
    };
    let file = std::fs::File::create(path).map_err(image::ImageError::IoError)?;
    let (width, height) = image.dimensions();
    let (bytes, color_type) = if image.color().has_alpha() {
        (image.to_rgba8().into_raw(), ExtendedColorType::Rgba8)
    } else {
        (image.to_rgb8().into_raw(), ExtendedColorType::Rgb8)
    };
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => {
            let mut encoder = image::codecs::png::PngEncoder::new(file);
            encoder
                .set_icc_profile(profile.clone())
                .map_err(image::ImageError::Unsupported)?;
            encoder.write_image(&bytes, width, height, color_type)
        }
        Some("jpg") | Some("jpeg") => {
            let mut encoder = image::codecs::jpeg::JpegEncoder::new(file);
            encoder
                .set_icc_profile(profile.clone())
                .map_err(image::ImageError::Unsupported)?;
            encoder.write_image(&bytes, width, height, color_type)
        }
        Some("webp") => {
            let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(file);
            encoder
                .set_icc_profile(profile.clone())
                .map_err(image::ImageError::Unsupported)?;
            encoder.write_image(&bytes, width, height, color_type)
        }
        Some("tif") | Some("tiff") => {
            let mut encoder = image::codecs::tiff::TiffEncoder::new(file);
            encoder
                .set_icc_profile(profile.clone())
                .map_err(image::ImageError::Unsupported)?;
            encoder.write_image(&bytes, width, height, color_type)
        }
        _ => image.save(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_bmp_export_composites_alpha() {
        let path = std::env::temp_dir().join(format!(
            "pointimg-test-{}-{}.bmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let image = RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 128]));
        save_rgba_for_output(&image, [255, 255, 255], &path).unwrap();
        let decoded = image::open(&path).unwrap().to_rgb8();
        let pixel = decoded.get_pixel(0, 0);
        assert!(pixel[0] > 100 && pixel[1] > 100);
        let _ = std::fs::remove_file(path);
    }

    // ── Parsing des couleurs de fond ─────────────────────────────────────────

    #[test]
    fn parse_bg_color_variants() {
        assert_eq!(parse_bg_color("white").unwrap(), ([255, 255, 255], false));
        assert_eq!(parse_bg_color("BLACK").unwrap(), ([0, 0, 0], false));
        assert_eq!(
            parse_bg_color("#1a1a2e").unwrap(),
            ([0x1a, 0x1a, 0x2e], false)
        );
        assert_eq!(
            parse_bg_color("1a1a2e").unwrap(),
            ([0x1a, 0x1a, 0x2e], false)
        );
        assert_eq!(parse_bg_color("transparent").unwrap(), ([0, 0, 0], true));
        assert_eq!(parse_bg_color("none").unwrap(), ([0, 0, 0], true));
        assert!(parse_bg_color("bleu").is_err());
        assert!(parse_bg_color("#12345").is_err());
        assert!(parse_bg_color("#zzzzzz").is_err());
    }

    // ── Parsing halftone / screening ─────────────────────────────────────────

    #[test]
    fn parse_halftone_variants() {
        assert_eq!(parse_halftone("off").unwrap(), HalftoneMode::Off);
        assert_eq!(parse_halftone("").unwrap(), HalftoneMode::Off);
        assert!(matches!(
            parse_halftone("cmyk").unwrap(),
            HalftoneMode::Cmyk { .. }
        ));
        assert!(matches!(
            parse_halftone("dominant-6").unwrap(),
            HalftoneMode::Dominant { n: 6, .. }
        ));
        assert!(parse_halftone("dominant-1").is_err());
        assert!(parse_halftone("dominant-x").is_err());
        assert!(parse_halftone("bogus").is_err());
    }

    #[test]
    fn parse_screening_variants() {
        assert_eq!(parse_screening("am").unwrap(), Screening::Am);
        assert_eq!(parse_screening("FM").unwrap(), Screening::Fm);
        assert!(parse_screening("xx").is_err());
    }

    // ── Parsing --preview WxH ────────────────────────────────────────────────

    #[test]
    fn parse_preview_size_valid_and_invalid() {
        assert_eq!(parse_preview_size("200x150").unwrap(), (200, 150));
        assert_eq!(parse_preview_size("200X150").unwrap(), (200, 150));
        assert!(parse_preview_size("200").is_err());
        assert!(parse_preview_size("0x10").is_err());
        assert!(parse_preview_size("a x b").is_err());
    }

    // ── Mapping flags → FilterParams ─────────────────────────────────────────

    #[test]
    fn algorithm_halftone_without_mode_defaults_to_cmyk() {
        let args = Args::parse_from(["pointimg", "--algorithm", "halftone", "-i", "in.png"]);
        let params = args.to_filter_params().unwrap();
        assert_eq!(params.algorithm, Algorithm::Halftone);
        assert!(matches!(params.halftone, HalftoneMode::Cmyk { .. }));
    }

    #[test]
    fn explicit_halftone_forces_algorithm() {
        let args = Args::parse_from(["pointimg", "--halftone", "dominant-4", "-i", "in.png"]);
        let params = args.to_filter_params().unwrap();
        assert_eq!(params.algorithm, Algorithm::Halftone);
        assert!(matches!(
            params.halftone,
            HalftoneMode::Dominant { n: 4, .. }
        ));
    }

    // ── Génération des chemins de sortie ─────────────────────────────────────

    #[test]
    fn resolve_output_path_substitutions() {
        let resolve = |pattern: &str, index: usize, total: usize| {
            resolve_output_path(
                pattern,
                std::path::Path::new("/tmp/photo.jpg"),
                index,
                total,
                false,
                false,
            )
        };
        assert_eq!(
            resolve("out/{stem}_{n}.png", 2, 10),
            PathBuf::from("out/photo_02.png")
        );
        assert_eq!(
            resolve("out/{name}.png", 0, 1),
            PathBuf::from("out/photo.jpg.png")
        );
        // Aucun token par-fichier + plusieurs fichiers → suffixe `_n` avant l'extension.
        assert_eq!(
            resolve("out/all.png", 3, 10),
            PathBuf::from("out/all_03.png")
        );
        // Sans extension, le suffixe est ajouté à la fin.
        assert_eq!(resolve("out/all", 3, 10), PathBuf::from("out/all_03"));
    }

    #[test]
    fn resolve_output_path_default_and_svg() {
        assert_eq!(
            resolve_output_path(
                "output.png",
                std::path::Path::new("/tmp/a.jpg"),
                0,
                1,
                true,
                false
            ),
            PathBuf::from("output.png")
        );
        assert_eq!(
            resolve_output_path(
                "output.png",
                std::path::Path::new("/tmp/a.jpg"),
                0,
                1,
                true,
                true
            ),
            PathBuf::from("output.svg")
        );
        // `--svg` force l'extension même sur un pattern explicite.
        assert_eq!(
            resolve_output_path(
                "out/x.png",
                std::path::Path::new("/tmp/a.jpg"),
                0,
                1,
                false,
                true
            ),
            PathBuf::from("out/x.svg")
        );
    }
}
