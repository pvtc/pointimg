use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use image::{ExtendedColorType, GenericImageView, ImageEncoder, ImageReader, RgbaImage};
use log::LevelFilter;
use pointimg::filter::{
    self, Algorithm, Dithering, DotShape, FilterParams, HalftoneMode, Screening,
};
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
#[command(author, version, about = pointimg::frontend::t("Filtre pointilliste — CLI"))]
struct Args {
    #[arg(short, long)]
    input: Option<String>,

    /// Output file. Pattern supported in batch: `{n}` (index),
    /// `{stem}` (source name without ext.), `{name}` (full source name).
    #[arg(short, long, default_value = "output.png")]
    output: String,

    /// Point placement algorithm
    #[arg(short, long, value_enum, default_value_t = Algorithm::Voronoi)]
    algorithm: Algorithm,

    /// Number of points (kmeans / voronoi / quadtree)
    #[arg(short, long, default_value_t = 800)]
    num_points: usize,

    /// Number of columns (grid only)
    #[arg(short, long, default_value_t = 80)]
    cols: u32,

    /// Minimum radius: fraction of min(width, height)
    #[arg(long, default_value_t = 0.003)]
    min_radius: f32,

    /// Maximum radius: fraction of min(width, height)
    #[arg(long, default_value_t = 0.06)]
    max_radius: f32,

    #[arg(short, long, default_value = "white")]
    bg: String,

    /// Shape of the dots
    #[arg(long, value_enum, default_value_t = ShapeArg::Circle)]
    shape: ShapeArg,

    /// Width/height ratio for the ellipse (e.g. 1.5)
    #[arg(long, default_value_t = 1.5, requires_if("ellipse", "shape"))]
    ellipse_aspect: f32,

    /// Rotation angle in degrees for the ellipse
    #[arg(long, default_value_t = 0.0, requires_if("ellipse", "shape"))]
    ellipse_angle: f32,

    /// Number of sides for the regular polygon (3-12)
    #[arg(long, default_value_t = 6, requires_if("polygon", "shape"))]
    polygon_sides: u8,

    /// Number of iterations (kmeans / voronoi)
    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// Variance sensitivity (0.0 = ignored, 1.0 = strong redistribution)
    #[arg(long, default_value_t = 0.7)]
    variance_sensitivity: f32,

    /// Maximum radius multiplier in uniform areas
    #[arg(long, default_value_t = 2.5)]
    max_boost: f32,

    /// RNG seed for reproducible results
    #[arg(long)]
    seed: Option<u64>,

    /// Number of colors for palette quantization
    #[arg(long)]
    palette: Option<usize>,

    /// Export as SVG instead of PNG/JPEG
    #[arg(long)]
    svg: bool,

    /// Gamma correction of the average colors of the dots (linear space)
    #[arg(long)]
    gamma: bool,

    /// Dithering on the quantized palette. Without a value: Floyd-Steinberg
    /// (the "offset print" effect). Alternatives: riemersma, bayer4, bayer8, off.
    ///
    /// `Option` so that omitting the option is accepted: with
    /// `num_args = 0..=1`, a plain type would make the argument mandatory.
    #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "floyd-steinberg", value_enum)]
    dithering: Option<Dithering>,

    /// Quick preview: subsamples the image to WxH max before the pipeline
    /// (preserves the aspect ratio). Format: `--preview 200x150`.
    #[arg(long, value_name = "WxH")]
    preview: Option<String>,

    /// Input ICC profile: auto (embedded profile), srgb, display-p3 or an
    /// .icc path.
    #[arg(long, default_value = "auto")]
    input_profile: String,

    /// Output ICC profile: sRGB by default or a path to an .icc file.
    #[arg(long, default_value = "srgb")]
    output_profile: String,

    /// Grid rotation angle (degrees), "halftone screen" effect (Grid only)
    #[arg(long, default_value_t = 0.0)]
    grid_angle: f32,

    /// Halftone mode: `cmyk` (4-channel rosette), `dominant-N` (N colors via
    /// k-means, e.g.: `dominant-6`). Off by default. Automatically activates
    /// the Halftone algorithm (ignores num_points, cols, iterations, etc.).
    #[arg(long, default_value = "off")]
    halftone: String,

    /// AM screening (rotated grid) or FM (stochastic blue noise)
    #[arg(long, default_value = "am")]
    screening: String,

    /// AM screening frequency (cells per min(W,H)) — e.g.: 60 = ~60 dots on the
    /// short side
    #[arg(long, default_value_t = 60.0)]
    halftone_freq: f32,

    /// Minimum radius of the halftone dot (fraction of min(W,H))
    #[arg(long, default_value_t = 0.002)]
    halftone_min_radius: f32,

    /// Maximum radius of the halftone dot (fraction of the screening step)
    #[arg(long, default_value_t = 0.85)]
    halftone_max_dot: f32,

    /// Verbosity level (`-v` = info, `-vv` = debug). By default: warnings only.
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    verbose: u8,

    /// Removes the information messages (warnings only).
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Load the FilterParams from a TOML file (replaces every individual flag).
    #[arg(long, value_name = "FILE")]
    preset: Option<PathBuf>,

    /// Save the current FilterParams to a TOML file.
    #[arg(long, value_name = "FILE")]
    save_preset: Option<PathBuf>,

    /// Number of files processed in parallel in batch mode (0 = all the CPUs).
    /// 1 = sequential (default, preserves the progress by iteration).
    #[arg(long, default_value_t = 1, value_name = "N")]
    jobs: usize,
}

/// Parses `--halftone`: `off`, `cmyk`, or `dominant-N` (e.g. `dominant-6`).
fn parse_halftone(s: &str) -> anyhow::Result<HalftoneMode> {
    let s = s.to_lowercase();
    if s == "off" || s.is_empty() {
        return Ok(HalftoneMode::Off);
    }
    if s == "cmyk" {
        return Ok(HalftoneMode::Cmyk {
            angles: [15.0, 75.0, 0.0, 45.0], // Cyan, Magenta, Yellow, Black
        });
    }
    if let Some(rest) = s.strip_prefix("dominant-") {
        let n: usize = rest.parse().map_err(|_| {
            anyhow::anyhow!(
                "{}",
                pointimg::frontend::t("`dominant-N` : N doit être un entier ≥ 2, got '{}'")
                    .replace("{}", rest)
            )
        })?;
        if n < 2 {
            anyhow::bail!(
                "{}",
                pointimg::frontend::t("`dominant-N` : N doit être ≥ 2, got {n}")
                    .replace("{n}", &n.to_string())
            );
        }
        return Ok(HalftoneMode::Dominant {
            n,
            base_angle_deg: 15.0,
        });
    }
    anyhow::bail!(
        pointimg::frontend::t(
            "halftone invalide '{s}'. Valeurs valides : off, cmyk, dominant-N (ex: dominant-6)"
        )
        .replace("{s}", &s)
    );
}

/// Parses `--screening`: `am` (rotated grid) or `fm` (blue noise).
fn parse_screening(s: &str) -> anyhow::Result<Screening> {
    match s.to_lowercase().as_str() {
        "am" => Ok(Screening::Am),
        "fm" => Ok(Screening::Fm),
        other => anyhow::bail!(
            "{}",
            pointimg::frontend::t("screening invalide '{other}'. Valeurs valides : am, fm")
                .replace("{other}", other)
        ),
    }
}

/// Parses a background color: "white", "black", "transparent" (alias "none"),
/// or "#rrggbb" / "rrggbb". Returns `(bg_color, transparent)`.
fn parse_bg_color(s: &str) -> anyhow::Result<([u8; 3], bool)> {
    match s.to_lowercase().as_str() {
        "white" => Ok(([255, 255, 255], false)),
        "black" => Ok(([0, 0, 0], false)),
        "transparent" | "none" => Ok(([0, 0, 0], true)),
        hex => {
            let hex = hex.strip_prefix('#').unwrap_or(hex);
            if hex.len() != 6 {
                anyhow::bail!(
                    "{}",
                    pointimg::frontend::t("Couleur invalide '{}'. Valeurs valides : white, black, transparent, #rrggbb").replace("{}", s)
                );
            }
            let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| {
                anyhow::anyhow!(
                    "{}",
                    pointimg::frontend::t("Couleur invalide '{}'").replace("{}", s)
                )
            })?;
            let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| {
                anyhow::anyhow!(
                    "{}",
                    pointimg::frontend::t("Couleur invalide '{}'").replace("{}", s)
                )
            })?;
            let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| {
                anyhow::anyhow!(
                    "{}",
                    pointimg::frontend::t("Couleur invalide '{}'").replace("{}", s)
                )
            })?;
            Ok(([r, g, b], false))
        }
    }
}

impl Args {
    /// Builds `FilterParams` from the CLI flags. Centralizes the
    /// flag-by-flag mapping so that `main()` only has to route (presets, batch,
    /// process_one). The invalid values (`--bg`, `--halftone`, `--screening`)
    /// are rejected here through the dedicated parsers.
    fn to_filter_params(&self) -> Result<FilterParams> {
        let (bg_color, transparent) = parse_bg_color(&self.bg)?;
        let halftone = parse_halftone(&self.halftone)?;
        let screening = parse_screening(&self.screening)?;
        // If --halftone is specified (≠ Off), algorithm is forced to Halftone.
        // Conversely, if --algorithm halftone is chosen without an explicit
        // --halftone, the default cmyk mode is activated (see below) instead of
        // producing an empty algorithm.
        let algorithm = if halftone != HalftoneMode::Off {
            Algorithm::Halftone
        } else {
            self.algorithm
        };
        // --algorithm halftone without an explicit --halftone → cmyk default.
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
            dithering: self.dithering.unwrap_or_default(),
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

    // Log configuration according to --verbose / --quiet. RUST_LOG can still
    // override these levels (parse_default_env).
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

    // Building of the FilterParams from the CLI flags. Centralized in
    // `Args::to_filter_params` to lighten `main()` up (the rest of main only
    // has to route: presets, batch, process_one).
    let mut params = args.to_filter_params()?;

    // TOML preset: loaded from a file, overrides the flags.
    // Saved to a file (early exit after writing).
    if let Some(ref preset_path) = args.preset {
        let txt = std::fs::read_to_string(preset_path).with_context(|| {
            pointimg::frontend::t("Impossible de lire le preset '{}'")
                .replace("{}", &preset_path.display().to_string())
        })?;
        let loaded = FilterParams::from_toml_str(&txt).with_context(|| {
            pointimg::frontend::t("Preset invalide : '{}'")
                .replace("{}", &preset_path.display().to_string())
        })?;
        params = loaded;
        log::info!(
            "{}",
            pointimg::frontend::t("Preset chargé : {}")
                .replace("{}", &preset_path.display().to_string())
        );
    }
    if let Some(ref save_path) = args.save_preset {
        let txt = params
            .to_toml_string()
            .with_context(|| pointimg::frontend::t("Impossible de sérialiser les paramètres"))?;
        std::fs::write(save_path, &txt).with_context(|| {
            pointimg::frontend::t("Impossible d'écrire '{}'")
                .replace("{}", &save_path.display().to_string())
        })?;
        println!(
            "{}",
            pointimg::frontend::t("Preset sauvegardé : {}")
                .replace("{}", &save_path.display().to_string())
        );
        // If --save-preset is given without --input, exit after the write
        // (no pipeline to run).
        if args.input.is_none() {
            return Ok(());
        }
    }

    // Expansion of the inputs: single file, glob (`*`, `?`, `[`), or folder.
    let input_str = args.input.as_deref().ok_or_else(|| {
        anyhow::anyhow!(pointimg::frontend::t(
            "--input est requis sauf si --save-preset seul est fourni (utilise --help)"
        ))
    })?;
    let inputs = expand_inputs(input_str).with_context(|| {
        pointimg::frontend::t("Impossible d'énumérer les entrées '{}'").replace("{}", input_str)
    })?;
    if inputs.is_empty() {
        anyhow::bail!(
            "{}",
            pointimg::frontend::t("Aucun fichier d'entrée trouvé pour '{}'")
                .replace("{}", input_str)
        );
    }
    log::info!(
        "{}",
        pointimg::frontend::t("{} fichier(s) à traiter").replace("{}", &inputs.len().to_string())
    );

    let preview_size = args
        .preview
        .as_deref()
        .map(parse_preview_size)
        .transpose()
        .context(pointimg::frontend::t("--preview invalide"))?;
    if let Some((pw, ph)) = preview_size {
        log::info!("{}", {
            let mut s = pointimg::frontend::t("Mode preview : {:?}x{:?} max").to_string();
            s = s.replacen("{:?}", &format!("{:?}", pw), 1);
            s = s.replacen("{:?}", &format!("{:?}", ph), 1);
            s
        });
    }

    let total = inputs.len();
    let default_output = args.output == "output.png";
    let parallel = args.jobs != 1 && total > 1;
    if parallel {
        let workers = if args.jobs == 0 {
            num_cpus_hint()
        } else {
            args.jobs
        };
        log::info!(
            "{}",
            pointimg::frontend::t("batch parallèle : {} worker(s)")
                .replace("{}", &workers.to_string())
        );
    }
    let failures = if parallel {
        // In parallel, the progress by iteration would be interleaved on
        // stderr: it is cut off and each file is left to render its result line
        // atomically.
        // `--jobs 0` lets rayon choose the number of threads (all the CPUs).
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build()
            .context(pointimg::frontend::t(
                "impossible de créer le pool de threads du batch",
            ))?;
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
                            log::error!("{}", {
                                let mut s = pointimg::frontend::t("échec '{}': {}").to_string();
                                s = s.replacen("{}", &input_path.display().to_string(), 1);
                                s = s.replacen("{}", &e.to_string(), 1);
                                s
                            });
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
                log::error!("{}", {
                    let mut s = pointimg::frontend::t("échec '{}': {}").to_string();
                    s = s.replacen("{}", &input_path.display().to_string(), 1);
                    s = s.replacen("{}", &e.to_string(), 1);
                    s
                });
                failures += 1;
                // Continue the batch: the whole queue is not stopped for one
                // defective file.
            }
        }
        failures
    };

    if failures > 0 {
        anyhow::bail!(
            "{}",
            pointimg::frontend::t("{} fichier(s) n'ont pas pu être traité(s)")
                .replace("{}", &failures.to_string())
        );
    }
    Ok(())
}

/// Number of CPUs available (through rayon's global pool, which reads
/// `available_parallelism`).
fn num_cpus_hint() -> usize {
    rayon::current_num_threads()
}

/// Parses `"WxH"` (case-insensitive) into `(u32, u32)`.
/// E.g.: `"200x150"` → `(200, 150)`.
fn parse_preview_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s.split_once(['x', 'X', '×']).ok_or_else(|| {
        anyhow::anyhow!(
            "{}",
            pointimg::frontend::t("format attendu : WxH (ex. 200x150), got '{}'").replace("{}", s)
        )
    })?;
    let w: u32 = w.parse().map_err(|_| {
        anyhow::anyhow!(
            "{}",
            pointimg::frontend::t("largeur invalide dans '{}'").replace("{}", s)
        )
    })?;
    let h: u32 = h.parse().map_err(|_| {
        anyhow::anyhow!(
            "{}",
            pointimg::frontend::t("hauteur invalide dans '{}'").replace("{}", s)
        )
    })?;
    if w == 0 || h == 0 {
        anyhow::bail!(
            "{}",
            pointimg::frontend::t("preview WxH doit être > 0, got '{}'").replace("{}", s)
        );
    }
    Ok((w, h))
}

/// Subsamples `src` so that it fits inside `max_w × max_h` while preserving
/// the aspect ratio. Returns `src` unchanged if it is already smaller, or
/// `None`.
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

/// Enumerates the input files from:
/// - a glob (`*`, `?`, `[` present) via `glob::glob`;
/// - a folder (lists the image files);
/// - otherwise a single path (returned as is even if it does not exist —
///   the error will come at open).
fn expand_inputs(input: &str) -> Result<Vec<PathBuf>> {
    let has_glob_chars = input.contains('*') || input.contains('?') || input.contains('[');
    if has_glob_chars {
        let mut out = Vec::new();
        for entry in glob::glob(input)
            .with_context(|| pointimg::frontend::t("glob invalide '{}'").replace("{}", input))?
        {
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
        // The list comes from the library: it used to be copied here, one entry
        // (`gif`) ahead of the one the interfaces have. A folder therefore
        // listed files that the interfaces refused.
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&p)? {
            let entry = entry?;
            let path = entry.path();
            if pointimg::frontend::is_supported_image(&path) && path.is_file() {
                out.push(path);
            }
        }
        out.sort();
        return Ok(out);
    }
    Ok(vec![p])
}

/// Computes the output path from an `--output` pattern, an input path
/// and the index in the batch.
///
/// Recognized substitutions:
///   `{n}`    → file number (0-padded, width = number of digits of the total)
///   `{stem}` → file name without extension (`photo.jpg` → `photo`)
///   `{name}` → full file name (`photo.jpg`)
///
/// If `--output` is the default value (`output.png`), `{stem}.png` is used (batch)
/// or `output.png` (single file). If `--svg`, the extension is forced to `.svg`.
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
        // If the user put no per-file token but we have several files, an _{n}
        // is added before the extension to avoid overwriting.
        if total > 1 && !has_per_file_token {
            if let Some(dot) = s.rfind('.') {
                s.insert_str(dot, &format!("_{}", n_padded));
            } else {
                s.push_str(&format!("_{}", n_padded));
            }
        }
        if svg {
            // Force the extension to `.svg` (replaces the last one).
            if let Some(dot) = s.rfind('.') {
                s.truncate(dot);
            }
            s.push_str(".svg");
        }
        s
    };
    PathBuf::from(resolved)
}

/// Processes a single input file: opens, pipeline (RGB/RGBA/SVG), saves.
/// `iter_progress` controls the display of the progress by iteration (cut off
/// in parallel batch mode to avoid interleaved stdout/stderr).
fn process_one(
    input: &std::path::Path,
    output: &std::path::Path,
    params: &FilterParams,
    args: &Args,
    preview_size: Option<(u32, u32)>,
    iter_progress: bool,
) -> Result<()> {
    let dimensions = ImageReader::open(input)
        .with_context(|| {
            pointimg::frontend::t("Impossible de lire les dimensions de '{}'")
                .replace("{}", &input.display().to_string())
        })?
        .into_dimensions()
        .with_context(|| {
            pointimg::frontend::t("Dimensions invalides pour '{}'")
                .replace("{}", &input.display().to_string())
        })?;
    if dimensions.0 == 0 || dimensions.1 == 0 {
        anyhow::bail!(
            "{}",
            pointimg::frontend::t("Image vide ({}x{})")
                .replace("{}", &dimensions.0.to_string())
                .replacen("{}", &dimensions.1.to_string(), 1)
        );
    }
    let (src_orig, profile_converted, was_resized) =
        pointimg::color::decode_to_srgb(input, &args.input_profile)?;
    if profile_converted {
        log::info!(
            "{}",
            pointimg::frontend::t("profil colorimétrique converti vers sRGB")
        );
    }
    if was_resized {
        log::warn!(
            "{}",
            pointimg::frontend::t(
                "image '{}' réduite automatiquement à {:?} pour respecter les limites mémoire"
            )
            .replace("{}", &input.display().to_string())
            .replace("{:?}", &format!("{:?}", src_orig.dimensions()))
        );
    }
    // Preview subsampling when requested.
    let src = match preview_size {
        Some((pw, ph)) => downscale_for_preview(&src_orig, pw, ph),
        None => src_orig,
    };

    let rgb = filter::flatten_to_rgb(&src, params.bg_color);
    let output_profile = pointimg::color::output_profile_from_spec(&args.output_profile)?;
    if args.svg && output_profile.is_some() {
        anyhow::bail!(pointimg::frontend::t(
            "--output-profile ne s'applique pas à l'export SVG"
        ));
    }
    log::info!(
        "{}",
        pointimg::frontend::t("mémoire de travail estimée : {} Mo").replace(
            "{}",
            &(filter::estimate_memory_bytes_for(rgb.width(), rgb.height(), params) / (1024 * 1024))
                .to_string()
        )
    );
    let never_cancel = AtomicBool::new(false);

    let show_progress = iter_progress
        && matches!(params.algorithm, Algorithm::Voronoi | Algorithm::Kmeans)
        && params.iterations > 1;

    if args.svg {
        // `apply_with_iteration_counts`: the image is never looked at here, and
        // rendering it at every iteration was the dominant cost of the run.
        let (_, dots) =
            filter::apply_with_iteration_counts(&rgb, params, &never_cancel, |iter, total| {
                if show_progress {
                    eprint!("\r  Iteration {iter}/{total}");
                }
            })
            .with_context(|| pointimg::frontend::t("Erreur lors du calcul du filtre"))?;
        if show_progress {
            eprintln!();
        }
        let (w, h) = rgb.dimensions();
        let svg = filter::render_svg_from_dots(w, h, &dots, params)
            .with_context(|| pointimg::frontend::t("Erreur lors du rendu SVG"))?;
        frontend::atomic_text_write(output, &svg).with_context(|| {
            pointimg::frontend::t("Impossible d'ecrire '{}'")
                .replace("{}", &output.display().to_string())
        })?;
        println!(
            "{}",
            pointimg::frontend::t("SVG sauvegarde : {}")
                .replace("{}", &output.display().to_string())
        );
    } else if params.transparent {
        let (dst, _dots) = filter::apply_rgba(&rgb, params)
            .with_context(|| pointimg::frontend::t("Erreur lors du calcul du filtre (RGBA)"))?;
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
            .with_context(|| {
                pointimg::frontend::t("Impossible de sauvegarder '{}'")
                    .replace("{}", &output.display().to_string())
            })?;
        } else {
            frontend::atomic_image_save(output, |tmp| {
                save_rgba_for_output(&dst, params.bg_color, tmp)
            })
            .with_context(|| {
                pointimg::frontend::t("Impossible de sauvegarder '{}'")
                    .replace("{}", &output.display().to_string())
            })?;
        }
        println!(
            "{}",
            pointimg::frontend::t("Sauvegarde (RGBA) : {}")
                .replace("{}", &output.display().to_string())
        );
    } else {
        let (dst, _dots) =
            filter::apply_with_iteration_counts(&rgb, params, &never_cancel, |iter, total| {
                if show_progress {
                    eprint!("\r  Iteration {iter}/{total}");
                }
            })
            .with_context(|| pointimg::frontend::t("Erreur lors du calcul du filtre"))?;
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
        .with_context(|| {
            pointimg::frontend::t("Impossible de sauvegarder '{}'")
                .replace("{}", &output.display().to_string())
        })?;
        println!(
            "{}",
            pointimg::frontend::t("Sauvegarde : {}").replace("{}", &output.display().to_string())
        );
    }
    Ok(())
}

fn save_rgba_for_output(
    image: &RgbaImage,
    bg: [u8; 3],
    path: &std::path::Path,
) -> image::ImageResult<()> {
    // PNG/WebP/TIFF preserve alpha; otherwise composite onto the background.
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

    // ── Parsing of the background colors ─────────────────────────────────────

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

    // ── Generation of the output paths ───────────────────────────────────────

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
        // No per-file token + several files → `_n` suffix before the extension.
        assert_eq!(
            resolve("out/all.png", 3, 10),
            PathBuf::from("out/all_03.png")
        );
        // Without an extension, the suffix is added at the end.
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
        // `--svg` forces the extension even on an explicit pattern.
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
