//! Logique du filtre pointilliste.
//!
//! Quatre algorithmes de partitionnement :
//!
//! - `Grid`     : grille uniforme
//! - `Kmeans`   : k-means spatial (x, y, r, g, b)
//! - `Voronoi`  : stippling de Lloyd pondéré par la density map
//! - `Quadtree` : subdivision récursive par contraste
//!
//! La `density_map` contrôle la distribution des points :
//!   densité élevée (zone détaillée) → plus de points, petits
//!   densité faible  (zone uniforme)  → moins de points, gros
//!
//! `variance_sensitivity` règle à quel point cette redistribution est forte.

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use rayon::prelude::*;
use std::f32::consts::PI;

pub use anyhow::{Result, anyhow};

// ─── Types publics ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Algorithm {
    Grid,
    Kmeans,
    Voronoi,
    Quadtree,
}

/// Forme des dots dessinés.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum DotShape {
    /// Disque plein (comportement historique)
    #[default]
    Circle,
    /// Carré plein, côté = 2×rayon
    Square,
    /// Ellipse, `aspect` = ratio width/height ∈ (0, 10], `angle_deg` = rotation en degrés
    Ellipse { aspect: f32, angle_deg: f32 },
    /// Polygone régulier à `sides` côtés (3 = triangle, 4 = losange, 6 = hexagone, …)
    RegularPolygon { sides: u8 },
}

#[derive(Clone, Debug)]
pub struct FilterParams {
    pub algorithm: Algorithm,
    /// Nombre de points cible (Kmeans / Voronoi / Quadtree)
    pub num_points: usize,
    /// Nombre de colonnes (Grid uniquement)
    pub cols: u32,
    /// Rayon minimum : fraction de min(largeur, hauteur) — ex. 0.002 = 0.2% de l'image
    pub min_radius_ratio: f32,
    /// Rayon maximum : fraction de min(largeur, hauteur) — ex. 0.08 = 8% de l'image
    pub max_radius_ratio: f32,
    /// Couleur de fond RGB — source de vérité unique (remplace l'ancien bg_white)
    pub bg_color: [u8; 3],
    /// Nombre d'itérations pour Voronoi / Kmeans
    pub iterations: usize,
    /// Sensibilité à la variance locale (0 = ignorée, 1 = forte redistribution)
    pub variance_sensitivity: f32,
    /// Multiplicateur maximum de la taille dans les zones uniformes (1.0 = pas de boost)
    pub max_boost: f32,
    /// Graine du RNG pour reproductibilité (None = aléatoire basé sur l'horloge)
    pub rng_seed: Option<u64>,
    /// Nombre de couleurs pour la quantification (None = pas de quantification)
    pub palette_size: Option<usize>,
    /// Forme des dots
    pub dot_shape: DotShape,
}

impl FilterParams {
    /// Le fond est-il clair ? Dérivé de `bg_color`.
    pub fn is_bg_light(&self) -> bool {
        let lum = 0.2126 * self.bg_color[0] as f32
            + 0.7152 * self.bg_color[1] as f32
            + 0.0722 * self.bg_color[2] as f32;
        lum > 127.5
    }
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Voronoi,
            num_points: 800,
            cols: 80,
            min_radius_ratio: 0.003,
            max_radius_ratio: 0.06,
            bg_color: [255, 255, 255],
            iterations: 10,
            variance_sensitivity: 0.7,
            max_boost: 2.5,
            rng_seed: None,
            palette_size: None,
            dot_shape: DotShape::Circle,
        }
    }
}

/// Un point résultant du filtre
#[derive(Clone, Debug)]
pub struct Dot {
    pub x: f32,
    pub y: f32,
    pub color: [u8; 3],
    pub radius: f32,
}

// ─── Validation commune ───────────────────────────────────────────────────────

fn validate_params(w: u32, h: u32, params: &FilterParams) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(anyhow!("Image vide ({}x{})", w, h));
    }
    if params.min_radius_ratio <= 0.0 {
        return Err(anyhow!("min_radius_ratio doit etre > 0"));
    }
    if params.max_radius_ratio < params.min_radius_ratio {
        return Err(anyhow!(
            "max_radius_ratio ({}) doit etre >= min_radius_ratio ({})",
            params.max_radius_ratio,
            params.min_radius_ratio
        ));
    }
    // Q5: bound max_radius_ratio from above (> 1.0 makes no sense)
    if params.max_radius_ratio > 1.0 {
        return Err(anyhow!(
            "max_radius_ratio ({}) doit etre <= 1.0",
            params.max_radius_ratio
        ));
    }
    if params.num_points == 0 {
        return Err(anyhow!("num_points doit etre > 0"));
    }
    if let DotShape::RegularPolygon { sides } = params.dot_shape
        && !(3..=12).contains(&sides)
    {
        return Err(anyhow!(
            "polygon_sides doit etre entre 3 et 12, got {}",
            sides
        ));
    }
    if let DotShape::Ellipse { aspect, .. } = params.dot_shape
        && (aspect <= 0.0 || aspect > 10.0)
    {
        return Err(anyhow!(
            "ellipse_aspect doit etre dans (0, 10], got {}",
            aspect
        ));
    }
    const MAX_DIMENSION: u32 = 65535;
    if w > MAX_DIMENSION || h > MAX_DIMENSION {
        return Err(anyhow!(
            "Image trop grande ({}x{}), maximum autorise: {}x{}",
            w,
            h,
            MAX_DIMENSION,
            MAX_DIMENSION
        ));
    }
    const MAX_PIXELS: u64 = 256 * 1024 * 1024; // 256M pixels = ~1GB RAM (RGBA)
    let pixels = (w as u64) * (h as u64);
    if pixels > MAX_PIXELS {
        return Err(anyhow!(
            "Image trop grande ({} pixels), maximum: {} pixels",
            pixels,
            MAX_PIXELS
        ));
    }
    Ok(())
}

// ─── Entrée principale ────────────────────────────────────────────────────────

/// Applique le filtre sur une `DynamicImage` (supporte RGBA, niveaux de gris, etc.)
/// Le fond transparent est composé sur `params.bg_color`.
pub fn apply_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<RgbImage> {
    let rgb = flatten_to_rgb(src, params.bg_color);
    apply(&rgb, params)
}

/// Applique le filtre itération par itération (Voronoï/K-means).
/// Appelle `on_progress(iter, total, image_intermediaire)` après chaque itération Lloyd.
/// Pratique pour la preview progressive dans la GUI.
/// Pour les autres algorithmes, appelle `on_progress` une seule fois à la fin.
///
/// Retourne `(image_finale, dots)` — les dots correspondent exactement à l'image
/// rendue, ce qui évite le double-calcul qu'on avait avant.
///
/// Le token `cancel` est vérifié entre chaque itération. Si `cancel` est vrai,
/// la fonction retourne `Err(anyhow!("cancelled"))`.
pub fn apply_with_progress<F>(
    src: &RgbImage,
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    mut on_progress: F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, &RgbImage),
{
    use std::sync::atomic::Ordering;
    let (w, h) = src.dimensions();
    validate_params(w, h, params)?;

    let density = compute_density_map(src, params.variance_sensitivity);

    match params.algorithm {
        Algorithm::Voronoi => {
            dots_voronoi_progressive(src, &density, params, cancel, &mut on_progress)
        }
        Algorithm::Kmeans => {
            dots_kmeans_progressive(src, &density, params, cancel, &mut on_progress)
        }
        _ => {
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!("cancelled"));
            }
            let dots = match params.algorithm {
                Algorithm::Grid => dots_grid(src, &density, params),
                Algorithm::Quadtree => dots_quadtree(src, params),
                _ => unreachable!(),
            };
            let result = render(src, &dots, params);
            on_progress(1, 1, &result);
            Ok((result, dots))
        }
    }
}

/// Retourne la density map normalisée comme image en niveaux de gris.
/// Utile pour la prévisualisation dans la GUI.
pub fn compute_density_image(src: &RgbImage, sensitivity: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let density = compute_density_map(src, sensitivity);
    GrayImage::from_fn(w, h, |x, y| {
        let v = density[(y * w + x) as usize];
        Luma([(v * 255.0) as u8])
    })
}

fn dots_voronoi_progressive<F>(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    on_progress: &mut F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, &RgbImage),
{
    use std::sync::atomic::Ordering;
    let (width, height) = src.dimensions();
    let k = params.num_points;
    let iters = params.iterations;

    let mut seeds: Vec<(f32, f32)> =
        importance_sample(density, width, height, k, make_rng_seed(params))
            .into_iter()
            .map(|(x, y)| (x as f32 + 0.5, y as f32 + 0.5))
            .collect();

    let power = 1.0 + params.variance_sensitivity * 3.0;
    let lloyd_weights: Vec<f64> = density
        .iter()
        .map(|&d| (d as f64).powf(power as f64))
        .collect();

    // Pré-allouer les accumulateurs (réutilisés à chaque itération)
    for iter in 0..iters {
        // ── Vérifier le cancel AVANT la lourde par_iter ─────────────────────
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }

        let grid = SeedGrid::new(&seeds, width, height);

        let (sum_x, sum_y, sum_w) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut sx, mut sy, mut sw), py| {
                    for px in 0..width {
                        let fx = px as f32 + 0.5;
                        let fy = py as f32 + 0.5;
                        let best = grid.nearest(fx, fy, &seeds);
                        let w = lloyd_weights[(py * width + px) as usize];
                        sx[best] += fx as f64 * w;
                        sy[best] += fy as f64 * w;
                        sw[best] += w;
                    }
                    (sx, sy, sw)
                },
            )
            .reduce(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut ax, mut ay, mut aw), (bx, by, bw)| {
                    for i in 0..k {
                        ax[i] += bx[i];
                        ay[i] += by[i];
                        aw[i] += bw[i];
                    }
                    (ax, ay, aw)
                },
            );

        for i in 0..k {
            if sum_w[i] > 0.0 {
                seeds[i] = ((sum_x[i] / sum_w[i]) as f32, (sum_y[i] / sum_w[i]) as f32);
            }
        }

        // Preview après chaque itération
        let dots = build_dots_from_seeds(src, density, &seeds, params);
        let preview = render(src, &dots, params);
        on_progress(iter + 1, iters, &preview);

        // P2: keep last iteration's result to avoid recomputing
        if iter + 1 == iters {
            return Ok((preview, dots));
        }
    }

    // Fallback (iters == 0)
    let dots = build_dots_from_seeds(src, density, &seeds, params);
    let img = render(src, &dots, params);
    Ok((img, dots))
}

fn dots_kmeans_progressive<F>(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
    on_progress: &mut F,
) -> Result<(RgbImage, Vec<Dot>)>
where
    F: FnMut(usize, usize, &RgbImage),
{
    use std::sync::atomic::Ordering;
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
    let k = params.num_points;
    let iters = params.iterations;

    let seeds_pos = importance_sample(density, width, height, k, make_rng_seed(params));

    let mut centers: Vec<[f32; 5]> = seeds_pos
        .iter()
        .map(|&(x, y)| {
            let p = src.get_pixel(x.min(width - 1), y.min(height - 1));
            [x as f32, y as f32, p[0] as f32, p[1] as f32, p[2] as f32]
        })
        .collect();

    let w_f = width as f32;
    let h_f = height as f32;

    for iter in 0..iters {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }

        // Pré-normaliser les centres une seule fois par itération (perf 9)
        let centers_norm: Vec<[f32; 5]> = centers
            .iter()
            .map(|c| {
                [
                    c[0] / w_f,
                    c[1] / h_f,
                    c[2] / 255.0,
                    c[3] / 255.0,
                    c[4] / 255.0,
                ]
            })
            .collect();

        let (sums, counts) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut sums, mut counts), py| {
                    for px in 0..width {
                        let p = src.get_pixel(px, py);
                        let feat = [
                            px as f32 / w_f,
                            py as f32 / h_f,
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                        ];
                        let best = centers_norm
                            .iter()
                            .enumerate()
                            .map(|(i, cn)| {
                                let d: f32 = feat
                                    .iter()
                                    .zip(cn.iter())
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum();
                                (i, d)
                            })
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        sums[best][0] += px as f64;
                        sums[best][1] += py as f64;
                        sums[best][2] += p[0] as f64;
                        sums[best][3] += p[1] as f64;
                        sums[best][4] += p[2] as f64;
                        counts[best] += 1;
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut as_, mut ac), (bs, bc)| {
                    for i in 0..centers.len() {
                        for j in 0..5 {
                            as_[i][j] += bs[i][j];
                        }
                        ac[i] += bc[i];
                    }
                    (as_, ac)
                },
            );

        for (i, c) in centers.iter_mut().enumerate() {
            let n = counts[i] as f64;
            if n > 0.0 {
                c[0] = (sums[i][0] / n) as f32;
                c[1] = (sums[i][1] / n) as f32;
                c[2] = (sums[i][2] / n) as f32;
                c[3] = (sums[i][3] / n) as f32;
                c[4] = (sums[i][4] / n) as f32;
            }
        }

        // Preview après chaque itération K-means (UX 12)
        let dots = dots_from_kmeans_centers(&centers, density, width, height, img_min, params);
        let preview = render(src, &dots, params);
        on_progress(iter + 1, iters, &preview);

        // P2: keep last iteration's result to avoid recomputing
        if iter + 1 == iters {
            return Ok((preview, dots));
        }
    }

    // Fallback (iters == 0)
    let dots = dots_from_kmeans_centers(&centers, density, width, height, img_min, params);
    let img = render(src, &dots, params);
    Ok((img, dots))
}

/// Construit les Dots depuis les centres K-means.
/// Extrait pour dédupliquer le code entre preview et résultat final.
fn dots_from_kmeans_centers(
    centers: &[[f32; 5]],
    density: &[f32],
    width: u32,
    height: u32,
    img_min: f32,
    params: &FilterParams,
) -> Vec<Dot> {
    centers
        .iter()
        .filter(|c| c[0] >= 0.0 && c[0] < width as f32 && c[1] >= 0.0 && c[1] < height as f32)
        .map(|c| {
            let avg = [c[2] as u8, c[3] as u8, c[4] as u8];
            let lum = luminance(avg[0], avg[1], avg[2]);
            let cx = (c[0] as u32).min(width - 1);
            let cy = (c[1] as u32).min(height - 1);
            let d = density[(cy * width + cx) as usize];
            Dot {
                x: c[0],
                y: c[1],
                color: avg,
                radius: radius_for_dot(lum, d, img_min, params),
            }
        })
        .collect()
}

/// Aplatit n'importe quelle `DynamicImage` vers RGB8 en composant l'alpha sur `bg`.
/// Fix #4 : convertit d'abord en RGBA8 pour gérer tous les formats alpha
/// (RGBA16, LumaA8, etc.), pas seulement ImageRgba8.
pub fn flatten_to_rgb(img: &DynamicImage, bg: [u8; 3]) -> RgbImage {
    // Q7: simplified — just check if the format has alpha and composite if so
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    // Convertir en RGBA8 pour compositeur uniformément
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    RgbImage::from_fn(w, h, |x, y| {
        let p = rgba.get_pixel(x, y);
        let a = p[3] as f32 / 255.0;
        Rgb([
            (p[0] as f32 * a + bg[0] as f32 * (1.0 - a)) as u8,
            (p[1] as f32 * a + bg[1] as f32 * (1.0 - a)) as u8,
            (p[2] as f32 * a + bg[2] as f32 * (1.0 - a)) as u8,
        ])
    })
}

pub fn apply(src: &RgbImage, params: &FilterParams) -> Result<RgbImage> {
    let (w, h) = src.dimensions();
    validate_params(w, h, params)?;

    match params.algorithm {
        Algorithm::Grid => {
            // P4: only compute density_map for algorithms that need it here
            let density = compute_density_map(src, params.variance_sensitivity);
            let dots = dots_grid(src, &density, params);
            Ok(render(src, &dots, params))
        }
        Algorithm::Quadtree => {
            let dots = dots_quadtree(src, params);
            Ok(render(src, &dots, params))
        }
        Algorithm::Voronoi | Algorithm::Kmeans => {
            // Delegate to apply_with_progress which computes its own density_map
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            let (img, _) = apply_with_progress(src, params, &never_cancel, |_, _, _| {})?;
            Ok(img)
        }
    }
}

// ─── Density map ─────────────────────────────────────────────────────────────
//
// Variance locale dans un voisinage 9×9, normalisée 0→1.
// sensitivity=0 → tout à 1 (uniforme), sensitivity=1 → variance pure.
//
// Bug 3 corrigé : calcul en f64 pour éviter la troncature entière.

fn compute_density_map(src: &RgbImage, sensitivity: f32) -> Vec<f32> {
    let (width, height) = src.dimensions();
    let radius: usize = 4;
    let w = width as usize;
    let h = height as usize;
    let n_pixels = w * h;

    // ── Summed-area tables (SAT) — P1 ────────────────────────────────────
    // 6 tables: sum and sum-of-squares for R, G, B.
    // SAT uses (w+1)x(h+1) layout with zero padding row/col for branchless queries.
    // Replaces the O(81) per-pixel loop with O(1) per-pixel after O(n) precomputation.
    let sat_len = (w + 1) * (h + 1);
    let mut sat_r = vec![0f64; sat_len];
    let mut sat_g = vec![0f64; sat_len];
    let mut sat_b = vec![0f64; sat_len];
    let mut sat_r2 = vec![0f64; sat_len];
    let mut sat_g2 = vec![0f64; sat_len];
    let mut sat_b2 = vec![0f64; sat_len];

    let stride = w + 1;
    for y in 0..h {
        for x in 0..w {
            let p = src.get_pixel(x as u32, y as u32);
            let (pr, pg, pb) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let idx = (y + 1) * stride + (x + 1);
            let left = idx - 1;
            let up = idx - stride;
            let diag = up - 1;
            sat_r[idx] = pr + sat_r[left] + sat_r[up] - sat_r[diag];
            sat_g[idx] = pg + sat_g[left] + sat_g[up] - sat_g[diag];
            sat_b[idx] = pb + sat_b[left] + sat_b[up] - sat_b[diag];
            sat_r2[idx] = pr * pr + sat_r2[left] + sat_r2[up] - sat_r2[diag];
            sat_g2[idx] = pg * pg + sat_g2[left] + sat_g2[up] - sat_g2[diag];
            sat_b2[idx] = pb * pb + sat_b2[left] + sat_b2[up] - sat_b2[diag];
        }
    }

    // Query helper: sum over rect [x0,y0]..[x1,y1] inclusive (0-indexed pixel coords)
    let sat_query = |sat: &[f64], x0: usize, y0: usize, x1: usize, y1: usize| -> f64 {
        sat[(y1 + 1) * stride + (x1 + 1)]
            - sat[y0 * stride + (x1 + 1)]
            - sat[(y1 + 1) * stride + x0]
            + sat[y0 * stride + x0]
    };

    // ── Compute variance per pixel using SAT (O(1) per pixel) ───────────
    let raw: Vec<f32> = (0..n_pixels)
        .into_par_iter()
        .map(|idx| {
            let px = idx % w;
            let py = idx / w;
            let x0 = px.saturating_sub(radius);
            let y0 = py.saturating_sub(radius);
            let x1 = (px + radius).min(w - 1);
            let y1 = (py + radius).min(h - 1);
            let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
            if n == 0.0 {
                return 0.0;
            }
            let sr = sat_query(&sat_r, x0, y0, x1, y1);
            let sg = sat_query(&sat_g, x0, y0, x1, y1);
            let sb = sat_query(&sat_b, x0, y0, x1, y1);
            let sr2 = sat_query(&sat_r2, x0, y0, x1, y1);
            let sg2 = sat_query(&sat_g2, x0, y0, x1, y1);
            let sb2 = sat_query(&sat_b2, x0, y0, x1, y1);
            // Var = E[X^2] - E[X]^2  — en f64 pour eviter troncature (bug 3 corrige)
            let var_r = (sr2 / n - (sr / n).powi(2)).max(0.0);
            let var_g = (sg2 / n - (sg / n).powi(2)).max(0.0);
            let var_b = (sb2 / n - (sb / n).powi(2)).max(0.0);
            ((var_r + var_g + var_b) / 3.0) as f32
        })
        .collect();

    // C3: use 1e-6 instead of 1.0 to preserve contrast on near-solid images
    let max_var = raw.iter().cloned().fold(0.0f32, f32::max).max(1e-6);

    raw.iter()
        .map(|&v| {
            let norm = (v / max_var).sqrt();
            // sensitivity=0 -> 1.0 partout ; sensitivity=1 -> norm pur
            1.0 - sensitivity * (1.0 - norm)
        })
        .collect()
}

// ─── Rendu commun ─────────────────────────────────────────────────────────────

fn render(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbImage {
    let (w, h) = src.dimensions();
    // Bug 6 corrigé : utiliser bg_color comme source de vérité
    let bg_rgb = params.bg_color;
    let mut dst = RgbImage::from_pixel(w, h, Rgb(bg_rgb));

    // Quantification optionnelle de la palette (perf 10 : seulement sur le résultat final)
    // Q1: avoid cloning dots when no quantization is needed
    let quantized: Vec<Dot>;
    let dots_to_draw: &[Dot] = if let Some(n_colors) = params.palette_size {
        quantized = quantize_dots(dots, n_colors.max(2));
        &quantized
    } else {
        dots
    };

    // Trier : les plus gros points d'abord (arrière-plan), les petits en dernier (avant-plan)
    let mut sorted: Vec<&Dot> = dots_to_draw.iter().collect();
    sorted.sort_unstable_by(|a, b| b.radius.total_cmp(&a.radius));

    for dot in sorted {
        let r = dot.radius.round() as i32;
        if r <= 0 {
            continue;
        }
        let color = Rgb(dot.color);
        let cx = dot.x as i32;
        let cy = dot.y as i32;
        draw_dot(&mut dst, cx, cy, r, color, params.dot_shape);
    }
    dst
}

/// Dessine un dot selon la forme choisie.
fn draw_dot(dst: &mut RgbImage, cx: i32, cy: i32, r: i32, color: Rgb<u8>, shape: DotShape) {
    let (iw, ih) = dst.dimensions();
    match shape {
        DotShape::Circle => {
            draw_filled_circle(dst, cx, cy, r, color);
        }
        DotShape::Square => {
            let x0 = (cx - r).max(0) as u32;
            let y0 = (cy - r).max(0) as u32;
            let x1 = (cx + r).min(iw as i32 - 1).max(0) as u32;
            let y1 = (cy + r).min(ih as i32 - 1).max(0) as u32;
            for py in y0..=y1 {
                for px in x0..=x1 {
                    dst.put_pixel(px, py, color);
                }
            }
        }
        DotShape::Ellipse { aspect, angle_deg } => {
            // Demi-axes
            let a = r as f32; // demi-axe x (avant rotation)
            let b = (r as f32 / aspect.max(0.01)).max(1.0); // demi-axe y
            let ang = angle_deg * PI / 180.0;
            let cos_a = ang.cos();
            let sin_a = ang.sin();
            let bbox = (a.max(b).ceil() as i32) + 1;
            for dy in -bbox..=bbox {
                for dx in -bbox..=bbox {
                    // Rotation inverse du point
                    let lx = dx as f32 * cos_a + dy as f32 * sin_a;
                    let ly = -dx as f32 * sin_a + dy as f32 * cos_a;
                    if (lx / a).powi(2) + (ly / b).powi(2) <= 1.0 {
                        let px = cx + dx;
                        let py = cy + dy;
                        if px >= 0 && py >= 0 && px < iw as i32 && py < ih as i32 {
                            dst.put_pixel(px as u32, py as u32, color);
                        }
                    }
                }
            }
        }
        DotShape::RegularPolygon { sides } => {
            let n = sides.max(3) as usize;
            let rf = r as f32;
            // Bounding box
            let bbox = r + 1;
            for dy in -bbox..=bbox {
                for dx in -bbox..=bbox {
                    if point_in_regular_polygon(dx as f32, dy as f32, rf, n) {
                        let px = cx + dx;
                        let py = cy + dy;
                        if px >= 0 && py >= 0 && px < iw as i32 && py < ih as i32 {
                            dst.put_pixel(px as u32, py as u32, color);
                        }
                    }
                }
            }
        }
    }
}

/// Test si (px, py) est dans un polygone régulier à `n` côtés de rayon `r`.
fn point_in_regular_polygon(px: f32, py: f32, r: f32, n: usize) -> bool {
    // Distance au centre
    let d = (px * px + py * py).sqrt();
    if d > r {
        return false;
    }
    if d == 0.0 {
        return true;
    }
    // Angle du point
    let angle = py.atan2(px);
    // Secteur angulaire du polygone contenant ce point
    let sector_angle = 2.0 * PI / n as f32;
    // Angle à l'intérieur du secteur [0, sector_angle)
    let sector = ((angle % sector_angle) + sector_angle) % sector_angle;
    // Distance au bord du polygone dans cette direction (projection sur l'apothème)
    let apothem_angle = sector - sector_angle / 2.0; // angle par rapport au milieu du côté
    let apothem = r * (PI / n as f32).cos(); // distance centre → milieu d'un côté
    let dist_to_edge = apothem / apothem_angle.cos().abs();
    d <= dist_to_edge
}

/// Remplace `draw_filled_circle_mut` de imageproc — même comportement, mais
/// sans dépendance sur l'API externe pour les formes custom.
fn draw_filled_circle(dst: &mut RgbImage, cx: i32, cy: i32, r: i32, color: Rgb<u8>) {
    let (iw, ih) = dst.dimensions();
    let r2 = (r * r) as i64;
    let x0 = (cx - r).max(0);
    let x1 = (cx + r).min(iw as i32 - 1);
    let y0 = (cy - r).max(0);
    let y1 = (cy + r).min(ih as i32 - 1);
    for py in y0..=y1 {
        let dy = (py - cy) as i64;
        for px in x0..=x1 {
            let dx = (px - cx) as i64;
            if dx * dx + dy * dy <= r2 {
                dst.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

/// Quantifie les couleurs des dots en `n` couleurs via k-means simple sur RGB.
fn quantize_dots(dots: &[Dot], n: usize) -> Vec<Dot> {
    if dots.is_empty() || n >= dots.len() {
        return dots.to_vec();
    }

    // Initialisation : sous-échantillonnage uniforme des dots comme centres
    let step = dots.len() / n;
    let mut centers: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let c = dots[i * step].color;
            [c[0] as f32, c[1] as f32, c[2] as f32]
        })
        .collect();

    // Q3: k-means with convergence check (max 10 iterations, stop early if converged)
    for _ in 0..10 {
        let mut sums = vec![[0f64; 3]; n];
        let mut counts = vec![0u64; n];

        for dot in dots {
            let best = centers
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let dr = c[0] - dot.color[0] as f32;
                    let dg = c[1] - dot.color[1] as f32;
                    let db = c[2] - dot.color[2] as f32;
                    (i, dr * dr + dg * dg + db * db)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            sums[best][0] += dot.color[0] as f64;
            sums[best][1] += dot.color[1] as f64;
            sums[best][2] += dot.color[2] as f64;
            counts[best] += 1;
        }

        let mut converged = true;
        for i in 0..n {
            let cnt = counts[i] as f64;
            if cnt > 0.0 {
                let new_center = [
                    (sums[i][0] / cnt) as f32,
                    (sums[i][1] / cnt) as f32,
                    (sums[i][2] / cnt) as f32,
                ];
                let shift = (new_center[0] - centers[i][0]).powi(2)
                    + (new_center[1] - centers[i][1]).powi(2)
                    + (new_center[2] - centers[i][2]).powi(2);
                if shift > 0.5 {
                    converged = false;
                }
                centers[i] = new_center;
            }
        }
        if converged {
            break;
        }
    }

    // Remapper chaque dot à la couleur de son centre le plus proche
    dots.iter()
        .map(|dot| {
            let best_color = centers
                .iter()
                .min_by(|a, b| {
                    let da = (a[0] - dot.color[0] as f32).powi(2)
                        + (a[1] - dot.color[1] as f32).powi(2)
                        + (a[2] - dot.color[2] as f32).powi(2);
                    let db = (b[0] - dot.color[0] as f32).powi(2)
                        + (b[1] - dot.color[1] as f32).powi(2)
                        + (b[2] - dot.color[2] as f32).powi(2);
                    da.total_cmp(&db)
                })
                .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
                .unwrap_or(dot.color);
            Dot {
                color: best_color,
                ..*dot
            }
        })
        .collect()
}

// ─── Calcul du rayon ─────────────────────────────────────────────────────────

fn radius_for_dot(
    lum: f32,
    local_density: f32, // 0=uniforme 1=détaillé
    img_min_side: f32,
    params: &FilterParams,
) -> f32 {
    let r_min = params.min_radius_ratio * img_min_side;
    let r_max = params.max_radius_ratio * img_min_side;

    // Halftone : sombre → grand (lum=0 → r_max, lum=1 → r_min)
    let r_lum = r_max - (r_max - r_min) * lum;

    // Uniformité : zone plate → on pousse encore plus grand
    let uniformity = 1.0 - local_density;
    let boost = 1.0 + uniformity * params.variance_sensitivity * (params.max_boost - 1.0);
    (r_lum * boost).min(params.max_boost * r_max)
}

/// Densité moyenne d'une zone rectangulaire de la density map.
fn zone_density(density: &[f32], x0: u32, y0: u32, w: u32, h: u32, img_w: u32, img_h: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for py in y0..(y0 + h).min(img_h) {
        for px in x0..(x0 + w).min(img_w) {
            sum += density[(py * img_w + px) as usize];
            n += 1;
        }
    }
    if n > 0 { sum / n as f32 } else { 1.0 }
}

// ─── Algorithme 1 : Grille ────────────────────────────────────────────────────

fn dots_grid(src: &RgbImage, density: &[f32], params: &FilterParams) -> Vec<Dot> {
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
    let cell = (width / params.cols).max(1);
    let cols = width / cell;
    let rows = height / cell;

    (0..rows)
        .flat_map(|row| (0..cols).map(move |col| (row, col)))
        .filter_map(|(row, col)| {
            let x0 = col * cell;
            let y0 = row * cell;
            let (sr, sg, sb, n) = pixel_sum(src, x0, y0, cell, cell);
            if n == 0 {
                return None;
            }
            let avg = [(sr / n) as u8, (sg / n) as u8, (sb / n) as u8];
            let lum = luminance(avg[0], avg[1], avg[2]);
            let d = zone_density(density, x0, y0, cell, cell, width, height);
            let r_nn_cap = cell as f32 * 0.5 * 0.8;
            Some(Dot {
                x: x0 as f32 + cell as f32 / 2.0,
                y: y0 as f32 + cell as f32 / 2.0,
                color: avg,
                radius: radius_for_dot(lum, d, img_min, params).min(r_nn_cap),
            })
        })
        .collect()
}

// ─── Algorithme 2 : K-means spatial ──────────────────────────────────────────
// (la version canonique est dots_kmeans_progressive, utilisée via apply_with_progress)

// ─── Grille spatiale pour nearest-seed ───────────────────────────────────────

struct SeedGrid {
    cells: Vec<Vec<usize>>,
    cols: usize,
    rows: usize,
    cell_w: f32,
    cell_h: f32,
}

impl SeedGrid {
    fn new(seeds: &[(f32, f32)], img_w: u32, img_h: u32) -> Self {
        let k = seeds.len().max(1);
        let area = (img_w * img_h) as f32;
        let cell_area = area / k as f32 * 4.0;
        let cell_size = cell_area.sqrt().max(1.0);

        let cols = ((img_w as f32 / cell_size).ceil() as usize).max(1);
        let rows = ((img_h as f32 / cell_size).ceil() as usize).max(1);
        let cell_w = img_w as f32 / cols as f32;
        let cell_h = img_h as f32 / rows as f32;

        let mut cells = vec![Vec::new(); cols * rows];
        for (i, &(sx, sy)) in seeds.iter().enumerate() {
            let cx = ((sx / cell_w) as usize).min(cols - 1);
            let cy = ((sy / cell_h) as usize).min(rows - 1);
            cells[cy * cols + cx].push(i);
        }
        SeedGrid {
            cells,
            cols,
            rows,
            cell_w,
            cell_h,
        }
    }

    /// Renvoie l'indice du seed le plus proche de (fx, fy).
    ///
    /// Bug 5 corrigé : la comparaison d'arrêt utilise min_possible² (distance au carré)
    /// comparé à best_dist qui est aussi une distance au carré.
    fn nearest(&self, fx: f32, fy: f32, seeds: &[(f32, f32)]) -> usize {
        let cx = ((fx / self.cell_w) as i64).clamp(0, self.cols as i64 - 1);
        let cy = ((fy / self.cell_h) as i64).clamp(0, self.rows as i64 - 1);

        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX; // distance AU CARRE

        let mut radius = 0i64;
        loop {
            // Q4: proper minimum distance from query point to the ring at `radius`.
            // The closest point in ring `radius` is at least `(radius-1)` cells away
            // from the query cell center, but we need the distance from the query point
            // to the nearest edge of cells in the ring.
            let min_possible_sq = if radius <= 1 {
                0.0f32
            } else {
                // Distance from query point to the nearest cell border at ring `radius`
                let dx = (radius as f32 - 1.0) * self.cell_w;
                let dy = (radius as f32 - 1.0) * self.cell_h;
                // The minimum distance is the smaller axis distance squared
                // (a point in a corner ring cell could be close on one axis)
                dx.min(dy).powi(2)
            };
            if min_possible_sq > best_dist && radius > 0 {
                break;
            }
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if radius > 0 && dx.abs() < radius && dy.abs() < radius {
                        continue;
                    }
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if nx < 0 || ny < 0 || nx >= self.cols as i64 || ny >= self.rows as i64 {
                        continue;
                    }
                    for &si in &self.cells[ny as usize * self.cols + nx as usize] {
                        let (sx, sy) = seeds[si];
                        let d = (fx - sx).powi(2) + (fy - sy).powi(2);
                        if d < best_dist {
                            best_dist = d;
                            best_idx = si;
                        }
                    }
                }
            }
            radius += 1;
            if radius > (self.cols.max(self.rows) as i64) {
                break;
            }
        }
        best_idx
    }
}

// ─── Algorithme 3 : Voronoï / Lloyd ──────────────────────────────────────────
// (la version canonique est dots_voronoi_progressive, utilisée via apply_with_progress)

// ─── Algorithme 4 : Quadtree adaptatif ───────────────────────────────────────

fn dots_quadtree(src: &RgbImage, params: &FilterParams) -> Vec<Dot> {
    let (width, height) = src.dimensions();
    let mut dots = Vec::new();
    let min_cell = ((width * height) as f32 / params.num_points as f32).sqrt() as u32 / 2;
    let min_cell = min_cell.max(2);
    let threshold = 800.0 * (1.0 - params.variance_sensitivity * 0.8);
    let img_min = width.min(height) as f32;
    subdivide(
        src, 0, 0, width, height, min_cell, threshold, img_min, params, &mut dots,
    );
    dots
}

#[allow(clippy::too_many_arguments)]
fn subdivide(
    src: &RgbImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    min_cell: u32,
    threshold: f32,
    img_min: f32,
    params: &FilterParams,
    dots: &mut Vec<Dot>,
) {
    // Bug 7 corrigé : cellule 1×1 → émet un dot au lieu de silencieusement ignorer
    if w == 0 || h == 0 {
        return;
    }
    let (sr, sg, sb, n) = pixel_sum(src, x, y, w, h);
    if n == 0 {
        return;
    }
    let avg = [(sr / n) as u8, (sg / n) as u8, (sb / n) as u8];

    if w == 1
        || h == 1
        || {
            let variance = pixel_variance(src, x, y, w, h, &avg);
            variance < threshold
        }
        || w <= min_cell
        || h <= min_cell
    {
        let lum = luminance(avg[0], avg[1], avg[2]);
        let cell_ratio = (w.min(h) as f32) / img_min;
        let local_density = cell_ratio.min(1.0);
        dots.push(Dot {
            x: x as f32 + w as f32 / 2.0,
            y: y as f32 + h as f32 / 2.0,
            color: avg,
            radius: radius_for_dot(lum, local_density, img_min, params).max(1.0),
        });
    } else {
        let hw = w / 2;
        let hh = h / 2;
        subdivide(
            src, x, y, hw, hh, min_cell, threshold, img_min, params, dots,
        );
        subdivide(
            src,
            x + hw,
            y,
            w - hw,
            hh,
            min_cell,
            threshold,
            img_min,
            params,
            dots,
        );
        subdivide(
            src,
            x,
            y + hh,
            hw,
            h - hh,
            min_cell,
            threshold,
            img_min,
            params,
            dots,
        );
        subdivide(
            src,
            x + hw,
            y + hh,
            w - hw,
            h - hh,
            min_cell,
            threshold,
            img_min,
            params,
            dots,
        );
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn luminance(r: u8, g: u8, b: u8) -> f32 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

fn pixel_sum(src: &RgbImage, x0: u32, y0: u32, w: u32, h: u32) -> (u64, u64, u64, u64) {
    let (iw, ih) = src.dimensions();
    let (mut sr, mut sg, mut sb, mut n) = (0u64, 0u64, 0u64, 0u64);
    for py in y0..(y0 + h).min(ih) {
        for px in x0..(x0 + w).min(iw) {
            let p = src.get_pixel(px, py);
            sr += p[0] as u64;
            sg += p[1] as u64;
            sb += p[2] as u64;
            n += 1;
        }
    }
    (sr, sg, sb, n)
}

fn pixel_variance(src: &RgbImage, x0: u32, y0: u32, w: u32, h: u32, avg: &[u8; 3]) -> f32 {
    let (iw, ih) = src.dimensions();
    let mut sum = 0f32;
    let mut n = 0u32;
    for py in y0..(y0 + h).min(ih) {
        for px in x0..(x0 + w).min(iw) {
            let p = src.get_pixel(px, py);
            let dr = p[0] as f32 - avg[0] as f32;
            let dg = p[1] as f32 - avg[1] as f32;
            let db = p[2] as f32 - avg[2] as f32;
            sum += dr * dr + dg * dg + db * db;
            n += 1;
        }
    }
    if n > 0 { sum / n as f32 } else { 0.0 }
}

/// Rend les dots en SVG (chaîne de caractères).
/// Chaque dot devient un élément SVG selon `params.dot_shape`.
///
/// Bug 6 corrigé : utilise `params.bg_color` au lieu de hard-coder blanc/noir.
/// Archi 19 : accepte les dots précalculés pour éviter de relancer le filtre.
pub fn render_svg_from_dots(w: u32, h: u32, dots: &[Dot], params: &FilterParams) -> Result<String> {
    if w == 0 || h == 0 {
        return Err(anyhow!("Image vide"));
    }

    // Q1: avoid cloning dots when no quantization is needed
    let quantized: Vec<Dot>;
    let dots_final: &[Dot] = if let Some(n_colors) = params.palette_size {
        quantized = quantize_dots(dots, n_colors.max(2));
        &quantized
    } else {
        dots
    };

    // Bug 6 corrigé : bg_color comme source de vérité
    let [br, bg_c, bb] = params.bg_color;
    let bg_hex = format!("#{:02x}{:02x}{:02x}", br, bg_c, bb);

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#
    );
    svg.push('\n');
    svg.push_str(&format!(
        r#"  <rect width="{w}" height="{h}" fill="{bg_hex}"/>"#
    ));
    svg.push('\n');

    let mut sorted: Vec<&Dot> = dots_final.iter().collect();
    sorted.sort_unstable_by(|a, b| b.radius.total_cmp(&a.radius));

    for dot in sorted {
        let r = dot.radius;
        if r < 0.5 {
            continue;
        }
        let [cr, cg, cb] = dot.color;
        let fill = format!("#{:02x}{:02x}{:02x}", cr, cg, cb);
        let elem = match params.dot_shape {
            DotShape::Circle => {
                format!(
                    "  <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{fill}\"/>",
                    dot.x, dot.y, r
                )
            }
            DotShape::Square => {
                let s = r * 2.0;
                format!(
                    "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{fill}\"/>",
                    dot.x - r,
                    dot.y - r,
                    s,
                    s
                )
            }
            DotShape::Ellipse { aspect, angle_deg } => {
                let rx = r;
                let ry = (r / aspect.max(0.01)).max(0.5);
                format!(
                    "  <ellipse cx=\"{:.1}\" cy=\"{:.1}\" rx=\"{:.1}\" ry=\"{:.1}\" transform=\"rotate({:.1},{:.1},{:.1})\" fill=\"{fill}\"/>",
                    dot.x, dot.y, rx, ry, angle_deg, dot.x, dot.y
                )
            }
            DotShape::RegularPolygon { sides } => {
                let n = sides.max(3) as usize;
                let pts: String = (0..n)
                    .map(|i| {
                        let a = 2.0 * PI * i as f32 / n as f32 - PI / 2.0;
                        format!("{:.1},{:.1}", dot.x + r * a.cos(), dot.y + r * a.sin())
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("  <polygon points=\"{pts}\" fill=\"{fill}\"/>")
            }
        };
        svg.push_str(&elem);
        svg.push('\n');
    }
    svg.push_str("</svg>\n");
    Ok(svg)
}

/// Rend les dots en SVG depuis une `RgbImage` (recalcule les dots).
/// Pour la sauvegarde CLI — préférer `render_svg_from_dots` en GUI.
pub fn render_svg(src: &RgbImage, params: &FilterParams) -> Result<String> {
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 {
        return Err(anyhow!("Image vide"));
    }
    let dots = compute_dots(src, params)?;
    render_svg_from_dots(w, h, &dots, params)
}

/// Rend les dots en SVG depuis une `DynamicImage` (supporte RGBA).
pub fn render_svg_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<String> {
    let rgb = flatten_to_rgb(src, params.bg_color);
    render_svg(&rgb, params)
}

/// Calcule les dots (sans rendu PNG) — permet de mettre en cache pour la GUI.
/// Q2: avoids rendering a full PNG image that would be discarded.
pub fn compute_dots(src: &RgbImage, params: &FilterParams) -> Result<Vec<Dot>> {
    let (w, h) = src.dimensions();
    validate_params(w, h, params)?;
    let density = compute_density_map(src, params.variance_sensitivity);

    let dots = match params.algorithm {
        Algorithm::Grid => dots_grid(src, &density, params),
        Algorithm::Quadtree => dots_quadtree(src, params),
        Algorithm::Voronoi => {
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            compute_dots_voronoi(src, &density, params, &never_cancel)?
        }
        Algorithm::Kmeans => {
            let never_cancel = std::sync::atomic::AtomicBool::new(false);
            compute_dots_kmeans(src, &density, params, &never_cancel)?
        }
    };
    Ok(dots)
}

/// Voronoi dot computation without rendering (Q2).
fn compute_dots_voronoi(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<Vec<Dot>> {
    use std::sync::atomic::Ordering;
    let (width, height) = src.dimensions();
    let k = params.num_points;
    let iters = params.iterations;

    let mut seeds: Vec<(f32, f32)> =
        importance_sample(density, width, height, k, make_rng_seed(params))
            .into_iter()
            .map(|(x, y)| (x as f32 + 0.5, y as f32 + 0.5))
            .collect();

    let power = 1.0 + params.variance_sensitivity * 3.0;
    let lloyd_weights: Vec<f64> = density
        .iter()
        .map(|&d| (d as f64).powf(power as f64))
        .collect();

    for _iter in 0..iters {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let grid = SeedGrid::new(&seeds, width, height);
        let (sum_x, sum_y, sum_w) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut sx, mut sy, mut sw), py| {
                    for px in 0..width {
                        let fx = px as f32 + 0.5;
                        let fy = py as f32 + 0.5;
                        let best = grid.nearest(fx, fy, &seeds);
                        let w = lloyd_weights[(py * width + px) as usize];
                        sx[best] += fx as f64 * w;
                        sy[best] += fy as f64 * w;
                        sw[best] += w;
                    }
                    (sx, sy, sw)
                },
            )
            .reduce(
                || (vec![0f64; k], vec![0f64; k], vec![0f64; k]),
                |(mut ax, mut ay, mut aw), (bx, by, bw)| {
                    for i in 0..k {
                        ax[i] += bx[i];
                        ay[i] += by[i];
                        aw[i] += bw[i];
                    }
                    (ax, ay, aw)
                },
            );
        for i in 0..k {
            if sum_w[i] > 0.0 {
                seeds[i] = ((sum_x[i] / sum_w[i]) as f32, (sum_y[i] / sum_w[i]) as f32);
            }
        }
    }
    Ok(build_dots_from_seeds(src, density, &seeds, params))
}

/// K-means dot computation without rendering (Q2).
fn compute_dots_kmeans(
    src: &RgbImage,
    density: &[f32],
    params: &FilterParams,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<Vec<Dot>> {
    use std::sync::atomic::Ordering;
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
    let k = params.num_points;
    let iters = params.iterations;

    let seeds_pos = importance_sample(density, width, height, k, make_rng_seed(params));
    let mut centers: Vec<[f32; 5]> = seeds_pos
        .iter()
        .map(|&(x, y)| {
            let p = src.get_pixel(x.min(width - 1), y.min(height - 1));
            [x as f32, y as f32, p[0] as f32, p[1] as f32, p[2] as f32]
        })
        .collect();

    let w_f = width as f32;
    let h_f = height as f32;

    for _iter in 0..iters {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        let centers_norm: Vec<[f32; 5]> = centers
            .iter()
            .map(|c| {
                [
                    c[0] / w_f,
                    c[1] / h_f,
                    c[2] / 255.0,
                    c[3] / 255.0,
                    c[4] / 255.0,
                ]
            })
            .collect();

        let (sums, counts) = (0..height)
            .into_par_iter()
            .fold(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut sums, mut counts), py| {
                    for px in 0..width {
                        let p = src.get_pixel(px, py);
                        let feat = [
                            px as f32 / w_f,
                            py as f32 / h_f,
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                        ];
                        let best = centers_norm
                            .iter()
                            .enumerate()
                            .map(|(i, cn)| {
                                let d: f32 = feat
                                    .iter()
                                    .zip(cn.iter())
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum();
                                (i, d)
                            })
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        sums[best][0] += px as f64;
                        sums[best][1] += py as f64;
                        sums[best][2] += p[0] as f64;
                        sums[best][3] += p[1] as f64;
                        sums[best][4] += p[2] as f64;
                        counts[best] += 1;
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![[0f64; 5]; centers.len()], vec![0u64; centers.len()]),
                |(mut as_, mut ac), (bs, bc)| {
                    for i in 0..centers.len() {
                        for j in 0..5 {
                            as_[i][j] += bs[i][j];
                        }
                        ac[i] += bc[i];
                    }
                    (as_, ac)
                },
            );

        for (i, c) in centers.iter_mut().enumerate() {
            let n = counts[i] as f64;
            if n > 0.0 {
                c[0] = (sums[i][0] / n) as f32;
                c[1] = (sums[i][1] / n) as f32;
                c[2] = (sums[i][2] / n) as f32;
                c[3] = (sums[i][3] / n) as f32;
                c[4] = (sums[i][4] / n) as f32;
            }
        }
    }
    Ok(dots_from_kmeans_centers(
        &centers, density, width, height, img_min, params,
    ))
}

fn make_rng_seed(params: &FilterParams) -> u64 {
    params.rng_seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64) // Bug 8 corrigé : nanos complets > subsec_nanos
            .unwrap_or(0xdeadbeef_cafebabe)
    })
}

fn importance_sample(
    weights: &[f32],
    width: u32,
    height: u32,
    k: usize,
    seed: u64,
) -> Vec<(u32, u32)> {
    let total: f64 = weights.iter().map(|&w| w as f64).sum();
    if total <= 0.0 {
        let mut rng: u64 = seed;
        return (0..k)
            .map(|_| {
                // Bug 9 corrigé : utiliser un clamp strict pour éviter coord == width/height
                let x = ((lcg_next(&mut rng) * width as f32) as u32).min(width - 1);
                let y = ((lcg_next(&mut rng) * height as f32) as u32).min(height - 1);
                (x, y)
            })
            .collect();
    }

    let step = total / k as f64;
    let mut rng: u64 = seed;
    let mut result = Vec::with_capacity(k);

    let mut cdf = 0.0f64;
    let mut pixel_idx: usize = 0;
    let n_pixels = (width * height) as usize;

    for i in 0..k {
        let jitter = lcg_next(&mut rng) as f64 - 0.5;
        let target = (i as f64 + 0.5 + jitter * 0.9) * step;
        let target = target.clamp(0.0, total - 1e-9);

        while pixel_idx < n_pixels && cdf + weights[pixel_idx] as f64 <= target {
            cdf += weights[pixel_idx] as f64;
            pixel_idx += 1;
        }
        let idx = pixel_idx.min(n_pixels - 1);
        let px = (idx as u32) % width;
        let py = (idx as u32) / width;
        // C2: sub-pixel jitter to reduce clustering/duplicates when k is large
        let jx = lcg_next(&mut rng);
        let jy = lcg_next(&mut rng);
        let fx = (px as f32 + jx).min(width as f32 - 0.01);
        let fy = (py as f32 + jy).min(height as f32 - 0.01);
        result.push((fx as u32, fy as u32));
    }

    result
}

/// Construit les Dots depuis les graines finales.
///
/// Perf 7 corrigé : parallélisé avec rayon.
fn build_dots_from_seeds(
    src: &RgbImage,
    density: &[f32],
    seeds: &[(f32, f32)],
    params: &FilterParams,
) -> Vec<Dot> {
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
    let k = seeds.len();

    let grid = SeedGrid::new(seeds, width, height);

    // Accumulation parallèle : chaque thread accumule ses propres vecteurs partiels
    let (sum_r, sum_g, sum_b, sum_d, counts) = (0..height)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0f64; k],
                    vec![0u64; k],
                )
            },
            |(mut sr, mut sg, mut sb, mut sd, mut cnt), py| {
                for px in 0..width {
                    let fx = px as f32 + 0.5;
                    let fy = py as f32 + 0.5;
                    let best = grid.nearest(fx, fy, seeds);
                    let p = src.get_pixel(px, py);
                    sr[best] += p[0] as u64;
                    sg[best] += p[1] as u64;
                    sb[best] += p[2] as u64;
                    sd[best] += density[(py * width + px) as usize] as f64;
                    cnt[best] += 1;
                }
                (sr, sg, sb, sd, cnt)
            },
        )
        .reduce(
            || {
                (
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0u64; k],
                    vec![0f64; k],
                    vec![0u64; k],
                )
            },
            |(mut ar, mut ag, mut ab, mut ad, mut ac), (br, bg_vals, bb, bd, bc)| {
                for i in 0..k {
                    ar[i] += br[i];
                    ag[i] += bg_vals[i];
                    ab[i] += bb[i];
                    ad[i] += bd[i];
                    ac[i] += bc[i];
                }
                (ar, ag, ab, ad, ac)
            },
        );

    let nn_radii = nearest_neighbor_radii(seeds);

    seeds
        .iter()
        .enumerate()
        .filter(|(i, _)| counts[*i] > 0)
        .map(|(i, &(x, y))| {
            let n = counts[i];
            let avg = [
                (sum_r[i] / n) as u8,
                (sum_g[i] / n) as u8,
                (sum_b[i] / n) as u8,
            ];
            let lum = luminance(avg[0], avg[1], avg[2]);
            let avg_density = (sum_d[i] / n as f64) as f32;
            let r_uncapped = radius_for_dot(lum, avg_density, img_min, params);
            let r_nn_cap = nn_radii[i] * 0.8;
            Dot {
                x,
                y,
                color: avg,
                radius: r_uncapped.min(r_nn_cap),
            }
        })
        .collect()
}

fn nearest_neighbor_radii(seeds: &[(f32, f32)]) -> Vec<f32> {
    let k = seeds.len();
    if k <= 1 {
        return vec![f32::MAX; k];
    }

    let mut by_x: Vec<usize> = (0..k).collect();
    by_x.sort_unstable_by(|&a, &b| seeds[a].0.total_cmp(&seeds[b].0));

    let mut min_dist_sq = vec![f32::MAX; k];

    for (rank, &i) in by_x.iter().enumerate() {
        let (ax, ay) = seeds[i];
        for &j in by_x.iter().skip(rank + 1) {
            let dx = seeds[j].0 - ax;
            let dx2 = dx * dx;
            if dx2 >= min_dist_sq[i] && dx2 >= min_dist_sq[j] {
                break;
            }
            let dy = seeds[j].1 - ay;
            let d2 = dx2 + dy * dy;
            if d2 < min_dist_sq[i] {
                min_dist_sq[i] = d2;
            }
            if d2 < min_dist_sq[j] {
                min_dist_sq[j] = d2;
            }
        }
    }

    min_dist_sq.iter().map(|&d| d.sqrt() * 0.5).collect()
}

fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Fix #2 : >> 32 donne un u32 complet, divisé par u32::MAX → [0.0, 1.0]
    ((*state >> 32) as f32) / (u32::MAX as f32)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use std::sync::atomic::AtomicBool;

    fn solid_image(w: u32, h: u32, r: u8, g: u8, b: u8) -> RgbImage {
        RgbImage::from_fn(w, h, |_, _| image::Rgb([r, g, b]))
    }

    fn checkerboard(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgb([255, 255, 255])
            } else {
                image::Rgb([0, 0, 0])
            }
        })
    }

    // ── luminance ────────────────────────────────────────────────────────────

    #[test]
    fn luminance_black_is_zero() {
        assert_eq!(luminance(0, 0, 0), 0.0);
    }

    #[test]
    fn luminance_white_is_one() {
        let l = luminance(255, 255, 255);
        assert!((l - 1.0).abs() < 0.001, "luminance blanc ≈ 1, got {l}");
    }

    #[test]
    fn luminance_pure_green_dominant() {
        let l_green = luminance(0, 255, 0);
        let l_red = luminance(255, 0, 0);
        let l_blue = luminance(0, 0, 255);
        assert!(l_green > l_red, "vert > rouge");
        assert!(l_green > l_blue, "vert > bleu");
    }

    // ── radius_for_dot ───────────────────────────────────────────────────────

    fn default_params() -> FilterParams {
        FilterParams::default()
    }

    #[test]
    fn radius_respects_minimum() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r = radius_for_dot(1.0, 1.0, img_min, &p);
        assert!(
            r >= p.min_radius_ratio * img_min * 0.9,
            "rayon trop petit : {r}"
        );
    }

    #[test]
    fn radius_dark_bigger_than_light() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r_dark = radius_for_dot(0.0, 0.5, img_min, &p);
        let r_light = radius_for_dot(1.0, 0.5, img_min, &p);
        assert!(
            r_dark > r_light,
            "sombre doit être plus grand : {r_dark} vs {r_light}"
        );
    }

    #[test]
    fn radius_uniform_zone_bigger_than_detailed() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r_uniform = radius_for_dot(0.5, 0.2, img_min, &p);
        let r_detailed = radius_for_dot(0.5, 1.0, img_min, &p);
        assert!(
            r_uniform > r_detailed,
            "uniforme doit être plus grand : {r_uniform} vs {r_detailed}"
        );
    }

    #[test]
    fn radius_never_exceeds_hard_cap() {
        let p = default_params();
        let img_min = 1000.0f32;
        let r_max_abs = p.max_boost * p.max_radius_ratio * img_min;
        for lum in [0.0f32, 0.5, 1.0] {
            for density in [0.0f32, 0.5, 1.0] {
                let r = radius_for_dot(lum, density, img_min, &p);
                assert!(
                    r <= r_max_abs + 0.01,
                    "rayon dépasse le cap : {r} > {r_max_abs}"
                );
            }
        }
    }

    // ── compute_density_map ──────────────────────────────────────────────────

    #[test]
    fn density_map_uniform_image_is_low_variance() {
        let img = solid_image(32, 32, 128, 128, 128);
        let sensitivity = 0.7;
        let density = compute_density_map(&img, sensitivity);
        let first = density[0];
        for &v in &density {
            assert!(
                (v - first).abs() < 1e-4,
                "densité non uniforme sur image unie"
            );
        }
        assert!(
            (first - 0.3).abs() < 0.05,
            "density sur image unie ≈ 0.3, got {first}"
        );
    }

    #[test]
    fn density_map_checkerboard_is_higher_than_solid() {
        let solid = solid_image(32, 32, 128, 128, 128);
        let checker = checkerboard(32, 32);
        let sensitivity = 1.0;
        let d_solid: f32 =
            compute_density_map(&solid, sensitivity).iter().sum::<f32>() / (32 * 32) as f32;
        let d_checker: f32 = compute_density_map(&checker, sensitivity)
            .iter()
            .sum::<f32>()
            / (32 * 32) as f32;
        assert!(
            d_checker > d_solid,
            "damier doit avoir density > image unie : {d_checker} vs {d_solid}"
        );
    }

    #[test]
    fn density_map_sensitivity_zero_is_all_ones() {
        let img = checkerboard(16, 16);
        let density = compute_density_map(&img, 0.0);
        for &v in &density {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "sensitivity=0 → density=1 partout, got {v}"
            );
        }
    }

    // ── nearest_neighbor_radii ───────────────────────────────────────────────

    #[test]
    fn nn_radii_single_seed_is_max() {
        let seeds = vec![(10.0f32, 20.0f32)];
        let radii = nearest_neighbor_radii(&seeds);
        assert_eq!(radii.len(), 1);
        assert_eq!(radii[0], f32::MAX);
    }

    #[test]
    fn nn_radii_empty_is_empty() {
        let seeds: Vec<(f32, f32)> = vec![];
        let radii = nearest_neighbor_radii(&seeds);
        assert!(radii.is_empty());
    }

    #[test]
    fn nn_radii_two_seeds_correct_distance() {
        let seeds = vec![(0.0f32, 0.0f32), (10.0f32, 0.0f32)];
        let radii = nearest_neighbor_radii(&seeds);
        assert_eq!(radii.len(), 2);
        assert!(
            (radii[0] - 5.0).abs() < 0.01,
            "rayon attendu 5.0, got {}",
            radii[0]
        );
        assert!(
            (radii[1] - 5.0).abs() < 0.01,
            "rayon attendu 5.0, got {}",
            radii[1]
        );
    }

    #[test]
    fn nn_radii_three_seeds_picks_closest() {
        let seeds = vec![(0.0f32, 0.0f32), (4.0f32, 0.0f32), (100.0f32, 0.0f32)];
        let radii = nearest_neighbor_radii(&seeds);
        assert!(
            (radii[0] - 2.0).abs() < 0.01,
            "A: demi-dist 2, got {}",
            radii[0]
        );
        assert!(
            (radii[1] - 2.0).abs() < 0.01,
            "B: demi-dist 2, got {}",
            radii[1]
        );
        assert!(
            (radii[2] - 48.0).abs() < 0.01,
            "C: demi-dist 48, got {}",
            radii[2]
        );
    }

    // ── SeedGrid ─────────────────────────────────────────────────────────────

    #[test]
    fn seedgrid_nearest_brute_force_match() {
        let mut rng: u64 = 0xdeadbeef;
        let seeds: Vec<(f32, f32)> = (0..50)
            .map(|_| (lcg_next(&mut rng) * 200.0, lcg_next(&mut rng) * 200.0))
            .collect();

        let grid = SeedGrid::new(&seeds, 200, 200);

        for qx in [5.0f32, 50.0, 100.0, 150.0, 195.0] {
            for qy in [5.0f32, 50.0, 100.0, 150.0, 195.0] {
                let grid_result = grid.nearest(qx, qy, &seeds);
                let bf_result = seeds
                    .iter()
                    .enumerate()
                    .map(|(i, &(sx, sy))| (i, (qx - sx).powi(2) + (qy - sy).powi(2)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i)
                    .unwrap();
                assert_eq!(
                    grid_result, bf_result,
                    "SeedGrid vs brute-force mismatch à ({qx},{qy}): grid={grid_result} bf={bf_result}"
                );
            }
        }
    }

    #[test]
    fn seedgrid_single_seed_always_returns_zero() {
        let seeds = vec![(50.0f32, 50.0f32)];
        let grid = SeedGrid::new(&seeds, 100, 100);
        for x in [0.0f32, 25.0, 50.0, 99.0] {
            for y in [0.0f32, 25.0, 50.0, 99.0] {
                assert_eq!(grid.nearest(x, y, &seeds), 0);
            }
        }
    }

    // ── apply (intégration) ──────────────────────────────────────────────────

    #[test]
    fn apply_produces_correct_dimensions() {
        let img = solid_image(64, 48, 200, 100, 50);
        let params = FilterParams {
            num_points: 20,
            iterations: 2,
            ..FilterParams::default()
        };
        let result = apply(&img, &params).unwrap();
        assert_eq!(result.dimensions(), (64, 48));
    }

    #[test]
    fn apply_bg_white_and_black() {
        let img = solid_image(32, 32, 0, 0, 0);
        let params_white = FilterParams {
            bg_color: [255, 255, 255],
            num_points: 10,
            iterations: 1,
            ..FilterParams::default()
        };
        let params_black = FilterParams {
            bg_color: [0, 0, 0],
            num_points: 10,
            iterations: 1,
            ..FilterParams::default()
        };
        let res_white = apply(&img, &params_white).unwrap();
        let res_black = apply(&img, &params_black).unwrap();
        let corner_w = res_white.get_pixel(0, 0);
        let corner_b = res_black.get_pixel(0, 0);
        let lum_w = luminance(corner_w[0], corner_w[1], corner_w[2]);
        let lum_b = luminance(corner_b[0], corner_b[1], corner_b[2]);
        assert!(lum_w > lum_b, "fond blanc plus clair que fond noir");
    }

    #[test]
    fn apply_all_algorithms_run() {
        let img = checkerboard(32, 32);
        for algo in [
            Algorithm::Grid,
            Algorithm::Kmeans,
            Algorithm::Voronoi,
            Algorithm::Quadtree,
        ] {
            let params = FilterParams {
                algorithm: algo,
                num_points: 20,
                cols: 8,
                iterations: 2,
                ..FilterParams::default()
            };
            let result = apply(&img, &params).unwrap();
            assert_eq!(
                result.dimensions(),
                (32, 32),
                "algo {algo:?} dimensions incorrectes"
            );
        }
    }

    #[test]
    fn apply_invalid_params_returns_error() {
        let img = solid_image(32, 32, 128, 128, 128);
        let params = FilterParams {
            min_radius_ratio: 0.5,
            max_radius_ratio: 0.1,
            ..FilterParams::default()
        };
        assert!(
            apply(&img, &params).is_err(),
            "min > max doit retourner une erreur"
        );

        let params2 = FilterParams {
            num_points: 0,
            ..FilterParams::default()
        };
        assert!(
            apply(&img, &params2).is_err(),
            "num_points=0 doit retourner une erreur"
        );
    }

    // ── Formes ───────────────────────────────────────────────────────────────

    #[test]
    fn all_dot_shapes_produce_correct_dimensions() {
        let img = checkerboard(32, 32);
        for shape in [
            DotShape::Circle,
            DotShape::Square,
            DotShape::Ellipse {
                aspect: 2.0,
                angle_deg: 30.0,
            },
            DotShape::RegularPolygon { sides: 6 },
        ] {
            let params = FilterParams {
                num_points: 10,
                iterations: 1,
                dot_shape: shape,
                ..FilterParams::default()
            };
            let result = apply(&img, &params).unwrap();
            assert_eq!(result.dimensions(), (32, 32), "shape {shape:?}");
        }
    }

    #[test]
    fn point_in_regular_polygon_center_is_inside() {
        for sides in [3u8, 4, 5, 6, 8] {
            assert!(
                point_in_regular_polygon(0.0, 0.0, 10.0, sides as usize),
                "centre doit être dans polygone à {sides} côtés"
            );
        }
    }

    #[test]
    fn point_in_regular_polygon_far_outside() {
        for sides in [3u8, 4, 6] {
            assert!(
                !point_in_regular_polygon(20.0, 0.0, 10.0, sides as usize),
                "point loin doit être hors polygone à {sides} côtés"
            );
        }
    }

    // ── apply_with_progress + cancel ─────────────────────────────────────────

    #[test]
    fn apply_with_progress_calls_callback() {
        let img = solid_image(32, 32, 100, 150, 200);
        let params = FilterParams {
            algorithm: Algorithm::Voronoi,
            num_points: 10,
            iterations: 3,
            ..FilterParams::default()
        };
        let cancel = AtomicBool::new(false);
        let mut call_count = 0usize;
        let result = apply_with_progress(&img, &params, &cancel, |iter, total, _preview| {
            assert!(iter <= total);
            call_count += 1;
        });
        assert!(result.is_ok());
        assert_eq!(call_count, 3, "callback appelé 3 fois pour 3 itérations");
    }

    #[test]
    fn apply_with_progress_cancel_returns_err() {
        let img = solid_image(64, 64, 100, 150, 200);
        let params = FilterParams {
            algorithm: Algorithm::Voronoi,
            num_points: 50,
            iterations: 10,
            ..FilterParams::default()
        };
        let cancel = AtomicBool::new(true); // déjà annulé
        let result = apply_with_progress(&img, &params, &cancel, |_, _, _| {});
        assert!(result.is_err(), "doit retourner Err quand cancel=true");
        assert!(
            result.unwrap_err().to_string().contains("cancelled"),
            "message d'erreur doit contenir 'cancelled'"
        );
    }

    // ── density image ────────────────────────────────────────────────────────

    #[test]
    fn compute_density_image_correct_size() {
        let img = checkerboard(40, 30);
        let di = compute_density_image(&img, 0.7);
        assert_eq!(di.dimensions(), (40, 30));
    }
}
