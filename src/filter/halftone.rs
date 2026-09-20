//! Mode halftone multi-canaux (rosette CMYK classique ou couleurs dominantes).
//!
//! Le pipeline historique produit un nuage de points uniformes avec des couleurs
//! "moyenne de zone". Le mode halftone, à l'inverse, sépare l'image en **N canaux
//! d'encre** (Cyan, Magenta, Jaune, Noir — ou N couleurs dominantes extraites par
//! k-means), puis place une **trame rotationnée indépendante par canal**. Les dots
//! d'un canal ont la couleur pleine de leur encre (ex. Cyan = [0, 255, 255]) et
//! un rayon proportionnel à la couverture d'encre à leur position.
//!
//! Le compositeur final superpose les canaux en mode **multiply** (mask de
//! transmission) — pas "over" additif — afin de reproduire correctement le
//! mélange soustractif des encres sur papier blanc.

use crate::filter::params::Dot;
use image::RgbImage;
use image::{Rgb, Rgb32FImage};
use serde::{Deserialize, Serialize};

// ─── Types d'API publique (vivent ici, référencés par params.rs) ──────────────

/// Mode de séparation en canaux d'encre. `Off` conserve le pipeline historique.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum HalftoneMode {
    /// Pipeline historique : 1 dot par zone, couleur moyenne. Par défaut.
    #[default]
    Off,
    /// 4 canaux Cyan, Magenta, Jaune, Noir avec angles classiques 15°/75°/0°/45°.
    Cmyk {
        /// Angles [cyan, magenta, jaune, noir] en degrés.
        angles: [f32; 4],
    },
    /// N canaux extraits via k-means sur les pixels de l'image source.
    Dominant {
        /// Nombre de canaux dominants (≥ 2).
        n: usize,
        /// Angle de base en degrés. Les canaux suivants sont placés à
        /// `base + i × 180° / n` (répartition uniforme évitant le moiré).
        base_angle_deg: f32,
    },
}

/// Mode de screening (placement des dots par canal).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum Screening {
    /// AM : grille régulière rotationnée par canal (rosette).
    #[default]
    Am,
    /// FM : blue noise stochastique, density variable.
    Fm,
}

// ─── Types internes (channels, config) ────────────────────────────────────────

/// Canal d'encre prêt à être tramé.
pub(crate) struct InkChannel {
    /// Couleur pleine de l'encre (sera le `Dot::color`).
    pub color: [u8; 3],
    /// Angle de trame AM en degrés (ignoré en FM).
    pub angle_deg: f32,
    /// Couverture par pixel quantifiée sur 8 bits (0..=255), dans l'ordre
    /// `(y * width + x)`. Le stockage `u8` divise par 4 la mémoire des cartes de
    /// couverture par rapport à `f32` — critique pour le mode dominant (jusqu'à
    /// `n` cartes pleine résolution).
    pub coverage: Vec<u8>,
}

/// Quantifie une couverture flottante ∈ [0, 1] sur 8 bits.
#[inline]
fn cov_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ─── Paramètres halftone centralisés ──────────────────────────────────────────

/// Vue sur les champs halftone de `FilterParams` pour éviter le traversing CLI.
pub struct HalftoneConfig<'a> {
    pub mode: &'a HalftoneMode,
    pub screening: Screening,
    /// Fréquence de trame AM en "cells per min(W,H)" (× ratio).
    /// 60 → environ 60 dots par la plus petite dimension (trame 60 lpp).
    pub screen_frequency: f32,
    #[allow(dead_code)]
    pub dot_shape: crate::filter::params::DotShape,
    /// Rayon minimum du dot halftone (fraction de `min(W,H)`).
    pub min_radius_ratio: f32,
    /// Rayon maximum du dot halftone (fraction du step de trame). > 1.0 → dots se touchent.
    pub max_dot_ratio: f32,
    /// Graine RNG pour le screening FM (déterminisme).
    pub rng_seed: Option<u64>,
}

// ─── Entrée publique ──────────────────────────────────────────────────────────

/// Produit l'ensemble des dots halftone (tous canaux confondus) à peindre dans
/// l'ordre dans lequel ils ont été générés. Pour le rendu multiply, l'ordre
/// n'a pas d'importance ( blending commutatif).
///
/// Chaque dot porte la couleur de son canal : en rendu, multiply va simuler
/// l'absorption soustractive.
pub(crate) fn dots_halftone(src: &RgbImage, cfg: &HalftoneConfig) -> Vec<Dot> {
    let channels = match cfg.mode {
        HalftoneMode::Off => return Vec::new(),
        HalftoneMode::Cmyk { angles } => separate_cmyk(src, *angles),
        HalftoneMode::Dominant { n, base_angle_deg } => {
            separate_dominant(src, *n, *base_angle_deg, cfg.rng_seed)
        }
    };
    let mut out = Vec::new();
    for ch in channels {
        let dots = match cfg.screening {
            Screening::Am => screen_am(src, ch, cfg),
            Screening::Fm => screen_fm(src, ch, cfg),
        };
        out.extend(dots);
    }
    out
}

/// High-precision CMYK separation for gamma-corrected halftone rendering.
/// Dominant-color screening keeps the historical RGB8 path for now because its
/// palette extraction is intentionally defined in display RGB space.
pub(crate) fn dots_halftone_linear(src: &Rgb32FImage, cfg: &HalftoneConfig) -> Vec<Dot> {
    let HalftoneMode::Cmyk { angles } = cfg.mode else {
        let rgb = RgbImage::from_fn(src.width(), src.height(), |x, y| {
            let p = src.get_pixel(x, y);
            Rgb([
                (p[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (p[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (p[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ])
        });
        return dots_halftone(&rgb, cfg);
    };
    let channels = separate_cmyk_linear(src, *angles);
    let mut out = Vec::new();
    for channel in channels {
        let dots = match cfg.screening {
            Screening::Am => screen_am_dimensions(src.width(), src.height(), channel, cfg),
            Screening::Fm => screen_fm_dimensions(src.width(), src.height(), channel, cfg),
        };
        out.extend(dots);
    }
    out
}

// ─── Séparation CMYK ─────────────────────────────────────────────────────────

/// Conversion RGB → CMYK avec Under Color Removal (UCR) simple :
/// si C+M+Y ≥ seuil, on remplace la composante neutre (min(C,M,Y)) par du Noir.
fn separate_cmyk(src: &RgbImage, angles: [f32; 4]) -> Vec<InkChannel> {
    let (w, h) = src.dimensions();
    let n = (w as usize) * (h as usize);

    let mut c_cov = vec![0u8; n];
    let mut m_cov = vec![0u8; n];
    let mut y_cov = vec![0u8; n];
    let mut k_cov = vec![0u8; n];

    // Seuil UCR : si la somme C+M+Y dépasse ce seuil, on remplace la partie
    // commune par du Noir. 0.3 = commencement modéré.
    const UCR_THRESHOLD: f32 = 0.30;

    for idx in 0..n {
        let x = idx as u32 % w;
        let y = idx as u32 / w;
        let p = src.get_pixel(x, y);
        let (r, g, b) = (
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
        );
        let k_inv = r.max(g).max(b);
        let k = 1.0 - k_inv; // couverture de noir "naturelle" (max RGB)
        // `scale` normalise les composantes CMY par `1 - k = max(r,g,b)`.
        // - Sur le noir pur (k = 1.0), on n'a pas besoin de CMY → 0.
        // - Sur une couleur saturée (cyan = (0,1,1)), max=1 → scale=1.0, et c=1.
        //   La formule précédente testait `k_inv < 1.0` ce qui zappait le cyan pur.
        let scale = if k < 1.0 { 1.0 / k_inv } else { 0.0 };
        let mut c = (1.0 - r - k) * scale;
        let mut m = (1.0 - g - k) * scale;
        let mut y = (1.0 - b - k) * scale;
        // Clamper → safety.
        c = c.clamp(0.0, 1.0);
        m = m.clamp(0.0, 1.0);
        y = y.clamp(0.0, 1.0);
        // UCR : si C+M+Y dépasse le seuil, enlever la partie commune.
        let sum = c + m + y;
        if sum > UCR_THRESHOLD {
            let grey = c.min(m).min(y); // composante neutre = gris
            c -= grey * 0.5; // retire 50% de la composante neutre (UCR partiel)
            m -= grey * 0.5;
            y -= grey * 0.5;
            // Le noir récupère la composante remplaçant la composante neutre retirée.
            // L'ink coverage totale reste la même à l'écran : c'est l'objectif UCR.
            // (Approximation : on ne calcule pas le gain exact.)
        }
        c_cov[idx] = cov_u8(c);
        m_cov[idx] = cov_u8(m);
        y_cov[idx] = cov_u8(y);
        k_cov[idx] = cov_u8(k);
    }

    vec![
        InkChannel {
            color: [0, 255, 255],
            angle_deg: angles[0],
            coverage: c_cov,
        },
        InkChannel {
            color: [255, 0, 255],
            angle_deg: angles[1],
            coverage: m_cov,
        },
        InkChannel {
            color: [255, 255, 0],
            angle_deg: angles[2],
            coverage: y_cov,
        },
        InkChannel {
            color: [0, 0, 0],
            angle_deg: angles[3],
            coverage: k_cov,
        },
    ]
}

fn separate_cmyk_linear(src: &Rgb32FImage, angles: [f32; 4]) -> Vec<InkChannel> {
    let (w, h) = src.dimensions();
    let n = (w as usize) * (h as usize);
    let mut c_cov = vec![0u8; n];
    let mut m_cov = vec![0u8; n];
    let mut y_cov = vec![0u8; n];
    let mut k_cov = vec![0u8; n];
    const UCR_THRESHOLD: f32 = 0.30;

    for idx in 0..n {
        let p = src.get_pixel(idx as u32 % w, idx as u32 / w);
        let (r, g, b) = (
            p[0].clamp(0.0, 1.0),
            p[1].clamp(0.0, 1.0),
            p[2].clamp(0.0, 1.0),
        );
        let k_inv = r.max(g).max(b);
        let k = 1.0 - k_inv;
        let scale = if k < 1.0 { 1.0 / k_inv.max(1e-6) } else { 0.0 };
        let mut c = ((1.0 - r - k) * scale).clamp(0.0, 1.0);
        let mut m = ((1.0 - g - k) * scale).clamp(0.0, 1.0);
        let mut y = ((1.0 - b - k) * scale).clamp(0.0, 1.0);
        let sum = c + m + y;
        if sum > UCR_THRESHOLD {
            let grey = c.min(m).min(y);
            c -= grey * 0.5;
            m -= grey * 0.5;
            y -= grey * 0.5;
        }
        c_cov[idx] = cov_u8(c);
        m_cov[idx] = cov_u8(m);
        y_cov[idx] = cov_u8(y);
        k_cov[idx] = cov_u8(k);
    }
    vec![
        InkChannel {
            color: [0, 255, 255],
            angle_deg: angles[0],
            coverage: c_cov,
        },
        InkChannel {
            color: [255, 0, 255],
            angle_deg: angles[1],
            coverage: m_cov,
        },
        InkChannel {
            color: [255, 255, 0],
            angle_deg: angles[2],
            coverage: y_cov,
        },
        InkChannel {
            color: [0, 0, 0],
            angle_deg: angles[3],
            coverage: k_cov,
        },
    ]
}

// ─── Séparation dominant colors (k-means) ─────────────────────────────────────

/// Déduplique les couleurs des pixels. Retourne les couleurs uniques (ordre
/// d'apparition) et, pour chaque pixel, l'index de sa couleur unique.
///
/// Le centre le plus proche ne dépend que de la couleur : dédupliquer évite de
/// répéter `n` multiplications par pixel (gain important sur les images
/// graphiques/plates), sans changer le résultat.
fn unique_pixel_colors(src: &RgbImage) -> (Vec<[u8; 3]>, Vec<u32>) {
    use std::collections::HashMap;
    let n_pixels = (src.width() as usize) * (src.height() as usize);
    // Plafond de capacité : évite de réserver des millions d'entrées d'un coup.
    let mut index_of: HashMap<[u8; 3], u32> = HashMap::with_capacity(n_pixels.min(1 << 16));
    let mut unique = Vec::new();
    let mut mapping = Vec::with_capacity(n_pixels);
    for p in src.pixels() {
        let color = [p[0], p[1], p[2]];
        let idx = *index_of.entry(color).or_insert_with(|| {
            unique.push(color);
            (unique.len() - 1) as u32
        });
        mapping.push(idx);
    }
    (unique, mapping)
}

/// K-means sur une liste de couleurs uniques, avec accumulation par pixel dans
/// l'ordre d'origine (sommes f64 stables). `mapping[pixel]` donne l'index de la
/// couleur unique du pixel.
fn compute_dominant_centers_dedup(
    unique: &[[u8; 3]],
    mapping: &[u32],
    n: usize,
    rng_seed: Option<u64>,
) -> Vec<[u8; 3]> {
    use rayon::prelude::*;
    let n_unique = unique.len();
    let n_pixels = mapping.len();
    if n_pixels == 0 || n == 0 || n_unique == 0 {
        return Vec::new();
    }

    // Initialisation : échantillonnage uniforme des couleurs uniques comme centres.
    let step = (n_unique / n.max(1)).max(1);
    let mut centers: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let c = unique[(i * step).min(n_unique - 1)];
            [c[0] as f32, c[1] as f32, c[2] as f32]
        })
        .collect();

    // K-means (max 10 itérations, stop anticipé). Réutilisation du pattern de
    // `render::compute_centers` mais appliqué aux pixels source.
    for _ in 0..10 {
        // Meilleur centre par couleur unique : une seule recherche par couleur.
        let best_per_unique: Vec<u32> = unique
            .par_iter()
            .map(|c| nearest_center(&centers, &[c[0] as f32, c[1] as f32, c[2] as f32]) as u32)
            .collect();

        let (sums, counts) = (0..n_pixels)
            .into_par_iter()
            .fold(
                || (vec![[0f64; 3]; n], vec![0u64; n]),
                |(mut sums, mut counts), idx| {
                    let u = mapping[idx] as usize;
                    let best = best_per_unique[u] as usize;
                    let c = unique[u];
                    sums[best][0] += c[0] as f64;
                    sums[best][1] += c[1] as f64;
                    sums[best][2] += c[2] as f64;
                    counts[best] += 1;
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![[0f64; 3]; n], vec![0u64; n]),
                |(mut s, mut c), (bs, bc)| {
                    for i in 0..n {
                        for j in 0..3 {
                            s[i][j] += bs[i][j];
                        }
                        c[i] += bc[i];
                    }
                    (s, c)
                },
            );

        let mut converged = true;
        for i in 0..n {
            let cnt = counts[i] as f64;
            if cnt > 0.0 {
                let new_c = [
                    (sums[i][0] / cnt) as f32,
                    (sums[i][1] / cnt) as f32,
                    (sums[i][2] / cnt) as f32,
                ];
                let shift = (new_c[0] - centers[i][0]).powi(2)
                    + (new_c[1] - centers[i][1]).powi(2)
                    + (new_c[2] - centers[i][2]).powi(2);
                if shift > 1.0 {
                    converged = false;
                }
                centers[i] = new_c;
            }
        }
        if converged {
            break;
        }
    }
    let _ = rng_seed; // (réservé pour une init k-means++ si l'on veut la variance)
    centers
        .iter()
        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
        .collect()
}

#[inline]
fn nearest_center(centers: &[[f32; 3]], p: &[f32; 3]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, c) in centers.iter().enumerate() {
        let dr = c[0] - p[0];
        let dg = c[1] - p[1];
        let db = c[2] - p[2];
        let d = dr * dr + dg * dg + db * db;
        if d < best_dist {
            best_dist = d;
            best = i;
            if d == 0.0 {
                break;
            }
        }
    }
    best
}

fn separate_dominant(
    src: &RgbImage,
    n: usize,
    base_angle_deg: f32,
    _rng_seed: Option<u64>,
) -> Vec<InkChannel> {
    let (w, h) = src.dimensions();
    let n_pixels = (w as usize) * (h as usize);
    let n = n.max(2);

    let (unique, mapping) = unique_pixel_colors(src);
    let colors = compute_dominant_centers_dedup(&unique, &mapping, n, _rng_seed);
    if colors.is_empty() {
        return Vec::new();
    }
    let centers_f: Vec<[f32; 3]> = colors
        .iter()
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect();

    // Meilleur centre par couleur unique (une recherche par couleur).
    let best_per_unique: Vec<u32> = unique
        .iter()
        .map(|c| nearest_center(&centers_f, &[c[0] as f32, c[1] as f32, c[2] as f32]) as u32)
        .collect();

    // Coverage par canal : "hard assign" sur le centre le plus proche,
    // pondéré par similarité (1 - normalized_distance). Cartes u8 (÷4 mémoire).
    let mut coverages: Vec<Vec<u8>> = (0..n).map(|_| vec![0u8; n_pixels]).collect();
    for (idx, &u) in mapping.iter().enumerate() {
        let ui = u as usize;
        let best = best_per_unique[ui] as usize;
        let c = centers_f[best];
        let p = unique[ui];
        let prgb = [p[0] as f32, p[1] as f32, p[2] as f32];
        let dist_sq =
            (c[0] - prgb[0]).powi(2) + (c[1] - prgb[1]).powi(2) + (c[2] - prgb[2]).powi(2);
        // Similarité ∈ [0, 1] : 1 quand identique, ~0 quand aux antipodes.
        let max_color_dist = 255.0 * 255.0 * 1.5; // distance max attendue
        let sim = 1.0 - (dist_sq / max_color_dist).clamp(0.0, 1.0);
        // Renforcer la sim pour qu'une attribution proche du vrai pixel donne une
        // couverture raisonnable (sinon tout est trop faible).
        let cov = (sim * 1.3).clamp(0.0, 1.0);
        coverages[best][idx] = cov_u8(cov);
    }

    // Angles répartis uniformément par section d'or pour éviter le moiré.
    let angles = (0..n)
        .map(|i| base_angle_deg + (i as f32) * 180.0 / n as f32)
        .collect::<Vec<_>>();

    (0..n)
        .map(|i| InkChannel {
            color: colors[i],
            angle_deg: angles[i],
            coverage: std::mem::take(&mut coverages[i]),
        })
        .collect()
}

// ─── Screening AM (rosette) ──────────────────────────────────────────────────
//
// Pour chaque canal, on génère une grille rotationnée centrée sur l'image au
// pas `screen_frequency` puis on émet un dot par intersection de la grille.
// Le rayon est proportionnel à la couverture moyenne dans la cellule.

#[allow(clippy::too_many_lines)]
fn screen_am(src: &RgbImage, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let (w, h) = src.dimensions();
    screen_am_dimensions(w, h, channel, cfg)
}

fn screen_am_dimensions(w: u32, h: u32, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let img_min = w.min(h) as f32;
    // Pas de trame en pixels : plus fréquence est élevée → plus cellule petite.
    let step = (img_min / cfg.screen_frequency).max(2.0);
    // Bornes de la grille en espace rotationné (assez large pour couvrir toute l'image).
    let half_diag = ((w as f32).hypot(h as f32)) / 2.0 + step;
    let extent = (half_diag / step).ceil() as i32 + 1;

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let ang = channel.angle_deg * std::f32::consts::PI / 180.0;
    let (ux, uy) = (ang.cos(), ang.sin());
    let (vx, vy) = (-uy, ux);

    // Rayon min/max en pixels.
    let r_min = cfg.min_radius_ratio * img_min;
    let r_max = cfg.max_dot_ratio * step; // ≤ step → dots se touchent

    // Cellule d'échantillonnage = 1×1 pixel dans l'espace image (pas step×step),
    // car on lit directement le coverage map indexé par (x_pix,y_pix). En
    // pratique on moyenne sur un disque de rayon ≈ step/2 autour du centre.

    let mut dots = Vec::new();
    for j in -extent..=extent {
        for i in -extent..=extent {
            // Coordonnées dans le repère rotationné (centré sur l'image).
            let sx = i as f32 * step;
            let sy = j as f32 * step;
            // Position dans l'espace image.
            let px = cx + sx * ux + sy * vx;
            let py = cy + sx * uy + sy * vy;
            if px < 0.0 || py < 0.0 || px >= w as f32 || py >= h as f32 {
                continue;
            }
            // Coverage moyenne dans un disque de rayon step/2 autour de (px, py).
            let cov = sample_coverage_disk(&channel.coverage, w, h, px, py, step * 0.5);
            if cov <= 0.005 {
                continue;
            }
            let radius = r_min + (r_max - r_min) * cov;
            // Anti suppression des trop petits.
            if radius < 0.5 {
                continue;
            }
            dots.push(Dot {
                x: px,
                y: py,
                color: channel.color,
                radius,
            });
        }
    }
    dots
}

/// Moyenne de `coverage` (quantifiée u8) dans un disque centré sur `(px, py)`
/// de rayon `r`. Retourne une valeur ∈ [0, 1].
fn sample_coverage_disk(coverage: &[u8], w: u32, h: u32, px: f32, py: f32, r: f32) -> f32 {
    let r_i = r.ceil() as i32;
    if r_i <= 0 {
        let ix = px as u32 % w;
        let iy = py as u32 % h;
        return coverage[(iy * w + ix) as usize] as f32 / 255.0;
    }
    let r2 = r * r;
    let mut sum = 0u32;
    let mut n = 0u32;
    let cx_i = px as i32;
    let cy_i = py as i32;
    for dy in -r_i..=r_i {
        let py_i = cy_i + dy;
        if py_i < 0 || py_i >= h as i32 {
            continue;
        }
        for dx in -r_i..=r_i {
            if (dx * dx + dy * dy) as f32 > r2 {
                continue;
            }
            let px_i = cx_i + dx;
            if px_i < 0 || px_i >= w as i32 {
                continue;
            }
            sum += coverage[(py_i as u32 * w + px_i as u32) as usize] as u32;
            n += 1;
        }
    }
    if n > 0 {
        sum as f32 / (n as f32 * 255.0)
    } else {
        // Fallback au centre exact.
        let ix = (px as u32).min(w - 1);
        let iy = (py as u32).min(h - 1);
        coverage[(iy * w + ix) as usize] as f32 / 255.0
    }
}

// ─── Screening FM (stochastique) ─────────────────────────────────────────────
//
// Variante pragmatique du "frequency-modulated" screening : on garde la grille
// rotationnée (comme AM), mais on regarde la coverage à chaque point et on n'émet
// le dot qu'avec probabilité ≈ coverage. Radius quasi constant (= r_max), c'est
// la **densité** des dots qui encode la couverture d'encre (à l'inverse de AM où
// c'est la **taille** qui encode). Le jitter sub-pixel casse l'aspect régulier de
// la grille → effet direct → rendu plus organique, moins moiré.

fn screen_fm(src: &RgbImage, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let (w, h) = src.dimensions();
    screen_fm_dimensions(w, h, channel, cfg)
}

fn screen_fm_dimensions(w: u32, h: u32, channel: InkChannel, cfg: &HalftoneConfig) -> Vec<Dot> {
    let img_min = w.min(h) as f32;
    let step = (img_min / cfg.screen_frequency).max(2.0);
    let half_diag = ((w as f32).hypot(h as f32)) / 2.0 + step;
    let extent = (half_diag / step).ceil() as i32 + 1;

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let ang = channel.angle_deg * std::f32::consts::PI / 180.0;
    let (ux, uy) = (ang.cos(), ang.sin());
    let (vx, vy) = (-uy, ux);

    let r_max = cfg.max_dot_ratio * step * 0.5;
    let r_min = cfg.min_radius_ratio * img_min;

    // RNG déterministe par (canal, seed). Dérive simple à partir du seed
    // et de la couleur du canal pour avoir des placements indépendants.
    let seed = cfg
        .rng_seed
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xdeadbeef)
        })
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(
            (channel.color[0] as u64) << 40
                | (channel.color[1] as u64) << 24
                | channel.color[2] as u64,
        );
    let mut rng: u64 = seed;

    let mut dots = Vec::new();
    for j in -extent..=extent {
        for i in -extent..=extent {
            // Coordonnées dans le repère rotationné (centré sur l'image).
            let sx = i as f32 * step;
            let sy = j as f32 * step;
            // Position dans l'espace image.
            let px = cx + sx * ux + sy * vx;
            let py = cy + sx * uy + sy * vy;
            if px < 0.0 || py < 0.0 || px >= w as f32 || py >= h as f32 {
                continue;
            }
            let cov = sample_coverage_disk(&channel.coverage, w, h, px, py, step * 0.5);
            if cov <= 0.005 {
                continue;
            }
            // Acceptation stochastique : on émet le dot avec probabilité = cov.
            let thr = lcg_next(&mut rng);
            if thr > cov {
                continue;
            }
            // Jitter sub-pixel légère pour casser la régularité de la grille.
            let jx = (lcg_next(&mut rng) - 0.5) * step * 0.4;
            let jy = (lcg_next(&mut rng) - 0.5) * step * 0.4;
            // Radius quasi constant (légère variation pour naturalité).
            let r = r_min + (r_max - r_min) * (0.7 + 0.3 * lcg_next(&mut rng));
            if r < 0.5 {
                continue;
            }
            dots.push(Dot {
                x: px + jx,
                y: py + jy,
                color: channel.color,
                radius: r,
            });
        }
    }
    dots
}

/// LCG dédié au screening FM (indépendant du crate sampling::lcg_next pour
/// éviter de briser l'encapsulation). Même algorithme.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 32) as f32) / (u32::MAX as f32)
}
