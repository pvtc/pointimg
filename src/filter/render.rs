use crate::filter::params::{Dot, DotShape, FilterParams};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use std::f32::consts::PI;

// ─── Rendu commun ─────────────────────────────────────────────────────────────

// ─── Anti-aliasing ───────────────────────────────────────────────────────────

/// Facteur de supersampling : 4×4 = 16 sous-pixels par pixel.
/// Compromis qualité/perf : couvre 16 niveaux d'opacity par pixel de bord.
const AA_SS: u32 = 4;
const AA_TOTAL: u32 = AA_SS * AA_SS;
const AA_STEP: f32 = 1.0 / AA_SS as f32;

/// Mélange le pixel `(px, py)` vers `color` selon `coverage ∈ (0, AA_TOTAL]`.
/// `coverage == AA_TOTAL` peint en opaque (chemin rapide).
#[inline]
fn blend_coverage(dst: &mut RgbImage, px: u32, py: u32, color: [u8; 3], coverage: u32) {
    debug_assert!(coverage > 0 && coverage <= AA_TOTAL);
    if coverage == AA_TOTAL {
        dst.put_pixel(px, py, Rgb(color));
        return;
    }
    let a = coverage as f32 / AA_TOTAL as f32;
    let inv = 1.0 - a;
    let p = dst.get_pixel_mut(px, py);
    for c in 0..3 {
        let v = p[c] as f32 * inv + color[c] as f32 * a + 0.5;
        p[c] = if v > 255.0 { 255 } else { v as u8 };
    }
}

/// Couverture (en 16ès de pixel) d'un pixel par une forme convexe,
/// en coords locales `(lx, ly)` centrées sur le dot.
///
/// Prétest des 4 coins du pixel : pour une forme convexe,
///   - 4 coins dedans ⇒ pixel entièrement couvert (chemin rapide).
///   - 4 coins dehors + centroïde dehors ⇒ pixel entièrement découvert (skip).
///   - sinon ⇒ supersampling 4×4 (chemin lent ≈ pixels de bord).
///
/// `cx`, `cy` : position float du centre du dot en coords image.
#[inline]
fn coverage_aa<F>(px: i32, py: i32, cx: f32, cy: f32, inside: F) -> u32
where
    F: Fn(f32, f32) -> bool,
{
    let lx_lo = px as f32 - 0.5 - cx;
    let lx_hi = lx_lo + 1.0;
    let ly_lo = py as f32 - 0.5 - cy;
    let ly_hi = ly_lo + 1.0;

    let c00 = inside(lx_lo, ly_lo);
    let c10 = inside(lx_hi, ly_lo);
    let c01 = inside(lx_lo, ly_hi);
    let c11 = inside(lx_hi, ly_hi);

    if c00 && c10 && c01 && c11 {
        return AA_TOTAL;
    }
    if !c00 && !c10 && !c01 && !c11 && !inside(lx_lo + 0.5, ly_lo + 0.5) {
        return 0;
    }

    let mut count = 0u32;
    for sy in 0..AA_SS {
        let ly = ly_lo + (sy as f32 + 0.5) * AA_STEP;
        for sx in 0..AA_SS {
            let lx = lx_lo + (sx as f32 + 0.5) * AA_STEP;
            if inside(lx, ly) {
                count += 1;
                if count == AA_TOTAL {
                    return AA_TOTAL;
                }
            }
        }
    }
    count
}

pub(crate) fn render(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbImage {
    let (w, h) = src.dimensions();
    let bg_rgb = params.bg_color;
    let mut dst = RgbImage::from_pixel(w, h, Rgb(bg_rgb));

    // Si palette + dithering, on garde les couleurs moyennes originales des dots
    // et on post-quantifie l'image rendue via Floyd-Steinberg.
    let quantized: Vec<Dot>;
    let palette_centers: Vec<[u8; 3]>;
    let (dots_to_draw, dither_palette): (&[Dot], &[[u8; 3]]) = match params.palette_size {
        Some(n_colors) if params.dithering => {
            let n = n_colors.max(2);
            palette_centers = quantize_palette_centers(dots, n);
            (dots, &palette_centers[..])
        }
        Some(n_colors) => {
            quantized = quantize_dots(dots, n_colors.max(2));
            (&quantized, &[][..])
        }
        None => (dots, &[][..]),
    };

    // Trier : les plus gros points d'abord (arrière-plan), les petits en dernier (avant-plan)
    let mut sorted: Vec<&Dot> = dots_to_draw.iter().collect();
    sorted.sort_unstable_by(|a, b| b.radius.total_cmp(&a.radius));

    for dot in sorted {
        if dot.radius <= 0.0 {
            continue;
        }
        draw_dot(
            &mut dst,
            dot.x,
            dot.y,
            dot.radius,
            Rgb(dot.color),
            params.dot_shape,
        );
    }

    if !dither_palette.is_empty() {
        crate::filter::dither::floyd_steinberg(&mut dst, dither_palette);
    }
    dst
}

/// Dessine un dot anti-aliasé selon la forme choisie.
///
/// `(cx_f, cy_f)` : centre en pixels (float, préserve la sous-pixel position).
/// `r_f` : rayon en pixels (float).
fn draw_dot(dst: &mut RgbImage, cx_f: f32, cy_f: f32, r_f: f32, color: Rgb<u8>, shape: DotShape) {
    let (iw, ih) = dst.dimensions();

    let bbox = match shape {
        DotShape::Ellipse { aspect, .. } => {
            let a = r_f;
            let b = (r_f / aspect.max(0.01)).max(1.0);
            (a.max(b).ceil() as i32) + 1
        }
        _ => (r_f.ceil() as i32) + 1,
    };
    let b = bbox as f32;

    let px_min = (cx_f - b).ceil() as i32;
    let px_max = (cx_f + b).floor() as i32;
    let py_min = (cy_f - b).ceil() as i32;
    let py_max = (cy_f + b).floor() as i32;

    match shape {
        DotShape::Circle => {
            let r2 = r_f * r_f;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| lx * lx + ly * ly <= r2);
                    if cov > 0 {
                        blend_coverage(dst, px as u32, py as u32, color.0, cov);
                    }
                }
            }
        }
        DotShape::Square => {
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        lx.abs() <= r_f && ly.abs() <= r_f
                    });
                    if cov > 0 {
                        blend_coverage(dst, px as u32, py as u32, color.0, cov);
                    }
                }
            }
        }
        DotShape::Ellipse { aspect, angle_deg } => {
            let a = r_f;
            let b_axis = (r_f / aspect.max(0.01)).max(1.0);
            let ang = angle_deg * PI / 180.0;
            let (cos_a, sin_a) = (ang.cos(), ang.sin());
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        // Rotation inverse : repère de l'ellipse.
                        let rx = lx * cos_a + ly * sin_a;
                        let ry = -lx * sin_a + ly * cos_a;
                        (rx / a).powi(2) + (ry / b_axis).powi(2) <= 1.0
                    });
                    if cov > 0 {
                        blend_coverage(dst, px as u32, py as u32, color.0, cov);
                    }
                }
            }
        }
        DotShape::RegularPolygon { sides } => {
            let n = sides.max(3) as usize;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        point_in_regular_polygon(lx, ly, r_f, n)
                    });
                    if cov > 0 {
                        blend_coverage(dst, px as u32, py as u32, color.0, cov);
                    }
                }
            }
        }
    }
}

/// Test si (px, py) est dans un polygone régulier à `n` côtés de rayon `r`.
pub(crate) fn point_in_regular_polygon(px: f32, py: f32, r: f32, n: usize) -> bool {
    let d = (px * px + py * py).sqrt();
    if d > r {
        return false;
    }
    if d == 0.0 {
        return true;
    }
    let angle = py.atan2(px);
    let sector_angle = 2.0 * PI / n as f32;
    let sector = ((angle % sector_angle) + sector_angle) % sector_angle;
    let apothem_angle = sector - sector_angle / 2.0;
    let apothem = r * (PI / n as f32).cos();
    let dist_to_edge = apothem / apothem_angle.cos().abs();
    d <= dist_to_edge
}

/// K-means sur les couleurs des `dots`, retournant `n` centres en f32.
/// Facteur commun entre `quantize_dots` (nearest-color mapping) et
/// `quantize_palette_centers` (centres seuls, pour dithering).
fn compute_centers(dots: &[Dot], n: usize) -> Vec<[f32; 3]> {
    if dots.is_empty() || n >= dots.len() {
        return Vec::new();
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
    centers
}

/// Centres de palette en u8 (utile pour le dithering Floyd-Steinberg).
pub(crate) fn quantize_palette_centers(dots: &[Dot], n: usize) -> Vec<[u8; 3]> {
    compute_centers(dots, n)
        .iter()
        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
        .collect()
}

/// Quantifie les couleurs des dots en `n` couleurs via k-means simple sur RGB,
/// puis remappe chaque dot sur son centre le plus proche (nearest-color).
pub(crate) fn quantize_dots(dots: &[Dot], n: usize) -> Vec<Dot> {
    if dots.is_empty() || n >= dots.len() {
        return dots.to_vec();
    }
    let centers = compute_centers(dots, n);

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

pub(crate) fn radius_for_dot(
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

// ─── Rendu RGBA (background transparent) ─────────────────────────────────────

/// Mélange le pixel `(px, py)` vers `color` selon `coverage ∈ (0, AA_TOTAL]`
/// en appliquant l'opérateur "over" sur le canal alpha existant.
/// `coverage == AA_TOTAL` peint en opaque (chemin rapide).
#[inline]
fn blend_coverage_rgba(dst: &mut RgbaImage, px: u32, py: u32, color: [u8; 3], coverage: u32) {
    debug_assert!(coverage > 0 && coverage <= AA_TOTAL);
    if coverage == AA_TOTAL {
        dst.put_pixel(px, py, Rgba([color[0], color[1], color[2], 255]));
        return;
    }
    let src_a = coverage as f32 / AA_TOTAL as f32;
    let p = dst.get_pixel_mut(px, py);
    let dst_a = p[3] as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a <= 0.0 {
        return;
    }
    let inv_out = 1.0 / out_a;
    let blend_w = dst_a * (1.0 - src_a) * inv_out;
    let src_w = src_a * inv_out;
    for c in 0..3 {
        let v = color[c] as f32 * src_w + p[c] as f32 * blend_w + 0.5;
        p[c] = if v > 255.0 { 255 } else { v as u8 };
    }
    p[3] = (out_a * 255.0 + 0.5) as u8;
}

/// Variant RGBA de `render`. Quand `params.transparent`, le fond initial est
/// transparent (alpha 0) ; les pixels non couverts par un dot restent transparents.
/// Sinon, le fond initial est `bg_color` opaque (identique à `render` mais en RGBA).
pub fn render_rgba(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbaImage {
    let (w, h) = src.dimensions();
    let bg_pixel = if params.transparent {
        Rgba([
            params.bg_color[0],
            params.bg_color[1],
            params.bg_color[2],
            0,
        ])
    } else {
        Rgba([
            params.bg_color[0],
            params.bg_color[1],
            params.bg_color[2],
            255,
        ])
    };
    let mut dst = RgbaImage::from_pixel(w, h, bg_pixel);

    let quantized: Vec<Dot>;
    let dots_to_draw: &[Dot] = if let Some(n_colors) = params.palette_size {
        quantized = quantize_dots(dots, n_colors.max(2));
        &quantized
    } else {
        dots
    };

    let mut sorted: Vec<&Dot> = dots_to_draw.iter().collect();
    sorted.sort_unstable_by(|a, b| b.radius.total_cmp(&a.radius));

    for dot in sorted {
        if dot.radius <= 0.0 {
            continue;
        }
        draw_dot_rgba(
            &mut dst,
            dot.x,
            dot.y,
            dot.radius,
            dot.color,
            params.dot_shape,
        );
    }
    dst
}

/// Dessine un dot anti-aliasé sur un canvas RGBA (opérateur "over" sur l'alpha).
fn draw_dot_rgba(
    dst: &mut RgbaImage,
    cx_f: f32,
    cy_f: f32,
    r_f: f32,
    color: [u8; 3],
    shape: DotShape,
) {
    let (iw, ih) = dst.dimensions();

    let bbox = match shape {
        DotShape::Ellipse { aspect, .. } => {
            let a = r_f;
            let b = (r_f / aspect.max(0.01)).max(1.0);
            (a.max(b).ceil() as i32) + 1
        }
        _ => (r_f.ceil() as i32) + 1,
    };
    let b = bbox as f32;

    let px_min = (cx_f - b).ceil() as i32;
    let px_max = (cx_f + b).floor() as i32;
    let py_min = (cy_f - b).ceil() as i32;
    let py_max = (cy_f + b).floor() as i32;

    match shape {
        DotShape::Circle => {
            let r2 = r_f * r_f;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| lx * lx + ly * ly <= r2);
                    if cov > 0 {
                        blend_coverage_rgba(dst, px as u32, py as u32, color, cov);
                    }
                }
            }
        }
        DotShape::Square => {
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        lx.abs() <= r_f && ly.abs() <= r_f
                    });
                    if cov > 0 {
                        blend_coverage_rgba(dst, px as u32, py as u32, color, cov);
                    }
                }
            }
        }
        DotShape::Ellipse { aspect, angle_deg } => {
            let a = r_f;
            let b_axis = (r_f / aspect.max(0.01)).max(1.0);
            let ang = angle_deg * PI / 180.0;
            let (cos_a, sin_a) = (ang.cos(), ang.sin());
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        let rx = lx * cos_a + ly * sin_a;
                        let ry = -lx * sin_a + ly * cos_a;
                        (rx / a).powi(2) + (ry / b_axis).powi(2) <= 1.0
                    });
                    if cov > 0 {
                        blend_coverage_rgba(dst, px as u32, py as u32, color, cov);
                    }
                }
            }
        }
        DotShape::RegularPolygon { sides } => {
            let n = sides.max(3) as usize;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        point_in_regular_polygon(lx, ly, r_f, n)
                    });
                    if cov > 0 {
                        blend_coverage_rgba(dst, px as u32, py as u32, color, cov);
                    }
                }
            }
        }
    }
}

// ─── Rendu halftone (composite multiply, ink-over-ink) ────────────────────────

/// Peint un dot sur un canvas RGBA en mode **multiply** : chaque pixel déjà peint
/// est multiplié par le mask de transmission de l'encre (modulé par coverage alpha).
///
/// Ceci simule le mélange soustractif des encres : partant du papier blanc, deux
/// encres successives font que chacune absorbe une partie complémentaire. À
/// l'inverse du mode "over", le résultat de Magenta + Jaune est bien Rouge et non
/// du jaune/magenta qui se superposent.
fn blend_multiply_coverage(dst: &mut RgbaImage, px: u32, py: u32, ink: [u8; 3], coverage: u32) {
    debug_assert!(coverage > 0 && coverage <= AA_TOTAL);
    // alpha en [0, 1] = fraction du pixel couverte par le dot.
    let a = coverage as f32 / AA_TOTAL as f32;
    if a >= 1.0 {
        // Multiply pleine : dst[i] = dst[i] * mask[i] / 255.
        let p = dst.get_pixel_mut(px, py);
        p[0] = ((p[0] as u32 * ink[0] as u32) / 255) as u8;
        p[1] = ((p[1] as u32 * ink[1] as u32) / 255) as u8;
        p[2] = ((p[2] as u32 * ink[2] as u32) / 255) as u8;
        // En mode multiply, le papier reste opaque : on force alpha à 255.
        p[3] = 255;
        return;
    }
    let p = dst.get_pixel_mut(px, py);
    for c in 0..3 {
        let dst_v = p[c] as f32;
        let mask = ink[c] as f32 / 255.0;
        // dst_after_full_ink = dst_v * mask
        // blended = dst_v * (1 - a) + dst_after_full_ink * a
        //         = dst_v * (1 - a + a * mask)
        let factor = 1.0 - a + a * mask;
        let v = dst_v * factor + 0.5;
        p[c] = if v > 255.0 { 255 } else { v as u8 };
    }
    p[3] = 255;
}

fn draw_dot_halftone(
    dst: &mut RgbaImage,
    cx_f: f32,
    cy_f: f32,
    r_f: f32,
    ink: [u8; 3],
    shape: DotShape,
) {
    let (iw, ih) = dst.dimensions();

    let bbox = match shape {
        DotShape::Ellipse { aspect, .. } => {
            let a_axis = r_f;
            let b_axis = (r_f / aspect.max(0.01)).max(1.0);
            (a_axis.max(b_axis).ceil() as i32) + 1
        }
        _ => (r_f.ceil() as i32) + 1,
    };
    let b = bbox as f32;

    let px_min = (cx_f - b).ceil() as i32;
    let px_max = (cx_f + b).floor() as i32;
    let py_min = (cy_f - b).ceil() as i32;
    let py_max = (cy_f + b).floor() as i32;

    match shape {
        DotShape::Circle => {
            let r2 = r_f * r_f;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| lx * lx + ly * ly <= r2);
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
        DotShape::Square => {
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        lx.abs() <= r_f && ly.abs() <= r_f
                    });
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
        DotShape::Ellipse { aspect, angle_deg } => {
            let a_axis = r_f;
            let b_axis = (r_f / aspect.max(0.01)).max(1.0);
            let ang = angle_deg * PI / 180.0;
            let (cos_a, sin_a) = (ang.cos(), ang.sin());
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        let rx = lx * cos_a + ly * sin_a;
                        let ry = -lx * sin_a + ly * cos_a;
                        (rx / a_axis).powi(2) + (ry / b_axis).powi(2) <= 1.0
                    });
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
        DotShape::RegularPolygon { sides } => {
            let n = sides.max(3) as usize;
            for py in py_min..=py_max {
                if py < 0 || py >= ih as i32 {
                    continue;
                }
                for px in px_min..=px_max {
                    if px < 0 || px >= iw as i32 {
                        continue;
                    }
                    let cov = coverage_aa(px, py, cx_f, cy_f, |lx, ly| {
                        point_in_regular_polygon(lx, ly, r_f, n)
                    });
                    if cov > 0 {
                        blend_multiply_coverage(dst, px as u32, py as u32, ink, cov);
                    }
                }
            }
        }
    }
}

/// Rendu halftone : part du papier (bg_color, ou blanc si multiplicatif), et
/// applique chaque dot en mode multiply. L'ordre n'a pas d'importance
/// (commutatif). Les pixels non couverts restent du papier.
pub fn render_halftone(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbaImage {
    let (w, h) = src.dimensions();
    // Papier : en halftone on part de blanc par défaut (multiply ne marche pas sur
    // noir). Si l'utilisateur a explicitement demandé une couleur, on l'utilise.
    let paper = if params.transparent {
        Rgba([255, 255, 255, 0])
    } else {
        Rgba([
            params.bg_color[0],
            params.bg_color[1],
            params.bg_color[2],
            255,
        ])
    };
    let mut dst = RgbaImage::from_pixel(w, h, paper);

    // Pas de tri nécessaire : multiply est commutatif. Mais on garde l'ordre de
    // génération (par canal) pour cohérence avec les tests Jest-like.
    for dot in dots {
        if dot.radius <= 0.0 {
            continue;
        }
        draw_dot_halftone(
            &mut dst,
            dot.x,
            dot.y,
            dot.radius,
            dot.color,
            params.dot_shape,
        );
    }
    dst
}
