use crate::filter::params::{Dot, FilterParams};
use crate::filter::render::radius_for_dot;
use crate::filter::util::{luminance, pixel_sum, pixel_variance};
use image::RgbImage;

// ─── Algorithme 4 : Quadtree adaptatif ───────────────────────────────────────

pub(crate) fn dots_quadtree(src: &RgbImage, params: &FilterParams) -> Vec<Dot> {
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
