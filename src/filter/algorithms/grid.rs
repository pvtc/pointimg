use crate::filter::density::zone_density;
use crate::filter::params::{Dot, FilterParams};
use crate::filter::render::radius_for_dot;
use crate::filter::util::{luminance, pixel_sum};
use image::RgbImage;
use std::f32::consts::PI;

// ─── Algorithme 1 : Grille ────────────────────────────────────────────────────

pub(crate) fn dots_grid(src: &RgbImage, density: &[f32], params: &FilterParams) -> Vec<Dot> {
    let (width, height) = src.dimensions();
    let img_min = width.min(height) as f32;
    let cell = (width / params.cols).max(1);
    let cols = width / cell;
    let rows = height / cell;

    // Centre de rotation = centre de l'image (coordDevice 0..W, 0..H).
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let angle = params.grid_angle_deg * PI / 180.0;
    let (cos_a, sin_a) = (angle.cos(), angle.sin());
    let rotated = params.grid_angle_deg.abs() > 1e-6;

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
            // Centre de la cellule.
            let cell_cx = x0 as f32 + cell as f32 / 2.0;
            let cell_cy = y0 as f32 + cell as f32 / 2.0;
            // Rotation autour du centre de l'image (effet halftone screen).
            let (dot_x, dot_y) = if rotated {
                let dx = cell_cx - cx;
                let dy = cell_cy - cy;
                (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
            } else {
                (cell_cx, cell_cy)
            };
            Some(Dot {
                x: dot_x,
                y: dot_y,
                color: avg,
                radius: radius_for_dot(lum, d, img_min, params).min(r_nn_cap),
            })
        })
        .collect()
}
