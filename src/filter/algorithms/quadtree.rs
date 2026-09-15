use crate::filter::params::{Dot, FilterParams};
use crate::filter::render::radius_for_dot;
use crate::filter::util::luminance;
use image::RgbImage;

// ─── Algorithme 4 : Quadtree adaptatif ───────────────────────────────────────
//
// Tables intégrales (summed-area tables) en entiers u64 : chaque requête de
// somme ou de somme de carrés sur un rectangle est O(1), au lieu de
// re-parcourir tous les pixels du nœud à chaque subdivision. L'arithmétique
// entière est exacte (aucun arrondi flottant par chemin), et la variance
// s'exprime en forme close : Σ(p−a)² = Σp² − 2a·Σp + n·a².

/// Tables intégrales RGB : sommes et sommes de carrés, layout (w+1)×(h+1)
/// avec une ligne/colonne de zéros pour des requêtes sans branches.
struct IntegralImage {
    sums: Vec<[u64; 3]>,
    squares: Vec<[u64; 3]>,
    stride: usize,
    img_w: u32,
    img_h: u32,
}

impl IntegralImage {
    fn build(src: &RgbImage) -> Self {
        let (w, h) = src.dimensions();
        let stride = w as usize + 1;
        let len = stride * (h as usize + 1);
        let mut sums = vec![[0u64; 3]; len];
        let mut squares = vec![[0u64; 3]; len];
        let raw = src.as_raw();
        let row_stride = w as usize * 3;
        for y in 0..h as usize {
            let row_start = y * row_stride;
            let row = &raw[row_start..row_start + row_stride];
            for (x, p) in row.as_chunks::<3>().0.iter().enumerate() {
                let idx = (y + 1) * stride + (x + 1);
                let left = idx - 1;
                let up = idx - stride;
                let diag = up - 1;
                for c in 0..3 {
                    let v = p[c] as u64;
                    sums[idx][c] = v + sums[left][c] + sums[up][c] - sums[diag][c];
                    squares[idx][c] = v * v + squares[left][c] + squares[up][c] - squares[diag][c];
                }
            }
        }
        Self {
            sums,
            squares,
            stride,
            img_w: w,
            img_h: h,
        }
    }

    /// Somme sur le rectangle [x0, x1) × [y0, y1). Arithmétique wrap-around :
    /// le résultat est exact mod 2⁶⁴, et le vrai résultat est ≥ 0.
    #[inline]
    fn rect_query(
        table: &[[u64; 3]],
        stride: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
    ) -> [u64; 3] {
        let a = table[y1 * stride + x1];
        let b = table[y0 * stride + x1];
        let c = table[y1 * stride + x0];
        let d = table[y0 * stride + x0];
        [
            a[0].wrapping_sub(b[0])
                .wrapping_sub(c[0])
                .wrapping_add(d[0]),
            a[1].wrapping_sub(b[1])
                .wrapping_sub(c[1])
                .wrapping_add(d[1]),
            a[2].wrapping_sub(b[2])
                .wrapping_sub(c[2])
                .wrapping_add(d[2]),
        ]
    }

    /// Sommes RGB et nombre de pixels du rectangle (clampé à l'image, comme
    /// l'ancien parcours direct).
    fn rect_sums(&self, x: u32, y: u32, w: u32, h: u32) -> ([u64; 3], u64) {
        let x0 = x as usize;
        let y0 = y as usize;
        let x1 = ((x + w) as usize).min(self.img_w as usize);
        let y1 = ((y + h) as usize).min(self.img_h as usize);
        let n = (x1 - x0) as u64 * (y1 - y0) as u64;
        (Self::rect_query(&self.sums, self.stride, x0, y0, x1, y1), n)
    }

    /// Σ(p − avg)² sur le rectangle, avg = moyenne tronquée par canal (u8),
    /// en entiers exacts : Σp² − 2a·Σp + n·a² par canal.
    fn rect_var_num(&self, x: u32, y: u32, w: u32, h: u32, avg: &[u8; 3]) -> (u64, u64) {
        let x0 = x as usize;
        let y0 = y as usize;
        let x1 = ((x + w) as usize).min(self.img_w as usize);
        let y1 = ((y + h) as usize).min(self.img_h as usize);
        let sums = Self::rect_query(&self.sums, self.stride, x0, y0, x1, y1);
        let squares = Self::rect_query(&self.squares, self.stride, x0, y0, x1, y1);
        let n = (x1 - x0) as u64 * (y1 - y0) as u64;
        let mut num = 0u64;
        for c in 0..3 {
            let a = avg[c] as u64;
            num = num
                .wrapping_add(squares[c])
                .wrapping_add(n.wrapping_mul(a.wrapping_mul(a)))
                .wrapping_sub(2u64.wrapping_mul(a).wrapping_mul(sums[c]));
        }
        (num, n)
    }
}

pub(crate) fn dots_quadtree(src: &RgbImage, params: &FilterParams) -> Vec<Dot> {
    let (width, height) = src.dimensions();
    let ii = IntegralImage::build(src);
    let mut dots = Vec::new();
    let min_cell = ((width * height) as f32 / params.num_points as f32).sqrt() as u32 / 2;
    let min_cell = min_cell.max(2);
    let threshold = 800.0 * (1.0 - params.variance_sensitivity * 0.8);
    let img_min = width.min(height) as f32;
    subdivide(
        &ii, 0, 0, width, height, min_cell, threshold, img_min, params, &mut dots,
    );
    dots
}

#[allow(clippy::too_many_arguments)]
fn subdivide(
    ii: &IntegralImage,
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
    // Emit a dot for 1×1 cells instead of silently dropping them.
    if w == 0 || h == 0 {
        return;
    }
    let (sums, n) = ii.rect_sums(x, y, w, h);
    if n == 0 {
        return;
    }
    let avg = [
        (sums[0] / n) as u8,
        (sums[1] / n) as u8,
        (sums[2] / n) as u8,
    ];

    if w == 1
        || h == 1
        || {
            let (var_num, var_n) = ii.rect_var_num(x, y, w, h, &avg);
            let variance = var_num as f32 / var_n as f32;
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
        subdivide(ii, x, y, hw, hh, min_cell, threshold, img_min, params, dots);
        subdivide(
            ii,
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
            ii,
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
            ii,
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
