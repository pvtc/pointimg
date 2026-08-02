// ─── Grille spatiale pour nearest-seed ───────────────────────────────────────

pub(crate) struct SeedGrid {
    cells: Vec<Vec<usize>>,
    cols: usize,
    rows: usize,
    cell_w: f32,
    cell_h: f32,
}

impl SeedGrid {
    pub(crate) fn new(seeds: &[(f32, f32)], img_w: u32, img_h: u32) -> Self {
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
    pub(crate) fn nearest(&self, fx: f32, fy: f32, seeds: &[(f32, f32)]) -> usize {
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
