//! Dithering Floyd-Steinberg pour la quantification de palette.
//!
//! Quand `FilterParams::dithering` est vrai (et `palette_size` est Some(n)),
//! l'image rendue finale est post-quantifiée en parcourant l'image ligne par
//! ligne, colonne par colonne, et en propageant l'erreur de quantification vers
//! les voisins non encore traités selon les poids classiques de Floyd-Steinberg :
//!
//! ```text
//!        *   7/16
//! 3/16  5/16  1/16
//! ```
//!
//! Chaque pixel est remplacé par la couleur de palette la plus proche, puis
//! l'erreur (originale - quantifiée) est distribuée aux 4 voisins futurs. À la
//! fin, la palette est respectée exactement, mais le diagramme moyenné garde
//! la teinte moyenne d'origine — d'où l'illusion de continuité.

use image::{Rgb, RgbImage};

/// Quantifie chaque pixel de `dst` vers la palette fournie en appliquant
/// Floyd-Steinberg. Opère **en place** sur l'image déjà rendue.
///
/// `palette` est une liste de couleurs RGB ; si elle contient < 2 entrées, on
/// ne fait rien (le rendu est supposé déjà opaque).
pub(crate) fn floyd_steinberg(dst: &mut RgbImage, palette: &[[u8; 3]]) {
    if palette.len() < 2 {
        return;
    }
    let (w, h) = dst.dimensions();
    // Travail sur un buffer mutable f32 pour propager l'erreur sans perte.
    let mut buf: Vec<[f32; 3]> = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = dst.get_pixel(x, y);
            buf.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    let idx = |x: u32, y: u32| -> usize { (y as usize) * w as usize + x as usize };

    let nearest = |c: &[f32; 3]| -> [u8; 3] {
        palette
            .iter()
            .min_by(|a, b| {
                let da = (a[0] as f32 - c[0]).powi(2)
                    + (a[1] as f32 - c[1]).powi(2)
                    + (a[2] as f32 - c[2]).powi(2);
                let db = (b[0] as f32 - c[0]).powi(2)
                    + (b[1] as f32 - c[1]).powi(2)
                    + (b[2] as f32 - c[2]).powi(2);
                da.total_cmp(&db)
            })
            .copied()
            .unwrap_or([0, 0, 0])
    };

    for y in 0..h {
        for x in 0..w {
            let cur = &mut buf[idx(x, y)];
            let old = *cur;
            let q = nearest(&old);
            // Écrire le pixel quantifié dans dst immédiatement.
            dst.put_pixel(x, y, Rgb(q));
            // Erreur en float.
            let err = [
                old[0] - q[0] as f32,
                old[1] - q[1] as f32,
                old[2] - q[2] as f32,
            ];
            // Propagation Floyd-Steinberg.
            let mut distribute = |nx: i32, ny: i32, weight: f32| {
                if nx >= 0 && (nx as u32) < w && ny >= 0 && (ny as u32) < h {
                    let n = &mut buf[idx(nx as u32, ny as u32)];
                    n[0] += err[0] * weight;
                    n[1] += err[1] * weight;
                    n[2] += err[2] * weight;
                }
            };
            distribute(x as i32 + 1, y as i32, 7.0 / 16.0);
            distribute(x as i32 - 1, y as i32 + 1, 3.0 / 16.0);
            distribute(x as i32, y as i32 + 1, 5.0 / 16.0);
            distribute(x as i32 + 1, y as i32 + 1, 1.0 / 16.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(c))
    }

    #[test]
    fn dither_solid_color_maps_to_palette_entry() {
        let mut img = solid(8, 8, [200, 30, 30]);
        floyd_steinberg(&mut img, &[[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
        // Tous les pixels doivent faire partie de la palette.
        for p in img.pixels() {
            let in_palette = p[0] == 255 && p[1] == 0 && p[2] == 0
                || p[0] == 0 && p[1] == 255 && p[2] == 0
                || p[0] == 0 && p[1] == 0 && p[2] == 255;
            assert!(in_palette, "pixel hors palette : {:?}", p.0);
        }
    }

    #[test]
    fn dither_empty_palette_is_noop() {
        let mut img = solid(4, 4, [123, 45, 67]);
        let before: Vec<_> = img.pixels().copied().collect();
        floyd_steinberg(&mut img, &[]);
        let after: Vec<_> = img.pixels().copied().collect();
        assert_eq!(before, after, "palette vide doit être no-op");
    }

    #[test]
    fn dither_keeps_dimensions() {
        let mut img = solid(10, 7, [128, 128, 128]);
        floyd_steinberg(&mut img, &[[0, 0, 0], [255, 255, 255]]);
        assert_eq!(img.dimensions(), (10, 7));
    }
}
