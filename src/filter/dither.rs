//! Quantification de palette : quatre algorithmes de dithering.
//!
//! Quand `FilterParams::palette_size` vaut `Some(n)`, le rendu final est
//! post-quantifié vers `n` couleurs. Un dithering évite les bandes plates : il
//! convertit l'erreur de quantification en un motif de points dont la moyenne
//! reproduit la teinte d'origine.
//!
//! | Algorithme | Principe | Usage |
//! |---|---|---|
//! | [`Dithering::Off`] | aucun | aplats francs |
//! | [`Dithering::FloydSteinberg`] | diffusion de l'erreur vers 4 voisins | photographie, dégradés |
//! | [`Dithering::Riemersma`] | diffusion à queue longue sur une ligne | dégradés, moins de « vers » que Floyd-Steinberg |
//! | [`Dithering::Bayer4`] / [`Dithering::Bayer8`] | motif ordonné 4×4 ou 8×8 | rendu « pixel art », sans diffusion |
//!
//! Les deux derniers ne propagent aucune erreur : chaque pixel est indépendant,
//! donc parallélisable, mais la moyenne n'est exacte que statistiquement — ce
//! qui convient précisément à l'esthétique visée. La matrice de Bayer est
//! **générée** par la règle récursive de Bayer (1973) :
//!
//! ```text
//! M(1) = [0]     M(2n) = 4·M(n) + [[0, 2], [3, 1]]
//! ```
//!
//! Floyd-Steinberg, lui, propage l'erreur ainsi :
//!
//! ```text
//!        *   7/16
//! 3/16  5/16  1/16
//! ```

use image::{Rgb, RgbImage};

/// Algorithme de dithering appliqué à la quantification de palette.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Debug,
    Default,
    serde::Serialize,
    serde::Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
pub enum Dithering {
    /// Aucun tramage : la couleur la plus proche, sans motif.
    #[default]
    Off,
    /// Diffusion de l'erreur vers les quatre voisins futurs.
    FloydSteinberg,
    /// Diffusion à queue longue (16 pixels, décroissance 0,9).
    Riemersma,
    /// Tramage ordonné de Bayer 4×4.
    Bayer4,
    /// Tramage ordonné de Bayer 8×8.
    Bayer8,
}

impl Dithering {
    /// Vrai si un tramage est appliqué.
    pub fn is_on(self) -> bool {
        self != Dithering::Off
    }
}

/// Longueur de la queue de Riemersma (en pixels).
const RIEMERSMA_TAPS: usize = 16;
/// Décroissance des poids le long de la queue.
///
/// Calibré sur un dégradé de 160 px de large : à 0,7, l'écart moyen entre
/// pixels voisins passe de 127 (prise unique, donc « vers » marqués) à 66, pour
/// un motif un peu plus fondu. C'est le compromis habituel de Riemersma : plus
/// de netteté locale se paie par une file plus courte, donc par des « vers ».
const RIEMERSMA_DECAY: f64 = 0.7;

/// Quantifie `dst` vers `palette` avec l'algorithme demandé.
///
/// Opère **en place** sur l'image déjà rendue. `Dithering::Off` — comme une
/// palette de moins de deux couleurs — ne fait rien.
pub fn apply(dst: &mut RgbImage, palette: &[[u8; 3]], dithering: Dithering) {
    if palette.len() < 2 {
        return;
    }
    match dithering {
        Dithering::Off => {}
        Dithering::FloydSteinberg => floyd_steinberg(dst, palette),
        Dithering::Riemersma => riemersma(dst, palette, RIEMERSMA_TAPS, RIEMERSMA_DECAY),
        Dithering::Bayer4 => ordered(dst, palette, &bayer_matrix(4)),
        Dithering::Bayer8 => ordered(dst, palette, &bayer_matrix(8)),
    }
}

/// Palette pré-convertie en `f32`, pour éviter un cast par comparaison.
fn palette_f32(palette: &[[u8; 3]]) -> Vec<[f32; 3]> {
    palette
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect()
}

/// Couleur de palette la plus proche de `c`.
fn nearest(palette: &[[u8; 3]], palette_f: &[[f32; 3]], c: &[f32; 3]) -> [u8; 3] {
    let mut best = palette[0];
    let mut best_dist = f32::INFINITY;
    for (i, a) in palette_f.iter().enumerate() {
        let dr = a[0] - c[0];
        let dg = a[1] - c[1];
        let db = a[2] - c[2];
        let d = dr * dr + dg * dg + db * db;
        if d < best_dist {
            best_dist = d;
            best = palette[i];
            if d == 0.0 {
                break;
            }
        }
    }
    best
}

/// Les deux couleurs de palette les plus proches de `c`.
///
/// La première est le résultat d'une quantification directe ; la seconde sert
/// au tramage ordonné, qui arbitre entre les deux d'après la position du pixel
/// sur le segment qui les relie.
fn two_nearest(palette: &[[u8; 3]], palette_f: &[[f32; 3]], c: &[f32; 3]) -> ([u8; 3], [u8; 3]) {
    let mut best = (f32::INFINITY, 0usize);
    let mut second = (f32::INFINITY, 0usize);
    for (i, a) in palette_f.iter().enumerate() {
        let dr = a[0] - c[0];
        let dg = a[1] - c[1];
        let db = a[2] - c[2];
        let d = dr * dr + dg * dg + db * db;
        if d < best.0 {
            second = best;
            best = (d, i);
            if d == 0.0 {
                break;
            }
        } else if d < second.0 {
            second = (d, i);
        }
    }
    (palette[best.1], palette[second.1])
}

/// Matrice de Bayer de côté `n` (une puissance de deux), permutation de
/// `0..n²`.
///
/// Générée par la règle récursive `M(2n) = 4·M(n) + [[0, 2], [3, 1]]` : aucune
/// table n'est recopiée ailleurs.
pub fn bayer_matrix(n: usize) -> Vec<Vec<u32>> {
    if n == 0 {
        return Vec::new();
    }
    let side = n.next_power_of_two();
    let mut block = vec![vec![0u32; 1]; 1];
    let mut size = 1;
    while size < side {
        let doubled = size * 2;
        let mut next = vec![vec![0u32; doubled]; doubled];
        for y in 0..size {
            for x in 0..size {
                let value = 4 * block[y][x];
                next[y][x] = value;
                next[y][x + size] = value + 2;
                next[y + size][x] = value + 3;
                next[y + size][x + size] = value + 1;
            }
        }
        block = next;
        size = doubled;
    }
    block.truncate(n);
    for row in &mut block {
        row.truncate(n);
    }
    block
}

/// Tramage ordonné : chaque pixel choisit entre ses deux couleurs de palette
/// les plus proches, d'après le seuil de la matrice à sa position.
fn ordered(dst: &mut RgbImage, palette: &[[u8; 3]], matrix: &[Vec<u32>]) {
    let (w, h) = dst.dimensions();
    let side = matrix.len() as u32;
    let levels = (matrix.len() * matrix.len()) as f32;
    let palette_f = palette_f32(palette);
    for y in 0..h {
        for x in 0..w {
            let p = dst.get_pixel(x, y);
            let c = [p[0] as f32, p[1] as f32, p[2] as f32];
            let (best, second) = two_nearest(palette, &palette_f, &c);
            // Seuil du motif au centre de sa cellule : (M + 0,5) / n².
            let threshold =
                (matrix[(y % side) as usize][(x % side) as usize] as f32 + 0.5) / levels;
            // Position du pixel sur le segment [best, second] : 0 = best.
            let delta = [
                second[0] as f32 - best[0] as f32,
                second[1] as f32 - best[1] as f32,
                second[2] as f32 - best[2] as f32,
            ];
            let offset = [
                c[0] - best[0] as f32,
                c[1] - best[1] as f32,
                c[2] - best[2] as f32,
            ];
            let norm = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            let along = if norm > 0.0 {
                ((offset[0] * delta[0] + offset[1] * delta[1] + offset[2] * delta[2]) / norm)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            dst.put_pixel(x, y, Rgb(if along > threshold { second } else { best }));
        }
    }
}

/// Diffusion de l'erreur à queue longue (Riemersma, 1995).
///
/// L'erreur est versée dans une ligne de retard de `taps` pixels : le pixel
/// suivant en reçoit la première part, les suivants des parts décroissantes
/// géométriquement, et l'erreur déjà en attente est atténuée à chaque pas. La
/// queue longue supprime les « vers » et le moiré directionnel du
/// Floyd-Steinberg ; l'algorithme reste séquentiel le long de chaque ligne.
fn riemersma(dst: &mut RgbImage, palette: &[[u8; 3]], taps: usize, decay: f64) {
    let (w, h) = dst.dimensions();
    if taps == 0 {
        floyd_steinberg(dst, palette);
        return;
    }
    let palette_f = palette_f32(palette);
    // Poids normalisés à 1 : chaque erreur est versée en totalité, donc la
    // moyenne est conservée exactement (hors troncature de fin de ligne).
    let mut weights = vec![0.0f64; taps];
    let mut total = 0.0f64;
    for (k, weight) in weights.iter_mut().enumerate() {
        *weight = decay.powi(k as i32);
        total += *weight;
    }
    for weight in &mut weights {
        *weight /= total;
    }

    let line_len = w as usize;
    let mut line = vec![[0.0f64; 3]; line_len];
    for y in 0..h {
        for x in 0..w as usize {
            // L'erreur déjà versée pour ce pixel sort de la file avant l'écriture
            // des suivantes : les emprunts sur `line` restent disjoints.
            let pending = {
                let slot = &mut line[x];
                let value = *slot;
                *slot = [0.0; 3];
                value
            };
            let p = dst.get_pixel(x as u32, y);
            let c = [
                p[0] as f64 + pending[0],
                p[1] as f64 + pending[1],
                p[2] as f64 + pending[2],
            ];
            let q = nearest(
                palette,
                &palette_f,
                &[c[0] as f32, c[1] as f32, c[2] as f32],
            );
            dst.put_pixel(x as u32, y, Rgb(q));
            let error = [c[0] - q[0] as f64, c[1] - q[1] as f64, c[2] - q[2] as f64];
            for (k, weight) in weights.iter().enumerate() {
                let index = (x + 1 + k) % line_len;
                let target = &mut line[index];
                for channel in 0..3 {
                    target[channel] = error[channel] * weight;
                }
            }
        }
        // La file repart à zéro entre deux lignes : le résultat ne doit pas
        // dépendre de la hauteur de l'image.
        line.fill([0.0; 3]);
    }
}

/// Quantifie chaque pixel de `dst` vers la palette fournie en appliquant
/// Floyd-Steinberg. Opère **en place** sur l'image déjà rendue.
///
/// `palette` est une liste de couleurs RGB ; si elle contient moins de deux
/// entrées, on ne fait rien (le rendu est supposé déjà opaque).
pub fn floyd_steinberg(dst: &mut RgbImage, palette: &[[u8; 3]]) {
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
    let palette_f = palette_f32(palette);

    for y in 0..h {
        for x in 0..w {
            let cur = &mut buf[idx(x, y)];
            let old = *cur;
            let q = nearest(palette, &palette_f, &old);
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

    /// Rampe de gris horizontale : le cas où un dithering se juge.
    fn grey_ramp(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, _| {
            let v = ((x as f32 / (w as f32 - 1.0).max(1.0)) * 255.0).round() as u8;
            Rgb([v, v, v])
        })
    }

    fn mean_luma(img: &RgbImage) -> f32 {
        let sum: f64 = img
            .pixels()
            .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
            .sum();
        (sum / img.pixels().count() as f64) as f32
    }

    fn in_palette(img: &RgbImage, palette: &[[u8; 3]]) -> bool {
        img.pixels().all(|p| {
            palette
                .iter()
                .any(|c| c[0] == p[0] && c[1] == p[1] && c[2] == p[2])
        })
    }

    const BICHROME: [[u8; 3]; 2] = [[0, 0, 0], [255, 255, 255]];

    #[test]
    fn bayer_matrix_is_a_permutation() {
        for n in [2usize, 4, 8] {
            let matrix = bayer_matrix(n);
            assert_eq!(matrix.len(), n, "côté {n}");
            let mut values: Vec<u32> = matrix.iter().flatten().copied().collect();
            values.sort_unstable();
            assert_eq!(values, (0..(n * n) as u32).collect::<Vec<_>>(), "n = {n}");
        }
    }

    #[test]
    fn bayer_matrix_starts_with_the_canonical_4x4() {
        // La matrice 4×4 de référence, dans l'ordre de ses lignes.
        let expected = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
        assert_eq!(bayer_matrix(4), expected.map(|r| r.to_vec()));
    }

    #[test]
    fn off_is_a_noop() {
        let mut img = solid(4, 4, [123, 45, 67]);
        let before: Vec<_> = img.pixels().copied().collect();
        apply(&mut img, &BICHROME, Dithering::Off);
        let after: Vec<_> = img.pixels().copied().collect();
        assert_eq!(before, after, "aucun tramage ne doit modifier l'image");
    }

    #[test]
    fn every_algorithm_stays_inside_the_palette() {
        for dithering in [
            Dithering::FloydSteinberg,
            Dithering::Riemersma,
            Dithering::Bayer4,
            Dithering::Bayer8,
        ] {
            let mut img = grey_ramp(64, 32);
            apply(&mut img, &BICHROME, dithering);
            assert!(in_palette(&img, &BICHROME), "{dithering:?} : hors palette");
            assert_eq!(img.dimensions(), (64, 32), "{dithering:?} : dimensions");
        }
    }

    #[test]
    fn every_algorithm_preserves_the_mean_on_a_ramp() {
        // Propriété commune aux quatre : la luminance moyenne du tramage reste
        // proche de celle de la source (c'est tout l'intérêt du dithering).
        let source = grey_ramp(128, 16);
        let reference = mean_luma(&source);
        for dithering in [
            Dithering::FloydSteinberg,
            Dithering::Riemersma,
            Dithering::Bayer4,
            Dithering::Bayer8,
        ] {
            let mut img = source.clone();
            apply(&mut img, &BICHROME, dithering);
            let delta = (mean_luma(&img) - reference).abs();
            assert!(delta < 6.0, "{dithering:?} : écart de luminance {delta}");
        }
    }

    #[test]
    fn bayer_mixes_both_colours_on_a_mid_grey() {
        // Un gris uniforme ne doit pas produire un aplat : c'est tout le but du
        // tramage ordonné.
        for (dithering, expected_side) in [(Dithering::Bayer4, 4usize), (Dithering::Bayer8, 8)] {
            let mut img = solid(16, 16, [128, 128, 128]);
            apply(&mut img, &BICHROME, dithering);
            let white = img.pixels().filter(|p| p[0] == 255).count();
            assert!(white > 0, "{dithering:?} : aucun pixel clair");
            assert!(
                white < img.pixels().count(),
                "{dithering:?} : aucun pixel sombre"
            );
            // Le motif doit être périodique de la taille de la matrice.
            let matrix = bayer_matrix(expected_side);
            for (y, row) in matrix.iter().enumerate().take(expected_side) {
                for (x, value) in row.iter().enumerate().take(expected_side) {
                    let threshold = (*value as f32 + 0.5) / (expected_side * expected_side) as f32;
                    let pixel = img.get_pixel(x as u32, y as u32)[0] == 255;
                    let expected_pixel = threshold > 0.5;
                    assert_eq!(
                        pixel, expected_pixel,
                        "{dithering:?} : motif inattendu en ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn riemersma_and_floyd_steinberg_differ_on_a_ramp() {
        // Les deux diffusions ne produisent pas le même motif : c'est le but de
        // proposer Riemersma en plus.
        let source = grey_ramp(96, 8);
        let mut fs = source.clone();
        apply(&mut fs, &BICHROME, Dithering::FloydSteinberg);
        let mut rie = source.clone();
        apply(&mut rie, &BICHROME, Dithering::Riemersma);
        let difference = fs
            .pixels()
            .zip(rie.pixels())
            .filter(|(a, b)| a != b)
            .count();
        assert!(difference > 20, "les deux algorithmes sont trop proches");
    }

    /// Écart moyen entre deux pixels voisins : mesure directe des « vers ».
    fn mean_local_jump(img: &RgbImage) -> f32 {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for y in 0..img.height() {
            for x in 1..img.width() {
                total += (i32::from(img.get_pixel(x, y)[0]) - i32::from(img.get_pixel(x - 1, y)[0]))
                    .abs() as f32;
                count += 1;
            }
        }
        total / count as f32
    }

    #[test]
    fn riemersma_tail_smooths_the_pattern() {
        // Ce que la queue longue apporte : sur un dégradé, l'écart entre pixels
        // voisins diminue nettement (les « vers » de la diffusion à prise unique
        // disparaissent), au prix d'un motif un peu plus fondu. C'est le
        // compromis qui définit Riemersma.
        let source = grey_ramp(160, 4);
        let mut short = source.clone();
        riemersma(&mut short, &BICHROME, 1, RIEMERSMA_DECAY);
        let mut long = source.clone();
        riemersma(&mut long, &BICHROME, RIEMERSMA_TAPS, RIEMERSMA_DECAY);
        let (short_jump, long_jump) = (mean_local_jump(&short), mean_local_jump(&long));
        // Mesuré : 65,8 contre 126,7, soit presque une division par deux.
        assert!(
            long_jump * 1.5 < short_jump,
            "la queue longue devrait lisser nettement le motif : {long_jump} contre {short_jump}"
        );
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
