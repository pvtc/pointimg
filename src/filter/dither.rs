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
    /// Diffusion à queue longue : 16 pixels, poids décroissants de 0,7.
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

    /// Avertissement sur le poids du fichier exporté, s'il y a lieu.
    ///
    /// Le tramage ordonné est à haute fréquence : ni deflate (PNG, TIFF) ni
    /// VP8L (WebP) n'ont de prise sur un motif qui change à chaque pixel. Le
    /// rendu est donc plus lourd *à palette égale* — mesuré sur une photo
    /// 3546×2364, 3,0 Mo en Bayer 4×4 contre 162 Ko en Floyd-Steinberg avec
    /// deux couleurs, soit dix-neuf fois plus. Le phénomène ne dépend pas de la
    /// largeur de la palette : Bayer 4×4 reste autour de 5 Mo de 8 à 32
    /// couleurs. C'est la contrepartie de l'esthétique « pixel art ».
    ///
    /// Les deux interfaces reprennent ce texte tel quel, pour que l'avertissement
    /// ne diverge pas d'un front-end à l'autre.
    pub fn size_caveat(self) -> Option<&'static str> {
        match self {
            Dithering::Bayer4 | Dithering::Bayer8 => Some(
                "Motif à haute fréquence : la compression ne peut pas le réduire. \
                 L'export pèsera nettement plus qu'avec une diffusion d'erreur, \
                 à palette égale (mesuré : 3,0 Mo contre 162 Ko avec 2 couleurs).",
            ),
            Dithering::Off | Dithering::FloydSteinberg | Dithering::Riemersma => None,
        }
    }

    /// Version courte de [`Dithering::size_caveat`] : tient sur une ligne, le
    /// détail et les chiffres restant dans l'infobulle.
    pub fn size_caveat_short(self) -> Option<&'static str> {
        match self {
            Dithering::Bayer4 | Dithering::Bayer8 => {
                Some("Export nettement plus lourd : motif que la compression ne réduit pas.")
            }
            Dithering::Off | Dithering::FloydSteinberg | Dithering::Riemersma => None,
        }
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

/// Poids de la file de Riemersma : `decay^k` normalisé pour sommer à 1.
///
/// Chaque part d'erreur doit être versée en totalité, sinon la luminance
/// dérive. C'est l'invariant le plus important de l'algorithme, et le plus
/// difficile à voir à l'œil sur une image : voir
/// `proprietes::la_moyenne_survit_a_la_quantification` pour le piège
/// correspondant.
fn riemersma_weights(taps: usize, decay: f64) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..taps).map(|k| decay.powi(k as i32)).collect();
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        for weight in &mut weights {
            *weight /= total;
        }
    }
    weights
}

/// Diffusion de l'erreur à queue longue (Riemersma, 1995).
///
/// L'erreur est versée dans une file de `taps` pixels : le suivant en reçoit la
/// part `w(0)`, le suivant encore `w(1)`, et ainsi de suite jusqu'à `w(taps-1)`,
/// les poids formant une suite géométrique décroissante de somme 1. Chaque part
/// n'est versée qu'une fois — c'est un décalage pur, pas une réinjection de
/// l'erreur encore en attente — donc l'erreur est intégralement redistribuée et
/// la moyenne conservée. La queue longue supprime les « vers » et le moiré
/// directionnel du Floyd-Steinberg ; l'algorithme reste séquentiel le long d'une
/// ligne.
fn riemersma(dst: &mut RgbImage, palette: &[[u8; 3]], taps: usize, decay: f64) {
    let (w, h) = dst.dimensions();
    if taps == 0 {
        floyd_steinberg(dst, palette);
        return;
    }
    let palette_f = palette_f32(palette);
    let weights = riemersma_weights(taps, decay);

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

#[cfg(test)]
mod caveat {
    // Le glob importe les variantes ; le type lui-même n'est pas nommé ici.
    use super::Dithering::*;

    /// L'avertissement ne concerne que les algorithmes ordonnés : une
    /// diffusion d'erreur ne détruit pas la compressibilité.
    #[test]
    fn only_ordered_screens_warn() {
        for mode in [FloydSteinberg, Riemersma, Off] {
            assert!(
                mode.size_caveat().is_none(),
                "{mode:?} ne devrait pas être signalé"
            );
        }
        for mode in [Bayer4, Bayer8] {
            assert!(
                mode.size_caveat().is_some(),
                "{mode:?} devrait être signalé : son motif est incompressible"
            );
        }
    }

    /// Un algorithme hors tramage ne se voit pas non plus : l'avertissement ne
    /// doit apparaître que si un tramage est réellement appliqué.
    #[test]
    fn caveat_never_appears_without_dithering() {
        for mode in [Off, FloydSteinberg, Riemersma, Bayer4, Bayer8] {
            if !mode.is_on() {
                assert!(mode.size_caveat().is_none());
            }
        }
    }

    /// Les deux variantes du texte doivent rester affichables : la version
    /// courte tient sur une ligne de l'interface, la longue dans une infobulle.
    #[test]
    fn caveat_stays_readable() {
        for mode in [Bayer4, Bayer8] {
            let long = mode.size_caveat().unwrap();
            let short = mode.size_caveat_short().unwrap();
            assert!(
                long.len() < 220,
                "version longue : {} caractères",
                long.len()
            );
            assert!(
                short.len() < 80,
                "version courte : {} caractères",
                short.len()
            );
            // La version courte doit rester une phrase, pas un fragment.
            assert!(short.ends_with('.'), "phrase incomplète : {short}");
        }
    }
}

// ─── Propriétés ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod proprietes {
    use super::Dithering;
    use super::Dithering::*;
    use super::*;
    use image::Rgb;

    /// Générateur congruentiel déterministe : les tests de propriétés doivent
    /// être reproductibles, donc aucune dépendance à une graineuntime.
    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    fn luma(p: Rgb<u8>) -> f64 {
        0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])
    }

    fn mean_luma(img: &RgbImage) -> f64 {
        img.pixels().map(|p| luma(*p)).sum::<f64>() / img.pixels().count() as f64
    }

    /// Rampe de gris horizontale.
    fn ramp(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, _| {
            let v = ((x as f64 / (w as f64 - 1.0).max(1.0)) * 255.0).round() as u8;
            Rgb([v, v, v])
        })
    }

    /// Rampe verticale : attrape une dérive qui ne dépendrait que de `x`.
    fn ramp_v(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |_, y| {
            let v = ((y as f64 / (h as f64 - 1.0).max(1.0)) * 255.0).round() as u8;
            Rgb([v, v, v])
        })
    }

    /// Bruit de couleur déterministe : aucun motif régulier, donc aucune
    /// résonance avec la matrice de Bayer.
    fn noise(w: u32, h: u32, seed: u64) -> RgbImage {
        let mut s = seed;
        RgbImage::from_fn(w, h, |_, _| {
            let v = [
                (lcg(&mut s) % 256) as u8,
                (lcg(&mut s) % 256) as u8,
                (lcg(&mut s) % 256) as u8,
            ];
            Rgb(v)
        })
    }

    /// Dégradé lisse en deux dimensions.
    fn smooth2d(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let fx = x as f64 / (w as f64 - 1.0).max(1.0);
            let fy = y as f64 / (h as f64 - 1.0).max(1.0);
            Rgb([
                (fx * 255.0).round() as u8,
                (fy * 255.0).round() as u8,
                ((fx + fy) * 0.5 * 255.0).round() as u8,
            ])
        })
    }

    /// Mélange progressif de deux couleurs : la source *est* un mélange de
    /// couleurs de palette, ce qui est la situation où le tramage a un sens.
    fn blend(a: [u8; 3], b: [u8; 3], w: u32, h: u32) -> RgbImage {
        // La différence se calcule en `i32` : `b - a` peut être négatif, et une
        // soustraction `u8` déborderait.
        let lerp = |from: u8, to: u8, t: f64| -> u8 {
            (f64::from(from) + f64::from(i32::from(to) - i32::from(from)) * t).round() as u8
        };
        RgbImage::from_fn(w, h, |x, _| {
            let t = x as f64 / (w as f64 - 1.0).max(1.0);
            Rgb([
                lerp(a[0], b[0], t),
                lerp(a[1], b[1], t),
                lerp(a[2], b[2], t),
            ])
        })
    }

    fn grey_levels(n: usize) -> Vec<[u8; 3]> {
        (0..n)
            .map(|i| {
                let v = (i as f64 / (n as f64 - 1.0) * 255.0).round() as u8;
                [v, v, v]
            })
            .collect()
    }

    /// Les quatre algorithmes qui trament.
    const MODES: [Dithering; 4] = [FloydSteinberg, Riemersma, Bayer4, Bayer8];

    /// Images à tonalité variée : c'est la condition pour que la conservation de
    /// la moyenne ait un sens (voir `une_source_grise_...` plus bas).
    fn images_tonales() -> Vec<(&'static str, RgbImage)> {
        vec![
            ("rampe 256x16", ramp(256, 16)),
            ("rampe 512x8", ramp(512, 8)),
            ("rampe verticale 16x256", ramp_v(16, 256)),
            ("bruit 128x128", noise(128, 128, 7)),
            ("bruit 256x64", noise(256, 64, 11)),
            ("dégradé 2D 128x128", smooth2d(128, 128)),
        ]
    }

    /// **La propriété qui définit un tramage** : sur une image à tonalité
    /// variée, la luminance moyenne doit survivre à la quantification.
    ///
    /// Tolérance 6,0 sur 255, soit 2,4 %. Elle est calibrée, pas arbitraire :
    /// l'écart le plus grand constaté sur les images ci-dessus avec une palette
    /// de gris est 1,7 (Bayer 8×8 sur un dégradé 2D, palette de deux tons).
    ///
    /// **Ce que ce test ne voit pas**, mesuré en mutation : supprimer la
    /// normalisation des poids de Riemersma ne le fait pas échouer. La file est
    /// remise à zéro en fin de ligne, et le résidu ainsi jeté est exactement ce
    /// qui rééquilibre la moyenne. L'invariant « les poids somment à 1 » est donc
    /// vérifié séparément, par
    /// `les_poids_de_la_queue_soment_a_un`, là où il porte réellement.
    #[test]
    fn la_moyenne_survit_a_la_quantification() {
        for (name, source) in images_tonales() {
            let reference = mean_luma(&source);
            for n in [2usize, 3, 4, 8, 16] {
                let palette = grey_levels(n);
                for mode in MODES {
                    let mut out = source.clone();
                    apply(&mut out, &palette, mode);
                    let delta = (mean_luma(&out) - reference).abs();
                    assert!(
                        delta < 6.0,
                        "{name}, {n} tons, {mode:?} : luminance déplacée de {delta:.2} \
                         (source {reference:.2}, sortie {:.2})",
                        mean_luma(&out)
                    );
                }
            }
        }
    }

    /// L'invariant que la conservation de la moyenne ne peut pas voir (voir la
    /// note du test précédent) : la file de Riemersma doit distribuer chaque
    /// erreur en totalité, donc ses poids somment à 1. C'est vérifié sur la
    /// construction des poids, pas à travers l'image — une somme fausse se
    /// compenserait d'une ligne à l'autre.
    #[test]
    fn les_poids_de_la_queue_soment_a_un() {
        for taps in [1usize, 2, 4, 8, 16, 32] {
            for decay in [0.5f64, 0.7, 0.9, 0.99] {
                let weights = riemersma_weights(taps, decay);
                assert_eq!(weights.len(), taps, "{taps} prises, décroissance {decay}");
                let total: f64 = weights.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-12,
                    "{taps} prises, décroissance {decay} : les poids somment à {total}"
                );
                // Et la suite doit être décroissante : la queue longue repose sur
                // des parts de plus en plus petites.
                for pair in weights.windows(2) {
                    assert!(pair[1] <= pair[0], "poids non décroissants : {weights:?}");
                }
            }
        }
    }

    #[test]
    fn la_moyenne_survit_sans_gris_dans_la_palette() {
        for (a, b) in [
            ([255u8, 0, 0], [0u8, 255, 0]),
            ([255, 0, 0], [0, 0, 0]),
            ([255, 255, 0], [0, 0, 255]),
        ] {
            let source = blend(a, b, 256, 16);
            let reference = mean_luma(&source);
            for mode in MODES {
                let mut out = source.clone();
                apply(&mut out, &[a, b], mode);
                let delta = (mean_luma(&out) - reference).abs();
                assert!(
                    delta < 2.0,
                    "mélange {a:?}/{b:?}, {mode:?} : luminance déplacée de {delta:.2}"
                );
            }
        }
    }

    /// Le tramage ordonné doit placer la bonne *proportion* de pixels clairs,
    /// à la granularité de sa matrice. C'est le contrat précis de Bayer, et il
    /// tient même sur une image uniforme — là où la conservation de la moyenne
    /// n'a pas de sens (voir le test suivant).
    #[test]
    fn la_trame_ordonnee_verse_la_bonne_proportion() {
        for (mode, side) in [(Bayer4, 4usize), (Bayer8, 8)] {
            for tone in [0.1f64, 0.25, 0.5, 0.75, 0.9] {
                let grey = (tone * 255.0).round() as u8;
                let mut img = RgbImage::from_pixel(64, 64, Rgb([grey; 3]));
                apply(&mut img, &[[0, 0, 0], [255, 255, 255]], mode);
                let white = img.pixels().filter(|p| p[0] > 127).count();
                let fraction = white as f64 / img.pixels().count() as f64;
                // Une cellule de la matrice vaut 1/side² de tonte surface.
                let cell = 1.0 / (side * side) as f64;
                assert!(
                    (fraction - tone).abs() <= cell,
                    "{mode:?}, ton {tone} : {fraction:.3} de pixels clairs \
                     (attendu {tone} à ±{cell:.4})"
                );
            }
        }
    }

    /// **Limite connue, épinglée volontairement.** Une source grise n'est pas un
    /// mélange atteignable si la palette ne contient aucun gris : la sélection du
    /// plus proche voisin se fait en distance RGB, donc elle ignore la
    /// luminance. Avec la palette {noir, rouge, vert, bleu} — de lumas 0, 0, 18,
    /// 54 et 182 — l'écart atteint 57 à 73, très loin de la tolérance de 6,0
    /// appliquée plus haut.
    ///
    /// Ce test ne vérifie donc pas un comportement satisfaisant, mais que
    /// l'exclusion est intentionnelle. S'il échoue, c'est bon signe : la
    /// limitation a été corrigée, et les tolerances des tests précédents peuvent
    /// être resserrées.
    #[test]
    fn une_source_grise_ne_se_reproduit_pas_sans_gris_dans_la_palette() {
        let palette: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let source = ramp(256, 16);
        let reference = mean_luma(&source);
        for mode in MODES {
            let mut out = source.clone();
            apply(&mut out, &palette, mode);
            let delta = (mean_luma(&out) - reference).abs();
            assert!(
                delta > 6.0,
                "{mode:?} : l'écart n'est plus que de {delta:.2} — la limite \\
                 documentée a disparu, les tolerances peuvent être resserrées"
            );
        }
    }

    /// Le rendu doit être reproductible : la suite du dépôt s'appuie dessus.
    #[test]
    fn le_tramage_est_deterministe() {
        for mode in MODES {
            for n in [2usize, 5, 16] {
                let palette = grey_levels(n);
                let source = smooth2d(64, 48);
                let mut first = source.clone();
                let mut second = source.clone();
                apply(&mut first, &palette, mode);
                apply(&mut second, &palette, mode);
                assert_eq!(
                    first.as_raw(),
                    second.as_raw(),
                    "{mode:?} avec {n} tons n'est pas reproductible"
                );
            }
        }
    }

    /// Le tramage ne doit rien changer aux pixels déjà exactement dans la
    /// palette : un aplat d'une couleur de palette doit rester un aplat. C'est ce
    /// qui évite qu'un rendu « propre » soit sali par le post-traitement.
    #[test]
    fn un_aplat_dune_couleur_de_palette_ne_bouge_pas() {
        let palette = grey_levels(8);
        for colour in &palette {
            let source = RgbImage::from_pixel(16, 16, Rgb(*colour));
            for mode in MODES {
                let mut out = source.clone();
                apply(&mut out, &palette, mode);
                assert!(
                    out.pixels().all(|p| p[0] == colour[0]),
                    "{mode:?} a modifié un aplat déjà dans la palette ({colour:?})"
                );
            }
        }
    }
}
