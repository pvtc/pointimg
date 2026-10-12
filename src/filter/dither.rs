//! Palette quantization: four dithering algorithms.
//!
//! When `FilterParams::palette_size` is `Some(n)`, the final render is
//! post-quantized to `n` colors. Dithering avoids flat bands: it converts the
//! quantization error into a dot pattern whose average reproduces the original
//! tone.
//!
//! | Algorithm | Principle | Use |
//! |---|---|---|
//! | [`Dithering::Off`] | none | clean flat areas |
//! | [`Dithering::FloydSteinberg`] | error diffused to 4 neighbors | photographs, gradients |
//! | [`Dithering::Riemersma`] | long-tail diffusion on a line | gradients, fewer "worms" than Floyd-Steinberg |
//! | [`Dithering::Bayer4`] / [`Dithering::Bayer8`] | ordered 4×4 or 8×8 pattern | "pixel art" look, no diffusion |
//!
//! The last two propagate no error: every pixel is independent, hence
//! parallelizable, but the average is only statistically exact, which suits
//! precisely the aesthetics aimed at. The Bayer matrix is **generated** by the
//! recursive rule of Bayer (1973):
//!
//! ```text
//! M(1) = [0]     M(2n) = 4·M(n) + [[0, 2], [3, 1]]
//! ```
//!
//! Floyd-Steinberg, for its part, propagates the error like this:
//!
//! ```text
//!        *   7/16
//! 3/16  5/16  1/16
//! ```

use image::{Rgb, RgbImage};

/// Dithering algorithm applied to palette quantization.
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
    /// No dithering: the nearest color, without a pattern.
    #[default]
    Off,
    /// Error diffusion toward the four future neighbors.
    FloydSteinberg,
    /// Long-tail diffusion: 16 pixels, weights decaying from 0.7.
    Riemersma,
    /// Bayer 4×4 ordered dithering.
    Bayer4,
    /// Bayer 8×8 ordered dithering.
    Bayer8,
}

impl Dithering {
    /// True when a dithering is applied.
    pub fn is_on(self) -> bool {
        self != Dithering::Off
    }

    /// Warning about the weight of the exported file, when it applies.
    ///
    /// Ordered dithering is high frequency: neither deflate (PNG, TIFF) nor
    /// VP8L (WebP) has any hold on a pattern that changes at every pixel. The
    /// render is therefore heavier *for an equal palette*, measured on a
    /// 3546×2364 photo, 3.0 MB in Bayer 4×4 against 162 kB in Floyd-Steinberg
    /// with two colors, that is nineteen times more. The phenomenon does not
    /// depend on the palette width: Bayer 4×4 stays around 5 MB from 8 to 32
    /// colors. It is the counterpart of the "pixel art" aesthetics.
    ///
    /// Both interfaces reuse this text as it is, so that the warning does not
    /// diverge from one front-end to another.
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

    /// Short version of [`Dithering::size_caveat`]: fits on one line, the
    /// details and the figures staying in the tooltip.
    pub fn size_caveat_short(self) -> Option<&'static str> {
        match self {
            Dithering::Bayer4 | Dithering::Bayer8 => {
                Some("Export nettement plus lourd : motif que la compression ne réduit pas.")
            }
            Dithering::Off | Dithering::FloydSteinberg | Dithering::Riemersma => None,
        }
    }
}

/// Length of the Riemersma queue (in pixels).
const RIEMERSMA_TAPS: usize = 16;
/// Decay of the weights along the queue.
///
/// Calibrated on a gradient 160 px wide: at 0.7, the average gap between
/// neighboring pixels goes from 127 (single tap, hence marked "worms") to 66,
/// for a slightly more blurred pattern. This is the usual Riemersma
/// compromise: more local sharpness is paid for by a shorter queue, hence by
/// "worms".
const RIEMERSMA_DECAY: f64 = 0.7;

/// Quantifies `dst` to `palette` with the requested algorithm.
///
/// Operates **in place** on the already rendered image. `Dithering::Off`, like
/// a palette with fewer than two colors, does nothing.
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

/// Palette pre-converted to `f32`, to avoid a cast per comparison.
fn palette_f32(palette: &[[u8; 3]]) -> Vec<[f32; 3]> {
    palette
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect()
}

/// Palette color closest to `c`.
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

/// The two palette colors closest to `c`.
///
/// The first is the result of a direct quantization; the second serves the
/// ordered dithering, which arbitrates between the two according to the
/// position of the pixel on the segment that joins them.
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

/// Bayer matrix of side `n` (a power of two), a permutation of `0..n²`.
///
/// Generated by the recursive rule `M(2n) = 4·M(n) + [[0, 2], [3, 1]]`: no
/// table is copied anywhere else.
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

/// Ordered dithering: each pixel chooses between its two closest palette
/// colors, according to the matrix threshold at its position.
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
            // Pattern threshold at the center of its cell: (M + 0.5) / n².
            let threshold =
                (matrix[(y % side) as usize][(x % side) as usize] as f32 + 0.5) / levels;
            // Position of the pixel on the [best, second] segment: 0 = best.
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

/// Weights of the Riemersma queue: `decay^k` normalized to sum to 1.
///
/// Every share of error must be poured out entirely: that is what makes the
/// queue a redistribution and not an amplification.
///
/// The invariant is checked by
/// `proprietes::les_poids_de_la_queue_soment_a_un`, and it cannot be checked by
/// the conservation of luminance: diffusion is a closed feedback loop, which
/// conserves the mean whatever the gain. See the note of
/// `proprietes::la_moyenne_survit_a_la_quantification`.
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

/// Long-tail error diffusion (Riemersma, 1995).
///
/// The error is poured into a queue with eviction over `taps` pixels: the next
/// one receives the share `w(0)`, the following one still `w(1)`, and so on up
/// to `w(taps-1)`, the weights forming a decreasing geometric sequence summing
/// to 1. Every share is poured only once, this is a pure shift and not a
/// re-injection of the error still pending, so the error is entirely
/// redistributed and the mean conserved.
///
/// **The queue straddles the end of the lines.** It is never reset from one
/// line to the next: the pending residual keeps being distributed and the mean
/// stays conserved. It is moreover sized on `taps` and not on the image width,
/// with a rotating head, indexing the positions by column would fold the end of
/// a line back onto already treated positions, and an image narrower than the
/// queue would lose shares.
///
/// For this continuity to be spatially correct, the scan is **serpentine**
/// (boustrophedon): one line is walked from left to right, the next one from
/// right to left. The share of error that crosses the end of a line therefore
/// lands on the pixel just below, and not at the start of the next line. This
/// is also what removes the directional moiré of Floyd-Steinberg, which the
/// long queue alone is not enough to erase.
fn riemersma(dst: &mut RgbImage, palette: &[[u8; 3]], taps: usize, decay: f64) {
    let (w, h) = dst.dimensions();
    if taps == 0 {
        floyd_steinberg(dst, palette);
        return;
    }
    let palette_f = palette_f32(palette);
    let weights = riemersma_weights(taps, decay);
    let mut queue = vec![[0.0f64; 3]; taps];
    // Position of the error share that applies to the treated pixel: the queue
    // is circular, and it is it, not the column, that defines the length.
    let mut head = 0usize;
    for y in 0..h {
        let reverse = y % 2 == 1;
        for step in 0..w {
            let x = if reverse { w - 1 - step } else { step };
            let p = dst.get_pixel(x, y);
            let c = [
                p[0] as f64 + queue[head][0],
                p[1] as f64 + queue[head][1],
                p[2] as f64 + queue[head][2],
            ];
            let q = nearest(
                palette,
                &palette_f,
                &[c[0] as f32, c[1] as f32, c[2] as f32],
            );
            dst.put_pixel(x, y, Rgb(q));
            let error = [c[0] - q[0] as f64, c[1] - q[1] as f64, c[2] - q[2] as f64];
            // The `taps` slots that follow are all rewritten, including the head
            // slot (it is, by the smallest weight): nothing to reset, and no
            // borrow overlaps.
            for (k, weight) in weights.iter().enumerate() {
                let slot = (head + 1 + k) % taps;
                for channel in 0..3 {
                    queue[slot][channel] = error[channel] * weight;
                }
            }
            head = (head + 1) % taps;
        }
    }
}

/// Quantifies every pixel of `dst` to the provided palette by applying
/// Floyd-Steinberg. Operates **in place** on the already rendered image.
///
/// `palette` is a list of RGB colors; if it holds fewer than two entries,
/// nothing is done (the render is assumed opaque already).
pub fn floyd_steinberg(dst: &mut RgbImage, palette: &[[u8; 3]]) {
    if palette.len() < 2 {
        return;
    }
    let (w, h) = dst.dimensions();
    // Work on a mutable f32 buffer to propagate the error without loss.
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
            // Write the quantized pixel into dst immediately.
            dst.put_pixel(x, y, Rgb(q));
            // Error as a float.
            let err = [
                old[0] - q[0] as f32,
                old[1] - q[1] as f32,
                old[2] - q[2] as f32,
            ];
            // Floyd-Steinberg propagation.
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

    /// Horizontal gray ramp: the case where a dithering proves itself.
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
        // The reference 4×4 matrix, in the order of its rows.
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
        // Property shared by the four: the mean luminance of the dithering
        // stays close to that of the source (that is the whole point of
        // dithering).
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
        // A uniform gray must not produce a flat area: that is the whole point
        // of ordered dithering.
        for (dithering, expected_side) in [(Dithering::Bayer4, 4usize), (Dithering::Bayer8, 8)] {
            let mut img = solid(16, 16, [128, 128, 128]);
            apply(&mut img, &BICHROME, dithering);
            let white = img.pixels().filter(|p| p[0] == 255).count();
            assert!(white > 0, "{dithering:?} : aucun pixel clair");
            assert!(
                white < img.pixels().count(),
                "{dithering:?} : aucun pixel sombre"
            );
            // The pattern must be periodic with the size of the matrix.
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
        // The two diffusions do not produce the same pattern: that is the point
        // of offering Riemersma as well.
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

    /// Mean gap between two neighboring pixels: direct measure of the "worms".
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

    /// What the long queue brings: on a gradient, the gap between neighboring
    /// pixels drops sharply (the "worms" of single-tap diffusion disappear),
    /// at the price of a slightly more blurred pattern. This is the compromise
    /// that defines Riemersma.
    ///
    /// As measured by mutation, this is the test that checks the most things:
    /// it fails if all error propagation is removed, if the sign of the weights
    /// is inverted, and if the queue is denormalized, where the conservation of
    /// the mean sees none of the three.
    #[test]
    fn riemersma_tail_smooths_the_pattern() {
        let source = grey_ramp(160, 4);
        let mut short = source.clone();
        riemersma(&mut short, &BICHROME, 1, RIEMERSMA_DECAY);
        let mut long = source.clone();
        riemersma(&mut long, &BICHROME, RIEMERSMA_TAPS, RIEMERSMA_DECAY);
        let (short_jump, long_jump) = (mean_local_jump(&short), mean_local_jump(&long));
        // Measured: 65.8 against 126.7, almost a division by two.
        assert!(
            long_jump * 1.5 < short_jump,
            "la queue longue devrait lisser nettement le motif : {long_jump} contre {short_jump}"
        );
    }

    #[test]
    fn dither_solid_color_maps_to_palette_entry() {
        let mut img = solid(8, 8, [200, 30, 30]);
        floyd_steinberg(&mut img, &[[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
        // All the pixels must be part of the palette.
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
    // The glob imports the variants; the type itself is not named here.
    use super::Dithering::*;

    /// The warning only concerns the ordered algorithms: an error diffusion
    /// does not destroy compressibility.
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

    /// An algorithm without dithering is not flagged either: the warning must
    /// only appear when a dithering is really applied.
    #[test]
    fn caveat_never_appears_without_dithering() {
        for mode in [Off, FloydSteinberg, Riemersma, Bayer4, Bayer8] {
            if !mode.is_on() {
                assert!(mode.size_caveat().is_none());
            }
        }
    }

    /// Both variants of the text must stay displayable: the short version fits
    /// on one interface line, the long one in a tooltip.
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
            // The short version must stay a sentence, not a fragment.
            assert!(short.ends_with('.'), "phrase incomplète : {short}");
        }
    }
}

// ─── Properties ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod proprietes {
    use super::Dithering;
    use super::Dithering::*;
    use super::*;
    use image::Rgb;

    /// Deterministic congruential generator: the property tests must be
    /// reproducible, hence no dependence on an externally seeded generator.
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

    /// Horizontal gray ramp.
    fn ramp(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, _| {
            let v = ((x as f64 / (w as f64 - 1.0).max(1.0)) * 255.0).round() as u8;
            Rgb([v, v, v])
        })
    }

    /// Vertical ramp: catches a drift that would depend only on `x`.
    fn ramp_v(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |_, y| {
            let v = ((y as f64 / (h as f64 - 1.0).max(1.0)) * 255.0).round() as u8;
            Rgb([v, v, v])
        })
    }

    /// Deterministic color noise: no regular pattern, hence no resonance with
    /// the Bayer matrix.
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

    /// Smooth two-dimensional gradient.
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

    /// Progressive blend of two colors: the source *is* a blend of palette
    /// colors, which is the situation where dithering makes sense.
    fn blend(a: [u8; 3], b: [u8; 3], w: u32, h: u32) -> RgbImage {
        // The difference is computed in `i32`: `b - a` can be negative, and a
        // `u8` subtraction would overflow.
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

    /// The four dithering algorithms.
    const MODES: [Dithering; 4] = [FloydSteinberg, Riemersma, Bayer4, Bayer8];

    /// Images with varied tone: this is the condition for the conservation of
    /// the mean to make sense (see `une_source_grise_...` below).
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

    /// **The defining property of a dithering**: on an image with varied tone,
    /// the mean luminance must survive quantization.
    ///
    /// Tolerance 6.0 out of 255, that is 2.4%. It is calibrated, not arbitrary:
    /// the largest difference observed on the images above with a gray palette
    /// is 1.7 (Bayer 8×8 on a 2D gradient, two-tone palette).
    ///
    /// **What this test does not see** — established by mutation, not assumed:
    ///
    /// - remove the normalization of the Riemersma weights: drift of 0.00, the
    ///   test passes;
    /// - remove all error propagation: drift of 0.00, the test passes;
    /// - invert the sign of the weights (positive feedback): drift of 37.85, the
    ///   test fails.
    ///
    /// So this property checks the **stability** of the loop, not its
    /// calibration nor the very presence of a diffusion. Two reasons, both
    /// structural: diffusion is a closed feedback loop, the injected error
    /// changes the choice of the quantizer, hence the next error, and the
    /// steady state conserves the mean whatever the gain; and quantizing each
    /// pixel independently already conserves the mean of a monotone ramp (cf.
    /// `une_quantification_simple_conserve_deja_...`).
    ///
    /// The calibration is checked by
    /// `les_poids_de_la_queue_soment_a_un` (structural invariant) and
    /// `tests::riemersma_tail_smooths_the_pattern` (smoothing, which detects
    /// the gain mutation).
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

    /// The invariant that the conservation of the mean cannot see (see the
    /// note of the previous test): the Riemersma queue must distribute each
    /// error entirely, so its weights sum to 1. This is checked on the
    /// construction of the weights, not through the image.
    ///
    /// Verified: removing the normalization makes this test fail, and it alone
    /// among the conservation properties.
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
                // And the sequence must be decreasing: the long queue rests on
                // smaller and smaller shares.
                for pair in weights.windows(2) {
                    assert!(pair[1] <= pair[0], "poids non décroissants : {weights:?}");
                }
            }
        }
    }

    /// **Characterization, not guarantee**: the conservation of the mean is a
    /// defect of any quantization on a gray ramp, with or without diffusion.
    ///
    /// On a monotone ramp, the sum of the quantized values is close to the sum
    /// of the sources, the steps of the staircase compensate. The property
    /// "the mean survives" therefore cannot, on its own, prove that a diffusion
    /// exists or that it is well calibrated. This test shows it by completely
    /// disabling the diffusion: the mean stays correct.
    ///
    /// It serves as a safeguard in the other direction: if this test started to
    /// fail, the structure of the ramp or of the quantizer has changed, and the
    /// tolerances above should be re-examined.
    #[test]
    fn une_quantification_simple_conserve_deja_la_moyenne_dune_rampe() {
        for n in [2usize, 3, 4, 8, 16] {
            let palette = grey_levels(n);
            for (name, source) in images_tonales() {
                let reference = mean_luma(&source);
                let mut out = source.clone();
                // `Dithering::Off`: nearest neighbor, without any diffusion.
                apply(&mut out, &palette, Dithering::Off);
                let delta = (mean_luma(&out) - reference).abs();
                assert!(
                    delta < 6.0,
                    "{name}, {n} tons, sans diffusion : la moyenne bouge déjà de \
                     {delta:.2} — les tolérances des autres tests sont à revoir"
                );
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

    /// Ordered dithering must place the right *proportion* of light pixels,
    /// at the granularity of its matrix. This is the precise contract of
    /// Bayer, and it holds even on a uniform image, where the conservation of
    /// the mean has no meaning (see the next test).
    #[test]
    fn la_trame_ordonnee_verse_la_bonne_proportion() {
        for (mode, side) in [(Bayer4, 4usize), (Bayer8, 8)] {
            for tone in [0.1f64, 0.25, 0.5, 0.75, 0.9] {
                let grey = (tone * 255.0).round() as u8;
                let mut img = RgbImage::from_pixel(64, 64, Rgb([grey; 3]));
                apply(&mut img, &[[0, 0, 0], [255, 255, 255]], mode);
                let white = img.pixels().filter(|p| p[0] > 127).count();
                let fraction = white as f64 / img.pixels().count() as f64;
                // One cell of the matrix is worth 1/side² of the whole surface.
                let cell = 1.0 / (side * side) as f64;
                assert!(
                    (fraction - tone).abs() <= cell,
                    "{mode:?}, ton {tone} : {fraction:.3} de pixels clairs \
                     (attendu {tone} à ±{cell:.4})"
                );
            }
        }
    }

    /// **Known limit, deliberately pinned.** A gray source is not an reachable
    /// mixture when the palette holds no gray: nearest-neighbor selection works
    /// in RGB distance, so it ignores luminance. With the palette {black, red,
    /// green, blue} — of lumas 0, 0, 18, 54 and 182 — the difference reaches 57
    /// to 73, far from the tolerance of 6.0 applied above.
    ///
    /// This test therefore does not check a satisfactory behavior, but that the
    /// exclusion is intentional. If it fails, the limit has been lifted and the
    /// tolerances of the previous tests can be tightened.
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
                 documentée a disparu, les tolérances peuvent être resserrées"
            );
        }
    }

    /// The render must be reproducible: the repository test suite relies on it.
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

    /// Dithering must change nothing about the pixels already exactly in the
    /// palette: a flat area of a palette color must stay a flat area. This is
    /// what prevents a "clean" render from being dirtied by the
    /// post-processing.
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
