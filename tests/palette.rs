//! Quantification de palette : le rendu SVG réduit le nombre de couleurs.

use pointimg::filter::{self, Algorithm, DotShape, FilterParams};
use std::sync::atomic::AtomicBool;

fn gradient_rgb(w: u32, h: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            (x * 255 / w.max(1)) as u8,
            (y * 255 / h.max(1)) as u8,
            ((x + y) * 255 / (w + h).max(1)) as u8,
        ])
    })
}

/// Compte les valeurs `fill="#rrggbb"` distinctes dans un SVG.
fn distinct_fill_colors(svg: &str) -> usize {
    svg.lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("  <")?;
            let rest = rest.strip_prefix("circle")?;
            rest.find("fill=\"").map(|i| {
                let s = &rest[i + 6..i + 6 + 7];
                s.to_string()
            })
        })
        .collect::<std::collections::HashSet<_>>()
        .len()
}

#[test]
fn palette_reduces_distinct_colors_in_svg() {
    let img = gradient_rgb(120, 120);
    let cancel = AtomicBool::new(false);

    let base = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 16,
        dot_shape: DotShape::Circle,
        rng_seed: Some(1),
        ..FilterParams::default()
    };
    let quantized = FilterParams {
        palette_size: Some(4),
        ..base.clone()
    };

    let (w, h) = img.dimensions();
    let (_, dots) = filter::apply_with_progress(&img, &base, &cancel, |_, _, _| {}).unwrap();
    let svg_full = filter::render_svg_from_dots(w, h, &dots, &base).unwrap();
    let svg_pal = filter::render_svg_from_dots(w, h, &dots, &quantized).unwrap();

    let n_full = distinct_fill_colors(&svg_full);
    let n_pal = distinct_fill_colors(&svg_pal);
    assert!(
        n_pal <= 4,
        "palette=4 doit produire ≤ 4 fill distincts, got {n_pal}"
    );
    assert!(
        n_pal < n_full,
        "palette doit réduire les fill : {n_pal} vs {n_full}"
    );
}
