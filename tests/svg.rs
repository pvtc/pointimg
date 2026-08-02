//! Tests de validité du rendu SVG via l'API publique.

use pointimg::filter::{self, Algorithm, DotShape, FilterParams};
use std::sync::atomic::AtomicBool;

fn checkerboard(w: u32, h: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        if (x + y) % 2 == 0 {
            image::Rgb([255, 255, 255])
        } else {
            image::Rgb([0, 0, 0])
        }
    })
}

#[test]
fn render_svg_from_dots_has_header_and_bg() {
    let img = checkerboard(20, 16);
    let cancel = AtomicBool::new(false);
    let params = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 8,
        rng_seed: Some(1),
        ..FilterParams::default()
    };
    let (_, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    let svg = filter::render_svg_from_dots(20, 16, &dots, &params).unwrap();

    assert!(svg.starts_with("<svg"), "doit commencer par <svg");
    assert!(svg.contains("</svg>"), "doit finir par </svg>");
    assert!(
        svg.contains(r#"width="20""#) && svg.contains(r#"height="16""#),
        "dimensions encodées"
    );
    assert!(
        svg.contains("#ffffff"),
        "couleur de fond white doit apparaître"
    );
}

#[test]
fn render_svg_emits_shape_elements() {
    let img = checkerboard(24, 24);
    let cancel = AtomicBool::new(false);

    for (shape, needle) in [
        (DotShape::Circle, "<circle"),
        (DotShape::Square, "<rect"),
        (
            DotShape::Ellipse {
                aspect: 2.0,
                angle_deg: 30.0,
            },
            "<ellipse",
        ),
        (DotShape::RegularPolygon { sides: 6 }, "<polygon"),
    ] {
        let params = FilterParams {
            algorithm: Algorithm::Grid,
            cols: 6,
            dot_shape: shape,
            rng_seed: Some(2),
            ..FilterParams::default()
        };
        let (_, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
        let svg = filter::render_svg_from_dots(24, 24, &dots, &params).unwrap();
        assert!(
            svg.contains(needle),
            "shape {:?} doit émettre {needle}",
            shape
        );
    }
}

#[test]
fn render_svg_dynamic_path_uses_bg() {
    use image::{DynamicImage, Rgba, RgbaImage};
    let rgba = RgbaImage::from_pixel(10, 10, Rgba([0, 128, 255, 255]));
    let dyn_img = DynamicImage::ImageRgba8(rgba);
    let params = FilterParams {
        algorithm: Algorithm::Grid,
        cols: 4,
        bg_color: [10, 20, 30],
        ..FilterParams::default()
    };
    let svg = filter::render_svg_dynamic(&dyn_img, &params).unwrap();
    assert!(svg.contains("#0a141e"), "bg hex #0a141e doit être présent");
}

#[test]
fn render_svg_empty_image_errors() {
    let params = FilterParams::default();
    let err = filter::render_svg_from_dots(0, 0, &[], &params);
    assert!(err.is_err(), "image vide doit échouer");
}
