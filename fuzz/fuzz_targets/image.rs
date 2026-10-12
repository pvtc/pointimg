#![no_main]

use image::{GenericImageView, ImageReader, Limits};
use libfuzzer_sys::fuzz_target;
use pointimg::filter::{self, Algorithm, FilterParams, HalftoneMode, Screening};
use std::io::Cursor;
use std::sync::atomic::AtomicBool;

fuzz_target!(|data: &[u8]| {
    let Ok(mut reader) = ImageReader::new(Cursor::new(data)).with_guessed_format() else {
        return;
    };
    let mut limits = Limits::default();
    limits.max_image_width = Some(65_535);
    limits.max_image_height = Some(65_535);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    let Ok(image) = reader.decode() else {
        return;
    };
    let (width, height) = image.dimensions();
    if pointimg::filter::validate_image_dimensions(width, height).is_err() {
        return;
    }
    let rgb = pointimg::filter::flatten_to_rgb(&image, [255, 255, 255]);
    let _ = pointimg::filter::compute_density_image(&rgb, 0.7);

    // Shrink the image before the full pipeline: libFuzzer must stay fast,
    // and the render/SVG/halftone paths do not depend on the resolution.
    let small = image::imageops::resize(
        &rgb,
        rgb.width().min(64),
        rgb.height().min(64),
        image::imageops::FilterType::Triangle,
    );
    let cancel = AtomicBool::new(false);

    for algorithm in [
        Algorithm::Grid,
        Algorithm::Kmeans,
        Algorithm::Voronoi,
        Algorithm::Quadtree,
    ] {
        let params = FilterParams {
            algorithm,
            num_points: 64,
            cols: 8,
            iterations: 2,
            rng_seed: Some(1),
            ..FilterParams::default()
        };
        if let Ok((_, dots)) =
            filter::apply_with_progress(&small, &params, &cancel, |_, _, _| {})
        {
            // Also covers SVG rendering (shapes, coordinates, palette).
            let _ = filter::render_svg_from_dots(
                small.width(),
                small.height(),
                &dots,
                &params,
            );
        }
    }

    // Dominant halftone + FM (k-means separation, stochastic screening).
    let halftone = FilterParams {
        algorithm: Algorithm::Halftone,
        halftone: HalftoneMode::Dominant {
            n: 3,
            base_angle_deg: 15.0,
        },
        screening: Screening::Fm,
        rng_seed: Some(2),
        ..FilterParams::default()
    };
    let _ = filter::apply_rgba(&small, &halftone);
});
