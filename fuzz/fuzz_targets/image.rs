#![no_main]

use image::{GenericImageView, ImageReader, Limits};
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

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
});
