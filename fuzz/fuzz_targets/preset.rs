#![no_main]

use libfuzzer_sys::fuzz_target;
use pointimg::filter::{self, FilterParams};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(params) = FilterParams::from_toml_str(text) else {
        return;
    };

    // Exercise both deserialization and the public validation path.
    let image = image::RgbImage::new(8, 8);
    let _ = filter::apply(&image, &params);
});
