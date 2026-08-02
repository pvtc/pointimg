//! Benches criterion pour les algorithmes de pointimg.
//!
//! Lance avec : `cargo bench` (compile puis exécute criterion).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use image::RgbImage;
use pointimg::filter::{self, Algorithm, FilterParams};
use std::sync::atomic::AtomicBool;

/// Image de test 400×400 : damier multi-couleurs + dégradé pour avoir
/// une densité de détail variée.
fn bench_image() -> RgbImage {
    let (w, h) = (400u32, 400u32);
    RgbImage::from_fn(w, h, |x, y| {
        let dx = (x / 40) % 4;
        let dy = (y / 40) % 4;
        match (dx + dy) % 4 {
            0 => image::Rgb([255, 0, 0]),
            1 => image::Rgb([0, 255, 0]),
            2 => image::Rgb([0, 0, 255]),
            _ => image::Rgb([
                (x as u8).wrapping_add(y as u8),
                (x as u8).wrapping_sub(y as u8),
                128,
            ]),
        }
    })
}

fn default_for(algo: Algorithm) -> FilterParams {
    FilterParams {
        algorithm: algo,
        num_points: 1000,
        cols: 60,
        iterations: 5,
        rng_seed: Some(99),
        ..FilterParams::default()
    }
}

fn bench_apply(c: &mut Criterion) {
    let img = bench_image();

    let mut group = c.benchmark_group("apply");
    for algo in [
        Algorithm::Grid,
        Algorithm::Kmeans,
        Algorithm::Voronoi,
        Algorithm::Quadtree,
    ] {
        let params = default_for(algo);
        group.bench_function(format!("{algo:?}"), |b| {
            b.iter(|| {
                let p = black_box(&params);
                filter::apply(black_box(&img), p).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_density_map(c: &mut Criterion) {
    let img = bench_image();
    c.bench_function("compute_density_image", |b| {
        b.iter(|| filter::compute_density_image(black_box(&img), 0.7));
    });
}

fn bench_svg(c: &mut Criterion) {
    let img = bench_image();
    let cancel = AtomicBool::new(false);
    let params = default_for(Algorithm::Voronoi);
    let (_, dots) = filter::apply_with_progress(&img, &params, &cancel, |_, _, _| {}).unwrap();
    let (w, h) = img.dimensions();

    c.bench_function("render_svg_from_dots", |b| {
        b.iter(|| {
            filter::render_svg_from_dots(w, h, black_box(&dots), black_box(&params)).unwrap();
        });
    });
}

criterion_group!(benches, bench_apply, bench_density_map, bench_svg);
criterion_main!(benches);
