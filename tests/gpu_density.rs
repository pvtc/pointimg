//! Parité CPU/GPU de la density map (feature `gpu`).
//!
//! Le test se désactive proprement si aucun adaptateur GPU n'est disponible
//! (CI sans GPU, drivers absents) : il vérifie d'abord `gpu_available()`.
//! Il compare directement `gpu::compute_density_map_raw` au chemin CPU, sans
//! dépendre de la variable d'environnement d'opt-in `POINTIMG_GPU`.

use image::RgbImage;
use pointimg::filter::{compute_density_map_cpu, gpu};

fn gradient_image(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let block = ((x / 7 + y / 5) % 4) as u8;
        let v = block * 60 + ((x * 3 + y * 5) % 64) as u8;
        image::Rgb([v, 255u8.saturating_sub(v), (v / 2).min(255)])
    })
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn gpu_density_map_matches_cpu() {
    if !gpu::gpu_available() {
        eprintln!("skip : aucun adaptateur GPU disponible");
        return;
    }
    let img = gradient_image(128, 96);
    for sensitivity in [0.0f32, 0.7, 1.0] {
        let cpu = compute_density_map_cpu(&img, sensitivity);
        let gpu_map = gpu::compute_density_map_raw(&img, sensitivity)
            .expect("le chemin GPU doit fonctionner quand un adaptateur est présent");
        assert_eq!(cpu.len(), gpu_map.len());
        let diff = max_abs_diff(&cpu, &gpu_map);
        assert!(
            diff < 2e-3,
            "density map CPU/GPU divergente (sensitivity={sensitivity}, max diff={diff})"
        );
    }
}
