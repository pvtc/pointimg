struct Params {
    width: u32,
    height: u32,
    radius: i32,
    sensitivity: f32,
}

@group(0) @binding(0)
var<storage, read> pixels: array<u32>;

@group(0) @binding(1)
var<storage, read_write> raw_variance: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn luminance(p: u32) -> f32 {
    let r = f32(p & 0xffu);
    let g = f32((p >> 8u) & 0xffu);
    let b = f32((p >> 16u) & 0xffu);
    // Mêmes pondérations que le chemin CPU/SAT (Rec. 709) pour garantir la parité.
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }

    // Fenêtre clampée aux bords de l'image — comportement identique au chemin
    // CPU/SAT (le bord est ignoré, pas répliqué).
    let x0 = i32(id.x);
    let y0 = i32(id.y);
    let ix0 = max(x0 - params.radius, 0);
    let iy0 = max(y0 - params.radius, 0);
    let ix1 = min(x0 + params.radius, i32(params.width) - 1);
    let iy1 = min(y0 + params.radius, i32(params.height) - 1);

    // Centrer les valeurs sur le pixel de référence évite la perte de précision
    // f32 de E[x²] - E[x]² (notamment sur une image uniforme à 128).
    let reference = luminance(pixels[u32(y0) * params.width + u32(x0)]);
    var sum_delta = 0.0;
    var sum_delta_sq = 0.0;
    var count = 0.0;

    for (var y = iy0; y <= iy1; y = y + 1) {
        for (var x = ix0; x <= ix1; x = x + 1) {
            let delta = luminance(pixels[u32(y) * params.width + u32(x)]) - reference;
            sum_delta = sum_delta + delta;
            sum_delta_sq = sum_delta_sq + delta * delta;
            count = count + 1.0;
        }
    }

    let mean_delta = sum_delta / count;
    let variance = max(sum_delta_sq / count - mean_delta * mean_delta, 0.0);
    raw_variance[id.y * params.width + id.x] = variance;
}
