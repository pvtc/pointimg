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

fn pixel_channel(p: u32, channel: u32) -> f32 {
    if channel == 0u {
        return f32(p & 0xffu);
    }
    if channel == 1u {
        return f32((p >> 8u) & 0xffu);
    }
    return f32((p >> 16u) & 0xffu);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }

    let x0 = i32(id.x);
    let y0 = i32(id.y);

    // Centrer les valeurs sur le pixel de référence évite la perte de précision
    // f32 de E[x²] - E[x]² (notamment sur une image uniforme à 128).
    let reference_p = pixels[u32(y0) * params.width + u32(x0)];
    let reference = vec3<f32>(
        pixel_channel(reference_p, 0u),
        pixel_channel(reference_p, 1u),
        pixel_channel(reference_p, 2u),
    );
    var sum_delta = vec3<f32>(0.0);
    var sum_delta_sq = vec3<f32>(0.0);
    var count = 0.0;

    for (var dy = -params.radius; dy <= params.radius; dy = dy + 1) {
        for (var dx = -params.radius; dx <= params.radius; dx = dx + 1) {
            let x = clamp(x0 + dx, 0, i32(params.width) - 1);
            let y = clamp(y0 + dy, 0, i32(params.height) - 1);
            let p = pixels[u32(y) * params.width + u32(x)];
            let rgb = vec3<f32>(
                pixel_channel(p, 0u),
                pixel_channel(p, 1u),
                pixel_channel(p, 2u),
            );
            let delta = rgb - reference;
            sum_delta = sum_delta + delta;
            sum_delta_sq = sum_delta_sq + delta * delta;
            count = count + 1.0;
        }
    }

    let mean_delta = sum_delta / count;
    let variance = max(sum_delta_sq / count - mean_delta * mean_delta, vec3<f32>(0.0));
    raw_variance[id.y * params.width + id.x] = (variance.x + variance.y + variance.z) / 3.0;
}
