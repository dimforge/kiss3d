// Weighted-Blended Order-Independent Transparency composite pass
// (McGuire & Bavoil, "Weighted Blended Order-Independent Transparency", 2013).
//
// The transparent geometry pass accumulated, per pixel:
//   accum.rgb = Σ colorᵢ·αᵢ·wᵢ,  accum.a = Σ αᵢ·wᵢ   (additive)
//   reveal     = Π (1 - αᵢ)                            (multiplicative)
// This pass resolves them to a single transparent color and emits it with a
// coverage alpha of (1 - reveal); the pipeline blends it over the opaque HDR
// scene with SrcAlpha / OneMinusSrcAlpha.

@group(0) @binding(0) var t_accum: texture_2d<f32>;
@group(0) @binding(1) var t_reveal: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.clip_position.xy);
    let accum = textureLoad(t_accum, coord, 0);
    let reveal = textureLoad(t_reveal, coord, 0).r;
    // Weighted-average color of all transparent fragments at this pixel.
    let color = accum.rgb / max(accum.a, 1.0e-5);
    return vec4<f32>(color, 1.0 - reveal);
}
