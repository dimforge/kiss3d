import package::common::fullscreen_uv_from_clip;
// FXAA (Fast Approximate Anti-Aliasing), Timothy Lottes' classic luma-edge
// variant. Runs as a post-processing effect on the tonemapped LDR image: it
// detects luminance edges and blurs along them, smoothing aliasing without a
// depth/normal buffer or multisampling.

struct FxaaUniforms {
    // 1 / render-target size, in UV units per texel.
    inv_resolution: vec2<f32>,
    // Relative luma contrast above which an edge is processed (e.g. 0.125).
    edge_threshold: f32,
    // Absolute minimum luma contrast, to ignore dark noise (e.g. 0.0312).
    edge_threshold_min: f32,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_color: sampler;
@group(0) @binding(2) var<uniform> u: FxaaUniforms;

const SPAN_MAX: f32 = 8.0;
const REDUCE_MUL: f32 = 0.125;   // 1/8
const REDUCE_MIN: f32 = 0.0078125; // 1/128

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = fullscreen_uv_from_clip(position);
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let inv = u.inv_resolution;

    // Use `textureSampleLevel` (explicit LOD 0), not `textureSample`: the latter
    // computes implicit derivatives and so is only valid in uniform control flow,
    // but the early `return` below makes the later taps non-uniform. WebGPU (Tint)
    // rejects that as a uniformity violation even though native naga tolerates it.
    // The post-process target has no mips, so LOD 0 is exactly equivalent.
    let rgb_m = textureSampleLevel(t_color, s_color, uv, 0.0).rgb;
    let l_m = luma(rgb_m);
    let l_nw = luma(textureSampleLevel(t_color, s_color, uv + vec2<f32>(-inv.x, -inv.y), 0.0).rgb);
    let l_ne = luma(textureSampleLevel(t_color, s_color, uv + vec2<f32>(inv.x, -inv.y), 0.0).rgb);
    let l_sw = luma(textureSampleLevel(t_color, s_color, uv + vec2<f32>(-inv.x, inv.y), 0.0).rgb);
    let l_se = luma(textureSampleLevel(t_color, s_color, uv + vec2<f32>(inv.x, inv.y), 0.0).rgb);

    let range_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let range_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    let range = range_max - range_min;

    // Skip near-flat regions.
    if range < max(u.edge_threshold_min, range_max * u.edge_threshold) {
        return vec4<f32>(rgb_m, 1.0);
    }

    // Estimate the edge direction from the corner luma gradient.
    var dir = vec2<f32>(
        -((l_nw + l_ne) - (l_sw + l_se)),
        ((l_nw + l_sw) - (l_ne + l_se)),
    );
    let dir_reduce = max((l_nw + l_ne + l_sw + l_se) * 0.25 * REDUCE_MUL, REDUCE_MIN);
    let rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    dir = clamp(dir * rcp_dir_min, vec2<f32>(-SPAN_MAX), vec2<f32>(SPAN_MAX)) * inv;

    let rgb_a = 0.5 * (
        textureSampleLevel(t_color, s_color, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb
        + textureSampleLevel(t_color, s_color, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb
    );
    let rgb_b = rgb_a * 0.5 + 0.25 * (
        textureSampleLevel(t_color, s_color, uv + dir * -0.5, 0.0).rgb
        + textureSampleLevel(t_color, s_color, uv + dir * 0.5, 0.0).rgb
    );

    let l_b = luma(rgb_b);
    if l_b < range_min || l_b > range_max {
        return vec4<f32>(rgb_a, 1.0);
    }
    return vec4<f32>(rgb_b, 1.0);
}
