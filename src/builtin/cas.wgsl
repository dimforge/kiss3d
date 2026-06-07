// Contrast Adaptive Sharpening (AMD FidelityFX CAS), simplified single-pass
// (no scaling). Runs as a post-processing effect on the tonemapped LDR image:
// it sharpens detail while adapting the strength to local contrast so flat areas
// and already-sharp edges are left mostly untouched. Pairs well after FXAA/TAA.

struct CasUniforms {
    // 1 / render-target size, in UV units per texel.
    inv_resolution: vec2<f32>,
    // Sharpening strength in [0, 1].
    sharpness: f32,
    _pad: f32,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var s_color: sampler;
@group(0) @binding(2) var<uniform> u: CasUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = (position + vec2<f32>(1.0, 1.0)) * 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

fn tap(uv: vec2<f32>, off: vec2<f32>) -> vec3<f32> {
    return textureSample(t_color, s_color, uv + off * u.inv_resolution).rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    // 3x3 neighborhood:  a b c / d e f / g h i
    let a = tap(uv, vec2<f32>(-1.0, -1.0));
    let b = tap(uv, vec2<f32>(0.0, -1.0));
    let c = tap(uv, vec2<f32>(1.0, -1.0));
    let d = tap(uv, vec2<f32>(-1.0, 0.0));
    let e = tap(uv, vec2<f32>(0.0, 0.0));
    let f = tap(uv, vec2<f32>(1.0, 0.0));
    let g = tap(uv, vec2<f32>(-1.0, 1.0));
    let h = tap(uv, vec2<f32>(0.0, 1.0));
    let i = tap(uv, vec2<f32>(1.0, 1.0));

    // Soft min/max of the cross, then extended to the full 3x3 (CAS).
    var mn = min(min(min(d, e), min(f, b)), h);
    let mn2 = min(mn, min(min(a, c), min(g, i)));
    mn = mn + mn2;

    var mx = max(max(max(d, e), max(f, b)), h);
    let mx2 = max(mx, max(max(a, c), max(g, i)));
    mx = mx + mx2;

    let rcp_mx = 1.0 / max(mx, vec3<f32>(1e-4));
    var amp = clamp(min(mn, vec3<f32>(2.0) - mx) * rcp_mx, vec3<f32>(0.0), vec3<f32>(1.0));
    amp = sqrt(amp);

    // Sharpening weight: peak between -1/8 (soft) and -1/5 (sharp).
    let peak = -1.0 / mix(8.0, 5.0, clamp(u.sharpness, 0.0, 1.0));
    let w = amp * peak;
    let rcp_w = 1.0 / (1.0 + 4.0 * w);

    let out_rgb = (b * w + d * w + f * w + h * w + e) * rcp_w;
    return vec4<f32>(max(out_rgb, vec3<f32>(0.0)), 1.0);
}
