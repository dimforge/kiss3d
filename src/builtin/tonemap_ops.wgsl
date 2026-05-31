// Shared tonemap operators, concatenated as a prefix into both the rasterizer's
// HDR resolve (`hdr_tonemap.wgsl`) and the path tracer's tonemap
// (`raytrace/tonemap.wgsl`), so both pipelines apply the identical operator.
//
// `apply_tonemap(hdr, op)` maps a LINEAR HDR color to a gamma-encoded display
// color. `op` matches `post_processing::Tonemap::as_u32`:
//   0 = None (clamp), 1 = ACES, 2 = Reinhard, 3 = AgX, 4 = Khronos PBR Neutral,
//   5 = Tony McMapface.
//
// The Tony McMapface 3D LUT is declared here at group(0) bindings 6 & 7; every
// pass that includes this file must bind the LUT + sampler there.

@group(0) @binding(6) var tony_lut: texture_3d<f32>;
@group(0) @binding(7) var tony_samp: sampler;

fn tm_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tm_reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (x + vec3<f32>(1.0));
}

// AgX (Minimal AgX, Benjamin Wrensch). Returns a LINEAR display-referred color.
fn tm_agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
         + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn tm_agx(color: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = inset * color;
    v = clamp(log2(max(v, vec3<f32>(1e-10))), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    v = tm_agx_contrast(v);
    v = outset * v;
    v = pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));
    return v;
}

// Khronos PBR Neutral (2024). Preserves in-gamut saturation; LINEAR output.
fn tm_pbr_neutral(color_in: vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    var color = color_in;
    let x = min(color.r, min(color.g, color.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    color = color - vec3<f32>(offset);
    let peak = max(color.r, max(color.g, color.b));
    if (peak < start_compression) {
        return color;
    }
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    color = color * (new_peak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(color, vec3<f32>(new_peak), g);
}

// Tony McMapface, sampled from the baked 3D LUT (LINEAR output). HDR is encoded
// into the LUT's [0,1] domain with x/(x+1) plus a half-texel inset (48³ LUT).
fn tm_tony(hdr: vec3<f32>) -> vec3<f32> {
    let dim = 48.0;
    let stimulus = max(hdr, vec3<f32>(0.0));
    let encoded = stimulus / (stimulus + vec3<f32>(1.0));
    let uv = encoded * ((dim - 1.0) / dim) + 0.5 / dim;
    return textureSampleLevel(tony_lut, tony_samp, uv, 0.0).rgb;
}

// Applies operator `op` to a LINEAR HDR color and gamma-encodes for the non-sRGB
// output surface.
fn apply_tonemap(hdr: vec3<f32>, op: u32) -> vec3<f32> {
    var c: vec3<f32>;
    if (op == 1u) {
        c = tm_aces(hdr);
    } else if (op == 2u) {
        c = tm_reinhard(hdr);
    } else if (op == 3u) {
        c = tm_agx(hdr);
    } else if (op == 4u) {
        c = tm_pbr_neutral(hdr);
    } else if (op == 5u) {
        c = tm_tony(hdr);
    } else {
        c = clamp(hdr, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return pow(c, vec3<f32>(1.0 / 2.2));
}
